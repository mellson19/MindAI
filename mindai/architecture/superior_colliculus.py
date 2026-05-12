"""Superior Colliculus — retinotopic saccade map driven by desire, not script.

Biological pathway (Wurtz & Goldberg 1989; Sparks 2002):

  SC superficial layers:
    Retinal ganglion cells (magnocellular) → SC: fast motion/onset signal
    V1/MT → SC: processed feature change after cortical delay

  SC deep/intermediate layers:
    FEF (Frontal Eye Fields) → SC: voluntary / goal-directed saccades
    Amygdala → SC: fear/reward orienting (Davis & Whalen 2001)
    Dopamine (SNc → SC): motivational salience gating
    Noradrenaline (LC → SC): arousal, boosts orienting responses
    ACh (brainstem PPT/LDT → SC): REM/waking gating, novelty detection

  SC internal dynamics:
    Fixation neurons (rostral SC): tonically active, suppress buildup cells.
      Release when cumulative drive overwhelms tonic inhibition.
      → high DA + high NA → fixation neurons less active → more saccades
      → satiated/calm state → fixation neurons strong → gaze holds (Munoz & Istvan 1998)

    Buildup neurons: leaky integrate-and-fire ramp toward threshold.
      Receive priority map × (1 - fixation_neuron_inhibition).
      When threshold crossed → burst → saccade command (Dorris & Munoz 1998).

    IOR: burst neurons enter refractory after firing.
      Implemented as per-location decay (τ ≈ 800 ms), NOT external Python logic.
      Emerges from the SC circuit itself (Posner & Cohen 1984; Klein 2000).

  What drives "desire to look":
    - Dopamine above baseline → wanting/curiosity → fixation neurons released
    - Noradrenaline × surprise → strong orienting to unexpected events
    - Amygdala threat → hypervigilant scanning (large priority everywhere)
    - ACh high → enhanced novelty detection in motion channel
    - All of these are driven by physiology (hunger, pain, world events)
      so gaze emerges from internal state, not from scripted rules.

References:
  Wurtz RH & Goldberg ME (1989) The Neurobiology of Saccadic Eye Movements.
  Munoz DP & Istvan PJ (1998) Lateral inhibitory interactions in the intermediate
      layers of the monkey superior colliculus. J Neurophysiol 79.
  Dorris MC & Munoz DP (1998) Saccadic probability influences motor preparation.
      J Neurosci 18.
  Posner MI & Cohen Y (1984) Components of visual orienting. Attention & Performance X.
  Klein RM (2000) Inhibition of return. Trends Cogn Sci 4.
  Davis M & Whalen PJ (2001) The amygdala: vigilance and emotion. Mol Psychiatry 6.
"""

from __future__ import annotations

import numpy as np


class SuperiorColliculus:
    """Retinotopic saccade map.

    Parameters
    ----------
    map_h, map_w : int
        Resolution of the SC retinotopic map. 15 × 20 matches retina's aspect ratio
        and is sufficient for coarse gaze control (SC resolves ~1–2° per cell).
    """

    # SC map resolution (rows × cols in visual-angle space)
    MAP_H: int = 15
    MAP_W: int = 20

    # Buildup neuron dynamics
    _LEAK:       float = 0.12   # per-tick leak (Dorris & Munoz 1998: τ ≈ 8 ticks)
    _THRESHOLD:  float = 0.65   # saccade threshold (calibrated to ~250 ms fixation)

    # Fixation neuron tonic strength and release thresholds
    _FIX_RECOVERY:  float = 0.04   # tonic recovery rate per tick
    _FIX_MAX:       float = 1.0

    # IOR parameters (Posner & Cohen 1984)
    _IOR_STRENGTH: float = 1.0    # full inhibition after firing
    _IOR_SIGMA:    float = 1.8    # spatial spread (cells)
    _IOR_DECAY:    float = 0.982  # per-tick: half-life ≈ 38 ticks ≈ 380 ms

    # Saccadic suppression duration (Matin 1974: ~40 ms)
    _SACCADE_TICKS: int = 4

    def __init__(self, map_h: int = MAP_H, map_w: int = MAP_W) -> None:
        self.map_h = map_h
        self.map_w = map_w
        n = map_h * map_w

        self._priority  = np.zeros(n, dtype=np.float32)
        self._buildup   = np.zeros(n, dtype=np.float32)
        self._ior       = np.zeros(n, dtype=np.float32)

        # Fixation neuron ensemble (rostral SC) — scalar proxy
        self._fixation_strength: float = 0.8

        # Current gaze in normalised image coords [-1, 1]
        self._fx: float = 0.0
        self._fy: float = 0.0

        # Saccadic suppression countdown
        self._suppressing: int = 0

        # Precompute IOR Gaussian kernel
        r   = max(1, int(self._IOR_SIGMA * 3))
        sz  = 2 * r + 1
        yy, xx = np.mgrid[-r:r+1, -r:r+1].astype(np.float32)
        self._ior_kernel = np.exp(-(xx**2 + yy**2) / (2 * self._IOR_SIGMA**2))
        self._ior_r = r

        # For diagnostics
        self.last_priority: np.ndarray = np.zeros(n, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        visual_motion:   np.ndarray,  # motion channel [4::5] per retinal receptor
        visual_luma:     np.ndarray,  # luma channel   [3::5] per retinal receptor
        surprise:        float,        # global prediction error (brain.surprise)
        threat:          float,        # amygdala.threat_level
        dopamine:        float,        # mesolimbic DA [0,1]; baseline ~0.5
        noradrenaline:   float,        # LC-NA [0,1]
        acetylcholine:   float,        # basal forebrain ACh [0,1]
        goal_drive:      float = 0.5,  # pfc.goal_persistence [0,1]
    ) -> tuple[float, float, bool]:
        """Compute one SC tick.

        Returns
        -------
        fx, fy : float
            Gaze fixation in normalised image coords [-1, 1].
        saccading : bool
            True during saccadic suppression window.
        """
        H, W = self.map_h, self.map_w

        # --- IOR decay (refractory of burst neurons) ----------------------
        self._ior *= self._IOR_DECAY

        # --- Saccadic suppression window ----------------------------------
        if self._suppressing > 0:
            self._suppressing -= 1
            return self._fx, self._fy, True

        # --- Build SC priority map ----------------------------------------
        # Two spatial signals, always active — their natural magnitudes determine
        # which dominates. No scripted switch between static/dynamic modes.
        #
        # motion_map: temporal change per location (Wurtz & Goldberg 1989)
        #   — zero on static images naturally; drives orienting on video
        #   — ACh gates how strongly motion triggers saccades (PPT→SC)
        #
        # luma_map: brain's neural activity in the vision channel after
        #   full recurrent processing (act_cpu, not raw retinal input)
        #   — includes top-down feedback: concept neurons activated by the
        #     current token context pre-activate associated visual neurons
        #     via STDP-learned weights → shows up as elevated luma activity
        #     at matching image locations (e.g. "дерево" boosts tree regions)
        #   — NA gates how strongly internal brain state drives gaze
        #
        # Together: on video, motion dominates. On static image, luma
        # (= brain's interpretation + top-down) dominates. Emergent, not scripted.
        motion_map  = self._downsample(visual_motion, H, W)
        luma_map    = self._downsample(visual_luma,   H, W)
        threat_gain = 1.0 + 3.0 * threat

        spatial_sal = (
            motion_map * (0.3 + 0.7 * acetylcholine)   # motion × ACh gate
            + luma_map * noradrenaline                  # top-down × arousal
            + float(np.clip(surprise / 10.0, 0.0, 1.0))
              * noradrenaline                           # global PE, uniform
        ) * threat_gain

        goal_salience = goal_drive * 0.08   # tonic PFC search drive, uniform

        self._priority = spatial_sal * (1.0 - self._ior) + goal_salience
        np.clip(self._priority, 0.0, 1.0, out=self._priority)
        self.last_priority = self._priority.copy()

        # --- Fixation neuron dynamics -------------------------------------
        # Release fixation when orienting drive is strong
        # (dopamine above baseline → wanting; NA × surprise → reflexive orienting)
        da_above_baseline = float(np.clip(dopamine - 0.5, 0.0, 0.5)) * 2.0
        orienting_drive   = min(1.0,
            da_above_baseline * 0.15
            + noradrenaline * float(np.clip(surprise / 5.0, 0.0, 1.0)) * 0.25
            + threat * 0.3
        )
        self._fixation_strength = float(np.clip(
            self._fixation_strength
            - orienting_drive
            + self._FIX_RECOVERY,
            0.0, self._FIX_MAX,
        ))

        # --- Buildup neurons (leaky integrator) ---------------------------
        # Suppressed by fixation neurons (Munoz & Istvan 1998)
        suppression_factor = 1.0 - self._fixation_strength * 0.8
        self._buildup += (
            self._priority * suppression_factor
            - self._LEAK * self._buildup
        )
        np.clip(self._buildup, 0.0, 1.0, out=self._buildup)

        # --- Winner-take-all: peak buildup → saccade ----------------------
        peak_idx  = int(np.argmax(self._buildup))
        peak_val  = float(self._buildup[peak_idx])

        if peak_val >= self._THRESHOLD:
            fy_i, fx_i = divmod(peak_idx, W)
            self._fx = float(fx_i / max(W - 1, 1) * 2.0 - 1.0)
            self._fy = float(fy_i / max(H - 1, 1) * 2.0 - 1.0)

            # IOR: burst-neuron refractory at fired location
            self._stamp_ior(peak_idx)

            # Buildup reset after saccade (all buildup suppressed briefly)
            self._buildup *= 0.08

            # Fixation neurons re-engage strongly after saccade (gaze stabilisation)
            self._fixation_strength = 0.85

            self._suppressing = self._SACCADE_TICKS
            return self._fx, self._fy, True

        return self._fx, self._fy, False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _downsample(self, signal: np.ndarray, H: int, W: int) -> np.ndarray:
        """Linearly interpolate 1-D signal onto H×W SC map."""
        n = len(signal)
        if n == 0:
            return np.zeros(H * W, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, n)
        x_new = np.linspace(0.0, 1.0, H * W)
        return np.interp(x_new, x_old, signal).astype(np.float32)

    def _stamp_ior(self, peak_idx: int) -> None:
        """Add Gaussian IOR blob centred at peak_idx (refractory of burst cell).
        Vectorised with numpy — no Python loop per saccade."""
        H, W = self.map_h, self.map_w
        cy, cx = divmod(peak_idx, W)
        r = self._ior_r
        ys = np.arange(max(0, cy - r), min(H, cy + r + 1))
        xs = np.arange(max(0, cx - r), min(W, cx + r + 1))
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        dy = yy - cy
        dx = xx - cx
        k = self._ior_kernel[dy + r, dx + r]
        idxs = (yy * W + xx).ravel()
        np.add.at(self._ior, idxs, (self._IOR_STRENGTH * k).ravel())
        np.clip(self._ior, 0.0, 1.0, out=self._ior)
