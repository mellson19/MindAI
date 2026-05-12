"""Periaqueductal Gray (PAG) — defensive behaviour switch.

Biological basis
----------------
The PAG is a midbrain structure that integrates threat signals and selects
one of three discrete defensive behaviours (Bandler & Shipley 1994):

    dorsolateral PAG (dlPAG)  →  active defence: fight / flight  (high arousal)
    ventrolateral PAG (vlPAG) →  passive defence: freeze         (immobility, analgesia)
    lateral PAG (lPAG)        →  flight when escape is possible

Selection between modes depends on perceived escapability and proximity
of threat (Fanselow 1994 "predatory imminence continuum"):
    distant threat   → freeze       (vlPAG, parasympathetic)
    nearby threat    → flight       (lPAG)
    contact threat   → fight        (dlPAG, sympathetic surge)

The PAG output gates substance-P release, motor activation, and opioid-
mediated stress analgesia (Reynolds 1969). During freeze, opioids spike
and pain perception is suppressed — biologically correct without scripted
reward shaping.

Why MindAI needs it
-------------------
Currently fear (amygdala) → noradrenaline linearly. Real defensive
responses are categorical, not graded. PAG provides the discrete switch
that decides whether the agent flees, freezes, or attacks — and exposes
that mode to the rest of the brain as a single state label that motor
systems can read.

References
----------
- Bandler R, Shipley MT (1994). Columnar organization in the midbrain
  periaqueductal gray. Trends Neurosci 17: 379-389.
- Fanselow MS (1994). Neural organization of the defensive behavior
  system responsible for fear. Psychon Bull Rev 1: 429-438.
- Reynolds DV (1969). Surgery in the rat during electrical analgesia
  induced by focal brain stimulation. Science 164: 444-445.
"""

from __future__ import annotations


class PAG:
    """Three-mode defensive switch driven by threat × distance × DA."""

    MODES = ('rest', 'freeze', 'flight', 'fight')

    def __init__(self):
        self.mode: str = 'rest'
        self._mode_steadiness = 0    # ticks remaining in current mode (hysteresis)
        self.opioid_analgesia: float = 0.0   # PAG → endogenous opioid release

    def update(
        self,
        threat:           float,
        distance_to_threat: float = float('inf'),
        dopamine:         float = 0.5,
    ) -> str:
        """Select defensive mode and emit analgesia signal.

        Parameters
        ----------
        threat:
            Amygdala threat level [0,1].
        distance_to_threat:
            Approximate distance — used for predatory imminence. Default inf
            (no proximate threat) so the agent stays calm by default.
        dopamine:
            Mesolimbic DA. Low DA + high threat biases toward freeze
            (learned helplessness). High DA + threat biases toward fight.

        Returns
        -------
        mode: one of MODES.
        """
        # Hysteresis: don't flip every tick
        if self._mode_steadiness > 0:
            self._mode_steadiness -= 1
            return self.mode

        # Below threshold → reset to calm
        if threat < 0.2:
            self.mode             = 'rest'
            self.opioid_analgesia = 0.0
            return self.mode

        # Predatory imminence: closer = more active defence
        if distance_to_threat > 10.0:
            new_mode = 'freeze'                     # distant: parasympathetic
        elif distance_to_threat > 2.0:
            new_mode = 'flight'                     # near: try to escape
        else:
            # Contact threat: fight or freeze depending on agency
            new_mode = 'fight' if dopamine > 0.4 else 'freeze'

        # Learned helplessness: very low DA always biases to freeze
        if dopamine < 0.2 and threat > 0.5:
            new_mode = 'freeze'

        self.mode             = new_mode
        self._mode_steadiness = 5    # stay in mode for ~500 ms

        # vlPAG → PAG opioid surge during freeze (Reynolds 1969)
        # Stress-induced analgesia: pain damped during immobile defence
        self.opioid_analgesia = 0.7 if new_mode == 'freeze' else 0.0

        return self.mode

    @property
    def is_active_defence(self) -> bool:
        return self.mode in ('flight', 'fight')

    @property
    def is_immobile(self) -> bool:
        """During freeze the PAG inhibits motor output."""
        return self.mode == 'freeze'
