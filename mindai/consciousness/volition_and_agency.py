"""Volition and agency — decision-making without scripted reward.

BasalGanglia
------------
Implements the corticostriatal three-factor Hebbian rule (Reynolds & Wickens
2002).  The update is:

    Δw = η × pre × post × (DA − DA_baseline)

DA above baseline → LTP on active corticostriatal synapses (approach).
DA below baseline → LTD (avoidance via indirect pathway suppression).

**Critically, pain does NOT directly modify weights.**  Pain suppresses VTA
dopamine in EndocrineSystem; the resulting DA drop below baseline naturally
causes LTD on whichever action was taken.  There is no developer-assigned
punishment signal — punishment is mediated entirely through the physiology
of the dopamine system.

FreeWillEngine
--------------
Implements the Libet readiness potential delay (Libet 1983) and somatic
marker bias (Damasio 1994).

Somatic Markers (biological mechanism)
---------------------------------------
The vmPFC stores co-occurrences of action × interoceptive body state.  When
the same action context is encountered, the OFC reconstructs the associated
body state via the insula → anterior cingulate pathway, producing a felt
"gut feeling" before deliberation.

What this means computationally:
- The marker IS the reconstructed felt distress (after substance P sensitisation)
- NOT a Q-value or Pavlovian weight — it is a body-state memory trace
- Fed by `felt_distress` (interoceptive signal) not raw world stress
- Influence: exponential suppression of the action's probability

Extinction (Milad 2006; Quirk & Mueller 2008):
  Safe re-exposure to a feared context produces rapid vmPFC-mediated
  extinction. Implemented as:
    1. Active extinction: re-taking an action with low distress applies
       stronger decay (* 0.80) than passive EMA (* 0.97).
    2. Passive decay: all markers fade slowly (* 0.9995/tick) even without
       re-exposure — biological forgetting of unreinforced fear traces.

Sense of authorship (Wegner & Wheatley 1999; Haggard et al. 2002):
  Agency = P(effect | action) - P(effect | no-action), approximated as:
  - Rises when an action reduces somatic marker distress (positive outcome)
  - Falls when an action increases it (negative outcome)
  - Decays slowly toward baseline when no actions are taken
"""

import numpy as np


class FreeWillEngine:

    def __init__(self, delay_ticks: int = 15):
        self.delay_ticks          = delay_ticks
        self.decision_queue       = []
        self.sense_of_authorship  = 1.0
        # Somatic marker traces — one per action; auto-grows with num_actions
        self.somatic_markers_pain = np.zeros(5)
        self._prev_markers        = np.zeros(5)   # for authorship delta

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _ensure_capacity(self, action_idx: int) -> None:
        if action_idx >= len(self.somatic_markers_pain):
            extra = action_idx - len(self.somatic_markers_pain) + 1
            self.somatic_markers_pain = np.concatenate(
                [self.somatic_markers_pain, np.zeros(extra)])
            self._prev_markers = np.concatenate(
                [self._prev_markers, np.zeros(extra)])

    # ------------------------------------------------------------------
    # Passive decay — all markers fade slowly even without re-exposure
    # (Milad 2006: extinction of unreinforced fear traces)
    # ------------------------------------------------------------------

    def decay_markers_passive(self) -> None:
        """Call once per tick from brain.py (awake path)."""
        self.somatic_markers_pain *= 0.9995

    # ------------------------------------------------------------------
    # Active update (vmPFC learning)
    # ------------------------------------------------------------------

    def update_somatic_markers(
        self, action_taken: int, felt_distress: float
    ) -> None:
        """Update body-state memory trace for the taken action.

        Parameters
        ----------
        action_taken:
            Index of the motor action just executed.
        felt_distress:
            Interoceptive signal [0,1] — substance-P-sensitised pain.
            This is what makes markers "somatic": they encode *how the body
            felt*, not how much raw stress the world delivered (Damasio 1994).

        Extinction logic (Milad 2006):
            Safe re-exposure (felt_distress < 0.05) triggers active extinction:
            the marker decays faster than normal EMA, modelling vmPFC-mediated
            safety learning via IL→BLA projection.
        """
        if action_taken is None:
            return
        self._ensure_capacity(action_taken)

        prev = self.somatic_markers_pain[action_taken]

        if felt_distress < 0.05 and self.somatic_markers_pain[action_taken] > 0.1:
            # Active extinction: safe re-exposure decays marker rapidly
            # vmPFC infralimbic cortex → amygdala BLA inhibition (Quirk 2008)
            self.somatic_markers_pain[action_taken] *= 0.80
        else:
            # Standard EMA: traumatic body memories persist (PTSD curves)
            self.somatic_markers_pain[action_taken] = (
                self.somatic_markers_pain[action_taken] * 0.97
                + felt_distress * 0.03
            )

        # Sense of authorship: did taking this action improve body state?
        delta = prev - self.somatic_markers_pain[action_taken]
        if delta > 0.01:
            # Marker reduced → action had a good outcome → agency rises
            self.sense_of_authorship = min(1.0, self.sense_of_authorship + 0.02)
        elif felt_distress > 0.3:
            # Distress increased → action had bad outcome → agency erodes
            self.sense_of_authorship = max(0.0, self.sense_of_authorship - 0.05)

    # ------------------------------------------------------------------
    # Motor bias via somatic markers
    # ------------------------------------------------------------------

    def bias_motor_probabilities(
        self, base_probabilities: np.ndarray, amygdala_arousal: float
    ) -> np.ndarray:
        """Apply somatic marker suppression to action probabilities.

        When amygdala arousal (noradrenaline proxy) is high, intuitive body
        memory dominates over deliberate cortical action selection.
        Exponential suppression matches the qualitative, all-or-nothing nature
        of aversion (Damasio 1994).
        """
        n                = len(base_probabilities)
        if len(self.somatic_markers_pain) < n:
            self.somatic_markers_pain = np.concatenate(
                [self.somatic_markers_pain, np.zeros(n - len(self.somatic_markers_pain))])
            self._prev_markers = np.concatenate(
                [self._prev_markers, np.zeros(n - len(self._prev_markers))])
        markers          = self.somatic_markers_pain[:n]
        intuition_weight = float(np.clip(amygdala_arousal, 0.0, 1.0))
        suppression      = np.exp(-markers / (0.2 + 1e-9))
        biased           = base_probabilities * (
            (1.0 - intuition_weight) + suppression * intuition_weight)
        total = np.sum(biased)
        if total < 1e-9:
            return np.ones(n) / n
        return biased / total

    # ------------------------------------------------------------------
    # Libet readiness potential delay
    # ------------------------------------------------------------------

    def unconscious_decision_making(
        self, motor_potentials: np.ndarray, amygdala_arousal: float = 0.5
    ) -> int:
        exp_vals   = np.exp(motor_potentials - np.max(motor_potentials))
        base_probs = exp_vals / np.sum(exp_vals)
        final_probs = self.bias_motor_probabilities(base_probs, amygdala_arousal)
        chosen = int(np.random.choice(len(final_probs), p=final_probs))
        self.decision_queue.append({
            'action':          chosen,
            'ticks_remaining': self.delay_ticks,
        })
        return chosen

    def conscious_veto_and_awareness(self) -> int | None:
        """Apply Libet delay (~200 ms readiness potential)."""
        action_to_realize = None
        for decision in self.decision_queue:
            decision['ticks_remaining'] -= 1
            if decision['ticks_remaining'] <= 0 and action_to_realize is None:
                action_to_realize = decision['action']
        self.decision_queue = [
            d for d in self.decision_queue if d['ticks_remaining'] > 0]
        return action_to_realize


# ---------------------------------------------------------------------------

class BasalGanglia:
    """Corticostriatal action selection: direct, indirect, and hyperdirect pathways.

    Biological basis:
      Direct pathway (D1 MSNs, striatum → GPi/SNr):
        Go-pathway. DA above baseline → D1 LTP → disinhibits thalamus → action.
        (Reynolds & Wickens 2002; Frank 2004)

      Indirect pathway (D2 MSNs, striatum → GPe → STN → GPi/SNr):
        No-go pathway. DA below baseline → D2 LTP → increases GPi inhibition
        of thalamus → suppresses action.  Opponent process to direct.
        (Frank 2004; Mink 1996)

      Hyperdirect pathway (cortex → STN → GPi/SNr, ~5 ms latency):
        Emergency brake. A fast cortical signal (via STN, bypassing striatum)
        immediately suppresses all actions before the slower direct/indirect
        paths resolve.  Critical for stopping a competing action when a new
        urgent stimulus arrives.
        (Aron & Poldrack 2006; Cavanagh 2011 — high conflict → cortex → STN)

    Learning rule (three-factor Hebbian, Reynolds & Wickens 2002):
      Direct:   Δw_D1 = +η × pre × (DA − baseline)   [LTP when DA high]
      Indirect: Δw_D2 = −η × pre × (DA − baseline)   [LTP when DA LOW — no-go]

    Pain does NOT enter directly.  Pain suppresses VTA dopamine; the DA drop
    below baseline produces D2 LTP (indirect no-go) — aversive learning
    is entirely mediated through physiology.
    """

    _DA_BASELINE = 0.5

    def __init__(self, motor_cortex_size: int, num_actions: int = 5):
        self.num_actions        = num_actions
        # Direct pathway weights (D1 MSNs) — go
        self.direct_weights     = np.random.uniform(
            0.05, 0.3, (motor_cortex_size, num_actions))
        # Indirect pathway weights (D2 MSNs) — no-go
        self.indirect_weights   = np.random.uniform(
            0.01, 0.1, (motor_cortex_size, num_actions))
        self.last_motor_pattern = None
        self.last_chosen_action = None

        # STN hyperdirect state — decays each tick
        self._stn_suppression: float = 0.0

    def hyperdirect_brake(self, conflict: float) -> None:
        """Cortex → STN fast suppression when response conflict is high.

        Called from brain.py when ACC reports conflict > threshold.
        STN suppression boosts GPi inhibition of all actions for ~2–3 ticks
        (Cavanagh 2011: theta-band cortical burst → STN → broad action hold).
        """
        self._stn_suppression = min(1.0, self._stn_suppression + conflict * 0.5)

    def map_to_action_potentials(
        self, motor_cortex_activity: np.ndarray
    ) -> np.ndarray:
        """Net striatal output: direct − indirect, modulated by STN brake."""
        self.last_motor_pattern = motor_cortex_activity.copy()
        direct   = np.dot(motor_cortex_activity, self.direct_weights)
        indirect = np.dot(motor_cortex_activity, self.indirect_weights)
        net      = direct - indirect
        # STN hyperdirect: uniform suppression across all actions
        net = net * (1.0 - self._stn_suppression * 0.7)
        # Decay STN suppression (resolves in ~3 ticks = 300 ms)
        self._stn_suppression = max(0.0, self._stn_suppression - 0.35)
        return net

    def reinforce_learning(
        self, chosen_action: int, dopamine: float
    ) -> None:
        """Three-factor Hebbian update on direct AND indirect pathways.

        Direct  (D1): DA > baseline → LTP → strengthen go-signal
        Indirect(D2): DA < baseline → LTP → strengthen no-go signal
        """
        if self.last_motor_pattern is None or chosen_action is None:
            return
        if chosen_action >= self.num_actions:
            return
        self.last_chosen_action = chosen_action

        da_error = dopamine - self._DA_BASELINE
        eta      = 0.003
        pre      = self.last_motor_pattern

        # Direct pathway: strengthened by positive prediction error
        d_direct = pre * max(0.0,  da_error) * eta
        self.direct_weights[:, chosen_action] = np.clip(
            self.direct_weights[:, chosen_action] + d_direct, 0.0, 1.0)

        # Indirect pathway: strengthened by negative prediction error
        d_indirect = pre * max(0.0, -da_error) * eta
        self.indirect_weights[:, chosen_action] = np.clip(
            self.indirect_weights[:, chosen_action] + d_indirect, 0.0, 1.0)

    # Backward compat — brain.py accesses action_weights for some paths
    @property
    def action_weights(self) -> np.ndarray:
        return self.direct_weights - self.indirect_weights
