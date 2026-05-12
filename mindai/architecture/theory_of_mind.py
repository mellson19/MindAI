"""Theory of Mind (ToM) — other agents' mental state modeling.

Biological basis (Premack & Woodruff 1978; Baron-Cohen 1985; Frith 1992):
  ToM (also called mentalising) is the ability to attribute mental states
  (beliefs, desires, intentions) to other agents and use these to predict
  and explain their behaviour.

  Neural substrate (Saxe & Kanwisher 2003; Frith & Frith 2006):
    TPJ (temporoparietal junction) — right TPJ is consistently activated
      when attributing beliefs to others.  Damage → inability to distinguish
      intentional from accidental actions.
    mPFC (medial prefrontal, area 10) — represents mental states of others;
      also active in self-reflection (shared circuitry: Frith 2007).
    STS (posterior, right hemisphere) — biological motion → intentional agent.
    Temporal poles — semantic knowledge of person-specific behaviour.

  First-order ToM: "I believe that X believes that..."
  Second-order ToM: "I believe that X believes that Y believes..."
  Children develop first-order at ~3–4 years, second-order at ~5–6 years.

  Link to mirror neurons (Gallese & Goldman 1998 — simulation theory):
    Mirror neurons provide the motor simulation of another's action.
    ToM adds the *inference* about the goal/intention behind that action.
    Shared representation hypothesis: understanding actions via simulation,
    then attributing goals = low-level ToM.

Implementation:
  The agent observes the human's action history (from world signals).
  A simple agent model: maintained belief about human's current goal state,
  updated by observed human actions.

  No scripted "theory" — the agent builds up statistical associations
  between observed human behaviours via Hebbian association in this module.
  The belief state is a soft probability distribution over possible
  human "modes" (approaching, retreating, idle, attacking).

  Association weights update Hebbian-style: when the agent observes
  a behaviour AND has a strong prediction, co-activation strengthens
  the relevant association.
"""

from __future__ import annotations
import numpy as np


# Human behavioural modes inferred from world signals
_MODES = ['approaching', 'retreating', 'idle', 'interacting', 'threat']


class TheoryOfMind:

    def __init__(self):
        n = len(_MODES)
        # Belief distribution over human modes — uniform prior
        self.belief: np.ndarray = np.ones(n, dtype=np.float32) / n

        # Association weights: observable features → mode probabilities
        # Features: [distance_change, proximity, interaction_flag, threat_signal]
        self._assoc_w = np.random.uniform(0.1, 0.3, (n, 4)).astype(np.float32)

        # Smoothed prediction confidence
        self.confidence: float = 0.0

        self._prev_distance: float = float('inf')

    def update(
        self,
        distance:          float,     # distance to human (world units)
        human_interacted:  bool,      # human pressed interaction key
        threat_signal:     float,     # amygdala threat level
    ) -> dict:
        """Update belief about human mental state.

        Returns:
          belief         : np.ndarray [5] — distribution over human modes
          most_likely    : str — highest-probability mode label
          confidence     : float [0,1] — entropy-based confidence
        """
        # Build feature vector from observable signals
        dist_change  = float(np.clip(
            (self._prev_distance - distance) / 10.0, -1.0, 1.0))
        proximity    = float(np.clip(1.0 - distance / 50.0, 0.0, 1.0))
        interact_f   = 1.0 if human_interacted else 0.0
        features     = np.array([dist_change, proximity, interact_f, threat_signal],
                                 dtype=np.float32)

        # Likelihood: dot product of features with association weights
        likelihood = np.dot(self._assoc_w, features)
        likelihood = np.exp(likelihood - likelihood.max())   # softmax
        likelihood /= likelihood.sum() + 1e-9

        # Bayesian update: posterior ∝ prior × likelihood
        posterior = self.belief * likelihood
        posterior /= posterior.sum() + 1e-9
        self.belief = 0.85 * self.belief + 0.15 * posterior

        # Hebbian weight update: co-activate features with current belief
        if self.belief.max() > 0.5:
            dominant = np.argmax(self.belief)
            self._assoc_w[dominant] = np.clip(
                self._assoc_w[dominant] + features * 0.005, 0.0, 1.0)

        # Confidence: 1 − normalised entropy
        p = self.belief + 1e-9
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(_MODES))
        self.confidence = float(np.clip(1.0 - entropy / max_entropy, 0.0, 1.0))

        self._prev_distance = distance

        return {
            'belief':       self.belief,
            'most_likely':  _MODES[int(np.argmax(self.belief))],
            'confidence':   self.confidence,
        }
