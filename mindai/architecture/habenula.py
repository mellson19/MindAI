"""Lateral Habenula (LHb) — anti-reward signal balancing the dopamine system.

Biological basis
----------------
LHb fires when an expected reward is OMITTED or worse-than-predicted
(Matsumoto & Hikosaka 2007). Its output projects to the rostromedial
tegmental nucleus (RMTg), which inhibits VTA dopamine neurons.

Function pairing with VTA:
    VTA dopamine: positive reward prediction error  (got reward unexpectedly)
    LHb       :   negative reward prediction error  (expected reward, got nothing)

Together they form a complete bidirectional teacher signal — exactly the
shape of the three-factor Hebbian rule's DA-baseline term.

Why MindAI needs it
-------------------
Without LHb, the brain's dopamine system can only signal "better than
expected" via depression toward baseline; "worse than expected" gets lost
in the floor. This causes oscillatory learning when predictions are
violated downward (cortisol spikes without DA dip → unstable mood).

LHb adds the missing signal: it tracks a slow reward expectation and
fires its anti-reward when reality undershoots. The output suppresses
mesolimbic DA — natural, biologically-correct LTD on the action that led
to the disappointment.

References
----------
- Matsumoto M, Hikosaka O (2007). Lateral habenula as a source of negative
  reward signals in dopamine neurons. Nature 447: 1111-1115.
- Hong S, Hikosaka O (2008). The globus pallidus sends reward-related
  signals to the lateral habenula. Neuron 60: 720-729.
"""

from __future__ import annotations


class Habenula:
    """Tracks expected reward and produces anti-reward when reality undershoots."""

    def __init__(self, expectation_decay: float = 0.99):
        # Slow EMA of recent reward — the "expected" baseline
        self._expected_reward: float = 0.0
        self._decay            = expectation_decay
        # Last LHb burst — read by Brain to suppress VTA DA
        self.anti_reward_signal: float = 0.0

    def update(self, actual_reward: float) -> float:
        """Update expectation and emit anti-reward burst.

        Returns
        -------
        anti_reward_signal in [0,1]
            Magnitude of LHb activation. Brain.py uses this to apply LTD
            on the just-chosen action via the indirect pathway.
        """
        actual = max(0.0, min(1.0, float(actual_reward)))

        # Negative prediction error: expected more than we got
        omission = max(0.0, self._expected_reward - actual)
        # LHb burst proportional to omission magnitude
        # Scaling 2.0 because typical omissions are ~0.1-0.3
        self.anti_reward_signal = min(1.0, omission * 2.0)

        # Slow update of expectation — Bayesian-like running average
        # Decay slow enough that one good reward doesn't immediately reset hope
        self._expected_reward = self._expected_reward * self._decay + actual * (1.0 - self._decay)

        return self.anti_reward_signal

    @property
    def expectation(self) -> float:
        """Current expected reward baseline (read-only)."""
        return self._expected_reward
