"""Default Mode Network (DMN) — self-referential thought and autobiographical memory.

Biological basis (Raichle et al. 2001; Buckner et al. 2008):
  The DMN is a set of brain regions (mPFC, PCC/precuneus, angular gyrus,
  hippocampus) that are most active during:
    - rest / mind-wandering
    - autobiographical memory retrieval
    - future simulation / mental time travel
    - theory of mind / social cognition
    - self-referential processing ("is this relevant to me?")

  The DMN is DEACTIVATED during externally-directed attention.
  Task-positive network (TPN) and DMN show anti-correlated activity
  (Fox et al. 2005): high external demand → DMN suppressed.

  mPFC (medial prefrontal cortex):
    Self-referential judgements ("am I capable of this?").
    Integrates past experience with current context.

  PCC/Precuneus:
    Autobiographical memory retrieval hub; interfaces with hippocampus.
    Sustained activation = ongoing self-narrative.

  Angular gyrus (TPJ overlap):
    Semantic integration; also implicated in social inference.

  Hippocampal subnetwork:
    Memory re-play and scene construction (Hassabis & Maguire 2007).
    Future simulation uses the same circuits as episodic memory.

Implementation:
  DMN activates when external arousal is LOW (boredom, rest).
  When active, it replays episodic memories and generates self-related
  content that biases the next attention cycle (PCC → thalamus).
  This is the "daydreaming" state already partially implemented in brain.py
  via `_is_daydreaming` — here we give it a structured form.
"""

from __future__ import annotations
import numpy as np
import random


class DefaultModeNetwork:

    def __init__(self, pattern_size: int):
        """
        pattern_size: size of activity patterns stored in episodic memory
        """
        self.pattern_size = pattern_size

        # Running DMN activation level
        self.activation: float = 0.0

        # Self-relevance signal from mPFC — rises when homeostatic state is salient
        self.self_relevance: float = 0.0

        # Currently replayed autobiographical content (injected into raw sensory)
        self.replay_pattern: np.ndarray = np.zeros(pattern_size, dtype=np.float32)

    def update(
        self,
        external_arousal: float,        # mean neural activity / thalamic drive
        episodic_memory:  list,         # list of episode dicts from Hippocampus
        wellbeing:        float,        # current homeostatic wellbeing
    ) -> dict:
        """Compute DMN state for one tick.

        Returns:
          activation    : float [0,1] — DMN activity level
          self_relevance: float [0,1] — mPFC self-referential signal
          replay_pattern: np.ndarray  — autobiographical content for injection
        """
        # DMN anti-correlates with external attention
        target_dmn = max(0.0, 1.0 - external_arousal * 2.0)
        self.activation = float(0.95 * self.activation + 0.05 * target_dmn)

        # Self-relevance: rises when something is wrong (homeostatic deficit)
        deficit = max(0.0, 1.0 - wellbeing)
        self.self_relevance = float(
            np.clip(0.9 * self.self_relevance + 0.1 * deficit, 0.0, 1.0))

        # Autobiographical replay when DMN is active and memories exist
        self.replay_pattern[:] = 0.0
        if self.activation > 0.3 and episodic_memory:
            episode = random.choice(episodic_memory)
            pattern = episode.get('pattern', np.zeros(self.pattern_size))
            n = min(len(pattern), self.pattern_size)
            # Amplitude scaled by DMN activation and self-relevance
            self.replay_pattern[:n] = (
                pattern[:n] * self.activation * (0.5 + self.self_relevance * 0.5))

        return {
            'activation':     self.activation,
            'self_relevance': self.self_relevance,
            'replay_pattern': self.replay_pattern,
        }
