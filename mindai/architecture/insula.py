"""Insula — interoception and body-state integration.

Biological basis (Craig 2002, 2009; Damasio 1994):
  The insula (insular cortex) receives interoceptive signals from the body
  (heartbeat, respiration, gut tension, muscle proprioception, nociception)
  and integrates them into a global body-state representation.

  Posterior insula: primary interoceptive cortex — receives visceral afferents
  via lamina I spinothalamic tract (Craig 2002).  Maps body-state topographically.

  Anterior insula: re-represents posterior insula in an abstracted,
  emotionally-valenced form.  Projects to ACC, OFC, amygdala.

  Subjective feeling states (Craig 2009):
    The anterior insula is activated by: heartbeat awareness, hunger, thirst,
    temperature, pain affect, disgust, social exclusion, empathy for pain.
    This is the substrate of "feeling" a body state.

  Mirror neuron role (Singer 2004; Carr 2003):
    Insula + ACC activate during both personal pain AND observed pain.
    The insula routes the emotional body-state of the observer toward the
    CeA→premotor path — enabling empathic resonance without scripting.

Implementation:
  Posterior insula: weighted average of nociceptive, visceral, and
    proprioceptive channels.
  Anterior insula: smoothed re-representation with emotional valence.
  Output: body_valence scalar [-1, 1] (negative = aversive)
"""

from __future__ import annotations
import numpy as np


class Insula:

    def __init__(self):
        # Running smoothed body-state (posterior insula)
        self.body_state: float = 0.0

        # Emotional valence of body state (anterior insula) [-1, 1]
        self.body_valence: float = 0.0

        # Interoceptive awareness — how clearly the brain is "feeling" the body
        # Rises with sustained interoceptive input, decays at rest
        self.interoceptive_awareness: float = 0.0

    def update(
        self,
        pain:    float,   # nociceptive signal [0,1]
        hunger:  float,   # energy deficit [0,1]
        thirst:  float,   # water deficit [0,1]
        arousal: float,   # global neural arousal [0,1]
    ) -> dict:
        """Compute body-state and emotional valence for one tick.

        Returns:
          body_state             : float [0,1] — posterior insula activity
          body_valence           : float [-1,1] — anterior insula valence
          interoceptive_awareness: float [0,1]
        """
        # Posterior insula: magnitude of all body signals
        raw_body = float(np.clip(
            pain * 0.5 + hunger * 0.25 + thirst * 0.25, 0.0, 1.0))
        self.body_state = float(0.85 * self.body_state + 0.15 * raw_body)

        # Anterior insula: valence — negative when body needs unmet
        raw_valence = -pain * 0.6 - hunger * 0.2 - thirst * 0.2
        self.body_valence = float(
            np.clip(0.9 * self.body_valence + 0.1 * raw_valence, -1.0, 1.0))

        # Interoceptive awareness: rises when body signals are strong
        if raw_body > 0.1 or arousal > 0.6:
            self.interoceptive_awareness = min(
                1.0, self.interoceptive_awareness + 0.01)
        else:
            self.interoceptive_awareness *= 0.995

        return {
            'body_state':              self.body_state,
            'body_valence':            self.body_valence,
            'interoceptive_awareness': self.interoceptive_awareness,
        }
