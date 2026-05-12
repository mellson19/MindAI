"""Olfactory bulb — non-thalamic sensory route, direct to limbic system.

Biological basis
----------------
Olfaction is unique among senses: it BYPASSES the thalamus and projects
directly to the piriform cortex and amygdala (Shepherd 2007). This
single anatomical fact explains why smell is so emotionally evocative —
it has a fast-track to fear/memory circuits that vision and audition
lack.

Pathway:
    olfactory receptor neurons → glomeruli (~2000 in human)
        → mitral cells (one per glomerulus, sparse code)
        → piriform cortex + AMYGDALA + entorhinal (NO thalamic relay)

Each odorant activates a unique combinatorial pattern of glomeruli
(Buck & Axel 1991 — Nobel 2004). Even with very few receptor types
(~350 in humans) the combinatorial code distinguishes thousands of
odours.

Why MindAI optionally needs this
--------------------------------
For text/multimodal agents olfaction is not directly used — there's no
physical odour input. But the architecture matters: the OB's direct-to-
amygdala pathway is a useful template for any future "fast aversive"
channel that needs to bypass cortical processing.

This module is a stub that becomes useful when:
  - Embodied robots add real chemical sensors (gas/CO2 detectors etc.).
  - Synthetic "odour-like" features are derived from auxiliary sensors
    (e.g., quick intensity histograms that should trigger amygdala fast).
  - Text-only mode: word-level "valence words" (smell-like primal
    signals) can be routed through here for direct emotional response
    without thalamic gating.

For now we provide the pathway and let it sit dormant if no input arrives.

References
----------
- Buck L, Axel R (1991). A novel multigene family may encode odorant
  receptors. Cell 65: 175-187.
- Shepherd GM (2007). Perspectives on olfactory processing.
  Ann N Y Acad Sci 1170: 87-101.
"""

from __future__ import annotations

import numpy as np


class OlfactoryBulb:
    """Combinatorial glomerular code → mitral cell sparse output.

    Bypasses thalamus — output goes directly to amygdala and piriform.
    """

    def __init__(self, num_glomeruli: int = 256, num_mitral: int = 256,
                 rng_seed: int = 11):
        rng = np.random.default_rng(rng_seed)
        # Glomeruli aren't independent — each odorant activates a sparse subset
        self.num_glomeruli = num_glomeruli
        self.num_mitral    = num_mitral
        # Glomeruli → mitral: 1:1 in mammals (one mitral per glomerulus)
        self._W_glom_mitral = np.eye(num_glomeruli, num_mitral, dtype=np.float32)
        # Lateral inhibition between mitral cells (granular interneurons)
        self._lateral_inhibition = (
            rng.random((num_mitral, num_mitral)).astype(np.float32) * 0.05)
        np.fill_diagonal(self._lateral_inhibition, 0.0)

        self.last_mitral: np.ndarray = np.zeros(num_mitral, dtype=np.float32)
        # Direct projection to amygdala — exposed for Brain to read
        self.amygdala_signal: float = 0.0

    def update(self, glomerular_input: np.ndarray | None = None) -> np.ndarray:
        """Process glomerular input and return mitral cell output.

        glomerular_input: (num_glomeruli,) array, values in [0,1].
                          None or zeros → no smell, mitral output decays.
        """
        if glomerular_input is None or not np.any(glomerular_input):
            self.last_mitral *= 0.8
            self.amygdala_signal = 0.0
            return self.last_mitral

        glom = np.asarray(glomerular_input, dtype=np.float32)
        glom = glom[:self.num_glomeruli]
        if glom.size < self.num_glomeruli:
            glom = np.pad(glom, (0, self.num_glomeruli - glom.size))

        # Mitral activation
        mitral = glom @ self._W_glom_mitral
        # Lateral inhibition (granule-cell mediated, sharpens odour code)
        mitral -= mitral @ self._lateral_inhibition
        mitral = np.clip(mitral, 0.0, 1.0)

        # k-WTA: olfactory code is naturally sparse (~5% active)
        k = max(1, self.num_mitral // 20)
        if k < self.num_mitral:
            thr = np.partition(mitral, self.num_mitral - k)[self.num_mitral - k]
            mitral = np.where(mitral >= thr, mitral, 0.0)

        self.last_mitral = mitral
        # Direct-to-amygdala signal: mean activity = "intensity" of odour
        # No specific affective valence here — that's learned downstream by amygdala
        self.amygdala_signal = float(np.mean(mitral))
        return mitral
