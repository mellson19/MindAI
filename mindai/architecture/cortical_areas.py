"""Cortical areas — functional specialisation of neocortical regions.

Biological basis (Zeki 1978; Ungerleider & Mishkin 1982; Rizzolatti 1996):
  The neocortex is subdivided into functionally specialised areas.  Each area
  has a distinct cytoarchitecture, connectivity pattern, and response profile.

  Visual hierarchy (ventral + dorsal streams):
    V1   — primary visual cortex; oriented edge detection (Hubel & Wiesel 1962)
    V2   — early visual; illusory contours
    V4   — colour and form (Zeki 1983)
    MT/V5 — motion detection (Zeki 1974; Newsome 1988)
    IT   — object recognition; invariant form representation (DiCarlo 2012)
    LOC  — lateral occipital; shape and 3D form

  Dorsal stream ("where/how"):
    V3A, V7 — spatial vision
    LIP     — parietal eye field; salience map; decision-related activity
    AIP     — grasp affordances (Sakata 1995)
    PMv/F5  — premotor hand grasp (mirror neurons here: Rizzolatti 1996)

  Auditory:
    A1  — primary auditory; tonotopic (low → high frequency medial → lateral)
    Belt (A2) — complex sounds, voice-selective
    STS — multisensory (voice+face integration; social observation)

  Somatosensory / Motor:
    S1  — primary somatosensory; homunculus (Penfield 1950)
    S2  — secondary somatosensory; pain affect
    M1  — primary motor; direct corticospinal output
    SMA — supplementary motor area; movement sequencing
    PMd — dorsal premotor; movement preparation

  Prefrontal:
    dlPFC (area 46/9) — working memory, cognitive control
    vmPFC (area 11/13) — somatic markers, reward valuation
    OFC   — economic value, sensory-specific satiety
    ACC   (area 24/32) — conflict monitoring (→ anterior_cingulate.py)
    IFG/Broca (area 44/45) — language production; mirror neurons

Implementation:
  This module maintains a topographic activity map — which areas are
  currently most active — derived from the neuron index range.
  Used by brain.py for area-specific modulation and reporting.

  Area boundaries are approximated from allometric proportions of
  the human neocortex (Elston 2003; Van Essen 2004).
"""

from __future__ import annotations
import numpy as np


# Area definitions: name → (fraction_start, fraction_end)
# Fractions of total neuron count.  Approximate but consistent.
AREA_FRACTIONS: dict[str, tuple[float, float]] = {
    # Visual hierarchy
    'V1':    (0.00, 0.06),
    'V2':    (0.06, 0.10),
    'V4':    (0.10, 0.13),
    'MT':    (0.13, 0.15),
    'IT':    (0.15, 0.19),
    # Dorsal stream
    'LIP':   (0.19, 0.22),
    'AIP':   (0.22, 0.24),
    # Auditory
    'A1':    (0.24, 0.27),
    'STS':   (0.27, 0.30),
    # Somatosensory / Motor
    'S1':    (0.30, 0.35),
    'M1':    (0.35, 0.39),
    'SMA':   (0.39, 0.41),
    'PMv':   (0.41, 0.44),   # mirror neuron area (F5 homolog)
    # Prefrontal
    'dlPFC': (0.44, 0.52),
    'vmPFC': (0.52, 0.56),
    'OFC':   (0.56, 0.60),
    'IFG':   (0.60, 0.64),   # Broca's area / language / mirror
    # Limbic / sub-cortical projection
    'ACC':   (0.64, 0.68),
    # Association
    'PCC':   (0.68, 0.73),   # DMN hub
    'TPJ':   (0.73, 0.78),   # theory of mind / multisensory
    'mPFC':  (0.78, 0.84),   # self-reference
    # Motor output / remaining
    'other': (0.84, 1.00),
}


class CorticalAreas:

    def __init__(self, num_neurons: int):
        self.num_neurons = num_neurons
        # Pre-compute slices
        self._slices: dict[str, slice] = {}
        for name, (f0, f1) in AREA_FRACTIONS.items():
            start = int(num_neurons * f0)
            stop  = int(num_neurons * f1)
            self._slices[name] = slice(start, stop)

        # Per-area running mean activity
        self.area_activity: dict[str, float] = {k: 0.0 for k in AREA_FRACTIONS}

    def slice(self, area: str) -> slice:
        return self._slices[area]

    def update(self, activity: np.ndarray) -> dict[str, float]:
        """Compute mean activity per area; update running averages."""
        for name, sl in self._slices.items():
            chunk = activity[sl]
            if len(chunk):
                self.area_activity[name] = float(
                    0.9 * self.area_activity[name] + 0.1 * np.mean(chunk))
        return self.area_activity

    def most_active(self, top_n: int = 3) -> list[str]:
        """Return names of the top_n most active areas this tick."""
        return sorted(
            self.area_activity, key=self.area_activity.__getitem__, reverse=True
        )[:top_n]
