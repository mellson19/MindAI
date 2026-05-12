"""Cortical layer canonical connectivity bias.

Biological basis (Douglas & Martin 1991; Bastos et al. 2012):
  The neocortex is organised into 6 layers within each cortical column (~0.5 mm).
  The canonical microcircuit (Douglas & Martin 1991) describes the dominant
  connectivity pattern within a column:

    L4 (granular)       — receives thalamic feedforward input
    L2/3 (supragranular)— lateral association, sends feedback to higher areas
    L5/6 (infragranular)— output to subcortical targets, feedback to thalamus

  Canonical connectivity strengths:
    L4  → L2/3 : strong excitatory (initial feedforward drive)
    L2/3 → L5  : strong excitatory (driving subcortical output)
    L5  → L4   : weak excitatory  (top-down expectation modulation)
    L2/3 → L2/3: recurrent lateral (attractor dynamics, working memory)

IMPORTANT — what this module does and does NOT do:
  - Applies the canonical weight bias ONCE at initialisation.
  - Does NOT route or suppress sensory input — that would be wrong at the
    whole-network scale (the layer structure exists within individual columns,
    not as global bands across 12k neurons).
  - STDP reshapes all weights freely after init. The bias only sets the
    starting connectivity distribution.
  - Layer index ranges here are a coarse approximation: real cortical areas
    each have their own layer structure with different proportions.
    CorticalAreas (cortical_areas.py) provides the area-level specialisation.
"""

from __future__ import annotations
import torch


class CorticalLayers:

    def __init__(self, num_nodes: int, device: torch.device):
        self.num_nodes = num_nodes
        self.device    = device

        # Approximate layer boundaries for the whole network.
        # These set initial weight biases only — not used for routing.
        # Allometric fractions: L4=20%, L2/3=50%, L5=30% (Elston 2003).
        l4_end  = int(num_nodes * 0.20)
        l23_end = int(num_nodes * 0.70)

        self.l4_slice  = slice(0,       l4_end)
        self.l23_slice = slice(l4_end,  l23_end)
        self.l5_slice  = slice(l23_end, num_nodes)

    def apply_canonical_bias(self, plasticity) -> None:
        """Bias initial weights toward canonical microcircuit (Douglas & Martin 1991).

        Called once at Brain init. STDP reshapes everything freely afterward.
        """
        src = plasticity.indices[0]
        dst = plasticity.indices[1]

        def in_layer(idx, sl):
            return (idx >= sl.start) & (idx < sl.stop)

        masks_and_factors = [
            (in_layer(src, self.l4_slice)  & in_layer(dst, self.l23_slice), 2.0),
            (in_layer(src, self.l23_slice) & in_layer(dst, self.l5_slice),  1.8),
            (in_layer(src, self.l5_slice)  & in_layer(dst, self.l4_slice),  0.5),
            (in_layer(src, self.l23_slice) & in_layer(dst, self.l23_slice), 1.5),
        ]
        for mask, factor in masks_and_factors:
            if mask.any():
                plasticity.weights_values[mask] = torch.clamp(
                    plasticity.weights_values[mask] * factor, -1.0, 1.0)
        plasticity._topology_changed = True
