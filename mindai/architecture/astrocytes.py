"""Astrocytes — glial regulation of synaptic stability.

Biological basis
----------------
Astrocytes form the third element of the tripartite synapse (Araque 1999).
They perform several critical roles that point neurons miss:

  1. Glutamate clearance (EAAT1/2 transporters) — prevents excitotoxicity.
     High-activity neurons get aggressive scavenging.
  2. K⁺ buffering (Kir4.1 channels) — sets resting potential, sinks
     accumulated extracellular K⁺ from heavy firing.
  3. GABA recycling — astrocytes shuttle glutamine back to GABAergic neurons.
  4. Synaptic pruning via complement (C1q/C3) tags — Stevens 2007:
     unused synapses get marked for microglial removal during development
     and sleep.
  5. Slow Ca²⁺ waves modulating long-term plasticity (Volterra & Meldolesi
     2005) — provide a slower stability layer beneath fast STDP.

Function in MindAI
------------------
We do NOT simulate Ca²⁺ waves directly. Instead astrocytes provide a
slow per-edge stability variable that acts as a filter on rapid weight
changes — preventing STDP "noise" from accumulating into permanent drift.

Mechanism (per tick during awake state):
    For every synapse, track a slow EMA of its weight (time constant
    ~5000 ticks). Each tick, weights are pulled gently back toward this
    EMA — but only when the synapse is currently inactive. Active synapses
    keep their fast STDP changes intact; inactive ones decay back toward
    their last stable state instead of staying at peaks driven by transient
    co-firing.

This is mathematically equivalent to a "slow plasticity" layer (Fusi 2005
cascade memory) that gives long-term stability without erasing short-term
learning.

References
----------
- Araque A et al. (1999). Tripartite synapses: glia, the unacknowledged
  partner. Trends Neurosci 22: 208-215.
- Volterra A, Meldolesi J (2005). Astrocytes, from brain glue to
  communication elements. Nat Rev Neurosci 6: 626-640.
- Stevens B et al. (2007). The classical complement cascade mediates CNS
  synapse elimination. Cell 131: 1164-1178.
- Fusi S, Drew PJ, Abbott LF (2005). Cascade models of synaptically
  stored memories. Neuron 45: 599-611.
"""

from __future__ import annotations

import torch


class Astrocytes:
    """Slow weight-stability layer atop fast STDP (Fusi cascade memory)."""

    # τ ≈ 5000 ticks → α = 0.9998 per-tick EMA
    _SLOW_DECAY = 0.9998
    # Pull-back rate for inactive synapses — gentle, biological
    _STABILITY_PULL = 0.001

    def __init__(self, device: torch.device | None = None):
        self.device         = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._slow_weights: torch.Tensor | None = None
        self._initialized   = False

    def initialize(self, current_weights: torch.Tensor) -> None:
        """Bind to a weight vector at the current size. Call after plasticity init."""
        self._slow_weights = current_weights.detach().clone()
        self._initialized  = True

    def step(
        self,
        weights:  torch.Tensor,
        pre_idx:  torch.Tensor,
        post_idx: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        """Apply astrocytic stabilisation. Returns updated weights tensor.

        Active synapses: untouched (fast STDP wins).
        Inactive synapses: pulled toward their slow EMA — the stable
        long-term value the astrocyte "remembers" for that connection.
        """
        # Resize if topology grew (neurogenesis added synapses)
        if (self._slow_weights is None
                or self._slow_weights.shape[0] != weights.shape[0]):
            n_extra = weights.shape[0] - (
                0 if self._slow_weights is None else self._slow_weights.shape[0])
            if self._slow_weights is None:
                self._slow_weights = weights.detach().clone()
            elif n_extra > 0:
                # New synapses: seed slow EMA at their current value
                extra = weights[-n_extra:].detach().clone()
                self._slow_weights = torch.cat([self._slow_weights, extra])
            else:
                # Pruning shrunk — truncate
                self._slow_weights = self._slow_weights[:weights.shape[0]]

        # Update slow EMA toward current weights — captures long-term value
        self._slow_weights = (self._slow_weights * self._SLOW_DECAY
                              + weights.detach() * (1.0 - self._SLOW_DECAY))

        # Determine active synapses: either pre or post fired
        active_pre  = activity[pre_idx]  > 0.5
        active_post = activity[post_idx] > 0.5
        active_syn  = active_pre | active_post

        # Pull inactive synapses toward slow EMA — preserves long-term memory
        # without forbidding short-term STDP on active ones
        inactive = ~active_syn
        if inactive.any():
            delta = (self._slow_weights[inactive] - weights[inactive]) * self._STABILITY_PULL
            weights[inactive] = weights[inactive] + delta

        return weights

    def state_dict(self) -> dict:
        return {'slow_weights': self._slow_weights.cpu().numpy()
                                if self._slow_weights is not None else None}

    def load_state_dict(self, state: dict) -> None:
        sw = state.get('slow_weights')
        if sw is not None:
            self._slow_weights = torch.tensor(sw, device=self.device)
            self._initialized  = True
