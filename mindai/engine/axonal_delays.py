"""Axonal delay queue — GPU ring buffer implementation.

Biological basis (Swadlow 1985; Salami 2003):
  Unmyelinated axons conduct at 0.5–2 m/s; myelinated at 5–70 m/s.
  In a cortical sphere of radius 10 cm, delays range from ~1 ms to ~200 ms.
  Delays shape synchrony windows and contribute to gamma/theta oscillations
  (König et al. 1995).

Implementation:
  Ring buffer of depth max_delay_ticks stored as a (max_delay, num_nodes) float32
  tensor on GPU. Each tick: arriving signals are read from current slot, slot is
  zeroed, buffer index advances. Spike delivery uses scatter_add_ per edge.
  O(active_edges) per tick — no CPU round-trip, no scipy.
"""

import torch


class DelayQueue:

    def __init__(
        self,
        num_nodes:       int,
        max_delay_ticks: int = 20,
        device:          torch.device | None = None,
    ):
        self.num_nodes       = num_nodes
        self.max_delay_ticks = max_delay_ticks
        self.device          = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._idx            = 0

        # Ring buffer: shape (max_delay_ticks, num_nodes) — post-synaptic potentials
        self._buf = torch.zeros(max_delay_ticks, num_nodes, device=self.device)

    def enqueue(
        self,
        activity:     torch.Tensor,   # (num_nodes,) current spike vector
        edge_src:     torch.Tensor,   # (E,) pre-synaptic indices
        edge_dst:     torch.Tensor,   # (E,) post-synaptic indices
        edge_weights: torch.Tensor,   # (E,) synaptic weights (already STP-scaled)
        edge_delays:  torch.Tensor,   # (E,) int16 delay in ticks
    ) -> None:
        """Schedule weighted spikes into future buffer slots.

        Only edges whose pre-synaptic neuron fired (activity > 0.5) generate
        a post-synaptic potential. scatter_add_ accumulates PSPs per destination.
        """
        pre_fired = activity[edge_src] > 0.5
        if not pre_fired.any():
            return

        src_f = edge_src[pre_fired]
        dst_f = edge_dst[pre_fired]
        w_f   = edge_weights[pre_fired] * activity[src_f]
        d_f   = edge_delays[pre_fired].long()

        # Clamp delays into valid ring buffer range
        d_f = d_f.clamp(1, self.max_delay_ticks - 1)

        # Target slot index in ring buffer for each spike
        slots = (self._idx + d_f) % self.max_delay_ticks  # (E_fired,)

        # Batch scatter into buffer — one scatter_add_ call per unique slot
        # (usually only a few distinct delay values → very few iterations)
        for slot_val in slots.unique():
            mask   = slots == slot_val
            target = self._buf[slot_val.item()]
            target.scatter_add_(0, dst_f[mask], w_f[mask])

    def dequeue(self) -> torch.Tensor:
        """Return PSPs arriving this tick; advance ring buffer index."""
        arriving = self._buf[self._idx].clone()
        self._buf[self._idx].zero_()
        self._idx = (self._idx + 1) % self.max_delay_ticks
        return arriving


def build_delay_tensor(
    edge_src:    torch.Tensor,
    edge_dst:    torch.Tensor,
    coordinates: "np.ndarray",         # (num_nodes, 3) float, in mm
    conduction_speed: float = 2.0,     # mm/tick at 10 Hz ≈ 20 mm/s (unmyelinated)
    device:      torch.device | None = None,
) -> torch.Tensor:
    """Pre-compute integer delay (ticks) for every edge from 3D coordinates.

    Called once at Brain init — result stored as int16 tensor on device.
    conduction_speed: 2 mm/tick at 10 Hz = 20 mm/s (Swadlow 1985 slow fibres).
    Myelinated fibres would use 50–70 mm/tick — kept slow for cortical local circuits.
    """
    import numpy as np
    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_np = edge_src.cpu().numpy()
    dst_np = edge_dst.cpu().numpy()
    diff   = coordinates[src_np] - coordinates[dst_np]
    dist   = np.sqrt((diff ** 2).sum(axis=1))
    delays = np.maximum(1, (dist / conduction_speed).astype(np.int16))
    return torch.tensor(delays, dtype=torch.int16, device=dev)
