"""BrainGeometry — Fibonacci-sphere coordinates for axonal-delay calculation.

Coordinates only — never the full N×N distance matrix (would be 3.6 TB for
the 400k-neuron default config).  Pairwise distances are computed on-demand
in vectorised form by ``build_delay_tensor`` (engine/axonal_delays.py),
which only needs distances for *existing* edges, i.e. O(synapses).
"""

import numpy as np


class BrainGeometry:

    def __init__(self, num_nodes: int, radius: float = 10.0):
        self.num_nodes   = num_nodes
        self.radius      = radius
        self.coordinates = self._generate_spherical_coordinates()

    def _generate_spherical_coordinates(self) -> np.ndarray:
        # Vectorised Fibonacci sphere — O(N), no Python loop
        n   = self.num_nodes
        idx = np.arange(n, dtype=np.float64)
        phi   = np.arccos(1.0 - 2.0 * (idx + 0.5) / n)
        theta = np.pi * (1.0 + 5.0 ** 0.5) * idx
        sin_phi = np.sin(phi)
        coords = np.empty((n, 3), dtype=np.float32)
        coords[:, 0] = self.radius * np.cos(theta) * sin_phi
        coords[:, 1] = self.radius * np.sin(theta) * sin_phi
        coords[:, 2] = self.radius * np.cos(phi)
        return coords

    def axonal_delay(self, node_a: int, node_b: int,
                     speed_of_conduction: float = 2.0) -> int:
        """Compute one pairwise delay on demand (O(1), no preallocated matrix)."""
        d = float(np.linalg.norm(self.coordinates[node_a] - self.coordinates[node_b]))
        return max(1, int(d / speed_of_conduction))