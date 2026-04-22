import numpy as np
import scipy.sparse as sp

class DelayQueue:

    def __init__(self, num_nodes: int, max_delay_ticks: int=50):
        self.num_nodes = num_nodes
        self.max_delay_ticks = max_delay_ticks
        self.spike_queue = [sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32) for _ in range(max_delay_ticks)]
        self.current_tick_idx = 0

    def add_spikes_to_queue(self, active_nodes: np.ndarray, delay_matrix: np.ndarray, weights: sp.csr_matrix):
        active_indices = np.where(active_nodes)[0]
        if len(active_indices) == 0:
            return
        for src in active_indices:
            row = weights.getrow(src)
            for tgt, weight in zip(row.indices, row.data):
                delay = int(delay_matrix[src, tgt])
                delay = min(delay, self.max_delay_ticks - 1)
                arrival_idx = (self.current_tick_idx + delay) % self.max_delay_ticks
                self.spike_queue[arrival_idx][src, tgt] += weight

    def get_arriving_signals(self) -> np.ndarray:
        arrived_matrix = self.spike_queue[self.current_tick_idx]
        total_incoming_potential = np.array(arrived_matrix.sum(axis=0)).flatten()
        self.spike_queue[self.current_tick_idx] = sp.lil_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.current_tick_idx = (self.current_tick_idx + 1) % self.max_delay_ticks
        return total_incoming_potential