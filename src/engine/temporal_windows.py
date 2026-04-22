import numpy as np
import torch

class HusserlianTime:

    def __init__(self, num_nodes: int, window_size: int=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes = num_nodes
        self.window_size = window_size
        self.retention_buffer = torch.zeros((window_size, num_nodes), device=self.device)
        decay_weights = torch.exp(-torch.arange(self.window_size, device=self.device) / 2.0)
        self.decay_weights = decay_weights / torch.sum(decay_weights)

    def create_conscious_now(self, primal_impression: torch.Tensor, protention_forecast: torch.Tensor) -> torch.Tensor:
        self.retention_buffer = torch.roll(self.retention_buffer, shifts=1, dims=0)
        self.retention_buffer[0] = primal_impression
        retention_smear = torch.sum(self.retention_buffer * self.decay_weights.unsqueeze(1), dim=0)
        thick_now = primal_impression * 0.6 + retention_smear * 0.3 + protention_forecast * 0.1
        return thick_now