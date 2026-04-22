import torch

class Thalamus:

    def __init__(self, num_nodes: int, device: torch.device):
        self.num_nodes = num_nodes
        self.device = device
        self.activation_threshold = 0.6

    def filter_attention(self, raw_activity: torch.Tensor, noradrenaline_level: float, boredom_level: float=0.0) -> torch.Tensor:
        salient_signal = torch.zeros_like(raw_activity)
        dynamic_shift = noradrenaline_level * 0.3 - boredom_level * 0.4
        current_threshold = torch.clamp(torch.tensor(self.activation_threshold + dynamic_shift), min=0.15, max=0.9).to(self.device)
        active_mask = raw_activity > current_threshold
        if active_mask.any():
            max_activation = torch.max(raw_activity[active_mask])
            winners_mask = raw_activity >= max_activation * 0.9
            salient_signal[winners_mask] = raw_activity[winners_mask]
        return salient_signal