import torch
import numpy as np

class PhaseCoupledWorkspace:

    def __init__(self, num_nodes: int, device: torch.device):
        self.num_nodes = num_nodes
        self.device = device
        self.history_buffer = []
        self.phases = torch.rand(num_nodes, device=self.device) * 2 * torch.pi
        self.natural_frequencies = torch.randn(num_nodes, device=self.device) * 0.02 + 0.1
        self.ignition_threshold = 0.7
        self.is_ignited = False
        self.ignition_cooldown = 0

    def broadcast_via_synchrony(self, salient_signal: torch.Tensor, network_activity: torch.Tensor) -> torch.Tensor:
        self.phases = (self.phases + self.natural_frequencies) % (2 * torch.pi)
        salient_mask = salient_signal > 0.1
        active_mask = network_activity > 0.1
        if salient_mask.any() and active_mask.any():
            leader_idx = torch.argmax(salient_signal)
            leader_phase = self.phases[leader_idx]
            coupling_strength = 0.5
            phase_diff = torch.sin(leader_phase - self.phases[active_mask])
            self.phases[active_mask] += coupling_strength * phase_diff
        phase_gate = (torch.sin(self.phases) + 1.0) / 2.0
        rhythmic_activity = network_activity * phase_gate
        if self.ignition_cooldown > 0:
            self.ignition_cooldown -= 1
        R = self.calculate_integration_metric(rhythmic_activity)
        if R > self.ignition_threshold and self.ignition_cooldown == 0:
            self.is_ignited = True
            self.ignition_cooldown = 30
            rhythmic_activity = torch.clamp(rhythmic_activity * 3.0, 0.0, 1.0)
        elif R < 0.3:
            self.is_ignited = False
        if active_mask.any() and phase_gate[active_mask].mean() > 0.8:
            self._snapshot_to_hippocampus(rhythmic_activity.cpu().numpy())
        return rhythmic_activity

    def _snapshot_to_hippocampus(self, activity: np.ndarray):
        self.history_buffer.append(activity.copy())
        if len(self.history_buffer) > 50:
            self.history_buffer.pop(0)

    def calculate_integration_metric(self, activity: torch.Tensor) -> float:
        active_mask = activity > 0.1
        if active_mask.sum() < 5:
            return 0.0
        phases_active = self.phases[active_mask]
        complex_x = torch.cos(phases_active).mean()
        complex_y = torch.sin(phases_active).mean()
        synchrony_R = torch.sqrt(complex_x ** 2 + complex_y ** 2)
        return synchrony_R.item()