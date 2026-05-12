import torch
import numpy as np
from collections import deque

class PhaseCoupledWorkspace:
    """Kuramoto phase-coupled global workspace (Baars 1988; Dehaene 2011).

    Biological basis:
    - Gamma-band (40 Hz) phase synchrony across cortical areas mediates
      conscious access (Tallon-Baudry & Bertrand 1999).
    - The "leader" neuron (highest salience) entrains others via Kuramoto
      coupling, approximating long-range corticocortical coherence.
    - Coupling strength scales with salience (not fixed) — high-salience
      signals recruit broader coalitions (Dehaene ignition).
    - Ignition threshold adaptive: tracks recent synchrony history so it
      scales across any network size and arousal state.
    - history_buffer: ring buffer of activity snapshots at high phase-gate
      moments; read by SemanticMemory during N2 sleep for concept extraction.
    """

    def __init__(self, num_nodes: int, device: torch.device):
        self.num_nodes = num_nodes
        self.device    = device
        # Ring buffer: read by sleep_consolidation → semantic_memory during N2
        self.history_buffer = deque(maxlen=50)
        # Phase oscillator — heterogeneous natural frequencies (Kuramoto 1984)
        # Biological gamma: 30–80 Hz; here normalised to [0.06, 0.14] rad/tick
        self.phases = torch.rand(num_nodes, device=self.device) * 2 * torch.pi
        self.natural_frequencies = (
            torch.rand(num_nodes, device=self.device) * 0.08 + 0.06)
        # Adaptive ignition threshold: running mean of recent synchrony R
        self._recent_R        = deque(maxlen=100)
        self._ignition_bias   = 0.15   # ignition fires when R > mean + bias
        self.ignition_threshold = 0.7  # initial value; adapts each tick
        self.is_ignited       = False
        self.ignition_cooldown = 0

    def broadcast_via_synchrony(
        self,
        salient_signal:   torch.Tensor,
        network_activity: torch.Tensor,
    ) -> torch.Tensor:
        # Phase advance
        self.phases = (self.phases + self.natural_frequencies) % (2 * torch.pi)

        active_mask  = network_activity > 0.1
        salient_mask = salient_signal   > 0.1

        if salient_mask.any() and active_mask.any():
            leader_idx   = torch.argmax(salient_signal)
            leader_phase = self.phases[leader_idx]

            # Coupling strength scales with leader salience (Dehaene 2011):
            # a barely-salient signal recruits a small coalition; a strong
            # signal pulls the whole active population into phase lock.
            leader_salience  = float(salient_signal[leader_idx])
            coupling_strength = float(torch.clamp(
                torch.tensor(leader_salience * 0.8), 0.1, 0.9))

            phase_diff = torch.sin(leader_phase - self.phases[active_mask])
            self.phases[active_mask] += coupling_strength * phase_diff

        phase_gate      = (torch.sin(self.phases) + 1.0) / 2.0
        rhythmic_activity = network_activity * phase_gate

        # --- Adaptive ignition threshold ---
        if self.ignition_cooldown > 0:
            self.ignition_cooldown -= 1
        R = self.calculate_integration_metric(rhythmic_activity)
        self._recent_R.append(R)
        if len(self._recent_R) >= 10:
            self.ignition_threshold = float(np.mean(self._recent_R)) + self._ignition_bias

        if R > self.ignition_threshold and self.ignition_cooldown == 0:
            self.is_ignited = True
            self.ignition_cooldown = 30
            rhythmic_activity = torch.clamp(rhythmic_activity * 3.0, 0.0, 1.0)
        elif R < 0.3:
            self.is_ignited = False

        # Snapshot at high phase coherence → read by SemanticMemory during N2.
        # Only when buffer has room — avoids redundant PCIe transfers once full.
        if (active_mask.any()
                and phase_gate[active_mask].mean() > 0.8
                and len(self.history_buffer) < self.history_buffer.maxlen):
            self.history_buffer.append(rhythmic_activity.cpu().numpy().copy())

        return rhythmic_activity

    def calculate_integration_metric(self, activity: torch.Tensor) -> float:
        active_mask = activity > 0.1
        if active_mask.sum() < 5:
            return 0.0
        phases_active = self.phases[active_mask]
        R = torch.sqrt(
            torch.cos(phases_active).mean() ** 2 +
            torch.sin(phases_active).mean() ** 2)
        return float(R)