import torch

class Thalamus:
    """Thalamocortical attention gate.

    Biological basis (Aston-Jones 2005; McCormick & Bal 1997):
    - LC noradrenaline acts on α₁ adrenergic receptors of thalamic relay
      neurons → MULTIPLICATIVE gain increase, not threshold shift.
      High NA amplifies strong signals and weak signals equally, improving
      signal-to-noise ratio (SNR) rather than raising the bar.
    - Boredom / low LC tone → burst mode (T-type Ca²⁺ channel de-inactivation)
      → lower effective threshold for novel input (paradoxically easier to
      capture attention when bored and something new appears).
    - Thalamic reticular nucleus provides feedback inhibition → winner-take-all
      via lateral inhibition among relay neurons.

    Threshold adapts to mean activity so it scales across any network size.
    """

    def __init__(self, num_nodes: int, device: torch.device):
        self.num_nodes = num_nodes
        self.device    = device
        # Running estimate of mean network activity (exponential moving average)
        self._mean_activity = 0.3   # initialised at typical resting level

    def filter_attention(
        self,
        raw_activity:      torch.Tensor,
        noradrenaline_level: float,
        boredom_level:     float = 0.0,
    ) -> torch.Tensor:
        # --- Adaptive threshold: mean activity + half a standard deviation ---
        # Scales automatically to any network size; replaces hardcoded 0.6.
        mean_act = float(raw_activity.mean())
        self._mean_activity = 0.99 * self._mean_activity + 0.01 * mean_act
        # Boredom lowers threshold (burst mode — easier to alert when idle)
        base_threshold = self._mean_activity + 0.15 - boredom_level * 0.08

        # --- NA: multiplicative gain on relay neurons (α₁ adrenergic) ---
        # High NA amplifies the signal, it does NOT change the threshold.
        gain = 1.0 + noradrenaline_level * 0.6   # up to 1.6× at max NA

        amplified = (raw_activity * gain).clamp(0.0, 1.0)

        # --- Thalamic reticular nucleus: winner-take-all ---
        salient_signal = torch.zeros_like(amplified)
        active_mask = amplified > base_threshold
        if active_mask.any():
            max_activation = torch.max(amplified[active_mask])
            winners_mask   = amplified >= max_activation * 0.9
            salient_signal[winners_mask] = amplified[winners_mask]
        return salient_signal