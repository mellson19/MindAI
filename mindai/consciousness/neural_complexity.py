"""Neural complexity — Lempel-Ziv complexity of firing patterns.

Biological basis (Tononi, Sporns & Edelman 1994):
  Neural complexity CN = mutual information between all bipartitions of the network.
  High CN = neither fully segregated (random) nor fully integrated (uniform) —
  the brain operates at maximal complexity between these extremes.

  Lempel-Ziv complexity (LZ76) approximates CN efficiently:
  - Compress the binary spike vector → short = repetitive (low complexity)
  - Long compressed representation = rich, differentiated activity = high complexity

  This is purely a measure of the information richness of firing patterns —
  no theoretical consciousness claims, no Φ, no partition minimisation.
  LZ complexity of EEG/LFP is experimentally measurable and correlates with
  arousal states (Casali 2013: highest in waking, lower in NREM, lowest in
  anaesthesia, intermediate in REM).

  Spectral radius of the connectivity sub-matrix reflects criticality:
  near-critical networks (ρ ≈ 1) show maximal dynamic range and information
  transmission (Beggs & Plenz 2003).
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def _lz76_complexity(binary_seq: np.ndarray) -> float:
    """Lempel-Ziv 1976 complexity — O(n log n)."""
    s   = binary_seq.astype(bool)
    n   = len(s)
    if n == 0:
        return 0.0
    c   = 1
    l   = 1
    i   = 0
    k   = 1
    k_max = 1
    stop = False
    while not stop:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c    += 1
                stop  = True
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c    += 1
                l    += k_max
                if l + 1 > n:
                    stop = True
                else:
                    i     = 0
                    k     = 1
                    k_max = 1
            else:
                k = 1
    return float(c)


class NeuralComplexity:
    """Measures richness of neural activity — no theoretical claims."""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def calculate(
        self,
        activity:         np.ndarray,          # (N,) float, firing rates
        weights_csr:      object = None,        # kept for back-compat; ignored when spectral_radius given
        *,
        spectral_radius:  float | None = None,  # precomputed GPU power-iteration value
    ) -> list[float]:
        """Return [lz_norm, spectral_radius_proxy, mean_activity].

        lz_norm:              LZ76 / (N / log2 N+1) — normalised to [0,1]
        spectral_radius_proxy: spectral radius proxy (Beggs 2003)
        mean_activity:        fraction of neurons firing (sparsity measure)

        If `spectral_radius` is provided (precomputed on GPU via power iteration),
        the CSR-based fallback is skipped entirely — no scipy matrix materialisation.
        """
        active = activity > 0.5
        n_active = int(active.sum())

        # --- LZ complexity of binary spike vector ---
        binary = (activity > 0.5).astype(np.uint8)
        lz_raw  = _lz76_complexity(binary)
        n       = len(binary)
        lz_norm = lz_raw / (n / (np.log2(n + 1) + 1e-6))
        lz_norm = float(np.clip(lz_norm, 0.0, 1.0))

        # --- Spectral radius proxy ---
        if spectral_radius is not None:
            # Precomputed on GPU — use directly, normalised to [0,1]
            sr_proxy = float(np.clip(spectral_radius / max(1, n_active), 0.0, 1.0))
        elif weights_csr is not None and n_active > 3:
            idx = np.where(active)[0]
            if idx.max() < weights_csr.shape[0]:
                sub = weights_csr[np.ix_(idx, idx)]
                v = np.random.randn(n_active).astype(np.float32)
                nrm = 0.0
                for _ in range(8):
                    v = sub.dot(v)
                    nrm = np.linalg.norm(v)
                    if nrm > 0:
                        v /= nrm
                sr_proxy = float(np.clip(nrm / (n_active + 1e-6), 0.0, 1.0))
            else:
                sr_proxy = 0.33
        else:
            sr_proxy = 0.33

        mean_act = float(np.mean(active))

        return [lz_norm, sr_proxy, mean_act]
