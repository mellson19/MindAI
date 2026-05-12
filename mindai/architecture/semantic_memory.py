"""SemanticMemory — concept extraction from episodic replay during N2 sleep.

Biological basis (Stickgold 2005; Diekelmann & Born 2010):
  Semantic gist is extracted during N2 (spindle phase) when the hippocampus
  replays episodic patterns into cortex.  Repeated co-activation across
  multiple episodes → Hebbian strengthening of shared cortical assemblies.

  The resulting semantic synapse represents a "prototype" — neurons that
  consistently fire together across emotionally distinct episodes become
  linked as an abstract concept (e.g. "food" regardless of which meal).

Sparsity (Olshausen & Field 1996):
  Semantic representations are SPARSE, not all-to-all.  Neurons that
  co-activate with correlation > threshold are linked; the rest are not.
  This is implemented via correlation-based selection instead of the O(N²)
  dense all-pairs graph that would explode with many active neurons.

  Max 64 new synapses per sleep call prevents runaway growth.
  Integrity starts at 0.5 (immature synapse) and grows through
  reconsolidation — not at 2.0 (fully mature) from birth.
"""

import numpy as np
import torch


_MAX_NEW_SYNAPSES  = 64    # per sleep call — prevents O(N²) explosion
_CORR_THRESHOLD    = 0.5   # minimum pairwise correlation to form a concept synapse
_INIT_INTEGRITY    = 0.5   # immature; grows through reconsolidation (not 2.0)
_INIT_WEIGHT       = 0.02  # weak initial semantic connection


class SemanticMemory:

    def extract_concept_during_sleep(
        self,
        memory_patterns,   # list or deque of np.ndarray activity snapshots
        plasticity,
    ) -> None:
        """Extract concept synapses from co-activation patterns.

        Parameters
        ----------
        memory_patterns:
            Recent high-synchrony activity snapshots from GlobalWorkspace
            history_buffer (written during phase-coherent ignition events).
        plasticity:
            StructuralPlasticity instance — concept synapses are injected here.
        """
        if not memory_patterns:
            return

        # Convert deque/list to array; handle variable-length snapshots
        patterns = list(memory_patterns)
        if len(patterns) < 2:
            return

        try:
            mat = np.stack(patterns, axis=0).astype(np.float32)   # (T, N)
        except ValueError:
            return   # snapshots have inconsistent shape (shouldn't happen)

        T, N = mat.shape

        # ------------------------------------------------------------------
        # 1. Find reliably active neurons (mean activity > 0.4 across episodes)
        # ------------------------------------------------------------------
        mean_act = mat.mean(axis=0)
        candidate_idx = np.where(mean_act > 0.4)[0]
        if len(candidate_idx) < 2:
            return

        # ------------------------------------------------------------------
        # 2. Compute pairwise correlation among candidates (sparse selection)
        # ------------------------------------------------------------------
        sub = mat[:, candidate_idx]    # (T, K)

        # Normalise columns for correlation
        mu  = sub.mean(axis=0, keepdims=True)
        std = sub.std(axis=0, keepdims=True) + 1e-8
        sub_norm = (sub - mu) / std    # (T, K)

        corr = (sub_norm.T @ sub_norm) / T   # (K, K) pairwise Pearson r

        # ------------------------------------------------------------------
        # 3. Select pairs above threshold (sparse: only strong co-activations)
        # ------------------------------------------------------------------
        K = len(candidate_idx)
        i_idx, j_idx = np.where((corr > _CORR_THRESHOLD) & (np.eye(K) == 0))

        if len(i_idx) == 0:
            return

        # Map back to full neuron indices
        src_full = candidate_idx[i_idx]
        tgt_full = candidate_idx[j_idx]

        # ------------------------------------------------------------------
        # 4. Cap at MAX_NEW_SYNAPSES (strongest correlations first)
        # ------------------------------------------------------------------
        if len(src_full) > _MAX_NEW_SYNAPSES:
            strengths = corr[i_idx, j_idx]
            top_k     = np.argsort(strengths)[-_MAX_NEW_SYNAPSES:]
            src_full  = src_full[top_k]
            tgt_full  = tgt_full[top_k]

        # ------------------------------------------------------------------
        # 5. Inject into StructuralPlasticity
        # ------------------------------------------------------------------
        device      = plasticity.device
        new_indices = torch.tensor(
            [src_full.tolist(), tgt_full.tolist()],
            dtype=torch.long, device=device)
        new_weights = torch.full(
            (len(src_full),), _INIT_WEIGHT,
            dtype=torch.float32, device=device)
        new_integrity = torch.full(
            (len(src_full),), _INIT_INTEGRITY,
            dtype=torch.float32, device=device)

        plasticity.indices          = torch.cat([plasticity.indices,          new_indices],   dim=1)
        plasticity.weights_values   = torch.cat([plasticity.weights_values,   new_weights])
        plasticity.integrity_values = torch.cat([plasticity.integrity_values, new_integrity])
        plasticity._topology_changed = True
