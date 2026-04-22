import numpy as np
import torch

class SemanticMemory:

    def extract_concept_during_sleep(self, memory_patterns: list, plasticity):
        if not memory_patterns:
            return
        avg_pattern = np.mean(memory_patterns, axis=0)
        active_core = np.where(avg_pattern > 0.5)[0]
        if len(active_core) > 2:
            src_list = []
            tgt_list = []
            for i in active_core:
                for j in active_core:
                    if i != j:
                        src_list.append(int(i))
                        tgt_list.append(int(j))
            if not src_list:
                return
            device = plasticity.device
            new_indices = torch.tensor([src_list, tgt_list], dtype=torch.long, device=device)
            new_weights = torch.full((len(src_list),), 0.01, dtype=torch.float32, device=device)
            new_integrity = torch.full((len(src_list),), 2.0, dtype=torch.float32, device=device)
            plasticity.indices = torch.cat([plasticity.indices, new_indices], dim=1)
            plasticity.weights_values = torch.cat([plasticity.weights_values, new_weights])
            plasticity.integrity_values = torch.cat([plasticity.integrity_values, new_integrity])
            plasticity._topology_changed = True