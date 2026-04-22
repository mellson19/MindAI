import torch
import numpy as np
import random

class StructuralPlasticity:

    def __init__(self, num_nodes: int, initial_density: float=0.01, inhibitory_ratio: float=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes = num_nodes
        self.active_limit = int(num_nodes * 0.8)
        self.epistemic_hunger = 0.0
        self.hunger_threshold = 100.0
        self.growth_cooldown = 0
        print(f'    [БИОЛОГИЯ] Выращивание графа на {self.device}. 80% Глутамат / 20% ГАМК...')
        self.is_inhibitory_tensor = torch.rand(num_nodes, device=self.device) < inhibitory_ratio
        self.is_inhibitory = self.is_inhibitory_tensor.cpu().numpy()
        num_connections = int(self.active_limit * self.active_limit * initial_density)
        indices = torch.randint(0, self.active_limit, (2, num_connections), device=self.device)
        mask = indices[0] != indices[1]
        self.indices = indices[:, mask]
        signs = torch.where(self.is_inhibitory_tensor[self.indices[0]], -1.0, 1.0)
        self.weights_values = (torch.rand(self.indices.shape[1], device=self.device) * 0.09 + 0.01) * signs
        self.integrity_values = torch.ones(self.indices.shape[1], device=self.device)
        self.pre_trace = torch.zeros(num_nodes, device=self.device)
        self.post_trace = torch.zeros(num_nodes, device=self.device)
        self._cached_sparse_weights = None
        self._topology_changed = True

    def _coalesce_state(self):
        combined_vals = torch.stack([self.weights_values, self.integrity_values], dim=1)
        temp_sparse = torch.sparse_coo_tensor(self.indices, combined_vals, (self.num_nodes, self.num_nodes, 2)).coalesce()
        self.indices = temp_sparse.indices()
        coalesced_vals = temp_sparse.values()
        self.weights_values = coalesced_vals[:, 0]
        self.integrity_values = torch.clamp(coalesced_vals[:, 1], 0.0, 2.0)
        self._cached_sparse_weights = torch.sparse_coo_tensor(self.indices, self.weights_values, (self.num_nodes, self.num_nodes))
        self._topology_changed = False

    def get_sparse_weights(self) -> torch.Tensor:
        if getattr(self, '_topology_changed', True) or self._cached_sparse_weights is None:
            self._coalesce_state()
        else:
            self._cached_sparse_weights._values().copy_(self.weights_values)
        return self._cached_sparse_weights

    def trigger_neurogenesis(self, surprise_level: float):
        if self.growth_cooldown > 0:
            self.growth_cooldown -= 1
            return
        self.epistemic_hunger += surprise_level
        if self.epistemic_hunger > self.hunger_threshold and self.active_limit < self.num_nodes - 5:
            self.active_limit += 5
            self.epistemic_hunger = 0.0
            self.growth_cooldown = 300

    def apply_stdp_learning(self, current_activity: torch.Tensor, neuromodulator_multiplier: float):
        self.pre_trace *= 0.9
        self.post_trace *= 0.9
        active_now_mask = current_activity > 0.5
        self.pre_trace[active_now_mask] = 1.0
        self.post_trace[active_now_mask] = 1.0
        if not active_now_mask.any():
            return
        pre_idx = self.indices[0]
        post_idx = self.indices[1]
        ltp_mask = (self.pre_trace[pre_idx] > 0.1) & active_now_mask[post_idx]
        ltd_mask = active_now_mask[pre_idx] & (self.post_trace[post_idx] > 0.1)
        signs = torch.where(self.is_inhibitory_tensor[pre_idx], -1.0, 1.0)
        if ltp_mask.any():
            delta_ltp = 0.05 * self.pre_trace[pre_idx][ltp_mask] * neuromodulator_multiplier
            self.weights_values[ltp_mask] = torch.clamp(self.weights_values[ltp_mask] + delta_ltp * signs[ltp_mask], -1.0, 1.0)
            self.integrity_values[ltp_mask] = torch.clamp(self.integrity_values[ltp_mask] + 0.1, 0.0, 2.0)
        if ltd_mask.any():
            delta_ltd = 0.02 * self.post_trace[post_idx][ltd_mask] * neuromodulator_multiplier
            self.weights_values[ltd_mask] = torch.clamp(self.weights_values[ltd_mask] - delta_ltd, -1.0, 1.0)

    def synaptogenesis_and_pruning(self, active_nodes: torch.Tensor, energy_level: float):
        if random.random() < 0.1:
            alive_mask = self.integrity_values > 0.0
            if not alive_mask.all():
                self.indices = self.indices[:, alive_mask]
                self.weights_values = self.weights_values[alive_mask]
                self.integrity_values = self.integrity_values[alive_mask]
                self._topology_changed = True
        if energy_level > 1000.0 and random.random() < 0.05:
            active_idx = torch.where(active_nodes[:self.active_limit] > 0.5)[0]
            if len(active_idx) > 1:
                src = active_idx[torch.randint(0, len(active_idx), (5,), device=self.device)]
                tgt = active_idx[torch.randint(0, len(active_idx), (5,), device=self.device)]
                new_indices = torch.stack([src, tgt])
                signs = torch.where(self.is_inhibitory_tensor[src], -1.0, 1.0)
                new_weights = torch.rand(5, device=self.device) * 0.1 * signs
                new_integrity = torch.ones(5, device=self.device) * 0.5
                self.indices = torch.cat([self.indices, new_indices], dim=1)
                self.weights_values = torch.cat([self.weights_values, new_weights])
                self.integrity_values = torch.cat([self.integrity_values, new_integrity])
                self._topology_changed = True

    def maintain_homeostasis(self):
        W = self.get_sparse_weights()
        incoming_sum = torch.sparse.sum(torch.abs(W), dim=0).to_dense()
        overloaded = incoming_sum > 5.0
        if overloaded.any():
            scale_factors = torch.ones(self.num_nodes, device=self.device)
            scale_factors[overloaded] = 5.0 / incoming_sum[overloaded]
            post_idx = self.indices[1]
            self.weights_values *= scale_factors[post_idx]

    def apply_cortisol_damage(self, cortisol_level: float):
        if cortisol_level > 0.5 and random.random() < 0.1:
            damage_chance = (cortisol_level - 0.5) * 0.05
            if random.random() < damage_chance:
                victim = random.randint(0, self.active_limit - 1)
                victim_mask = self.indices[0] == victim
                self.weights_values[victim_mask] *= 0.8
                self.integrity_values[victim_mask] = torch.clamp(self.integrity_values[victim_mask] - 0.2, min=0.0)
                print(f'    [ПСИХИАТРИЯ] Нейротоксичность! Кортизол разрушил связи узла {victim}.')