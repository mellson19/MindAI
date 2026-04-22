import torch

class PredictiveMicrocircuits:

    def __init__(self, num_nodes: int, initial_density: float=0.005):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes = num_nodes
        num_connections = int(num_nodes * num_nodes * initial_density)
        self.td_indices = torch.randint(0, num_nodes, (2, num_connections), device=self.device)
        self.td_values = torch.rand(num_connections, device=self.device) * 0.1
        self.bu_indices = torch.randint(0, num_nodes, (2, num_connections), device=self.device)
        self.bu_values = torch.rand(num_connections, device=self.device) * 0.1
        self.prediction_neurons = torch.zeros(num_nodes, device=self.device)
        self.error_neurons = torch.zeros(num_nodes, device=self.device)

    def process_inference_step(self, sensory_input: torch.Tensor, internal_state: torch.Tensor, plasticity_rate: float):
        W_top = torch.sparse_coo_tensor(self.td_indices, self.td_values, (self.num_nodes, self.num_nodes)).coalesce()
        self.prediction_neurons = torch.clamp(torch.sparse.mm(W_top, internal_state.unsqueeze(1)).squeeze(1), 0.0, 1.0)
        self.error_neurons = torch.relu(sensory_input - self.prediction_neurons)
        W_bot = torch.sparse_coo_tensor(self.bu_indices, self.bu_values, (self.num_nodes, self.num_nodes)).coalesce()
        bottom_up_drive = torch.sparse.mm(W_bot, self.error_neurons.unsqueeze(1)).squeeze(1)
        updated_internal_state = torch.clamp(internal_state + bottom_up_drive, 0.0, 1.0)
        active_error_mask = self.error_neurons > 0.1
        active_state_mask = internal_state > 0.1
        td_active = active_error_mask[self.td_indices[0]] & active_state_mask[self.td_indices[1]]
        if td_active.any():
            self.td_values[td_active] += 0.01 * plasticity_rate
        bu_active = active_state_mask[self.bu_indices[0]] & active_error_mask[self.bu_indices[1]]
        if bu_active.any():
            self.bu_values[bu_active] += 0.01 * plasticity_rate
        total_surprise_physical = torch.sum(self.error_neurons)
        return (total_surprise_physical, updated_internal_state)