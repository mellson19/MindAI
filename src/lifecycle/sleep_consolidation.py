import numpy as np

class SleepCycle:

    def __init__(self):
        self.is_sleeping = False
        self.sleep_queue = []

    def process_sleep_tick(self, hippocampus, plasticity, current_cortisol: float, current_mood_vector: np.ndarray) -> bool:
        if not self.sleep_queue:
            self.sleep_queue = hippocampus.retrieve_for_consolidation(current_cortisol, current_mood_vector)
            hippocampus.episodic_memory.clear()
        if not self.sleep_queue:
            self.is_sleeping = False
            return False
        memory_pattern = self.sleep_queue.pop(0)
        if hasattr(plasticity, 'apply_stdp_learning'):
            import torch
            device = plasticity.device
            memory_tensor = torch.tensor(memory_pattern, dtype=torch.float32, device=device)
            plasticity.apply_stdp_learning(current_activity=memory_tensor, neuromodulator_multiplier=5.0)
        return True