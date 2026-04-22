import numpy as np

class Hippocampus:

    def __init__(self, max_capacity: int=1000):
        self.max_capacity = max_capacity
        self.episodic_memory = []
        self.current_time_index = 0

    def encode_episode(self, workspace_pattern: np.ndarray, emotional_valence: float):
        self.current_time_index += 1
        if abs(emotional_valence) < 0.2:
            return
        episode = {'timestamp': self.current_time_index, 'pattern': workspace_pattern.copy(), 'valence': emotional_valence}
        self.episodic_memory.append(episode)
        if len(self.episodic_memory) > self.max_capacity:
            self.episodic_memory.pop(0)

    def retrieve_for_consolidation(self, current_cortisol: float, current_mood_vector: np.ndarray) -> list:
        retrieved_patterns = []
        for ep in self.episodic_memory:
            pattern = ep['pattern'].copy()
            age = self.current_time_index - ep['timestamp']
            decay = np.exp(-age / 1000.0)
            pattern *= decay
            if current_cortisol > 0.3:
                flip_probability = (current_cortisol - 0.3) * 0.2
                mutation_mask = np.random.rand(*pattern.shape) < flip_probability
                pattern = np.where(mutation_mask, 1.0 - pattern, pattern)
            pattern = pattern * 0.8 + current_mood_vector * 0.2
            retrieved_patterns.append(pattern)
        return retrieved_patterns