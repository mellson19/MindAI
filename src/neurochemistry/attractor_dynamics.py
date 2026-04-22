import numpy as np

class MoodAttractors:

    def __init__(self):
        self.attractors = {'calm': {'energy_target': 0.9, 'stress_target': 0.1, 'depth': 0.5}, 'anxiety': {'energy_target': 0.5, 'stress_target': 0.6, 'depth': 0.8}, 'depression': {'energy_target': 0.2, 'stress_target': 0.9, 'depth': 1.5}}

    def apply_attractor_pull(self, current_energy: float, current_stress: float, dopamine: float, base_energy: float=5000.0):
        e_ratio = current_energy / base_energy
        s_ratio = min(1.0, current_stress / base_energy)
        current_state = np.array([e_ratio, s_ratio])
        pull_forces = {}
        for name, attr in self.attractors.items():
            target = np.array([attr['energy_target'], attr['stress_target']])
            distance = np.linalg.norm(current_state - target)
            force = attr['depth'] / (distance + 0.1) * (1.0 - dopamine * 0.8)
            pull_forces[name] = force
        dominant_mood = max(pull_forces, key=pull_forces.get)
        return (dominant_mood, 0, 0)