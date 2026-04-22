import numpy as np

class EgoModel:

    def __init__(self, expected_baseline: float=5000.0):
        self.predicted_internal_energy = expected_baseline
        self.predicted_stress = 0.0
        self.sense_of_agency = 1.0

    def evaluate_self(self, actual_energy: float, actual_stress: float) -> float:
        energy_error = abs(self.predicted_internal_energy - actual_energy)
        stress_error = abs(self.predicted_stress - actual_stress)
        total_self_error = energy_error + stress_error
        self.predicted_internal_energy = self.predicted_internal_energy * 0.9 + actual_energy * 0.1
        self.predicted_stress = self.predicted_stress * 0.9 + actual_stress * 0.1
        normalized_error = total_self_error / (self.predicted_internal_energy + 1.0)
        if normalized_error > 0.5:
            self.sense_of_agency = max(0.0, self.sense_of_agency - 0.1)
        else:
            self.sense_of_agency = min(1.0, self.sense_of_agency + 0.05)
        return total_self_error