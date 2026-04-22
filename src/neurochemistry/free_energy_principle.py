import numpy as np

class ActiveInferenceEngine:

    def __init__(self):
        self.precision = 1.0
        self.boredom_accumulator = 0.0

    def calculate_vfe(self, predicted_sensory: np.ndarray, actual_sensory: np.ndarray, prior_beliefs: np.ndarray) -> float:
        actual_safe = np.clip(actual_sensory, 0.0, 1.0)
        pred_safe = np.clip(predicted_sensory, 0.0, 1.0)
        prediction_error = np.sum((actual_safe - pred_safe) ** 2)
        kl_divergence = np.sum(np.abs(actual_safe - prior_beliefs))
        num_nodes = len(actual_sensory)
        free_energy = (self.precision * prediction_error + kl_divergence) / (num_nodes / 1000.0)
        if np.isnan(free_energy) or np.isinf(free_energy):
            return 1000.0
        return min(free_energy, 1000.0)

    def generate_action_gradient(self, free_energy: float, motor_options: np.ndarray, hunger_distress: float=0.0, pain_distress: float=0.0, adrenaline: float=0.0) -> np.ndarray:
        base_t = free_energy / 15.0
        panic = pain_distress * 3.0 * (1.0 - adrenaline)
        focus = hunger_distress * 2.0 + adrenaline * 3.0
        if free_energy < 0.5 and pain_distress < 0.1 and (hunger_distress < 0.3):
            self.boredom_accumulator = min(10.0, self.boredom_accumulator + 0.1)
        else:
            self.boredom_accumulator *= 0.5
        T = max(0.05, min(5.0, base_t + panic - focus + self.boredom_accumulator))
        exp_vals = np.exp(motor_options / T)
        sum_probs = np.sum(exp_vals)
        if sum_probs == 0 or np.isnan(sum_probs):
            return np.ones_like(motor_options) / len(motor_options)
        return exp_vals / sum_probs