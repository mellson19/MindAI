import numpy as np

class FreeWillEngine:

    def __init__(self, delay_ticks: int=15):
        self.delay_ticks = delay_ticks
        self.decision_queue = []
        self.sense_of_authorship = 1.0
        self.somatic_markers_pain = np.zeros(5)

    def update_somatic_markers(self, action_taken: int, resulting_stress: float):
        if action_taken is not None and action_taken < 5:
            self.somatic_markers_pain[action_taken] = self.somatic_markers_pain[action_taken] * 0.9 + resulting_stress * 0.1

    def bias_motor_probabilities(self, base_probabilities: np.ndarray, amygdala_arousal: float) -> np.ndarray:
        intuition_weight = min(1.0, amygdala_arousal)
        suppression_factors = np.exp(-self.somatic_markers_pain / (100.0 + 1e-05))
        biased_probs = base_probabilities * (1.0 - intuition_weight + suppression_factors * intuition_weight)
        sum_p = np.sum(biased_probs)
        if sum_p == 0:
            return np.ones_like(base_probabilities) / len(base_probabilities)
        return biased_probs / sum_p

    def unconscious_decision_making(self, motor_potentials: np.ndarray, amygdala_arousal: float=0.5) -> int:
        exp_vals = np.exp(motor_potentials - np.max(motor_potentials))
        base_probs = exp_vals / np.sum(exp_vals)
        final_probs = self.bias_motor_probabilities(base_probs, amygdala_arousal)
        chosen_action_id = np.random.choice(len(final_probs), p=final_probs)
        self.decision_queue.append({'action': chosen_action_id, 'ticks_remaining': self.delay_ticks})
        return chosen_action_id

    def conscious_veto_and_awareness(self) -> int:
        action_to_realize = None
        for decision in self.decision_queue:
            decision['ticks_remaining'] -= 1
            if decision['ticks_remaining'] == 0:
                action_to_realize = decision['action']
                self.sense_of_authorship = min(1.0, self.sense_of_authorship + 0.01)
        self.decision_queue = [d for d in self.decision_queue if d['ticks_remaining'] > 0]
        return action_to_realize

class BasalGanglia:

    def __init__(self, motor_cortex_size: int, num_actions: int=5):
        self.num_actions = num_actions
        self.action_weights = np.random.uniform(0.1, 0.5, (motor_cortex_size, num_actions))
        self.last_motor_pattern = None
        self.last_chosen_action = None

    def map_to_action_potentials(self, motor_cortex_activity: np.ndarray) -> np.ndarray:
        self.last_motor_pattern = motor_cortex_activity.copy()
        action_potentials = np.dot(motor_cortex_activity, self.action_weights)
        return action_potentials

    def reinforce_learning(self, chosen_action: int, dopamine: float, pain: float):
        if self.last_motor_pattern is None or chosen_action is None or chosen_action >= self.num_actions:
            return
        self.last_chosen_action = chosen_action
        learning_signal = dopamine * 0.1 - pain * 0.05
        self.action_weights[:, chosen_action] += self.last_motor_pattern * learning_signal
        self.action_weights = np.clip(self.action_weights, 0.0, 1.0)