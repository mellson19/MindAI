import numpy as np
import copy

class ImaginationSimulator:

    def __init__(self):
        self.is_imagining = False

    def simulate_future_scenario(self, current_brain_state, hypothetical_action: int):
        print(f"[Внутренний диалог]: 'А что если я выберу действие {hypothetical_action}?'")
        simulated_brain = copy.deepcopy(current_brain_state)
        simulated_stress = 0.0
        for _ in range(10):
            simulated_stress += simulated_brain.predict_stress_level()
        print(f"[Внутренний диалог]: 'Ожидаемый стресс: {simulated_stress}.'")
        return simulated_stress