import numpy as np

class PrefrontalCortex:

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.current_goal_vector = np.zeros(num_nodes)
        self.goal_persistence = 0.0

    def formulate_goal(self, energy: float, water: float, base_resource: float):
        self.current_goal_vector.fill(0.0)
        if water < base_resource * 0.4 and water < energy:
            self.current_goal_vector[300:305] = 1.0
            self.goal_persistence = 1.0
        elif energy < base_resource * 0.4:
            self.current_goal_vector[305:310] = 1.0
            self.goal_persistence = 1.0
        else:
            self.goal_persistence *= 0.9
        return self.current_goal_vector * self.goal_persistence