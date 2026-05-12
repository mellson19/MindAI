import numpy as np

class PrefrontalCortex:
    """Prefrontal working-memory bias toward current homeostatic goal.

    Biologically: vlPFC/OFC encodes goal-state representations that bias
    downstream motor cortex and striatum toward deficit-reducing actions
    (Wallis 2007; Rushworth 2011).

    Goal representation is a distributed pattern across motor-adjacent
    neurons, not hardcoded indices — the layout slice is computed relative
    to num_nodes so the module scales across any network size.

    Two goal channels (hunger, thirst) are placed at the top 10 neurons
    of the motor region (last neurons in the array, furthest from sensory
    input, closest to output layer by convention in this layout).
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.current_goal_vector = np.zeros(num_nodes)
        self.goal_persistence = 0.0
        # Goal neuron indices: last 10 neurons, split hunger/thirst
        # Relative indices — safe for any num_nodes >= 10
        self._hunger_slice = slice(max(0, num_nodes - 10), max(0, num_nodes - 5))
        self._thirst_slice = slice(max(0, num_nodes - 5),  num_nodes)

    def formulate_goal(self, energy: float, water: float, base_resource: float):
        self.current_goal_vector.fill(0.0)
        if water < base_resource * 0.4 and water < energy:
            self.current_goal_vector[self._thirst_slice] = 1.0
            self.goal_persistence = 1.0
        elif energy < base_resource * 0.4:
            self.current_goal_vector[self._hunger_slice] = 1.0
            self.goal_persistence = 1.0
        else:
            self.goal_persistence *= 0.9
        return self.current_goal_vector * self.goal_persistence