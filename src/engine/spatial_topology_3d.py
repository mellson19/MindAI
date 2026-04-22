import numpy as np

class BrainGeometry:

    def __init__(self, num_nodes: int, radius: float=10.0):
        self.num_nodes = num_nodes
        self.radius = radius
        self.coordinates = self._generate_spherical_coordinates()
        self.distance_matrix = self._calculate_distance_matrix()

    def _generate_spherical_coordinates(self) -> np.ndarray:
        coords = np.zeros((self.num_nodes, 3))
        for i in range(self.num_nodes):
            phi = np.arccos(1 - 2 * (i + 0.5) / self.num_nodes)
            theta = np.pi * (1 + 5 ** 0.5) * i
            x = self.radius * np.cos(theta) * np.sin(phi)
            y = self.radius * np.sin(theta) * np.sin(phi)
            z = self.radius * np.cos(phi)
            coords[i] = [x, y, z]
        return coords

    def _calculate_distance_matrix(self) -> np.ndarray:
        diff = self.coordinates[:, np.newaxis, :] - self.coordinates[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=-1))
        return distances

    def get_axonal_delay(self, node_a: int, node_b: int) -> int:
        distance = self.distance_matrix[node_a, node_b]
        speed_of_conduction = 2.0
        return max(1, int(distance / speed_of_conduction))