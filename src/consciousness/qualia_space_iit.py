import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

class QualiaSpace:

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.last_qualia_signature = [0.33, 0.33, 0.33]
        self.tick_counter = 0

    def calculate_qualia_shape(self, active_nodes: np.ndarray, weight_matrix: sp.spmatrix) -> list:
        self.tick_counter += 1
        if self.tick_counter % 10 != 0:
            return self.last_qualia_signature
        active_indices = np.where(active_nodes > 0.5)[0]
        if len(active_indices) > 100:
            active_indices = np.argsort(active_nodes)[-100:]
        if len(active_indices) < 3:
            return [0.33, 0.33, 0.33]
        subgraph = weight_matrix[active_indices, :][:, active_indices].tocsr()
        if subgraph.nnz == 0:
            return [0.33, 0.33, 0.33]
        try:
            eigenvalues, _ = sla.eigs(subgraph, k=3, which='LM', return_eigenvectors=True, tol=0.01)
            qualia_signature = np.abs(eigenvalues)
            sum_q = np.sum(qualia_signature) + 1e-09
            self.last_qualia_signature = (qualia_signature / sum_q).tolist()
        except:
            pass
        return self.last_qualia_signature