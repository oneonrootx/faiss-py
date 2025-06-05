import numpy as np

class IndexFlatL2:

    def __init__(self, d: int):
        self.d = d
        self.vectors = np.empty((0, d), dtype=np.float32)

    def train(self, vectors):
        raise NotImplementedError("`IndexFlatL2` is not trainable")

    def add(self, vectors):
        self.vectors = np.concatenate((self.vectors, vectors), axis=0)

    def search(self, query, k: int):
        k = min(k, len(self.vectors))

        if query.ndim == 1:
            query = query[None, :]

        diff = self.vectors[None, :, :] - query[:, None, :] # (1, N, d) - (M, 1, d) = (M, N, d)
        D = np.linalg.norm(diff, axis=2)
        I = np.argpartition(D, kth=k - 1)[:, :k]
        return D[np.arange(len(query))[:, None], I], I
