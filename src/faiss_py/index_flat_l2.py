import numpy as np
import numpy.typing as npt
from faiss_py.core.exceptions import IndexNotTrainable
from faiss_py.core.index import Index


class IndexFlatL2(Index):

    def __init__(self, d: int):
        self._vectors: npt.NDArray[np.float32] = np.empty((0, d), dtype=np.float32)
        super().__init__(d)

    def _l2_norm(self, vector: npt.NDArray[np.float32], batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.linalg.norm(batch - vector, axis=1)

    def add(self, vectors: npt.NDArray[np.float32]):
        self._vectors = np.concatenate((self._vectors, vectors), axis=0)

    def train(self, vectors: npt.NDArray[np.float32]):
        raise IndexNotTrainable()

    def search(self, query: npt.NDArray[np.float32], k: int):
        k = min(k, len(self._vectors))
        if k == 0: return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        D = self._l2_norm(query, self._vectors)
        topk_idx = np.argpartition(D, k - 1)[:k]
        sorted_topk_idx = topk_idx[np.argsort(D[topk_idx])]
        return D[sorted_topk_idx], sorted_topk_idx
