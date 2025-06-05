import numpy as np

from faiss_py.core.index import Index

class IndexFlatL2(Index):
    """
    Flat (brute-force) L2 (Euclidean) distance index.

    Stores all vectors in a flat numpy array and performs brute-force search
    for nearest neighbors using L2 distance.
    """

    def __init__(self, d: int):
        """
        Initialize the index.

        Parameters
        ----------
        d : int
            Dimensionality of the vectors to be stored.
        """
        super().__init__(d)
        self.database = np.empty((0, d), dtype=np.float32)

    def train(self, vectors):
        """
        Not implemented for flat indices.

        Parameters
        ----------
        vectors : np.ndarray
            Input vectors (ignored).

        Raises
        ------
        NotImplementedError
            Always, since flat indices do not require training.
        """
        raise NotImplementedError("`IndexFlatL2` is not trainable, it's flat")

    def add(self, vectors):
        """
        Add vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray
            Array of shape (n, d) containing the vectors to add.
        """
        self.database = np.concatenate((self.database, vectors), axis=0)

    def search(self, query, k: int):
        """
        Search for the k nearest neighbors of the query vectors.

        Parameters
        ----------
        query : np.ndarray
            Query vectors of shape (m, d) or (d,).
        k : int
            Number of nearest neighbors to return.

        Returns
        -------
        D : np.ndarray
            Array of shape (m, k) with the L2 distances to the nearest neighbors.
        I : np.ndarray
            Array of shape (m, k) with the indices of the nearest neighbors.
        """
        k = min(k, len(self.database))
        query = np.asarray(query)

        if query.ndim == 1:
            query = query[None, :]

        diff = self.database[None, :, :] - query[:, None, :] # (1, N, d) - (M, 1, d) = (M, N, d)
        Dall = np.linalg.norm(diff, axis=2)
        
        Iunsorted = np.argpartition(Dall, kth=k - 1)[:, :k]
        Dunsorted = Dall[np.arange(len(query))[:, None], Iunsorted]

        order = np.argsort(Dunsorted, axis=1)
        I = np.take_along_axis(Iunsorted, order, axis=1)
        D = np.take_along_axis(Dunsorted, order, axis=1)

        return D, I
