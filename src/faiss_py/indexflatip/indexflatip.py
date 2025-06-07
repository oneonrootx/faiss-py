import numpy as np

from faiss_py.core.index import Index


class IndexFlatIP(Index):
    """
    Flat (brute-force) Inner Product (dot product) index.

    Stores all vectors in a flat numpy array and performs brute-force search
    for nearest neighbors using inner product similarity.
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
            Always raised, as flat indices are not trainable.
        """
        raise NotImplementedError("`IndexFlatIP` is not trainable, it's flat")
    
    def add(self, vectors):
        """
        Add vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray
            Vectors to add, shape (n, d).
        """
        self.database = np.concatenate((self.database, vectors), axis=0)

    def search(self, query, k: int):
        """
        Search the index for the top-k vectors with highest inner product.

        Parameters
        ----------
        query : np.ndarray
            Query vector(s), shape (d,) or (nq, d).
        k : int
            Number of nearest neighbors to return.

        Returns
        -------
        D : np.ndarray
            Inner products of the top-k results, shape (nq, k).
        I : np.ndarray
            Indices of the top-k results, shape (nq, k).
        """
        k = min(k, len(self.database))
        query = np.asarray(query)

        if query.ndim == 1:
            query = query[None, :]

        Dall = np.dot(query, self.database.T)
        
        Iunsorted = np.argpartition(-Dall, kth=k - 1)[:, :k]
        Dunsorted = Dall[np.arange(len(query))[:, None], Iunsorted]

        order = np.argsort(-Dunsorted, axis=1)
        I = np.take_along_axis(Iunsorted, order, axis=1)
        D = np.take_along_axis(Dunsorted, order, axis=1)

        return D, I
