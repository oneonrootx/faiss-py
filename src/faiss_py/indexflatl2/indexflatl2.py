import numpy as np
from tqdm import tqdm
from faiss_py import logger
from faiss_py.core.index import Index

class IndexFlatL2(Index):
    """
    Flat (brute-force) L2 (Euclidean) distance index.

    Stores all vectors in a flat numpy array and performs brute-force search
    for nearest neighbors using L2 distance.
    """

    def __init__(self, d: int, verbose: bool = False):
        """
        Initialize the index.

        Parameters
        ----------
        d : int
            Dimensionality of the vectors to be stored.
        verbose : bool, optional
            Enable verbose logging. Defaults to False.
        """
        super().__init__(d)
        self.verbose = verbose
        self.database = np.empty((0, d), dtype=np.float32)
        if self.verbose:
            logger.info("Initialized IndexFlatL2 with d=%d", d)

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
        if self.verbose:
            logger.info("Adding %d vectors to IndexFlatL2", len(vectors))
        vectors = np.asarray(vectors)
        self.database = np.concatenate((self.database, vectors), axis=0)
        if self.verbose:
            logger.info("Database now contains %d vectors", len(self.database))

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

        if self.verbose:
            logger.info("Searching %d queries for k=%d neighbors in database of %d vectors", 
                       len(query), k, len(self.database))

        diff = self.database[None, :, :] - query[:, None, :] # (1, N, d) - (M, 1, d) = (M, N, d)
        Dall = np.linalg.norm(diff, axis=2)
        
        Iunsorted = np.argpartition(Dall, kth=k - 1)[:, :k]
        Dunsorted = Dall[np.arange(len(query))[:, None], Iunsorted]

        order = np.argsort(Dunsorted, axis=1)
        I = np.take_along_axis(Iunsorted, order, axis=1)
        D = np.take_along_axis(Dunsorted, order, axis=1)

        if self.verbose:
            logger.info("Search completed, found %d results per query", I.shape[1])

        return D, I
