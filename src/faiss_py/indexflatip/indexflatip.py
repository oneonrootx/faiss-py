import numpy as np
from tqdm import tqdm
from faiss_py import logger
from faiss_py.core.index import Index


class IndexFlatIP(Index):
    """
    Flat (brute-force) Inner Product (dot product) index.

    Stores all vectors in a flat numpy array and performs brute-force search
    for nearest neighbors using inner product similarity.
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
            logger.info("Initialized IndexFlatIP with d=%d", d)

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
        if self.verbose:
            logger.info("Adding %d vectors to IndexFlatIP", len(vectors))
        self.database = np.concatenate((self.database, vectors), axis=0)
        if self.verbose:
            logger.info("Database now contains %d vectors", len(self.database))

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

        if self.verbose:
            logger.info("Searching %d queries for k=%d neighbors in database of %d vectors", 
                       len(query), k, len(self.database))

        Dall = np.dot(query, self.database.T)
        
        Iunsorted = np.argpartition(-Dall, kth=k - 1)[:, :k]
        Dunsorted = Dall[np.arange(len(query))[:, None], Iunsorted]

        order = np.argsort(-Dunsorted, axis=1)
        I = np.take_along_axis(Iunsorted, order, axis=1)
        D = np.take_along_axis(Dunsorted, order, axis=1)

        if self.verbose:
            logger.info("Search completed, found %d results per query", I.shape[1])

        return D, I
