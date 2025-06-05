from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

class Index(ABC):

    def __init__(self, d: int):
        """
        Initialize the index with the dimension of the vectors.

        Args:
            d (int): The dimension of the vectors to be indexed.
        """
        self.d = d

    @abstractmethod
    def train(self, vectors: npt.NDArray[np.float32]):
        """Train the index with the provided vectors.

        Args:
            vectors (np.array): A numpy array of shape (n, d), where n is the number of vectors and d is the dimension of each vector.
        """
        ...
    
    @abstractmethod
    def add(self, vectors: npt.NDArray[np.float32]):
        """Add one or more vectors to the index.

        Args:
            vectors (np.array): A numpy array of shape (n, d), where n is the number of vectors and d is the dimension of each vector.
        """
        ...

    @abstractmethod
    def search(self, query: npt.NDArray[np.float32], k: int):
        """
        Search the index for the k nearest neighbors of the query vector(s).

        Args:
            query (np.array): A numpy array of shape (m, d), where m is the number of query vectors and d is the dimension.
            k (int): The number of nearest neighbors to return.

        Returns:
            distances (np.array): A numpy array of shape (m, k) with the distances to the nearest neighbors.
            indices (np.array): A numpy array of shape (m, k) with the indices of the nearest neighbors.
        """
        ...
