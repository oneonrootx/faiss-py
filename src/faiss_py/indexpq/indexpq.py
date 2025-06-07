import itertools
from faiss_py.core.index import Index
from faiss_py.kmeans.kmeans import Kmeans

import numpy as np
from tqdm import tqdm



class IndexPQ(Index):
    """
    Product Quantization (PQ) index for approximate nearest neighbor search.

    Attributes:
        M (int): Number of subquantizers (subspaces).
        S (int): Number of subspaces (d // M).
        nbits (int): Number of centroids per subspace.
        verbose (bool): If True, print progress during training.
        codebooks (dict): Dictionary mapping subspace index to trained Kmeans index.
        codebook_centoids (dict): Dictionary mapping subspace index to centroids.
        database (np.ndarray): Encoded database vectors (codes).
    """

    def __init__(self, d: int, M: int, nbits: int, verbose: bool = False):
        """
        Initialize the IndexPQ.

        Args:
            d (int): Dimensionality of the input vectors.
            M (int): Number of subquantizers (subspaces).
            nbits (int): Number of centroids per subspace.
            verbose (bool, optional): If True, print progress during training. Defaults to False.

        Raises:
            ValueError: If d is not divisible by M.
        """
        super().__init__(d)
        
        if d % M != 0: 
            raise ValueError(f"M must divide d ({d} % {M} == {d % M})")
        
        self.M = M
        self.S = self.d // self.M
        self.nbits = nbits
        self.verbose = verbose
        self.codebooks = None
        self.codebook_centoids = None
        self.database = np.empty((0, self.S), dtype=np.int16)


    def train(self, vectors):
        """
        Train the PQ codebooks using KMeans for each subspace.

        Args:
            vectors (np.ndarray): Training vectors of shape (N, d).
        """
        subvectors = vectors.reshape(vectors.shape[0], -1, self.M)
        subspaces = subvectors.transpose(1, 0, 2)
        self.codebooks = {}
        self.codebook_centoids = {}
        for subspace_idx, subspace in tqdm(enumerate(subspaces), total=len(subspaces), desc="Building Codebooks", disable=not self.verbose):
            kmeans = Kmeans(d=self.M, k=self.nbits)
            kmeans.train(subspace)
            self.codebooks[subspace_idx] = kmeans.index
            self.codebook_centoids[subspace_idx] = kmeans.centroids

    def encode(self, vector):
        """
        Encode input vectors into PQ codes.

        Args:
            vector (np.ndarray): Input vector(s) of shape (d,) or (N, d).

        Returns:
            np.ndarray: Encoded PQ codes of shape (N, S).
        """
        vector = np.asarray(vector)

        if vector.ndim == 1:
            vector = vector[None, :]

        SV = vector.reshape(vector.shape[0], -1, self.M)
        S = SV.transpose(1, 0, 2)

        codesT = []
        for subspace_idx, subspace in enumerate(S):
            index: Index = self.codebooks[subspace_idx]
            _, I = index.search(subspace, k=1)
            codesT.append(I)
        
        codes = np.array(codesT).T[0]
        return codes
    
    def decode(self, codes):
        """
        Decode PQ codes back to approximate vectors.

        Args:
            codes (np.ndarray): PQ codes of shape (N, S) or (S,).

        Returns:
            np.ndarray: Reconstructed vectors of shape (N, d).
        """
        codes = np.asarray(codes)

        if codes.ndim == 1:
            codes = codes[None, :]

        vectors = []
        for code in codes:
            vector = np.array(list(itertools.chain(*[self.codebook_centoids[subspace_id][code_id] for subspace_id, code_id in enumerate(code)])))
            vectors.append(vector)

        return np.array(vectors)

    def add(self, vectors):
        """
        Add vectors to the PQ database (by encoding them).

        Args:
            vectors (np.ndarray): Vectors to add, shape (N, d).
        """
        self.database = np.concatenate((self.database, self.encode(vectors)), axis=0)
    
    def search(self, query, k: int):
        """
        Search for the k nearest neighbors of the query vectors.

        Args:
            query (np.ndarray): Query vector(s) of shape (d,) or (Nq, d).
            k (int): Number of nearest neighbors to return.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - D_sorted: Distances to the k nearest neighbors, shape (Nq, k).
                - I_sorted: Indices of the k nearest neighbors, shape (Nq, k).
        """
        k = min(k, len(self.database))
        query = np.asarray(query)

        if query.ndim == 1:
            query = query[None, :] # (Nq, d)

        query_split = query.reshape(query.shape[0], -1, self.M) # (Nq, S, M)

        subspace_centroids = np.array([
            self.codebooks[subspace_index].database
            for subspace_index in range(self.S)
        ]) # (S, nbits, M)

        # Broadcasting
        # (Nq,    S,      1,      M)
        # (1,     S,      nbits,  M) = 
        # (Nq,    S,      nbits,  M)
        Hdiff = query_split[:, :, None, :] - subspace_centroids[None, :, :, :]
        H = np.sum(Hdiff ** 2, axis=-1) # (Nq, S, nbits)


        # Broadcasting
        # (Nq,  1,  1) x
        # (1,   1,  S) x
        # (1,   N,  S) = 
        # (Nq,  N,  S)
        query_idx = np.arange(len(query))[:, None, None] # (Nq, 1, 1)
        subspace_idx = np.arange(self.S)[None, None, :] # (1, 1, S)
        databse_idx = self.database[None, :, :] # (1, N, S)
        Ddiff = H[query_idx, subspace_idx, databse_idx] # (Nq, N, S)
        
        # Sum up the subspace distances
        D = np.sum(Ddiff, axis=-1) # (Nq, N)

        # Compute I
        I = np.argpartition(D, kth=k-1, axis=1)[:, :k]  # (Nq, k)

        # Filter D
        query_indices = np.arange(D.shape[0])[:, None]
        D = D[query_indices, I]

        # Sort these as argpartition doesn't garentee order
        sorted_order = np.argsort(D, axis=1)
        I_sorted = I[query_indices, sorted_order]
        D_sorted = D[query_indices, sorted_order]

        return D_sorted, I_sorted






