import itertools
import numpy as np

from faiss_py.core.index import Index
from faiss_py.kmeans.kmeans import Kmeans


class IndexIVFFlat(Index):
    """
    Inverted File with Flat quantization (IVFFlat) index.

    This index partitions the vector space into `nlist` cells using a quantizer (e.g., KMeans).
    Each cell contains a subset of the database vectors. During search, only the vectors in the
    closest `nprobe` cells are searched using a flat index (L2 or IP).
    """

    def __init__(self, quantizer: type[Index], d: int, nlist: int, nprobe: int):
        """
        Initialize the IVFFlat index.

        Parameters
        ----------
        quantizer : type[Index]
            The class of the quantizer to use for cell assignment (e.g., IndexFlatL2 or IndexFlatIP).
        d : int
            Dimensionality of the vectors.
        nlist : int
            Number of cells (clusters) to partition the space into.
        nprobe : int
            Number of cells to probe during search.
        """
        super().__init__(d)
        self.quantizer = quantizer
        self.nlist = nlist
        self.nprobe = nprobe
        self.database = np.empty((0, d), dtype=np.float32)
        
        self.cell_index = None
        self.cells = None

    def train(self, vectors):
        """
        Train the quantizer (KMeans) and assign database vectors to cells.

        Parameters
        ----------
        vectors : np.ndarray
            Training vectors of shape (N, d).
        """
        kmeans = Kmeans(self.d, k=self.nlist)
        kmeans.train(vectors)
        self.cell_index = kmeans.index

        labels = kmeans.labels  # shape (N,)
        self.cells = {}

        for cell in np.unique(labels):
            mask = labels == cell
            self.cells[cell] = {
                "vectors": vectors[mask],
                "indices": np.nonzero(mask)[0]  # global positions in original `vectors`
            }

    def add(self, vectors):
        """
        Add vectors to the database.

        Parameters
        ----------
        vectors : np.ndarray
            Vectors to add, shape (n, d).
        """
        self.database = np.concatenate((self.database, vectors))


    def add(self, vectors):
        """
        Add vectors to the index database. These vectors will be available for search after being assigned to cells.

        Parameters
        ----------
        vectors : np.ndarray
            Vectors to add, shape (n, d).
        """
        self.database = np.concatenate((self.database, vectors))


    def search(self, query, k: int):
        """
        Search the index for the top-k nearest neighbors of each query vector.

        Parameters
        ----------
        query : np.ndarray
            Query vectors of shape (m, d) or (d,).
        k : int
            Number of nearest neighbors to return.

        Returns
        -------
        D : np.ndarray
            Array of shape (m, k) with the distances (or similarities) to the nearest neighbors.
        I : np.ndarray
            Array of shape (m, k) with the indices of the nearest neighbors in the original database.
        """

        # 1. Get top-nprobe cell IDs for each query
        _, I = self.cell_index.search(query, k=self.nprobe)
        
        Dout, Iout = [], []

        for Iq, q in zip(I, query):
            # 2. Collect all vectors and their original indices from the selected cells
            filtered_vectors = []
            filtered_indices = []

            for cell_id in Iq:
                cell = self.cells[cell_id]
                filtered_vectors.append(cell["vectors"])
                filtered_indices.append(cell["indices"])

            # 3. Concatenate for use in flat index
            filtered_vectors = np.concatenate(filtered_vectors, axis=0)
            filtered_indices = np.concatenate(filtered_indices, axis=0)

            # 4. Build temporary index over filtered vectors
            index: Index = self.quantizer(self.d)
            index.add(filtered_vectors)

            # 5. Run search over local filtered database
            Dtemp, Itemp_local = index.search(q[None, :], k=k)  # shape (1, k)
            Dtemp = Dtemp[0]
            Itemp_local = Itemp_local[0]

            # 6. Map local result indices back to global indices
            Itemp_global = filtered_indices[Itemp_local]

            # 7. Append results
            Dout.append(Dtemp)
            Iout.append(Itemp_global)

        return np.array(Dout), np.array(Iout)


