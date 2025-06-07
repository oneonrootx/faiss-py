import numpy as np
from tqdm import tqdm
from faiss_py import logger
from faiss_py.indexflatl2 import IndexFlatL2


class Kmeans:
    """
    Kmeans clustering class for partitioning data into k clusters using L2 distance.

    Features:
    - k-means++ initialization
    - multiple random restarts (rounds), returns best clustering by WCSS
    - builds an IndexFlatL2 over centroids after training

    Attributes
    ----------
    d : int
        Dimensionality of the input vectors.
    k : int
        Number of clusters.
    centroids : np.ndarray
        Array of shape (k, d) with the cluster centroids after training.
    index : IndexFlatL2
        IndexFlatL2 built over the centroids for fast nearest-centroid search.
    labels : np.ndarray
        Array of shape (n_samples,) with the cluster assignment for each input vector after training.
    """

    def __init__(self, d: int, k: int, verbose: bool = False):
        self.d = d
        self.k = k
        self.verbose = verbose
        self.centroids = None
        self.index = None
        self.labels = None
        
        if self.verbose:
            logger.info("Initialized Kmeans with d=%d, k=%d", d, k)

    def _initialize_centroids_kmeans_pp(self, x, seed=None):
        """
        Initialize centroids using k-means++ algorithm.

        Parameters
        ----------
        x : np.ndarray
            Data array (n_samples, d)
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        centroids : np.ndarray
            Initialized centroids array (k, d)
        """
        if seed is not None:
            np.random.seed(seed)

        n_samples, d = x.shape
        centroids = np.empty((self.k, d), dtype=x.dtype)

        # Choose the first centroid randomly
        centroid_idx = np.random.choice(n_samples)
        centroids[0] = x[centroid_idx]

        # Compute initial distances from first centroid
        dist_sq = np.sum((x - centroids[0]) ** 2, axis=1)

        for i in range(1, self.k):
            # Choose next centroid weighted by squared distance
            probs = dist_sq / dist_sq.sum()
            centroid_idx = np.random.choice(n_samples, p=probs)
            centroids[i] = x[centroid_idx]

            # Update distances with new centroid
            new_dist_sq = np.sum((x - centroids[i]) ** 2, axis=1)
            dist_sq = np.minimum(dist_sq, new_dist_sq)

        return centroids

    def train(self, x, weights=None, init_centroids=None, niter=100, nrounds=20, seed=None):
        """
        Train the K-means clustering model on the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (n_samples, d).
        weights : np.ndarray, optional
            Sample weights (not implemented).
        init_centroids : np.ndarray, optional
            Initial centroids to use for clustering.
        niter : int, default=100
            Number of iterations for each K-means run.
        nrounds : int, default=20
            Number of random restarts.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        labels : np.ndarray
            Cluster assignments for each input vector.
        """
        if seed is not None:
            np.random.seed(seed)

        if self.k >= len(x):
            raise ValueError("Kmeans requires k < len(x), pick a smaller k or pass more vectors to cluster.")
        if weights is not None:
            raise NotImplementedError("`weights` argument has not been implemented for Kmeans.")

        if self.verbose:
            logger.info("Training Kmeans on %d vectors with niter=%d, nrounds=%d", len(x), niter, nrounds)

        soln_candidates = []

        rounds = 1 if init_centroids is not None else nrounds

        for _ in tqdm(range(rounds), desc="KMeans rounds", disable=not self.verbose):
            # Initialize centroids
            if init_centroids is not None:
                centroids = init_centroids
            else:
                centroids = self._initialize_centroids_kmeans_pp(x, seed=seed)

            prev_labels = None

            for iter in range(niter):
                # Vectorized distance: (k, n_samples)
                # squared L2 distances without sqrt for speed
                diff = x[None, :, :] - centroids[:, None, :]  # (k, n_samples, d)
                D_centroids = np.sum(diff ** 2, axis=-1)       # (k, n_samples)

                labels = np.argmin(D_centroids, axis=0)        # (n_samples,)

                if prev_labels is not None and np.array_equal(labels, prev_labels):
                    break

                # Update centroids
                new_centroids = []
                for i in range(self.k):
                    assigned = x[labels == i]
                    if len(assigned) == 0:
                        # Handle empty cluster by random reinitialization
                        new_centroids.append(x[np.random.choice(len(x))])
                    else:
                        new_centroids.append(assigned.mean(axis=0))
                centroids = np.vstack(new_centroids)

                prev_labels = labels

            # Compute total within-cluster sum of squares (WCSS)
            total_wcss = 0.0
            for i in range(self.k):
                assigned = x[labels == i]
                if len(assigned) > 0:
                    total_wcss += np.sum((assigned - centroids[i]) ** 2)

            soln_candidates.append((total_wcss, labels.copy(), centroids.copy()))

        # Pick best solution by minimal WCSS
        best_idx = np.argmin([wcss for wcss, _, _ in soln_candidates])
        best_wcss, best_labels, best_centroids = soln_candidates[best_idx]

        self.centroids = best_centroids
        self.labels = best_labels
        self.index = IndexFlatL2(self.d, verbose=False)
        self.index.add(self.centroids)

        if self.verbose:
            logger.info("Kmeans training completed, best WCSS: %.6f", best_wcss)

        return self.labels
