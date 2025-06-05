import numpy as np

class Kmeans:

    def __init__(self, d: int, k: int):
        self.d = d
        self.k = k
        self.centroids = None
    
    def train(self, x, weights = None, init_centroids = None, niter: int = 100, nrounds: int = 20, seed: int  = None):
        """
        Trains the K-means clustering model on the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data of shape (n_samples, n_features).
        weights : np.ndarray, optional
            Sample weights (not currently used).
        init_centroids : np.ndarray, optional
            Initial centroids to use for clustering (not currently used).
        niter : int, default=100
            Number of iterations for each K-means run.
        nrounds : int, default=20
            Number of times to run K-means with different centroid seeds.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        None
            Updates the centroids attribute of the class.

        TODO:
        -------
        - [ ] create `IndexFlatL2` from centroids
        - [ ] use weights
        """
        if seed: np.random.seed(seed)

        # assertions
        if self.k >= len(x):
            raise ValueError("Kmeans requires k < len(x), pick a smaller k or pass more vectors to cluster.")
        if weights:
            raise NotImplementedError("`weights` argument has not been implemented for Kmeans.")


        # 0. initialise solution candidates
        soln_candidates = []

        for _ in range(nrounds if init_centroids is None else 1):
            
            # 1. choose k random points
            centroids = init_centroids if init_centroids is not None else x[np.random.choice(np.arange(len(x)), size=self.k)]

            for iter in range(niter):
                
                # 2. compute the distances between each vector and the centroids
                D_centroids = np.array([
                    np.linalg.norm(x - centroid, axis=1) for centroid in centroids
                ])

                # 3. get the label for each vector (index of closest centroid)
                labels = np.argmin(D_centroids, axis=0)

                # 4. if the labels haven't changes, we stop
                if iter > 0 and all(prev_labels == labels):
                    break

                # 4. get the new centroids
                centroids = [
                    x[labels == i].mean(axis=0) if np.any(labels == i) else x[np.random.choice(len(x))] for i in range(self.k)
                ]

                prev_labels = labels

            # 5. get within cluster sum of squares (residuals)
            total_wcss = sum(
                np.sum((x[labels == i] - centroids[i])**2)
                for i in range(self.k)
            )

            soln_candidates.append((total_wcss, labels))

        # 6. update state, assign centroids and create index
        self.centroids = centroids
        # self.index = ... TODO: add centroids to an index after `IndexFlatL2`

        # 7. return labels with lowest total variance
        soln = soln_candidates[np.argmin([var for var, _ in soln_candidates])]
        return soln[1]






        
        
    
