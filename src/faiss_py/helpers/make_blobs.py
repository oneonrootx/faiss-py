import numpy as np

def make_blobs(n: int, d: int, nblobs: int, spread: float = 0.1, seed: int = 42):
    """Generate n vectors of dimension d that concentrate around nblobs centroids.
    
    Parameters:
    -----
    n : int
        The number of vectors to returbn
    d : int
        The number of dimensions for each vector
    nblobs : int
        The number of blobs to generate 
    spread: float [0...1]
        How closely the vectors congregate around the centroids, e.g. spread = 0 means all points are exactly the centroid
    """
    np.random.seed(seed)
    
    blobs = []
    
    # Generate centroids that are well spread out in [0, 1]^d by multiplying by a factor (e.g., 8)
    blob_centroids = np.random.uniform(low=0.1, high=0.9, size=(nblobs, d)) * 8
    for i in range(n):
        centroid =  blob_centroids[i % nblobs]
        vector = centroid + (np.random.randn(d) * spread)
        blobs.append(vector)
    return np.array(blobs), blob_centroids
