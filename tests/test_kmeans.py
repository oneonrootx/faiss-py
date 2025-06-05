

import faiss_py
import numpy as np


from faiss_py.helpers.make_blobs import make_blobs
import faiss_py.kmeans


def test_kmeans():
    d, n, k = 2, 1000, 3
    kmeans_py = faiss_py.kmeans.Kmeans(d, k)
    blobs, true_centroids = make_blobs(n=n, d=d, nblobs=k, spread=0.01, seed=42)
    kmeans_py.train(blobs, seed=42)

    # For each found centroid, find the closest true centroid and sum the distances
    found_centroids = kmeans_py.centroids
    distances = []
    for fc in found_centroids:
        min_dist = np.min(np.linalg.norm(true_centroids - fc, axis=1))
        distances.append(min_dist)
    total_distance = np.sum(distances)

    assert (total_distance / k) < 0.01

def test_kmeans_init_centroids():
    d, n, k = 2, 1000, 3
    kmeans_py = faiss_py.kmeans.Kmeans(d, k)
    blobs, true_centroids = make_blobs(n=n, d=d, nblobs=k, spread=0.01, seed=42)
    kmeans_py.train(blobs, init_centroids=true_centroids, seed=42)

    # For each found centroid, find the closest true centroid and sum the distances
    found_centroids = kmeans_py.centroids
    distances = []
    for fc in found_centroids:
        min_dist = np.min(np.linalg.norm(true_centroids - fc, axis=1))
        distances.append(min_dist)
    total_distance = np.sum(distances)

    assert (total_distance / k) < 0.01



def test_kmeans_index_search():
    d, n, k = 2, 100, 4
    kmeans_py = faiss_py.kmeans.Kmeans(d, k)
    blobs, true_centroids = make_blobs(n=n, d=d, nblobs=k, spread=0.05, seed=123)
    kmeans_py.train(blobs, seed=123)

    # Check that index is created and is an IndexFlatL2
    assert kmeans_py.index is not None
    assert hasattr(kmeans_py.index, "search")
    # The centroids should be in the index
    assert kmeans_py.index.database.shape == (k, d)

    # Now, for each centroid, search for its nearest centroid in the index
    # Should be itself (distance 0)
    D, I = kmeans_py.index.search(kmeans_py.centroids, k=1)
    assert D.shape == (k, 1)
    assert I.shape == (k, 1)
    # Each centroid's nearest neighbor in the index should be itself (index matches)
    for idx in range(k):
        assert I[idx, 0] == idx
        assert np.isclose(D[idx, 0], 0.0)
