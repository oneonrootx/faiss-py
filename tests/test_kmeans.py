

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


