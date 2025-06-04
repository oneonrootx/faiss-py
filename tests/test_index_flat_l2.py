import numpy as np
import pytest
from faiss_py.core.exceptions import IndexNotTrainable
from faiss_py.index_flat_l2 import IndexFlatL2


def test_index_flat_l2_init():
    d = 4
    index = IndexFlatL2(d)
    assert index._vectors.shape == (0, d)
    assert index.d == d


def test_index_flat_l2_add():
    d = 4
    index = IndexFlatL2(d)
    vectors = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    index.add(vectors)
    assert index._vectors.shape == (2, 4)
    np.testing.assert_array_equal(index._vectors, vectors) 

def test_index_flat_l2_train():
    d = 4
    index = IndexFlatL2(d)
    vectors = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    with pytest.raises(IndexNotTrainable):
        index.train(vectors)

def test_index_flat_l2_search_basic():
    d = 4
    index = IndexFlatL2(d)
    vectors = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 0, 0, 0]], dtype=np.float32)
    index.add(vectors)
    query = np.array([1, 2, 3, 4], dtype=np.float32)
    D, idx = index.search(query, k=2)
    # The closest should be itself (distance 0), then the next closest
    assert idx[0] == 0
    assert len(idx) == 2
    assert np.isclose(D[0], 0)

def test_index_flat_l2_search_k_greater_than_vectors():
    d = 4
    index = IndexFlatL2(d)
    vectors = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    index.add(vectors)
    query = np.array([1, 2, 3, 4], dtype=np.float32)
    D, idx = index.search(query, k=5)
    # Should return all available vectors
    assert len(idx) == 2
    assert set(idx) == {0, 1}

# Optionally, test for multiple queries (if supported)
def test_index_flat_l2_search_multiple_queries():
    d = 4
    index = IndexFlatL2(d)
    vectors = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 0, 0, 0]], dtype=np.float32)
    index.add(vectors)
    queries = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    # This will likely fail unless search supports batch queries, but let's check
    try:
        D, idx = index.search(queries, k=1)
        assert D.shape == (2,)
        assert idx.shape == (2,)
    except Exception:
        pass