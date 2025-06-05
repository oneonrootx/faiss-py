import numpy as np
import pytest

from faiss_py.indexflatip import IndexFlatIP

def test_indexflatip_add_and_search_basic():
    d = 4
    index = IndexFlatIP(d)
    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    index.add(vectors)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    D, I = index.search(query, k=2)
    assert D.shape == (1, 2)
    assert I.shape == (1, 2)
    # The closest should be the first vector (index 0)
    assert I[0, 0] == 0
    # The dot product with itself should be 1.0
    assert np.isclose(D[0, 0], 1.0)

def test_indexflatip_search_multiple_queries():
    d = 3
    index = IndexFlatIP(d)
    vectors = np.eye(d, dtype=np.float32)
    index.add(vectors)
    queries = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    D, I = index.search(queries, k=1)
    assert D.shape == (2, 1)
    assert I.shape == (2, 1)
    assert I[0, 0] == 0
    assert I[1, 0] == 1
    assert np.isclose(D[0, 0], 1.0)
    assert np.isclose(D[1, 0], 1.0)

def test_indexflatip_add_multiple_times():
    d = 2
    index = IndexFlatIP(d)
    v1 = np.array([[1.0, 0.0]], dtype=np.float32)
    v2 = np.array([[0.0, 1.0]], dtype=np.float32)
    index.add(v1)
    index.add(v2)
    assert index.database.shape == (2, d)
    query = np.array([0.0, 1.0], dtype=np.float32)
    D, I = index.search(query, k=2)
    assert set(I[0]) == {0, 1}

def test_indexflatip_search_k_greater_than_database():
    d = 2
    index = IndexFlatIP(d)
    vectors = np.eye(d, dtype=np.float32)
    index.add(vectors)
    query = np.array([1.0, 1.0], dtype=np.float32)
    # k > number of vectors in database
    D, I = index.search(query, k=10)
    # Should only return as many as in database
    assert D.shape == (1, 2)
    assert I.shape == (1, 2)

def test_indexflatip_train_raises():
    d = 2
    index = IndexFlatIP(d)
    with pytest.raises(NotImplementedError):
        index.train(np.eye(d, dtype=np.float32))

def test_indexflatip_empty_database_search():
    d = 3
    index = IndexFlatIP(d)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Should not crash, but k=0
    D, I = index.search(query, k=1)
    assert D.shape == (1, 0)
    assert I.shape == (1, 0)
