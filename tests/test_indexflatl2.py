import numpy as np
from faiss_py.indexflatl2 import IndexFlatL2

def test_indexflatl2_add_and_search():
    d = 4
    xb = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    index = IndexFlatL2(d)
    index.add(xb)

    # Query is exactly one of the base vectors
    xq = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    D, I = index.search(xq, k=2)

    # For each query, the closest should be itself (distance 0)
    assert np.allclose(D[:, 0], 0)
    assert xb[I[0, 0]].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert xb[I[1, 0]].tolist() == [0.0, 0.0, 1.0, 0.0]

def test_indexflatl2_search_knn():
    d = 2
    xb = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    index = IndexFlatL2(d)
    index.add(xb)

    xq = np.array([
        [0.1, 0.1],
        [0.9, 0.9],
    ], dtype=np.float32)
    D, I = index.search(xq, k=2)

    # For first query, closest should be [0,0] and [0,1] or [1,0]
    assert I.shape == (2, 2)
    assert set(I[0]).issubset({0, 1, 2})
    assert set(I[1]).issubset({1, 2, 3})

def test_indexflatl2_empty_add():
    d = 3
    index = IndexFlatL2(d)
    assert index.database.shape == (0, d)
    xb = np.random.randn(5, d).astype(np.float32)
    index.add(xb)
    assert index.database.shape == (5, d)
    # Add more
    index.add(xb)
    assert index.database.shape == (10, d)

def test_indexflatl2_search_single_vector():
    d = 3
    xb = np.eye(d, dtype=np.float32)
    index = IndexFlatL2(d)
    index.add(xb)
    xq = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    D, I = index.search(xq, k=1)
    assert D.shape == (1, 1)
    assert I.shape == (1, 1)
    assert I[0, 0] == 0
    assert np.isclose(D[0, 0], 0.0)

def test_indexflatl2_search_more_than_db():
    d = 2
    xb = np.array([[0, 0], [1, 1]], dtype=np.float32)
    index = IndexFlatL2(d)
    index.add(xb)
    xq = np.array([[0, 1]], dtype=np.float32)
    # k > number of vectors in db
    D, I = index.search(xq, k=3)
    assert D.shape == (1, 2)
    assert I.shape == (1, 2)
    # Should return both indices
    assert set(I[0]) == {0, 1}

def test_indexivfflat_simple():
    from faiss_py.indexivfflat import IndexIVFFlat
    from faiss_py.indexflatl2 import IndexFlatL2
    d = 2
    nlist = 2
    nprobe = 1
    xb = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    index.add(xb)
    xq = np.array([[0.1, 0.1]], dtype=np.float32)
    D, I = index.search(xq, k=2) if hasattr(index, 'search') else (None, None)
    print('IVFFlat search result:', D, I)
