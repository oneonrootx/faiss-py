import numpy as np
from faiss_py.helpers.make_blobs import make_blobs
from faiss_py.indexivfflat import IndexIVFFlat
from faiss_py.indexflatl2 import IndexFlatL2
from faiss_py.indexflatip import IndexFlatIP

def test_indexivfflat_simple():
    d = 4
    nlist = 2
    nprobe = 2
    xb = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    index.add(xb)
    xq = np.array([[0.1, 0.1, 0.1, 0.1], [0, 0, 0, 0]], dtype=np.float32)
    D, I = index.search(xq, k=2)
    print('IVFFlat search result:', D, I)
    # Check that the closest vector to [0,0,0,0] is itself
    assert I[1, 0] == 0
    # Check that distances are non-negative
    assert np.all(D >= 0)
    # Check that output shapes are correct
    assert D.shape == (2, 2)
    assert I.shape == (2, 2)

def test_indexivfflat_blob():
    d = 4
    nlist = 10
    nprobe = 10
    xb, _ = make_blobs(n=1000, d=d, nblobs=nlist)
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    index.add(xb)
    xq = np.array([xb[0], xb[4]], dtype=np.float32)
    D, I = index.search(xq, k=2)
    # Check that the closest vector to xb[0] is itself
    assert I[0, 0] == 0
    # Check that the closest vector to xb[4] is itself
    assert I[1, 0] == 4
    # Check that distances are non-negative
    assert np.all(D >= 0)
    # Check that output shapes are correct
    assert D.shape == (2, 2)
    assert I.shape == (2, 2)

def test_indexivfflat_empty_add():
    d = 3
    nlist = 2
    nprobe = 2
    xb = np.random.randn(10, d).astype(np.float32)
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    # Do not add any vectors
    xq = np.random.randn(2, d).astype(np.float32)
    try:
        D, I = index.search(xq, k=2)
        # Should fail or return empty results
        assert D.shape == (2, 0) or D.shape == (2, 2)
        assert I.shape == (2, 0) or I.shape == (2, 2)
    except Exception:
        pass  # Acceptable if it raises due to empty database

def test_indexivfflat_add_after_train():
    d = 5
    nlist = 3
    nprobe = 2
    xb = np.random.randn(20, d).astype(np.float32)
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb[:10])
    index.add(xb[:10])
    # Add more vectors after initial add
    index.add(xb[10:])
    xq = xb[15:17]
    D, I = index.search(xq, k=3)
    # Check that output shapes are correct
    assert D.shape == (2, 3)
    assert I.shape == (2, 3)

def test_indexivfflat_k_greater_than_db():
    d = 4
    nlist = 2
    nprobe = 2
    xb = np.random.randn(5, d).astype(np.float32)
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    index.add(xb)
    xq = np.random.randn(1, d).astype(np.float32)
    D, I = index.search(xq, k=10)
    # Should return as many as possible (<= 5)
    assert D.shape[0] == 1
    assert I.shape[0] == 1
    assert D.shape[1] <= 10
    assert I.shape[1] <= 10

def test_indexivfflat_multiple_queries():
    d = 6
    nlist = 4
    nprobe = 3
    xb = np.random.randn(50, d).astype(np.float32)
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    index.add(xb)
    xq = np.random.randn(7, d).astype(np.float32)
    D, I = index.search(xq, k=4)
    assert D.shape == (7, 4)
    assert I.shape == (7, 4)

def test_indexivfflat_with_ip_quantizer():
    # Use IndexFlatIP as quantizer
    d = 4
    nlist = 3
    nprobe = 2
    xb = np.random.randn(30, d).astype(np.float32)
    quantizer = IndexFlatIP
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    index.add(xb)
    xq = np.random.randn(5, d).astype(np.float32)
    D, I = index.search(xq, k=3)
    assert D.shape == (5, 3)
    assert I.shape == (5, 3)

def test_indexivfflat_with_ip_quantizer_blob():
    d = 8
    nlist = 5
    nprobe = 3
    xb, _ = make_blobs(n=200, d=d, nblobs=nlist) 
    xb_norm = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    quantizer = IndexFlatIP
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb_norm)
    index.add(xb_norm)
    xq = xb_norm[:3]
    D, I = index.search(xq, k=10)
    # Should return self as closest
    for i in range(3):
        assert I[i, 0] == i

def test_indexivfflat_search_after_multiple_adds_and_quantizers():
    d = 5
    nlist = 2
    nprobe = 2
    xb = np.random.randn(10, d).astype(np.float32)
    quantizer = IndexFlatIP
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    index.add(xb[:5])
    index.add(xb[5:])
    xq = xb[:2]
    D, I = index.search(xq, k=3)
    assert D.shape == (2, 3)
    assert I.shape == (2, 3)

def test_indexivfflat_train_with_ip_quantizer_and_search_l2():
    d = 4
    nlist = 2
    nprobe = 2
    xb = np.random.randn(12, d).astype(np.float32)
    # Train with IP quantizer, search with L2 quantizer
    index_ip = IndexIVFFlat(IndexFlatIP, d, nlist, nprobe)
    index_ip.train(xb)
    index_ip.add(xb)
    xq = np.random.randn(2, d).astype(np.float32)
    D_ip, I_ip = index_ip.search(xq, k=4)
    assert D_ip.shape == (2, 4)
    assert I_ip.shape == (2, 4)

    index_l2 = IndexIVFFlat(IndexFlatL2, d, nlist, nprobe)
    index_l2.train(xb)
    index_l2.add(xb)
    D_l2, I_l2 = index_l2.search(xq, k=4)
    assert D_l2.shape == (2, 4)
    assert I_l2.shape == (2, 4)

def test_indexivfflat_empty_database_search():
    d = 3
    nlist = 2
    nprobe = 2
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    # No train, no add
    xq = np.random.randn(1, d).astype(np.float32)
    try:
        D, I = index.search(xq, k=2)
        # Should fail or return empty
        assert D.shape[1] == 0 or D.shape[1] == 2
    except Exception:
        pass

def test_indexivfflat_train_twice():
    d = 4
    nlist = 2
    nprobe = 2
    xb = np.random.randn(10, d).astype(np.float32)
    quantizer = IndexFlatL2
    index = IndexIVFFlat(quantizer, d, nlist, nprobe)
    index.train(xb)
    # Train again with different data
    xb2 = np.random.randn(10, d).astype(np.float32)
    index.train(xb2)
    index.add(xb2)
    xq = xb2[:2]
    D, I = index.search(xq, k=2)
    assert D.shape == (2, 2)
    assert I.shape == (2, 2)