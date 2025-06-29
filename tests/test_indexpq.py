
import numpy as np
from faiss_py.helpers.make_blobs import make_blobs
from faiss_py.indexflatl2 import IndexFlatL2
from faiss_py.indexpq import IndexPQ



def test_indexpq_search():
    d, N, M, nbits = 64, 100, 8, 16
    index = IndexPQ(d=d, M=M, nbits=nbits, verbose=True)
    xb, _ = make_blobs(N, d, nbits, seed=42)
    xq = xb[:3]

    index.train(xb)
    index.add(xb)

    tindex = IndexFlatL2(d)
    tindex.add(xb)

    _, Itest = tindex.search(xq, k=7)
    _, I = index.search(xq, k=7)

    # assert sets of indexes overlap to some extent
    for i in range(len(xq)):
        intersection = len(set(I[i]) & set(Itest[i]))
        overlap_ratio = intersection / len(I[i])
        assert overlap_ratio >= 0.3, f"Query {i}: overlap ratio {overlap_ratio:.2f} too low"



def test_quantization_error_is_reasonable():
    d, N, M, nbits = 4, 10, 2, 4
    index = IndexPQ(d=d, M=M, nbits=nbits, verbose=True)
    xb, _ = make_blobs(N, d, nbits, seed=42)
    
    index.train(xb)

    vectors_to_encode = xb[np.random.choice(np.arange(len(xb)), size=5)]
    codes = index.encode(vectors_to_encode)
    reconstructed_vectors = index.decode(codes)

    for reconstructed_vector, original_vector in zip(reconstructed_vectors, vectors_to_encode):
        qerror = np.linalg.norm(reconstructed_vector - original_vector)
        assert qerror < 1, f"Quantization error too large: {qerror:.2f}"


def test_encode_decode_shape():
    d, M, nbits = 8, 4, 16
    index = IndexPQ(d=d, M=M, nbits=nbits)
    xb, _ = make_blobs(100, d, nbits, seed=42)
    
    index.train(xb)
    
    S = d // M  # Number of subspaces
    
    # Test single vector
    single_vector = xb[0]
    codes = index.encode(single_vector)
    assert codes.shape == (1, S), f"Expected shape (1, {S}), got {codes.shape}"
    
    reconstructed = index.decode(codes)
    assert reconstructed.shape == (1, d), f"Expected shape (1, {d}), got {reconstructed.shape}"
    
    # Test multiple vectors
    multi_vectors = xb[:5]
    codes = index.encode(multi_vectors)
    assert codes.shape == (5, S), f"Expected shape (5, {S}), got {codes.shape}"
    
    reconstructed = index.decode(codes)
    assert reconstructed.shape == (5, d), f"Expected shape (5, {d}), got {reconstructed.shape}"


def test_indexpq_edge_cases():
    d, M, nbits = 4, 2, 8
    index = IndexPQ(d=d, M=M, nbits=nbits)
    xb, _ = make_blobs(50, d, nbits, seed=42)
    
    index.train(xb)
    index.add(xb)
    
    # Test k=0
    xq = xb[:2]
    D, I = index.search(xq, k=0)
    assert D.shape == (2, 0), f"Expected shape (2, 0), got {D.shape}"
    assert I.shape == (2, 0), f"Expected shape (2, 0), got {I.shape}"
    
    # Test k larger than database
    D, I = index.search(xq, k=100)
    assert D.shape == (2, len(xb)), f"Expected shape (2, {len(xb)}), got {D.shape}"
    assert I.shape == (2, len(xb)), f"Expected shape (2, {len(xb)}), got {I.shape}"


def test_indexpq_dimension_validation():
    d, M, nbits = 7, 4, 16  # d not divisible by M
    
    try:
        IndexPQ(d=d, M=M, nbits=nbits)
        assert False, "Should have raised ValueError for d not divisible by M"
    except ValueError as e:
        assert "M must divide d" in str(e)


def test_indexpq_train_search_consistency():
    d, M, nbits = 16, 8, 32
    index = IndexPQ(d=d, M=M, nbits=nbits)
    xb, _ = make_blobs(200, d, nbits, seed=42)
    
    index.train(xb)
    index.add(xb)
    
    # Search for training vectors themselves
    xq = xb[:5]
    D, I = index.search(xq, k=1)
    
    # Check shapes
    assert D.shape == (5, 1), f"Expected shape (5, 1), got {D.shape}"
    assert I.shape == (5, 1), f"Expected shape (5, 1), got {I.shape}"
    
    # All distances should be relatively small since we're searching for vectors in the training set
    assert np.all(D < 2.0), f"Distances too large: {D.max():.2f}"
