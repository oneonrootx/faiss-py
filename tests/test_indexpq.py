
import numpy as np
from faiss_py.helpers.make_blobs import make_blobs
from faiss_py.indexflatl2.indexflatl2 import IndexFlatL2
from faiss_py.indexpq.indexpq import IndexPQ



def test_indexpq_search():
    d, N, M, nbits = 64, 1000, 8, 256
    index = IndexPQ(d=d, M=M, nbits=nbits, verbose=True)
    xb, _ = make_blobs(N, d, nbits, seed=42)
    xq = xb[:3]

    index.train(xb)
    index.add(xb)

    tindex = IndexFlatL2(d)
    tindex.add(xb)

    _, It = tindex.search(xq, k=7)

    _, I = index.search(xq, k=7)
    print(I, It)



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
