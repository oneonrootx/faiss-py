# faiss-py

This is a python implementation of [faiss](https://github.com/facebookresearch/faiss) for understanding the implementation of `faiss` without needing to understand C++. The only dependancy we will allow ourselves is numpy.

From the faiss wiki:

> Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python (versions 2 and 3). Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Meta AI Research with help from external contributors.

___

## Roadmap

### Faiss building blocks

- [x] `Kmeans` - Clustering
- [ ] `PCAMatrix` - Dimensionality Reduction
- [ ] `Quantizer` - Quantization

### Basic Indexes

- [x] `IndexFlatL2` - Exact search for L2
- [x] `IndexFlatIP` - Exact search for inner product
- [ ] `IndexHNSWFlat` - Hierarchical Navigable Small World graph exploration
- [x] `IndexIVFFlat` - Inverted file with exact post-verification
- [ ] `IndexLSH` - Locality-Sensitive Hashing (binary flat index)
- [ ] `IndexScalarQuantizer` - Scalar quantizer (SQ) in flat mode
- [x] `IndexPQ` - Product quantizer (PQ) in flat mode
- [ ] `IndexIVFScalarQuantizer` - IVF and scalar quantizer
- [ ] `IndexIVFPQ` - IVFADC (coarse quantizer + PQ on residuals)
- [ ] `IndexIVFPQR` - IVFADC+R (same as IVFADC with re-ranking based on codes)
