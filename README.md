# faiss-py

This is a python implementation of [faiss](https://github.com/facebookresearch/faiss) for understanding the implementation of `faiss` without needing to understand C++. The only dependancy we will allow ourselves is numpy.

From the faiss wiki:

> Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python (versions 2 and 3). Some of the most useful algorithms are implemented on the GPU. It is developed primarily at Meta AI Research with help from external contributors.

___

Faiss is based on years of research. Of which, we will implement:

- [ ] The inverted file from “Video google: A text retrieval approach to object matching in videos.”, Sivic & Zisserman, ICCV 2003. This is the key to non-exhaustive search in large datasets. Otherwise all searches would need to scan all elements in the index, which is prohibitive even if the operation to apply for each element is fast

- [ ] The product quantization (PQ) method from “Product quantization for nearest neighbor search”, Jégou & al., PAMI 2011. This can be seen as a lossy compression technique for high-dimensional vectors, that allows relatively accurate reconstructions and distance computations in the compressed domain.

- [ ] The three-level quantization (IVFADC-R aka IndexIVFPQR) method from "Searching in one billion vectors: re-rank with source coding", Tavenard & al., ICASSP'11.

- [ ] The inverted multi-index from “The inverted multi-index”, Babenko & Lempitsky, CVPR 2012. This method greatly improves the speed of inverted indexing for fast/less accurate operating points.

- [ ] The optimized PQ from “Optimized product quantization”, He & al, CVPR 2013. This method can be seen as a linear transformation of the vector space to make it more amenable for indexing with a product quantizer.

- [ ] The pre-filtering of product quantizer distances from “Polysemous codes”, Douze & al., ECCV 2016. This technique performs a binary filtering stage before computing PQ distances.

- [ ] The GPU implementation and fast k-selection is described in “Billion-scale similarity search with GPUs”, Johnson & al, ArXiv 1702.08734, 2017

- [ ] The HNSW indexing method from "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", Malkov & al., ArXiv 1603.09320, 2016

- [ ] The in-register vector comparisons from "Quicker ADC : Unlocking the Hidden Potential of Product Quantization with SIMD", André et al, PAMI'19, also used in "Accelerating Large-Scale Inference with Anisotropic Vector Quantization", Guo, Sun et al, ICML'20.

- [ ] The binary multi-index hashing method from "Fast Search in Hamming Space with Multi-Index Hashing", Norouzi et al, CVPR’12.

- [ ] The graph-based indexing method NSG from "Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph", Cong Fu et al, VLDB 2019.

- [ ] The Local search quantization method from "Revisiting additive quantization", Julieta Martinez, et al. ECCV 2016 and "LSQ++: Lower running time and higher recall in multi-codebook quantization", Julieta Martinez, et al. ECCV 2018.

- [ ] The residual quantizer implementation from "Improved Residual Vector Quantization for High-dimensional Approximate Nearest Neighbor Search", Shicong Liu et al, AAAI'15.