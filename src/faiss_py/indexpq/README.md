# IndexPQ
___

Will come back and flesh out the basics later, rn I need some space to think about ADC:

## ADC (Asymmetric Distance Computation)

ADC, or Asymmetric Distance Computation, is a key technique in Product Quantization (PQ) for efficient approximate nearest neighbor search. In ADC, the database vectors are encoded (quantized) using PQ, while the query vector remains uncompressed. The search is performed by computing distances between the uncompressed query and the quantized representations of the database vectors.

## Vectorising

Asm. we have a codebook (map between subspaces and `IndexFlatL2`s with centroids for that subspace).

Consider:

- `N` the number of vectors (quantized) in the database
- `d` the dimension of the vectors
- `M` the dimension of the subspaces
- `S` the number of subspaces (`d // M`)
- `nbits` the number of centroids per subspace
- `Nq` the number of query vectors

1. Partition the input queries

`query` has shape `(Nq, d)` and we would like to reshape it as `(Nq, S, M)`, this has just split up every `d` length vector into `S` lenght `M` vectors. Nice, we do this with:

```python
query_split = query.reshape(query.shape[0], -1, M)
```

2. Now, we want to find the distance from each vector and each centroid in the relevant subspace.

We can build the `subspace_centroids` from the cookbooks constructed during training:

```python
subspace_centroids = np.array([
    [codebooks[subspace_index].database] 
    for subspace_index in range(S)
])
```

There `subspace_centroids` then has shape `(S, nbits, M)` where the index on the zeor axis corresponds to the subspace.

3. Now, we want to compute H `(Nq, S, nbits)`, where `H_ijk` is the distance squared between `kth` centroid in the `jth` subspace for `ith` query vector's `jth` subvector.

So, how to we take `subspace_centroids (S, nbits, M)` and `query_split (Nq, S, M)` and produce `H`.

We might start by considering a `Hdiff` which is the element-wise difference between the centroids and subvectors, such a tensort would have shape `(Nq, S, nbits, M)` and now the broadcasting looks a bit more obvious

```python
(Nq,    S,      1,      M) -
(1,     S,      nbits,  M) = 
(Nq,    S,      nbits,  M)
```

Nice, now the broadcasting will work so we can build `Hdiff` via:

```python
Hdiff = query_split[:, :, None, :] - subspace_centroids[None, :, :, :]
```

And then we can compute `H` via:

```python
H = np.sum(Hdiff ** 2, axis=-1) # (Nq, S, nbits)
```

4. ADC time, we now can take the codes in our database and read out the quantised distances in each subspace from `H` and then sum them up! That's our quantised distance between the query vector and the database vector.

Note:

```python
database (N, M) # where each value is a number in [0, nbits)
```

So, how do we do this! We want to get `(Nq, N, S)` and then summing along the last axis gives us `D (Nq, N)` where `D_ij` is the quantised distance between query vector `i` and database vector `j`. At that point it's easy to compute argmax's and return a result!