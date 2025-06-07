# IndexHSSWFlat

From the `fiass` wiki:

---

The Hierarchical Navigable Small World indexing method is based on a graph built on the indexed vectors. At search time, the graph is explored in a way that converges to the nearest neighbors as quickly as possible. The IndexHNSW uses a flat index as underlying storage to quickly access the database vectors and abstract the compression / decompression of vectors. HNSW depends on a few important parameters:

- `M` is the number of neighbors used in the graph. A larger M is more accurate but uses more memory

- `efConstruction` is the depth of exploration at add time

- `efSearch` is the depth of exploration of the search

## Supported encodings

IndexHNSW supports the following Flat indexes: IndexHNSWFlat (no encoding), IndexHNSWSQ (scalar quantizer), IndexHNSWPQ (product quantizer), IndexHNSW2Level (two-level encoding).

## Supported operations

In addition to the restrictions of the Flat index HNSW uses, HNSW does not support removing vectors from the index. This would destroy the graph structure.

---

## So, how does it work?



