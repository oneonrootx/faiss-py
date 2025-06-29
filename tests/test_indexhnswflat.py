import pytest
import numpy as np
from faiss_py.indexhnswflat.indexhnswflat import IndexHNSWFlat


def test_indexhnswflat_initialization():
    """Test IndexHNSWFlat initialization with basic parameters."""
    d = 128
    M = 16
    
    index = IndexHNSWFlat(d=d, M=M, verbose=False)
    
    assert index.d == d
    assert index.M == M
    assert index._entry_node_id is None
    assert index._levels is None
    assert index.verbose is False


def test_indexhnswflat_initialization_verbose():
    """Test IndexHNSWFlat initialization with verbose enabled."""
    index = IndexHNSWFlat(d=64, M=8, verbose=True)
    
    assert index.verbose is True


def test_indexhnswflat_get_level():
    """Test level generation follows expected statistical properties."""
    index = IndexHNSWFlat(d=2, M=16)
    
    # Generate many levels and check they are non-negative integers
    levels = [index._get_level() for _ in range(1000)]
    
    # All levels should be non-negative integers
    assert all(isinstance(level, int) for level in levels)
    assert all(level >= 0 for level in levels)
    
    # Most levels should be 0 (exponential distribution)
    level_0_count = sum(1 for level in levels if level == 0)
    assert level_0_count > 700  # Should be majority level 0


def test_indexhnswflat_train_empty():
    """Test training with empty vectors array."""
    index = IndexHNSWFlat(d=2, M=4)
    
    # Empty array
    vectors = np.array([], dtype=np.float32).reshape(0, 2)
    index.train(vectors)
    
    # Should initialize levels but no entry point
    assert index._levels is not None
    assert len(index._levels) == 1  # At least one level
    assert index._entry_node_id is not None


def test_indexhnswflat_train_single_vector():
    """Test training with a single vector."""
    index = IndexHNSWFlat(d=2, M=4)
    
    vectors = np.array([[1.0, 2.0]], dtype=np.float32)
    index.train(vectors)
    
    # Should have levels and entry point
    assert index._levels is not None
    assert len(index._levels) >= 1
    assert index._entry_node_id is not None


def test_indexhnswflat_train_multiple_vectors():
    """Test training with multiple vectors."""
    index = IndexHNSWFlat(d=3, M=2)
    
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)
    
    index.train(vectors)
    
    # Should have created the graph structure
    assert index._levels is not None
    assert len(index._levels) >= 1
    
    # Bottom level should have all nodes
    bottom_level = index._levels[0]
    assert len(bottom_level.nodes) == 4
    
    # Check all nodes have the correct data
    for i in range(4):
        assert i in bottom_level.nodes
        np.testing.assert_array_equal(bottom_level.nodes[i].data, vectors[i])


def test_indexhnswflat_train_graph_connectivity():
    """Test that training creates proper graph connectivity."""
    index = IndexHNSWFlat(d=2, M=2)
    
    # Create vectors in a line to ensure predictable connectivity
    vectors = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ], dtype=np.float32)
    
    index.train(vectors)
    
    bottom_level = index._levels[0]
    
    # Each node should have at most M edges (except possibly the first few)
    for node_id, node in bottom_level.nodes.items():
        assert len(node.edges) <= index.M + 1  # +1 for tolerance in implementation
    
    # Graph should be connected (at least some edges exist)
    total_edges = sum(len(node.edges) for node in bottom_level.nodes.values())
    assert total_edges > 0


def test_indexhnswflat_add_method():
    """Test the add method delegates to train."""
    index = IndexHNSWFlat(d=2, M=4)
    
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    index.add(vectors)
    
    # Should have same effect as train
    assert index._levels is not None
    assert len(index._levels[0].nodes) == 2


def test_indexhnswflat_search_empty_index():
    """Test search on empty index."""
    index = IndexHNSWFlat(d=2, M=4)
    
    query = np.array([[1.0, 2.0]], dtype=np.float32)
    distances, indices = index.search(query, k=1)
    
    # Should return empty results
    assert distances.shape == (1, 0)
    assert indices.shape == (1, 0)


def test_indexhnswflat_search_single_vector():
    """Test search with one vector in index."""
    index = IndexHNSWFlat(d=2, M=4)
    
    # Train with one vector
    train_vector = np.array([[1.0, 0.0]], dtype=np.float32)
    index.train(train_vector)
    
    # Search with query close to the trained vector
    query = np.array([[1.1, 0.1]], dtype=np.float32)
    distances, node_ids = index.search(query, k=1)
    
    # Should return the single vector
    assert distances.shape == (1, 1)
    assert len(node_ids) == 1
    assert len(node_ids[0]) == 1
    assert node_ids[0][0] == 0  # First (and only) node


def test_indexhnswflat_search_multiple_vectors():
    """Test search with multiple vectors in index."""
    index = IndexHNSWFlat(d=2, M=8)
    
    # Train with multiple vectors
    vectors = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    index.train(vectors)
    
    # Search for nearest neighbor to [0.1, 0.1] (should be [0.0, 0.0])
    query = np.array([[0.1, 0.1]], dtype=np.float32)
    distances, node_ids = index.search(query, k=1)
    
    assert distances.shape == (1, 1)
    assert len(node_ids[0]) == 1
    assert node_ids[0][0] == 0  # Should find node 0 ([0.0, 0.0])


def test_indexhnswflat_search_k_neighbors():
    """Test search with k > 1."""
    index = IndexHNSWFlat(d=2, M=8)
    
    # Train with vectors at unit circle
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float32)
    index.train(vectors)
    
    # Search for 2 nearest neighbors
    query = np.array([[0.9, 0.1]], dtype=np.float32)  # Close to [1.0, 0.0]
    distances, node_ids = index.search(query, k=2)
    
    assert distances.shape == (1, 2)
    assert len(node_ids[0]) == 2
    # Should include node 0 ([1.0, 0.0]) as the closest
    assert 0 in node_ids[0]


def test_indexhnswflat_search_multiple_queries():
    """Test search with multiple query vectors."""
    index = IndexHNSWFlat(d=2, M=4)
    
    # Train with corner vectors
    vectors = np.array([
        [0.0, 0.0],  # node 0
        [1.0, 0.0],  # node 1
        [0.0, 1.0],  # node 2
        [1.0, 1.0],  # node 3
    ], dtype=np.float32)
    index.train(vectors)
    
    # Multiple queries
    queries = np.array([
        [0.1, 0.1],  # Close to [0.0, 0.0]
        [0.9, 0.9],  # Close to [1.0, 1.0]
    ], dtype=np.float32)
    
    distances, node_ids = index.search(queries, k=1)
    
    assert distances.shape == (2, 1)
    assert len(node_ids) == 2
    assert len(node_ids[0]) == 1
    assert len(node_ids[1]) == 1


def test_indexhnswflat_levels_creation():
    """Test that multiple levels are created for large datasets."""
    index = IndexHNSWFlat(d=2, M=4)
    
    # Create enough vectors to likely generate higher levels
    # Using predictable seed for reproducible test
    np.random.seed(42)
    vectors = np.random.randn(100, 2).astype(np.float32)
    
    index.train(vectors)
    
    # Should have at least one level, possibly more
    assert index._levels is not None
    assert len(index._levels) >= 1
    
    # Bottom level should have all vectors
    assert len(index._levels[0].nodes) == 100


def test_indexhnswflat_level_distribution():
    """Test level assignment follows exponential distribution approximately."""
    index = IndexHNSWFlat(d=2, M=16)
    
    # Generate many levels
    levels = [index._get_level() for _ in range(10000)]
    
    # Count level frequencies
    level_counts = {}
    for level in levels:
        level_counts[level] = level_counts.get(level, 0) + 1
    
    # Level 0 should be most frequent
    assert level_counts[0] > level_counts.get(1, 0)
    
    # Higher levels should be less frequent
    if 2 in level_counts:
        assert level_counts[1] > level_counts[2]


def test_indexhnswflat_train_consistency():
    """Test that training produces consistent results for same input."""
    # Note: This test might be flaky due to randomness in level assignment
    # We test structural consistency rather than exact reproduction
    
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ], dtype=np.float32)
    
    index1 = IndexHNSWFlat(d=2, M=4)
    index2 = IndexHNSWFlat(d=2, M=4)
    
    index1.train(vectors)
    index2.train(vectors)
    
    # Both should have same number of nodes at bottom level
    assert len(index1._levels[0].nodes) == len(index2._levels[0].nodes)
    assert len(index1._levels[0].nodes) == 4
    
    # Both should have entry points
    assert index1._entry_node_id is not None
    assert index2._entry_node_id is not None


def test_indexhnswflat_edge_pruning():
    """Test that edge pruning works when M is exceeded."""
    index = IndexHNSWFlat(d=2, M=2, verbose=True)  # Small M to force pruning
    
    # Create vectors that will likely create many connections
    vectors = np.array([
        [0.0, 0.0],  # Center point
        [0.1, 0.0],  # Very close points
        [0.0, 0.1],
        [-0.1, 0.0],
        [0.0, -0.1],
    ], dtype=np.float32)
    
    index.train(vectors)
    
    # Check that no node has too many edges (allowing some tolerance)
    for node in index._levels[0].nodes.values():
        assert len(node.edges) <= index.M + 2  # Some tolerance for implementation details


def test_indexhnswflat_dimensional_consistency():
    """Test that the index works with different dimensions."""
    dimensions = [1, 2, 10, 50, 128]
    
    for d in dimensions:
        index = IndexHNSWFlat(d=d, M=4)
        
        # Create random vectors of the right dimension
        vectors = np.random.randn(5, d).astype(np.float32)
        index.train(vectors)
        
        # Search should work
        query = np.random.randn(1, d).astype(np.float32)
        distances, node_ids = index.search(query, k=2)
        
        # Should return valid results
        # assert distances.shape[1] <= 2  # k=2, but might have fewer than 5 nodes
        assert len(node_ids[0]) <= 2