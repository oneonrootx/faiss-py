import random
import numpy as np
import pytest
from faiss_py.core.graph import Edge, Graph, Node, NodeNotFoundError, NodeVectorIP, NodeVectorL2
from faiss_py.indexflatl2 import IndexFlatL2
import matplotlib.pyplot as plt

N, d = 1000, 2

@pytest.fixture
def fully_connected_vgraph():
    k = 4
    graph = Graph()
    nodes = [NodeVectorL2(id=idx, data=2 * np.random.random(d) - 1) for idx in range(N)]
    for node in nodes: graph.add_node(node)
    
    index = IndexFlatL2(d=d)
    index.add(np.array([node.data for node in nodes]))

    for node in nodes:
        _, I = index.search(node.data.reshape(1, -1), k=k)
        neighbours = np.array(nodes)[I[0]].tolist()
        for neighbour in neighbours:
            graph.add_edge(Edge(node.id, neighbour.id))
            graph.add_edge(Edge(neighbour.id, node.id))

    return graph

@pytest.fixture
def fully_connected_vgraph_ip():
    k = 4
    graph = Graph()
    nodes = []
    for idx in range(N):
        point = 2 * np.random.random(d) - 1
        point = point / np.linalg.norm(point) + np.random.random(d) / 5
        node = NodeVectorIP(id=idx, data=point)
        nodes.append(node)
    for node in nodes: graph.add_node(node)
    
    index = IndexFlatL2(d=d)
    index.add(np.array([node.data for node in nodes]))

    for node in nodes:
        _, I = index.search(node.data.reshape(1, -1), k=k)
        neighbours = np.array(nodes)[I[0]].tolist()
        for neighbour in neighbours:
            graph.add_edge(Edge(node.id, neighbour.id))
            graph.add_edge(Edge(neighbour.id, node.id))

    return graph



        


def test_init_NodeVectorL2():

    with pytest.raises(ValueError):
        NodeVectorL2(id=1, data=[[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        NodeVectorL2(id=1, data=np.random.random((2, d)))

    with pytest.raises(ValueError):
        NodeVectorL2(id=1, data=["a", "b"])

    NodeVectorL2(id=1, data=np.random.random((1, d)))


def test_add_node():
    node = NodeVectorL2(id=1, data=np.random.random((1, d)))
    graph = Graph()
    graph.add_node(node)


def test_add_edge():
    graph = Graph[np.typing.ArrayLike, Node]()

    node_from = NodeVectorL2(id=1, data=np.random.random(d))
    graph.add_node(node_from)
    
    node_to = NodeVectorL2(id=2, data=np.random.random((d)))
    graph.add_node(node_to)

    good_edge = Edge(from_id=node_from.id, to_id=node_to.id, weight=0)
    graph.add_edge(good_edge)

    len(graph.nodes) == 2

    assert isinstance(graph.nodes[1].edges[2], Edge)

    bad_edge = Edge(from_id=100, to_id=200, weight=0)
    with pytest.raises(NodeNotFoundError):
        graph.add_edge(bad_edge)


def test_depth_first_search(fully_connected_vgraph: Graph):
    tol = 0.01
    query = random.choice(fully_connected_vgraph.nodes).data
    best_node, path = fully_connected_vgraph.depth_first_search(entry_node_id=0, query=query, tol=tol)
    anim = fully_connected_vgraph.animate_search(path, query, best_node, interval=250)
    plt.show()


def test_breadth_first_search(fully_connected_vgraph: Graph):
    tol = 0.01
    query = random.choice(fully_connected_vgraph.nodes).data
    best_node, path = fully_connected_vgraph.breadth_first_search(entry_node_id=0, query=query, tol=tol)
    anim = fully_connected_vgraph.animate_search(path, query, best_node, interval=250)
    plt.show()


def test_greedy_search(fully_connected_vgraph: Graph):
    tol = 1e-3
    query = random.choice(fully_connected_vgraph.nodes).data
    best_node, path = fully_connected_vgraph.greedy_search(entry_node_id=0, query=query, tol=tol)
    anim = fully_connected_vgraph.animate_search(path, query, best_node, interval=250)
    plt.show() 


def test_best_first_search(fully_connected_vgraph: Graph):
    tol = 1e-3
    query = random.choice(fully_connected_vgraph.nodes).data
    best_node, path = fully_connected_vgraph.best_first_search(entry_node_id=0, query=query, tol=tol)
    anim = fully_connected_vgraph.animate_search(path, query, best_node, interval=250)
    plt.show()


def test_best_first_search_ip(fully_connected_vgraph_ip: Graph):
    tol = 1e-10
    query = random.choice(fully_connected_vgraph_ip.nodes).data
    best_node, path = fully_connected_vgraph_ip.best_first_search(entry_node_id=0, query=query, tol=tol)
    anim = fully_connected_vgraph_ip.animate_search(path, query, best_node, interval=250)
    plt.show()





