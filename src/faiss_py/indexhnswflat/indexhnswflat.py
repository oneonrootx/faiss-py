from typing import List, Sequence, Tuple
import numpy as np
import logging
from faiss_py.core.index import Index
from faiss_py.core.graph import Node, VGraph


logger = logging.getLogger(__name__)

class IndexHNSWFlat(Index):
    
    def __init__(self, d: int, M: int, verbose: bool = False):
        super().__init__(d)
        self.M: int = M
        self._entry_node_id: int | None = None
        self._levels: List[VGraph] | None = None
        self.verbose = verbose

    def _get_level(self):
        """Here we define a distribution for getting the level as a function of M"""
        return int(np.floor(-np.log(np.random.random()) * (np.log(self.M) ** -1)))

    def train(self, vectors):
        vectors = np.asarray(vectors) # (N, d)
        # Let's start with a looped approach
        # we create the bottom level `l = 1`
        self._levels = [VGraph(directed=True, d=self.d)]
        self._entry_node_id = 0 # default entry node
        for curr_node_id, v in enumerate(vectors): # v ~ (d)
            l = self._get_level()
            node = Node[np.array](id=curr_node_id, data=v)

            # if the level has not been created, create it and update the entrypoint
            if l + 1 > len(self._levels):
                num_new_levels = l + 1 - len(self._levels)
                for _ in range(num_new_levels): 
                    self._levels.append(VGraph(directed=True, d=self.d))
                self._entry_node_id = node.id
            
            # add node to l, l - 1, .., 0 graphs
            for level_index in range(l + 1):
                vgraph = self._levels[level_index]

                # Add node first
                vgraph.add_node(curr_node_id, v)
                
                # 1. Get the M closest nodes in the layer (if any exist)
                if len(vgraph._node_ids) > 1:  # More than just the current node
                    _, node_ids = vgraph.knn(v, k=min(self.M, len(vgraph._node_ids) - 1))
                else:
                    node_ids = [[]]  # No other nodes to connect to

                # 2. set the edges from the new node to them
                for to_id in node_ids[0]:
                    vgraph.add_edge(from_id=curr_node_id, to_id=to_id)
                    vgraph.add_edge(from_id=to_id, to_id=curr_node_id)

                # 2. update the edges now (when Ne > M) to remove the worst neighbour
                if len(node_ids[0]):  # Only if there are nodes to connect to
                    edge_nodes = [vgraph.nodes[node_id] for node_id in node_ids[0]]
                    for edge_node in edge_nodes:
                        if len(edge_node.edges) > self.M:
                            dropped_node_id = vgraph.drop_worst_edge(edge_node.id)
                            if self.verbose: 
                                logger.info(f"Dropping edge {edge_node.id} -> {dropped_node_id}")

    def add(self, vectors):
        """Add vectors to the index (same as train for HNSW)"""
        return self.train(vectors)
    
    def search(self, query, k: int):
        """Search for k nearest neighbors"""
        if self._entry_node_id is None or not self._levels:
            raise Exception("Nothing to search; you must train the index first.")
        
        curr_entry_node_id = self._entry_node_id
        full_path: List[Tuple[int, Node[np.array]]] = []
        for level, vgraph in enumerate(reversed(self._levels)):
            final_node, path = vgraph.greedy_descent(curr_entry_node_id, query=query)
            full_path.extend([(level, node) for node in path])
            curr_entry_node_id = final_node.id # The next level must contain this node as S_{l} is in S_{l + 1}

        # need to return D & I
        return final_node, full_path



            
            



    

