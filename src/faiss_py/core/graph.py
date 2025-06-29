"""
Basic Graph Class with basic search utilities 
* greedy search
* bfs
* best-first search (when heuristic is defined)
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Deque, Iterable, Set
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

import numpy as np

class GraphError(Exception): 
    """Base exception for graph-related errors."""
    ...
class NodeAlreadyExistsError(GraphError): 
    """Raised when attempting to add a node that already exists."""
    ...
class NodeNotFoundError(GraphError): 
    """Raised when attempting to access a node that doesn't exist."""
    ...


class Edge:
    """Represents a weighted edge between two nodes in a graph.
    
    Args:
        from_id: ID of the source node
        to_id: ID of the destination node  
        weight: Weight of the edge (default: 0.0)
    """

    def __init__(self, from_id: int, to_id: int, weight: np.float64 = 0.0):
        self.from_id = from_id
        self.to_id = to_id
        self.weight = weight

class Node[TData](ABC):
    """Abstract base class for graph nodes.
    
    Args:
        id: Unique identifier for the node
        data: Data stored in the node
    """

    def __init__(self, id: int, data: TData) -> None:
        self.id = id
        self.data = data
        self.edges: dict[int, Edge] = dict()

    @abstractmethod
    def metric(self, data: TData) -> np.float64: 
        """Calculate distance/similarity metric between this node's data and query data.
        
        Args:
            data: Query data to compare against
            
        Returns:
            Distance/similarity score
        """
        ...

    def __repr__(self):
        return f"<Node id={self.id}>"



class Graph[TData, TNode: Node[TData]]:
    """Generic graph data structure with search algorithms.
    
    Supports various search strategies including DFS, BFS, best-first, and greedy search.
    """

    def __init__(self):
        self.nodes: dict[int, TNode] = dict()

    def add_node(self, node: TNode):
        """Add a node to the graph.
        
        Args:
            node: Node to add
            
        Raises:
            NodeAlreadyExistsError: If a node with the same ID already exists
        """
        if node.id in self.nodes:
            raise NodeAlreadyExistsError
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        """Add an edge to the graph.
        
        Args:
            edge: Edge to add
            
        Raises:
            NodeNotFoundError: If either source or destination node doesn't exist
        """
        if edge.from_id not in self.nodes:
            raise NodeNotFoundError(f"Node with id {edge.from_id} not found in graph.")
        if edge.to_id not in self.nodes:
            raise NodeNotFoundError(f"Node with id {edge.to_id} not found in graph.")
        self.nodes[edge.from_id].edges[edge.to_id] = edge

    def _generic_search[TFrontier: Iterable](
        self, 
        entry_node_id: int, 
        query: TData, 
        frontier_factory: Callable[[], TFrontier],
        pop_fn: Callable[[TFrontier], TNode], 
        push_fn: Callable[[TFrontier, list[TNode]], None], 
        expand_filter_fn: Callable[[list[TNode]], list[TNode]] | None  = None, 
        *, 
        tol: float = 0.0  
    ):
        """Generic search algorithm template.
        
        Args:
            entry_node_id: ID of the starting node
            query: Query data to search for
            pop_fn: Function to pop nodes from frontier (determines search order)
            push_fn: Function to add nodes to frontier
            tol: Distance tolerance for early termination (default: 0.0)
            
        Returns:
            Tuple of (best_node, search_path)
            
        Raises:
            NodeNotFoundError: If entry node doesn't exist
        """ 
        if entry_node_id not in self.nodes:
            raise NodeNotFoundError

        entry_node = self.nodes[entry_node_id]

        visited: Set[int] = set()
        seen: Set[int] = set([entry_node_id])
        frontier: TFrontier = frontier_factory()
        push_fn(frontier, [entry_node])

        path: list[TNode] = []
        best_node: TNode | Node = None
        best_distance: float = float("inf")
        
        while frontier:
            # Pop & Handle
            current_node = pop_fn(frontier)
            path.append(current_node)
            visited.add(current_node.id)
                
            if (distance := current_node.metric(query)) < best_distance:
                best_distance = distance
                best_node = current_node
                # Break early if node is good enough
                if float(distance) <= tol: break
            
            # Expand
            neighbours = [self.nodes[node_id] for node_id in current_node.edges.keys() if node_id not in visited | seen]
            neighbours = expand_filter_fn(neighbours) if expand_filter_fn and neighbours else neighbours
            seen.update([node.id for node in neighbours])
            push_fn(frontier, neighbours)

        return best_node, path

    def depth_first_search(self, entry_node_id: int, query: TData, *, tol: float = 0.0):
        """Perform depth-first search to find the best matching node.
        
        Args:
            entry_node_id: ID of the starting node
            query: Query data to search for
            tol: Distance tolerance for early termination (default: 0.0)
            
        Returns:
            Tuple of (best_node, search_path)
        """
        return self._generic_search(
            entry_node_id=entry_node_id,
            query=query,
            frontier_factory=lambda: deque(),
            pop_fn=lambda q: q.pop(),
            push_fn=lambda q, nodes: q.extend(nodes),
            tol=tol
        )

    def breadth_first_search(self, entry_node_id: int, query: TData, *, tol: float = 0.0):
        """Perform breadth-first search to find the best matching node.
        
        Args:
            entry_node_id: ID of the starting node
            query: Query data to search for
            tol: Distance tolerance for early termination (default: 0.0)
            
        Returns:
            Tuple of (best_node, search_path)
        """
        return self._generic_search(
            entry_node_id=entry_node_id,
            query=query,
            frontier_factory=lambda: deque(),
            pop_fn=lambda q: q.popleft(),
            push_fn=lambda q, nodes: q.extend(nodes),
            tol=tol
        )

    def best_first_search(self, entry_node_id: int, query: TData, *, tol: float = 0.0): 
        """Perform best-first search to find the best matching node.
        
        Args:
            entry_node_id: ID of the starting node
            query: Query data to search for
            
        Returns:
            Tuple of (best_node, search_path)
        """
        def best_first_search_pop_fn(frontier: list[TNode]):
            _, node = heapq.heappop(frontier)
            return node
        
        def best_first_search_push_fn(frontier: list[TNode], nodes: list[TNode]):
            for node in nodes:
                heapq.heappush(frontier, (node.metric(query), node))

        return self._generic_search(
            entry_node_id=entry_node_id,
            query=query,
            frontier_factory=lambda: list(),
            pop_fn=best_first_search_pop_fn,
            push_fn=best_first_search_push_fn,
            tol=tol
        )
        

    def greedy_search(self, entry_node_id: int, query: TData, *, tol: float = 0.0): 
        """Perform greedy search to find the best matching node.
        
        Args:
            entry_node_id: ID of the starting node
            query: Query data to search for
            
        Returns:
            Tuple of (best_node, search_path)
        """
        def greedy_expand_filter_fn(nodes: list[TNode]):
            D = [node.metric(query) for node in nodes]
            idx = np.argmin(D)
            return [nodes[idx]]

        return self._generic_search(
            entry_node_id=entry_node_id,
            query=query,
            frontier_factory=lambda: deque(),
            pop_fn=lambda q: q.pop(),
            push_fn=lambda q, nodes: q.extend(nodes),
            expand_filter_fn=greedy_expand_filter_fn,
            tol=tol
        )
    
    def visualize(self, path: list[Node] | None = None, query: TData | None = None, best: TNode = None) -> plt.Figure:
        """Visualize the graph with optional search path and query point.
        
        Args:
            path: List of nodes representing the search path (optional)
            query: Query point to highlight (optional)
            best: Best node found (optional)
            
        Returns:
            matplotlib Figure object
        """
        # Determine dimensionality from first node
        if not self.nodes:
            fig, ax = plt.subplots(figsize=(10, 8))
            return fig
            
        first_node = next(iter(self.nodes.values()))
        is_3d = len(first_node.data) >= 3
        
        if is_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract path information
        path_ids = [node.id for node in path] if path else []
        best_node_id = best.id if best is not None else None
        start_node_id = path_ids[0] if path_ids else None
        
        # Draw query point with emphasis
        if query is not None:
            if is_3d:
                query_pos = [query[0], query[1], query[2]]
                ax.scatter(*query_pos, marker='*', color='green', s=200, 
                          label='Query Point', edgecolors='black', linewidth=2)
            else:
                query_pos = [query[0], query[1]]
                ax.plot(*query_pos, marker='*', color='green', markersize=15, 
                       label='Query Point', markeredgecolor='black', markeredgewidth=2)
        
        # Collect all unique edges to avoid duplicates
        drawn_edges = set()
        
        # Draw all edges first (background layer)
        for node in self.nodes.values():
            if is_3d:
                node_pos = [node.data[0], node.data[1], node.data[2]]
            else:
                node_pos = [node.data[0], node.data[1]]
                
            for neighbour_id in node.edges.keys():
                # Create a unique edge identifier (smaller ID first to avoid duplicates)
                edge_key = (min(node.id, neighbour_id), max(node.id, neighbour_id))
                
                # Skip if we've already drawn this edge
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)
                
                neighbour = self.nodes[neighbour_id]
                if is_3d:
                    neighbour_pos = [neighbour.data[0], neighbour.data[1], neighbour.data[2]]
                else:
                    neighbour_pos = [neighbour.data[0], neighbour.data[1]]
                
                # Check if this edge is part of the search path
                is_path_edge = (node.id in path_ids and neighbour_id in path_ids and 
                              abs(path_ids.index(node.id) - path_ids.index(neighbour_id)) == 1)
                
                if is_path_edge:
                    # Emphasize path edges
                    if is_3d:
                        ax.plot([node_pos[0], neighbour_pos[0]], [node_pos[1], neighbour_pos[1]], 
                               [node_pos[2], neighbour_pos[2]], 'r-', linewidth=3, alpha=0.8)
                    else:
                        ax.plot([node_pos[0], neighbour_pos[0]], [node_pos[1], neighbour_pos[1]], 
                               'r-', linewidth=3, alpha=0.8, zorder=2)
                else:
                    # Regular edges
                    if is_3d:
                        ax.plot([node_pos[0], neighbour_pos[0]], [node_pos[1], neighbour_pos[1]], 
                               [node_pos[2], neighbour_pos[2]], 'lightgray', linewidth=1, alpha=0.3)
                    else:
                        ax.plot([node_pos[0], neighbour_pos[0]], [node_pos[1], neighbour_pos[1]], 
                               'lightgray', linewidth=1, alpha=0.5, zorder=1)
        
        # Draw path order annotations
        if path_ids:
            for i, node_id in enumerate(path_ids):
                node = self.nodes[node_id]
                if is_3d:
                    node_pos = [node.data[0], node.data[1], node.data[2]]
                    ax.text(node_pos[0], node_pos[1], node_pos[2], f'{i+1}', 
                           fontsize=10, fontweight='bold', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    node_pos = [node.data[0], node.data[1]]
                    ax.annotate(f'{i+1}', (node_pos[0], node_pos[1]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Draw nodes (foreground layer)
        for node in self.nodes.values():
            if is_3d:
                node_pos = [node.data[0], node.data[1], node.data[2]]
            else:
                node_pos = [node.data[0], node.data[1]]
            
            # Determine node appearance based on role
            if node.id == best_node_id:
                # Best/final node
                if is_3d:
                    ax.scatter(*node_pos, marker='*', color='red', s=150, 
                             label='Best Node', edgecolors='black', linewidth=1)
                else:
                    ax.plot(*node_pos, marker='*', color='red', markersize=12, 
                           label='Best Node', markeredgecolor='black', markeredgewidth=1, zorder=4)
            elif node.id == start_node_id:
                # Start node
                if is_3d:
                    ax.scatter(*node_pos, marker='o', color='orange', s=100, 
                             label='Start Node', edgecolors='black', linewidth=1)
                else:
                    ax.plot(*node_pos, marker='o', color='orange', markersize=10, 
                           label='Start Node', markeredgecolor='black', markeredgewidth=1, zorder=4)
            elif node.id in path_ids:
                # Path nodes
                if is_3d:
                    ax.scatter(*node_pos, marker='o', color='red', s=80, 
                             edgecolors='darkred', linewidth=1)
                else:
                    ax.plot(*node_pos, marker='o', color='red', markersize=8, 
                           markeredgecolor='darkred', markeredgewidth=1, zorder=3)
            else:
                # Regular nodes
                if is_3d:
                    ax.scatter(*node_pos, marker='o', color='lightblue', s=50, 
                             edgecolors='blue', linewidth=0.5, alpha=0.7)
                else:
                    ax.plot(*node_pos, marker='o', color='lightblue', markersize=6, 
                           markeredgecolor='blue', markeredgewidth=0.5, alpha=0.7, zorder=2)
        
        # Styling and layout
        if is_3d:
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.set_zlabel('Z Coordinate', fontsize=12)
            ax.set_title('3D Graph Search Visualization', fontsize=14, fontweight='bold')
        else:
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title('Graph Search Visualization', fontsize=14, fontweight='bold')
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', framealpha=0.9)
        
        # Add path statistics if path exists
        if path_ids:
            path_length = len(path_ids)
            if best_node_id and query is not None:
                final_distance = self.nodes[best_node_id].metric(query)
                stats_text = f'Path Length: {path_length}\nFinal Distance: {final_distance:.4f}'
                if is_3d:
                    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                             verticalalignment='top',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
                else:
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def animate_search(self, path: list[Node] | None = None, query: TData | None = None, 
                      best: TNode = None, interval: int = 500) -> FuncAnimation:
        """Create an animated visualization of the search path progression.
        
        Args:
            path: List of nodes representing the search path (optional)
            query: Query point to highlight (optional)
            best: Best node found (optional)
            interval: Delay between frames in milliseconds (default: 500)
            
        Returns:
            matplotlib FuncAnimation object
        """
        if not path:
            return self.visualize(path, query, best)
        
        # Determine dimensionality from first node
        if not self.nodes:
            fig, ax = plt.subplots(figsize=(10, 8))
            return FuncAnimation(fig, lambda frame: [], frames=1, interval=interval)
            
        first_node = next(iter(self.nodes.values()))
        is_3d = len(first_node.data) >= 3
        
        if is_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract path information
        path_ids = [node.id for node in path]
        best_node_id = best.id if best is not None else None
        start_node_id = path_ids[0] if path_ids else None
        
        # Draw query point with emphasis (static)
        if query is not None:
            if is_3d:
                query_pos = [query[0], query[1], query[2]]
                ax.scatter(*query_pos, marker='*', color='green', s=200, 
                          label='Query Point', edgecolors='black', linewidth=2)
            else:
                query_pos = [query[0], query[1]]
                ax.plot(*query_pos, marker='*', color='green', markersize=15, 
                       label='Query Point', markeredgecolor='black', markeredgewidth=2)
        
        # Collect all unique edges and draw static background edges
        drawn_edges = set()
        for node in self.nodes.values():
            for neighbour_id in node.edges.keys():
                edge_key = (min(node.id, neighbour_id), max(node.id, neighbour_id))
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)
                
                neighbour = self.nodes[neighbour_id]
                if is_3d:
                    node_pos = [node.data[0], node.data[1], node.data[2]]
                    neighbour_pos = [neighbour.data[0], neighbour.data[1], neighbour.data[2]]
                    ax.plot([node_pos[0], neighbour_pos[0]], [node_pos[1], neighbour_pos[1]], 
                           [node_pos[2], neighbour_pos[2]], 'lightgray', linewidth=1, alpha=0.2)
                else:
                    node_pos = [node.data[0], node.data[1]]
                    neighbour_pos = [neighbour.data[0], neighbour.data[1]]
                    ax.plot([node_pos[0], neighbour_pos[0]], [node_pos[1], neighbour_pos[1]], 
                           'lightgray', linewidth=1, alpha=0.3, zorder=1)
        
        # Draw all nodes as background (static)
        for node in self.nodes.values():
            if is_3d:
                node_pos = [node.data[0], node.data[1], node.data[2]]
            else:
                node_pos = [node.data[0], node.data[1]]
                
            if node.id == start_node_id:
                if is_3d:
                    ax.scatter(*node_pos, marker='o', color='orange', s=100, 
                             label='Start Node', edgecolors='black', linewidth=1)
                else:
                    ax.plot(*node_pos, marker='o', color='orange', markersize=10, 
                           label='Start Node', markeredgecolor='black', markeredgewidth=1, zorder=2)
            elif node.id == best_node_id:
                if is_3d:
                    ax.scatter(*node_pos, marker='*', color='red', s=150, 
                             label='Best Node', edgecolors='black', linewidth=1)
                else:
                    ax.plot(*node_pos, marker='*', color='red', markersize=12, 
                           label='Best Node', markeredgecolor='black', markeredgewidth=1, zorder=2)
            else:
                if is_3d:
                    ax.scatter(*node_pos, marker='o', color='lightblue', s=50, 
                             edgecolors='blue', linewidth=0.5, alpha=0.7)
                else:
                    ax.plot(*node_pos, marker='o', color='lightblue', markersize=6, 
                           markeredgecolor='blue', markeredgewidth=0.5, alpha=0.7, zorder=2)
        
        # Initialize animated elements
        path_lines = []
        visited_nodes = []
        step_annotations = []
        
        # Styling and layout
        if is_3d:
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.set_zlabel('Z Coordinate', fontsize=12)
            ax.set_title('Animated 3D Graph Search', fontsize=14, fontweight='bold')
        else:
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title('Animated Graph Search', fontsize=14, fontweight='bold')
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', framealpha=0.9)
        
        # Animation statistics text
        if is_3d:
            stats_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        else:
            stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        def animate(frame):
            # Clear previous animated elements
            for line in path_lines:
                line.remove()
            for node in visited_nodes:
                node.remove()
            for annotation in step_annotations:
                annotation.remove()
            path_lines.clear()
            visited_nodes.clear()
            step_annotations.clear()
            
            # Current step in the animation
            current_step = min(frame + 1, len(path_ids))
            current_path = path_ids[:current_step]
            
            # Draw path edges up to current step
            for i in range(len(current_path) - 1):
                node_id = current_path[i]
                next_node_id = current_path[i + 1]
                
                node = self.nodes[node_id]
                next_node = self.nodes[next_node_id]
                
                if is_3d:
                    node_pos = [node.data[0], node.data[1], node.data[2]]
                    next_pos = [next_node.data[0], next_node.data[1], next_node.data[2]]
                    line, = ax.plot([node_pos[0], next_pos[0]], [node_pos[1], next_pos[1]], 
                                   [node_pos[2], next_pos[2]], 'r-', linewidth=3, alpha=0.8)
                else:
                    node_pos = [node.data[0], node.data[1]]
                    next_pos = [next_node.data[0], next_node.data[1]]
                    line, = ax.plot([node_pos[0], next_pos[0]], [node_pos[1], next_pos[1]], 
                                   'r-', linewidth=3, alpha=0.8, zorder=3)
                path_lines.append(line)
            
            # Highlight visited nodes in the current path
            for i, node_id in enumerate(current_path):
                node = self.nodes[node_id]
                if is_3d:
                    node_pos = [node.data[0], node.data[1], node.data[2]]
                else:
                    node_pos = [node.data[0], node.data[1]]
                
                # Skip start and best nodes (already drawn)
                if node_id == start_node_id or node_id == best_node_id:
                    continue
                
                if is_3d:
                    visited_node = ax.scatter(*node_pos, marker='o', color='red', s=80, 
                                            edgecolors='darkred', linewidth=1)
                    visited_nodes.append(visited_node)
                    
                    # Add step number annotation for 3D
                    annotation = ax.text(node_pos[0], node_pos[1], node_pos[2], f'{i+1}', 
                                       fontsize=10, fontweight='bold', color='darkred',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    step_annotations.append(annotation)
                else:
                    visited_node, = ax.plot(*node_pos, marker='o', color='red', markersize=8, 
                                           markeredgecolor='darkred', markeredgewidth=1, zorder=4)
                    visited_nodes.append(visited_node)
                    
                    # Add step number annotation for 2D
                    annotation = ax.annotate(f'{i+1}', (node_pos[0], node_pos[1]), 
                                           xytext=(5, 5), textcoords='offset points',
                                           fontsize=10, fontweight='bold', color='darkred',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                                           zorder=5)
                    step_annotations.append(annotation)
            
            # Update statistics
            if current_step > 0 and query is not None and best is not None:
                current_node = self.nodes[current_path[-1]]
                current_distance = current_node.metric(query)
                best_distance = best.metric(query) if best else float('inf')
                
                stats_text.set_text(f'Step: {current_step}/{len(path_ids)}\n'
                                   f'Current Distance: {current_distance:.4f}\n'
                                   f'Best Distance: {best_distance:.4f}')
            else:
                stats_text.set_text(f'Step: {current_step}/{len(path_ids)}')
            
            return path_lines + visited_nodes + step_annotations + [stats_text]
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(path_ids), interval=interval, 
                           blit=False, repeat=True)
        
        return anim




        
class NodeVectorL2(Node[np.typing.ArrayLike]): 
    """Vector node for storing array-like data with L2 distance metric.
    
    Args:
        id: Unique identifier for the node
        data: Array-like data (will be converted to float array with shape (1, D))
        
    Raises:
        TypeError: If data cannot be converted to float dtype
        ValueError: If data doesn't have shape (1, D)
    """

    def __init__(self, id: int, data: np.typing.ArrayLike) -> None:
        data = np.asarray(data, dtype=float)
        if not np.issubdtype(data.dtype, np.floating):
            raise TypeError(f"Expected float dtype, got {data.dtype}")
        if data.ndim != 1:
            raise ValueError(f"Expected data.ndim 1, got {data.shape}")
        super().__init__(id, data)
    
    def metric(self, data: np.typing.ArrayLike) -> np.float64:
        """Calculate L2 (Euclidean) distance between this node's data and query data.
        
        Args:
            data: Query data to compare against
            
        Returns:
            L2 distance as a float64
        """
        data = np.asarray(data)
        return np.linalg.norm(data - self.data) 
    

class NodeVectorIP(Node):
    
    def __init__(self, id: int, data: np.typing.ArrayLike):
        super().__init__(id, data)

    def metric(self, data: np.typing.ArrayLike) -> np.float64:
        """Calculate negative IP between this node's data and query data.
        
        Args:
            data: Query data to compare against
            
        Returns:
            Negative dot product as a float64
        """
        data = np.asarray(data)
        return 1 - np.dot(data / np.linalg.norm(data), self.data / np.linalg.norm(self.data))


