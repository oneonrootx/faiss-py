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

Let's start with an analogous problem. Navigating the UK, imagine each address is a node in a graph, and addresses that are sufficently close share an edge. Now let's say we're trying to get from `10 Downing Street, London` (`starting_address`) to `1 Princess St, Edinburgh` (`target_address`). Consider the following strategy for achieving this:


```python
current_address = starting_address  # Start at some initial address (entry point)
while True:
    # Compute the distance from the current address to the target
    distance_to_target = distance(current_address, target_address)
    
    # If we're close enough to the target, stop searching
    if distance_to_target < tolerance:
        break

    # Look at all neighboring addresses of the current address
    neighbor_addresses = get_neighbors(current_address)
    
    # For each neighbor, compute how far it is from the target
    distances = []
    for neighbor in neighbor_addresses:
        d = distance(neighbor, target_address)
        distances.append(d)
    
    # Find the neighbor that is closest to the target
    index_of_best = index_of_smallest_value(distances)
    best_neighbor = neighbor_addresses[index_of_best]
    
    # Move to that neighbor and repeat
    current_address = best_neighbor
```

This is a very basic greedy graph traversal algorithm. What's wrong with it? I like to always start a problem like this by thinking about how I would attack this problem. Personally, I would think `How do I get to Edinburgh` and then once in Edinburgh, I'd ask myself how do I get to `New Town` and then how do I get to `Princess Street` and then only once I'm on  `Princess Street` would I look at the street numbers and find where `1 Princess Street` is. I certainly would't think, which of the houses on `Downing Street` are closest to `1 Princess Street`? Better go there first!

### So, what can we do? 

Now, imagine instead of relying on just one giant map with every address in the UK, we have **a series of maps at different resolutions**:

* The **top-level map** only shows major cities — London, Manchester, Edinburgh.
* The **mid-level map** shows neighborhoods within each city — New Town in Edinburgh, Soho in London.
* The **lowest-level map** shows individual streets and addresses — like 1 Princess Street.

But we don’t put *every* address on *every* map. That would be redundant and cluttered. Instead, we **randomly promote** a few addresses to be visible on the higher-resolution maps. For example, maybe only 1 in 10,000 addresses appears on the city-level map. These promoted addresses act as **landmarks or waypoints** that help us traverse the space quickly at a coarse level before zooming in.

### How does this help?

Let’s say we’re starting in London and trying to find our way to 1 Princess Street in Edinburgh:

1. **At the top layer (coarse map):**
   We don’t see individual streets, just some major waypoints like “Edinburgh” and “Manchester.” We pick a waypoint that brings us roughly in the right direction — say, “Edinburgh” — and descend.

2. **At the middle layer:**
   Now that we’re near Edinburgh, we see neighborhoods like “Old Town” and “New Town.” We move toward “New Town” and descend again.

3. **At the bottom layer (fine-grained map):**
   Finally, we search at street-level detail to locate “Princess Street” and eventually “1 Princess Street.”

This is the **core idea of the Hierarchical Navigable Small World graph (HNSW)**: build a multi-level graph where each level provides shortcuts over longer distances. At each layer, we use local greedy search to move closer to the target. Once we reach the most promising node, we descend a level and continue.

Each level in the graph is a **random subset** of the level below. When a new vector is added, it’s randomly assigned a highest layer (e.g., level 2 or level 0), and then linked into the graph at each layer from the top down. This randomness is what gives HNSW its “small-world” properties — long-range links that dramatically reduce the number of steps needed to reach a target.

Whenever you plan a journey using a series of maps—starting with a high-level map to get you from city to city, then zooming in to neighborhoods, and finally down to street-level—you’re essentially running a manual version of Hierarchical Navigable Small World (HNSW) search.

HNSW structures data into multiple layers, where each layer represents a different "resolution" of the space. Higher layers contain fewer, more broadly connected nodes (like major cities), and lower layers contain finer-grained nodes (like individual addresses).

When searching, HNSW starts at the top layer, moving greedily toward the goal. Once it can’t improve further at that level, it drops to a more detailed one and continues. This layered approach helps avoid getting stuck in local minima and lets you navigate vast spaces quickly—just like how you'd navigate from London to 1 Princess Street by switching to more detailed maps as you go.