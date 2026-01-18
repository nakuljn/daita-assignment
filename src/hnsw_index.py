import numpy as np
import heapq
from typing import List, Tuple, Set, Dict


class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) index for approximate nearest neighbor search.
    Uses multi-layer graph structure with greedy search for efficient high-dimensional similarity queries.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2
        self.ef_construction = ef_construction
        self.vectors: np.ndarray = None
        self.graphs: List[Dict[int, List[int]]] = []
        self.entry_point: int = -1
        self.max_level: int = -1
        self.node_levels: List[int] = []

    def build(self, vectors: np.ndarray, verbose: bool = True) -> None:
        self.vectors = vectors
        n = len(vectors)
        
        if verbose:
            print(f"[BUILD] Inserting {n} vectors...")
        
        for i in range(n):
            self._insert_node(i)
            if verbose and (i + 1) % 2000 == 0:
                print(f"[BUILD] {i + 1}/{n} done")
        
        if verbose:
            print(f"[BUILD] Complete: {n} vectors, {len(self.graphs)} layers")

    def _insert_node(self, node_id: int) -> None:
        vector = self.vectors[node_id]
        level = min(int(-np.log(np.random.random() + 1e-10) / np.log(self.M)), 10)
        self.node_levels.append(level)
        
        while len(self.graphs) <= level:
            self.graphs.append({})
        
        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_level = level
            for l in range(level + 1):
                self.graphs[l][node_id] = []
            return
        
        curr = self.entry_point
        for l in range(self.max_level, level, -1):
            curr = self._greedy_closest(vector, curr, l)
        
        for l in range(min(level, self.max_level), -1, -1):
            neighbors = self._search_layer(vector, curr, self.ef_construction, l)
            M_max = self.M_max0 if l == 0 else self.M
            self._connect(node_id, neighbors[:M_max], l)
            if neighbors:
                curr = neighbors[0][1]
        
        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def _greedy_closest(self, query: np.ndarray, start: int, layer: int) -> int:
        curr = start
        curr_dist = self._dist(query, curr)
        
        improved = True
        while improved:
            improved = False
            for neighbor in self.graphs[layer].get(curr, []):
                d = self._dist(query, neighbor)
                if d < curr_dist:
                    curr, curr_dist = neighbor, d
                    improved = True
                    break
        return curr

    def _search_layer(self, query: np.ndarray, entry: int, ef: int, layer: int) -> List[Tuple[float, int]]:
        visited = {entry}
        candidates = [(self._dist(query, entry), entry)]
        results = [(-candidates[0][0], entry)]
        heapq.heapify(candidates)
        heapq.heapify(results)
        
        while candidates:
            c_dist, c_node = heapq.heappop(candidates)
            if c_dist > -results[0][0]:
                break
            
            for neighbor in self.graphs[layer].get(c_node, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                
                n_dist = self._dist(query, neighbor)
                if n_dist < -results[0][0] or len(results) < ef:
                    heapq.heappush(candidates, (n_dist, neighbor))
                    heapq.heappush(results, (-n_dist, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        
        return sorted([(-d, n) for d, n in results])

    def _connect(self, node_id: int, neighbors: List[Tuple[float, int]], layer: int) -> None:
        M_max = self.M_max0 if layer == 0 else self.M
        neighbor_ids = [n for _, n in neighbors]
        self.graphs[layer][node_id] = neighbor_ids
        
        for nid in neighbor_ids:
            if nid not in self.graphs[layer]:
                self.graphs[layer][nid] = []
            conns = self.graphs[layer][nid]
            
            if node_id not in conns:
                if len(conns) < M_max:
                    conns.append(node_id)
                else:
                    nid_vec = self.vectors[nid]
                    all_n = [(1.0 - np.dot(nid_vec, self.vectors[x]), x) for x in conns + [node_id]]
                    all_n.sort()
                    self.graphs[layer][nid] = [x for _, x in all_n[:M_max]]

    def _dist(self, query: np.ndarray, node_id: int) -> float:
        return 1.0 - np.dot(query, self.vectors[node_id])

    def query(self, query: np.ndarray, k: int, ef_search: int = 100) -> List[Tuple[int, float]]:
        if self.entry_point == -1:
            return []
        
        curr = self.entry_point
        for l in range(self.max_level, 0, -1):
            curr = self._greedy_closest(query, curr, l)
        
        candidates = self._search_layer(query, curr, ef_search, 0)
        return [(n, 1.0 - d) for d, n in candidates[:k]]

    def get_stats(self) -> dict:
        edges = sum(len(n) for g in self.graphs for n in g.values())
        return {
            "vectors": len(self.vectors) if self.vectors is not None else 0,
            "layers": len(self.graphs),
            "max_level": self.max_level,
            "edges": edges,
            "M": self.M
        }
