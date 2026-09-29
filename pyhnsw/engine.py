"""Resumable HNSW best-first search in Python, on the graph extracted from
the Faiss index.

Restored from the pre-cleanup `search.py` (the loop mirrors hnswlib's
searchBaseLayerST; it was validated to +-0.0002 recall against Faiss on
SIFT1M), with the PIP / Ada-ef policies removed and one addition: the
search is an object whose state (visited set, candidate heap, result heap,
counters) survives between calls, so a search run at ef0 can be inspected
and then CONTINUED at a larger ef instead of being restarted.

    s = Search(ctx, q, k)
    s.run(100)             # the peek: beam search until it converges at ef=100
    ids, dists = s.top(100)  # what it has found so far (no ground truth needed)
    s.run(1200)            # keep going with a longer shortlist; nothing is recomputed

What a resumable engine has to keep: `spill`, the nodes whose distance was
computed but which did not make the ef-sized result heap at the time.
When ef grows, the best of them are re-admitted (no new distance
computations). With spill=False they are lost, and the continued search
can only follow candidates that are still in the heap.

Cost: n_dist = distance computations, n_hops = expanded nodes, cumulative
over every run() call on the same object. Faiss uses max(ef, k) as the
beam; so does this loop.
"""

import heapq

import numpy as np


class Engine:
    """Per-dataset state shared by all searches: vectors, metric, norms, and
    the visited-tag array (one int per node, bumped per search)."""

    def __init__(self, ds, graph):
        self.graph = graph
        self.X = ds.train
        self.metric = ds.metric
        self.sq = np.einsum("ij,ij->i", self.X, self.X).astype(np.float32) if ds.metric == "l2" else None
        self.visited = np.zeros(len(self.X), dtype=np.int32)
        self.tag = 0

    def dists(self, ids, q, q_sq):
        if self.metric == "cosine":
            return 1.0 - self.X[ids] @ q
        return self.sq[ids] - 2.0 * (self.X[ids] @ q) + q_sq

    def descend(self, q, q_sq):
        """Greedy descent through the upper layers (as Faiss does before the
        level-0 beam). Returns (level-0 entry point, its distance, n_dist)."""
        g = self.graph
        cur = g.entry_point
        cur_dist = float(self.dists(np.array([cur]), q, q_sq)[0])
        n_dist = 1
        for lvl in range(g.max_level, 0, -1):
            table = g.upper[lvl]
            changed = True
            while changed:
                changed = False
                nbrs = table.get(cur)
                if nbrs is None or len(nbrs) == 0:
                    break
                d = self.dists(nbrs, q, q_sq)
                n_dist += len(nbrs)
                j = int(np.argmin(d))
                if d[j] < cur_dist:
                    cur_dist, cur, changed = float(d[j]), int(nbrs[j]), True
        return cur, cur_dist, n_dist


class Search:
    """One query's search, resumable across increasing ef."""

    def __init__(self, engine, q, k, spill=True):
        self.e = engine
        self.k = k
        self.keep_spill = spill
        self.q = np.ascontiguousarray(q, dtype=np.float32)
        self.q_sq = float(self.q @ self.q) if engine.metric == "l2" else 0.0
        engine.tag += 1
        self.tag = engine.tag
        ep, ep_dist, self.n_dist = engine.descend(self.q, self.q_sq)
        self.n_hops = 0
        self.ep, self.ep_dist = ep, ep_dist
        engine.visited[ep] = self.tag
        self.candidates = [(ep_dist, ep)]  # min-heap: nodes not yet expanded
        self.results = [(-ep_dist, ep)]  # max-heap (as -dist): the best `ef` nodes seen
        self.spill = []  # min-heap: seen, not in results, not expanded
        self.ef = 0

    def run(self, ef):
        """Continue the beam search until it converges at this ef (>= the
        previous ef). Returns the cumulative (n_dist, n_hops)."""
        ef = max(int(ef), self.k)
        assert ef >= self.ef, "ef can only grow on a resumed search"
        self.ef = ef
        e, adj0, visited, tag = self.e, self.e.graph.adj0, self.e.visited, self.tag
        candidates, results, spill = self.candidates, self.results, self.spill
        # a longer shortlist: re-admit the best spilled nodes (their distances are known)
        while spill and len(results) < ef:
            d, node = heapq.heappop(spill)
            heapq.heappush(results, (-d, node))
            heapq.heappush(candidates, (d, node))
        lower_bound = -results[0][0]

        while candidates:
            cand_dist, cand = candidates[0]
            if cand_dist > lower_bound and len(results) >= ef:
                break
            heapq.heappop(candidates)
            self.n_hops += 1
            nbrs = adj0[cand]
            nbrs = nbrs[nbrs >= 0]
            unvisited = nbrs[visited[nbrs] != tag]
            if not len(unvisited):
                continue
            visited[unvisited] = tag
            dists = e.dists(unvisited, self.q, self.q_sq)
            self.n_dist += len(unvisited)
            for d, node in zip(dists.tolist(), unvisited.tolist()):
                if len(results) < ef or d < lower_bound:
                    heapq.heappush(candidates, (d, node))
                    heapq.heappush(results, (-d, node))
                    if len(results) > ef:
                        nd, evicted = heapq.heappop(results)
                        if self.keep_spill:
                            heapq.heappush(spill, (-nd, evicted))
                    lower_bound = -results[0][0]
                elif self.keep_spill:
                    heapq.heappush(spill, (d, node))
        return self.n_dist, self.n_hops

    def top(self, n=None):
        """The best n nodes found so far: (ids, distances), ascending."""
        n = self.k if n is None else n
        best = heapq.nlargest(n, self.results)
        best.sort(key=lambda t: -t[0])
        return np.array([node for _, node in best], dtype=np.int64), np.array([-d for d, _ in best], dtype=np.float32)
