"""Instrumented HNSW best-first search with pluggable termination policies.

One search loop (mirroring hnswlib's searchBaseLayerST / the Ada-ef paper's
Algorithm 2) serves three strategies:

  - fixed ef        : standard HNSW (baseline)
  - PIP             : patience early termination (Teofili & Lin, ECIR 2025),
                      semantics copied from the authors' C++ reference in
                      hnsw-ada-ef/hnswlib/hnswalg.h (searchBaseLayerSTWith-
                      PatienceInProximity): after each expanded node, phi =
                      |top-k ∩ previous top-k| / k; a counter increments while
                      phi >= gamma and resets otherwise; stop at Delta.
  - Ada-ef          : adaptive ef (Zhang & Miller, SIGMOD 2026): start with
                      ef = inf, record the first l distances, score the query
                      against the estimated distance distribution, look up ef
                      in the offline table, truncate, continue normally.

Cost metrics returned per query: distance computations (n_dist), expanded
nodes (n_hops). Distances are computed with numpy over each node's neighbor
list at once.
"""

import heapq

import numpy as np


def true_dists(vectors, ids, q, metric, sq_norms=None, q_sq=None):
    """True distances from q to vectors[ids] (cosine distance or L2 squared)."""
    if metric == "cosine":
        return 1.0 - vectors[ids] @ q
    return sq_norms[ids] - 2.0 * (vectors[ids] @ q) + q_sq


def descend(graph, vectors, q, metric, sq_norms=None, q_sq=None):
    """Greedy descent through upper levels; returns (entry node, n_dist)."""
    cur = graph.entry_point
    cur_dist = true_dists(vectors, np.array([cur]), q, metric, sq_norms, q_sq)[0]
    n_dist = 1
    for lvl in range(graph.max_level, 0, -1):
        table = graph.upper[lvl]
        changed = True
        while changed:
            changed = False
            nbrs = table.get(cur)
            if nbrs is None or len(nbrs) == 0:
                break
            d = true_dists(vectors, nbrs, q, metric, sq_norms, q_sq)
            n_dist += len(nbrs)
            j = int(np.argmin(d))
            if d[j] < cur_dist:
                cur_dist = float(d[j])
                cur = int(nbrs[j])
                changed = True
    return cur, cur_dist, n_dist


def search(
    graph,
    vectors,
    q,
    k,
    metric,
    ef=None,
    pip=None,  # (gamma, delta) with gamma in [0,1]
    ada=None,  # AdaEf object from ada_ef.py
    sq_norms=None,
    visited=None,
    visit_tag=None,
    trace=None,  # optional dict populated with per-hop phi / |candidates| lists
    entry=None,  # force this node as the level-0 entry point (skips descend)
):
    """Returns (ids ascending by distance, stats dict)."""
    q = np.ascontiguousarray(q, dtype=np.float32)
    q_sq = float(q @ q) if metric == "l2" else None

    if entry is not None:
        ep = int(entry)
        ep_dist = float(true_dists(vectors, np.array([ep]), q, metric, sq_norms, q_sq)[0])
        n_dist = 1
    else:
        ep, ep_dist, n_dist = descend(graph, vectors, q, metric, sq_norms, q_sq)

    if visited is None:
        visited = np.zeros(len(vectors), dtype=np.int32)
        visit_tag = 1
    visited[ep] = visit_tag

    adj0 = graph.adj0
    # candidates: min-heap of (dist, id); results: max-heap as (-dist, id)
    candidates = [(ep_dist, ep)]
    results = [(-ep_dist, ep)]
    lower_bound = ep_dist

    collecting = ada is not None
    if collecting:
        eff_ef = float("inf")
        collected = [ep_dist]
        l_limit = ada.l_limit(graph)
    else:
        eff_ef = ef
    est_ef = None
    score = None

    # PIP state. delta may be a float in (0, 1]: then it is a *fraction of the
    # per-query estimated ef* (hybrid mode, requires ada) — patience scales with
    # how hard Ada-ef thinks the query is.
    #
    # phi = |top-k ∩ previous top-k| / k is tracked incrementally: a size-k
    # max-heap of everything ever considered equals the top-k of `results`
    # (same (-d, id) tuple order as nlargest), and since the top-k set has
    # constant size k once full, |prev ∩ cur| = k − (#members that entered
    # since the last check). O(log k) per insert instead of O(ef·log k) per
    # hop, which is what makes k=1000 tractable.
    if pip is not None:
        gamma, delta = pip
        stable_counter = 0
        pip_started = False
        topk_heap = [(-ep_dist, ep)]  # min-heap of (-d, id): root = worst of the top-k
        pip_period = 0
        entered = {ep: 0}  # id -> pip_period when it entered the top-k
        new_this_period = 1

    n_hops = 0
    while candidates:
        cand_dist, cand = candidates[0]
        if cand_dist > lower_bound and len(results) >= min(eff_ef, 1 << 30):
            break
        heapq.heappop(candidates)
        n_hops += 1

        nbrs = adj0[cand]
        nbrs = nbrs[nbrs >= 0]
        unvisited = nbrs[visited[nbrs] != visit_tag]
        if len(unvisited):
            visited[unvisited] = visit_tag
            dists = true_dists(vectors, unvisited, q, metric, sq_norms, q_sq)
            n_dist += len(unvisited)

            for d, node in zip(dists.tolist(), unvisited.tolist()):
                if collecting:
                    consider = True
                    if len(collected) >= l_limit:
                        collecting = False
                        score = ada.compute_score(q, np.array(collected, dtype=np.float32))
                        est_ef = ada.estimate_ef(score)
                        eff_ef = max(est_ef, k)
                        if trace is not None:
                            trace["collected"] = list(collected)
                            trace["score"] = score
                            trace["est_ef"] = est_ef
                        if pip is not None and isinstance(delta, float) and delta <= 1.0:
                            delta = max(30, int(delta * eff_ef))
                        while len(results) > eff_ef:
                            heapq.heappop(results)
                        lower_bound = -results[0][0]
                else:
                    consider = len(results) < eff_ef or d < lower_bound
                if consider:
                    heapq.heappush(candidates, (d, node))
                    heapq.heappush(results, (-d, node))
                    if collecting:
                        collected.append(d)
                    if len(results) > eff_ef:
                        heapq.heappop(results)
                    lower_bound = -results[0][0]
                    if pip is not None:
                        if len(topk_heap) < k:
                            heapq.heappush(topk_heap, (-d, node))
                            entered[node] = pip_period
                            new_this_period += 1
                        elif (-d, node) > topk_heap[0]:
                            _, ev = heapq.heappushpop(topk_heap, (-d, node))
                            if entered.pop(ev, -1) == pip_period:
                                new_this_period -= 1
                            entered[node] = pip_period
                            new_this_period += 1

        # ---- PIP saturation check (once per expanded node) ----
        # With Ada-ef active, patience only starts after the ef estimate:
        # the collection phase must finish so the hybrid gets its budget first.
        if pip is not None and not collecting and len(results) >= k:
            if pip_started:
                phi = (k - new_this_period) / k
            else:  # first check: no previous top-k snapshot yet
                phi = 0.0
                pip_started = True
            pip_period += 1
            new_this_period = 0
            if trace is not None:
                trace.setdefault("phi", []).append(phi)
                trace.setdefault("n_results", []).append(len(results))
                trace.setdefault("counter", []).append(stable_counter)
            if phi >= gamma:
                stable_counter += 1
            else:
                stable_counter = 0
            if stable_counter >= delta:
                break
        elif trace is not None:
            trace.setdefault("n_results", []).append(len(results))

    top = heapq.nlargest(k, results)  # largest -dist = smallest dist
    top.sort(key=lambda t: -t[0])
    ids = np.array([node for _, node in top], dtype=np.int64)
    stats = {"n_dist": n_dist, "n_hops": n_hops}
    if ada is not None:
        stats["score"] = score
        stats["est_ef"] = est_ef
    return ids, stats


class SearchContext:
    """Reusable per-dataset state: squared norms + visited tag array."""

    def __init__(self, ds, graph):
        self.ds = ds
        self.graph = graph
        self.sq_norms = (
            np.einsum("ij,ij->i", ds.train, ds.train).astype(np.float32)
            if ds.metric == "l2"
            else None
        )
        self.visited = np.zeros(len(ds.train), dtype=np.int32)
        self.tag = 0

    def search(self, q, k, **kw):
        self.tag += 1
        return search(
            self.graph,
            self.ds.train,
            q,
            k,
            self.ds.metric,
            sq_norms=self.sq_norms,
            visited=self.visited,
            visit_tag=self.tag,
            **kw,
        )


def recall_at_k(found_ids, gt_row, k):
    return len(set(found_ids[:k]) & set(gt_row[:k].tolist())) / k
