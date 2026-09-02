"""E11 — avg-dist, exactly as on the whiteboard (2026-09-01).

E10 (`hardness3.py`) implemented the professor's avg-dist metric with a
4-hop BFS cap for speed. This experiment removes every shortcut and
computes the method literally:

    given q
      pick two nodes u, v from GT (the exact top-100 of q)
      d(u, v) = min( shortest-path-dist(u, v), shortest-path-dist(v, u) )
      repeat for all C(100, 2) = 4950 pairs   (or --sample I random pairs)
      avg-dist(q) = mean of d over the pairs
    repeat for q_1 ... q_1000

Shortest-path-dist is the hop count over the level-0 HNSW graph's
out-edges (the graph the beam search actually walks). Each BFS runs until
every other GT node has been reached or the reachable set is exhausted —
no cap. Pairs that are unreachable in BOTH directions have no finite
distance; they are excluded from the average and counted separately
(`n_unreachable`), so nothing invented enters avg-dist.

Per query we also record the median and max pair distance, and the pooled
histogram of exact pair distances over all queries.

Algorithm (exact, but avoids expanding a million-node 4th ring):
  meet-in-the-middle  forward BFS 3 rings from u, backward BFS 2 rings
                      (in-edges) from every target v.  Any u->v path of
                      length d <= 5 has a node w with fwd(u,w) + bwd(w,v) = d,
                      and every such w gives a real path, so
                      d(u,v) = min_w fwd + bwd exactly whenever d <= 5.
  second pass         pairs unresolved in both directions get a 3-ring
                      backward layer as well (resolves d <= 6 exactly).
  fallback            pairs still unresolved (d > 6) get an uncapped BFS
                      from each endpoint until the other is reached or the
                      reachable set is exhausted.
  Verified identical to a plain uncapped BFS on sample queries.

Runtime: ~2-5 s per query on GloVe-100 (1000 queries: ~1 h).
Usage: .venv/bin/python -m pyhnsw.hardness4 [--sample I] [--n-queries N]
"""

import argparse
import json
import time

import numpy as np

from .experiments import RESULTS_DIR, get_ctx
from .graph import DATA_DIR
from .hardness import LEVELS, spearman

K_GT = 100
INF = np.iinfo(np.int16).max


class ExactBFS:
    """Uncapped BFS over a -1-padded adjacency matrix, early-stopping when
    all targets are found."""

    def __init__(self, adj0):
        self.adj0 = adj0
        n = adj0.shape[0]
        self.visited = np.zeros(n, dtype=bool)
        self.tpos = np.full(n, -1, dtype=np.int32)  # node -> index in target set

    def set_targets(self, targets):
        self.targets = np.asarray(targets, dtype=np.int64)
        self.tpos[self.targets] = np.arange(len(self.targets), dtype=np.int32)

    def clear_targets(self):
        self.tpos[self.targets] = -1

    def hops_from(self, src_idx, needed=None):
        """Exact hop distance from targets[src_idx] to every target
        (INF if unreachable). Stops early once all `needed` targets (bool
        mask; default: all) have been reached."""
        adj0, visited, tpos = self.adj0, self.visited, self.tpos
        n_t = len(self.targets)
        dist = np.full(n_t, INF, dtype=np.int16)
        dist[src_idx] = 0
        if needed is None:
            needed = np.ones(n_t, dtype=bool)
        needed = needed.copy()
        needed[src_idx] = False
        remaining = int(needed.sum())
        src = self.targets[src_idx]
        frontier = np.array([src])
        visited[src] = True
        touched = [frontier]
        h = 0
        while frontier.size:
            h += 1
            nb = adj0[frontier].ravel()
            nb = nb[nb >= 0]
            nb = nb[~visited[nb]]
            if nb.size == 0:
                break  # reachable set exhausted
            nb = np.unique(nb)
            visited[nb] = True
            touched.append(nb)
            hits = tpos[nb]
            hits = hits[hits >= 0]
            if hits.size:
                dist[hits] = h  # first time seen == shortest path
                remaining -= int(needed[hits].sum())
                if remaining == 0:
                    break
            frontier = nb
        for t in touched:
            visited[t] = False
        return dist

    def pairwise(self, targets):
        """Symmetrised exact hop-distance matrix among targets (plain
        uncapped BFS from every target; slow reference implementation)."""
        self.set_targets(targets)
        n_t = len(targets)
        D = np.empty((n_t, n_t), dtype=np.int16)
        for i in range(n_t):
            D[i] = self.hops_from(i)
        self.clear_targets()
        return np.minimum(D, D.T)


def build_reverse_csr(adj0, cache_path):
    """In-edge lists as CSR (indptr, sources), cached on disk."""
    if cache_path.exists():
        z = np.load(cache_path)
        return z["indptr"], z["rsrc"]
    n, m0 = adj0.shape
    src = np.repeat(np.arange(n, dtype=np.int32), m0)
    dst = adj0.ravel()
    ok = dst >= 0
    src, dst = src[ok], dst[ok]
    order = np.argsort(dst, kind="stable")
    rsrc = src[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(dst, minlength=n))
    np.savez(cache_path, indptr=indptr, rsrc=rsrc)
    return indptr, rsrc


FWD_DEPTH, BWD_DEPTH = 3, 2
UNSEEN = 100  # forward-distance sentinel; UNSEEN + BWD_DEPTH > FWD+BWD


class MeetInMiddle:
    """Exact pairwise hop distances among a target set, resolving every
    pair with d <= FWD_DEPTH + BWD_DEPTH via forward/backward BFS layers
    and delegating the rest to an uncapped BFS (ExactBFS)."""

    def __init__(self, adj0, indptr, rsrc):
        self.adj0, self.indptr, self.rsrc = adj0, indptr, rsrc
        self.fd = np.full(adj0.shape[0], UNSEEN, dtype=np.int16)
        self.fallback = ExactBFS(adj0)

    def _in_neighbors(self, nodes):
        ip, rs = self.indptr, self.rsrc
        return np.concatenate([rs[ip[v]:ip[v + 1]] for v in nodes]) if len(nodes) else nodes

    def _backward_layers(self, targets):
        """Concatenated backward layers of all targets: nodes B, their layer
        depth Bj, and segment boundaries (one segment per target)."""
        nodes, depth, bounds = [], [], [0]
        for v in targets:
            layer = np.array([v], dtype=np.int32)
            for j in range(BWD_DEPTH + 1):
                nodes.append(layer)
                depth.append(np.full(layer.size, j, dtype=np.int16))
                if j < BWD_DEPTH:
                    layer = self._in_neighbors(layer)
            bounds.append(bounds[-1] + sum(x.size for x in nodes[-(BWD_DEPTH + 1):]))
        return np.concatenate(nodes), np.concatenate(depth), np.array(bounds[:-1])

    def _forward_layers(self, src):
        """Mark fd[w] = exact forward hop distance for all w within FWD_DEPTH."""
        adj0, fd = self.adj0, self.fd
        frontier = np.array([src])
        fd[src] = 0
        touched = [frontier]
        for h in range(1, FWD_DEPTH + 1):
            nb = adj0[frontier].ravel()
            nb = nb[nb >= 0]
            nb = nb[fd[nb] == UNSEEN]
            if nb.size == 0:
                break
            nb = np.unique(nb)
            fd[nb] = h
            touched.append(nb)
            frontier = nb
        return touched

    def pairwise(self, targets):
        targets = np.asarray(targets, dtype=np.int64)
        n_t = len(targets)
        B, Bj, bounds = self._backward_layers(targets)
        D = np.full((n_t, n_t), INF, dtype=np.int16)
        limit = FWD_DEPTH + BWD_DEPTH
        for i, u in enumerate(targets):
            touched = self._forward_layers(u)
            row = np.minimum.reduceat(self.fd[B] + Bj, bounds)
            row[row > limit] = INF
            D[i] = row
            for t in touched:
                self.fd[t] = UNSEEN
        D = np.minimum(D, D.T)

        # second pass for pairs > limit in both directions: extend the
        # backward side to depth 3 (FWD 3 + BWD 3 resolves d <= 6 exactly),
        # only for the targets/sources involved.
        iu, ju = np.where(np.triu(D == INF, 1))
        if iu.size:
            deep = {}
            def bwd3(t):
                if t not in deep:
                    layer = np.array([targets[t]], dtype=np.int32)
                    for _ in range(BWD_DEPTH + 1):
                        layer = np.unique(self._in_neighbors(layer))
                    deep[t] = layer
                return deep[t]
            partners = {}
            for a, b in zip(iu, ju):
                partners.setdefault(a, []).append(b)
                partners.setdefault(b, []).append(a)
            for s_, ts in partners.items():
                touched = self._forward_layers(targets[s_])
                for t in ts:
                    layer = bwd3(t)
                    if layer.size == 0:
                        continue  # target has no in-edges: unreachable
                    d = int(self.fd[layer].min()) + BWD_DEPTH + 1
                    if d <= limit + 1:
                        D[s_, t] = min(D[s_, t], d)
                for t in touched:
                    self.fd[t] = UNSEEN
            D = np.minimum(D, D.T)

        # fallback: pairs > limit + 1 in both directions (rare) -> uncapped BFS
        iu, ju = np.where(np.triu(D == INF, 1))
        if iu.size:
            self.fallback.set_targets(targets)
            for s in np.unique(np.concatenate([iu, ju])):
                needed = np.zeros(n_t, dtype=bool)
                needed[ju[iu == s]] = True
                needed[iu[ju == s]] = True
                row = self.fallback.hops_from(int(s), needed)
                D[s] = np.minimum(D[s], row)
            self.fallback.clear_targets()
            D = np.minimum(D, D.T)
        return D


def pair_values(D, rng=None, sample=None):
    """The board's pair list: all C(k,2) pairs, or `sample` random ones."""
    n = D.shape[0]
    iu, ju = np.triu_indices(n, 1)
    if sample is not None and sample < len(iu):
        pick = rng.choice(len(iu), sample, replace=False)
        iu, ju = iu[pick], ju[pick]
    return D[iu, ju]


def summarize(vals):
    finite = vals[vals < INF].astype(float)
    return {
        "avg_dist": float(finite.mean()) if finite.size else None,
        "median_dist": float(np.median(finite)) if finite.size else None,
        "max_dist": int(finite.max()) if finite.size else None,
        "n_pairs": int(vals.size),
        "n_unreachable": int((vals == INF).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="board's method 2: draw I random pairs instead of all C(k,2)")
    ap.add_argument("--n-queries", type=int, default=None)
    args = ap.parse_args()

    ctx = get_ctx("glove100")
    ds, graph = ctx.ds, ctx.graph
    indptr, rsrc = build_reverse_csr(
        graph.adj0, DATA_DIR / "glove100_radj_m16_efc200.npz")
    bfs = MeetInMiddle(graph.adj0, indptr, rsrc)
    rng = np.random.default_rng(0)

    e4 = json.loads((RESULTS_DIR / "e4_recall_skew_glove100.json").read_text())
    recalls = np.array(e4["configs"]["baseline_ef160"]["recalls"])
    e9 = json.loads((RESULTS_DIR / "e9_hardness_causal_glove100.json").read_text())
    n_dists = np.array(e9["cost_n_dists_ef160"])
    n_q = args.n_queries or len(recalls)
    recalls, n_dists = recalls[:n_q], n_dists[:n_q]

    keys = ["avg_dist", "median_dist", "max_dist", "n_unreachable"]
    per_q = {k: [] for k in keys}
    hist = {}
    t0 = time.perf_counter()
    for i in range(n_q):
        gt = ds.ground_truth[i][:K_GT]
        D = bfs.pairwise(gt)
        vals = pair_values(D, rng, args.sample)
        s = summarize(vals)
        for k in keys:
            per_q[k].append(s[k])
        for v, c in zip(*np.unique(vals, return_counts=True)):
            key = "inf" if v == INF else str(int(v))
            hist[key] = hist.get(key, 0) + int(c)
        if (i + 1) % 25 == 0 or i == n_q - 1:
            el = time.perf_counter() - t0
            print(
                f"{i + 1:4d}/{n_q}  {el:6.0f}s  ({el / (i + 1):.1f}s/q)  "
                f"avg-dist so far {np.mean(per_q['avg_dist']):.3f}  "
                f"unreachable pairs so far {sum(per_q['n_unreachable'])}",
                flush=True,
            )

    avg = np.array(per_q["avg_dist"], dtype=float)
    corr_recall = spearman(recalls, avg)
    corr_cost = spearman(n_dists, avg)

    # how much did E10's 4-hop cap matter?
    e10_path = RESULTS_DIR / "e10_gt_graphdist_glove100.json"
    vs_e10 = None
    if e10_path.exists():
        e10 = json.loads(e10_path.read_text())
        capped = np.array(e10["per_query"]["gt_hops_mean"])[:n_q]
        vs_e10 = {
            "spearman_exact_vs_capped": spearman(avg, capped),
            "mean_abs_diff": float(np.abs(avg - capped).mean()),
            "max_abs_diff": float(np.abs(avg - capped).max()),
        }

    levels_out = []
    for name, lo, hi in LEVELS:
        sel = (recalls >= lo) & (recalls <= hi)
        levels_out.append({
            "level": name,
            "n_queries": int(sel.sum()),
            "avg_dist": {"mean": float(avg[sel].mean()),
                         "median": float(np.median(avg[sel]))},
            "max_dist": {"mean": float(np.array(per_q["max_dist"])[sel].mean())},
            "n_unreachable": {"sum": int(np.array(per_q["n_unreachable"])[sel].sum())},
        })

    out = {
        "dataset": "glove100",
        "hardness_from": "recall@10, HNSW ef=160",
        "graph": "faiss HNSW M=16 efC=200, level-0 out-edges, uncapped directed BFS",
        "k_gt": K_GT,
        "pairs": "all C(k,2)" if args.sample is None else f"{args.sample} random",
        "distance": "min(hops u->v, hops v->u); unreachable excluded from avg",
        "n_queries": n_q,
        "pair_hop_histogram": dict(sorted(hist.items(), key=lambda kv: (kv[0] == "inf", kv[0]))),
        "spearman_recall_vs_avg_dist": corr_recall,
        "spearman_ndist_vs_avg_dist": corr_cost,
        "vs_e10_capped": vs_e10,
        "levels": levels_out,
        "per_query": {"recalls": recalls.tolist(), "n_dists": n_dists.tolist(), **per_q},
        "elapsed_s": time.perf_counter() - t0,
    }
    suffix = "" if args.sample is None else f"_sample{args.sample}"
    path = RESULTS_DIR / f"e11_avgdist_glove100{suffix}.json"
    path.write_text(json.dumps(out, indent=1))

    print(f"\navg-dist: corr(recall) {corr_recall:+.3f}   corr(n_dist) {corr_cost:+.3f}")
    print(f"{'level':>14} {'n':>5} {'avg-dist':>9} {'median':>8} {'max':>6} {'unreach':>8}")
    for lv in levels_out:
        print(f"{lv['level']:>14} {lv['n_queries']:>5} {lv['avg_dist']['mean']:>9.3f} "
              f"{lv['avg_dist']['median']:>8.3f} {lv['max_dist']['mean']:>6.2f} "
              f"{lv['n_unreachable']['sum']:>8d}")
    print("pair hop histogram:", out["pair_hop_histogram"])
    if vs_e10:
        print("vs E10 (4-hop cap):", {k: round(v, 4) for k, v in vs_e10.items()})
    print(f"elapsed {out['elapsed_s']:.0f}s; saved {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
