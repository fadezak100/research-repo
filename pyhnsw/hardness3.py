"""E10 — graph distance among a query's true neighbors (whiteboard 2026-09-01).

E7 measured how spread out a query's ground-truth top-100 (GT) is in *vector*
space (gt_pairwise = mean cosine distance between GT members). This
experiment measures the same spread in the *graph*:

    d_G(u, v) = shortest-path hop count from u to v in the level-0 HNSW graph

  for every pair (u, v) of GT members — C(100, 2) = 4950 pairs, i.e. the
  "~5000" on the board — and aggregates per query:

  gt_hops_mean    mean d_G over all GT pairs          ("avg-dist" on the board)
  gt_hops_median  median d_G over all GT pairs
  gt_hops_far     fraction of pairs NOT reachable within MAX_HOPS hops
  gt_nn_hops      mean over GT nodes of d_G to its *closest* other GT node
                  (1 = every true neighbor is adjacent to some other true
                  neighbor; > 1 = isolated members the beam must jump to)

Two points that matter for interpreting the numbers:

  * Directed.  Search follows out-edges only, so BFS uses out-edges. We take
    d_G(u, v) = min(hops u->v, hops v->u) — pairs are unordered on the board.
  * Censored.  BFS is capped at MAX_HOPS; pairs not reached by then count as
    MAX_HOPS + 1 (the graph's own diameter is ~log_32(1.2M) ≈ 4, so
    "> 4 hops" already means "not in the same neighbourhood").  gt_hops_far
    reports how many pairs were censored so the mean is never read blind.
    Checked: re-running with MAX_HOPS=5 (15x slower) moves gt_hops_mean by
    < 0.01 and drives gt_hops_far to ~0 — i.e. essentially every censored
    pair is exactly 5 hops apart, so the 4-hop numbers are near-exact.

Implementation: one hop-bounded BFS from each of the 100 GT nodes.  Each hop
is a numpy gather over adj0 plus np.unique; the final hop is gather-only
(no dedup needed because we never expand it).  ~4 min on GloVe-100.

Outputs results/e10_gt_graphdist_glove100.json — per-query values, Spearman
correlations with hardness (recall@10 at ef=160), with search cost
(n_dist at ef=160, from E9), and with the E7 vector-space metrics.

Usage: .venv/bin/python -m pyhnsw.hardness3
"""

import json
import time

import numpy as np

from .experiments import RESULTS_DIR, get_ctx
from .hardness import LEVELS, spearman

MAX_HOPS = 4
K_GT = 100


class HopBFS:
    """Reusable hop-bounded BFS over a -1-padded adjacency matrix."""

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

    def hops_from(self, src_idx, max_hops):
        """Hop distance from targets[src_idx] to every target (max_hops+1 if
        not reached within max_hops)."""
        adj0, visited, tpos = self.adj0, self.visited, self.tpos
        n_t = len(self.targets)
        dist = np.full(n_t, max_hops + 1, dtype=np.int16)
        dist[src_idx] = 0
        found = 1
        src = self.targets[src_idx]
        frontier = np.array([src])
        visited[src] = True
        touched = [frontier]
        for h in range(1, max_hops + 1):
            nb = adj0[frontier].ravel()
            nb = nb[nb >= 0]
            if h < max_hops:
                nb = nb[~visited[nb]]
                if nb.size == 0:
                    break
                nb = np.unique(nb)
                visited[nb] = True
                touched.append(nb)
            # record any targets first seen at this hop
            hits = tpos[nb]
            hits = hits[hits >= 0]
            if hits.size:
                new = hits[dist[hits] > h]
                if new.size:
                    dist[new] = h
                    found += len(np.unique(new))
            if found == n_t:
                break
            frontier = nb
        for t in touched:
            visited[t] = False
        return dist

    def pairwise(self, targets, max_hops=MAX_HOPS):
        """Symmetrised (min of both directions) hop-distance matrix among targets."""
        self.set_targets(targets)
        n_t = len(targets)
        D = np.empty((n_t, n_t), dtype=np.int16)
        for i in range(n_t):
            D[i] = self.hops_from(i, max_hops)
        self.clear_targets()
        return np.minimum(D, D.T)


def summarize(D, max_hops=MAX_HOPS):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    pairs = D[iu].astype(float)
    off = D + np.eye(n, dtype=np.int16) * (max_hops + 2)  # ignore the diagonal
    return {
        "gt_hops_mean": float(pairs.mean()),
        "gt_hops_median": float(np.median(pairs)),
        "gt_hops_far": float((pairs > max_hops).mean()),
        "gt_nn_hops": float(off.min(axis=1).mean()),
    }


def main():
    ctx = get_ctx("glove100")
    ds, graph = ctx.ds, ctx.graph
    bfs = HopBFS(graph.adj0)

    e4 = json.loads((RESULTS_DIR / "e4_recall_skew_glove100.json").read_text())
    recalls = np.array(e4["configs"]["baseline_ef160"]["recalls"])
    e7 = json.loads((RESULTS_DIR / "e7_hardness_glove100.json").read_text())
    e9 = json.loads((RESULTS_DIR / "e9_hardness_causal_glove100.json").read_text())
    n_dists = np.array(e9["cost_n_dists_ef160"])
    n_q = len(recalls)

    keys = ["gt_hops_mean", "gt_hops_median", "gt_hops_far", "gt_nn_hops"]
    metrics = {k: [] for k in keys}
    hist = np.zeros(MAX_HOPS + 2, dtype=np.int64)  # pooled pair-distance histogram
    t0 = time.perf_counter()
    for i in range(n_q):
        gt = ds.ground_truth[i][:K_GT]
        D = bfs.pairwise(gt)
        s = summarize(D)
        for k in keys:
            metrics[k].append(s[k])
        hist += np.bincount(D[np.triu_indices(K_GT, 1)], minlength=MAX_HOPS + 2)
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(
                f"{i + 1:4d}/{n_q}  {el:6.0f}s  mean hops so far "
                f"{np.mean(metrics['gt_hops_mean']):.3f}",
                flush=True,
            )

    arr = {k: np.array(v) for k, v in metrics.items()}
    corr_recall = {k: spearman(recalls, arr[k]) for k in keys}
    corr_cost = {k: spearman(n_dists, arr[k]) for k in keys}
    e7_keys = ["gt_pairwise", "q_gt_mean", "rel_contrast"]
    corr_e7 = {
        k: {j: spearman(np.array(e7["per_query"][j]), arr[k]) for j in e7_keys}
        for k in keys
    }

    levels_out = []
    for name, lo, hi in LEVELS:
        sel = (recalls >= lo) & (recalls <= hi)
        levels_out.append(
            {
                "level": name,
                "n_queries": int(sel.sum()),
                **{
                    k: {"mean": float(arr[k][sel].mean()),
                        "median": float(np.median(arr[k][sel]))}
                    for k in keys
                },
            }
        )

    out = {
        "dataset": "glove100",
        "hardness_from": "recall@10, HNSW ef=160",
        "graph": "faiss HNSW M=16 efC=200, level-0 out-edges, directed BFS",
        "k_gt": K_GT,
        "max_hops": MAX_HOPS,
        "censored_value": MAX_HOPS + 1,
        "n_queries": n_q,
        "pair_hop_histogram": {str(h): int(c) for h, c in enumerate(hist)},
        "spearman_recall_vs_metric": corr_recall,
        "spearman_ndist_vs_metric": corr_cost,
        "spearman_vs_e7": corr_e7,
        "levels": levels_out,
        "per_query": {"recalls": recalls.tolist(), "n_dists": n_dists.tolist(),
                      **{k: arr[k].tolist() for k in keys}},
        "elapsed_s": time.perf_counter() - t0,
    }
    (RESULTS_DIR / "e10_gt_graphdist_glove100.json").write_text(json.dumps(out, indent=1))

    print(
        f"\n{'metric':>14} {'corr(recall)':>12} {'corr(n_dist)':>12}   "
        + "  ".join(f"{name:>14}" for name, *_ in LEVELS)
    )
    for k in keys:
        row = "  ".join(f"{lv[k]['mean']:>14.3f}" for lv in levels_out)
        print(f"{k:>14} {corr_recall[k]:>12.3f} {corr_cost[k]:>12.3f}   {row}")
    print("\npair hop histogram (all queries pooled):",
          {h: int(c) for h, c in enumerate(hist)})
    print("corr with E7 vector metrics:")
    for k in keys:
        print(f"  {k:>14}: " + ", ".join(f"{j}={v:+.2f}" for j, v in corr_e7[k].items()))
    print(f"\nelapsed {out['elapsed_s']:.0f}s; saved results/e10_gt_graphdist_glove100.json")


if __name__ == "__main__":
    main()
