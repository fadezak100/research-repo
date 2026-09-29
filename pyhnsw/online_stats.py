"""E13 — query-time hardness statistics from a prefix search (external queries).

The offline experiments (E7, E11) measure statistics of a query's TRUE
neighbors and show they separate hard from easy queries. At query time the
true neighbors are unknown, so the same statistics must be computed on the
neighbors a cheap prefix search has FOUND. This experiment measures how much
of the hardness signal survives that substitution, statistic by statistic.

Prefix search: Faiss HNSW at k=100, ef=100 (the cheapest search that returns
100 candidates). Every query-time statistic below uses only its output: the
100 found ids, their distances, and the hop / distance counters.

Query-time statistics (found top-100)            offline counterpart (true top-100)
  d10, d100          mean distance to found top-10/100    d10_gt, d100_gt
  contrast10/100     expected distance to a random base   contrast10_gt/100_gt
                     vector / d10 or d100 (label-free,    (E7 rel_contrast with the
                     one dot product with the mean vector)  exact expectation)
  spread             mean pairwise distance among found   spread_gt (E7 gt_pairwise)
  edges              level-0 out-edges with both ends     edges_gt
                     in the found set
  avgdist            exact avg-dist (BFS) among found     avgdist_gt (E11, per query)
  n_hops, n_dist     effort of the ef=100 search          -
  ep_dist, ep_nbr_std, ep_nbr_mean, ep_degree, descent_ndist, global_ep_dist
                     QBAT-style entry-time features from the greedy descent
                     through the upper layers (before the level-0 beam)
Distances are the index's native ones: cosine distance (1 - ip) on GloVe,
squared L2 on SIFT1M. Expected distance to a random base vector is exact:
cosine 1 - q.xbar; L2 |q|^2 - 2 q.xbar + mean |x|^2.

Targets per query
  recall     recall@100 of the ef=100 search (the hardness label, k100_ef100)
  ef_min(T)  smallest ef on EF_GRID reaching recall@100 >= T (T = 0.8, 0.9);
             censored queries (never reach T by the cap) are ranked last
  cost(T)    distance computations at ef_min(T)

Reported: tie-averaged Spearman of every statistic with the three targets,
AUC for detecting the hard bin (bin 0 of the dataset's k100_ef100 bins), and
for each statistic the Spearman with its offline counterpart (what the
substitution loses). Per-query values, the recall and cost matrices of the ef
sweep are saved for later policy replay.

Output: results/e13_online_stats_{dataset}.json
Usage:  .venv/bin/python -m pyhnsw.online_stats [--dataset glove100|sift1m] [--n 1000]
"""

import argparse
import json
import time

import numpy as np

from .avgdist import INF, MeetInMiddle, build_reverse_csr, pair_values, summarize
from .graph import DATA_DIR
from .index import RESULTS_DIR, load, recall_at_k, save, search_with_cost
from .labels import bin_mask, load_labels

K = 100
EF0 = 100
LABEL = "k100_ef100"
EF_GRID = [100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]
TARGETS = (0.8, 0.9)
E11_FILES = {  # per-query avg-dist among the true top-100 of the same 1000 queries
    "glove100": "e11_avgdist_glove100_k1000_ef1000_gt100",
    "sift1m": "e11_avgdist_sift1m_k100_ef100_gt100",
}
PAIRS = [("d10", "d10_gt"), ("d100", "d100_gt"), ("contrast10", "contrast10_gt"),
         ("contrast100", "contrast100_gt"), ("spread", "spread_gt"),
         ("edges", "edges_gt"), ("avgdist", "avgdist_gt")]
ONLINE = ["d10", "d100", "contrast10", "contrast100", "spread", "edges", "avgdist",
          "n_hops", "n_dist", "ep_dist", "ep_nbr_mean", "ep_nbr_std", "ep_degree",
          "descent_ndist", "global_ep_dist"]
OFFLINE = [b for _, b in PAIRS]


# ------------------------------------------------------------------ stats
def rank_avg(x):
    """Tie-averaged ranks (1-based)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rank_avg(x), rank_avg(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")  # constant target (e.g. every query reaches T at ef=k)
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(stat, positive):
    """Mann-Whitney AUC of `stat` for the positive class; returned oriented
    (>= 0.5) with the direction that flags positives."""
    r = rank_avg(stat)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    a = (r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return (a, "high") if a >= 0.5 else (1 - a, "low")


# --------------------------------------------------------------- geometry
class Metric:
    """Native distances of the index: cosine distance or squared L2."""

    def __init__(self, ds):
        self.kind = ds.metric
        X = ds.train
        self.X = X
        self.xbar = X.mean(axis=0).astype(np.float32)
        if self.kind == "l2":
            self.sq = np.einsum("ij,ij->i", X, X)
            self.msq = float(self.sq.mean())
        else:
            self.sq, self.msq = None, None

    def to_q(self, ids, q):
        """Distances from q to X[ids]."""
        if self.kind == "cosine":
            return 1.0 - self.X[ids] @ q
        return self.sq[ids] - 2.0 * (self.X[ids] @ q) + float(q @ q)

    def faiss_to_dist(self, D):
        return 1.0 - D if self.kind == "cosine" else D

    def expected(self, q):
        """Exact mean distance from q to a random base vector."""
        if self.kind == "cosine":
            return 1.0 - float(q @ self.xbar)
        return float(q @ q) - 2.0 * float(q @ self.xbar) + self.msq

    def pairwise_mean(self, ids):
        G = self.X[ids]
        if self.kind == "cosine":
            P = 1.0 - G @ G.T
        else:
            s = self.sq[ids]
            P = s[:, None] + s[None, :] - 2.0 * (G @ G.T)
        iu = np.triu_indices(len(ids), 1)
        return float(P[iu].mean())


def descend(graph, metric, q):
    """Greedy descent through the upper layers (as Faiss does before the
    level-0 beam). Returns the level-0 entry point, its distance, and the
    number of distance computations spent."""
    cur = graph.entry_point
    cur_dist = float(metric.to_q(np.array([cur]), q)[0])
    n_dist = 1
    for lvl in range(graph.max_level, 0, -1):
        table = graph.upper[lvl]
        changed = True
        while changed:
            changed = False
            nbrs = table.get(cur)
            if nbrs is None or len(nbrs) == 0:
                break
            d = metric.to_q(nbrs, q)
            n_dist += len(nbrs)
            j = int(np.argmin(d))
            if d[j] < cur_dist:
                cur_dist, cur, changed = float(d[j]), int(nbrs[j]), True
    return cur, cur_dist, n_dist


def edges_inside(adj0, ids):
    """Directed level-0 edges u->v with both endpoints in `ids`."""
    nb = adj0[ids]
    return int(np.isin(nb[nb >= 0], ids).sum())


# ------------------------------------------------------------------- main
def ef_sweep(index, Q, gt, k, grid):
    """recall@k and distance computations of every query at every ef."""
    rec = np.empty((len(Q), len(grid)))
    cost = np.empty((len(Q), len(grid)), dtype=np.int64)
    for j, ef in enumerate(grid):
        t = time.perf_counter()
        ids, ndis, _ = search_with_cost(index, Q, k, ef)
        rec[:, j] = [recall_at_k(ids[i], gt[i], k) for i in range(len(Q))]
        cost[:, j] = ndis
        print(f"  ef={ef:5d}  mean recall {rec[:, j].mean():.3f}  mean ndis {ndis.mean():.0f}  "
              f"[{time.perf_counter() - t:.0f}s]", flush=True)
    return rec, cost


def ef_min(rec, cost, grid, T):
    """Per query: first grid ef with recall >= T (censored -> -1), and its cost."""
    ok = rec >= T
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    efm = np.where(first >= 0, np.array(grid)[np.maximum(first, 0)], -1)
    c = cost[np.arange(len(rec)), np.where(first >= 0, first, len(grid) - 1)]
    return efm, c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    dataset, n = args.dataset, args.n
    t0 = time.perf_counter()

    ctx = load(dataset)
    ds, index, graph = ctx.ds, ctx.index, ctx.graph
    metric = Metric(ds)
    Q = ds.test[:n]
    GT = ds.ground_truth[:n, :K]
    indptr, rsrc = build_reverse_csr(graph.adj0, DATA_DIR / f"{dataset}_radj_m16_efc200.npz")
    bfs = MeetInMiddle(graph.adj0, indptr, rsrc)
    print(f"{dataset}: {n} external queries, k={K}, prefix ef={EF0}  [{time.perf_counter() - t0:.0f}s]")

    # 1. prefix search and its recall (must equal the stored k100_ef100 label)
    F, ndis0, nhops0, D0 = search_with_cost(index, Q, K, EF0, with_dists=True)
    dF = metric.faiss_to_dist(D0)
    recall = np.array([recall_at_k(F[i], GT[i], K) for i in range(n)])
    lab_rec, _, bins, _ = load_labels(dataset, "external", LABEL)
    mism = int((np.abs(lab_rec[:n] - recall) > 1e-9).sum())
    print(f"prefix search: mean recall {recall.mean():.3f}, mean ndis {ndis0.mean():.0f}; "
          f"queries whose recall differs from the stored label: {mism}")

    # 2. ef sweep -> oracle ef per target
    print("ef sweep:")
    rec_grid, cost_grid = ef_sweep(index, Q, GT, K, EF_GRID)
    targets = {"recall": recall}
    censored = {}
    for T in TARGETS:
        efm, c = ef_min(rec_grid, cost_grid, EF_GRID, T)
        censored[T] = int((efm < 0).sum())
        targets[f"ef_min_{T}"] = np.where(efm < 0, 2 * EF_GRID[-1], efm).astype(float)
        targets[f"cost_{T}"] = c.astype(float)
        print(f"  T={T}: censored (never reach T by ef={EF_GRID[-1]}): {censored[T]}/{n}; "
              f"ef_min distribution: " + ", ".join(f"{e}:{int((efm == e).sum())}" for e in EF_GRID))

    # 3. per-query statistics
    e11 = json.loads((RESULTS_DIR / f"{E11_FILES[dataset]}.json").read_text())
    assert e11["n_queries"] >= n and e11["k_gt"] == K
    avgdist_gt = np.array(e11["per_query"]["avg_dist"][:n], dtype=float)
    stats = {s: np.empty(n) for s in ONLINE + OFFLINE}
    stats["avgdist_gt"] = avgdist_gt
    t = time.perf_counter()
    for i in range(n):
        q = Q[i]
        f, d = F[i], dF[i]
        g = GT[i].astype(np.int64)
        dg = metric.to_q(g, q)
        ex = metric.expected(q)
        stats["d10"][i], stats["d100"][i] = d[:10].mean(), d.mean()
        stats["contrast10"][i], stats["contrast100"][i] = ex / d[:10].mean(), ex / d.mean()
        stats["d10_gt"][i], stats["d100_gt"][i] = dg[:10].mean(), dg.mean()
        stats["contrast10_gt"][i], stats["contrast100_gt"][i] = ex / dg[:10].mean(), ex / dg.mean()
        stats["spread"][i] = metric.pairwise_mean(f)
        stats["spread_gt"][i] = metric.pairwise_mean(g)
        stats["edges"][i] = edges_inside(graph.adj0, f)
        stats["edges_gt"][i] = edges_inside(graph.adj0, g)
        s = summarize(pair_values(bfs.pairwise(f)))
        stats["avgdist"][i] = s["avg_dist"] if s["avg_dist"] is not None else np.nan
        stats["n_hops"][i], stats["n_dist"][i] = nhops0[i], ndis0[i]
        ep, epd, nd = descend(graph, metric, q)
        nb = graph.adj0[ep]
        nb = nb[nb >= 0]
        dn = metric.to_q(nb, q)
        stats["ep_dist"][i], stats["descent_ndist"][i], stats["ep_degree"][i] = epd, nd, len(nb)
        stats["ep_nbr_mean"][i], stats["ep_nbr_std"][i] = dn.mean(), dn.std()
        stats["global_ep_dist"][i] = metric.to_q(np.array([graph.entry_point]), q)[0]
        if (i + 1) % 200 == 0:
            print(f"  stats {i + 1}/{n}  [{time.perf_counter() - t:.0f}s]", flush=True)

    # 4. correlations, hard-bin detection, offline pairing
    hard = bin_mask(recall, bins[0])
    table = {}
    for name in ONLINE + OFFLINE:
        v = stats[name]
        row = {tn: spearman(v, tv) for tn, tv in targets.items()}
        a, direction = auc(v, hard)
        row["auc_hard"], row["hard_when"] = a, direction
        table[name] = row
    pairing = {a: {"offline": b, "spearman_online_vs_offline": spearman(stats[a], stats[b])}
               for a, b in PAIRS}

    out = {
        "dataset": dataset, "query_type": "external (test set)", "n_queries": n,
        "k": K, "prefix_ef": EF0, "label": LABEL, "hard_bin": bins[0],
        "n_hard": int(hard.sum()), "ef_grid": EF_GRID, "targets": list(TARGETS),
        "censored": {str(T): censored[T] for T in TARGETS},
        "distance": "cosine distance" if metric.kind == "cosine" else "squared L2",
        "online_stats": ONLINE, "offline_stats": OFFLINE,
        "spearman": table, "pairing": pairing,
        "per_query": {**{k_: v.tolist() for k_, v in stats.items()},
                      **{k_: v.tolist() for k_, v in targets.items()},
                      "recall_grid": rec_grid.tolist(), "cost_grid": cost_grid.tolist(),
                      "found_ids": F.tolist()},
        "elapsed_s": time.perf_counter() - t0,
    }
    save(f"e13_online_stats_{dataset}", out)

    tn = list(targets)
    print(f"\n{dataset}: Spearman with the targets ({n} external queries; hard bin = "
          f"{bins[0]['level']}, n={int(hard.sum())})")
    print(f"{'statistic':>16} {'recall':>8} {'ef_min.8':>9} {'ef_min.9':>9} {'cost.9':>8} "
          f"{'AUC hard':>9} {'offline twin':>14} {'twin:recall':>12} {'online~twin':>12}")
    twin = {a: b for a, b in PAIRS}
    for name in ONLINE:
        r = table[name]
        b = twin.get(name)
        tw = f"{b:>14} {table[b]['recall']:>+12.3f} {pairing[name]['spearman_online_vs_offline']:>+12.3f}" if b else ""
        print(f"{name:>16} {r['recall']:>+8.3f} {r['ef_min_0.8']:>+9.3f} {r['ef_min_0.9']:>+9.3f} "
              f"{r['cost_0.9']:>+8.3f} {r['auc_hard']:>6.3f}({r['hard_when'][0]}) {tw}")
    print(f"elapsed {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
