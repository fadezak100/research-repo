"""Hardness labels: plain HNSW (Faiss) at a fixed k and ef, per-query recall
and cost, and the hard / mid / easy bins.

A query is "hard" when plain HNSW, given the same fixed budget as everyone
else, still misses most of its true neighbors. The recall it reaches at that
budget is its label; the bins group the labels for presentation.

Query sets
  external  the dataset's held-out test vectors; ground truth = the shipped
            top-100 (k <= 100) or the brute-forced top-1000 (gt_for_k).
  internal  index nodes used as queries (internal_queries.py); the node is in
            the index, so we search for k+1, drop the node's own id from the
            result, and score against its self-excluded ground truth.

Operating points (ef fixed per k):
  k10_ef160     recall@10 at ef=160      bins <=0.25 / ~0.5 / ~0.75 / ~1.0 (closed)
  k10_ef40      recall@10 at ef=40       same bins
  k100_ef100    recall@100 at ef=100     bins <0.6 / 0.6-0.7 / 0.7-0.8 / >=0.8 (half-open)
  k1000_ef1000  recall@1000 at ef=1000   same bins

Output: results/labels_{external,internal}_{dataset}.json with, per operating
point, the per-query recalls, distance computations (ndis) and hops, and the
bin table. Consumers read it through load_labels().

Usage: .venv/bin/python -m pyhnsw.labels [--dataset glove100]
           [--queries external|internal] [--only k10_ef160,k1000_ef1000]
"""

import argparse
import json
import time

import numpy as np

from .index import RESULTS_DIR, gt_for_k, load, recall_at_k, save, search_with_cost

# recall@10 takes only the values 0.0, 0.1, ..., 1.0 -> closed bins
LEVELS_K10 = [
    ("<=0.25 (hard)", 0.0, 0.25),
    ("~0.5", 0.26, 0.5),
    ("~0.75", 0.51, 0.8),
    ("~1.0 (easy)", 0.81, 1.0),
]
# recall@100 / @1000 is near-continuous -> half-open bins [lo, hi)
LEVELS_CONT = [
    ("<0.6 (hard)", 0.0, 0.6),
    ("0.6-0.7", 0.6, 0.7),
    ("0.7-0.8", 0.7, 0.8),
    (">=0.8 (easy)", 0.8, 1.01),
]

CONFIGS = {  # label -> (k, ef)
    "k10_ef160": (10, 160),
    "k10_ef40": (10, 40),
    "k100_ef100": (100, 100),
    "k1000_ef1000": (1000, 1000),
}


def levels_for(k):
    """Bin definitions and whether they are closed intervals."""
    return (LEVELS_K10, True) if k == 10 else (LEVELS_CONT, False)


def bin_mask(recalls, b):
    """Boolean mask of the queries in bin b (a dict with lo/hi/inclusive or a
    (name, lo, hi) tuple plus `inclusive`)."""
    recalls = np.asarray(recalls)
    return (recalls >= b["lo"]) & ((recalls <= b["hi"]) if b["inclusive"] else (recalls < b["hi"]))


def bins_for(k):
    levels, inclusive = levels_for(k)
    return [{"level": n, "lo": lo, "hi": hi, "inclusive": inclusive} for n, lo, hi in levels]


def bin_table(recalls, n_dists, k):
    recalls, n_dists = np.asarray(recalls), np.asarray(n_dists)
    rows = []
    for b in bins_for(k):
        sel = bin_mask(recalls, b)
        rows.append({
            **b,
            "n_queries": int(sel.sum()),
            "recall_mean": float(recalls[sel].mean()) if sel.any() else None,
            "n_dist_mean": float(n_dists[sel].mean()) if sel.any() else None,
        })
    return rows


def label(index, Q, gt, k, ef, self_ids=None):
    """Plain HNSW on the query vectors Q at (k, ef). gt[i] is the true
    neighbor list of query i. If self_ids is given (internal queries), ask
    for k+1 and drop each query's own id from its result."""
    ask = k + 1 if self_ids is not None else k
    ids, ndis, nhops = search_with_cost(index, Q, ask, ef)
    recalls = np.empty(len(Q))
    self_not_found = 0
    self_rank = []
    for i in range(len(Q)):
        row = ids[i]
        if self_ids is not None:
            pos = np.flatnonzero(row == self_ids[i])
            if pos.size:
                self_rank.append(int(pos[0]))
                row = np.delete(row, pos[0])
            else:
                self_not_found += 1
        recalls[i] = recall_at_k(row, gt[i], k)
    out = {"recalls": recalls, "n_dists": ndis, "n_hops": nhops}
    if self_ids is not None:
        out["self_not_found"] = self_not_found
        out["self_rank_hist"] = np.bincount(self_rank).tolist() if self_rank else []
    return out


def run(dataset="glove100", queries="external", only=None, n=1000, seed=0):
    ctx = load(dataset)
    ds = ctx.ds
    if queries == "external":
        Q = ds.test[:n]
        gt_all = gt_for_k(dataset, 1000, n)
        self_ids, qids = None, None
    else:
        from .internal_queries import internal_gt
        qids, gt_all = internal_gt(dataset, 1000, n, seed)
        Q = ds.train[qids]
        self_ids = qids
    labels = only.split(",") if only else list(CONFIGS)
    configs = {}
    for lab in labels:
        k, ef = CONFIGS[lab]
        t0 = time.perf_counter()
        r = label(ctx.index, Q, gt_all, k, ef, self_ids)
        el = time.perf_counter() - t0
        bins = bin_table(r["recalls"], r["n_dists"], k)
        configs[lab] = {
            "k": k, "ef": ef,
            "recall_mean": float(r["recalls"].mean()),
            "recall_p5": float(np.percentile(r["recalls"], 5)),
            "n_dist_mean": float(r["n_dists"].mean()),
            "seconds": el,
            "recalls": r["recalls"].tolist(),
            "n_dists": r["n_dists"].tolist(),
            "n_hops": r["n_hops"].tolist(),
            "bins": bins,
            **({k_: v for k_, v in r.items() if k_.startswith("self_")}),
        }
        extra = f"  self_not_found {r['self_not_found']}" if self_ids is not None else ""
        print(f"\n== {lab}: k={k} ef={ef}  mean recall {r['recalls'].mean():.3f}  "
              f"p5 {np.percentile(r['recalls'], 5):.2f}  mean ndis {r['n_dists'].mean():.0f}{extra}  [{el:.0f}s]")
        print(f"   {'bin':>14} {'n':>5} {'recall':>7} {'ndis':>8}")
        for b in bins:
            rm = f"{b['recall_mean']:.3f}" if b["recall_mean"] is not None else "-"
            dm = f"{b['n_dist_mean']:.0f}" if b["n_dist_mean"] is not None else "-"
            print(f"   {b['level']:>14} {b['n_queries']:>5} {rm:>7} {dm:>8}")
    payload = {
        "dataset": dataset,
        "query_type": queries,
        "search": "faiss IndexHNSWFlat, fixed efSearch, cost = hnsw_stats.ndis",
        "n_queries": int(len(Q)),
        "configs": configs,
    }
    if qids is not None:
        payload["seed"] = seed
        payload["qids"] = qids.tolist()
    return save(f"labels_{queries}_{dataset}", payload)


def load_labels(dataset, queries, config):
    """(recalls, n_dists, bins, description) for one operating point of a
    saved labels file."""
    d = json.loads((RESULTS_DIR / f"labels_{queries}_{dataset}.json").read_text())
    c = d["configs"][config]
    desc = f"recall@{c['k']}, HNSW ef={c['ef']}"
    return np.array(c["recalls"]), np.array(c["n_dists"]), c["bins"], desc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    ap.add_argument("--queries", default="external", choices=("external", "internal"))
    ap.add_argument("--only", default=None, help="comma-separated subset of configs")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.dataset, args.queries, args.only, args.n, args.seed)


if __name__ == "__main__":
    main()
