"""E12a — hardness labels for INTERNAL queries (index nodes as queries).

Step 1 (`internal_queries.py`) sampled 1000 node ids and brute-forced their
exact top-1000 true neighbors with the node itself removed. This step runs
plain HNSW (fixed ef, no early termination) with each node's own vector as
the query and labels the node hard/easy by the recall it reaches.

Per node, per configuration:
  1. search for k+1 results (the node itself comes back at distance 0)
  2. delete the node's own id from the result; if it is missing keep the
     first k and count it (`self_not_found`)
  3. recall@k = |found[:k] & true_top_k| / k      (both sides self-excluded)
  4. cost = number of distance computations

Configurations (ef fixed per k, matching the external-query labels):
  k10        k=10   ef=160   bins <=0.25 / ~0.5 / ~0.75 / ~1.0   (closed, E4/E11)
  k10_ef40   k=10   ef=40    same bins — backup if ef=160 leaves no hard tail
  k100       k=100  ef=100   bins <0.6 / 0.6-0.7 / 0.7-0.8 / >=0.8 (half-open)
  k1000      k=1000 ef=1000  same half-open bins (E11 k=1000 label)

Adjacency check (no search): per node, how many of its 2M=32 out-edges are
in its true top-100 / top-1000, and how many of its 16 nearest true
neighbors are direct out-edges ("first M neighbors ~1 hop by construction").

Output: results/e12_hardness_internal_glove100.json
Usage:  .venv/bin/python -m pyhnsw.hardness5 [--dataset glove100] [--n 1000]
            [--seed 0] [--only k10,k100,...]
"""

import argparse
import json
import time

import numpy as np

from .experiments import RESULTS_DIR, get_ctx, save
from .hardness import LEVELS
from .hardness4 import LEVELS_K1000
from .internal_queries import K_GT, N_INTERNAL, SEED, internal_gt
from .search import recall_at_k

# label -> (k, ef, bins, bins are closed intervals?)
CONFIGS = {
    "k10": (10, 160, LEVELS, True),
    "k10_ef40": (10, 40, LEVELS, True),
    "k100": (100, 100, LEVELS_K1000, False),
    "k1000": (1000, 1000, LEVELS_K1000, False),
}

# mean recall of the 1000 EXTERNAL queries at the same operating points
EXTERNAL_REF = {"k10": 0.853, "k10_ef40": 0.715, "k100": 0.696, "k1000": 0.798}

M = 16  # graph parameter; level-0 degree is 2M


def bin_mask(recalls, lo, hi, inclusive):
    return (recalls >= lo) & ((recalls <= hi) if inclusive else (recalls < hi))


def bin_table(recalls, n_dists, levels, inclusive):
    rows = []
    for name, lo, hi in levels:
        sel = bin_mask(recalls, lo, hi, inclusive)
        rows.append({
            "level": name, "lo": lo, "hi": hi, "inclusive": inclusive,
            "n_queries": int(sel.sum()),
            "recall_mean": float(recalls[sel].mean()) if sel.any() else None,
            "n_dist_mean": float(n_dists[sel].mean()) if sel.any() else None,
        })
    return rows


def label_queries(ctx, qids, gt, k, ef):
    """Plain HNSW on each node's own vector; self removed from the result."""
    ds = ctx.ds
    recalls = np.empty(len(qids))
    n_dists = np.empty(len(qids), dtype=np.int64)
    self_not_found = 0
    self_rank = np.full(len(qids), -1, dtype=np.int32)  # position of self in result
    t0 = time.perf_counter()
    for i, q in enumerate(qids):
        ids, st = ctx.search(ds.train[q], k + 1, ef=ef)
        ids = np.asarray(ids)
        pos = np.flatnonzero(ids == q)
        if pos.size:
            self_rank[i] = pos[0]
            ids = np.delete(ids, pos[0])
        else:
            self_not_found += 1
        recalls[i] = recall_at_k(ids.tolist(), gt[i], k)
        n_dists[i] = st["n_dist"]
    elapsed = time.perf_counter() - t0
    return recalls, n_dists, self_not_found, self_rank, elapsed


def adjacency_check(adj0, qids, gt):
    """How much of each node's true neighborhood is wired directly to it."""
    out = {"edges_in_gt100": [], "edges_in_gt1000": [], "top16_direct": [],
           "out_degree": []}
    for i, q in enumerate(qids):
        nb = adj0[q]
        nb = set(nb[nb >= 0].tolist())
        g100, g1000 = set(gt[i, :100].tolist()), set(gt[i, :1000].tolist())
        out["out_degree"].append(len(nb))
        out["edges_in_gt100"].append(len(nb & g100))
        out["edges_in_gt1000"].append(len(nb & g1000))
        out["top16_direct"].append(len(set(gt[i, :M].tolist()) & nb))
    summary = {key: {"mean": float(np.mean(v)), "min": int(np.min(v)),
                     "max": int(np.max(v)),
                     "hist": np.bincount(v, minlength=2 * M + 1).tolist()}
               for key, v in out.items()}
    return {"per_query": out, "summary": summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    ap.add_argument("--n", type=int, default=N_INTERNAL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of configs, e.g. k10,k100")
    args = ap.parse_args()

    ctx = get_ctx(args.dataset)
    qids, gt = internal_gt(args.dataset, K_GT, args.n, args.seed)
    assert gt.shape == (len(qids), K_GT), "qids/gt mismatch (n or seed differ)"
    assert not np.any(gt == qids[:, None]), "self id present in GT"
    print(f"{len(qids)} internal queries, GT top-{K_GT} (self excluded)")

    labels = args.only.split(",") if args.only else list(CONFIGS)
    results = {}
    for label in labels:
        k, ef, levels, inclusive = CONFIGS[label]
        recalls, n_dists, snf, self_rank, el = label_queries(ctx, qids, gt, k, ef)
        bins = bin_table(recalls, n_dists, levels, inclusive)
        results[label] = {
            "k": k, "ef": ef,
            "recall_mean": float(recalls.mean()),
            "recall_p5": float(np.percentile(recalls, 5)),
            "n_dist_mean": float(n_dists.mean()),
            "self_not_found": snf,
            "self_rank_hist": np.bincount(self_rank[self_rank >= 0]).tolist(),
            "seconds": el,
            "recalls": recalls.tolist(),
            "n_dists": n_dists.tolist(),
            "bins": bins,
        }
        print(f"\n== {label}: k={k} ef={ef}  mean recall {recalls.mean():.3f} "
              f"(external {EXTERNAL_REF.get(label, float('nan')):.3f})  "
              f"mean n_dist {n_dists.mean():.0f}  self_not_found {snf}  [{el:.0f}s]")
        print(f"   {'bin':>14} {'n':>5} {'recall':>7} {'n_dist':>8}")
        for b in bins:
            r = f"{b['recall_mean']:.3f}" if b["recall_mean"] is not None else "-"
            d = f"{b['n_dist_mean']:.0f}" if b["n_dist_mean"] is not None else "-"
            print(f"   {b['level']:>14} {b['n_queries']:>5} {r:>7} {d:>8}")

    adj = adjacency_check(ctx.graph.adj0, qids, gt)
    s = adj["summary"]
    print(f"\n== adjacency (no search): of each node's {2 * M} out-edges, "
          f"{s['edges_in_gt100']['mean']:.1f} are in its GT-100 and "
          f"{s['edges_in_gt1000']['mean']:.1f} in its GT-1000; "
          f"{s['top16_direct']['mean']:.1f} of its 16 nearest true neighbors "
          f"are direct out-edges (min {s['top16_direct']['min']}, "
          f"max {s['top16_direct']['max']})")

    save(f"e12_hardness_internal_{args.dataset}", {
        "dataset": args.dataset,
        "query_type": "internal (index nodes, self excluded from GT and results)",
        "n_queries": len(qids), "seed": args.seed, "qids": qids.tolist(),
        "external_reference_recall_mean": EXTERNAL_REF,
        "configs": results,
        "adjacency": adj,
    })


if __name__ == "__main__":
    main()
