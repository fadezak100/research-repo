"""E12a — graph wiring around the internal query nodes (no search).

For each of the 1000 sampled nodes (internal_queries.py), how much of its
true neighborhood is wired directly to it in the level-0 graph:

  top16_direct      of its 16 (= M) nearest true neighbors, how many are
                    direct out-edges — "are the first M neighbors one hop
                    away by construction?"
  edges_in_gt100    of its <=32 out-edges, how many are in its true top-100
  edges_in_gt1000   ... in its true top-1000
  out_degree        number of out-edges actually used (of 2M = 32 slots)

Pure set intersection between the node's adjacency row and its ground
truth; nothing is searched. The hardness labels themselves come from
`labels.py --queries internal`.

Output: results/adjacency_internal_{dataset}.json
Usage:  .venv/bin/python -m pyhnsw.adjacency [--dataset glove100] [--n 1000] [--seed 0]
"""

import argparse

import numpy as np

from .index import load, save
from .internal_queries import K_GT, N_INTERNAL, SEED, internal_gt

M = 16  # graph parameter; level-0 degree is 2M


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
    args = ap.parse_args()

    ctx = load(args.dataset)
    qids, gt = internal_gt(args.dataset, K_GT, args.n, args.seed)
    assert not np.any(gt == qids[:, None]), "self id present in GT"
    adj = adjacency_check(ctx.graph.adj0, qids, gt)
    s = adj["summary"]
    print(f"of each node's {2 * M} out-edges, {s['edges_in_gt100']['mean']:.1f} are in its "
          f"GT-100 and {s['edges_in_gt1000']['mean']:.1f} in its GT-1000; "
          f"{s['top16_direct']['mean']:.1f} of its 16 nearest true neighbors are direct "
          f"out-edges (min {s['top16_direct']['min']}, max {s['top16_direct']['max']}); "
          f"mean out-degree {s['out_degree']['mean']:.1f}")
    save(f"adjacency_internal_{args.dataset}", {
        "dataset": args.dataset,
        "graph": "faiss HNSW M=16 efC=200, level-0 out-edges",
        "n_queries": int(len(qids)), "seed": args.seed, "qids": qids.tolist(),
        **adj,
    })


if __name__ == "__main__":
    main()
