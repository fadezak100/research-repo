"""E7 — what makes a query hard?

Professor's question: bin queries by hardness (recall at a fixed setting),
look at each query's ground-truth top-100 (GT), and compute "closeness"
metrics on the GT set — do harder queries have ground truths that are, e.g.,
further from each other?

We compute the metrics for ALL 1000 queries (not just 10 per bin) so we can
report rank correlations with hardness; the 4 hardness levels are used for
the box-plot presentation view.

Hardness label: per-query recall@10 of plain HNSW at ef=160 on GloVe-100
(from E4's saved results). Levels: <=0.25 (hard), ~0.5, ~0.75, ~1.0 (easy).

Metrics per query (GT = its exact top-100 neighbor vectors):
  q_gt_mean     mean cosine distance from the query to its GT
                ("does this query have close friends at all?")
  gt_pairwise   mean pairwise cosine distance among GT members
                (professor's hypothesis: hard queries' GT are spread out)
  contrast      d100 / d1 — how much worse is the 100th neighbor than the 1st
                (flat profile = neighbors indistinguishable from each other)
  rel_contrast  mean distance to 2000 random vectors / mean distance to GT-10
                ("how much do the true neighbors stand out from noise?")
  gt_components number of connected islands the GT forms inside the HNSW
                graph (edges of the graph restricted to GT members)
  gt_indegree   mean in-degree of GT nodes in the full graph — how "popular" /
                reachable they are (hubness): rarely-linked GT = hard to reach

Usage: .venv/bin/python -m pyhnsw.hardness
"""

import json

import numpy as np

from .experiments import RESULTS_DIR, get_ctx

LEVELS = [
    ("<=0.25 (hard)", 0.0, 0.25),
    ("~0.5", 0.26, 0.5),
    ("~0.75", 0.51, 0.8),
    ("~1.0 (easy)", 0.81, 1.0),
]


def components_within(gt_ids, adj0):
    """Connected components of the HNSW graph restricted to the GT set."""
    idx = {int(g): i for i, g in enumerate(gt_ids)}
    parent = list(range(len(gt_ids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, g in enumerate(gt_ids):
        for nb in adj0[g]:
            j = idx.get(int(nb))
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return len({find(i) for i in range(len(gt_ids))})


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ctx = get_ctx("glove100")
    ds, graph = ctx.ds, ctx.graph
    adj0 = graph.adj0

    e4 = json.loads((RESULTS_DIR / "e4_recall_skew_glove100.json").read_text())
    recalls = np.array(e4["configs"]["baseline_ef160"]["recalls"])
    n_q = len(recalls)

    # in-degree of every node in the level-0 graph (how often it is linked to)
    indegree = np.bincount(adj0[adj0 >= 0].ravel(), minlength=len(ds.train))

    rng = np.random.default_rng(0)
    rand_ids = rng.choice(len(ds.train), 2000, replace=False)
    rand_vecs = ds.train[rand_ids]

    metrics = {k: [] for k in
               ["q_gt_mean", "gt_pairwise", "contrast", "rel_contrast",
                "gt_components", "gt_indegree"]}
    for i in range(n_q):
        q = ds.test[i]
        gt = ds.ground_truth[i][:100]
        G = ds.train[gt]
        d_qgt = 1.0 - G @ q                       # query -> each GT member
        pair = 1.0 - G @ G.T                      # GT <-> GT
        off = pair[np.triu_indices(len(gt), 1)]
        d_rand = 1.0 - rand_vecs @ q

        metrics["q_gt_mean"].append(float(d_qgt.mean()))
        metrics["gt_pairwise"].append(float(off.mean()))
        metrics["contrast"].append(float(d_qgt.max() / max(d_qgt.min(), 1e-9)))
        metrics["rel_contrast"].append(float(d_rand.mean() / d_qgt[:10].mean()))
        metrics["gt_components"].append(components_within(gt, adj0))
        metrics["gt_indegree"].append(float(indegree[gt].mean()))

    # correlations with hardness over all queries (hard = low recall)
    corr = {k: spearman(recalls, np.array(v)) for k, v in metrics.items()}

    # per-level summaries
    levels_out = []
    for name, lo, hi in LEVELS:
        sel = (recalls >= lo) & (recalls <= hi)
        levels_out.append({
            "level": name, "n_queries": int(sel.sum()),
            **{k: {"mean": float(np.array(v)[sel].mean()),
                   "median": float(np.median(np.array(v)[sel]))}
               for k, v in metrics.items()},
        })

    out = {
        "dataset": "glove100", "hardness_from": "recall@10, HNSW ef=160",
        "n_queries": n_q,
        "spearman_recall_vs_metric": corr,
        "levels": levels_out,
        "per_query": {"recalls": recalls.tolist(), **metrics},
    }
    (RESULTS_DIR / "e7_hardness_glove100.json").write_text(json.dumps(out, indent=1))

    print(f"{'metric':>14} {'corr(recall)':>12}   " +
          "  ".join(f"{name:>14}" for name, *_ in LEVELS))
    for k in metrics:
        row = "  ".join(f"{lv[k]['mean']:>14.3f}" for lv in levels_out)
        print(f"{k:>14} {corr[k]:>12.3f}   {row}")
    print("\nlevel sizes:", [lv["n_queries"] for lv in levels_out])
    print("saved results/e7_hardness_glove100.json")


if __name__ == "__main__":
    main()
