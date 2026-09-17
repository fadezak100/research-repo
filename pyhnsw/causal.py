"""E9 — is query hardness a budget problem?

E7 showed hard queries' GT-100 is spread out (gt_pairwise) and barely stands
out from noise (rel_contrast). Those are correlations. E9 adds two tests:

  asymptote  : run the hard bin (recall@10 <= 0.25 at ef=160) with ever larger
               ef. If recall -> 1.0 the graph CAN find the GT — hardness is a
               per-query budget problem. If it plateaus, the GT is
               structurally unreachable.
  cost link  : per-query distance computations at fixed ef vs the E7 metrics —
               hard queries are also the expensive ones.
  samples    : the professor's protocol — 10 queries per hardness level with
               their GT-closeness metrics, as a slide-ready table.

Hardness label: the k10_ef160 operating point of labels.py (recall@10 of
plain HNSW at ef=160, GloVe-100). Searches run through Faiss.
Usage: .venv/bin/python -m pyhnsw.causal
"""

import json

import numpy as np

from .gt_metrics import spearman
from .index import RESULTS_DIR, load, recall_at_k, save, search_with_cost
from .labels import bin_mask, load_labels

ASYMPTOTE_EFS = [160, 320, 640, 1280, 2560, 5120]
K = 10
LABEL = "k10_ef160"


def main():
    ctx = load("glove100")
    ds = ctx.ds

    recalls160, nd160, bins, desc = load_labels("glove100", "external", LABEL)
    e7 = json.loads((RESULTS_DIR / "e7_hardness_glove100.json").read_text())
    pq = e7["per_query"]

    rng = np.random.default_rng(0)
    idx_by_level = {b["level"]: np.where(bin_mask(recalls160, b))[0] for b in bins}
    hard = idx_by_level["<=0.25 (hard)"]
    easy_ctl = rng.choice(idx_by_level["~1.0 (easy)"], 20, replace=False)

    # ---- professor's protocol: 10 sampled queries per level, GT metrics ----
    samples = []
    for name, idx in idx_by_level.items():
        take = rng.choice(idx, min(10, len(idx)), replace=False)
        for qi in sorted(int(i) for i in take):
            samples.append({
                "level": name, "query": qi, "recall@10_ef160": float(recalls160[qi]),
                **{m: round(float(pq[m][qi]), 3)
                   for m in ["q_gt_mean", "gt_pairwise", "contrast",
                             "rel_contrast", "gt_indegree"]},
            })

    # ---- asymptote test: recall of the hard bin as ef grows ----
    def bin_recall(idx, ef):
        ids, ndis, _ = search_with_cost(ctx.index, ds.test[idx], K, ef)
        rs = [recall_at_k(ids[j], ds.ground_truth[qi], K) for j, qi in enumerate(idx)]
        return float(np.mean(rs)), float(ndis.mean()), rs

    asymptote = []
    for ef in ASYMPTOTE_EFS:
        h_avg, h_nd, h_rs = bin_recall(hard, ef)
        e_avg, e_nd, _ = bin_recall(easy_ctl, ef)
        asymptote.append({
            "ef": ef, "hard_recall": h_avg, "hard_n_dist": h_nd,
            "easy_recall": e_avg, "easy_n_dist": e_nd,
            "hard_recalls": h_rs,
        })
        print(f"ef={ef:>5}: hard bin recall={h_avg:.3f} (ndis={h_nd:.0f})  "
              f"easy ctl recall={e_avg:.3f}")

    # ---- cost link: per-query distance computations at ef=160 vs E7 metrics ----
    cost_corr = {m: spearman(nd160, np.array(pq[m]))
                 for m in ["rel_contrast", "gt_pairwise", "q_gt_mean"]}
    cost_corr["recall"] = spearman(nd160, recalls160)
    print("spearman(ndis@ef160, ·):", {k: round(v, 3) for k, v in cost_corr.items()})

    save("e9_hardness_causal_glove100", {
        "dataset": "glove100", "k": K,
        "hardness_from": f"{desc} (labels.py {LABEL})",
        "search": "faiss IndexHNSWFlat, cost = hnsw_stats.ndis",
        "bin_sizes": {name: int(len(idx)) for name, idx in idx_by_level.items()},
        "samples": samples,
        "asymptote": asymptote,
        "cost_n_dists_ef160": nd160.tolist(),
        "cost_spearman": cost_corr,
    })


if __name__ == "__main__":
    main()
