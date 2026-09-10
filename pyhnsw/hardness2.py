"""E9 — is query hardness a data problem or an algorithm problem?

E7 showed hard queries' GT-100 is spread out (gt_pairwise) and barely stands
out from noise (rel_contrast). Those are correlations. E9 adds the causal tests:

  asymptote  : run the hard bin (recall@10 <= 0.25 at ef=160) with ever larger
               ef. If recall -> 1.0 the graph CAN find the GT — hardness is a
               per-query budget problem (algorithmic, fixable with adaptive
               ef). If it plateaus, the GT is structurally unreachable.
  oracle     : re-run hard queries with the search entry point forced to the
               query's true nearest neighbor (skipping the upper-layer
               descent). If recall recovers, the problem is *routing to the
               right region*. If it stays low, even starting inside the GT
               region the beam cannot reach the rest of the GT — a
               data-geometry problem.
  cost link  : per-query distance computations at fixed ef vs the E7 metrics —
               hard queries are also the expensive ones.
  samples    : the professor's protocol — 10 queries per hardness level with
               their GT-closeness metrics, as a slide-ready table.

Hardness label matches E7: per-query recall@10 of plain HNSW at ef=160 on
GloVe-100 (from E4). Usage: .venv/bin/python -m pyhnsw.hardness2
"""

import json

import numpy as np

from .experiments import RESULTS_DIR, get_ctx, run_config
from .hardness import LEVELS, spearman

ASYMPTOTE_EFS = [160, 320, 640, 1280, 2560, 5120]
K = 10


def main():
    ctx = get_ctx("glove100")
    ds = ctx.ds

    e4 = json.loads((RESULTS_DIR / "e4_recall_skew_glove100.json").read_text())
    recalls160 = np.array(e4["configs"]["baseline_ef160"]["recalls"])
    e7 = json.loads((RESULTS_DIR / "e7_hardness_glove100.json").read_text())
    pq = e7["per_query"]

    rng = np.random.default_rng(0)
    bins = {}
    for name, lo, hi in LEVELS:
        idx = np.where((recalls160 >= lo) & (recalls160 <= hi))[0]
        bins[name] = idx
    hard = bins["<=0.25 (hard)"]
    easy_ctl = rng.choice(bins["~1.0 (easy)"], 20, replace=False)

    # ---- professor's protocol: 10 sampled queries per level, GT metrics ----
    samples = []
    for name, _, _ in LEVELS:
        take = rng.choice(bins[name], min(10, len(bins[name])), replace=False)
        for qi in sorted(int(i) for i in take):
            samples.append({
                "level": name, "query": qi, "recall@10_ef160": float(recalls160[qi]),
                **{m: round(float(pq[m][qi]), 3)
                   for m in ["q_gt_mean", "gt_pairwise", "contrast",
                             "rel_contrast", "gt_indegree"]},
            })

    # ---- asymptote test: recall of the hard bin as ef grows ----
    def bin_recall(idx, **kw):
        rs, nds = [], []
        for qi in idx:
            ids, st = ctx.search(ds.test[qi], K, **kw)
            rs.append(len(set(ids[:K]) & set(ds.ground_truth[qi][:K].tolist())) / K)
            nds.append(st["n_dist"])
        return float(np.mean(rs)), float(np.mean(nds)), rs

    asymptote = []
    for ef in ASYMPTOTE_EFS:
        h_avg, h_nd, h_rs = bin_recall(hard, ef=ef)
        e_avg, e_nd, _ = bin_recall(easy_ctl, ef=ef)
        asymptote.append({
            "ef": ef, "hard_recall": h_avg, "hard_n_dist": h_nd,
            "easy_recall": e_avg, "easy_n_dist": e_nd,
            "hard_recalls": h_rs,
        })
        print(f"ef={ef:>5}: hard bin recall={h_avg:.3f} (nd={h_nd:.0f})  "
              f"easy ctl recall={e_avg:.3f}")

    # ---- oracle entry test: start the search at the true nearest neighbor ----
    oracle = {}
    for label, kw in [
        ("normal_ef160", {"ef": 160}),
        ("oracle_ef160", {"ef": 160, "oracle": True}),
        ("normal_ef640", {"ef": 640}),
        ("oracle_ef640", {"ef": 640, "oracle": True}),
    ]:
        use_oracle = kw.pop("oracle", False)
        rs = []
        for qi in hard:
            entry = int(ds.ground_truth[qi][0]) if use_oracle else None
            ids, _ = ctx.search(ds.test[qi], K, entry=entry, **kw)
            rs.append(len(set(ids[:K]) & set(ds.ground_truth[qi][:K].tolist())) / K)
        oracle[label] = {"recall_avg": float(np.mean(rs)), "recalls": rs}
        print(f"{label}: hard bin recall={np.mean(rs):.3f}")

    # ---- cost link: per-query n_dist at ef=160 vs E7 metrics ----
    r160 = run_config(ctx, ef=160)
    nd = np.array(r160["n_dists"])
    cost_corr = {m: spearman(nd, np.array(pq[m]))
                 for m in ["rel_contrast", "gt_pairwise", "q_gt_mean"]}
    cost_corr["recall"] = spearman(nd, recalls160)
    print("spearman(n_dist@ef160, ·):",
          {k: round(v, 3) for k, v in cost_corr.items()})

    out = {
        "dataset": "glove100", "k": K,
        "hardness_from": "recall@10, HNSW ef=160 (E4)",
        "bin_sizes": {name: int(len(idx)) for name, idx in bins.items()},
        "samples": samples,
        "asymptote": asymptote,
        "oracle": oracle,
        "cost_n_dists_ef160": nd.tolist(),
        "cost_spearman": cost_corr,
    }
    (RESULTS_DIR / "e9_hardness_causal_glove100.json").write_text(
        json.dumps(out, indent=1))
    print("saved results/e9_hardness_causal_glove100.json")


if __name__ == "__main__":
    main()
