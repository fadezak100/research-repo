"""E12b — graph distances for INTERNAL queries, compared with external ones.

Inputs: the 1000 sampled nodes + self-excluded GT (`internal_queries.py`)
and their hardness labels (`hardness5.py`, results/e12_hardness_internal_*).

Per node q, two measurements on the level-0 HNSW graph (out-edges, the graph
the beam search walks; exact, uncapped BFS — same code as E11):

  avg-dist     mean shortest-path hops between every pair of q's true
               neighbors, d(u,v) = min(hops u->v, hops v->u). Identical to
               the external-query measurement, so the two are comparable.
  q->GT hops   NEW, only possible when q is a node: hops from q itself to
               each of its true neighbors (directed, q -> v). Reported as the
               mean, the fraction that are direct out-edges (1 hop), and the
               mean by true-neighbor rank (1-16, 17-32, 33-100, 101-1000).
               One BFS per node instead of C(k,2) pairs.

Both are binned by the hardness label and correlated (Spearman) with the
node's recall and search cost. The matching EXTERNAL-query E11 result is
loaded and its per-bin avg-dist is written next to the internal one:
  label k10   + GT-100  <-> e11_avgdist_glove100.json
  label k1000 + GT-100  <-> e11_avgdist_glove100_k1000_gt100.json
  label k1000 + GT-1000 <-> e11_avgdist_glove100_k1000_gt1000.json
  label k100            no external run exists (noted in the output)

Output: results/e12_avgdist_internal_glove100_{label}_gt{K}.json
Usage:  .venv/bin/python -m pyhnsw.hardness6 [--label k10|k100|k1000]
            [--k-gt 100|1000] [--sample I] [--n-queries N]
Runtime: GT-100 ~3 min; GT-1000 ~2 h (499,500 pairs per node). Checkpoints
every 25 nodes in faiss/data/ and resumes.
"""

import argparse
import json
import time

import numpy as np

from .experiments import RESULTS_DIR, get_ctx
from .graph import DATA_DIR
from .hardness import spearman
from .hardness4 import (
    INF,
    MeetInMiddle,
    build_reverse_csr,
    pair_values,
    summarize,
)
from .internal_queries import K_GT, N_INTERNAL, SEED, internal_gt

RANK_GROUPS = [("1-16", 0, 16), ("17-32", 16, 32), ("33-100", 32, 100), ("101-1000", 100, 1000)]

EXTERNAL_FILES = {
    ("k10", 100): "e11_avgdist_glove100",
    ("k1000", 100): "e11_avgdist_glove100_k1000_gt100",
    ("k1000", 1000): "e11_avgdist_glove100_k1000_gt1000",
}


def bin_mask(recalls, b):
    return (recalls >= b["lo"]) & ((recalls <= b["hi"]) if b["inclusive"] else (recalls < b["hi"]))


def hops_from_query(bfs, q, gt):
    """Directed hop distance q -> each GT node (INF if unreachable)."""
    ex = bfs.fallback  # ExactBFS on the same adjacency
    targets = np.concatenate(([q], gt)).astype(np.int64)
    ex.set_targets(targets)
    d = ex.hops_from(0)
    ex.clear_targets()
    return d[1:]


def summarize_q_hops(h, k_gt):
    finite = h[h < INF].astype(float)
    out = {
        "q_hops_mean": float(finite.mean()) if finite.size else None,
        "q_hops_max": int(finite.max()) if finite.size else None,
        "q_frac_1hop": float((h == 1).mean()),
        "q_unreachable": int((h == INF).sum()),
    }
    for name, lo, hi in RANK_GROUPS:
        if lo >= k_gt:
            break
        seg = h[lo:min(hi, k_gt)]
        fin = seg[seg < INF].astype(float)
        out[f"q_hops_rank_{name}"] = float(fin.mean()) if fin.size else None
    return out


def load_external(label, k_gt):
    name = EXTERNAL_FILES.get((label, k_gt))
    if name is None or not (RESULTS_DIR / f"{name}.json").exists():
        return None
    e = json.loads((RESULTS_DIR / f"{name}.json").read_text())
    return {
        "file": f"{name}.json",
        "hardness_from": e["hardness_from"],
        "n_queries": e["n_queries"],
        "spearman_recall_vs_avg_dist": e["spearman_recall_vs_avg_dist"],
        "spearman_ndist_vs_avg_dist": e["spearman_ndist_vs_avg_dist"],
        "levels": [
            {"level": lv["level"], "n_queries": lv["n_queries"],
             "avg_dist_mean": lv["avg_dist"]["mean"]}
            for lv in e["levels"]
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    ap.add_argument("--label", default="k1000", choices=("k10", "k10_ef40", "k100", "k1000"),
                    help="hardness label from e12_hardness_internal (which k/ef bins)")
    ap.add_argument("--k-gt", type=int, default=100, choices=(100, 1000),
                    help="size of the true-neighbor set whose pairs are measured")
    ap.add_argument("--sample", type=int, default=None,
                    help="draw I random pairs instead of all C(k,2)")
    ap.add_argument("--n-queries", type=int, default=None)
    ap.add_argument("--n", type=int, default=N_INTERNAL)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    k_gt = args.k_gt

    ctx = get_ctx(args.dataset)
    graph = ctx.graph
    indptr, rsrc = build_reverse_csr(graph.adj0, DATA_DIR / f"{args.dataset}_radj_m16_efc200.npz")
    bfs = MeetInMiddle(graph.adj0, indptr, rsrc)
    rng = np.random.default_rng(0)

    qids, gt_all = internal_gt(args.dataset, K_GT, args.n, args.seed)
    labels = json.loads((RESULTS_DIR / f"e12_hardness_internal_{args.dataset}.json").read_text())
    assert labels["qids"] == qids.tolist(), "hardness labels were made for different nodes"
    cfg = labels["configs"][args.label]
    recalls = np.array(cfg["recalls"])
    n_dists = np.array(cfg["n_dists"])
    bins = cfg["bins"]
    n_q = args.n_queries or len(qids)
    qids, gt_all, recalls, n_dists = qids[:n_q], gt_all[:n_q], recalls[:n_q], n_dists[:n_q]
    print(f"hardness label {args.label} (k={cfg['k']}, ef={cfg['ef']}); GT top-{k_gt}; "
          f"{n_q} internal queries")

    keys = ["avg_dist", "median_dist", "max_dist", "n_unreachable",
            "q_hops_mean", "q_hops_max", "q_frac_1hop", "q_unreachable"]
    keys += [f"q_hops_rank_{g}" for g, lo, _ in RANK_GROUPS if lo < k_gt]
    per_q = {k: [] for k in keys}
    hist, q_hist = {}, {}
    tag = f"{args.label}_gt{k_gt}_s{args.sample}" + (f"_n{n_q}" if args.n_queries else "")
    ckpt = DATA_DIR / f"e12b_ckpt_{tag}.json"
    if ckpt.exists():
        saved = json.loads(ckpt.read_text())
        per_q, hist, q_hist = saved["per_q"], saved["hist"], saved["q_hist"]
        print(f"resuming from checkpoint: {len(per_q['avg_dist'])} queries done")
    t0 = time.perf_counter()
    done0 = len(per_q["avg_dist"])
    for i in range(done0, n_q):
        q, gt = int(qids[i]), gt_all[i][:k_gt]
        # (1) pairwise avg-dist among the true neighbors — as for external queries
        D = bfs.pairwise(gt)
        vals = pair_values(D, rng, args.sample)
        s = summarize(vals)
        for v, c in zip(*np.unique(vals, return_counts=True)):
            key = "inf" if v == INF else str(int(v))
            hist[key] = hist.get(key, 0) + int(c)
        # (2) hops from the node itself to each true neighbor — internal only
        h = hops_from_query(bfs, q, gt)
        s.update(summarize_q_hops(h, k_gt))
        for v, c in zip(*np.unique(h, return_counts=True)):
            key = "inf" if v == INF else str(int(v))
            q_hist[key] = q_hist.get(key, 0) + int(c)
        for k in keys:
            per_q[k].append(s[k])
        if (i + 1) % 25 == 0 or i == n_q - 1:
            ckpt.write_text(json.dumps({"per_q": per_q, "hist": hist, "q_hist": q_hist}))
            el = time.perf_counter() - t0
            print(f"{i + 1:4d}/{n_q}  {el:6.0f}s  ({el / (i + 1 - done0):.1f}s/q)  "
                  f"avg-dist so far {np.mean(per_q['avg_dist']):.3f}  "
                  f"q->GT hops so far {np.mean(per_q['q_hops_mean']):.3f}", flush=True)

    avg = np.array(per_q["avg_dist"], dtype=float)
    qh = np.array(per_q["q_hops_mean"], dtype=float)
    frac1 = np.array(per_q["q_frac_1hop"], dtype=float)
    corr = {
        "recall_vs_avg_dist": spearman(recalls, avg),
        "ndist_vs_avg_dist": spearman(n_dists, avg),
        "recall_vs_q_hops": spearman(recalls, qh),
        "ndist_vs_q_hops": spearman(n_dists, qh),
        "recall_vs_q_frac_1hop": spearman(recalls, frac1),
        "avg_dist_vs_q_hops": spearman(avg, qh),
    }

    external = load_external(args.label, k_gt)
    ext_by_level = {lv["level"]: lv for lv in external["levels"]} if external else {}

    levels_out = []
    for b in bins:
        sel = bin_mask(recalls, b)
        row = {
            "level": b["level"], "n_queries": int(sel.sum()),
            "avg_dist": {"mean": float(avg[sel].mean()) if sel.any() else None,
                         "median": float(np.median(avg[sel])) if sel.any() else None},
            "q_hops": {"mean": float(qh[sel].mean()) if sel.any() else None,
                       "frac_1hop": float(frac1[sel].mean()) if sel.any() else None},
            "n_unreachable": {"sum": int(np.array(per_q["n_unreachable"])[sel].sum())},
            "q_unreachable": {"sum": int(np.array(per_q["q_unreachable"])[sel].sum())},
        }
        for g, lo, _ in RANK_GROUPS:
            if lo < k_gt:
                v = np.array(per_q[f"q_hops_rank_{g}"], dtype=float)
                row["q_hops"][f"rank_{g}"] = float(np.nanmean(v[sel])) if sel.any() else None
        ext = ext_by_level.get(b["level"])
        row["external"] = ({"n_queries": ext["n_queries"], "avg_dist_mean": ext["avg_dist_mean"]}
                           if ext else None)
        levels_out.append(row)

    out = {
        "dataset": args.dataset,
        "query_type": "internal (index nodes; self excluded from GT)",
        "hardness_label": args.label,
        "hardness_from": f"recall@{cfg['k']}, HNSW ef={cfg['ef']}",
        "levels_def": bins,
        "graph": "faiss HNSW M=16 efC=200, level-0 out-edges, uncapped directed BFS",
        "k_gt": k_gt,
        "pairs": "all C(k,2)" if args.sample is None else f"{args.sample} random",
        "distance": "avg-dist: min(hops u->v, hops v->u); q_hops: directed q->v; "
                    "unreachable excluded from means",
        "n_queries": n_q,
        "pair_hop_histogram": dict(sorted(hist.items(), key=lambda kv: (kv[0] == "inf", kv[0]))),
        "q_hop_histogram": dict(sorted(q_hist.items(), key=lambda kv: (kv[0] == "inf", kv[0]))),
        "spearman": corr,
        "external_reference": external,
        "levels": levels_out,
        "per_query": {"qids": qids.tolist(), "recalls": recalls.tolist(),
                      "n_dists": n_dists.tolist(), **per_q},
        "elapsed_s": time.perf_counter() - t0,
    }
    suffix = f"_{args.label}_gt{k_gt}"
    if args.sample is not None:
        suffix += f"_sample{args.sample}"
    if args.n_queries:
        suffix += f"_n{n_q}"
    path = RESULTS_DIR / f"e12_avgdist_internal_{args.dataset}{suffix}.json"
    path.write_text(json.dumps(out, indent=1))

    print(f"\nSpearman  avg-dist: recall {corr['recall_vs_avg_dist']:+.3f}  cost {corr['ndist_vs_avg_dist']:+.3f}"
          f"   |  q->GT hops: recall {corr['recall_vs_q_hops']:+.3f}  cost {corr['ndist_vs_q_hops']:+.3f}"
          f"   |  avg-dist vs q->GT {corr['avg_dist_vs_q_hops']:+.3f}")
    ext_hdr = "ext avg-dist" if external else "ext (none)"
    print(f"{'level':>14} {'n':>5} {'avg-dist':>9} {ext_hdr:>13} {'q->GT':>7} {'%1hop':>6} "
          f"{'r1-16':>6} {'r17-32':>7} {'r33-100':>8}")
    def f(v, spec):
        return "-" if v is None else format(v, spec)

    for lv in levels_out:
        e = lv["external"]
        ext = f"{e['avg_dist_mean']:.3f} (n={e['n_queries']})" if e else "-"
        qh_ = lv["q_hops"]
        pct = None if qh_["frac_1hop"] is None else 100 * qh_["frac_1hop"]
        print(f"{lv['level']:>14} {lv['n_queries']:>5} {f(lv['avg_dist']['mean'], '.3f'):>9} {ext:>13} "
              f"{f(qh_['mean'], '.3f'):>7} {f(pct, '.1f'):>5}% "
              f"{f(qh_.get('rank_1-16'), '.2f'):>6} {f(qh_.get('rank_17-32'), '.2f'):>7} "
              f"{f(qh_.get('rank_33-100'), '.2f'):>8}")
    if external:
        print(f"external file: {external['file']}  (Spearman recall vs avg-dist "
              f"{external['spearman_recall_vs_avg_dist']:+.3f})")
    else:
        print(f"no external-query run exists for label {args.label} with GT-{k_gt}")
    print(f"elapsed {out['elapsed_s']:.0f}s; saved {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
