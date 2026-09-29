"""E16 — the query-time statistics measured INSIDE a resumable search.

E13 computed contrast and edges on the output of a separate Faiss search
and E14 replayed the cost of "peek, then widen ef" from saved sweeps. Here
both are done for real in the Python engine (pyhnsw/engine.py):

  1. peek     run the beam search until it converges at ef0 (=100); read the
              current top-100 and compute the statistics on it (edges,
              contrast, d10/d100, and the entry-time ep_dist for reference)
  2. resume   keep the SAME search and widen ef along the grid
              100 -> 150 -> ... -> 3200, recording recall@100 and the
              cumulative distance computations after each step
  3. fresh    (reference) for the same grid, the Faiss recall and cost of a
              search started from scratch, taken from E13's saved sweep

Checks
  engine ~ Faiss   the engine's peek recall must equal the stored
                   k100_ef100 label; its statistics must match E13's
  correlation      Spearman / AUC / catch rate of each in-engine statistic
                   with the peek recall and with the oracle ef, where the
                   oracle ef is now the smallest ef at which the RESUMED
                   search reaches the target (measured, not replayed)
  resume cost      per grid ef: resumed recall vs fresh recall, resumed
                   cumulative cost vs fresh cost vs two-stage cost
                   (peek + fresh), with and without the spill heap

Usage:  .venv/bin/python -m pyhnsw.resumable_stats [--dataset glove100|sift1m] [--n 1000]
Output: results/e16_resumable_{dataset}.json   (~10 min per dataset)
"""

import argparse
import json
import time

import numpy as np

from .engine import Engine, Search
from .index import RESULTS_DIR, load, recall_at_k, save
from .labels import bin_mask, load_labels
from .online_stats import Metric, auc, edges_inside, spearman

K, EF0, LABEL = 100, 100, "k100_ef100"
EF_GRID = [100, 150, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]
TARGETS = (0.8, 0.9)
STATS = ["edges", "contrast10", "contrast100", "d10", "d100", "n_dist", "ep_dist"]


def ef_min(rec_grid, T):
    """Smallest grid ef whose recall reaches T; 2x the cap when none does."""
    ok = rec_grid >= T
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    return np.where(first < 0, 2 * EF_GRID[-1], np.array(EF_GRID)[np.maximum(first, 0)]).astype(float)


def catch_rate(stat, hard, hard_when):
    """Flag the n_hard most suspicious queries; share of them that are hard."""
    n_hard = int(hard.sum())
    order = np.argsort(-stat if hard_when == "high" else stat, kind="mergesort")[:n_hard]
    return float(hard[order].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    dataset, n = args.dataset, args.n
    t0 = time.perf_counter()

    ctx = load(dataset)
    ds, graph = ctx.ds, ctx.graph
    metric = Metric(ds)
    eng = Engine(ds, graph)
    Q, GT = ds.test[:n], ds.ground_truth[:n, :K]
    e13 = json.loads((RESULTS_DIR / f"e13_online_stats_{dataset}.json").read_text())
    pq13 = e13["per_query"]
    assert e13["ef_grid"] == EF_GRID and e13["n_queries"] >= n
    lab_rec, _, bins, _ = load_labels(dataset, "external", LABEL)
    print(f"{dataset}: {n} external queries, k={K}, peek ef={EF0}, resume grid {EF_GRID}")

    stats = {s: np.empty(n) for s in STATS}
    recall0 = np.empty(n)
    rec_res = {True: np.empty((n, len(EF_GRID))), False: np.empty((n, len(EF_GRID)))}
    cost_res = {True: np.empty((n, len(EF_GRID))), False: np.empty((n, len(EF_GRID)))}
    hops0 = np.empty(n)
    t = time.perf_counter()
    for i in range(n):
        q = Q[i]
        for spill in (True, False):
            s = Search(eng, q, K, spill=spill)
            for j, ef in enumerate(EF_GRID):
                s.run(ef)
                ids, dists = s.top(K)
                rec_res[spill][i, j] = recall_at_k(ids, GT[i], K)
                cost_res[spill][i, j] = s.n_dist
                if j == 0 and spill:  # the peek: statistics on what the search has now
                    ex = metric.expected(q)
                    stats["d10"][i], stats["d100"][i] = dists[:10].mean(), dists.mean()
                    stats["contrast10"][i], stats["contrast100"][i] = ex / dists[:10].mean(), ex / dists.mean()
                    stats["edges"][i] = edges_inside(graph.adj0, ids)
                    stats["n_dist"][i], hops0[i] = s.n_dist, s.n_hops
                    stats["ep_dist"][i] = s.ep_dist
                    recall0[i] = rec_res[spill][i, 0]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}  [{time.perf_counter() - t:.0f}s]", flush=True)

    # ---- engine ~ Faiss
    mism = int((np.abs(lab_rec[:n] - recall0) > 1e-9).sum())
    agree = {s: {"spearman_vs_e13": spearman(stats[s], np.array(pq13[s][:n])),
                 "max_abs_diff": float(np.max(np.abs(stats[s] - np.array(pq13[s][:n]))))} for s in STATS}
    rec_fresh = np.array(pq13["recall_grid"])[:n]
    cost_fresh = np.array(pq13["cost_grid"])[:n].astype(float)

    # ---- targets and correlations
    hard = bin_mask(recall0, bins[0])
    targets = {"recall": recall0}
    for T in TARGETS:
        targets[f"ef_min_{T}_resumed"] = ef_min(rec_res[True], T)
        targets[f"ef_min_{T}_fresh"] = ef_min(rec_fresh, T)
    table = {}
    for s in STATS:
        row = {tn: spearman(stats[s], tv) for tn, tv in targets.items()}
        a, direction = auc(stats[s], hard)
        row["auc_hard"], row["hard_when"] = a, direction
        row["catch_rate"] = catch_rate(stats[s], hard, direction)
        table[s] = row

    # ---- resume accounting per grid ef
    two_stage = cost_fresh[:, :1] + np.where(np.arange(len(EF_GRID)) > 0, cost_fresh, 0)
    resume = []
    for j, ef in enumerate(EF_GRID):
        resume.append({
            "ef": ef,
            "recall_fresh": float(rec_fresh[:, j].mean()),
            "recall_resumed": float(rec_res[True][:, j].mean()),
            "recall_resumed_no_spill": float(rec_res[False][:, j].mean()),
            "cost_fresh": float(cost_fresh[:, j].mean()),
            "cost_resumed": float(cost_res[True][:, j].mean()),
            "cost_resumed_no_spill": float(cost_res[False][:, j].mean()),
            "cost_two_stage": float(two_stage[:, j].mean()),
            "frac_0.9_fresh": float((rec_fresh[:, j] >= 0.9).mean()),
            "frac_0.9_resumed": float((rec_res[True][:, j] >= 0.9).mean()),
            "frac_0.9_resumed_no_spill": float((rec_res[False][:, j] >= 0.9).mean()),
        })

    out = {
        "dataset": dataset, "n_queries": n, "k": K, "peek_ef": EF0, "ef_grid": EF_GRID, "label": LABEL,
        "hard_bin": bins[0], "n_hard": int(hard.sum()),
        "engine_vs_faiss": {"peek_recall_mismatches": mism, "statistics": agree},
        "spearman": table, "resume": resume,
        "per_query": {**{s: v.tolist() for s, v in stats.items()}, "n_hops": hops0.tolist(),
                      **{tn: tv.tolist() for tn, tv in targets.items()},
                      "recall_resumed": rec_res[True].tolist(), "cost_resumed": cost_res[True].tolist(),
                      "recall_resumed_no_spill": rec_res[False].tolist(), "cost_resumed_no_spill": cost_res[False].tolist()},
        "elapsed_s": time.perf_counter() - t0,
    }
    save(f"e16_resumable_{dataset}", out)

    print(f"\nengine vs Faiss: peek recall differs from the stored label for {mism}/{n} queries")
    for s in STATS:
        print(f"  {s:12} Spearman with E13 value {agree[s]['spearman_vs_e13']:+.4f}, max |diff| {agree[s]['max_abs_diff']:.4g}")
    print(f"\nin-engine statistics (hard bin {bins[0]['level']}, n={int(hard.sum())}):")
    print(f"{'statistic':>12} {'rho recall':>11} {'rho ef.9 res':>13} {'rho ef.9 fresh':>15} {'AUC':>6} {'catch':>6}")
    for s in STATS:
        r = table[s]
        print(f"{s:>12} {r['recall']:>+11.3f} {r['ef_min_0.9_resumed']:>+13.3f} {r['ef_min_0.9_fresh']:>+15.3f} "
              f"{r['auc_hard']:>6.3f} {100 * r['catch_rate']:>5.0f}%")
    print(f"\nresume accounting (means over {n} queries):")
    print(f"{'ef':>5} {'recall fresh':>13} {'resumed':>8} {'no spill':>9} {'cost fresh':>11} {'resumed':>9} {'no spill':>9} {'two-stage':>10}")
    for r in resume:
        print(f"{r['ef']:>5} {r['recall_fresh']:>13.3f} {r['recall_resumed']:>8.3f} {r['recall_resumed_no_spill']:>9.3f} "
              f"{r['cost_fresh']:>11,.0f} {r['cost_resumed']:>9,.0f} {r['cost_resumed_no_spill']:>9,.0f} {r['cost_two_stage']:>10,.0f}")
    print(f"elapsed {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
