"""E17 — the peek-then-widen policy run end to end in the resumable engine.

E14 chose an ef per query from a statistic of the ef=100 peek and REPLAYED
the outcome from saved Faiss sweeps. Here the same policy is executed as
one search per query in pyhnsw/engine.py:

    s = Search(q, k); s.run(100)             # peek
    stat = edges / contrast10 of s.top(100)  # read the live result heap
    ef = table[stat]                         # E14's table (same training queries, q80 rule)
    if ef > 100: s.run(ef)                   # keep searching, do not restart
    return s.top(k)                          # measured recall and cost

The table is fit exactly as in E14 (10 equal-count bins on the 9000
training queries 1000..9999, each bin the smallest grid ef at which 80%
of its queries reach T). Baselines are measured in the same engine: fixed
ef at every grid value and the per-query oracle, both taken from E16's
resumed sweeps (which equal fresh searches). Each policy is compared with
the fixed-ef frontier interpolated at the policy's own mean cost, with a
paired bootstrap CI, and with what E14 predicted under resume accounting.
Wall-clock per stage is reported for the Python engine (indicative only).

Usage:  .venv/bin/python -m pyhnsw.end_to_end [--dataset glove100|sift1m] [--targets 0.9,0.8]
Output: results/e17_end_to_end_{dataset}.json   (~3 min per dataset)
"""

import argparse
import json
import time

import numpy as np

from .ef_policy import EF0, EF_GRID, K, Score, Table, frontier_at, metrics, train_features
from .engine import Engine, Search
from .index import RESULTS_DIR, load, recall_at_k, save
from .online_stats import Metric, edges_inside

STATS = ["edges", "contrast10"]
N_BOOT = 500


def run_policy(eng, metric, adj0, Q, GT, score, table, T):
    """Execute peek -> statistic -> table -> continue for every query."""
    n = len(Q)
    rec, cost, ef_used, stat_val = np.empty(n), np.empty(n), np.empty(n, dtype=int), np.empty(n)
    rec_peek, cost_peek = np.empty(n), np.empty(n)
    t_peek = t_stat = t_cont = 0.0
    for i in range(n):
        q = Q[i]
        t0 = time.perf_counter()
        s = Search(eng, q, K)
        s.run(EF0)
        ids, dists = s.top(K)
        t1 = time.perf_counter()
        if score.parts == ["edges"]:
            v = edges_inside(adj0, ids)
        else:
            v = metric.expected(q) / dists[:10].mean()
        ef = int(table.lookup(score({score.parts[0]: np.array([v])}))[0])
        t2 = time.perf_counter()
        rec_peek[i], cost_peek[i] = recall_at_k(ids, GT[i], K), s.n_dist
        if ef > EF0:
            s.run(ef)
            ids, _ = s.top(K)
        t3 = time.perf_counter()
        rec[i], cost[i], ef_used[i], stat_val[i] = recall_at_k(ids, GT[i], K), s.n_dist, ef, v
        t_peek, t_stat, t_cont = t_peek + (t1 - t0), t_stat + (t2 - t1), t_cont + (t3 - t2)
    return {"recall": rec, "cost": cost, "ef": ef_used, "stat": stat_val, "recall_peek": rec_peek,
            "cost_peek": cost_peek, "ms_peek": 1e3 * t_peek / n, "ms_stat": 1e3 * t_stat / n, "ms_continue": 1e3 * t_cont / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    ap.add_argument("--targets", default="0.9,0.8")
    args = ap.parse_args()
    dataset = args.dataset
    targets = [float(x) for x in args.targets.split(",")]
    t_start = time.perf_counter()

    ctx = load(dataset)
    ds, graph = ctx.ds, ctx.graph
    metric = Metric(ds)
    eng = Engine(ds, graph)
    train = train_features(dataset, ctx, metric)
    e16 = json.loads((RESULTS_DIR / f"e16_resumable_{dataset}.json").read_text())
    e14 = json.loads((RESULTS_DIR / f"e14_policy_{dataset}.json").read_text())
    n = e16["n_queries"]
    Q, GT = ds.test[:n], ds.ground_truth[:n, :K]
    rec_grid = np.array(e16["per_query"]["recall_resumed"])  # == fresh Faiss recall, E16
    cost_grid = np.array(e16["per_query"]["cost_resumed"], dtype=float)
    rng = np.random.default_rng(0)
    print(f"{dataset}: {n} queries end to end in the resumable engine; targets {targets}; statistics {STATS}")

    results = {}
    for T in targets:
        frontier = [{"ef": ef, **metrics(rec_grid[:, j], cost_grid[:, j], T)} for j, ef in enumerate(EF_GRID)]
        fc = np.array([f["cost"] for f in frontier])
        ok = rec_grid >= T
        j_or = np.where(ok.any(axis=1), ok.argmax(axis=1), len(EF_GRID) - 1)
        rows = np.arange(n)
        oracle = metrics(rec_grid[rows, j_or], cost_grid[rows, j_or], T)
        policies = []
        for sname in STATS:
            score = Score(sname, train)
            table = Table(score(train), train["recall_grid"], T, q=0.8)
            t0 = time.perf_counter()
            r = run_policy(eng, metric, graph.adj0, Q, GT, score, table, T)
            m = metrics(r["recall"], r["cost"], T)
            fixed = {k: frontier_at(fc, np.array([f[k] for f in frontier]), m["cost"]) for k in ["mean", "p5", "frac"]}
            ff = np.array([f["frac"] for f in frontier])
            same_frac_cost = float(np.interp(m["frac"], ff, fc)) if m["frac"] <= ff.max() else None
            diffs = {"p5": [], "frac": []}
            for _ in range(N_BOOT):
                b = rng.integers(0, n, n)
                fcb = cost_grid[b].mean(axis=0)
                for k, fn in [("p5", lambda x: np.percentile(x, 5)), ("frac", lambda x: (x >= T).mean())]:
                    fixed_b = np.interp(r["cost"][b].mean(), fcb, [fn(rec_grid[b, j]) for j in range(len(EF_GRID))])
                    diffs[k].append(fn(r["recall"][b]) - fixed_b)
            # what E14 predicted for this policy under resume accounting
            pred = next(p for p in e14["results"][str(T)]["policies"] if p["stat"] == sname and p["rule"] == "q80")["resume"]
            # per-query agreement with the replay: the chosen ef's recall/cost from E16's grid
            j_ch = np.searchsorted(EF_GRID, r["ef"])
            rec_replay, cost_replay = rec_grid[rows, j_ch], cost_grid[rows, j_ch]
            entry = {
                "stat": sname, "table_ef_hard_to_easy": table.efs, "bin_lower_edge": table.edges.tolist(),
                "measured": {**m, "fixed_ef_at_same_cost": fixed, "fixed_ef_cost_at_same_frac": same_frac_cost,
                             "diff_vs_fixed": {k: {"point": m[k] - fixed[k],
                                                   "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                                               for k, v in diffs.items()},
                             "frac_continued": float((r["ef"] > EF0).mean()),
                             "ef_hist": {str(e): int((r["ef"] == e).sum()) for e in EF_GRID}},
                "e14_replay_prediction": {k: pred[k] for k in ["mean", "p5", "frac", "cost"]},
                "per_query_vs_replay": {"recall_mismatches": int((np.abs(r["recall"] - rec_replay) > 1e-9).sum()),
                                        "max_abs_cost_diff": float(np.max(np.abs(r["cost"] - cost_replay)))},
                "wall_clock_ms_per_query": {"peek": r["ms_peek"], "statistic": r["ms_stat"], "continue": r["ms_continue"]},
                "per_query": {"stat": r["stat"].tolist(), "ef": r["ef"].tolist(), "recall_peek": r["recall_peek"].tolist(),
                              "recall": r["recall"].tolist(), "cost": r["cost"].tolist()},
            }
            policies.append(entry)
            print(f"\n  T={T} {sname:10}  [{time.perf_counter() - t0:.0f}s]  continued {100 * entry['measured']['frac_continued']:.0f}% of queries")
            print(f"    measured : cost {m['cost']:8,.0f}  mean {m['mean']:.3f}  p5 {m['p5']:.2f}  frac>={T} {m['frac']:.3f}")
            print(f"    E14 said : cost {pred['cost']:8,.0f}  mean {pred['mean']:.3f}  p5 {pred['p5']:.2f}  frac>={T} {pred['frac']:.3f}")
            print(f"    fixed ef at the same cost: frac {fixed['frac']:.3f}  p5 {fixed['p5']:.2f}  ->  "
                  f"diff frac {entry['measured']['diff_vs_fixed']['frac']['point']:+.3f} "
                  f"[{entry['measured']['diff_vs_fixed']['frac']['ci95'][0]:+.3f}, {entry['measured']['diff_vs_fixed']['frac']['ci95'][1]:+.3f}]")
            sf = f"{same_frac_cost:,.0f} ({100 * (1 - m['cost'] / same_frac_cost):+.0f}%)" if same_frac_cost else "n/a"
            print(f"    fixed-ef cost for the same frac: {sf};  oracle: cost {oracle['cost']:,.0f} frac {oracle['frac']:.3f}")
            print(f"    per-query vs replay: {entry['per_query_vs_replay']['recall_mismatches']} recall mismatches, "
                  f"max cost diff {entry['per_query_vs_replay']['max_abs_cost_diff']:.0f};  "
                  f"wall-clock ms/query: peek {r['ms_peek']:.1f}, statistic {r['ms_stat']:.2f}, continue {r['ms_continue']:.1f}")
        results[str(T)] = {"target": T, "frontier": frontier, "oracle": oracle, "policies": policies}

    out = {"dataset": dataset, "n_queries": n, "k": K, "peek_ef": EF0, "ef_grid": EF_GRID, "rule": "q80",
           "train_queries": "test[1000:10000] (as E14)", "results": results, "elapsed_s": time.perf_counter() - t_start}
    save(f"e17_end_to_end_{dataset}", out)
    print(f"elapsed {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
