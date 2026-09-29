"""E14 — from a query-time statistic to an ef: two-stage policy vs fixed ef.

Policy, per query
  1. search at k=100, ef=100 (stage 1) and compute a statistic on the 100
     found neighbors (E13): `edges` = level-0 out-edges among the found set,
     `contrast10` = expected random distance / mean distance to the found
     top-10, or `both` = mean of their percentile ranks
  2. look the value up in a table: statistic bin -> ef
  3. if ef > 100, search again at that ef (stage 2) and return that result

Table (fit on the 9000 test queries NOT used anywhere else, indices
1000..9999): sort the training queries by the statistic (oriented so higher
= easier), split into N_BINS equal-count bins; each bin gets the smallest
grid ef at which the bin reaches the target:
  q80   at least 80% of the bin's queries have recall@100 >= T
  mean  the bin's mean recall@100 >= T (Ada-ef's rule; known to under-serve
        the per-query contract)
Bins that never reach the target get the cap (largest grid ef); ef is then
forced non-increasing from hard to easy bins.

Evaluation: replayed on the 1000 held-out queries of E13 from their saved
ef-sweep recall / cost matrices (no new searches). Per policy: mean recall,
p5, fraction of queries with recall >= T, and mean cost in distance
computations, accounted two ways: `two_stage` = stage-1 cost + stage-2 cost
for re-searched queries (what this implementation pays), `resume` = the cost
of the chosen ef alone (what an engine that continues the search would pay).
Baselines: fixed ef at every grid value (the frontier) and the oracle (per
query the smallest grid ef reaching T). Each policy is compared with the
fixed-ef frontier interpolated at the policy's own cost, with a paired
bootstrap CI on the differences in p5 and fraction >= T.

Output: results/e14_policy_{dataset}.json
        --per-query STAT also writes results/e14_per_query_{dataset}_T{T}_{STAT}.csv/.xlsx:
        one row per evaluation query with the statistic, its bin, ef before (100) and
        after, recall before and after, cost before and after, and the oracle ef
        ("not by 3200" when no grid ef reaches T; larger ef does, see SESSION_2026-09-18).
Usage:  .venv/bin/python -m pyhnsw.ef_policy [--dataset glove100|sift1m] [--targets 0.9,0.8]
            [--per-query contrast10|edges|both]
"""

import argparse
import json
import time

import numpy as np

from .graph import DATA_DIR
from .index import RESULTS_DIR, load, save
from .online_stats import EF0, EF_GRID, K, Metric, edges_inside, spearman

TRAIN_SLICE = (1000, 10000)
STATS = ["edges", "contrast10", "both"]
RULES = {"q80": 0.8, "mean": None}
N_BINS = 10
N_BOOT = 500


# --------------------------------------------------------------- training
def train_features(dataset, ctx, metric):
    """Stage-1 statistics and the ef-sweep recall matrix of the training
    queries (batched Faiss search; cached)."""
    cache = DATA_DIR / f"{dataset}_policy_train_k{K}_ef{EF0}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    ds, index = ctx.ds, ctx.index
    lo, hi = TRAIN_SLICE
    Q = np.ascontiguousarray(ds.test[lo:hi], dtype=np.float32)
    GT = ds.ground_truth[lo:hi, :K]
    n = len(Q)
    t = time.perf_counter()
    index.hnsw.efSearch = EF0
    D, I = index.search(Q, K)
    dF = metric.faiss_to_dist(D)
    edges = np.array([edges_inside(ctx.graph.adj0, I[i]) for i in range(n)], dtype=float)
    contrast10 = np.array([metric.expected(Q[i]) / dF[i, :10].mean() for i in range(n)])
    print(f"  training stage-1 statistics for {n} queries [{time.perf_counter() - t:.0f}s]")
    rec = np.empty((n, len(EF_GRID)))
    for j, ef in enumerate(EF_GRID):
        t = time.perf_counter()
        index.hnsw.efSearch = ef
        _, I = index.search(Q, K)
        rec[:, j] = [np.isin(I[i], GT[i]).sum() / K for i in range(n)]
        print(f"  training sweep ef={ef:5d}  mean recall {rec[:, j].mean():.3f}  [{time.perf_counter() - t:.0f}s]",
              flush=True)
    out = {"edges": edges, "contrast10": contrast10, "recall_grid": rec}
    np.savez(cache, **out)
    return out


class Score:
    """Orients a statistic so that higher = easier (sign fixed on the
    training set) and, for `both`, averages percentile ranks against the
    training distribution."""

    def __init__(self, name, train):
        self.name = name
        self.parts = ["edges", "contrast10"] if name == "both" else [name]
        self.sign = {}
        self.sorted = {}
        for p in self.parts:
            self.sign[p] = 1.0 if spearman(train[p], train["recall_grid"][:, 0]) >= 0 else -1.0
            self.sorted[p] = np.sort(self.sign[p] * train[p])

    def __call__(self, vals):
        if len(self.parts) == 1:
            p = self.parts[0]
            return self.sign[p] * np.asarray(vals[p], dtype=float)
        pct = [np.searchsorted(self.sorted[p], self.sign[p] * np.asarray(vals[p], dtype=float),
                               side="right") / len(self.sorted[p]) for p in self.parts]
        return np.mean(pct, axis=0)


class Table:
    """Equal-count score bins -> ef."""

    def __init__(self, score, rec_grid, T, n_bins=N_BINS, q=0.8):
        order = np.argsort(score, kind="mergesort")
        chunks = np.array_split(order, n_bins)
        self.edges = np.array([score[c].min() for c in chunks])  # lower edge, ascending
        efs = []
        for c in chunks:
            r = rec_grid[c]
            ok = (r >= T).mean(axis=0) >= q if q is not None else r.mean(axis=0) >= T
            efs.append(EF_GRID[int(np.argmax(ok))] if ok.any() else EF_GRID[-1])
        for i in range(n_bins - 2, -1, -1):  # harder bins never get a smaller ef
            efs[i] = max(efs[i], efs[i + 1])
        self.efs = efs
        self.n_per_bin = [len(c) for c in chunks]

    def lookup(self, score):
        b = np.clip(np.searchsorted(self.edges, score, side="right") - 1, 0, len(self.efs) - 1)
        return np.array(self.efs)[b]


# -------------------------------------------------------------- evaluation
def metrics(rec, cost, T):
    return {"mean": float(rec.mean()), "p5": float(np.percentile(rec, 5)),
            "p1": float(np.percentile(rec, 1)), "frac": float((rec >= T).mean()),
            "cost": float(cost.mean())}


def replay(ef_choice, rec_grid, cost_grid, T):
    """Recall and both cost accountings of a per-query ef choice."""
    j = np.searchsorted(EF_GRID, ef_choice)
    rows = np.arange(len(ef_choice))
    rec = rec_grid[rows, j]
    resume = cost_grid[rows, j]
    two_stage = cost_grid[:, 0] + np.where(j > 0, resume, 0)
    return rec, resume, two_stage


def frontier_at(frontier_cost, frontier_val, cost):
    """Linear interpolation of a fixed-ef metric at a given mean cost."""
    return float(np.interp(cost, frontier_cost, frontier_val))


def evaluate(dataset, T, train, ev, boot_rng):
    rec_grid, cost_grid = ev["recall_grid"], ev["cost_grid"]
    n = len(rec_grid)
    # fixed-ef frontier and oracle
    frontier = [{"ef": ef, **metrics(rec_grid[:, j], cost_grid[:, j], T)} for j, ef in enumerate(EF_GRID)]
    fc = np.array([f["cost"] for f in frontier])
    ok = rec_grid >= T
    j_or = np.where(ok.any(axis=1), ok.argmax(axis=1), len(EF_GRID) - 1)
    rows = np.arange(n)
    oracle_rec, oracle_cost = rec_grid[rows, j_or], cost_grid[rows, j_or]
    oracle = {"resume": metrics(oracle_rec, oracle_cost, T),
              "two_stage": metrics(oracle_rec, cost_grid[:, 0] + np.where(j_or > 0, oracle_cost, 0), T),
              "censored": int((~ok.any(axis=1)).sum())}

    policies = []
    for sname in STATS:
        score = Score(sname, train)
        s_train, s_eval = score(train), score(ev)
        for rname, q in RULES.items():
            table = Table(s_train, train["recall_grid"], T, q=q)
            ef_choice = table.lookup(s_eval)
            rec, resume, two = replay(ef_choice, rec_grid, cost_grid, T)
            entry = {"stat": sname, "rule": rname, "table_ef_hard_to_easy": table.efs,
                     "bin_lower_edge": table.edges.tolist(), "n_per_bin": table.n_per_bin,
                     "frac_researched": float((ef_choice > EF0).mean()),
                     "ef_choice_hist": {str(e): int((ef_choice == e).sum()) for e in EF_GRID}}
            for acc, cost in [("two_stage", two), ("resume", resume)]:
                m = metrics(rec, cost, T)
                fixed = {k: frontier_at(fc, np.array([f[k] for f in frontier]), m["cost"])
                         for k in ["mean", "p5", "frac"]}
                # cost a fixed ef needs for the same fraction >= T (inverse interpolation)
                ff = np.array([f["frac"] for f in frontier])
                same_frac_cost = float(np.interp(m["frac"], ff, fc)) if m["frac"] <= ff.max() else None
                # paired bootstrap on the differences vs fixed ef at matched cost
                diffs = {"p5": [], "frac": []}
                for _ in range(N_BOOT):
                    b = boot_rng.integers(0, n, n)
                    rb, cb = rec[b], cost[b]
                    fcb = cost_grid[b].mean(axis=0)
                    for k, fn in [("p5", lambda x: np.percentile(x, 5)), ("frac", lambda x: (x >= T).mean())]:
                        fixed_b = np.interp(cb.mean(), fcb, [fn(rec_grid[b, j]) for j in range(len(EF_GRID))])
                        diffs[k].append(fn(rb) - fixed_b)
                entry[acc] = {**m, "fixed_ef_at_same_cost": fixed,
                              "fixed_ef_cost_at_same_frac": same_frac_cost,
                              "diff_vs_fixed": {k: {"point": m[k] - fixed[k],
                                                    "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
                                                for k, v in diffs.items()}}
            policies.append(entry)
    return {"target": T, "frontier": frontier, "oracle": oracle, "policies": policies}


def write_per_query(dataset, T, stat, train, ev):
    """One row per evaluation query: statistic, bin, ef / recall / cost before
    (fixed ef=100) and after (the table's ef), and the oracle ef."""
    import csv

    score = Score(stat, train)
    table = Table(score(train), train["recall_grid"], T, q=0.8)
    b = np.clip(np.searchsorted(table.edges, score(ev), side="right") - 1, 0, len(table.efs) - 1)
    R, C = ev["recall_grid"], ev["cost_grid"]
    rows = []
    for i in range(len(b)):
        ef_after = table.efs[b[i]]
        j = EF_GRID.index(ef_after)
        hit = np.flatnonzero(R[i] >= T)
        rows.append({"query": i, stat: round(float(ev[stat][i]), 3), "bin": int(b[i]),
                     "ef_before": EF0, "ef_after": ef_after,
                     "recall_before": round(float(R[i, 0]), 2), "recall_after": round(float(R[i, j]), 2),
                     "cost_before": int(C[i, 0]), "cost_after": int(C[i, 0] + (C[i, j] if j > 0 else 0)),
                     "oracle_ef": EF_GRID[hit[0]] if hit.size else f"not by {EF_GRID[-1]}"})
    path = RESULTS_DIR / f"e14_per_query_{dataset}_T{T}_{stat}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = f"{dataset} T={T} {stat}"
        ws.append(list(rows[0]))
        for r in rows:
            ws.append(list(r.values()))
        wb.save(str(path.with_suffix(".xlsx")))
    except ImportError:
        pass
    print(f"saved {path.relative_to(RESULTS_DIR.parent)} (+ .xlsx, {len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    ap.add_argument("--targets", default="0.9,0.8")
    ap.add_argument("--per-query", default=None, choices=STATS,
                    help="also write the per-query before/after table for this statistic")
    args = ap.parse_args()
    dataset = args.dataset
    targets = [float(x) for x in args.targets.split(",")]
    t0 = time.perf_counter()

    ctx = load(dataset)
    metric = Metric(ctx.ds)
    print(f"{dataset}: training queries {TRAIN_SLICE[0]}..{TRAIN_SLICE[1] - 1}, evaluation = E13's 1000")
    train = train_features(dataset, ctx, metric)
    e13 = json.loads((RESULTS_DIR / f"e13_online_stats_{dataset}.json").read_text())
    assert e13["ef_grid"] == EF_GRID and e13["k"] == K and e13["prefix_ef"] == EF0
    pq = e13["per_query"]
    ev = {"edges": np.array(pq["edges"]), "contrast10": np.array(pq["contrast10"]),
          "recall_grid": np.array(pq["recall_grid"]), "cost_grid": np.array(pq["cost_grid"], dtype=float)}
    rng = np.random.default_rng(0)

    results = {}
    for T in targets:
        r = evaluate(dataset, T, train, ev, rng)
        results[str(T)] = r
        if args.per_query:
            write_per_query(dataset, T, args.per_query, train, ev)
        print(f"\n== {dataset}  T={T}  (oracle: {r['oracle']['censored']} of 1000 never reach T by ef={EF_GRID[-1]})")
        print(f"{'policy':>18} {'cost':>7} {'mean':>6} {'p5':>6} {'frac':>6} | fixed ef @ same cost: "
              f"{'p5':>6} {'frac':>6} | {'d p5 [95% CI]':>22} {'d frac [95% CI]':>22}  re-searched / table (hard->easy)")
        for f in r["frontier"]:
            print(f"{'fixed ef=' + str(f['ef']):>18} {f['cost']:>7,.0f} {f['mean']:>6.3f} {f['p5']:>6.3f} {f['frac']:>6.3f}")
        for acc in ["two_stage", "resume"]:
            print(f"  -- cost accounting: {acc} --")
            o = r["oracle"][acc]
            print(f"{'oracle':>18} {o['cost']:>7,.0f} {o['mean']:>6.3f} {o['p5']:>6.3f} {o['frac']:>6.3f}")
            for p in r["policies"]:
                a = p[acc]
                dp, df = a["diff_vs_fixed"]["p5"], a["diff_vs_fixed"]["frac"]
                print(f"{p['stat'] + '/' + p['rule']:>18} {a['cost']:>7,.0f} {a['mean']:>6.3f} {a['p5']:>6.3f} {a['frac']:>6.3f} "
                      f"{'':>22}{a['fixed_ef_at_same_cost']['p5']:>6.3f} {a['fixed_ef_at_same_cost']['frac']:>6.3f}   "
                      f"{dp['point']:>+7.3f} [{dp['ci95'][0]:+.3f},{dp['ci95'][1]:+.3f}] "
                      f"{df['point']:>+7.3f} [{df['ci95'][0]:+.3f},{df['ci95'][1]:+.3f}]  "
                      f"{p['frac_researched']:.2f} {p['table_ef_hard_to_easy']}")

    out = {"dataset": dataset, "k": K, "prefix_ef": EF0, "ef_grid": EF_GRID,
           "train_queries": list(TRAIN_SLICE), "n_train": int(len(train["edges"])),
           "eval": "E13's 1000 external queries (replayed from their ef sweep)",
           "n_bins": N_BINS, "rules": {"q80": "smallest ef with >=80% of the bin at recall>=T",
                                       "mean": "smallest ef with bin-mean recall>=T"},
           "results": results, "elapsed_s": time.perf_counter() - t0}
    save(f"e14_policy_{dataset}", out)
    print(f"elapsed {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
