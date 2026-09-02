"""Analyze the Lucene PIP replication.

The paper reports one operating point per method and a QPS delta. That comparison
is only meaningful if the two methods are at the same recall. This script does
the comparison the paper omits: interpolate the baseline efSearch sweep to
patience's exact recall, and report the cost gap there.

Usage: python analyze_replication.py results/lucene_pip_replication.json
"""

import json
import sys
from pathlib import Path

import numpy as np


def frontier(rows):
    """Baseline sweep sorted by cost, as (recalls, costs)."""
    rows = sorted(rows, key=lambda r: r["visited_avg"])
    return [r["recall"] for r in rows], [r["visited_avg"] for r in rows], rows


def main(path):
    d = json.loads(Path(path).read_text())
    sweep = d["sweep"]
    ks = sorted({r["k"] for r in sweep})

    print(f"engine: {d['engine']}")
    print(f"patience defaults: {d['patience_defaults']}")
    print(f"queries: {d['n_queries']}  index: {d['index']}\n")

    for k in ks:
        base = [r for r in sweep if r["k"] == k and r["method"] == "baseline"]
        pat = [r for r in sweep if r["k"] == k and r["method"] == "patience"]
        R, C, brows = frontier(base)

        print("=" * 88)
        print(f"k = {k}")
        print("=" * 88)
        print("  raw sweep (ef == k is the true Lucene default operating point)")
        print(f"    {'ef':>6} {'Δ':>5} | {'base R':>8} {'base cost':>10} {'base QPS':>9}"
              f" | {'PIP R':>8} {'PIP cost':>10} {'PIP QPS':>9}")
        for b, p in zip(sorted(base, key=lambda r: r["ef"]),
                        sorted(pat, key=lambda r: r["ef"])):
            star = "  <- Lucene default" if b["ef"] == k else ""
            print(f"    {b['ef']:>6} {p['delta']:>5} | {b['recall']:>8.4f}"
                  f" {b['visited_avg']:>10.1f} {b['qps']:>9.1f}"
                  f" | {p['recall']:>8.4f} {p['visited_avg']:>10.1f} {p['qps']:>9.1f}{star}")

        print("\n  ISO-RECALL: baseline cost interpolated to PIP's exact recall")
        print(f"    {'ef':>6} | {'PIP R':>8} {'PIP cost':>10} {'base cost':>10}"
              f" {'saving':>9}")
        savings = []
        for p in sorted(pat, key=lambda r: r["ef"]):
            rec, cost = p["recall"], p["visited_avg"]
            if rec < R[0] or rec > R[-1]:
                print(f"    {p['ef']:>6} | {rec:>8.4f} {cost:>10.1f} "
                      f"{'outside baseline sweep':>21}")
                continue
            eq = float(np.interp(rec, R, C))
            sav = (eq - cost) / eq * 100
            savings.append(sav)
            print(f"    {p['ef']:>6} | {rec:>8.4f} {cost:>10.1f} {eq:>10.1f} {sav:>+8.1f}%")
        if savings:
            print(f"    -> median iso-recall saving: {np.median(savings):+.1f}%"
                  f"   (max {max(savings):+.1f}%)")

        # The paper's own style of comparison: PIP's best point vs the baseline's
        # most expensive point, ignoring that they sit at different recall.
        pbest = max(pat, key=lambda r: r["recall"])
        bmax = max(base, key=lambda r: r["visited_avg"])
        print(f"\n  PAPER-STYLE (unmatched recall) at k={k}:")
        print(f"    'PIP {pbest['recall']:.4f} @ {pbest['visited_avg']:.0f} vs "
              f"HNSW {bmax['recall']:.4f} @ {bmax['visited_avg']:.0f}' "
              f"=> claims {(1 - pbest['visited_avg'] / bmax['visited_avg']) * 100:.0f}% fewer visits")
        print("    (but those are different recall levels — see iso-recall above)\n")

        # Where does PIP move the budget? Split by baseline difficulty at the
        # closest matched-recall pair.
        p = min(pat, key=lambda r: abs(r["recall"] - 0.99)) if k == 10 else \
            max(pat, key=lambda r: r["recall"])
        b = min(base, key=lambda r: abs(r["recall"] - p["recall"]))
        br = np.array(b["recalls"]); bv = np.array(b["visited"], float)
        pr = np.array(p["recalls"]); pv = np.array(p["visited"], float)
        order = np.argsort(br, kind="stable")
        n = len(br)
        buckets = [("hardest 10%", order[:n // 10]),
                   ("middle 40%", order[n // 10:n // 2]),
                   ("easiest 50%", order[n // 2:])]
        print(f"  BUDGET REALLOCATION: PIP ef={p['ef']} (R={p['recall']:.4f}) vs "
              f"baseline ef={b['ef']} (R={b['recall']:.4f})")
        print(f"    {'bucket':<14}{'base cost':>10}{'PIP cost':>10}{'Δcost':>9}"
              f"{'base R':>9}{'PIP R':>8}")
        for name, ix in buckets:
            print(f"    {name:<14}{bv[ix].mean():>10.0f}{pv[ix].mean():>10.0f}"
                  f"{(pv[ix].mean() / bv[ix].mean() - 1) * 100:>+8.1f}%"
                  f"{br[ix].mean():>9.3f}{pr[ix].mean():>8.3f}")
        print(f"    {'OVERALL':<14}{bv.mean():>10.0f}{pv.mean():>10.0f}"
              f"{(pv.mean() / bv.mean() - 1) * 100:>+8.1f}%"
              f"{br.mean():>9.3f}{pr.mean():>8.3f}")
        print(f"    latency tail: base p99={np.percentile(bv, 99):.0f} "
              f"PIP p99={np.percentile(pv, 99):.0f} "
              f"({np.percentile(pv, 99) / np.percentile(bv, 99):.2f}x)")
        print(f"    recall tail:  base p5={np.percentile(br, 5):.3f} "
              f"PIP p5={np.percentile(pr, 5):.3f}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "results/lucene_pip_replication.json")
