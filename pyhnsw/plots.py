"""Charts for the meeting deck, generated from results/*.json.

Usage:  .venv/bin/python -m pyhnsw.plots
Writes PNGs into results/figs/.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = RESULTS / "figs"

# reference palette (light mode)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 7,
    }
)


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def savefig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/{name}.png")


def fig_e1():
    data = load("e1_validation_sift1m")
    rows = data["rows"]
    efs = [r["ef"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(efs, [r["faiss_recall"] for r in rows], "-o", color=BLUE, label="Faiss C++")
    ax.plot(
        efs, [r["py_recall"] for r in rows], "--s", color=ORANGE,
        markerfacecolor="none", label="pyhnsw (ours)",
    )
    ax.set_xscale("log")
    ax.set_xticks(efs, [str(e) for e in efs])
    ax.set_xlabel("efSearch")
    ax.set_ylabel(f"recall@{data['k']}")
    ax.set_title("E1 — Python engine matches Faiss exactly (SIFT1M)")
    ax.legend()
    savefig(fig, "e1_validation")


def fig_e2(dataset, title):
    data = load(f"e2_pip_{dataset}")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    b = data["baseline"]
    p = data["pip_sweep"]
    ax.plot(
        [r["n_dist_avg"] for r in b], [r["recall_avg"] for r in b],
        "-o", color=BLUE, label="HNSW baseline (ef sweep)",
    )
    ax.plot(
        [r["n_dist_avg"] for r in p], [r["recall_avg"] for r in p],
        "-s", color=ORANGE, label="PIP γ=0.95 Δ=30 (ef sweep)",
    )
    g = data["pip_grid"]
    ax.scatter(
        [r["n_dist_avg"] for r in g], [r["recall_avg"] for r in g],
        marker="^", s=55, facecolors="none", edgecolors=AQUA, linewidths=1.6,
        label="PIP grid @ ef=320 (γ, Δ varied)", zorder=3,
    )
    for r in g:
        if r["delta"] == 50 or (r["gamma"], r["delta"]) == (0.8, 10):
            ax.annotate(
                f"γ={r['gamma']}, Δ={r['delta']}",
                (r["n_dist_avg"], r["recall_avg"]),
                textcoords="offset points", xytext=(6, -10), fontsize=8, color=INK2,
            )
    ax.set_xlabel("distance computations per query (avg)")
    ax.set_ylabel(f"recall@{data['k']}")
    ax.set_title(title)
    ax.legend(loc="lower right")
    savefig(fig, f"e2_tradeoff_{dataset}")


def fig_e3():
    data = load("e3_saturation_sift1m")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, (label, color) in zip(axes, [("easy", BLUE), ("hard", ORANGE)]):
        q = data["queries"][label]
        hops = np.arange(len(q["phi"]))
        ax.plot(hops, q["phi"], color=color, label="φ (top-k overlap)")
        thirty = np.array(q["counter"]) >= 30
        if thirty.any():
            first = int(np.argmax(thirty))
            ax.axvline(first, color=MUTED, linestyle=":", linewidth=1.5)
            ax.annotate(
                "PIP (Δ=30)\nwould stop here", (first, 0.35),
                textcoords="offset points", xytext=(8, 0), fontsize=8, color=INK2,
            )
        ax.set_title(f"{label} query")
        ax.set_xlabel("expanded nodes (hops)")
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("φ = |top-k ∩ prev top-k| / k")
    axes[0].legend(loc="lower right")
    fig.suptitle("E3 — saturation of the top-k set during traversal (SIFT1M, ef=320)", y=1.02)
    savefig(fig, "e3_saturation")


def fig_e4():
    data = load("e4_recall_skew_glove100")
    cfgs = data["configs"]
    order = [
        ("baseline_ef40", "HNSW ef=40", BLUE),
        ("baseline_ef160", "HNSW ef=160", AQUA),
        ("pip_ef320", "PIP γ=0.95 Δ=30", ORANGE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    bins = np.arange(0, 1.15, 0.1) - 0.05
    for ax, (key, label, color) in zip(axes, order):
        c = cfgs[key]
        ax.hist(c["recalls"], bins=bins, color=color, edgecolor=SURFACE, linewidth=0.8)
        ax.set_title(
            f"{label}\navg={c['recall_avg']:.3f}  p5={c['recall_p5']:.2f}  "
            f"cost={c['n_dist_avg']:.0f}",
            fontsize=10,
        )
        ax.set_xlabel(f"per-query recall@{data['k']}")
    axes[0].set_ylabel("queries")
    fig.suptitle(
        "E4 — the same setting treats queries very differently (GloVe-100, 1000 queries)",
        y=1.06,
    )
    savefig(fig, "e4_recall_skew")


def fig_e5():
    e5 = load("e5_adaef_glove100")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    efs = e5["ada"]["est_efs"]
    ax.hist([e for e in efs if e is not None], bins=40, color=AQUA, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlabel("ef assigned by Ada-ef")
    ax.set_ylabel("queries")
    ax.set_yscale("log")
    ax.set_title(
        f"E5 — per-query ef chosen by Ada-ef (GloVe-100, target recall "
        f"{e5['target_recall']}, WAE={e5['wae']:.0f})"
    )
    savefig(fig, "e5_ef_distribution")


def fig_headline():
    """Frontier: baseline curve, PIP curve, Ada-ef and hybrid points (GloVe)."""
    e2 = load("e2_pip_glove100")
    e6 = load("e6_head_to_head_glove100")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    b, p = e2["baseline"], e2["pip_sweep"]
    ax.plot(
        [r["n_dist_avg"] for r in b], [r["recall_avg"] for r in b],
        "-o", color=BLUE, label="HNSW baseline (ef sweep)",
    )
    ax.plot(
        [r["n_dist_avg"] for r in p], [r["recall_avg"] for r in p],
        "-s", color=ORANGE, label="PIP γ=0.95 Δ=30 (ef sweep)",
    )
    marks = [
        ("ada", "Ada-ef (target 0.95)", AQUA, "D"),
        ("hybrid_scaled", "Hybrid: Ada-ef + scaled patience", YELLOW, "^"),
    ]
    for key, label, color, marker in marks:
        c = e6["configs"][key]
        ax.scatter(
            [c["n_dist_avg"]], [c["recall_avg"]], marker=marker, s=110,
            color=color, edgecolors=INK, linewidths=0.8, zorder=4, label=label,
        )
        ax.annotate(
            f"avg {c['recall_avg']:.3f}\np5  {c['recall_p5']:.2f}",
            (c["n_dist_avg"], c["recall_avg"]),
            textcoords="offset points", xytext=(10, -18), fontsize=8, color=INK2,
        )
    ax.axhline(0.95, color=MUTED, linestyle=":", linewidth=1.3)
    ax.annotate("target recall 0.95", (ax.get_xlim()[1], 0.95),
                fontsize=8, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel("distance computations per query (avg)")
    ax.set_ylabel(f"recall@{e2['k']}")
    ax.set_title("GloVe-100 — cost/recall frontier: fixed ef vs PIP vs Ada-ef vs hybrid")
    ax.legend(loc="lower right", fontsize=9)
    savefig(fig, "headline_frontier_glove100")


def fig_e6_percentiles():
    e6 = load("e6_head_to_head_glove100")
    labels = [
        ("pip", "PIP\nγ=0.95 Δ=30", ORANGE),
        ("ada", "Ada-ef\ntarget 0.95", AQUA),
        ("hybrid_fixed", "Hybrid\nΔ=400", YELLOW),
        ("hybrid_scaled", "Hybrid\nΔ=0.5·ef", "#e87ba4"),
    ]
    metrics = ["recall_avg", "recall_p5", "recall_p1"]
    metric_names = ["average", "5th percentile", "1st percentile"]
    x = np.arange(len(metrics))
    width = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    ax = axes[0]
    for i, (key, label, color) in enumerate(labels):
        c = e6["configs"][key]
        vals = [c[m] for m in metrics]
        bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.92, color=color,
                      label=label.replace("\n", " "))
        for b_, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b_.get_x() + b_.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8, color=INK2)
    ax.axhline(e6["target_recall"], color=MUTED, linestyle=":", linewidth=1.3)
    ax.set_xticks(x, metric_names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(f"recall@{e6['k']}")
    ax.set_title("recall: average and tail")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    for i, (key, label, color) in enumerate(labels):
        c = e6["configs"][key]
        ax.bar(i, c["n_dist_avg"], 0.62, color=color)
        ax.annotate(f"{c['n_dist_avg']:.0f}", (i, c["n_dist_avg"]),
                    ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(labels)), [l for _, l, _ in labels], fontsize=8)
    ax.set_ylabel("distance computations / query")
    ax.set_title("cost")
    fig.suptitle("E6 — head-to-head on GloVe-100 (1000 queries)", y=1.04)
    savefig(fig, "e6_head_to_head")


def fig_e7():
    data = load("e7_hardness_glove100")
    pq = data["per_query"]
    recalls = np.array(pq["recalls"])
    levels = [("<=0.25\n(hard)", 0.0, 0.25), ("~0.5", 0.26, 0.5),
              ("~0.75", 0.51, 0.8), ("~1.0\n(easy)", 0.81, 1.0)]
    panels = [
        ("rel_contrast", "how much GT stands out vs random\n(random dist ÷ top-10 dist)", "ratio"),
        ("gt_pairwise", "GT spread: mean distance\nbetween GT members", "cosine distance"),
        ("q_gt_mean", "query → its GT:\nmean distance", "cosine distance"),
        ("contrast", "neighbor profile flatness\n(100th ÷ 1st distance)", "ratio"),
        ("gt_indegree", "GT popularity in graph\n(mean in-degree) — no signal", "links"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    fig.subplots_adjust(hspace=0.55, wspace=0.3)
    corr = data["spearman_recall_vs_metric"]
    for ax, (key, title, ylab) in zip(axes.ravel(), panels):
        vals = np.array(pq[key])
        groups = [vals[(recalls >= lo) & (recalls <= hi)] for _, lo, hi in levels]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color=INK, linewidth=1.6))
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.55)
            patch.set_edgecolor(BASE)
        ax.set_xticks(range(1, 5), [name for name, *_ in levels], fontsize=8.5)
        ax.set_title(f"{title}\ncorr with recall: {corr[key]:+.2f}", fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=8.5)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        "E7 — what makes a query hard? GT-100 structure by hardness level (GloVe-100, 1000 queries)",
        y=0.99, fontsize=13,
    )
    savefig(fig, "e7_hardness")


def fig_e7_key():
    """Two-panel version for the slide: GT spread and contrast vs random."""
    data = load("e7_hardness_glove100")
    pq = data["per_query"]
    recalls = np.array(pq["recalls"])
    levels = [("<=0.25\n(hard)", 0.0, 0.25), ("~0.5", 0.26, 0.5),
              ("~0.75", 0.51, 0.8), ("~1.0\n(easy)", 0.81, 1.0)]
    panels = [
        ("gt_pairwise", "GT spread:\nmean distance between the 100 true neighbors",
         "cosine distance"),
        ("rel_contrast", "contrast vs random:\nmean distance to random vectors ÷ to top-10",
         "ratio"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    fig.subplots_adjust(wspace=0.25, top=0.76)
    corr = data["spearman_recall_vs_metric"]
    for ax, (key, title, ylab) in zip(axes, panels):
        vals = np.array(pq[key])
        groups = [vals[(recalls >= lo) & (recalls <= hi)] for _, lo, hi in levels]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color=INK, linewidth=1.8))
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.55)
            patch.set_edgecolor(BASE)
        ax.set_xticks(range(1, 5), [name for name, *_ in levels], fontsize=11)
        ax.set_title(f"{title}\ncorr with recall: {corr[key]:+.2f}", fontsize=12)
        ax.set_ylabel(ylab, fontsize=11)
    fig.suptitle(
        "E7 — hard queries' true neighbors are far apart and barely stand out "
        "from random vectors (GloVe-100, 1000 queries)",
        y=1.0, fontsize=13,
    )
    savefig(fig, "e7_hardness_key")


def fig_e10():
    """E10 — graph distance among the GT (whiteboard): d_G vs d_vec."""
    data = load("e10_gt_graphdist_glove100")
    e7 = load("e7_hardness_glove100")
    pq = data["per_query"]
    recalls = np.array(pq["recalls"])
    hops = np.array(pq["gt_hops_mean"])
    far = np.array(pq["gt_hops_far"])
    dvec = np.array(e7["per_query"]["gt_pairwise"])
    corr = data["spearman_recall_vs_metric"]
    H = data["max_hops"]
    levels = [("<=0.25\n(hard)", 0.0, 0.25), ("~0.5", 0.26, 0.5),
              ("~0.75", 0.51, 0.8), ("~1.0\n(easy)", 0.81, 1.0)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.32, top=0.78)

    def box(ax, vals, title, ylab, key):
        groups = [vals[(recalls >= lo) & (recalls <= hi)] for _, lo, hi in levels]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color=INK, linewidth=1.8))
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.55)
            patch.set_edgecolor(BASE)
        ax.set_xticks(range(1, 5), [name for name, *_ in levels], fontsize=10)
        ax.set_title(f"{title}\ncorr with recall: {corr[key]:+.2f}", fontsize=11)
        ax.set_ylabel(ylab, fontsize=10)

    box(axes[0], hops, "avg-dist: mean shortest-path hops\nbetween GT-100 pairs (4950 pairs)",
        "hops (level-0 graph)", "gt_hops_mean")
    box(axes[1], far, f"fraction of GT pairs more than\n{H} hops apart",
        "fraction of pairs", "gt_hops_far")

    ax = axes[2]
    hard = recalls <= 0.25
    easy = recalls >= 0.81
    mid = ~hard & ~easy
    ax.scatter(dvec[mid], hops[mid], s=9, color=MUTED, alpha=0.35, label="in between")
    ax.scatter(dvec[easy], hops[easy], s=11, color=BLUE, alpha=0.6, label="easy (recall ≥ 0.81)")
    ax.scatter(dvec[hard], hops[hard], s=13, color=ORANGE, alpha=0.8, label="hard (recall ≤ 0.25)")
    rho = spearman_np(dvec, hops)
    ax.set_xlabel("d_vec: mean cosine distance between GT pairs", fontsize=10)
    ax.set_ylabel("d_G: mean hops between GT pairs", fontsize=10)
    ax.set_title(f"vector spread vs graph spread\nSpearman ρ = {rho:+.2f}", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "E10 — hard queries' true neighbors are many hops apart in the HNSW graph "
        "(GloVe-100, 1000 queries, GT-100)",
        y=1.0, fontsize=13,
    )
    savefig(fig, "e10_gt_graphdist")


def fig_e11(name="e11_avgdist_glove100", out="e11_avgdist"):
    """Exact avg-dist (whiteboard method, no hop cap). `name` selects the
    results file; hardness bins and GT size are read from it."""
    data = load(name)
    pq = data["per_query"]
    recalls = np.array(pq["recalls"])
    avg = np.array(pq["avg_dist"], dtype=float)
    mx = np.array(pq["max_dist"], dtype=float)
    hist = data["pair_hop_histogram"]
    if "levels_def" in data:
        inclusive = data["levels_def"][0]["inclusive"]
        levels = [(lv["level"].replace(" (", "\n("), lv["lo"], lv["hi"])
                  for lv in data["levels_def"]]
    else:
        inclusive = True
        levels = [("<=0.25\n(hard)", 0.0, 0.25), ("~0.5", 0.26, 0.5),
                  ("~0.75", 0.51, 0.8), ("~1.0\n(easy)", 0.81, 1.0)]
    k_gt = data.get("k_gt", 100)
    n_pairs = k_gt * (k_gt - 1) // 2
    label = data.get("hardness_from", "recall@10, HNSW ef=160")

    def sel(lo, hi):
        return (recalls >= lo) & ((recalls <= hi) if inclusive else (recalls < hi))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.32, top=0.78)

    def box(ax, vals, title, ylab):
        groups = [vals[sel(lo, hi)] for _, lo, hi in levels]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.55,
                        medianprops=dict(color=INK, linewidth=1.8))
        for patch in bp["boxes"]:
            patch.set_facecolor(BLUE)
            patch.set_alpha(0.55)
            patch.set_edgecolor(BASE)
        ax.set_xticks(range(1, 5), [name for name, *_ in levels], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylab, fontsize=10)

    box(axes[0], avg,
        f"avg-dist: mean shortest-path hops between\nGT-{k_gt} pairs (exact, {n_pairs:,} pairs)"
        f"\ncorr with recall: {data['spearman_recall_vs_avg_dist']:+.2f}",
        "hops (level-0 graph)")
    ax = axes[1]
    hop_vals = sorted(set(int(v) for v in mx))
    shades = ([BLUE, AQUA, YELLOW, ORANGE, "#b03a2e", "#6c3483", "#1b4f72", "#7b7d7d"]
              * 2)[: len(hop_vals)]
    bottom = np.zeros(len(levels))
    for hv, col in zip(hop_vals, shades):
        frac = np.array([np.mean(mx[sel(lo, hi)] == hv) for _, lo, hi in levels])
        ax.bar(range(len(levels)), frac, bottom=bottom, color=col, width=0.6,
               label=f"{hv} hops")
        for x, (b, f) in enumerate(zip(bottom, frac)):
            if f >= 0.08:
                ax.annotate(f"{f:.0%}", (x, b + f / 2), ha="center", va="center",
                            fontsize=9, color="white")
        bottom += frac
    ax.set_xticks(range(len(levels)), [name for name, *_ in levels], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of queries", fontsize=10)
    ax.set_title("max-dist: how far apart is the\nfarthest pair of true neighbors?", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, -0.14), ncol=len(hop_vals))

    ax = axes[2]
    keys = [k for k in hist if k != "inf"]
    xs = [int(k) for k in keys]
    ys = np.array([hist[k] for k in keys], dtype=float)
    ys /= ys.sum()
    ax.bar(xs, ys, color=BLUE, alpha=0.75, width=0.7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1%}", (x, y), ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(xs)
    ax.set_xlabel("hops between a pair of true neighbors", fontsize=10)
    ax.set_ylabel("fraction of all pairs", fontsize=10)
    unreach = hist.get("inf", 0)
    ax.set_title(f"exact pair-distance distribution\n(all queries pooled; unreachable pairs: {unreach})",
                 fontsize=11)

    fig.suptitle(
        "avg-dist: hard queries' true neighbors are farther apart "
        f"in the HNSW graph (GloVe-100, 1000 queries; hardness = {label})",
        y=1.0, fontsize=13,
    )
    savefig(fig, out)


def fig_pip_iso_recall(k=1000):
    """Lucene PIP replication at one k: the saving the paper's comparison
    method reports vs the saving at matched recall, and the frontier behind it."""
    path = RESULTS.parent / "patience-in-proximity" / "results" / "lucene_pip_replication.json"
    d = json.loads(path.read_text())
    sweep = d["sweep"]
    base = sorted([r for r in sweep if r["k"] == k and r["method"] == "baseline"],
                  key=lambda r: r["visited_avg"])
    pat = sorted([r for r in sweep if r["k"] == k and r["method"] == "patience"],
                 key=lambda r: r["ef"])
    R = [r["recall"] for r in base]
    C = [r["visited_avg"] for r in base]
    sav = []
    for p_ in pat:
        if R[0] <= p_["recall"] <= R[-1]:
            eq = float(np.interp(p_["recall"], R, C))
            sav.append((eq - p_["visited_avg"]) / eq * 100)
    pbest = max(pat, key=lambda r: r["recall"])
    bmax = max(base, key=lambda r: r["visited_avg"])
    paper = (1 - pbest["visited_avg"] / bmax["visited_avg"]) * 100
    med, best = float(np.median(sav)), float(max(sav))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                  gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.3})

    # --- left: two bars
    ax.bar([0], [paper], 0.55, color=ORANGE)
    ax.bar([1], [med], 0.55, color=BLUE)
    ax.scatter([1], [best], marker="_", s=600, color=INK, linewidths=2, zorder=3)
    ax.annotate(f"{paper:.0f}%", (0, paper), ha="center", va="bottom", fontsize=12,
                color=INK2, xytext=(0, 3), textcoords="offset points")
    ax.annotate(f"median {med:+.1f}%\nbest {best:+.1f}%", (1, max(best, 0)),
                ha="center", va="bottom", fontsize=10.5, color=INK2,
                xytext=(0, 5), textcoords="offset points")
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xticks([0, 1], ["paper's comparison method\n(PIP best vs HNSW most\n"
                           "expensive; recall not matched)",
                           "at matched recall\n(HNSW ef interpolated\nto PIP's recall)"],
                  fontsize=9.5)
    ax.set_ylabel("reduction in nodes visited vs plain HNSW (%)", fontsize=10)
    ax.set_ylim(-5, max(paper, best) * 1.25)
    ax.set_title(f"k = {k}: the saving PIP appears to give\nvs the saving it actually gives",
                 fontsize=11)

    # --- right: the frontier
    ax2.plot([r["visited_avg"] for r in base], R, "-o", color=BLUE, markersize=5,
             label="plain HNSW (efSearch sweep)")
    ax2.plot([r["visited_avg"] for r in pat], [r["recall"] for r in pat], "s",
             color=ORANGE, markersize=6.5, markerfacecolor="white", markeredgewidth=1.8,
             label="PIP (same efSearch values)")
    for r in pat:
        ax2.annotate(f"ef={r['ef']}", (r["visited_avg"], r["recall"]), fontsize=8,
                     color=INK2, xytext=(6, -10), textcoords="offset points")
    ax2.set_xlabel("nodes visited / query", fontsize=10)
    ax2.set_ylabel(f"recall@{k}", fontsize=10)
    lo = min(r["recall"] for r in pat + base)
    ax2.set_ylim(lo - 0.01, 1.003)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.set_title(f"k = {k}: PIP sits on the plain-HNSW frontier —\n"
                  "each PIP point costs what plain HNSW costs at that recall", fontsize=11)

    fig.suptitle(
        f"PIP in Lucene 10.5, SIFT1M, 1000 queries, k = {k} — all percentages are from "
        "our sweep; the paper reports 39–42% fewer visits on BEIR (p. 406)",
        y=1.03, fontsize=12,
    )
    savefig(fig, f"pip_lucene_iso_recall_k{k}")


def spearman_np(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


DATASET_TITLES = {"glove100": "GloVe-100", "sift1m": "SIFT1M"}


def fig_e8_frontier(dataset):
    """Recall/cost frontier at k=10 (from E2) vs k=100 vs k=1000 (E8)."""
    title = DATASET_TITLES[dataset]
    e2 = load(f"e2_pip_{dataset}")
    panels = [(10, e2["baseline"], e2["pip_sweep"], "PIP γ=0.95 Δ=30 (ef sweep)")]
    for k in (100, 1000):
        e8 = load(f"e8_ksweep_{dataset}_k{k}")
        panels.append((k, e8["baseline"], e8["pip"], "PIP γ=0.95, Δ varied"))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, (k, base, pip, pip_label) in zip(axes, panels):
        ax.plot([r["n_dist_avg"] for r in base], [r["recall_avg"] for r in base],
                "-o", color=BLUE, label="HNSW baseline (ef sweep)")
        if k == 10:
            ax.plot([r["n_dist_avg"] for r in pip], [r["recall_avg"] for r in pip],
                    "-s", color=ORANGE, label=pip_label)
        else:
            loose = [r for r in pip if r["gamma"] <= 0.95]
            tight, seen = [], set()
            for r in pip:  # γ=0.995 / 0.999 can stop identically — plot once
                key = (round(r["n_dist_avg"]), round(r["recall_avg"], 4))
                if r["gamma"] > 0.95 and key not in seen:
                    seen.add(key)
                    tight.append(r)
            ax.scatter([r["n_dist_avg"] for r in loose], [r["recall_avg"] for r in loose],
                       marker="s", s=70, color=ORANGE, edgecolors=INK,
                       linewidths=0.7, zorder=4, label="PIP γ=0.95, Δ varied")
            ax.scatter([r["n_dist_avg"] for r in tight], [r["recall_avg"] for r in tight],
                       marker="D", s=70, color=AQUA, edgecolors=INK,
                       linewidths=0.7, zorder=4,
                       label="PIP Lucene defaults (γ≥0.995, Δ=0.3k)")
            for r in loose:
                ax.annotate(f"Δ={r['delta']}", (r["n_dist_avg"], r["recall_avg"]),
                            textcoords="offset points", xytext=(6, -11),
                            fontsize=8, color=INK2)
            for r in tight:
                ax.annotate(f"γ={r['gamma']}", (r["n_dist_avg"], r["recall_avg"]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, color=INK2)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("distance computations / query")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("recall@k")
    fig.suptitle(
        f"E8 — the PIP story flips with k ({title}, 1000 queries): "
        "at k=10 PIP caps recall; at large k it matches the baseline for less cost",
        y=1.04,
    )
    savefig(fig, f"e8_frontier_{dataset}")


def _closest_baseline(base, target_recall):
    """Baseline sweep entry whose recall is closest to target from above."""
    above = [r for r in base if r["recall_avg"] >= target_recall]
    pool = above or base
    return min(pool, key=lambda r: abs(r["recall_avg"] - target_recall))


def fig_e8_hist(dataset):
    """E4-style per-query recall histograms at k=100 / k=1000: does the left
    tail survive at large k? PIP (best Δ) vs recall-matched baseline."""
    title = DATASET_TITLES[dataset]
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2), sharey="row")
    bins = np.arange(0, 1.15, 0.1) - 0.05
    for row, k in enumerate((100, 1000)):
        e8 = load(f"e8_ksweep_{dataset}_k{k}")
        pip = max(e8["pip"], key=lambda r: r["recall_avg"])
        base = _closest_baseline(e8["baseline"], pip["recall_avg"])
        for col, (r, label, color) in enumerate([
            (base, f"HNSW ef={base['ef']}", BLUE),
            (pip, f"PIP γ={pip['gamma']} Δ={pip['delta']} @ ef={pip['ef']}", ORANGE),
        ]):
            ax = axes[row][col]
            ax.hist(r["recalls"], bins=bins, color=color, edgecolor=SURFACE,
                    linewidth=0.8)
            ax.set_title(
                f"k={k} — {label}\navg={r['recall_avg']:.3f}  "
                f"p5={r['recall_p5']:.2f}  cost={r['n_dist_avg']:.0f}",
                fontsize=10,
            )
            ax.set_xlabel(f"per-query recall@{k}")
        axes[row][0].set_ylabel("queries")
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(
        f"E8 — per-query recall at large k ({title}, 1000 queries)", y=0.99
    )
    savefig(fig, f"e8_hist_{dataset}")


def fig_e9_asymptote():
    data = load("e9_hardness_causal_glove100")
    rows = data["asymptote"]
    efs = [r["ef"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(efs, [r["easy_recall"] for r in rows], "-o", color=BLUE,
            label="easy queries (control, n=20)")
    ax.plot(efs, [r["hard_recall"] for r in rows], "-s", color=ORANGE,
            label=f"hard bin (recall ≤ 0.25 @ ef=160, n={data['bin_sizes']['<=0.25 (hard)']})")
    for r in rows:
        ax.annotate(f"{r['hard_recall']:.2f}", (r["ef"], r["hard_recall"]),
                    textcoords="offset points", xytext=(0, -14), fontsize=8,
                    color=INK2, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(efs, [str(e) for e in efs])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("ef (search beam width)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("E9 — can a bigger budget fix hard queries? (GloVe-100)")
    savefig(fig, "e9_asymptote")


def fig_e9_oracle():
    data = load("e9_hardness_causal_glove100")
    o = data["oracle"]
    pairs = [("ef=160", "normal_ef160", "oracle_ef160"),
             ("ef=640", "normal_ef640", "oracle_ef640")]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax = axes[0]
    x = np.arange(len(pairs))
    for i, (key_idx, label, color) in enumerate(
        [(1, "normal entry (descend from top)", BLUE),
         (2, "oracle entry (start at true NN)", AQUA)]
    ):
        vals = [o[p[key_idx]]["recall_avg"] for p in pairs]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.33, color=color, label=label)
        for b_, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b_.get_x() + b_.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hard-bin recall@10 (avg)")
    ax.set_title("does teleporting to the answer help?")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    normal = np.array(o["normal_ef160"]["recalls"])
    oracle = np.array(o["oracle_ef160"]["recalls"])
    jitter = (np.random.default_rng(1).random(len(normal)) - 0.5) * 0.04
    ax.scatter(normal + jitter, oracle + jitter, s=42, color=ORANGE, alpha=0.75,
               edgecolors=INK, linewidths=0.5)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1.2)
    ax.set_xlabel("recall@10, normal entry (ef=160)")
    ax.set_ylabel("recall@10, oracle entry (ef=160)")
    ax.set_title("per-query: points above the line = routing helped")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.suptitle("E9 — oracle-entry test on the hard bin (GloVe-100)", y=1.05)
    savefig(fig, "e9_oracle")


def main():
    fig_e1()
    fig_e2("sift1m", "E2 — PIP recall/cost tradeoff (SIFT1M)")
    fig_e2("glove100", "E2 — PIP recall/cost tradeoff (GloVe-100)")
    fig_e3()
    fig_e4()
    fig_e5()
    fig_headline()
    fig_e6_percentiles()


if __name__ == "__main__":
    main()


def fig_e12_labels(label="k10", out=None):
    """E12a — internal-query hardness labels vs the external-query labels at
    the same operating point (k=10, ef=160 by default)."""
    d = load("e12_hardness_internal_glove100")
    c = d["configs"][label]
    k, ef = c["k"], c["ef"]
    r_int, n_int = np.array(c["recalls"]), np.array(c["n_dists"])
    e4 = load("e4_recall_skew_glove100")["configs"][f"baseline_ef{ef}"]
    r_ext = np.array(e4["recalls"])
    n_ext = np.array(load("e9_hardness_causal_glove100")[f"cost_n_dists_ef{ef}"])
    bins = c["bins"]
    inclusive = bins[0]["inclusive"]

    def sel(r, lo, hi):
        return (r >= lo) & ((r <= hi) if inclusive else (r < hi))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    w = 0.38
    series = [("internal (nodes)", BLUE, r_int, n_int),
              ("external (test set)", ORANGE, r_ext, n_ext)]

    # (a) recall distribution
    ax = axes[0, 0]
    vals = np.round(np.arange(0, k + 1) / k, 2)
    x = np.arange(len(vals))
    for j, (name, col, r, _) in enumerate(series):
        cnt = np.array([(np.isclose(r, v)).sum() for v in vals])
        ax.bar(x + (j - 0.5) * w, cnt, w, color=col, label=name)
        ax.text(x[-1] + (j - 0.5) * w, cnt[-1], f"{cnt[-1]}", ha="center",
                va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [f"{v:g}" for v in vals])
    ax.set_xlabel(f"recall@{k} at ef={ef}")
    ax.set_ylabel("queries")
    ax.set_title(f"(a) recall@{k} distribution, 1000 queries each", loc="left")
    ax.legend(frameon=False)

    # (b) bin counts
    ax = axes[0, 1]
    x = np.arange(len(bins))
    for j, (name, col, r, _) in enumerate(series):
        cnt = [int(sel(r, b["lo"], b["hi"]).sum()) for b in bins]
        bars = ax.bar(x + (j - 0.5) * w, cnt, w, color=col, label=name)
        for b_, v in zip(bars, cnt):
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v}", ha="center",
                    va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [b["level"].replace(" (", "\n(") for b in bins])
    ax.set_ylabel("queries")
    ax.set_title("(b) hardness bins", loc="left")
    ax.legend(frameon=False)

    # (c) cost by bin
    ax = axes[1, 0]
    for j, (name, col, r, n) in enumerate(series):
        m = [n[sel(r, b["lo"], b["hi"])].mean() if sel(r, b["lo"], b["hi"]).any()
             else 0 for b in bins]
        bars = ax.bar(x + (j - 0.5) * w, m, w, color=col, label=name)
        for b_, v in zip(bars, m):
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [b["level"].replace(" (", "\n(") for b in bins])
    ax.set_ylabel("distance computations (mean)")
    ax.set_title("(c) search cost by bin", loc="left")
    ax.set_ylim(0, max(n_int.mean(), n_ext.mean()) * 1.9)
    ax.legend(frameon=False, loc="upper right")

    # (d) adjacency: how many of the 16 nearest true neighbors are 1 hop away
    ax = axes[1, 1]
    h = np.array(d["adjacency"]["summary"]["top16_direct"]["hist"][:17])
    ax.bar(np.arange(17), h, 0.7, color=BLUE)
    mean = d["adjacency"]["summary"]["top16_direct"]["mean"]
    ax.axvline(mean, color=INK2, lw=1, ls="--")
    ax.text(mean + 0.3, h.max() * 1.04, f"mean {mean:.1f} of 16", color=INK2,
            fontsize=9)
    ax.set_ylim(0, h.max() * 1.15)
    ax.set_xticks(np.arange(0, 17, 2))
    ax.set_xlabel("of a node's 16 nearest true neighbors, how many are direct out-edges")
    ax.set_ylabel("nodes")
    ax.set_title("(d) are the first M=16 neighbors one hop away?", loc="left")

    fig.suptitle(
        f"E12a — internal queries (index nodes, self excluded) vs external queries, "
        f"GloVe-100, k={k} ef={ef}; self not found in {c['self_not_found']} / 1000",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out or f"e12_hardness_internal_{label}")


# ---------------------------------------------------------------- E12 helpers
E12_LABELS = ["k10", "k10_ef40", "k100", "k1000"]


def _e12_external(k, ef):
    """Per-query (recalls, n_dists-or-None) of the 1000 EXTERNAL queries at
    the same operating point as an E12 config."""
    if k == 10:
        e4 = load("e4_recall_skew_glove100")["configs"][f"baseline_ef{ef}"]
        r = np.array(e4["recalls"])
        n = None
        if ef == 160:
            n = np.array(load("e9_hardness_causal_glove100")["cost_n_dists_ef160"])
        return r, n
    row = next(r for r in load(f"e8_ksweep_glove100_k{k}")["baseline"] if r["ef"] == ef)
    return np.array(row["recalls"]), np.array(row["n_dists"])


def _e12_sel(r, b):
    return (r >= b["lo"]) & ((r <= b["hi"]) if b["inclusive"] else (r < b["hi"]))


def _bin_xticks(ax, bins):
    ax.set_xticks(np.arange(len(bins)), [b["level"].replace(" (", "\n(") for b in bins])


def fig_e12_bins(out="e12_bins"):
    """E12a — hard / mid / easy counts for every configuration, internal vs
    external queries."""
    d = load("e12_hardness_internal_glove100")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    w = 0.38
    for ax, label in zip(axes.ravel(), E12_LABELS):
        c = d["configs"][label]
        bins = c["bins"]
        r_int = np.array(c["recalls"])
        r_ext, _ = _e12_external(c["k"], c["ef"])
        x = np.arange(len(bins))
        for j, (name, col, r) in enumerate(
            [("internal (nodes)", BLUE, r_int), ("external (test set)", ORANGE, r_ext)]
        ):
            cnt = [int(_e12_sel(r, b).sum()) for b in bins]
            bars = ax.bar(x + (j - 0.5) * w, cnt, w, color=col, label=name)
            for b_, v in zip(bars, cnt):
                ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v}", ha="center",
                        va="bottom", fontsize=8, color=INK2)
        _bin_xticks(ax, bins)
        ax.set_ylim(0, 1000)
        ax.set_ylabel("queries (of 1000)")
        ax.set_title(f"k={c['k']}, ef={c['ef']}  —  mean recall {r_int.mean():.3f} "
                     f"internal vs {r_ext.mean():.3f} external", loc="left")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("E12a — hardness bins per operating point, internal vs external "
                 "queries (GloVe-100, 1000 queries each)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out)


def fig_e12_recall_dist(out="e12_recall_dist"):
    """E12a — recall distributions where recall is near-continuous
    (k=100, k=1000) plus the k=10 ef=40 backup label."""
    d = load("e12_hardness_internal_glove100")
    labels = ["k10_ef40", "k100", "k1000"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, label in zip(axes, labels):
        c = d["configs"][label]
        r_int = np.array(c["recalls"])
        r_ext, _ = _e12_external(c["k"], c["ef"])
        if c["k"] == 10:
            edges = np.arange(-0.05, 1.06, 0.1)
        else:
            edges = np.linspace(0, 1, 21)
        for name, col, r in [("internal (nodes)", BLUE, r_int),
                             ("external (test set)", ORANGE, r_ext)]:
            ax.hist(r, bins=edges, color=col, alpha=0.75, label=name,
                    histtype="stepfilled", edgecolor=SURFACE, linewidth=0.8)
        for b in c["bins"][:-1]:
            ax.axvline(b["hi"] if not b["inclusive"] else b["hi"] + 0.005,
                       color=BASE, lw=1, ls=":")
        ax.set_xlabel(f"recall@{c['k']} at ef={c['ef']}")
        ax.set_ylabel("queries")
        ax.set_title(f"k={c['k']}, ef={c['ef']}", loc="left")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("E12a — recall distribution, internal vs external queries "
                 "(dotted lines = bin edges)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out)


def fig_e12_cost(out="e12_cost"):
    """E12a — search cost per hardness bin, internal vs external."""
    d = load("e12_hardness_internal_glove100")
    labels = ["k10", "k100", "k1000"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    w = 0.38
    for ax, label in zip(axes, labels):
        c = d["configs"][label]
        bins = c["bins"]
        r_int, n_int = np.array(c["recalls"]), np.array(c["n_dists"])
        r_ext, n_ext = _e12_external(c["k"], c["ef"])
        x = np.arange(len(bins))
        top = 0
        for j, (name, col, r, n) in enumerate(
            [("internal (nodes)", BLUE, r_int, n_int), ("external (test set)", ORANGE, r_ext, n_ext)]
        ):
            m = [n[_e12_sel(r, b)].mean() if _e12_sel(r, b).any() else 0 for b in bins]
            top = max(top, max(m))
            bars = ax.bar(x + (j - 0.5) * w, m, w, color=col, label=name)
            for b_, v in zip(bars, m):
                txt = f"{v / 1000:.1f}k" if v >= 10000 else f"{v:,.0f}"
                ax.text(b_.get_x() + b_.get_width() / 2, v, txt, ha="center",
                        va="bottom", fontsize=7.5, color=INK2)
        _bin_xticks(ax, bins)
        ax.set_ylim(0, top * 1.3)
        ax.set_ylabel("distance computations (mean)")
        ax.set_title(f"k={c['k']}, ef={c['ef']}", loc="left")
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle("E12a — search cost by hardness bin: hard queries cost more, "
                 "and internal ≈ external", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out)


def fig_e12_adjacency(out="e12_adjacency"):
    """E12a — how much of a node's true neighborhood is wired directly to it."""
    d = load("e12_hardness_internal_glove100")
    s = d["adjacency"]["summary"]
    panels = [
        ("top16_direct", 16, "of the 16 nearest true neighbors,\nhow many are direct out-edges",
         "(a) are the first M=16 neighbors one hop away?"),
        ("edges_in_gt100", 32, "of the node's out-edges (≤32),\nhow many are in its true top-100",
         "(b) out-edges that are true top-100 neighbors"),
        ("edges_in_gt1000", 32, "of the node's out-edges (≤32),\nhow many are in its true top-1000",
         "(c) out-edges that are true top-1000 neighbors"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, n_max, xlabel, title) in zip(axes, panels):
        h = np.array(s[key]["hist"][: n_max + 1])
        ax.bar(np.arange(n_max + 1), h, 0.7, color=BLUE)
        mean = s[key]["mean"]
        ax.axvline(mean, color=INK2, lw=1, ls="--")
        ax.text(mean + 0.3, h.max() * 1.04, f"mean {mean:.1f} of {n_max}", color=INK2,
                fontsize=9)
        ax.set_ylim(0, h.max() * 1.15)
        ax.set_xticks(np.arange(0, n_max + 1, 4 if n_max > 16 else 2))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("nodes")
        ax.set_title(title, loc="left")
    fig.suptitle("E12a — graph wiring around each of the 1000 sampled nodes "
                 "(no search; M=16, level-0 degree ≤32)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out)


def fig_e12_hops(label="k1000", k_gt=100, out=None):
    """E12b — graph distances for internal queries vs the external-query
    result at the same operating point: avg-dist per bin (both), hops from
    the node itself to its true neighbors per bin, and those hops by
    neighbor rank."""
    d = load(f"e12_avgdist_internal_glove100_{label}_gt{k_gt}")
    lv = d["levels"]
    ext = d.get("external_reference")
    names = [l["level"].replace(" (", "\n(") for l in lv]
    x = np.arange(len(lv))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) avg-dist among true neighbors, internal vs external
    ax = axes[0]
    vi = [l["avg_dist"]["mean"] or 0 for l in lv]
    w = 0.34
    b1 = ax.bar(x - 0.22, vi, w, color=BLUE, label="internal (nodes)")
    for b_, v, l in zip(b1, vi, lv):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}\nn={l['n_queries']}",
                ha="center", va="bottom", fontsize=7, color=INK2)
    if ext:
        ve = [(l["external"] or {}).get("avg_dist_mean", 0) or 0 for l in lv]
        b2 = ax.bar(x + 0.22, ve, w, color=ORANGE, label="external (test set)")
        for b_, v, l in zip(b2, ve, lv):
            n = (l["external"] or {}).get("n_queries", "")
            ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}\nn={n}",
                    ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(vi) * 1.35)
    ax.set_ylabel(f"avg-dist: mean hops between\npairs of the {k_gt} true neighbors")
    ax.set_title("(a) avg-dist among true neighbors, by bin", loc="left")
    ax.legend(frameon=False, loc="upper right")

    # (b) hops from the node itself to its true neighbors, by bin
    ax = axes[1]
    vq = [l["q_hops"]["mean"] or 0 for l in lv]
    f1 = [100 * (l["q_hops"]["frac_1hop"] or 0) for l in lv]
    bars = ax.bar(x, vq, 0.6, color=BLUE)
    for b_, v, f in zip(bars, vq, f1):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.2f}\n{f:.0f}% at 1 hop",
                ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(vq) * 1.4)
    ax.set_ylabel(f"mean hops from the node\nto its {k_gt} true neighbors")
    ax.set_title("(b) hops from the node itself, by bin (internal only)", loc="left")

    # (c) hops from the node by neighbor rank, hard vs easy
    ax = axes[2]
    groups = [g for g in ["1-16", "17-32", "33-100", "101-1000"]
              if f"rank_{g}" in lv[0]["q_hops"]]
    xg = np.arange(len(groups))
    shades = [BLUE, AQUA, YELLOW, ORANGE][: len(lv)]
    wg = 0.8 / len(lv)
    for j, (l, col) in enumerate(zip(lv, shades)):
        vals = [l["q_hops"].get(f"rank_{g}") or 0 for g in groups]
        ax.bar(xg + (j - (len(lv) - 1) / 2) * wg, vals, wg, color=col,
               label=l["level"])
    ax.set_xticks(xg, [f"ranks {g}" for g in groups])
    ax.set_ylabel("mean hops from the node")
    ax.set_title("(c) hops from the node, by true-neighbor rank", loc="left")
    ax.legend(frameon=False, fontsize=8, title="hardness bin", title_fontsize=8)

    sp = d["spearman"]
    fig.suptitle(
        f"E12b — internal queries: {d['hardness_from']} label, true top-{k_gt}. "
        f"Spearman with recall: avg-dist {sp['recall_vs_avg_dist']:+.2f}, "
        f"hops-from-node {sp['recall_vs_q_hops']:+.2f}"
        + (f"; external avg-dist {ext['spearman_recall_vs_avg_dist']:+.2f}" if ext else ""),
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out or f"e12_hops_{label}_gt{k_gt}")


# ------------------------------------------------------- explainer diagrams
def fig_gt_normalize(out="explain_normalize"):
    """Normalization: same direction, length 1. (3,4) -> (0.6,0.8)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 4.4)
    ax.set_ylim(-0.3, 4.6)
    circ = plt.Circle((0, 0), 1, fill=False, color=BASE, lw=1.2, ls="--")
    ax.add_patch(circ)
    ax.annotate("", xy=(3, 4), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
    ax.annotate("", xy=(0.6, 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=3))
    ax.text(3.05, 4.05, "v = (3, 4)\nlength 5", color=ORANGE, fontsize=11, va="bottom")
    ax.text(0.75, 0.55, "v / 5 = (0.6, 0.8)\nlength 1", color=BLUE, fontsize=11, va="top")
    ax.text(1.05, -0.15, "unit circle", color=MUTED, fontsize=9)
    ax.text(0.05, 4.3, "same direction, length set to 1", color=INK2, fontsize=11)
    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.grid(True, color=GRID, lw=0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    savefig(fig, out)


def fig_gt_matmul(out="explain_matmul"):
    """One matrix multiply scores every query against every base vector."""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, color, label, sub):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="none", alpha=0.9))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y - 0.35, sub, ha="center", va="top", color=INK2, fontsize=10)

    # Q: 1000 x 100 (tall-ish, narrow)
    box(0.5, 1.5, 1.6, 3.2, BLUE, "Q", "1000 queries\n× 100 dims")
    ax.text(2.6, 3.1, "·", fontsize=26, ha="center", va="center", color=INK)
    # X^T: 100 x 1.18M (short, very wide)
    box(3.1, 2.7, 8.0, 0.8, AQUA, "Xᵀ", "100 dims × 1,183,514 base vectors")
    ax.text(11.6, 3.1, "=", fontsize=22, ha="center", va="center", color=INK)
    # S: 1000 x 1.18M
    box(12.1, 1.5, 9.4, 3.2, ORANGE, "S", "1000 × 1,183,514 similarities\nS[i, j] = how similar query i is to node j")
    ax.text(0.5, 5.6, "one matrix multiply — every query compared with every vector (nothing skipped)",
            fontsize=12, color=INK, va="center")
    ax.text(21.5, 5.6, "≈ 1.2 × 10¹¹ multiply-adds → a few seconds", fontsize=10, color=INK2,
            va="center", ha="right")
    fig.tight_layout()
    savefig(fig, out)


def fig_hops_example(out="explain_hops"):
    """Tiny directed graph for the step-3 explainer: query node q, its five
    true neighbors A-E, one non-neighbor X. Left: the out-edges. Right: the
    same graph colored by BFS ring (hops from q)."""
    pos = {"q": (0, 1), "A": (1.5, 2), "X": (1.5, 0), "B": (3, 2), "C": (3, 0),
           "D": (4.5, 2), "E": (4.5, 0)}
    edges = [("q", "A"), ("q", "X"), ("A", "B"), ("A", "q"), ("B", "D"), ("B", "A"),
             ("X", "C"), ("C", "E"), ("C", "X"), ("D", "E"), ("E", "A")]
    gt = {"A", "B", "C", "D", "E"}
    hops = {"q": 0, "A": 1, "X": 1, "B": 2, "C": 2, "D": 3, "E": 3}
    ring_col = {0: ORANGE, 1: BLUE, 2: AQUA, 3: YELLOW}

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    for ax, mode in zip(axes, ["edges", "rings"]):
        ax.set_xlim(-0.6, 5.2)
        ax.set_ylim(-1.15, 2.7)
        ax.set_aspect("equal")
        ax.axis("off")
        for u, v in edges:
            (x1, y1), (x2, y2) = pos[u], pos[v]
            # offset reciprocal edges slightly so both arrows show
            rec = (v, u) in edges
            dx, dy = x2 - x1, y2 - y1
            n = (dx * dx + dy * dy) ** 0.5
            ox, oy = (-dy / n * 0.08, dx / n * 0.08) if rec else (0, 0)
            ax.annotate("", xy=(x2 + ox, y2 + oy), xytext=(x1 + ox, y1 + oy),
                        arrowprops=dict(arrowstyle="-|>", color=INK2 if mode == "edges" else BASE,
                                        lw=1.6, shrinkA=14, shrinkB=14))
        for name, (x, y) in pos.items():
            if mode == "edges":
                col = ORANGE if name == "q" else (BLUE if name in gt else MUTED)
            else:
                col = ring_col[hops[name]]
            ax.add_patch(plt.Circle((x, y), 0.26, color=col, zorder=3))
            ax.text(x, y, name, ha="center", va="center", color="white", fontsize=12,
                    fontweight="bold", zorder=4)
            if mode == "rings":
                ax.text(x, y - 0.42, f"{hops[name]} hop{'s' if hops[name] != 1 else ''}",
                        ha="center", va="top", color=INK2, fontsize=9)
        if mode == "edges":
            ax.set_title("(a) out-edges; q = query node, blue = true neighbors, gray = not",
                         loc="left", fontsize=10.5)
        else:
            ax.set_title("(b) BFS from q: colored by ring (hops from q)",
                         loc="left", fontsize=10.5)
            ax.text(-0.5, -1.0, "hops from q to A–E: 1, 2, 2, 3, 3  →  mean 2.2; 1 of 5 is one hop away",
                    fontsize=9.5, color=INK2)
    fig.tight_layout()
    savefig(fig, out)
