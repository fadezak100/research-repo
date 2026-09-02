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
        ("gt_components", "GT forms N islands in the graph\n(connected components of GT-100)", "islands"),
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
    fig.suptitle(
        "E7 — what makes a query hard? GT-100 structure by hardness level (GloVe-100, 1000 queries)",
        y=0.99, fontsize=13,
    )
    savefig(fig, "e7_hardness")


def fig_e7_key():
    """Two-panel version for the slide: the cause (GT spread) and its
    consequence in the graph (GT islands)."""
    data = load("e7_hardness_glove100")
    pq = data["per_query"]
    recalls = np.array(pq["recalls"])
    levels = [("<=0.25\n(hard)", 0.0, 0.25), ("~0.5", 0.26, 0.5),
              ("~0.75", 0.51, 0.8), ("~1.0\n(easy)", 0.81, 1.0)]
    panels = [
        ("gt_pairwise", "the cause — GT spread:\nmean distance between the 100 true neighbors",
         "cosine distance"),
        ("gt_components", "the consequence — GT islands:\nconnected components of GT-100 in the graph",
         "islands"),
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
        "E7 — hard queries' true neighbors are far apart, so the graph leaves them "
        "as disconnected islands (GloVe-100, 1000 queries)",
        y=1.0, fontsize=13,
    )
    savefig(fig, "e7_hardness_key")


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
