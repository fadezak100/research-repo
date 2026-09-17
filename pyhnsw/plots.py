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


def fig_e11(name="e11_avgdist_glove100_k10_ef160_gt100", out="e11_avgdist_k10_ef160_gt100"):
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


def main():
    fig_e7()
    fig_e7_key()
    fig_e9_asymptote()
    for lab, kg in [("k10_ef160", 100), ("k1000_ef1000", 100), ("k1000_ef1000", 1000)]:
        fig_e11(f"e11_avgdist_glove100_{lab}_gt{kg}", f"e11_avgdist_{lab}_gt{kg}")
    fig_e12_labels("k10_ef160")
    fig_e12_bins()
    fig_e12_recall_dist()
    fig_e12_cost()
    fig_e12_adjacency()
    for lab in ["k10_ef160", "k100_ef100", "k1000_ef1000"]:
        fig_e12_hops(lab, 100)
    fig_gt_normalize()
    fig_gt_matmul()
    fig_hops_example()



def fig_e12_labels(label="k10_ef160", out=None):
    """E12a — internal-query hardness labels vs the external-query labels at
    the same operating point (k=10, ef=160 by default)."""
    d = load("labels_internal_glove100")
    c = d["configs"][label]
    k, ef = c["k"], c["ef"]
    r_int, n_int = np.array(c["recalls"]), np.array(c["n_dists"])
    r_ext, n_ext = _e12_external(label)
    adj = load("adjacency_internal_glove100")["summary"]
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
    h = np.array(adj["top16_direct"]["hist"][:17])
    ax.bar(np.arange(17), h, 0.7, color=BLUE)
    mean = adj["top16_direct"]["mean"]
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
    savefig(fig, out or f"e12_labels_{label}")


# ---------------------------------------------------------------- E12 helpers
E12_LABELS = ["k10_ef160", "k10_ef40", "k100_ef100", "k1000_ef1000"]


def _e12_external(label):
    """Per-query (recalls, n_dists) of the 1000 EXTERNAL queries at the same
    operating point (labels.py --queries external)."""
    c = load("labels_external_glove100")["configs"][label]
    return np.array(c["recalls"]), np.array(c["n_dists"])


def _e12_sel(r, b):
    return (r >= b["lo"]) & ((r <= b["hi"]) if b["inclusive"] else (r < b["hi"]))


def _bin_xticks(ax, bins):
    ax.set_xticks(np.arange(len(bins)), [b["level"].replace(" (", "\n(") for b in bins])


def fig_e12_bins(out="e12_bins"):
    """E12a — hard / mid / easy counts for every configuration, internal vs
    external queries."""
    d = load("labels_internal_glove100")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    w = 0.38
    for ax, label in zip(axes.ravel(), E12_LABELS):
        c = d["configs"][label]
        bins = c["bins"]
        r_int = np.array(c["recalls"])
        r_ext, _ = _e12_external(label)
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
    d = load("labels_internal_glove100")
    labels = ["k10_ef40", "k100_ef100", "k1000_ef1000"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, label in zip(axes, labels):
        c = d["configs"][label]
        r_int = np.array(c["recalls"])
        r_ext, _ = _e12_external(label)
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
    d = load("labels_internal_glove100")
    labels = ["k10_ef160", "k100_ef100", "k1000_ef1000"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    w = 0.38
    for ax, label in zip(axes, labels):
        c = d["configs"][label]
        bins = c["bins"]
        r_int, n_int = np.array(c["recalls"]), np.array(c["n_dists"])
        r_ext, n_ext = _e12_external(label)
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
    d = load("adjacency_internal_glove100")
    s = d["summary"]
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


def fig_e12_hops(label="k1000_ef1000", k_gt=100, out=None):
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


if __name__ == "__main__":
    main()
