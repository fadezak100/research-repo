"""Charts for the meeting deck, generated from results/*.json.

Usage:  .venv/bin/python -m pyhnsw.plots [--dataset glove100|sift1m]
Writes PNGs into results/figs/ (GloVe figures keep their original names;
other datasets get a _{dataset} suffix).
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
ACCENT_TXT = "#1f5fae"

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


DATASET_TITLE = {"glove100": "GloVe-100", "sift1m": "SIFT1M"}


def _suffix(dataset):
    """Figure-name suffix: GloVe figures keep their original names."""
    return "" if dataset == "glove100" else f"_{dataset}"


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

    ds_title = DATASET_TITLE.get(data.get("dataset", "glove100"), data.get("dataset"))
    fig.suptitle(
        "avg-dist: hard queries' true neighbors are farther apart "
        f"in the HNSW graph ({ds_title}, {data['n_queries']} queries; hardness = {label})",
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


# which operating points (labels.py CONFIGS) each dataset's figures show
E12_FIGS = {
    "glove100": {
        "all": ["k10_ef160", "k10_ef40", "k100_ef100", "k1000_ef1000"],
        "k10": "k10_ef160",
        "three": ["k10_ef160", "k100_ef100", "k1000_ef1000"],
        "recall_dist": ["k10_ef40", "k100_ef100", "k1000_ef1000"],
        "e11": [("k10_ef160", 100), ("k1000_ef1000", 100), ("k1000_ef1000", 1000)],
    },
    "sift1m": {
        "all": ["k10_ef16", "k10_ef40", "k100_ef100", "k1000_ef1000"],
        "k10": "k10_ef16",
        "three": ["k10_ef16", "k100_ef100", "k1000_ef1000"],
        "recall_dist": ["k10_ef16", "k100_ef100", "k1000_ef1000"],
        "e11": [("k10_ef16", 100), ("k100_ef100", 100), ("k1000_ef1000", 100)],
    },
}


def main(dataset="glove100"):
    cfg = E12_FIGS[dataset]
    if dataset == "glove100":  # E7 / E9 and the explainer diagrams exist only for GloVe
        fig_e7()
        fig_e7_key()
        fig_e9_asymptote()
    for lab, kg in cfg["e11"]:
        fig_e11(f"e11_avgdist_{dataset}_{lab}_gt{kg}",
                f"e11_avgdist_{lab}_gt{kg}{_suffix(dataset)}")
    fig_e12_labels(cfg["k10"], dataset)
    fig_e12_bins(dataset)
    fig_e12_recall_dist(dataset)
    fig_e12_cost(dataset)
    fig_e12_adjacency(dataset)
    for lab in cfg["three"]:
        fig_e12_hops(lab, 100, dataset)
    if (RESULTS / f"e13_online_stats_{dataset}.json").exists():
        fig_e13_scatter(dataset)
        fig_e13_stats()
    if (RESULTS / f"e14_policy_{dataset}.json").exists():
        fig_e14(dataset)
    if (RESULTS / f"e15_shap_{dataset}.json").exists():
        fig_e15()
    if (RESULTS / f"e16_resumable_{dataset}.json").exists():
        fig_e16()
    if (RESULTS / f"e17_end_to_end_{dataset}.json").exists():
        fig_e17()
    if dataset == "glove100":
        fig_gt_normalize()
        fig_gt_matmul()
        fig_hops_example()
        fig_entry_features()
        fig_system()



def fig_e12_labels(label="k10_ef160", dataset="glove100", out=None):
    """E12a — internal-query hardness labels vs the external-query labels at
    the same operating point (k=10, ef=160 by default)."""
    d = load(f"labels_internal_{dataset}")
    c = d["configs"][label]
    k, ef = c["k"], c["ef"]
    r_int, n_int = np.array(c["recalls"]), np.array(c["n_dists"])
    r_ext, n_ext = _e12_external(label, dataset)
    adj = load(f"adjacency_internal_{dataset}")["summary"]
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
        f"{DATASET_TITLE[dataset]}, k={k} ef={ef}; self not found in {c['self_not_found']} / 1000",
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out or f"e12_labels_{label}{_suffix(dataset)}")


# ---------------------------------------------------------------- E12 helpers
def _e12_external(label, dataset="glove100"):
    """Per-query (recalls, n_dists) of the 1000 EXTERNAL queries at the same
    operating point (labels.py --queries external)."""
    c = load(f"labels_external_{dataset}")["configs"][label]
    return np.array(c["recalls"]), np.array(c["n_dists"])


def _e12_sel(r, b):
    return (r >= b["lo"]) & ((r <= b["hi"]) if b["inclusive"] else (r < b["hi"]))


def _bin_xticks(ax, bins):
    ax.set_xticks(np.arange(len(bins)), [b["level"].replace(" (", "\n(") for b in bins])


def fig_e12_bins(dataset="glove100", out=None):
    """E12a — hard / mid / easy counts for every configuration, internal vs
    external queries."""
    d = load(f"labels_internal_{dataset}")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    w = 0.38
    for ax, label in zip(axes.ravel(), E12_FIGS[dataset]["all"]):
        c = d["configs"][label]
        bins = c["bins"]
        r_int = np.array(c["recalls"])
        r_ext, _ = _e12_external(label, dataset)
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
                 f"queries ({DATASET_TITLE[dataset]}, 1000 queries each)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out or f"e12_bins{_suffix(dataset)}")


def fig_e12_recall_dist(dataset="glove100", out=None):
    """E12a — recall distributions where recall is near-continuous
    (k=100, k=1000) plus one k=10 label."""
    d = load(f"labels_internal_{dataset}")
    labels = E12_FIGS[dataset]["recall_dist"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, label in zip(axes, labels):
        c = d["configs"][label]
        r_int = np.array(c["recalls"])
        r_ext, _ = _e12_external(label, dataset)
        if c["k"] == 10:
            edges = np.arange(-0.05, 1.06, 0.1)
        elif min(r_int.min(), r_ext.min()) < 0.6:
            edges = np.linspace(0, 1, 21)
        else:  # narrow range (SIFT1M): finer bins over the top half
            edges = np.linspace(0.5, 1, 51)
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
    fig.suptitle(f"E12a — recall distribution, internal vs external queries, "
                 f"{DATASET_TITLE[dataset]} (dotted lines = bin edges)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out or f"e12_recall_dist{_suffix(dataset)}")


def fig_e12_cost(dataset="glove100", out=None):
    """E12a — search cost per hardness bin, internal vs external."""
    d = load(f"labels_internal_{dataset}")
    labels = E12_FIGS[dataset]["three"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    w = 0.38
    for ax, label in zip(axes, labels):
        c = d["configs"][label]
        bins = c["bins"]
        r_int, n_int = np.array(c["recalls"]), np.array(c["n_dists"])
        r_ext, n_ext = _e12_external(label, dataset)
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
                 f"and internal ≈ external ({DATASET_TITLE[dataset]})",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out or f"e12_cost{_suffix(dataset)}")


def fig_e12_adjacency(dataset="glove100", out=None):
    """E12a — how much of a node's true neighborhood is wired directly to it."""
    d = load(f"adjacency_internal_{dataset}")
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
                 f"({DATASET_TITLE[dataset]}; no search; M=16, level-0 degree ≤32)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out or f"e12_adjacency{_suffix(dataset)}")


def fig_e12_hops(label="k1000_ef1000", k_gt=100, dataset="glove100", out=None):
    """E12b — graph distances for internal queries vs the external-query
    result at the same operating point: avg-dist per bin (both), hops from
    the node itself to its true neighbors per bin, and those hops by
    neighbor rank."""
    d = load(f"e12_avgdist_internal_{dataset}_{label}_gt{k_gt}")
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
        f"E12b — internal queries, {DATASET_TITLE[dataset]}: {d['hardness_from']} label, "
        f"true top-{k_gt}. "
        f"Spearman with recall: avg-dist {sp['recall_vs_avg_dist']:+.2f}, "
        f"hops-from-node {sp['recall_vs_q_hops']:+.2f}"
        + (f"; external avg-dist {ext['spearman_recall_vs_avg_dist']:+.2f}" if ext else ""),
        x=0.01, ha="left", fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out or f"e12_hops_{label}_gt{k_gt}{_suffix(dataset)}")


# ------------------------------------------------------------------ E13
E13_NAMES = {
    "d10": "distance to found top-10", "d100": "distance to found top-100",
    "contrast10": "contrast (top-10)", "contrast100": "contrast (top-100)",
    "spread": "spread of found top-100", "edges": "edges among found top-100",
    "avgdist": "avg-dist among found top-100", "n_hops": "hops of the ef=100 search",
    "n_dist": "distances of the ef=100 search", "ep_dist": "entry: dist to level-0 entry",
    "ep_nbr_mean": "entry: mean dist to its neighbors", "ep_nbr_std": "entry: std of neighbor dists",
    "ep_degree": "entry: degree", "descent_ndist": "entry: upper-layer distances",
    "global_ep_dist": "entry: dist to global entry",
}


def fig_e13_stats(datasets=("glove100", "sift1m"), out="e13_stats"):
    """E13 — Spearman of each query-time statistic (found top-100 of an
    ef=100 search) with the recall label, per dataset; the offline twin
    (same statistic on the true top-100) drawn as a marker."""
    ds_have = [d for d in datasets if (RESULTS / f"e13_online_stats_{d}.json").exists()]
    fig, axes = plt.subplots(1, len(ds_have), figsize=(6.2 * len(ds_have), 6.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, dsn in zip(axes, ds_have):
        d = load(f"e13_online_stats_{dsn}")
        names = d["online_stats"]
        twins = {a: v["offline"] for a, v in d["pairing"].items()}
        y = np.arange(len(names))[::-1]
        rho = [d["spearman"][nm]["recall"] for nm in names]
        ax.barh(y, rho, 0.62, color=BLUE, label="query-time (found top-100)")
        for yi, nm, r in zip(y, names, rho):
            ax.text(r + (0.02 if r >= 0 else -0.02), yi, f"{r:+.2f}", va="center",
                    ha="left" if r >= 0 else "right", fontsize=8, color=INK2)
            if nm in twins:
                rt = d["spearman"][twins[nm]]["recall"]
                ax.plot([rt], [yi], marker="D", color=ORANGE, ms=7, ls="none",
                        label="offline twin (true top-100)" if nm == names[0] else None)
        ax.axvline(0, color=BASE, lw=1)
        ax.set_yticks(y, [E13_NAMES.get(nm, nm) for nm in names])
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel(f"Spearman with recall@100 at ef=100")
        ax.set_title(f"{DATASET_TITLE[dsn]}: hard bin {d['hard_bin']['level']} (n={d['n_hard']})",
                     loc="left")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, loc="lower center", ncol=2, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("E13 — how much hardness signal survives when the statistic is computed on "
                 "the FOUND neighbors instead of the true ones (1000 external queries)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, out)


def fig_e13_scatter(dataset="glove100", out=None):
    """E13 — the two strongest families against recall: contrast (top-100)
    and avg-dist among the found top-100, hard bin highlighted."""
    d = load(f"e13_online_stats_{dataset}")
    pq = d["per_query"]
    rec = np.array(pq["recall"])
    hb = d["hard_bin"]
    hard = (rec >= hb["lo"]) & ((rec <= hb["hi"]) if hb["inclusive"] else (rec < hb["hi"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key in zip(axes, ["contrast100", "avgdist"]):
        x = np.array(pq[key], dtype=float)
        ax.scatter(x[~hard], rec[~hard], s=9, color=BLUE, alpha=0.45, label="other")
        ax.scatter(x[hard], rec[hard], s=11, color=ORANGE, alpha=0.7, label=f"hard ({hb['level']})")
        r = d["spearman"][key]
        if key == "contrast100":
            ax.set_xscale("log")
        ax.set_xlabel(E13_NAMES[key] + (" (log scale)" if key == "contrast100" else ""))
        ax.set_title(f"Spearman with recall {r['recall']:+.2f}, with oracle ef at target 0.9 "
                     f"{r['ef_min_0.9']:+.2f}", loc="left", fontsize=10)
        ax.set_ylabel("recall@100 at ef=100")
        ax.legend(frameon=False, loc="lower right" if key == "contrast100" else "lower left")
    fig.suptitle(f"E13 — query-time statistics vs recall, {DATASET_TITLE[dataset]} "
                 f"(prefix search k=100, ef=100; {d['n_queries']} external queries)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out or f"e13_scatter{_suffix(dataset)}")


# ------------------------------------------------------------------ E15
def fig_e15(datasets=("glove100", "sift1m"), target="ef_0.9", out="e15_shap"):
    """E15 — SHAP summary (QBAT Fig. 4 style) of a LightGBM model predicting
    log2 of the oracle ef from the 14 query-time statistics, one row per
    dataset. Left: features sorted by mean |SHAP|; one dot per query,
    jittered vertically, coloured by the feature's value rank (light = low,
    dark = high). Right: held-out R^2 of each feature alone, with the R^2 of
    the top-2 / top-5 features by SHAP as reference lines."""
    ds_have = [d for d in datasets if (RESULTS / f"e15_shap_{d}.json").exists()]
    fig, axes = plt.subplots(len(ds_have), 2, figsize=(13, 5.6 * len(ds_have)), squeeze=False,
                             gridspec_kw={"width_ratios": [3.2, 1.0]})
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue_ramp", ["#cfe0f7", BLUE, "#0d3a75"])
    rng = np.random.default_rng(0)
    for (ax, axr), dsn in zip(axes, ds_have):
        d = load(f"e15_shap_{dsn}")
        r = d["results"][target]
        feats = [f["feature"] for f in r["shap"]]
        sv = np.array(r["per_query"]["shap"])
        col = {f: j for j, f in enumerate(d["features"])}
        y = np.arange(len(feats))[::-1]
        for yi, f in zip(y, feats):
            x = sv[:, col[f]]
            v = np.array(d["feature_values"][f], dtype=float)
            v = np.argsort(np.argsort(v)) / (len(v) - 1)  # rank -> colour, robust to outliers
            ax.scatter(x, yi + rng.uniform(-0.28, 0.28, len(x)), c=v, cmap=ramp, s=7, alpha=0.75,
                       linewidths=0, rasterized=True)
        ax.axvline(0, color=BASE, lw=1)
        ax.set_yticks(y, [f"{E13_NAMES.get(f, f)}  ({100 * s['share']:.0f}%)"
                          for f, s in zip(feats, r["shap"])])
        ax.set_xlabel("SHAP value: shift of the predicted log2(ef) for that query   "
                      "(in brackets: share of the total |SHAP|)")
        ax.set_title(f"{DATASET_TITLE[dsn]}: predicting the smallest ef that reaches recall {target[3:]}; "
                     f"all 14 features, held-out R² {r['fit_all_features']['r2']:.2f}", loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
        alone = [r["single_feature"][f]["r2"] for f in feats]
        axr.barh(y, np.clip(alone, 0, None), 0.62, color=BLUE)
        for yi, a in zip(y, alone):
            axr.text(max(a, 0) + 0.015, yi, f"{a:.2f}", va="center", fontsize=8, color=INK2)
        tk = r["top_k_by_shap"]
        for k, ls in (("2", "--"), ("5", ":")):
            axr.axvline(tk[k]["r2"], color=ORANGE, lw=1.3, ls=ls)
        axr.set_title(f"orange: top 2 by SHAP together {tk['2']['r2']:.2f} (dashed), "
                      f"top 5 {tk['5']['r2']:.2f} (dotted)", loc="left", fontsize=8.5, color=INK2)
        axr.set_xlim(0, 1.05)
        axr.set_yticks(y, ["" for _ in feats])
        axr.set_xlabel("held-out R² of the feature alone")
        axr.grid(axis="y", visible=False)
    sm = matplotlib.cm.ScalarMappable(cmap=ramp)
    cb = fig.colorbar(sm, ax=axes[:, 0].tolist(), fraction=0.02, pad=0.01, ticks=[0, 1], aspect=40)
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("the query's value of that feature (rank)", fontsize=9)
    fig.suptitle("E15 — which statistics does a learned ef predictor use? LightGBM + SHAP on the held-out "
                 "fold (5-fold CV × 3 seeds, 1000 external queries)", x=0.01, ha="left", fontsize=11)
    savefig(fig, out)


# ------------------------------------------------------------------ E16
def fig_e16(datasets=("glove100", "sift1m"), out="e16_resumable"):
    """E16 — the resumable Python engine vs fresh Faiss searches, one row per
    dataset. Left: mean recall@100 against ef for a fresh search, a search
    resumed from the ef=100 peek (with the spill heap), and resumed without
    it. Middle: mean distance computations against ef for the same three,
    plus the two-stage cost (peek + fresh). Right: Spearman of the in-engine
    statistics with the peek recall and with the measured oracle ef."""
    ds_have = [d for d in datasets if (RESULTS / f"e16_resumable_{d}.json").exists()]
    fig, axes = plt.subplots(len(ds_have), 3, figsize=(16, 4.8 * len(ds_have)), squeeze=False)
    for (ax_r, ax_c, ax_s), dsn in zip(axes, ds_have):
        d = load(f"e16_resumable_{dsn}")
        r = d["resume"]
        efs = [x["ef"] for x in r]
        for key, col, ls, lw, lab in [("fresh", MUTED, "-", 6, "fresh search (Faiss); the blue line sits on top of it"),
                                      ("resumed", BLUE, "-", 1.8, "resumed from the ef=100 peek, spill kept"),
                                      ("resumed_no_spill", ORANGE, "--", 1.8, "resumed, spill dropped")]:
            ax_r.plot(efs, [x[f"recall_{key}"] for x in r], ls, marker="o", ms=4, lw=lw, color=col, alpha=0.5 if lw > 3 else 1, label=lab)
            ax_c.plot(efs, [x[f"cost_{key}"] for x in r], ls, marker="o", ms=4, lw=lw, color=col, alpha=0.5 if lw > 3 else 1, label=lab)
        ax_c.plot(efs, [x["cost_two_stage"] for x in r], ":", marker="s", ms=4, color=AQUA, label="two-stage: peek, then fresh")
        for ax in (ax_r, ax_c):
            ax.set_xscale("log")
            ax.set_xticks(efs[::2], [str(e) for e in efs[::2]])
            ax.minorticks_off()
            ax.set_xlabel("ef")
        ax_r.set_ylabel("mean recall@100")
        ax_c.set_ylabel("mean distance computations")
        ax_r.set_title(f"{DATASET_TITLE[dsn]}: recall reached", loc="left", fontsize=10.5)
        ax_c.set_title(f"{DATASET_TITLE[dsn]}: what it cost", loc="left", fontsize=10.5)
        ax_r.legend(fontsize=8, loc="lower right")
        ax_c.legend(fontsize=8, loc="upper left")
        names = list(d["spearman"])
        y = np.arange(len(names))[::-1]
        ax_s.barh(y + 0.18, [d["spearman"][s]["recall"] for s in names], 0.34, color=BLUE, label="with peek recall")
        ax_s.barh(y - 0.18, [-d["spearman"][s]["ef_min_0.9_resumed"] for s in names], 0.34, color=ORANGE,
                  label="with oracle ef @0.9 (sign flipped)")
        for yi, s in zip(y, names):
            ax_s.text(1.03, yi, f"catch {100 * d['spearman'][s]['catch_rate']:.0f}%", va="center", fontsize=8, color=INK2)
        ax_s.axvline(0, color=BASE, lw=1)
        ax_s.set_yticks(y, [E13_NAMES.get(s, s) for s in names])
        ax_s.set_xlim(-1.05, 1.35)
        ax_s.set_xlabel("Spearman ρ (measured inside the engine)")
        ax_s.set_title(f"{DATASET_TITLE[dsn]}: statistics at the peek, hard bin n={d['n_hard']}", loc="left", fontsize=10.5)
        ax_s.legend(fontsize=8, loc="lower left")
        ax_s.grid(axis="y", visible=False)
    fig.suptitle("E16 — peek at ef=100, compute the statistic, keep searching: measured in a resumable Python engine "
                 "(1000 external queries, k=100)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out)


# ------------------------------------------------------------------ E17
def fig_e17(out="e17_policy"):
    """E17 — the policy executed in the engine vs the fixed-ef frontier and
    the oracle, measured: fraction of queries reaching the target against
    mean distance computations, one panel per (dataset, target)."""
    panels = [(d, T) for d in ("glove100", "sift1m") if (RESULTS / f"e17_end_to_end_{d}.json").exists()
              for T in load(f"e17_end_to_end_{d}")["results"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    axes = np.atleast_1d(axes)
    colors = {"edges": BLUE, "contrast10": ORANGE}
    for ax, (dsn, T) in zip(axes, panels):
        r = load(f"e17_end_to_end_{dsn}")["results"][T]
        fr = r["frontier"]
        fc, ff = [f["cost"] for f in fr], [f["frac"] for f in fr]
        ax.plot(fc, ff, "-o", color=MUTED, ms=4, lw=1.5, label="one ef for every query")
        for f in fr:
            if f["ef"] in (100, 200, 400, 800, 1600, 3200):
                ax.annotate(f"ef {f['ef']}", (f["cost"], f["frac"]), xytext=(4, -10), textcoords="offset points",
                            fontsize=7.5, color=INK2)
        for pol in r["policies"]:
            m = pol["measured"]
            ax.plot([m["cost"]], [m["frac"]], "o", color=colors[pol["stat"]], ms=10, mec=SURFACE, mew=1,
                    label=f"peek + {E13_NAMES[pol['stat']].split(' (')[0]} -> ef")
            ax.annotate("", (m["cost"], m["fixed_ef_at_same_cost"]["frac"]), (m["cost"], m["frac"]),
                        arrowprops=dict(arrowstyle="-", color=colors[pol["stat"]], lw=1, ls=":"))
        o = r["oracle"]
        ax.plot([o["cost"]], [o["frac"]], "*", color=INK, ms=13, label="oracle (each query its own ef)")
        xmax = max(max(pol["measured"]["cost"] for pol in r["policies"]), o["cost"]) * 1.9
        ax.set_xlim(0, min(xmax, max(fc)))
        ax.set_ylim(min(ff) - 0.03, 1.02)
        ax.set_xlabel("mean distance computations per query")
        ax.set_ylabel(f"share of queries with recall@100 ≥ {T}")
        ax.set_title(f"{DATASET_TITLE[dsn]}, target {T}", loc="left", fontsize=10.5)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("E17 — peek at ef=100, read the statistic, keep searching at the table's ef: measured end to end "
                 "(1000 queries, k=100)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    savefig(fig, out)


# ------------------------------------------------------------------ E14
def fig_e14(dataset="glove100", out=None):
    """E14 — two-stage ef policies vs the fixed-ef frontier and the oracle:
    mean recall, p5 and fraction >= T against mean cost, one row per target.
    Filled markers = two-stage cost (prefix + re-search), hollow = resume
    cost (chosen ef alone); a thin line joins the two for each policy."""
    d = load(f"e14_policy_{dataset}")
    targets = list(d["results"])
    fig, axes = plt.subplots(len(targets), 3, figsize=(15, 4.4 * len(targets)), squeeze=False)
    colors = {"edges": BLUE, "contrast10": ORANGE, "both": AQUA}
    for row, T in zip(axes, targets):
        r = d["results"][T]
        fr = r["frontier"]
        fc = np.array([f["cost"] for f in fr])
        xmax = max(p["two_stage"]["cost"] for p in r["policies"]) * 1.6
        xmax = max(xmax, r["oracle"]["two_stage"]["cost"] * 1.3)
        for ax, key, ylab in zip(row, ["mean", "p5", "frac"],
                                 ["mean recall@100", "p5 recall@100", f"fraction of queries with recall >= {T}"]):
            fv = np.array([f[key] for f in fr])
            ax.plot(fc, fv, "-o", color=MUTED, ms=5, lw=1.5, label="fixed ef (frontier)")
            for f in fr:
                if f["cost"] <= xmax:
                    ax.annotate(f"ef={f['ef']}", (f["cost"], f[key]), fontsize=7, color=MUTED,
                                xytext=(3, -9), textcoords="offset points")
            o = r["oracle"]
            ax.plot([o["two_stage"]["cost"]], [o["two_stage"][key]], marker="*", ms=13, color=INK, ls="none",
                    label="oracle (two-stage cost)")
            ax.plot([o["resume"]["cost"]], [o["resume"][key]], marker="*", ms=13, mfc="none", color=INK, ls="none",
                    label="oracle (resume cost)")
            for p in r["policies"]:
                col = colors[p["stat"]]
                mk = "o" if p["rule"] == "q80" else "s"
                a, b = p["two_stage"], p["resume"]
                ax.plot([a["cost"], b["cost"]], [a[key], b[key]], "-", color=col, lw=0.8, alpha=0.6)
                ax.plot([a["cost"]], [a[key]], marker=mk, ms=8, color=col, ls="none",
                        label=f"{p['stat']} / {p['rule']}" if key == "mean" else None)
                ax.plot([b["cost"]], [b[key]], marker=mk, ms=8, mfc="none", color=col, ls="none")
            ax.set_xlim(0, xmax)
            lo = min(min(fv[fc <= xmax]), min(p["two_stage"][key] for p in r["policies"]))
            ax.set_ylim(max(0, lo - 0.05), 1.02)
            ax.set_xlabel("mean distance computations per query")
            ax.set_ylabel(ylab)
            ax.set_title(f"T={T}: {ylab}", loc="left", fontsize=10)
        row[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(f"E14 — two-stage ef policy (prefix k=100 ef=100 -> statistic -> table -> re-search) vs fixed ef, "
                 f"{DATASET_TITLE[dataset]}; filled = two-stage cost, hollow = resume cost; "
                 f"circle = q80 rule, square = bin-mean rule", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    savefig(fig, out or f"e14_policy{_suffix(dataset)}")


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


def fig_entry_features(out="explain_entry_features"):
    """Explainer for the six entry-time statistics (QBAT's HNSW features):
    a toy three-layer HNSW graph, one query, the greedy descent through the
    upper layers, and each statistic drawn where it is measured. The graph
    and the descent are computed in the drawn coordinates (so lengths on the
    page are the real distances); the layer-0 entry point is where the
    layer-1 greedy walk ends, as in online_stats.descend (except that a
    node already measured on the way down is not measured again here)."""
    import textwrap

    rng = np.random.default_rng(11)
    base = {2: 5.95, 1: 3.3, 0: 0.55}
    SHEAR, H, W, X0 = 1.6, 2.0, 7.4, 0.7

    def T(p, lvl):  # unit-square layer coordinates -> page coordinates (sheared layers)
        return X0 + W * p[0] + SHEAR * p[1], base[lvl] + H * p[1]

    q = np.array([0.72, 0.58])
    pts = rng.uniform(0.05, 0.95, size=(30, 2))
    vq = np.array(T(q, 0))
    vis = np.array([T(p, 0) for p in pts])
    pts = pts[np.linalg.norm(vis - vq, axis=1) > 1.15]  # keep the query's own patch empty
    cluster = q + rng.normal(0, 0.085, size=(7, 2))  # the true neighbors, unseen at this stage
    pts = np.vstack([pts, cluster])
    n0 = len(pts)
    vis = np.array([T(p, 0) for p in pts])
    dq = np.linalg.norm(vis - vq, axis=1)
    far = [i for i in np.argsort(-dq) if dq[i] > 2.3]
    upper1 = sorted(far[:4] + rng.choice(far[4:], 6, replace=False).tolist())
    ep = int(np.argmax(dq))
    upper2 = sorted({ep, upper1[len(upper1) // 3], upper1[2 * len(upper1) // 3]})

    def knn_edges(ids, k):
        e = set()
        for i in ids:
            d = np.linalg.norm(vis[ids] - vis[i], axis=1)
            for b in np.argsort(d)[1:k + 1]:
                e.add(tuple(sorted((i, ids[b]))))
        return e

    edges = {0: knn_edges(list(range(n0)), 3), 1: knn_edges(upper1, 2), 2: knn_edges(upper2, 2)}
    nbrs = {lvl: {i: sorted({b if a == i else a for a, b in e if i in (a, b)}) for i in range(n0)}
            for lvl, e in edges.items()}

    # greedy descent, as in online_stats.descend: layers max..1, landing node = level-0 entry
    # measurements = every "how far is this node from the query?" question, in order; a node
    # already measured on the way down is not measured again (a simplification for the drawing)
    cur, path, measured = ep, {2: [], 1: []}, [(2, ep)]
    for lvl in (2, 1):
        path[lvl].append(cur)
        changed = True
        while changed:
            changed = False
            nb = nbrs[lvl][cur]
            measured += [(lvl, i) for i in nb if (lvl, i) not in measured]
            if nb and dq[nb].min() < dq[cur]:
                cur, changed = nb[int(np.argmin(dq[nb]))], True
                path[lvl].append(cur)
    land = cur
    land_nb = nbrs[0][land]
    n_desc = len(measured)

    fig, ax = plt.subplots(figsize=(15.5, 9.9))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-0.5, 9.4)
    ax.axis("off")
    ax.grid(False)
    cluster_ids = set(range(n0 - 7, n0))
    for lvl, name in ((2, "top layer: a few nodes"), (1, "middle layer"), (0, "layer 0: every node")):
        corners = [T(p, lvl) for p in ((0, 0), (1, 0), (1, 1), (0, 1))]
        ax.add_patch(plt.Polygon(corners, closed=True, fc="#f3f2ee", ec=BASE, lw=1, zorder=0))
        ax.text(corners[3][0] - 0.05, corners[3][1] + 0.08, name, fontsize=10, color=INK2, ha="left")
        ids = {2: upper2, 1: upper1, 0: list(range(n0))}[lvl]
        for a, b in edges[lvl]:
            (xa, ya), (xb, yb) = T(pts[a], lvl), T(pts[b], lvl)
            ax.plot([xa, xb], [ya, yb], color=BASE, lw=0.9, zorder=1)
        for i in ids:
            x, y = T(pts[i], lvl)
            c = YELLOW if (lvl == 0 and i in cluster_ids) else MUTED
            ax.plot(x, y, "o", ms=5.5, color=c, mec=SURFACE, mew=0.6, zorder=3)
        # the query's spot on every layer (it is not a node)
        x, y = T(q, lvl)
        ax.plot(x, y, marker="*", ms=13 if lvl == 0 else 12, color=ORANGE, mec=INK, mew=0.5,
                alpha=1 if lvl == 0 else 0.55, ls="none", zorder=6)
        if lvl != 0:
            ax.plot([x, x], [y, base[lvl - 1] + H * q[1]], color=ORANGE, lw=0.8, ls=":", alpha=0.6, zorder=1)
    qx, qy = T(q, 0)
    ax.text(qx + 0.22, qy - 0.02, "query", fontsize=10.5, color=INK, fontweight="bold", va="center")

    # descent path: greedy steps within a layer (solid), drops between layers (dashed)
    for lvl in (2, 1):
        for a, b in zip(path[lvl], path[lvl][1:]):
            ax.annotate("", xy=T(pts[b], lvl), xytext=T(pts[a], lvl),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, shrinkA=4, shrinkB=4), zorder=5)
        top = path[lvl][-1]
        ax.annotate("", xy=T(pts[top], lvl - 1), xytext=T(pts[top], lvl),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, ls="--", shrinkA=5, shrinkB=5), zorder=5)
        ax.plot(*T(pts[path[lvl][0]], lvl), "o", ms=8, color=BLUE, mec=SURFACE, zorder=5)
    for k, (lvl, i) in enumerate(measured, 1):
        (x, y), (sx, sy) = T(pts[i], lvl), T(q, lvl)
        ax.plot([x, sx], [y, sy], color=BLUE, lw=0.9, ls=(0, (1.5, 2.5)), zorder=2)
        ax.plot(x, y, "o", ms=10, mfc="none", mec=BLUE, mew=1.3, zorder=4)
        ax.text(x + 0.18 * (sx - x), y + 0.18 * (sy - y) + 0.13, str(k), fontsize=7.5, color=BLUE,
                fontweight="bold", ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none"))
    ax.plot(*T(pts[ep], 2), "o", ms=9, color=INK, zorder=6)
    ax.annotate("global entry point\n(every search starts here)", T(pts[ep], 2), xytext=(-16, 22),
                textcoords="offset points", fontsize=9.5, color=INK, ha="right",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    # the landing node and its layer-0 neighbours
    lx, ly = T(pts[land], 0)
    ax.plot(lx, ly, "o", ms=11, color=BLUE, mec=INK, mew=0.8, zorder=7)
    ax.annotate("layer-0 entry point (where the real search begins)", (lx, ly), xytext=(lx - 0.3, 0.05),
                textcoords="data", fontsize=9.5, color=INK, ha="right", va="top",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=6))
    for k, i in enumerate(land_nb, 1):
        x, y = T(pts[i], 0)
        ax.plot([lx, x], [ly, y], color=BLUE, lw=2.2, zorder=4)
        ax.plot(x, y, "o", ms=7, color=BLUE, mec=SURFACE, zorder=5)
        ax.plot([qx, x], [qy, y], color=AQUA, lw=1.2, ls="--", zorder=3)
        ux, uy = (x - lx), (y - ly)  # number sits near the far end, pushed off the link sideways
        nx, ny = -uy / np.hypot(ux, uy) * 0.16, ux / np.hypot(ux, uy) * 0.16
        ax.text(lx + 0.72 * ux + nx, ly + 0.72 * uy + ny, str(k), fontsize=8, color=BLUE, fontweight="bold",
                ha="center", va="center", zorder=8, bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none"))

    # the true neighbourhood, not yet visited
    ax.add_patch(plt.Circle((qx, qy), 1.0, fc="none", ec=YELLOW, lw=1.4, ls=(0, (3, 2)), zorder=2))
    ax.annotate("yellow dots: the query's true neighbors. The six numbers are measured before\n"
                "the search gets here; contrast (7) is measured after, from what it found.",
                (qx - 0.3, qy - 0.95), xytext=(qx - 2.3, -0.05), textcoords="data", fontsize=9, color=INK2,
                ha="left", va="top", arrowprops=dict(arrowstyle="-", color=YELLOW, lw=1))

    # ---- the six statistics, drawn where they are measured
    def tag(num, xy, dx, dy):
        ax.annotate(num, xy, xytext=(dx, dy), textcoords="offset points", fontsize=12, fontweight="bold",
                    color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.25", fc=SURFACE, ec=INK, lw=1), zorder=9)

    ax.plot([qx, lx], [qy, ly], color=ORANGE, lw=2.6, zorder=6)  # 1
    tag("1", (0.35 * qx + 0.65 * lx, 0.35 * qy + 0.65 * ly), 0, -15)
    # 2 and 6: on the neighbour whose dashed line is longest (most room for the labels)
    far_nb = max(land_nb, key=lambda i: dq[i])
    fx, fy = T(pts[far_nb], 0)
    tag("2", (0.45 * qx + 0.55 * fx, 0.45 * qy + 0.55 * fy), 0, 15)
    tag("6", (0.7 * qx + 0.3 * fx, 0.7 * qy + 0.3 * fy), 0, 15)
    # 4: on the thick link that points away from the query
    away_nb = max(land_nb, key=lambda i: np.linalg.norm(vis[i] - vq))
    ax_, ay_ = T(pts[away_nb], 0)
    tag("4", ((lx + ax_) / 2, (ly + ay_) / 2), -22, 10)
    ax.text((lx + ax_) / 2 - 0.45, (ly + ay_) / 2 + 0.42, f"{len(land_nb)} thick links", fontsize=9,
            color=INK, ha="center", va="bottom")
    # 3: to the right of the query's ghost on the middle layer, where all its dotted lines end
    sx, sy = T(q, 1)
    tag("3", (sx, sy), 30, 0)
    ax.text(sx + 0.7, sy, f"{n_desc} dotted lines in total\n= {n_desc} distance measurements\non the way down",
            fontsize=9, color=INK, ha="left", va="center", linespacing=1.3)
    tx, ty = T(q, 2)
    ex_, ey_ = T(pts[ep], 2)
    ax.plot([tx, ex_], [ty, ey_], color=ORANGE, lw=2.2, alpha=0.85, zorder=6)  # 5
    tag("5", ((tx + ex_) / 2, (ty + ey_) / 2), 0, 14)
    # 7: contrast = typical distance to a random vector / mean distance to the found top-10 (the yellow dots here)
    for i in cluster_ids:
        x, y = T(pts[i], 0)
        ax.plot([qx, x], [qy, y], color=YELLOW, lw=1.0, zorder=3)
    others = [i for i in range(n0) if i not in cluster_ids]
    typical = dq[others].mean()
    near_typ = [i for i in others if abs(dq[i] - typical) < 0.3 * typical]
    rnd = min(near_typ, key=lambda i: vis[i][1] + 0.3 * vis[i][0])  # typical distance, bottom-left corner (open space)
    rx, ry = T(pts[rnd], 0)
    ax.plot([qx, rx], [qy, ry], color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=3)
    ax.plot(rx, ry, "o", ms=8, mfc="none", mec=INK2, mew=1.4, zorder=5)
    ax.annotate("a random vector\n(typical distance)", (rx, ry), xytext=(0, -12), textcoords="offset points",
                fontsize=8.5, color=INK2, ha="center", va="top")
    tag("7", (0.3 * qx + 0.7 * rx, 0.3 * qy + 0.7 * ry), 0, 14)
    contrast_toy = dq[others].mean() / np.mean([dq[i] for i in cluster_ids])

    # ---- legend column
    lines = [
        ("Blue arrows", "the descent: on each upper layer, keep jumping to a linked node that is closer to the query "
                        "(solid), then drop one layer down at that node (dashed)."),
        ("Blue rings and dotted lines", "each dotted line is one question asked on the way down: "
                                         "'how far is this node from the query?' The small number is the order."),
        ("", ""),
        ("1  ep_dist", "length of the orange line on layer 0: how far the layer-0 entry point is from the query. "
                       "Long = the maps dropped us off far from home."),
        ("2  ep_nbr_std", "the dashed green lines are the query's distances to each neighbor of the entry point. "
                          "How different are those lengths from each other? Small = every direction looks alike."),
        ("3  descent_ndist", f"count the dotted lines: {n_desc} here. That is how many nodes had to be measured "
                             "before the search reached layer 0. Many = a zigzag trip down."),
        ("4  ep_degree", f"count the thick blue links leaving the layer-0 entry point (numbered 1 to {len(land_nb)}): "
                         f"{len(land_nb)} here. Few = a dead-end corner with few ways out."),
        ("5  global_ep_dist", "length of the orange line on the top layer: how far the query is from the node where "
                              "every search starts."),
        ("6  ep_nbr_mean", "the same green lines as 2, but their average length. Long = even the entry point's "
                           "neighbors are far from the query."),
        ("", ""),
        ("7  contrast  (not one of the six)", "measured AFTER the layer-0 search has found the query's 10 best "
                                              "neighbors (the yellow dots): the dashed grey line, the typical distance "
                                              "to a random vector, divided by the average length of the short yellow "
                                              f"lines. Toy value {contrast_toy:.1f}: the found neighbors are {contrast_toy:.0f}x "
                                              "closer than a random vector (real queries: about 1.5 to 4 on GloVe). "
                                              "Near 1 = no real neighborhood = hard."),
    ]
    y = 8.6
    for head, body in lines:
        if not head:
            y -= 0.12
            continue
        ax.text(11.4, y, head, fontsize=10, fontweight="bold", color=INK, va="top")
        wrapped = textwrap.fill(body, 46)
        ax.text(11.4, y - 0.24, wrapped, fontsize=8.8, color=INK2, va="top", linespacing=1.25)
        y -= 0.24 + 0.19 * (wrapped.count("\n") + 1) + 0.14
    ax.text(0.3, 9.15, "The six entry-time statistics (QBAT's HNSW features) and contrast, drawn on a toy HNSW graph",
            fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(0.3, 8.8, "The star is the query; it is not a node. Grey dots and lines are the index. The three layers hold "
                      "the same nodes:\nthe upper layers keep only a few of them. Lengths on the page are the real distances.",
            fontsize=10, color=INK2, va="top", linespacing=1.3)
    savefig(fig, out)


def fig_system(out="explain_system"):
    """Flowchart of the peek-then-widen system: the offline lane (once per
    index and target recall) and the online lane (every query), with the
    lookup table as the only thing that crosses between them."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

    fig, ax = plt.subplots(figsize=(16, 9.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.6)
    ax.axis("off")
    ax.grid(False)
    LANE = {"off": (5.55, 3.55), "on": (0.08, 5.17)}  # (bottom, height)
    for key, title, sub in (("off", "OFFLINE  ·  once per index and target recall T",
                             "profiling on a few thousand training queries; minutes"),
                            ("on", "ONLINE  ·  every query, inside one search",
                             "uses only what the search has already computed; no ground truth")):
        b, h = LANE[key]
        ax.add_patch(FancyBboxPatch((0.25, b), 15.5, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#f3f2ee", ec=BASE, lw=1, zorder=0))
        ax.text(0.5, b + h - 0.18, title, fontsize=12, fontweight="bold", color=ACCENT_TXT, va="top")
        ax.text(0.5, b + h - 0.52, sub, fontsize=9.5, color=INK2, va="top")

    def box(x, y, w, h, head, body, fc=SURFACE, ec=BASE, head_col=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.2, zorder=2))
        ax.text(x + 0.14, y + h - 0.14, head, fontsize=10, fontweight="bold", color=head_col, va="top", zorder=3)
        ax.text(x + 0.14, y + h - 0.44, body, fontsize=8.6, color=INK2, va="top", linespacing=1.25, zorder=3)
        return (x, y, w, h)

    def arrow(a, b, side="r", color=INK2, ls="-", label=None, lw=1.4):
        (xa, ya, wa, ha), (xb, yb, wb, hb) = a, b
        if side == "r":
            p0, p1 = (xa + wa, ya + ha / 2), (xb, yb + hb / 2)
        elif side == "d":
            p0, p1 = (xa + wa / 2, ya), (xb + wb / 2, yb + hb)
        elif side == "l":
            p0, p1 = (xa, ya + ha / 2), (xb + wb, yb + hb / 2)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw, ls=ls,
                                     shrinkA=2, shrinkB=2, zorder=4))
        if label:
            ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.12, label, fontsize=8, color=color,
                    ha="center", va="bottom", zorder=5, bbox=dict(fc=SURFACE, ec="none", pad=1))

    # ---------------- offline lane
    y0, h0 = 5.85, 2.35
    o1 = box(0.55, y0, 2.9, h0, "1  Build the index",
             "HNSW as usual (Faiss / hnswlib):\nM, efConstruction.\n\nStore one extra vector: the\nmean of the database, x̄\n(needed by contrast).")
    o2 = box(3.85, y0, 3.9, h0, "2  Profile training queries",
             "For each of ~3,000 queries (never\nused for evaluation):\n• peek at ef0, compute the\n  statistic s (edges / contrast)\n"
             "• keep searching through an ef\n  grid, record recall at each ef\n• oracle ef* = smallest ef with\n  recall ≥ T")
    o3 = box(8.15, y0, 3.7, h0, "3  Fit the table  s → ef",
             "Sort the queries by s, cut into\n10 equal bins. Each bin gets the\nsmallest ef at which 80% of its\n"
             "queries reach T (or fit a\nmonotone curve instead of bins).\nRefit when the index is rebuilt\n(edges depends on M).")
    o4 = box(12.25, y0 + 0.55, 3.3, 1.25, "TABLE", "low s  → big ef\nhigh s → ef0 (stop here)", fc="#fde9d9", ec=ORANGE, head_col=ORANGE)
    arrow(o1, o2)
    arrow(o2, o3)
    arrow(o3, o4)

    # ---------------- online lane
    y1, h1 = 2.55, 1.85
    n1 = box(0.55, y1, 2.2, h1, "A  Query arrives", "q, k = 100,\ntarget recall T")
    n2 = box(3.1, y1, 2.55, h1, "B  Descend", "Greedy walk down the\nupper layers to the\nlayer-0 entry point.\n(entry-time features\nlive here: not used)")
    n3 = box(6.0, y1, 2.75, h1, "C  Peek", "Beam search at ef0 = 100\nuntil it converges.\n≈ 2,400 distance\ncomputations on GloVe.\nThis is the first part of\nthe final search, not extra.")
    n4 = box(9.1, y1, 2.9, h1, "D  Statistic on the top-100",
             "read the search's own result\nheap (no ground truth)\n• edges: 100 link-list reads\n• contrast: 1 dot product with x̄\n→ s")
    n5 = box(12.35, y1, 3.2, h1, "E  Look up  ef = TABLE[s]", "the only thing that crosses\nfrom offline to online")
    arrow(n1, n2)
    arrow(n2, n3)
    arrow(n3, n4)
    arrow(n4, n5)
    arrow(o4, n5, side="d", color=ORANGE, ls="--", label="table")

    # decision + two exits
    dy = 0.65
    dx, dw, dh = 13.95, 1.5, 0.95
    ax.add_patch(Polygon([(dx - dw / 2, dy + dh / 2), (dx, dy + dh), (dx + dw / 2, dy + dh / 2), (dx, dy)], closed=True,
                         fc=SURFACE, ec=BASE, lw=1.2, zorder=2))
    ax.text(dx, dy + dh / 2, "ef > ef0 ?", fontsize=9.5, ha="center", va="center", fontweight="bold", zorder=3)
    ax.add_patch(FancyArrowPatch((dx, y1), (dx, dy + dh), arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.4,
                                 shrinkA=2, shrinkB=2, zorder=4))
    n6 = box(6.0, 0.45, 5.9, 1.55, "F  Keep searching  (resume, do not restart)",
             "The same search continues with the longer shortlist ef: the visited\nset and heaps are kept, pruned nodes are re-admitted from the spill\n"
             "heap. Measured in E16: exactly a fresh ef search's recall and cost.", fc="#ddebf7", ec=BLUE, head_col=BLUE)
    n7 = box(0.55, 0.45, 4.9, 1.55, "G  Return the top-k",
             "Easy queries (high s) return straight from the peek: ~30% of\nqueries on GloVe pay only ≈ 2,400 comparisons.\n"
             "Hard queries return after F with the ef they needed.", fc="#e2efda", ec=AQUA, head_col="#137a55")
    ax.add_patch(FancyArrowPatch((dx - dw / 2, dy + dh / 2), (n6[0] + n6[2], dy + dh / 2), arrowstyle="-|>", mutation_scale=14,
                                 color=INK2, lw=1.4, shrinkA=2, shrinkB=2, zorder=4))
    ax.text(dx - dw / 2 - 0.35, dy + dh / 2 + 0.12, "yes", fontsize=8.5, color=INK2, ha="center", va="bottom")
    arrow(n6, n7, side="l")
    # "no" path: from the diamond bottom, around under F, into G
    ax.add_patch(FancyArrowPatch((dx, dy), (dx, 0.2), arrowstyle="-", color=INK2, lw=1.4, zorder=4))
    ax.add_patch(FancyArrowPatch((dx, 0.2), (n7[0] + n7[2] / 2, 0.2), arrowstyle="-", color=INK2, lw=1.4, zorder=4))
    ax.add_patch(FancyArrowPatch((n7[0] + n7[2] / 2, 0.2), (n7[0] + n7[2] / 2, n7[1]), arrowstyle="-|>", mutation_scale=14,
                                 color=INK2, lw=1.4, shrinkB=2, zorder=4))
    ax.text(dx + 0.15, 0.38, "no", fontsize=8.5, color=INK2, ha="left", va="bottom")

    ax.text(0.3, 9.42, "Peek, then widen: how the per-query ef would work as a system",
            fontsize=13.5, fontweight="bold", color=INK, va="top")
    savefig(fig, out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100", choices=tuple(E12_FIGS))
    main(ap.parse_args().dataset)
