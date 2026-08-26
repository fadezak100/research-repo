"""Figures for the beginner-friendly Ada-ef explainer deck.

Every chart is computed from the real GloVe-100 data and the real fitted
estimator — no fake numbers. Also dumps the worked-example numbers to
results/explain_numbers.json so the slide text uses the same values.

Usage: .venv/bin/python -m pyhnsw.explain_figs
"""

import json
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .ada_ef import QUANTILE_STEP, AdaEf
from .experiments import RESULTS_DIR, fit_ada
from .graph import load_graph
from .search import SearchContext

FIGS = RESULTS_DIR / "figs" / "explain"

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
    }
)


def savefig(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figs/explain/{name}.png")


def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def main():
    ctx, est = fit_ada("glove100")
    ds = ctx.ds
    numbers = {}

    # ---- pick a typical, an easy, and a hard query (by Ada-ef's own score)
    infos = []
    for i in range(300):
        trace = {}
        ctx.search(ds.test[i], 10, ada=est, trace=trace)
        infos.append((i, trace.get("score"), trace.get("est_ef"), trace.get("collected")))
    infos = [t for t in infos if t[1] is not None]
    infos.sort(key=lambda t: t[1])
    # hard: low score AND a genuinely large assigned ef (score ~0 groups are
    # distorted by the zero-recall filter and just get the WAE floor)
    lowish = [t for t in infos if 2 <= t[1] <= 8] or infos[:5]
    hard = max(lowish, key=lambda t: t[2])
    easy = infos[-4]  # high score = easy
    typical = infos[len(infos) // 2]

    # ================================================== fig 1: the distance list
    qi = typical[0]
    q = ds.test[qi]
    all_dists = 1.0 - ds.train @ q  # the full distance list, for real
    mu_pred = 1.0 - float(q @ est.mu)
    var_pred = float(q @ est.sigma @ q)
    sd_pred = float(np.sqrt(var_pred))
    numbers["typical"] = {
        "query_index": int(qi),
        "n_vectors": int(len(ds.train)),
        "mu_pred": mu_pred,
        "sd_pred": sd_pred,
        "actual_mean": float(all_dists.mean()),
        "actual_sd": float(all_dists.std()),
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(all_dists, bins=160, color=BLUE, edgecolor="none")
    ax.axvline(np.min(all_dists), color=ORANGE, linewidth=2)
    ax.annotate(
        "the true nearest neighbors\nlive here, in the far left tail",
        (float(np.min(all_dists)), ax.get_ylim()[1] * 0.55),
        xytext=(30, 0), textcoords="offset points", fontsize=10, color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE),
    )
    ax.set_xlabel("distance from the query to a vector  (0 = identical, 1 = unrelated)")
    ax.set_ylabel("how many vectors")
    ax.set_title(f"One real query vs all {len(ds.train):,} GloVe vectors — every distance, computed")
    savefig(fig, "1_distance_list")

    # ================================================== fig 2: prediction overlay
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    counts, bins, _ = ax.hist(
        all_dists, bins=160, color=BLUE, edgecolor="none", label="actual distances (slow way)"
    )
    xs = np.linspace(bins[0], bins[-1], 400)
    scale = len(all_dists) * (bins[1] - bins[0])
    ax.plot(
        xs, gaussian_pdf(xs, mu_pred, sd_pred) * scale, color=ORANGE, linewidth=2.5,
        label="predicted bell curve (two numbers, instant)",
    )
    ax.annotate(
        f"center = 1 − q·(mean vector) = {mu_pred:.3f}\nwidth  = from the covariance table = {sd_pred:.3f}",
        (0.02, 0.80), xycoords="axes fraction", fontsize=10, color=INK2,
    )
    ax.set_xlabel("distance")
    ax.set_ylabel("how many vectors")
    ax.set_title("The trick: we can predict this whole shape without computing any distance")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.72))
    savefig(fig, "2_prediction_overlay")

    # ================================================== fig 3: toy mean vector
    rng = np.random.default_rng(7)
    toy = np.round(rng.uniform(-0.9, 0.9, size=(4, 3)), 1)
    toy_mean = np.round(toy.mean(axis=0), 2)
    numbers["toy"] = {"vectors": toy.tolist(), "mean": toy_mean.tolist()}
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axis("off")
    ax.grid(False)
    for r in range(4):
        for c in range(3):
            ax.text(c, 3 - r, f"{toy[r, c]:+.1f}", ha="center", va="center",
                    fontsize=15, color=INK,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f0fb", edgecolor=BASE))
        ax.text(-0.85, 3 - r, f"vector {r + 1}", ha="center", va="center", fontsize=10, color=INK2)
    for c in range(3):
        ax.text(c, 3.8, f"dim {c + 1}", ha="center", va="center", fontsize=10, color=INK2)
        ax.annotate("", (c, -0.55), (c, -0.15), arrowprops=dict(arrowstyle="->", color=MUTED))
        ax.text(c, -0.9, f"{toy_mean[c]:+.2f}", ha="center", va="center",
                fontsize=15, color=SURFACE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=ORANGE, edgecolor="none"))
    ax.text(-0.85, -0.9, "mean\nvector", ha="center", va="center", fontsize=10, color=ORANGE)
    ax.text(1, -1.55, "average each column → one “average vector” for the whole dataset",
            ha="center", fontsize=11, color=INK2)
    ax.set_xlim(-1.5, 2.6)
    ax.set_ylim(-1.8, 4.2)
    savefig(fig, "3_mean_vector_toy")

    # ================================================== fig 4: covariance scatters
    sig = est.sigma.copy()
    d = sig.shape[0]
    off = np.abs(sig - np.diag(np.diag(sig)))
    i1, j1 = np.unravel_index(np.argmax(off), off.shape)  # strongly linked pair
    # weakly linked pair with similar variances
    flat = np.argsort(np.abs(sig[np.triu_indices(d, 1)]))
    iu = np.triu_indices(d, 1)
    i0, j0 = iu[0][flat[0]], iu[1][flat[0]]
    sample = ds.train[np.random.default_rng(0).choice(len(ds.train), 2500, replace=False)]
    numbers["cov"] = {
        "linked_dims": [int(i1), int(j1)], "linked_value": float(sig[i1, j1]),
        "unlinked_dims": [int(i0), int(j0)], "unlinked_value": float(sig[i0, j0]),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9))
    for ax, (a, b), title, color in [
        (axes[0], (i1, j1),
         f"dims {i1} & {j1}: they move together\ncovariance = {sig[i1, j1]:+.4f}  (≠ 0)", BLUE),
        (axes[1], (i0, j0),
         f"dims {i0} & {j0}: unrelated\ncovariance = {sig[i0, j0]:+.5f}  (≈ 0)", AQUA),
    ]:
        ax.scatter(sample[:, a], sample[:, b], s=6, alpha=0.35, color=color, edgecolors="none")
        ax.set_xlabel(f"value in dim {a}")
        ax.set_ylabel(f"value in dim {b}")
        ax.set_title(title, fontsize=10)
    fig.suptitle("2,500 real GloVe vectors — what one cell of the covariance table measures", y=1.04)
    savefig(fig, "4_covariance_pairs")

    # ================================================== fig 5: easy vs hard collected
    nd = NormalDist()
    zs = [nd.inv_cdf(QUANTILE_STEP * (i + 1)) for i in range(5)]
    panels = []
    for label, (qi_, score_, ef_, collected_) in [("easy", easy), ("hard", hard)]:
        q_ = ds.test[qi_]
        mu_ = 1.0 - float(q_ @ est.mu)
        sd_ = float(np.sqrt(q_ @ est.sigma @ q_))
        thresholds = [mu_ + z * sd_ for z in zs]
        col = np.array(collected_)
        bins_ = np.searchsorted(thresholds, col, side="right")
        cnt = np.bincount(bins_, minlength=6)[:5]
        panels.append((label, qi_, score_, ef_, col, thresholds, cnt, mu_, sd_))
        numbers[label] = {
            "query_index": int(qi_), "mu": mu_, "sd": sd_,
            "thresholds": [float(t) for t in thresholds],
            "counts": cnt.tolist(), "n_collected": int(len(col)),
            "score": float(score_), "est_ef": int(ef_),
        }

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=False)
    fig.subplots_adjust(top=0.78, wspace=0.25)
    for ax, (label, qi_, score_, ef_, col, thr, cnt, mu_, sd_) in zip(axes, panels):
        color = AQUA if label == "easy" else ORANGE
        lo = min(col.min(), thr[0]) - 0.01
        hi = mu_ - 1.2 * sd_
        ax.hist(col[col < hi], bins=48, color=color, edgecolor="none")
        for t in thr:
            ax.axvline(t, color=MUTED, linewidth=1.1, linestyle=":")
        ax.axvline(thr[0], color=INK2, linewidth=1.4)
        inside = int(cnt[0])
        ax.set_title(
            f"{label} query:  score {score_:.0f}  →  ef = {ef_}", fontsize=12
        )
        ax.annotate(
            f"{inside} of {len(col)} visited vectors\nare inside the top-0.1% cut-off\n(solid line)",
            (0.97, 0.95) if label == "easy" else (0.03, 0.95),
            xycoords="axes fraction", fontsize=9.5, color=INK2,
            ha="right" if label == "easy" else "left", va="top",
        )
        ax.set_xlabel("distance to visited vector")
        ax.set_ylabel("how many visited vectors")
        ax.set_xlim(lo, hi)
    fig.suptitle(
        "Two real queries, same first-1000 peek — dotted lines are the five “closest 0.1%…0.5%” cut-offs",
        y=0.94, fontsize=12,
    )
    savefig(fig, "5_easy_vs_hard")

    # ================================================== fig 5b: where cut-offs come from
    h_mu, h_sd = numbers["hard"]["mu"], numbers["hard"]["sd"]
    h_thr = numbers["hard"]["thresholds"]
    n_total = len(ds.train)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), width_ratios=[1, 1.25])
    fig.subplots_adjust(top=0.80, wspace=0.14)

    ax = axes[0]
    xs = np.linspace(h_mu - 4.6 * h_sd, h_mu + 3.6 * h_sd, 600)
    ys = gaussian_pdf(xs, h_mu, h_sd)
    ax.plot(xs, ys, color=BLUE, linewidth=2.2)
    tail = xs <= h_thr[4]
    ax.fill_between(xs[tail], ys[tail], color=ORANGE, alpha=0.9)
    ax.annotate(
        "orange sliver = the closest 0.5%\nof ALL vectors — zoomed →",
        (0.02, 0.97), xycoords="axes fraction", fontsize=10, color=INK2, va="top",
    )
    ax.annotate(
        "", (h_thr[4] - 0.01, gaussian_pdf(h_thr[4], h_mu, h_sd) * 2.5),
        xytext=(h_mu - 3.0 * h_sd, ys.max() * 0.55),
        arrowprops=dict(arrowstyle="->", color=ORANGE),
    )
    ax.axvline(h_mu, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate(f" center {h_mu:.3f}", (h_mu, ys.max() * 0.55), fontsize=9, color=MUTED)
    ax.set_xlabel("distance")
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_title("predicted bell curve (hard query)\ncenter 0.841, width 0.109", fontsize=11)

    ax = axes[1]
    lo, hi = 0.47, 0.585
    xs = np.linspace(lo, hi, 400)
    ys = gaussian_pdf(xs, h_mu, h_sd)
    top = gaussian_pdf(hi, h_mu, h_sd) * 1.35
    bands = [h_mu - 5 * h_sd] + list(h_thr)
    for i in range(5):
        seg = (xs >= bands[i]) & (xs <= bands[i + 1])
        ax.fill_between(xs[seg], ys[seg], color=ORANGE, alpha=0.85 - 0.15 * i)
    ax.plot(xs, ys, color=BLUE, linewidth=2.2)
    labels_y = [0.96, 0.86, 0.96, 0.86, 0.96]
    for i, t in enumerate(h_thr):
        ax.axvline(t, color=INK2, linewidth=1.2)
        ax.annotate(
            f"{0.1 * (i + 1):.1f}%\n{t:.3f}",
            (t, top * labels_y[i]), xytext=(0, 2), textcoords="offset points",
            ha="center", va="top", fontsize=9, color=INK2,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor=BASE),
        )
    ax.annotate(
        f"“closest 0.1%” means: only ≈ {int(0.001 * n_total):,} of the\n"
        f"{n_total:,} vectors can lie left of that line",
        (0.03, 0.55), xycoords="axes fraction", fontsize=10, color=INK2, va="top",
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top)
    ax.set_xlabel("distance (left tail only)")
    ax.set_yticks([])
    ax.set_title("the five cut-off lines\ncut-off = center − (3.09 … 2.58) × width", fontsize=11)
    fig.suptitle(
        "A cut-off is a distance value on the ruler, not a vector — the curve's two numbers tell us where each “closest X%” line sits",
        y=0.97, fontsize=12.5,
    )
    savefig(fig, "5b_cutoffs")

    # ================================================== fig 6: the cheat sheet
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    table = est.table
    picks = []
    # score ~3 (genuinely hard), ~30 (middle), ~90 (easy); the score-0 group is
    # distorted by the zero-recall filter, so skip it for the illustration
    for want in [3, 30, 90]:
        g = min(table, key=lambda t: abs(t[0] - want))
        if g not in picks:
            picks.append(g)
    colors = [ORANGE, YELLOW, AQUA]
    for (score_key, entries), color in zip(picks, colors):
        entries = sorted(entries)[:12]
        efs = [e for e, _ in entries]
        recs = [r for _, r in entries]
        ax.plot(efs, recs, "-o", color=color, markersize=5,
                label=f"practice queries with score ≈ {score_key}")
        passing = [e for e, r in entries if r >= est.target_recall]
        if passing:
            ax.scatter([passing[0]], [dict(entries)[passing[0]]], s=140, facecolors="none",
                       edgecolors=INK, linewidths=1.6, zorder=5)
    ax.axhline(est.target_recall, color=MUTED, linestyle=":", linewidth=1.4)
    ax.annotate(f"target recall {est.target_recall}", (ax.get_xlim()[1], est.target_recall),
                ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("ef we tried (log scale)")
    ax.set_ylabel("fraction of true neighbors found")
    ax.set_title("The cheat-sheet, three real rows: low-score groups need a much bigger ef\n(circled = the ef the table hands back for that group)")
    ax.legend(loc="lower right", fontsize=9)
    savefig(fig, "6_cheat_sheet")

    # ================================================== fig 7: score → ef (all queries)
    e5 = json.loads((RESULTS_DIR / "e5_adaef_glove100.json").read_text())
    scores = [s for s in e5["ada"]["scores"] if s is not None]
    efs = [e for e in e5["ada"]["est_efs"] if e is not None]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter(scores, efs, s=10, alpha=0.4, color=BLUE, edgecolors="none")
    ax.set_yscale("log")
    ax.set_xlabel("query score (0 = nothing close found early → hard, 100 = lots found → easy)")
    ax.set_ylabel("ef assigned (log scale)")
    ax.set_title("All 1000 real queries: high score → small search, low score → big search")
    savefig(fig, "7_score_vs_ef")

    (RESULTS_DIR / "explain_numbers.json").write_text(json.dumps(numbers, indent=1))
    print("wrote results/explain_numbers.json")


if __name__ == "__main__":
    main()
