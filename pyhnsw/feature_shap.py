"""E15 — which query-time statistics does a learned budget predictor rely on?

E13 scores each statistic on its own (Spearman with recall / the oracle ef).
That cannot say what a statistic adds ON TOP of the others, and many of ours
are near-copies (contrast10 ~ d10, edges ~ avgdist). This experiment looks at
all of them at once, the way QBAT (Sec. 3) does for its centroid-distance
features: train a gradient-boosted tree model (LightGBM) to predict a query's
ideal budget from the statistics, and attribute its predictions with SHAP.

Input: results/e13_online_stats_{dataset}.json only (no searches).

Features   the 14 deployable query-time statistics of E13; avgdist among the
           found set is left out (BFS, lab-only). A second SHAP run with
           avgdist included is reported as `with_avgdist`.
Targets    ef_0.9 / ef_0.8  log2 of the oracle ef (smallest grid ef reaching
                            recall 0.9 / 0.8; censored queries = 2 x the cap)
           recall           recall@100 of the ef=100 prefix search
           A target is skipped when one value covers > 90% of the queries
           (SIFT1M ef_0.8: 961 of 1000 queries need ef=100).
Protocol   5-fold cross-validation x 3 seeds. SHAP values (TreeExplainer,
           tree-path-dependent) are computed on the held-out fold only, so
           every query's attribution comes from a model that never saw it.

SHAP splits the credit of a shared signal among correlated features, so it
is read together with three retrain checks on held-out R^2 (same folds):
  single       each feature alone
  family       each family alone, and all features minus that family
               (distance / graph / effort / entry)
  forward      greedy forward selection: which feature adds the most next
  top-k        the k features with the largest mean |SHAP| (QBAT's check:
               do the top two carry the model?)

Faiss is never imported here: Faiss and LightGBM each bring an OpenMP
runtime and the process segfaults when both are loaded.

Usage:  .venv/bin/python -m pyhnsw.feature_shap [--dataset glove100|sift1m]
Output: results/e15_shap_{dataset}.json   (~1 min)
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import shap
from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

FAMILIES = {
    "distance": ["d10", "d100", "contrast10", "contrast100", "spread"],
    "graph": ["edges"],
    "effort": ["n_hops", "n_dist"],
    "entry": ["ep_dist", "ep_nbr_mean", "ep_nbr_std", "ep_degree", "descent_ndist", "global_ep_dist"],
}
FEATURES = [f for fam in FAMILIES.values() for f in fam]
LAB_ONLY = ["avgdist"]
N_FOLDS, SEEDS, FORWARD_STEPS = 5, (0, 1, 2), 5
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=1.0, verbose=-1, n_jobs=1)


def rho(x, y):
    return float(spearmanr(x, y).statistic)


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def folds(n, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::N_FOLDS] for i in range(N_FOLDS)]


def cv_fit(X, y, seed, with_shap=False):
    """Out-of-fold predictions (and SHAP values) for one seed."""
    n = len(y)
    pred, sv, per_fold = np.empty(n), np.zeros(X.shape) if with_shap else None, []
    for te in folds(n, seed):
        tr = np.setdiff1d(np.arange(n), te)
        model = lgb.LGBMRegressor(random_state=seed, **PARAMS).fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        if with_shap:
            sv[te] = shap.TreeExplainer(model).shap_values(X[te])
            per_fold.append(np.abs(sv[te]).mean(axis=0))
    return pred, sv, per_fold


def heldout(X, y, seeds=SEEDS):
    """Mean held-out R^2 / Spearman over the seeds."""
    s = [(r2(y, p), rho(y, p)) for p in (cv_fit(X, y, seed)[0] for seed in seeds)]
    return {"r2": float(np.mean([a for a, _ in s])), "spearman": float(np.mean([b for _, b in s]))}


def shap_run(X, y, names):
    sv_all, imp_folds, preds = [], [], []
    for seed in SEEDS:
        pred, sv, per_fold = cv_fit(X, y, seed, with_shap=True)
        sv_all.append(sv)
        imp_folds += per_fold
        preds.append(pred)
    sv = np.mean(sv_all, axis=0)  # per query: mean of its 3 out-of-fold attributions
    imp, imp_folds = np.abs(sv).mean(axis=0), np.array(imp_folds)
    feats = []
    for j, name in enumerate(names):
        feats.append({
            "feature": name, "mean_abs_shap": float(imp[j]), "share": float(imp[j] / imp.sum()),
            "std_over_folds": float(imp_folds[:, j].std()),
            # sign of the effect: + = a high value pushes the prediction up
            "direction": rho(X[:, j], sv[:, j]) if imp[j] > 0 else 0.0,
        })
    feats.sort(key=lambda f: -f["mean_abs_shap"])
    fit = {"r2": float(np.mean([r2(y, p) for p in preds])), "spearman": float(np.mean([rho(y, p) for p in preds]))}
    return sv, feats, fit


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    args = ap.parse_args()
    dataset, t0 = args.dataset, time.perf_counter()
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    e13 = json.loads((RESULTS_DIR / f"e13_online_stats_{dataset}.json").read_text())
    pq = e13["per_query"]
    X = np.column_stack([np.array(pq[f], dtype=float) for f in FEATURES])
    X_lab = np.column_stack([X] + [np.array(pq[f], dtype=float) for f in LAB_ONLY])
    col = {f: j for j, f in enumerate(FEATURES)}
    all_targets = {"ef_0.9": np.log2(np.array(pq["ef_min_0.9"])), "ef_0.8": np.log2(np.array(pq["ef_min_0.8"])),
                   "recall": np.array(pq["recall"], dtype=float)}

    results, skipped = {}, {}
    for tname, y in all_targets.items():
        top_share = float(np.unique(y, return_counts=True)[1].max() / len(y))
        if top_share > 0.9:
            skipped[tname] = f"one value covers {100 * top_share:.0f}% of the queries"
            print(f"{dataset} / {tname}: skipped ({skipped[tname]})")
            continue
        sv, feats, fit = shap_run(X, y, FEATURES)
        fam_imp = {fam: float(np.abs(sv[:, [col[f] for f in fs]].sum(axis=1)).mean()) for fam, fs in FAMILIES.items()}
        _, feats_lab, fit_lab = shap_run(X_lab, y, FEATURES + LAB_ONLY)

        single = {f: heldout(X[:, [col[f]]], y) for f in FEATURES}
        family = {fam: {"alone": heldout(X[:, [col[f] for f in fs]], y),
                        "all_minus": heldout(X[:, [j for f, j in col.items() if f not in fs]], y)}
                  for fam, fs in FAMILIES.items()}
        ranked = [f["feature"] for f in feats]
        top_k = {str(k): {"features": ranked[:k], **heldout(X[:, [col[f] for f in ranked[:k]]], y)} for k in (1, 2, 3, 5)}

        chosen, forward = [], []
        for _ in range(FORWARD_STEPS):  # one seed per candidate; the chosen set is re-scored on all seeds
            cand = {f: heldout(X[:, [col[g] for g in chosen + [f]]], y, seeds=SEEDS[:1])["r2"]
                    for f in FEATURES if f not in chosen}
            best = max(cand, key=cand.get)
            chosen.append(best)
            forward.append({"added": best, **heldout(X[:, [col[g] for g in chosen]], y)})

        results[tname] = {
            "fit_all_features": fit, "shap": feats, "shap_by_family": fam_imp,
            "with_avgdist": {"fit": fit_lab, "shap": feats_lab},
            "single_feature": single, "family": family, "top_k_by_shap": top_k, "forward_selection": forward,
            "per_query": {"shap": np.round(sv, 5).tolist(), "target": y.tolist()},
        }

        print(f"\n{dataset} / target {tname}: all {len(FEATURES)} features, held-out R2 {fit['r2']:.3f}, "
              f"Spearman {fit['spearman']:+.3f}")
        print(f"{'feature':>16} {'mean|SHAP|':>11} {'share':>7} {'±folds':>8} {'direction':>10} {'alone R2':>9}")
        for f in feats:
            print(f"{f['feature']:>16} {f['mean_abs_shap']:>11.4f} {100 * f['share']:>6.1f}% {f['std_over_folds']:>8.4f} "
                  f"{f['direction']:>+10.2f} {single[f['feature']]['r2']:>9.3f}")
        print("family      mean|SHAP|   alone R2   all-minus R2")
        for fam in FAMILIES:
            print(f"{fam:>10} {fam_imp[fam]:>11.4f} {family[fam]['alone']['r2']:>10.3f} {family[fam]['all_minus']['r2']:>14.3f}")
        print("top-k by SHAP: " + ";  ".join(f"k={k}: R2 {v['r2']:.3f}" for k, v in top_k.items()))
        print("forward selection: " + "  ->  ".join(f"+{s['added']} {s['r2']:.3f}" for s in forward))
        print(f"with avgdist (lab-only): R2 {fit_lab['r2']:.3f}; top 4 by SHAP: "
              + ", ".join(f"{f['feature']} {100 * f['share']:.0f}%" for f in feats_lab[:4]))

    out = {
        "dataset": dataset, "source": f"e13_online_stats_{dataset}.json", "n_queries": int(len(X)),
        "features": FEATURES, "families": FAMILIES, "lab_only": LAB_ONLY,
        "model": {"library": f"lightgbm {lgb.__version__}", "params": PARAMS, "shap": f"shap {shap.__version__} TreeExplainer"},
        "protocol": {"folds": N_FOLDS, "seeds": list(SEEDS), "shap_on": "held-out fold"},
        "feature_values": {f: pq[f] for f in FEATURES},
        "skipped_targets": skipped, "results": results, "elapsed_s": time.perf_counter() - t0,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"e15_shap_{dataset}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nsaved {path.relative_to(RESULTS_DIR.parent)}  [{out['elapsed_s']:.0f}s]")


if __name__ == "__main__":
    main()
