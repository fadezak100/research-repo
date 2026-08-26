"""Experiment runner: E1–E6 from the meeting plan.

Each experiment writes a JSON file into results/. Run all with:

    .venv/bin/python -m pyhnsw.experiments all

or a single one, e.g.:

    .venv/bin/python -m pyhnsw.experiments e2_sift1m
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from .ada_ef import AdaEf
from .graph import DATA_DIR, load_graph
from .search import SearchContext, recall_at_k

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
K = 10
N_QUERIES = 1000
EF_SWEEP = [10, 20, 40, 80, 160, 320, 640]
PIP_DEFAULT = (0.95, 30)  # gamma, delta — defaults used in the Ada-ef paper
PIP_GRID = [(g, d) for g in (0.8, 0.9, 0.95) for d in (10, 30, 50)]
TARGET_RECALL = 0.95

# E8: baseline ef sweeps per k (ef must be >= k for the beam to hold k results)
EF_BY_K = {
    100: [100, 150, 200, 300, 400, 600, 800, 1200],
    1000: [1000, 1250, 1500, 2000, 3000, 4000],
}
E8_PIP_DELTAS = [30, 100, 300]

_CTX_CACHE = {}


def get_ctx(dataset):
    if dataset not in _CTX_CACHE:
        ds, _, graph = load_graph(dataset)
        _CTX_CACHE[dataset] = SearchContext(ds, graph)
    return _CTX_CACHE[dataset]


def run_config(ctx, n_queries=N_QUERIES, k=K, gt=None, sub_ks=None, **kw):
    """Run one search config over the query set; per-query recalls + costs.

    gt overrides ds.ground_truth (needed when k > the 100 neighbors shipped in
    the hdf5). sub_ks additionally records recall@j of the result-list prefix
    for each j in sub_ks (PIP-paper-style R@10/R@100/R@1000 from one run).
    """
    ds = ctx.ds
    if gt is None:
        gt = ds.ground_truth
    recalls, n_dists, scores, est_efs = [], [], [], []
    sub = {j: [] for j in (sub_ks or [])}
    t0 = time.perf_counter()
    for i in range(n_queries):
        ids, st = ctx.search(ds.test[i], k, **kw)
        recalls.append(recall_at_k(ids, gt[i], k))
        n_dists.append(st["n_dist"])
        for j in sub:
            sub[j].append(recall_at_k(ids, gt[i], j))
        if "score" in st:
            scores.append(st["score"])
            est_efs.append(st["est_ef"])
    elapsed = time.perf_counter() - t0
    recalls, n_dists = np.array(recalls), np.array(n_dists)
    out = {
        "recall_avg": float(recalls.mean()),
        "recall_p5": float(np.percentile(recalls, 5)),
        "recall_p1": float(np.percentile(recalls, 1)),
        "n_dist_avg": float(n_dists.mean()),
        "ms_per_query": 1000 * elapsed / n_queries,
        "recalls": recalls.tolist(),
        "n_dists": n_dists.tolist(),
    }
    for j in sub:
        out[f"recall_at_{j}"] = float(np.mean(sub[j]))
    if scores:
        out["scores"] = scores
        out["est_efs"] = est_efs
    return out


def save(name, payload):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"saved {path}")


# ---------------------------------------------------------------- E1
def e1_validation():
    """Python FixedEf vs Faiss C++ recall on SIFT1M — engine correctness."""
    ds, index, graph = load_graph("sift1m")
    ctx = get_ctx("sift1m")
    rows = []
    for ef in EF_SWEEP:
        index.hnsw.efSearch = ef
        _, found = index.search(ds.test[:N_QUERIES], K)
        faiss_recall = float(
            np.mean(
                [recall_at_k(found[i], ds.ground_truth[i], K) for i in range(N_QUERIES)]
            )
        )
        r = run_config(ctx, ef=ef)
        rows.append(
            {"ef": ef, "py_recall": r["recall_avg"], "faiss_recall": faiss_recall,
             "n_dist_avg": r["n_dist_avg"], "ms_per_query": r["ms_per_query"]}
        )
        print(f"ef={ef}: py={r['recall_avg']:.4f} faiss={faiss_recall:.4f}")
    save("e1_validation_sift1m", {"k": K, "n_queries": N_QUERIES, "rows": rows})


# ---------------------------------------------------------------- E2
def e2(dataset):
    """Baseline ef sweep vs PIP (default + grid) — recall/cost tradeoff."""
    ctx = get_ctx(dataset)
    out = {"k": K, "n_queries": N_QUERIES, "baseline": [], "pip_sweep": [], "pip_grid": []}
    for ef in EF_SWEEP:
        r = run_config(ctx, ef=ef)
        del r["recalls"], r["n_dists"]
        out["baseline"].append({"ef": ef, **r})
        print(f"[{dataset}] baseline ef={ef}: recall={r['recall_avg']:.4f} nd={r['n_dist_avg']:.0f}")
    for ef in EF_SWEEP:
        r = run_config(ctx, ef=ef, pip=PIP_DEFAULT)
        del r["recalls"], r["n_dists"]
        out["pip_sweep"].append({"ef": ef, "gamma": PIP_DEFAULT[0], "delta": PIP_DEFAULT[1], **r})
        print(f"[{dataset}] pip ef={ef}: recall={r['recall_avg']:.4f} nd={r['n_dist_avg']:.0f}")
    for gamma, delta in PIP_GRID:
        r = run_config(ctx, ef=320, pip=(gamma, delta))
        del r["recalls"], r["n_dists"]
        out["pip_grid"].append({"ef": 320, "gamma": gamma, "delta": delta, **r})
        print(f"[{dataset}] pip grid g={gamma} d={delta}: recall={r['recall_avg']:.4f} nd={r['n_dist_avg']:.0f}")
    save(f"e2_pip_{dataset}", out)


# ---------------------------------------------------------------- E3
def e3(dataset="sift1m"):
    """Saturation anatomy: per-hop phi / result-set size for sample queries."""
    ctx = get_ctx(dataset)
    ds = ctx.ds
    # pick an easy and a hard query by baseline cost at ef=320
    costs = []
    for i in range(200):
        _, st = ctx.search(ds.test[i], K, ef=320)
        costs.append(st["n_dist"])
    easy, hard = int(np.argmin(costs)), int(np.argmax(costs))
    out = {"k": K, "queries": {}}
    for label, qi in [("easy", easy), ("hard", hard)]:
        trace = {}
        # trace with patience *tracking* but no early stop (delta = huge)
        ctx.search(ds.test[qi], K, ef=320, pip=(0.95, 10**9), trace=trace)
        out["queries"][label] = {
            "query_index": qi,
            "phi": trace.get("phi", []),
            "n_results": trace.get("n_results", []),
            "counter": trace.get("counter", []),
        }
        print(f"[{dataset}] {label} q{qi}: {len(trace.get('phi', []))} hops")
    save(f"e3_saturation_{dataset}", out)


# ---------------------------------------------------------------- E4
def e4(dataset="glove100"):
    """Per-query recall skew at fixed ef, baseline vs PIP."""
    ctx = get_ctx(dataset)
    out = {"k": K, "n_queries": N_QUERIES, "configs": {}}
    for label, kw in [
        ("baseline_ef40", {"ef": 40}),
        ("baseline_ef160", {"ef": 160}),
        ("pip_ef320", {"ef": 320, "pip": PIP_DEFAULT}),
    ]:
        r = run_config(ctx, **kw)
        out["configs"][label] = {
            "recalls": r["recalls"], "n_dist_avg": r["n_dist_avg"],
            "recall_avg": r["recall_avg"], "recall_p5": r["recall_p5"], "recall_p1": r["recall_p1"],
        }
        print(f"[{dataset}] {label}: avg={r['recall_avg']:.4f} p5={r['recall_p5']:.2f} p1={r['recall_p1']:.2f}")
    save(f"e4_recall_skew_{dataset}", out)


# ---------------------------------------------------------------- E8
def gt_for_k(dataset, k, n_queries=N_QUERIES):
    """Exact top-k ground truth. hdf5 files ship 100 neighbors; for k > 100
    brute-force with a flat Faiss index over the first n_queries test vectors
    and cache the result."""
    ctx = get_ctx(dataset)
    ds = ctx.ds
    if k <= ds.ground_truth.shape[1]:
        return ds.ground_truth
    cache = DATA_DIR / f"{dataset}_gt{k}_q{n_queries}.npy"
    if cache.exists():
        return np.load(cache)
    import faiss

    d = ds.train.shape[1]
    index = (
        faiss.IndexFlatIP(d) if ds.metric == "cosine" else faiss.IndexFlatL2(d)
    )
    index.add(ds.train)
    _, gt = index.search(ds.test[:n_queries], k)
    # sanity: top-100 prefix must agree with the shipped ground truth
    overlap = np.mean(
        [
            len(set(gt[i, :100].tolist()) & set(ds.ground_truth[i, :100].tolist())) / 100
            for i in range(n_queries)
        ]
    )
    print(f"gt{k} sanity: top-100 overlap with hdf5 gt = {overlap:.4f}")
    assert overlap > 0.999, "brute-force GT disagrees with shipped ground truth"
    np.save(cache, gt)
    return gt


def e8(dataset, k):
    """PIP-paper regime: baseline ef sweep vs PIP at large k (100 / 1000).

    At k=1000 also records R@10/R@100/R@1000 of each run — the PIP paper's
    Table-1 metrics (their claim: recall unchanged, big efficiency win)."""
    ctx = get_ctx(dataset)
    gt = gt_for_k(dataset, k)
    sub_ks = [10, 100, 1000] if k == 1000 else [10, 100]
    efs = EF_BY_K[k]
    pip_ef = efs[-1]
    out = {"k": k, "n_queries": N_QUERIES, "baseline": [], "pip": []}
    for ef in efs:
        r = run_config(ctx, k=k, gt=gt, sub_ks=sub_ks, ef=ef)
        out["baseline"].append({"ef": ef, **r})
        print(f"[{dataset} k={k}] baseline ef={ef}: recall={r['recall_avg']:.4f} "
              f"p5={r['recall_p5']:.2f} nd={r['n_dist_avg']:.0f}")
    # γ=0.95 comes from the Ada-ef paper's PIP baseline; Lucene's production
    # defaults are γ=0.995, Δ=max(7, 0.3k) (PatienceKnnVectorQuery) — at large
    # k the loose γ=0.95 lets 5% of the top-k churn per hop and still count as
    # "saturated", so both parameterizations are worth showing.
    configs = [(0.95, d) for d in E8_PIP_DELTAS]
    lucene_delta = max(7, int(0.3 * k))
    configs += [(0.995, lucene_delta), (0.999, lucene_delta)]
    for gamma, delta in configs:
        r = run_config(ctx, k=k, gt=gt, sub_ks=sub_ks, ef=pip_ef, pip=(gamma, delta))
        out["pip"].append({"ef": pip_ef, "gamma": gamma, "delta": delta, **r})
        print(f"[{dataset} k={k}] pip g={gamma} d={delta}: recall={r['recall_avg']:.4f} "
              f"p5={r['recall_p5']:.2f} nd={r['n_dist_avg']:.0f}")
    save(f"e8_ksweep_{dataset}_k{k}", out)


def e8_addendum(dataset, k):
    """Append the Lucene-default PIP configs to an already-saved e8 json."""
    path = RESULTS_DIR / f"e8_ksweep_{dataset}_k{k}.json"
    out = json.loads(path.read_text())
    ctx = get_ctx(dataset)
    gt = gt_for_k(dataset, k)
    sub_ks = [10, 100, 1000] if k == 1000 else [10, 100]
    pip_ef = EF_BY_K[k][-1]
    lucene_delta = max(7, int(0.3 * k))
    done = {(r["gamma"], r["delta"]) for r in out["pip"]}
    for gamma in (0.995, 0.999):
        if (gamma, lucene_delta) in done:
            continue
        r = run_config(ctx, k=k, gt=gt, sub_ks=sub_ks, ef=pip_ef, pip=(gamma, lucene_delta))
        out["pip"].append({"ef": pip_ef, "gamma": gamma, "delta": lucene_delta, **r})
        print(f"[{dataset} k={k}] pip g={gamma} d={lucene_delta}: "
              f"recall={r['recall_avg']:.4f} p5={r['recall_p5']:.2f} nd={r['n_dist_avg']:.0f}")
    path.write_text(json.dumps(out, indent=1))
    print(f"updated {path}")


# ---------------------------------------------------------------- E5 + E6
def fit_ada(dataset="glove100", target=TARGET_RECALL):
    ctx = get_ctx(dataset)
    cache = RESULTS_DIR / f"ada_ef_{dataset}_k{K}_t{int(target * 100)}"
    if (cache.parent / (cache.name + ".json")).exists():
        est = AdaEf.load(cache)
        print(f"loaded cached Ada-ef estimator (WAE={est.wae:.0f})")
    else:
        t0 = time.perf_counter()
        est = AdaEf.fit(ctx, K, target)
        est.timings["total_s"] = time.perf_counter() - t0
        RESULTS_DIR.mkdir(exist_ok=True)
        est.save(cache)
    return ctx, est


def e5(dataset="glove100"):
    """Ada-ef declarative recall vs baseline sweep vs PIP."""
    ctx, est = fit_ada(dataset)
    r = run_config(ctx, ada=est)
    out = {
        "k": K, "n_queries": N_QUERIES, "target_recall": TARGET_RECALL,
        "wae": est.wae, "table": est.table,
        "offline_timings": getattr(est, "timings", None),
        "ada": r,
    }
    print(f"[{dataset}] ada-ef: avg={r['recall_avg']:.4f} p5={r['recall_p5']:.2f} "
          f"p1={r['recall_p1']:.2f} nd={r['n_dist_avg']:.0f}")
    save(f"e5_adaef_{dataset}", out)


def e6(dataset="glove100"):
    """Head-to-head per-query PIP vs Ada-ef + hybrid (Ada-ef ef + patience)."""
    ctx, est = fit_ada(dataset)
    out = {"k": K, "n_queries": N_QUERIES, "target_recall": TARGET_RECALL, "configs": {}}
    for label, kw in [
        ("pip", {"ef": 320, "pip": PIP_DEFAULT}),
        ("ada", {"ada": est}),
        # hybrid: Ada-ef budget + patience early-exit after the ef estimate.
        # delta=400 is a fixed patience; delta=0.5 scales patience with the
        # per-query estimated ef (our proposed distribution-aware patience).
        ("hybrid_fixed", {"ada": est, "pip": (0.95, 400)}),
        ("hybrid_scaled", {"ada": est, "pip": (0.95, 0.5)}),
    ]:
        r = run_config(ctx, **kw)
        out["configs"][label] = r
        print(f"[{dataset}] {label}: avg={r['recall_avg']:.4f} p5={r['recall_p5']:.2f} "
              f"p1={r['recall_p1']:.2f} nd={r['n_dist_avg']:.0f}")
    save(f"e6_head_to_head_{dataset}", out)


EXPERIMENTS = {
    "e1": e1_validation,
    "e2_sift1m": lambda: e2("sift1m"),
    "e2_glove100": lambda: e2("glove100"),
    "e3": lambda: e3("sift1m"),
    "e4": lambda: e4("glove100"),
    "e5": lambda: e5("glove100"),
    "e6": lambda: e6("glove100"),
    "e8_glove100_k100": lambda: e8("glove100", 100),
    "e8_glove100_k1000": lambda: e8("glove100", 1000),
    "e8_sift1m_k100": lambda: e8("sift1m", 100),
    "e8_sift1m_k1000": lambda: e8("sift1m", 1000),
}


def main():
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(EXPERIMENTS)
    for name in names:
        print(f"=== {name} ===")
        t0 = time.perf_counter()
        EXPERIMENTS[name]()
        print(f"=== {name} done in {time.perf_counter() - t0:.0f}s ===\n")


if __name__ == "__main__":
    main()
