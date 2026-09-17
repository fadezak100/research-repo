"""Dataset + Faiss HNSW index + extracted level-0 graph, and the two ways to
search it.

All searches go through Faiss (`IndexHNSWFlat.search`). Two entry points:

  search(index, Q, k, ef)            batched, multithreaded, no cost — for
                                     sweeps where only the results matter
  search_with_cost(index, Q, k, ef)  one query at a time, single-threaded,
                                     reading faiss.cvar.hnsw_stats after each
                                     query -> ids, ndis, nhops. ndis is the
                                     number of distance computations, the
                                     cost metric used everywhere in this repo.

Faiss uses max(efSearch, k) as the beam, so ef < k behaves as ef = k.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .graph import DATA_DIR, Dataset, HnswGraph, load_graph

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_QUERIES = 1000

_CACHE = {}


@dataclass
class Ctx:
    ds: Dataset
    index: faiss.Index
    graph: HnswGraph


def load(dataset, m=16, ef_construction=200):
    """Dataset, Faiss index and extracted graph (all cached on disk and in
    memory)."""
    key = (dataset, m, ef_construction)
    if key not in _CACHE:
        ds, index, graph = load_graph(dataset, m, ef_construction)
        _CACHE[key] = Ctx(ds, index, graph)
    return _CACHE[key]


def search(index, Q, k, ef, threads=None):
    """Batched search; returns ids of shape (len(Q), k)."""
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    if Q.ndim == 1:
        Q = Q[None, :]
    if threads:
        faiss.omp_set_num_threads(threads)
    index.hnsw.efSearch = int(ef)
    _, ids = index.search(Q, int(k))
    return ids


def search_with_cost(index, Q, k, ef):
    """One query at a time, single-threaded, so faiss.cvar.hnsw_stats gives
    the per-query cost. Returns (ids (n,k), ndis (n,), nhops (n,))."""
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    if Q.ndim == 1:
        Q = Q[None, :]
    stats = faiss.cvar.hnsw_stats
    prev_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    index.hnsw.efSearch = int(ef)
    ids = np.empty((len(Q), k), dtype=np.int64)
    ndis = np.empty(len(Q), dtype=np.int64)
    nhops = np.empty(len(Q), dtype=np.int64)
    try:
        for i in range(len(Q)):
            stats.reset()
            _, I = index.search(Q[i : i + 1], int(k))
            ids[i] = I[0]
            ndis[i] = stats.ndis
            nhops[i] = stats.nhops
    finally:
        faiss.omp_set_num_threads(prev_threads)
    return ids, ndis, nhops


def recall_at_k(found_ids, gt_row, k):
    return len(set(np.asarray(found_ids)[:k].tolist()) & set(np.asarray(gt_row)[:k].tolist())) / k


def gt_for_k(dataset, k, n_queries=N_QUERIES):
    """Exact top-k ground truth for the external (test) queries. The hdf5
    files ship 100 neighbors; for k > 100 brute-force with a flat Faiss index
    over the first n_queries test vectors and cache the result."""
    ds = load(dataset).ds
    if k <= ds.ground_truth.shape[1]:
        return ds.ground_truth
    cache = DATA_DIR / f"{dataset}_gt{k}_q{n_queries}.npy"
    if cache.exists():
        return np.load(cache)
    d = ds.train.shape[1]
    flat = faiss.IndexFlatIP(d) if ds.metric == "cosine" else faiss.IndexFlatL2(d)
    flat.add(ds.train)
    _, gt = flat.search(ds.test[:n_queries], k)
    overlap = np.mean(
        [len(set(gt[i, :100].tolist()) & set(ds.ground_truth[i, :100].tolist())) / 100
         for i in range(n_queries)]
    )
    print(f"gt{k} sanity: top-100 overlap with hdf5 gt = {overlap:.4f}")
    assert overlap > 0.999, "brute-force GT disagrees with shipped ground truth"
    np.save(cache, gt)
    return gt


def save(name, payload):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"saved {path.relative_to(RESULTS_DIR.parent)}")
    return path
