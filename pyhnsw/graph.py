"""Build an HNSW index with Faiss and extract its graph into numpy arrays.

Faiss does the expensive C++ index construction; we pull the adjacency lists
out (`index.hnsw.neighbors/offsets/levels`) so the Python search loop in
search.py has full control over traversal — which is what both PIP and Ada-ef
need to hook into.

Datasets are ann-benchmarks hdf5 files living in faiss/data/.
"""

from dataclasses import dataclass
from pathlib import Path

import faiss
import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "faiss" / "data"

DATASETS = {
    # name -> (hdf5 file, metric)
    "sift1m": ("sift-128-euclidean.hdf5", "l2"),
    "glove100": ("glove-100-angular.hdf5", "cosine"),
}


@dataclass
class Dataset:
    name: str
    metric: str  # "l2" or "cosine"
    train: np.ndarray  # (n, d) float32; normalized when metric == "cosine"
    test: np.ndarray  # (nq, d) float32; normalized when metric == "cosine"
    ground_truth: np.ndarray  # (nq, 100) exact top-100 ids


def load_dataset(name: str) -> Dataset:
    fname, metric = DATASETS[name]
    with h5py.File(DATA_DIR / fname, "r") as f:
        train = np.ascontiguousarray(f["train"], dtype=np.float32)
        test = np.ascontiguousarray(f["test"], dtype=np.float32)
        gt = np.array(f["neighbors"])
    if metric == "cosine":
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        test /= np.linalg.norm(test, axis=1, keepdims=True)
    return Dataset(name, metric, train, test, gt)


def build_or_load_index(ds: Dataset, m: int = 16, ef_construction: int = 200):
    cache = DATA_DIR / f"{ds.name}_hnsw_m{m}_efc{ef_construction}.faiss"
    if cache.exists():
        return faiss.read_index(str(cache))
    faiss_metric = (
        faiss.METRIC_INNER_PRODUCT if ds.metric == "cosine" else faiss.METRIC_L2
    )
    index = faiss.IndexHNSWFlat(ds.train.shape[1], m, faiss_metric)
    index.hnsw.efConstruction = ef_construction
    index.add(ds.train)
    faiss.write_index(index, str(cache))
    return index


@dataclass
class HnswGraph:
    """HNSW adjacency extracted from a Faiss index.

    adj0: (n, 2M) int32 level-0 neighbor lists, -1 padded.
    upper: dict level -> {node_id: int32 array of neighbors} for levels >= 1.
    """

    adj0: np.ndarray
    upper: dict
    entry_point: int
    max_level: int
    m: int


def extract_graph(index) -> HnswGraph:
    hnsw = index.hnsw
    neighbors = faiss.vector_to_array(hnsw.neighbors)
    offsets = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    levels = faiss.vector_to_array(hnsw.levels)  # levels[i] = 1 + top level of i
    cum = faiss.vector_to_array(hnsw.cum_nneighbor_per_level)
    n = levels.shape[0]
    m0 = cum[1] - cum[0]  # level-0 degree = 2M

    # Level 0: every node's block starts with its 2M level-0 slots.
    adj0 = neighbors[(offsets[:-1, None] + np.arange(m0)[None, :])].astype(np.int32)

    # Upper levels: only nodes with levels[i] > lvl have entries.
    upper = {}
    for lvl in range(1, hnsw.max_level + 1):
        node_ids = np.where(levels > lvl)[0]
        lo, hi = cum[lvl], cum[lvl + 1]
        table = {}
        for i in node_ids:
            nbrs = neighbors[offsets[i] + lo : offsets[i] + hi]
            table[int(i)] = nbrs[nbrs >= 0].astype(np.int32)
        upper[lvl] = table

    return HnswGraph(
        adj0=adj0,
        upper=upper,
        entry_point=int(hnsw.entry_point),
        max_level=int(hnsw.max_level),
        m=int(m0 // 2),
    )


def load_graph(name: str, m: int = 16, ef_construction: int = 200):
    """Convenience: dataset + faiss index + extracted graph, with graph caching."""
    ds = load_dataset(name)
    index = build_or_load_index(ds, m, ef_construction)
    cache = DATA_DIR / f"{ds.name}_graph_m{m}_efc{ef_construction}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        graph = HnswGraph(
            adj0=z["adj0"],
            upper=z["upper"].item(),
            entry_point=int(z["entry_point"]),
            max_level=int(z["max_level"]),
            m=int(z["m"]),
        )
    else:
        graph = extract_graph(index)
        np.savez_compressed(
            cache,
            adj0=graph.adj0,
            upper=np.array(graph.upper, dtype=object),
            entry_point=graph.entry_point,
            max_level=graph.max_level,
            m=graph.m,
        )
    return ds, index, graph
