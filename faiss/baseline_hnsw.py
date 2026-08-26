"""Phase 1 baseline: Faiss HNSW on SIFT1M (ann-benchmarks format).

Builds an HNSW index over the 1M SIFT vectors, then sweeps efSearch and
reports recall@10 and queries-per-second against the exact ground truth
shipped with the dataset.

Usage:
    python baseline_hnsw.py [--m 16] [--ef-construction 200]
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

import faiss

DATA_FILE = Path(__file__).parent / "data" / "sift-128-euclidean.hdf5"
RESULTS_DIR = Path(__file__).parent / "results"
K = 10
EF_SEARCH_VALUES = [10, 20, 40, 80, 160, 320]


def load_dataset():
    with h5py.File(DATA_FILE, "r") as f:
        train = np.array(
            f["train"], dtype=np.float32
        )  # 1M SIFT vectors * 128 dimensions
        test = np.array(f["test"], dtype=np.float32)  # 10k queries * 128 dimensions
        neighbors = np.array(f["neighbors"])  # 10k queries * 100 nearest neighbors
    return train, test, neighbors


def build_or_load_index(train, m, ef_construction):
    index_file = DATA_FILE.parent / f"sift1m_hnsw_m{m}_efc{ef_construction}.faiss"
    if index_file.exists():
        print(f"Loading cached index from {index_file.name}")
        return faiss.read_index(str(index_file))

    print(f"Building HNSW index (M={m}, efConstruction={ef_construction}) ...")
    index = faiss.IndexHNSWFlat(
        train.shape[1], m
    )  # train.shape[1] is the number of dimensions
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    index.add(train)
    print(f"Build took {time.perf_counter() - t0:.1f}s")
    faiss.write_index(index, str(index_file))
    return index


def recall_at_k(found, ground_truth, k):
    hits = 0
    for i in range(found.shape[0]):  # shape[0] is the number of queries
        hits += len(set(found[i, :k]) & set(ground_truth[i, :k]))
    return hits / (found.shape[0] * k)


def main():
    parser = argparse.ArgumentParser()
    # M: max connections per node in the graph (fixed at build time).
    # Higher = better recall, but more memory, slower build, and slower queries.
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)

    args = parser.parse_args()

    # Load the dataset
    print(f"Faiss {faiss.__version__}, {faiss.omp_get_max_threads()} threads")
    train, test, neighbors = load_dataset()
    print(f"train {train.shape}, test {test.shape}")

    # Building the index
    index = build_or_load_index(train, args.m, args.ef_construction)

    # Experimenting with different exploration factors
    results = []
    print(f"\n{'efSearch':>8} {'recall@10':>10} {'QPS':>10} {'ms/query':>10}")
    for ef in EF_SEARCH_VALUES:
        index.hnsw.efSearch = ef
        index.search(test[:100], K)  # warmup
        t0 = time.perf_counter()
        _, found = index.search(test, K)
        elapsed = time.perf_counter() - t0
        recall = recall_at_k(found, neighbors, K)
        qps = test.shape[0] / elapsed
        print(
            f"{ef:>8} {recall:>10.4f} {qps:>10.0f} {1000 * elapsed / test.shape[0]:>10.3f}"
        )
        results.append({"efSearch": ef, "recall@10": recall, "qps": qps})

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"baseline_sift1m_m{args.m}_efc{args.ef_construction}.json"
    out.write_text(
        json.dumps(
            {
                "dataset": "sift-128-euclidean",
                "index": {
                    "type": "HNSWFlat",
                    "M": args.m,
                    "efConstruction": args.ef_construction,
                },
                "k": K,
                "threads": faiss.omp_get_max_threads(),
                "sweep": results,
            },
            indent=2,
        )
    )
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
