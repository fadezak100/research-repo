"""E12 step 1 — internal queries and their ground truth.

E10/E11 used *external* queries (the hdf5 `test` vectors, never inserted
into the index). Here the queries are index nodes themselves: rows of the
`train` array, which is exactly the set Faiss inserted, so row i == node i
of the HNSW graph.

  sample   pick N node ids uniformly at random (fixed seed); cache them so
           every later step (hardness labels, graph measurements) uses the
           same nodes.
  gt       exact top-K true neighbors of each sampled node, *excluding the
           node itself*: brute-force top-(K+1) with a flat Faiss index over
           all base vectors, then drop the query's own id from each row.
           The self id is removed by id, not by position, because GloVe
           has duplicate rows: a duplicate can tie the query at distance 0
           and land at rank 0 instead of it.

Caches (faiss/data/):
  {dataset}_internal_qids_n{N}_s{SEED}.npy      (N,)   int64 node ids
  {dataset}_internal_gt{K}_n{N}_s{SEED}.npy     (N, K) int32 neighbor ids
                                                        (self excluded)
GT-100 is the first 100 columns of the K=1000 file.

Usage: .venv/bin/python -m pyhnsw.internal_queries [--dataset glove100]
           [--n 1000] [--seed 0] [--k 1000]
"""

import argparse

import numpy as np

from .experiments import get_ctx
from .graph import DATA_DIR

N_INTERNAL = 1000
SEED = 0
K_GT = 1000


def qids_path(dataset, n=N_INTERNAL, seed=SEED):
    return DATA_DIR / f"{dataset}_internal_qids_n{n}_s{seed}.npy"


def gt_path(dataset, k=K_GT, n=N_INTERNAL, seed=SEED):
    return DATA_DIR / f"{dataset}_internal_gt{k}_n{n}_s{seed}.npy"


def sample_internal_queries(dataset, n=N_INTERNAL, seed=SEED):
    """N distinct node ids drawn uniformly from the index; cached."""
    path = qids_path(dataset, n, seed)
    if path.exists():
        return np.load(path)
    n_base = len(get_ctx(dataset).ds.train)
    rng = np.random.default_rng(seed)
    qids = np.sort(rng.choice(n_base, size=n, replace=False)).astype(np.int64)
    np.save(path, qids)
    print(f"sampled {n} internal queries from {n_base} nodes (seed {seed}) -> {path}")
    return qids


def internal_gt(dataset, k=K_GT, n=N_INTERNAL, seed=SEED):
    """Exact top-k neighbors of each sampled node, self excluded; cached.

    Returns (qids, gt) with gt.shape == (n, k)."""
    qids = sample_internal_queries(dataset, n, seed)
    path = gt_path(dataset, k, n, seed)
    if path.exists():
        return qids, np.load(path)
    import faiss

    ds = get_ctx(dataset).ds
    d = ds.train.shape[1]
    index = faiss.IndexFlatIP(d) if ds.metric == "cosine" else faiss.IndexFlatL2(d)
    index.add(ds.train)
    # k+1 because the query vector is in the index and must come back once
    sims, ids = index.search(ds.train[qids], k + 1)

    gt = np.empty((n, k), dtype=np.int32)
    n_dup = 0  # queries with a distinct node at distance 0 (duplicate vector)
    for i, q in enumerate(qids):
        row = ids[i]
        self_pos = np.flatnonzero(row == q)
        assert self_pos.size == 1, f"query {q} not found in its own top-{k + 1}"
        keep = np.delete(row, self_pos[0])
        gt[i] = keep[:k]
        # a duplicate neighbor scores exactly like the query itself
        d0 = sims[i, self_pos[0]]
        others = np.delete(sims[i], self_pos[0])
        if np.any(np.isclose(others[:1], d0)):
            n_dup += 1
    assert not np.any(gt == qids[:, None]), "self id leaked into GT"
    assert all(len(set(r.tolist())) == k for r in gt), "duplicate ids in a GT row"

    np.save(path, gt)
    print(f"internal gt{k}: {n} queries, self excluded; "
          f"{n_dup} queries have a duplicate vector at distance 0 -> {path}")
    return qids, gt


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="glove100")
    ap.add_argument("--n", type=int, default=N_INTERNAL)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--k", type=int, default=K_GT)
    args = ap.parse_args()
    qids, gt = internal_gt(args.dataset, args.k, args.n, args.seed)
    print(f"qids: {qids.shape} (first 5: {qids[:5].tolist()})")
    print(f"gt:   {gt.shape}  e.g. node {qids[0]} -> top-5 {gt[0, :5].tolist()}")


if __name__ == "__main__":
    main()
