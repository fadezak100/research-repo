"""Export the SIFT1M HDF5 arrays to raw binary files for the Lucene (Java) benchmark.

Java has no h5py, so we dump the three arrays as flat little-endian binaries:
    train.bin      1M x 128 float32
    test.bin       10k x 128 float32
    neighbors.bin  10k x 100 int32
and, for LuceneRecallBenchmark (which grades k=1000), the exact top-1000 of the
first 1000 queries, brute-forced by pyhnsw.experiments.gt_for_k and cached at
faiss/data/sift1m_gt1000_q1000.npy:
    neighbors1000.bin  1000 x 1000 int32
"""

from pathlib import Path

import h5py
import numpy as np

SRC = Path(__file__).parent.parent / "faiss" / "data" / "sift-128-euclidean.hdf5"
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

with h5py.File(SRC, "r") as f:
    np.array(f["train"], dtype="<f4").tofile(OUT / "train.bin")
    np.array(f["test"], dtype="<f4").tofile(OUT / "test.bin")
    np.array(f["neighbors"], dtype="<i4").tofile(OUT / "neighbors.bin")

GT1000 = SRC.parent / "sift1m_gt1000_q1000.npy"
if not GT1000.exists():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pyhnsw.experiments import gt_for_k  # brute-forces and caches the file
    gt_for_k("sift1m", 1000)
np.load(GT1000).astype("<i4").tofile(OUT / "neighbors1000.bin")

print("wrote", sorted(p.name for p in OUT.iterdir()))
