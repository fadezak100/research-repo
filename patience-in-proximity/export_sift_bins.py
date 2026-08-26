"""Export the SIFT1M HDF5 arrays to raw binary files for the Lucene (Java) benchmark.

Java has no h5py, so we dump the three arrays as flat little-endian binaries:
    train.bin      1M x 128 float32
    test.bin       10k x 128 float32
    neighbors.bin  10k x 100 int32
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

print("wrote", [p.name for p in OUT.iterdir()])
