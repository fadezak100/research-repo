# Query hardness in HNSW

Why does the same HNSW index, at the same search budget, find all the true
neighbors of one query and a quarter of another's? This repo measures that:
it labels queries hard / mid / easy by the recall plain HNSW reaches at a
fixed `ef`, then measures what is different about the hard ones — in vector
space and in the graph the search walks.

Everything runs on GloVe-100 (1.18M vectors, cosine) with a Faiss HNSW index
(M=16, efConstruction=200). All searches go through Faiss; the level-0 graph
is extracted to numpy only for the graph measurements (BFS).

## Repo layout

```
pyhnsw/
  graph.py             load dataset, build/load the Faiss index, extract the level-0 graph
  index.py             search (batched) / search_with_cost (per-query ndis via hnsw_stats), gt_for_k
  labels.py            hardness labels: recall + cost at fixed (k, ef), hard/mid/easy bins
  internal_queries.py  1000 random index nodes as queries + their exact GT (self removed)
  gt_metrics.py        E7   vector-space metrics of a query's true neighbors vs hardness
  causal.py            E9   does more budget fix hard queries? (asymptote, cost link)
  avgdist.py           E11  exact avg-dist: hops between pairs of true neighbors (BFS engine)
  adjacency.py         E12a how much of a node's true neighborhood is wired to it
  internal_avgdist.py  E12b avg-dist + hops-from-node for internal queries, vs external
  query_sheet.py       Excel walkthrough of avg-dist for one query
  plots.py             all figures (results/figs/*.png)
results/               experiment JSON + figs/
faiss/data/            hdf5 datasets, built indexes, graph/GT caches (not committed)
tools/                 slide builders, decks, session notes (not committed)
```

## Datasets

From [ann-benchmarks](https://ann-benchmarks.com), 10k queries and exact
top-100 ground truth each. Only the first 1000 queries are used.

- **GloVe-100** (`glove-100-angular`, cosine): 1.18M × 100.
  `curl -L -o faiss/data/glove-100-angular.hdf5 http://ann-benchmarks.com/glove-100-angular.hdf5`
- **SIFT1M** (`sift-128-euclidean`, L2): 1M × 128, loadable but unused so far.
  `curl -L -o faiss/data/sift-128-euclidean.hdf5 http://ann-benchmarks.com/sift-128-euclidean.hdf5`

Cosine datasets are L2-normalized on load so Faiss's inner-product index
computes cosine similarity. Exact top-1000 ground truth is brute-forced with
a flat index and cached (`gt_for_k`).

## Running

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m pyhnsw.labels --queries external   # results/labels_external_glove100.json
.venv/bin/python -m pyhnsw.internal_queries            # 1000 random nodes + exact GT (cached)
.venv/bin/python -m pyhnsw.labels --queries internal   # results/labels_internal_glove100.json
.venv/bin/python -m pyhnsw.gt_metrics                  # E7  -> e7_hardness_glove100.json
.venv/bin/python -m pyhnsw.causal                      # E9  -> e9_hardness_causal_glove100.json
.venv/bin/python -m pyhnsw.avgdist --label k1000_ef1000 --k-gt 100     # E11 (~4 min; GT-1000 ~2 h)
.venv/bin/python -m pyhnsw.adjacency                   # E12a -> adjacency_internal_glove100.json
.venv/bin/python -m pyhnsw.internal_avgdist --label k1000_ef1000 --k-gt 100   # E12b (~4 min)
.venv/bin/python -m pyhnsw.plots                       # all figures
```

The first run builds the index (~1 min) and caches it under `faiss/data/`.

## Method

**Labels.** Plain HNSW at a fixed `ef`, asking for `k`; the recall it reaches
is the query's label. Cost is the number of distance computations
(`faiss.cvar.hnsw_stats.ndis`, single-threaded). Operating points:

| label | k | ef | bins |
|---|---|---|---|
| `k10_ef160` | 10 | 160 | ≤0.25 / ~0.5 / ~0.75 / ~1.0 (closed) |
| `k10_ef40` | 10 | 40 | same |
| `k100_ef100` | 100 | 100 | <0.6 / 0.6–0.7 / 0.7–0.8 / ≥0.8 (half-open) |
| `k1000_ef1000` | 1000 | 1000 | same |

**Internal queries.** 1000 index nodes drawn at random (seed 0) and used as
queries with their own vector. Exact ground truth is brute-forced and the
node itself removed by id; the search asks for k+1 and drops the node from
its results, so a perfect search still scores 1.0.

**avg-dist** (the professor's whiteboard definition): for a query's true
top-100, the mean over all 4,950 pairs of the shortest-path hop count in the
level-0 graph, `min(hops u→v, hops v→u)`, exact via bidirectional BFS
(forward 3 rings, backward 2 over a cached reverse adjacency; deeper pass and
plain BFS as fallbacks). Pairs unreachable in both directions are excluded and
counted (none on GloVe).

**Hops from the node** (internal queries only): one BFS from the node to
each of its true neighbors; reported as the mean, the share at one hop, and
the mean by neighbor rank (1–16, 17–32, 33–100).

## Results (GloVe-100, 1000 queries per set)

**E7 — vector-space metrics.** Hard queries' true neighbors barely stand out
from random vectors (rel_contrast, Spearman +0.71 with recall) and are spread
far apart (gt_pairwise −0.52); GT in-degree shows no signal.

**E9 — budget.** The hard bin (34 queries, recall@10 ≤ 0.25 at ef=160)
climbs 0.16 → 0.93 as ef goes 160 → 5120 with no plateau: the neighbors are
reachable, at ~22× the cost. Per-query cost correlates −0.82 with
rel_contrast and +0.77 with GT spread — hard and expensive for the same
reason.

**E11 / E12 — avg-dist, external vs internal queries** (true top-100):

| hardness label | queries | hard | mid | mid | easy | Spearman vs recall |
|---|---|---|---|---|---|---|
| recall@1000, ef=1000 | internal (nodes) | 4.13 (n=127) | 3.73 (128) | 3.36 (165) | 2.84 (580) | −0.88 |
| | external (test set) | 4.10 (n=145) | 3.75 (150) | 3.39 (156) | 2.89 (549) | −0.88 |
| recall@10, ef=160 | internal | 4.22 (17) | 4.14 (63) | 3.79 (158) | 2.98 (762) | −0.65 |
| | external | 4.14 (34) | 4.07 (94) | 3.71 (182) | 3.00 (690) | −0.70 |
| recall@100, ef=100 | internal | 3.95 (261) | 3.45 (89) | 3.22 (111) | 2.79 (539) | −0.96 |

Internal and external queries agree bin for bin: the signal belongs to the
node's neighborhood in the graph, not to the query being an outsider. Labels
and cost per bin also match (k=1000: hard ≈ 21.8k vs easy ≈ 13.4k distance
computations, both sets).

**E12 — wiring.** Of a node's 16 nearest true neighbors, only 6.3 are direct
out-edges (range 0–15); of its ≤32 out-edges, 15.5 are in its true top-100.
Hops from the node to its true top-100: hard 3.56 vs easy 2.45 (Spearman
−0.73 with recall); only 12–17% are one hop away.

**Not in this repo any more:** the PIP / Ada-ef comparison work and the
Lucene benchmark (tag `pre-cleanup`), and the index-repair / per-query-budget
pilots (session notes under `tools/`). Their conclusions: local graph repair
gains ~1 point; a denser rebuild (M=32, efC=500) saves 20–28% at matched
recall but hard queries stay 1.6–1.7× as expensive; a per-query budget from
an online contrast score ties PIP and beats fixed ef on the tail.

## Next

1. Per-hop trace: how many new vectors enter the result list at each step of
   the search, averaged per hardness bin (raw material for a dynamic stop).
2. Rebuild with a shuffled insertion order and check whether the same nodes
   stay hard.
3. avg-dist within rank segments of the true neighbor list.
