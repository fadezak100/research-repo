# Ada-ef, explained (offline phase + online phase)

*Zhang & Miller, "Distribution-Aware Exploration for Adaptive HNSW Search",
SIGMOD 2026. Cross-referenced against the authors' code in `hnsw-ada-ef/` and
our Python re-implementation in `pyhnsw/ada_ef.py`.*

## The problem it solves

HNSW search quality is controlled by one knob, `ef` (the size of the result
priority queue at the base layer). Systems set one global `ef` for every query.
But queries differ wildly in difficulty (see `results/figs/e4_recall_skew.png`):
a global `ef` simultaneously **over-searches** easy queries (wasted compute) and
**under-searches** hard ones (terrible tail recall) — and gives **no recall
guarantee** either way. Ada-ef replaces the knob with a contract: *"give me
recall ≥ 0.95"*, and picks `ef` per query at runtime.

## The core insight

For a query **q** and dataset **V**, consider the *Full Distance List* — the
distances from q to every vector in V. For inner-product/cosine on
high-dimensional learned embeddings, this list is approximately **Gaussian**
(a Central Limit Theorem argument: q·v = Σᵢ qᵢvᵢ is a sum of many terms), with
closed-form parameters:

- cosine distance mean: μ = 1 − q·**mean(V)**
- variance: σ² = q **Σ** qᵀ, where **Σ** is the covariance matrix of V

So with just the dataset's **mean vector** and **covariance matrix** (computed
once offline), the whole distance distribution of any future query can be
estimated in O(d²) at query time — without computing any actual distances.
Knowing the distribution means knowing where the p-th percentile distance lies,
i.e. what counts as "an unusually close neighbor" *for this query*.

## Offline phase (all of it is cheap — ~9s total on GloVe-100 on a laptop)

Code: `pyhnsw/ada_ef.py::AdaEf.fit`, mirroring `EfAdapter::init` in
`hnsw-ada-ef/hnswlib/adaptive_ef.h` and `Estimator` in `distribution.h`.

1. **Dataset statistics** (`Stats`): column mean μ⃗ (d values) and covariance
   matrix Σ (d×d). One pass over the data. *(0.1s)*
2. **Proxy queries + ground truth** (`Samp.`): uniformly sample ~200 data
   vectors (we use 500 at k=10; see note below), treat them as stand-in
   queries, and compute their exact top-k by brute force. *(0.7s)*
3. **ef-estimation table** (`EF-Est.`): for each proxy query compute its
   **query score** (defined below), cast to integer, and group queries by
   score. For each score group, record (ef, average recall) pairs: start at
   ef = k and ef = 1.5k, then grow ef with secant-style steps until the group
   reaches the target recall (cap: ef = 5000). Result: a table
   `score → [(ef₁, r₁), (ef₂, r₂), …]` sorted ascending. *(8s)*
4. **WAE** (weighted-average ef): across score groups (weighted by group
   size), the smallest passing ef. Used as a floor at query time.

Key property: steps 1–2 don't depend on k or the target recall, so changing
the target only rebuilds the table (seconds). Insertions/deletions update the
mean/covariance incrementally (rank-one-style updates — §6.3 of the paper).
This is why the paper reports 50× less offline compute and 100× less offline
memory than the learned alternatives (DARTH/LAET train GBDT models on
hundreds of millions of feature rows).

## Online phase (per query)

Code: `pyhnsw/search.py` (the `ada` branch), mirroring
`adaptiveSearchBaseLayerST` in `hnswalg.h` + `Sketch` in `sketch.h`.

1. **Descend** the upper layers greedily to the base layer (standard HNSW).
2. **Collection**: run best-first search with **ef = ∞** until
   l = 1 + 2M + (2M−1)·2M distances are recorded (l = the 2-hop neighborhood
   bound of the entry point; 1025 for M=16). ef=∞ means: visit everything
   reachable, no pruning yet.
3. **Score the query**: estimate μ, σ from the offline stats (two dot
   products + one matrix-vector product). Build m=5 quantile thresholds
   θᵢ = μ + Φ⁻¹(0.001·i)·σ (the 0.1%…0.5% quantiles of the estimated
   distribution). Count how many of the l collected distances fall in each
   quantile bin: cᵢ. The score is Σ (cᵢ/l)·wᵢ with exponentially decaying
   weights wᵢ = 100·e⁻ⁱ.
   **Intuition**: a high score means many of the first ~1000 nodes visited are
   already in the extreme tail of this query's distance distribution — the
   neighborhood is dense with good candidates → easy query → small ef
   suffices. A low score means good candidates are rare → hard query → big ef.
4. **Look up ef**: find the table's nearest score groups (the score, and its
   ±1 neighbors, averaged), take the smallest ef whose recorded recall meets
   the target. Final ef = max(lookup, WAE, k).
5. **Truncate and continue**: shrink the result queue to ef and finish the
   search with the standard fixed-ef rule.

The collection phase is not wasted work — everything visited was real search
progress; the truncation just re-imposes a budget.

## What we measured (GloVe-100, k=10, target 0.95, 1000 queries)

| method | avg recall | p5 | p1 | dist comps/query |
|---|---|---|---|---|
| HNSW ef=160 | 0.853 | 0.30 | 0.10 | 3,429 |
| HNSW ef=640 | 0.934 | 0.60 | 0.30 | 11,033 |
| PIP (γ=.95, Δ=30) | 0.799 | 0.20 | 0.10 | 1,942 |
| **Ada-ef (target .95)** | **0.980** | **0.90** | **0.70** | 30,800 |
| Hybrid: Ada-ef + patience Δ=0.5·ef | 0.976 | 0.90 | 0.70 | 23,247 |

Ada-ef is the only method that meets the declarative target (overshooting on
GloVe, as the paper's own GloVe results do), and the tail improvement is the
headline: the 5th-percentile query goes from 0.30 (fixed ef=160) to 0.90.

## Deviations from the paper in our re-implementation

- k=10 instead of the paper's k=100/1000; per-query recall then has 0.1
  granularity, so we sample 500 proxy queries instead of 200 to stabilize
  per-group averages.
- Table probing runs in collection mode (ef=∞ for the first l nodes, then
  truncate), matching what the online search actually executes. The authors'
  code mixes probe modes between the initial entries and the refinement loop.
- The authors' table refinement stops as soon as group recall fails to improve
  between two probes. At k=10 that noise strands the hardest score groups at
  tiny ef (we observed the hardest group frozen at ef=20). We keep probing
  with doubling steps until the target or the ef cap (bounded attempts).
- Single-threaded Python; cost measured in distance computations, not seconds.

## Limits (and why they're research openings)

- The Gaussian theory covers inner product / cosine / cosine distance. **L2 is
  open** — the squared terms break the simple CLT argument. (The authors left
  a commented-out `SquaredEuclideanDistanceEstimator` draft in
  `distribution.h`.)
- Within its chosen ef, Ada-ef still runs the search to exhaustion — no early
  exit. That's exactly the gap PIP-style saturation could fill (our E6).
- The table is built from *data vectors as proxy queries*; real queries from a
  different distribution (e.g. text→image in LAION) degrade it — the paper's
  Laion-T2I results show this is the hard case.
