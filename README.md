# Query hardness in HNSW

Why does the same HNSW index, at the same search budget, find all the true
neighbors of one query and a quarter of another's? This repo measures that:
it labels queries hard / mid / easy by the recall plain HNSW reaches at a
fixed `ef`, then measures what is different about the hard ones — in vector
space and in the graph the search walks.

The main line runs on GloVe-100 (1.18M vectors, cosine); the internal-vs-
external comparison (E12) is replicated on SIFT1M (1M vectors, L2). Each
dataset gets a Faiss HNSW index (M=16, efConstruction=200). All searches go
through Faiss; the level-0 graph is extracted to numpy only for the graph
measurements (BFS).

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
  online_stats.py      E13  query-time statistics on the FOUND neighbors of a prefix search
  ef_policy.py         E14  statistic -> ef table -> re-search: two-stage policy vs fixed ef / oracle
  feature_shap.py      E15  all statistics at once: LightGBM ef predictor + SHAP attribution (QBAT Sec. 3)
  engine.py            resumable HNSW beam search in Python (peek at ef0, inspect, keep searching at a larger ef)
  resumable_stats.py   E16  the statistics measured inside the resumable engine; resume vs fresh vs two-stage cost
  end_to_end.py        E17  the peek-then-widen policy executed as one search per query in the engine
  query_sheet.py       Excel walkthrough of avg-dist for one query
  stats_sheet.py       Excel export of the E13 per-query statistics (results/e13_per_query_stats.xlsx)
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
- **SIFT1M** (`sift-128-euclidean`, L2): 1M × 128. Used for the E12
  replication (labels, adjacency, E11, E12b); E7 / E9 are GloVe-only.
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
.venv/bin/python -m pyhnsw.online_stats                # E13 -> e13_online_stats_glove100.json (~3 min)
.venv/bin/python -m pyhnsw.ef_policy                   # E14 -> e14_policy_glove100.json (~1 min; needs E13)
.venv/bin/python -m pyhnsw.feature_shap                # E15 -> e15_shap_glove100.json (~2 min; needs E13, no Faiss)
.venv/bin/python -m pyhnsw.resumable_stats             # E16 -> e16_resumable_glove100.json (~5 min; needs E13)
.venv/bin/python -m pyhnsw.end_to_end                  # E17 -> e17_end_to_end_glove100.json (~1 min; needs E14, E16)
.venv/bin/python -m pyhnsw.plots                       # all figures
```

The first run builds the index (~1 min) and caches it under `faiss/data/`.

Every script takes `--dataset sift1m` (default `glove100`); the SIFT1M
replication is the same sequence minus E7 / E9, with the E11 runs at the
SIFT operating points:

```bash
for q in external internal; do .venv/bin/python -m pyhnsw.labels --dataset sift1m --queries $q; done
.venv/bin/python -m pyhnsw.adjacency --dataset sift1m
for lab in k10_ef16 k100_ef100 k1000_ef1000; do
  .venv/bin/python -m pyhnsw.avgdist          --dataset sift1m --label $lab --k-gt 100   # ~70 s each
  .venv/bin/python -m pyhnsw.internal_avgdist --dataset sift1m --label $lab --k-gt 100   # ~90 s each
done
.venv/bin/python -m pyhnsw.plots --dataset sift1m      # figs/*_sift1m.png
```

## Method

**Labels.** Plain HNSW at a fixed `ef`, asking for `k`; the recall it reaches
is the query's label. Cost is the number of distance computations
(`faiss.cvar.hnsw_stats.ndis`, single-threaded). Operating points:

| label | k | ef | GloVe-100 bins | SIFT1M bins |
|---|---|---|---|---|
| `k10_ef160` | 10 | 160 | ≤0.25 / ~0.5 / ~0.75 / ~1.0 (closed) | (saturated, not run) |
| `k10_ef40` | 10 | 40 | same | ≤0.5 / 0.6–0.7 / 0.8–0.9 / 1.0 (closed) |
| `k10_ef16` | 10 | 16 | (not run) | same |
| `k100_ef100` | 100 | 100 | <0.6 / 0.6–0.7 / 0.7–0.8 / ≥0.8 (half-open) | <0.85 / 0.85–0.90 / 0.90–0.95 / ≥0.95 (half-open) |
| `k1000_ef1000` | 1000 | 1000 | same | same |

Bin edges are per dataset (`labels.BINS`): SIFT1M is far easier for HNSW,
so its edges sit higher (chosen so the hard bin holds ~12% of the external
queries, as on GloVe). Faiss's beam is max(ef, k), so ef=100 / ef=1000 are
already the cheapest settings for k=100 / k=1000; on SIFT1M the k=1000
label therefore cannot separate queries (recall@1000 ≥ 0.93 for all of
them) and k=10 needs ef=16 to spread.

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

**E12 replicated on SIFT1M** (true top-100, 1000 queries per set; SIFT bins,
see the label table). Figures: `figs/*_sift1m.png`.

| hardness label | queries | hard | mid | mid | easy | Spearman vs recall |
|---|---|---|---|---|---|---|
| recall@100, ef=100 | internal (nodes) | 3.28 (n=38) | 3.15 (67) | 3.01 (239) | 2.80 (656) | −0.84 |
| | external (test set) | 3.27 (n=116) | 3.13 (148) | 3.03 (253) | 2.82 (483) | −0.86 |
| recall@10, ef=16 | internal | 3.35 (5) | 3.11 (25) | 2.99 (301) | 2.84 (669) | −0.35 |
| | external | 3.20 (123) | 3.07 (232) | 2.96 (371) | 2.81 (274) | −0.60 |
| recall@1000, ef=1000 | internal | – (0) | 3.64 (1) | 3.37 (8) | 2.89 (991) | −0.77 |
| | external | – (0) | – (0) | 3.42 (10) | 2.97 (990) | −0.84 |

The GloVe finding holds: at the k=100 label the per-bin avg-dist agrees
within 0.02 hops between internal and external queries and the Spearman
correlations match (−0.84 vs −0.86), on a graph where the whole range is
compressed (2.8–3.3 hops vs 2.8–4.1 on GloVe; pair distances 1–7 hops, no
unreachable pairs). Cost per bin matches too (k=100: hard ≈ 2.2k vs easy
≈ 1.65k distance computations, both sets). Two differences from GloVe:
internal queries are easier than external ones at small k (recall@10 at
ef=16: 0.95 vs 0.79; the node's own out-edges hand it much of its top-10),
which compresses the internal k=10 label and its correlation; and the
hops-from-node signal is weaker (hard 2.74 vs easy 2.32 at k=100, Spearman
−0.43 vs −0.73 on GloVe). Wiring is similar: 7.2 of a node's 16 nearest true
neighbors are direct out-edges (GloVe 6.3), 15.7 of its out-edges are in its
true top-100 (15.5), 14–16% of the top-100 are one hop away.

**E12 — wiring.** Of a node's 16 nearest true neighbors, only 6.3 are direct
out-edges (range 0–15); of its ≤32 out-edges, 15.5 are in its true top-100.
Hops from the node to its true top-100: hard 3.56 vs easy 2.45 (Spearman
−0.73 with recall); only 12–17% are one hop away.

**E13 — query-time statistics on the found neighbors.** At query time the
true neighbors are unknown, so each offline statistic is recomputed on the
100 neighbors a prefix search (Faiss, k=100, ef=100) has found, using only
that search's output. Spearman with recall@100 at ef=100, 1000 external
queries; the offline twin is the same statistic on the true top-100:

| statistic on the found top-100 | GloVe-100 | SIFT1M | offline twin GloVe / SIFT |
|---|---|---|---|
| contrast: expected distance to a random base vector ÷ distance to found top-10 (one dot product) | +0.93 | +0.69 | +0.92 / +0.69 |
| edges among the found top-100 (adjacency reads, no distances) | +0.92 | +0.86 | +0.97 / +0.92 |
| avg-dist among the found top-100 (exact BFS) | −0.89 | −0.83 | −0.96 / −0.87 |
| distance to the found top-10 | −0.89 | −0.63 | −0.86 / −0.63 |
| distance computations of the ef=100 search | −0.82 | −0.78 | – |
| spread (mean pairwise distance) of the found top-100 | −0.61 | −0.46 | −0.76 / −0.47 |
| QBAT-style entry-time features (best: distance to the level-0 entry point) | −0.51 | −0.54 | – |

Distance-based statistics are the strongest on GloVe but weaken on SIFT1M,
where the found top-10 is the true top-10 for nearly every query and the
distances say little about whether the other 90 will be reached. The graph
statistic holds on both datasets (AUC for the hard bin 0.96 GloVe / 0.93
SIFT1M) and costs no distance computations. Entry-time features, the ones
QBAT uses for HNSW, are weak on both. The same ordering holds against the
oracle ef (smallest ef reaching recall 0.9): on GloVe contrast −0.91 and
edges −0.88; on SIFT1M edges −0.65 vs contrast −0.53. 108 of the 1000 GloVe
queries do not reach 0.9 within the grid (cap ef=3200; one more doubling
serves them, at ~114k distance computations each); none on SIFT1M. Figures
`figs/e13_stats.png`, `figs/e13_scatter*.png`.

**E14 — from the statistic to an ef.** Two-stage policy: search at
ef=100, compute the statistic on the found top-100, look ef up in a table,
re-search only the flagged queries. The table has 10 equal-count bins of
the statistic, fit on the 9000 test queries not used anywhere else; each
bin gets the smallest grid ef at which 80% of its queries reach recall T
(`q80`; the bin-mean rule, Ada-ef's, is also run). Replayed on E13's 1000
queries from their saved ef sweep. Cost is counted two ways: two-stage
(prefix plus re-search, what this implementation pays) and resume (the
chosen ef alone, what an engine that continues the search would pay).
Baselines: fixed ef at every grid value and the per-query oracle.

| dataset, target | policy (q80) | cost | frac ≥ T | p5 | fixed ef at same cost: frac / p5 | fixed-ef cost for the same frac | oracle cost / frac |
|---|---|---|---|---|---|---|---|
| GloVe, T=0.8 | contrast | 17.8k | 0.917 | 0.77 | 0.848 / 0.65 | 28.1k (−37%) | 13.3k / 0.978 |
| GloVe, T=0.8 | edges | 17.0k | 0.867 | 0.72 | 0.841 / 0.64 | 20.0k (−15%) | |
| GloVe, T=0.9 | contrast | 23.9k | 0.811 | 0.82 | 0.773 / 0.73 | 30.1k (−21%) | 20.3k / 0.892 |
| GloVe, T=0.9 | edges | 25.6k | 0.810 | 0.81 | 0.784 / 0.75 | 29.9k (−14%) | |
| SIFT1M, T=0.9 | edges | 3.3k | 0.946 | 0.89 | 0.983 / 0.92 | 2.8k (+16%) | 2.8k / 1.000 |
| SIFT1M, T=0.9, resume cost | edges | 2.4k | 0.946 | 0.89 | 0.862 / 0.85 | 2.8k (−17%) | 2.2k / 1.000 |

Two-stage cost unless noted; all differences vs fixed ef at equal cost have
paired-bootstrap 95% CIs excluding zero. On GloVe the policy beats fixed ef
on mean, tail and fraction-at-target under both accountings; contrast is
the better policy signal there and edges holds up. On SIFT1M the useful ef
range is 100–200 and the ef=100 prefix alone is most of the budget, so a
two-stage implementation loses to plain fixed ef=200; an engine that can
continue the prefix search gains +0.08 in fraction ≥ 0.9 at equal cost. The
bin-mean rule gains on p5 but loses on the fraction at target on both
datasets. The gap to the oracle stays large (GloVe T=0.8: oracle 0.978 at
13.3k vs the best policy 0.917 at 17.8k). Figures `figs/e14_policy*.png`.

**E15 — all statistics at once: which ones does a learned ef predictor
use?** E13 scores each statistic alone, which cannot say what a statistic
adds on top of the others (contrast ≈ d10, edges ≈ avg-dist). Following
QBAT (Sec. 3), a LightGBM model predicts log2 of the oracle ef (smallest
grid ef reaching recall 0.9) from the 14 deployable query-time statistics,
and SHAP attributes each held-out prediction to the features (5-fold CV ×
3 seeds; every query's attribution comes from a model that never saw it).
Because SHAP splits the credit of a shared signal among correlated
features, it is read with retrain checks: held-out R² of each feature
alone, of the top-k features by SHAP, and greedy forward selection.

| GloVe-100 (R² all 14 = 0.92) | SHAP share | alone R² | | SIFT1M (R² all 14 = 0.55) | SHAP share | alone R² |
|---|---|---|---|---|---|---|
| contrast (top-100) | 64% | 0.88 | | edges among found | 39% | 0.53 |
| edges among found | 13% | 0.71 | | contrast (top-10) | 12% | 0.26 |
| contrast (top-10) | 6% | 0.83 | | distance count of the search | 9% | 0.30 |
| distance to found top-10 | 4% | 0.76 | | spread | 9% | 0.01 |
| distance count of the search | 3% | 0.60 | | contrast (top-100) | 5% | 0.19 |
| six entry-time features together | 6% | ≤ 0.28 | | six entry-time features together | 16% | ≤ 0.09 |
| top 2 by SHAP together | | 0.91 | | top 2 by SHAP together | | 0.55 |
| forward selection | contrast100 0.88 → +edges 0.91 → +d100 0.92 | | | forward selection | edges 0.53 → +contrast10 0.55 → +spread 0.56 | |

The model confirms E13 and adds the interaction picture: two features carry
it (the distance signal and the graph signal), and their order flips
between datasets — contrast first on GloVe, edges first on SIFT1M. Dropping
the whole distance family costs GloVe 0.92 → 0.77 while dropping the graph
family costs 0.92 → 0.92 (contrast covers it); on SIFT1M dropping the graph
family costs 0.55 → 0.47 and dropping the distance family 0.55 → 0.48, so
there the two are complementary. Entry-time features add nothing on either
dataset (R² without them is unchanged or higher). The learned model's held-out
Spearman with the oracle ef (+0.95 GloVe, +0.63 SIFT1M) is only slightly
above the best single statistic (−0.92 / −0.65 in E13): the ceiling is the
statistics, not the model. Including the lab-only avg-dist changes nothing
on GloVe (R² 0.92) and little on SIFT1M (0.56). SIFT1M's oracle ef takes
only five distinct values, so R² there is bounded by the coarse grid.
Figure `figs/e15_shap.png`; per-query SHAP values are in
`results/e15_shap_*.json`.

**E16 — the statistics measured inside a resumable search.** E13 computed
the statistics on the output of a separate Faiss search and E14 replayed
the cost of "peek, then widen ef" from saved sweeps. `pyhnsw/engine.py`
restores the pre-cleanup Python beam search (same loop as hnswlib's
`searchBaseLayerST`, beam = max(ef, k) as in Faiss) as a resumable object:
run it until it converges at ef=100, read the current top-100, compute
the statistics, then keep the same visited set and heaps and continue at
a larger ef. What such an engine must keep is a *spill* heap of nodes
whose distance was computed but which did not fit the ef-sized result
heap; when ef grows the best of them are re-admitted at no cost.

- **Engine = Faiss.** Peek recall equals the stored `k100_ef100` label for
  1000/1000 GloVe queries (997/1000 on SIFT1M, the three known distance
  ties); every statistic matches E13's value (Spearman 1.000, differences
  at float precision; `n_dist` +1 as before).
- **Correlations hold inside the engine**, with the oracle ef now measured
  on the resumed search rather than replayed: GloVe contrast10 +0.93 /
  edges +0.92 with peek recall, −0.91 / −0.88 with the oracle ef, catch
  rates 85% / 83%; SIFT1M edges +0.86 / −0.65, catch 59%. The in-engine
  oracle ef is identical to the fresh-search one for every query.
- **Resuming is free.** With the spill heap, a search resumed from the
  ef=100 peek reaches exactly the recall of a fresh search at every grid
  ef, for exactly the fresh search's distance computations (+1). E14's
  "resume" accounting was therefore exact, and the peek costs nothing
  beyond the statistic itself. Two-stage (peek, then a fresh search) pays
  the peek twice: +2,369 distance computations per re-searched query on
  GloVe, +1,882 on SIFT1M, which is what made E14 lose on SIFT1M.
- **Without the spill heap** the continued search cannot re-admit pruned
  nodes and instead keeps expanding until the result heap fills: recall
  ends higher (GloVe ef=3200: 0.980 vs 0.967) at about twice the cost
  (91k vs 45k), i.e. it is a different, wider search, not a resumption.

Figure `figs/e16_resumable.png`; per-query resumed recall/cost curves are in
`results/e16_resumable_*.json`.

**E17 — the policy executed end to end.** E14's policy (peek at ef=100,
statistic, 10-bin table fit on the training queries with the q80 rule)
run as one search per query in the resumable engine: `s.run(100)`, read
the live top-100, look up ef, `s.run(ef)` if larger, return the top-100.
Baselines (fixed ef at every grid value, the per-query oracle) are the
engine's own sweeps from E16. Measured, 1000 evaluation queries:

| dataset, target | statistic | continued | cost | frac ≥ T | p5 | fixed ef at same cost: frac / p5 | Δ frac [95% CI] | fixed-ef cost for same frac | oracle cost / frac |
|---|---|---|---|---|---|---|---|---|---|
| GloVe, T=0.8 | contrast | 71% | 15.9k | 0.917 | 0.77 | 0.831 / 0.63 | +0.086 [+0.07, +0.11] | 28.1k (+43%) | 11.8k / 0.978 |
| GloVe, T=0.8 | edges | 60% | 15.4k | 0.867 | 0.72 | 0.825 / 0.62 | +0.042 [+0.02, +0.06] | 20.0k (+23%) | |
| GloVe, T=0.9 | contrast | 80% | 21.9k | 0.811 | 0.82 | 0.756 / 0.71 | +0.055 [+0.03, +0.08] | 30.1k (+27%) | 18.5k / 0.892 |
| GloVe, T=0.9 | edges | 78% | 23.6k | 0.810 | 0.81 | 0.770 / 0.73 | +0.040 [+0.02, +0.06] | 29.9k (+21%) | |
| SIFT1M, T=0.9 | edges | 44% | 2.36k | 0.946 | 0.89 | 0.862 / 0.85 | +0.084 [+0.07, +0.10] | 2.85k (+17%) | 2.21k / 1.000 |
| SIFT1M, T=0.9 | contrast | 56% | 2.42k | 0.935 | 0.88 | 0.877 / 0.86 | +0.058 [+0.05, +0.07] | 2.72k (+11%) | |

Every measured number equals E14's resume-accounting prediction: per query,
0 recall mismatches and 0 cost difference against the replay. The policy
beats fixed ef at equal cost on both datasets, including SIFT1M, where the
two-stage implementation had lost only because it paid the peek twice.
Wall-clock in the Python engine (indicative): the statistic costs 0.02 ms
per query for contrast and 0.15 ms for edges against 1.5 ms for the peek
itself. The gap to the oracle is unchanged (GloVe T=0.8: 0.917 at 15.9k vs
0.978 at 11.8k). Results `results/e17_end_to_end_*.json` (per-query
statistic, chosen ef, recall and cost).

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
