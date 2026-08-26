"""Ada-ef (Zhang & Miller, SIGMOD 2026) in Python — offline + online phases.

Offline phase (mirrors the authors' EfAdapter::init in
hnsw-ada-ef/hnswlib/adaptive_ef.h and Estimator in distribution.h):
  1. Dataset statistics: column mean vector mu (d,) and covariance matrix
     Sigma (d, d) of the (normalized) data vectors.
  2. Sample 200 data vectors as proxy queries; compute their exact ground
     truth by brute force.
  3. ef-estimation table: score each proxy query (see below), group queries
     by integer score, and for each group record (ef, avg recall) pairs —
     starting at ef = k and 1.5k, then growing ef by secant steps until the
     group's average recall reaches the target (cap: ef_upper_bound).
     WAE = count-weighted average over groups of the smallest passing ef.

Online phase (per query q, mirrors adaptiveSearchBaseLayerST + Sketch):
  1. Search starts with ef = inf and records the distances of the first
     l = 1 + 2M + (2M-1)*2M visited nodes (entry point's 2-hop bound).
  2. Estimated distance distribution: cosine distance ~ N(1 - q.mu, q Sigma qT).
  3. Query score: bin the l collected distances by the estimated quantile
     thresholds theta_i = mean + z_i * std (z_i = Phi^-1(delta*(i+1)),
     delta = 0.001, m = 5 bins), score = sum (c_i / l) * 100 * e^-i.
  4. ef = max(mean of table lookups at int scores {s-1, s, s+1}, WAE, k);
     the result heap is truncated to ef and the search continues normally.
"""

import json
from bisect import bisect_left
from statistics import NormalDist

import numpy as np

QUANTILE_STEP = 0.001
NUM_BINS = 5
EF_UPPER_BOUND = 5000
# The paper samples 200 data vectors, at k=100/1000. We run k=10, where a single
# query's recall has 0.1 granularity, so score groups need more members for the
# per-group average recall to be stable.
N_PROXY_QUERIES = 500


class AdaEf:
    """Online estimator. Build with AdaEf.fit(ctx, k, target_recall)."""

    def __init__(self, mu, sigma, k, target_recall, table=None, wae=None, probe_ef=None):
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)
        self.k = k
        self.target_recall = target_recall
        self.table = table  # sorted list of [int_score, [(ef, recall), ...]]
        self.wae = wae
        self.probe_ef = probe_ef  # fixed ef used while building the table
        nd = NormalDist()
        z = np.array([nd.inv_cdf(QUANTILE_STEP * (i + 1)) for i in range(NUM_BINS)])
        self._z = z
        self._weights = 100.0 * np.exp(-np.arange(NUM_BINS))
        self._links = self._build_links() if table else None

    # ---------- online ----------

    def l_limit(self, graph):
        m0 = 2 * graph.m
        return 1 + m0 + (m0 - 1) * m0

    def compute_score(self, q, collected):
        mean = 1.0 - float(q @ self.mu)  # cosine distance
        var = float(q @ self.sigma @ q)
        std = np.sqrt(max(var, 0.0))
        thresholds = mean + self._z * std
        # c_i = number of collected distances in bin i (below theta_i, first hit)
        bins = np.searchsorted(thresholds, collected, side="right")
        cnt = np.bincount(bins, minlength=NUM_BINS + 1)[:NUM_BINS]
        return float((cnt / len(collected)) @ self._weights)

    def _build_links(self):
        """links[i] = index of the score group closest to integer score i."""
        keys = [g[0] for g in self.table]
        links = []
        for i in range(101):
            j = bisect_left(keys, i)
            a = j - 1 if j > 0 else -1
            b = j if j < len(keys) else -1
            if a != -1 and b != -1:
                links.append(a if abs(keys[a] - i) <= abs(keys[b] - i) else b)
            else:
                links.append(a if a != -1 else b)
        return links

    def _lookup(self, int_score):
        group = self.table[self._links[int(np.clip(int_score, 0, 100))]]
        for ef, recall in group[1]:
            if recall >= self.target_recall:
                return ef
        return group[1][-1][0]

    def estimate_ef(self, score):
        if self.table is None:
            return self.probe_ef  # table-building probe mode
        first = self._lookup(score)
        if score < 1 or score >= 100:
            est = first
        else:
            est = (first + self._lookup(score - 1) + self._lookup(score + 1)) // 3
        return int(max(est, self.wae, self.k))

    # ---------- offline ----------

    @classmethod
    def fit(cls, ctx, k, target_recall, seed=42, n_proxy=N_PROXY_QUERIES, log=print):
        """Full offline phase against a SearchContext (cosine metric only)."""
        import time

        ds = ctx.ds
        assert ds.metric == "cosine", "Ada-ef theory covers IP/cosine only"
        timings = {}

        t0 = time.perf_counter()
        mu = ds.train.mean(axis=0)
        centered = ds.train - mu
        sigma = (centered.T @ centered) / (len(ds.train) - 1)
        timings["stats_s"] = time.perf_counter() - t0
        log(f"[offline] mean+covariance: {timings['stats_s']:.1f}s")

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        proxy_ids = rng.choice(len(ds.train), size=n_proxy, replace=False)
        proxies = ds.train[proxy_ids]
        # exact ground truth for proxies (brute force, top-k by cosine distance)
        import faiss

        _, gt = faiss.knn(proxies, ds.train, k, metric=faiss.METRIC_INNER_PRODUCT)
        timings["sample_gt_s"] = time.perf_counter() - t0
        log(f"[offline] {n_proxy} proxy queries + exact GT: {timings['sample_gt_s']:.1f}s")

        t0 = time.perf_counter()
        est = cls(mu, sigma, k, target_recall)

        def run_group(query_idx, ef):
            # Probe in collection mode (probe_ef) so table recalls match what the
            # online search actually does: explore l nodes with ef=inf, then
            # truncate to ef and continue.
            est.probe_ef = ef
            recalls = []
            for i in query_idx:
                ids, _ = ctx.search(proxies[i], k, ada=est)
                recalls.append(len(set(ids) & set(gt[i].tolist())) / k)
            kept = [r for r in recalls if r >= 1e-3]  # C++ drops zero-recall queries
            return float(np.mean(kept)) if kept else 0.0

        # score every proxy query (probe mode: collection + fixed ef = k)
        est.probe_ef = k
        scores = []
        for i in range(n_proxy):
            _, stats = ctx.search(proxies[i], k, ada=est)
            scores.append(stats["score"])
        groups = {}
        for i, s in enumerate(scores):
            groups.setdefault(int(s), []).append(i)

        table = []
        for int_score in sorted(groups):
            qidx = groups[int_score]
            ef1, ef2 = k, int(1.5 * k)
            r1, r2 = run_group(qidx, ef1), run_group(qidx, ef2)
            entries = [(ef1, r1), (ef2, r2)]
            latest_ef, latest_recall = ef2, r2
            ef_diff, recall_diff = ef2 - ef1, r2 - r1
            # The C++ breaks as soon as recall stops improving between probes.
            # At k=10 group recall is noisy (0.1 granularity), so that strands
            # the hardest groups at tiny ef values; instead keep pushing with
            # doubling steps until the target or the cap (bounded probes).
            attempts = 0
            while (
                target_recall - latest_recall > 1e-4
                and latest_ef < EF_UPPER_BOUND
                and attempts < 15
            ):
                attempts += 1
                if recall_diff > 1e-6:
                    step = int(ef_diff * (target_recall - latest_recall) / recall_diff)
                    step = max(step, int(k * 0.5))
                else:
                    step = max(latest_ef, int(k * 0.5))  # flat/noisy: double ef
                ef_diff = step
                ef = min(latest_ef + step, EF_UPPER_BOUND)
                r = run_group(qidx, ef)
                entries.append((ef, r))
                recall_diff = r - latest_recall
                latest_ef, latest_recall = ef, r
            table.append([int_score, entries])

        # WAE: weighted by group size, smallest ef reaching the target
        wae = 0.0
        for int_score, entries in table:
            ef = entries[-1][0]
            for e, r in entries:
                if r >= target_recall:
                    ef = e
                    break
            wae += len(groups[int_score]) * ef / n_proxy
        timings["table_s"] = time.perf_counter() - t0
        log(f"[offline] ef-estimation table ({len(table)} score groups), "
            f"WAE={wae:.0f}: {timings['table_s']:.1f}s")

        est.table = table
        est.wae = wae
        est._links = est._build_links()
        est.timings = timings
        return est

    # ---------- persistence ----------

    def save(self, path):
        np.savez_compressed(
            str(path) + ".npz", mu=self.mu, sigma=self.sigma
        )
        with open(str(path) + ".json", "w") as f:
            json.dump(
                {
                    "k": self.k,
                    "target_recall": self.target_recall,
                    "wae": self.wae,
                    "table": self.table,
                    "timings": getattr(self, "timings", None),
                },
                f,
            )

    @classmethod
    def load(cls, path):
        z = np.load(str(path) + ".npz")
        with open(str(path) + ".json") as f:
            meta = json.load(f)
        table = [[g[0], [tuple(p) for p in g[1]]] for g in meta["table"]]
        return cls(
            z["mu"], z["sigma"], meta["k"], meta["target_recall"],
            table=table, wae=meta["wae"],
        )
