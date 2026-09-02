// Lives in Lucene's own package so it can override the package-visible
// AbstractKnnVectorQuery.mergeLeafResults hook. That hook receives one TopDocs
// per segment whose totalHits is the KnnCollector's visitedCount() — i.e. the
// number of vectors actually scored during the graph walk. Lucene drops this
// number when it rewrites the kNN query into a DocAndScoreQuery, so capturing
// it here is the only way to get an exact, machine-independent cost measure.
package org.apache.lucene.search;

/** Plain HNSW search that records how many vectors it scored. */
final class CountingKnnQuery extends KnnFloatVectorQuery {
  long visited;

  CountingKnnQuery(String field, float[] target, int k) {
    super(field, target, k);
  }

  @Override
  protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
    long v = 0;
    for (TopDocs td : perLeafResults) {
      v += td.totalHits.value();
    }
    visited = v;
    return super.mergeLeafResults(perLeafResults);
  }
}

/** The authors' merged early-termination query, same instrumentation. */
final class CountingPatienceQuery extends PatienceKnnVectorQuery {
  long visited;

  CountingPatienceQuery(KnnFloatVectorQuery delegate, double saturationThreshold, int patience) {
    super(delegate, saturationThreshold, patience);
  }

  @Override
  protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
    long v = 0;
    for (TopDocs td : perLeafResults) {
      v += td.totalHits.value();
    }
    visited = v;
    return super.mergeLeafResults(perLeafResults);
  }
}

/** Factory reachable from the default package. */
public final class CountingQueries {
  private CountingQueries() {}

  /** Lucene 10.5.0 defaults, read off PatienceKnnVectorQuery's bytecode. */
  public static final double DEFAULT_GAMMA = 0.995;

  public static int defaultPatience(int k) {
    return Math.max(7, (int) (k * 0.3));
  }

  public static Query baseline(String field, float[] q, int ef) {
    return new CountingKnnQuery(field, q, ef);
  }

  public static Query patience(String field, float[] q, int ef) {
    // Exactly what PatienceKnnVectorQuery.fromFloatQuery(knn) does: gamma=0.995,
    // patience=max(7, 0.3 * the query's k) — and the query's k here is ef.
    KnnFloatVectorQuery delegate = new CountingKnnQuery(field, q, ef);
    return new CountingPatienceQuery(delegate, DEFAULT_GAMMA, defaultPatience(ef));
  }

  public static long visitedOf(Query q) {
    if (q instanceof CountingPatienceQuery p) {
      return p.visited;
    }
    return ((CountingKnnQuery) q).visited;
  }
}
