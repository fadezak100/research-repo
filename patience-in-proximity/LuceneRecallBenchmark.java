/**
 * Replication of "Patience in Proximity" (Teofili & Lin, ECIR 2025) using the
 * authors' OWN implementation — PatienceKnnVectorQuery, which they merged into
 * Apache Lucene (>= 10.2) — at its shipped defaults (gamma=0.995,
 * patience=max(7, 0.3k)).
 *
 * What the paper reports (Table 1): one operating point per method, per dataset,
 * with QPS and recall. What it does NOT report: an efSearch sweep. Without a
 * sweep you cannot tell a genuine frontier shift from "we stopped early and
 * landed on a cheaper point of the same curve". This benchmark adds the sweep.
 *
 * Cost is measured as vectors scored per query (exact, deterministic, machine
 * independent) as well as QPS, because QPS alone cannot be compared across
 * machines or JIT states.
 *
 * Usage:
 *   java LuceneRecallBenchmark <dataDir> <indexDir> <resultsJson>
 */

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.StringJoiner;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.search.CountingQueries;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class LuceneRecallBenchmark {

  static final String FIELD = "vec";
  static final int DIM = 128;
  static final int NQ = 1000;        // queries graded (we have exact GT@1000 for these)
  static final int GT_COLS = 1000;   // neighbors1000.bin is NQ x 1000
  static final int WARMUP = 100;

  // Same efSearch grids as the Python E8 sweep, so the two engines are
  // directly comparable. The FIRST entry of each grid is ef == k: that is the
  // real Lucene default, because Query-level Lucene has no separate ef knob —
  // an application asking for k neighbors gets a beam of exactly k.
  static int[] efGrid(int k) {
    return switch (k) {
      case 10 -> new int[] {10, 20, 40, 80, 160, 320, 640};
      case 100 -> new int[] {100, 150, 200, 300, 400, 600, 800, 1200};
      case 1000 -> new int[] {1000, 1250, 1500, 2000, 3000, 4000};
      default -> throw new IllegalArgumentException("no grid for k=" + k);
    };
  }

  public static void main(String[] args) throws Exception {
    Path dataDir = Path.of(args[0]);
    Path indexDir = Path.of(args[1]);
    Path outJson = Path.of(args[2]);

    float[][] test = readFloats(dataDir.resolve("test.bin"), NQ);
    int[][] gt = readInts(dataDir.resolve("neighbors1000.bin"), NQ, GT_COLS);

    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      IndexSearcher searcher = new IndexSearcher(reader);
      System.out.println("segments=" + reader.leaves().size() + " docs=" + reader.maxDoc());
      if (reader.leaves().size() != 1) {
        throw new IllegalStateException("expected a single segment; per-leaf topK "
            + "reduction would make the ef sweep mean something else");
      }
      int[] docToRow = loadDocToRow(reader);

      List<String> rows = new ArrayList<>();
      System.out.printf(Locale.ROOT, "%n%5s %-9s %6s %6s %10s %12s %10s%n",
          "k", "method", "ef", "Delta", "recall@k", "visited/q", "QPS");

      for (int k : new int[] {10, 100, 1000}) {
        for (String method : new String[] {"baseline", "patience"}) {
          for (int ef : efGrid(k)) {
            for (int q = 0; q < WARMUP; q++) {
              searcher.search(makeQuery(method, test[q], ef), k);
            }

            double[] recalls = new double[NQ];
            long[] visited = new long[NQ];
            long t0 = System.nanoTime();
            for (int q = 0; q < NQ; q++) {
              Query query = makeQuery(method, test[q], ef);
              TopDocs td = searcher.search(query, k);
              visited[q] = CountingQueries.visitedOf(query);
              recalls[q] = recallOf(td, docToRow, gt[q], k);
            }
            double elapsed = (System.nanoTime() - t0) / 1e9;

            double recall = Arrays.stream(recalls).average().orElse(0);
            double visitAvg = Arrays.stream(visited).average().orElse(0);
            double qps = NQ / elapsed;
            int delta = method.equals("patience") ? CountingQueries.defaultPatience(ef) : 0;

            System.out.printf(Locale.ROOT, "%5d %-9s %6d %6d %10.4f %12.1f %10.1f%n",
                k, method, ef, delta, recall, visitAvg, qps);
            rows.add(row(k, method, ef, delta, recall, visitAvg, qps, recalls, visited));
          }
        }
      }

      String json = "{\n"
          + "  \"dataset\": \"sift-128-euclidean\",\n"
          + "  \"engine\": \"lucene-10.5.0 (PatienceKnnVectorQuery, author implementation)\",\n"
          + "  \"index\": {\"type\": \"HNSW\", \"M\": 16, \"beamWidth\": 100},\n"
          + "  \"patience_defaults\": {\"gamma\": " + CountingQueries.DEFAULT_GAMMA
          + ", \"delta\": \"max(7, 0.3*ef)\"},\n"
          + "  \"n_queries\": " + NQ + ",\n"
          + "  \"threads\": 1,\n"
          + "  \"note\": \"ef is the Lucene query k; results truncated to the graded k. "
          + "ef == k is the true Lucene default operating point.\",\n"
          + "  \"sweep\": [\n" + String.join(",\n", rows) + "\n  ]\n}\n";
      Files.write(outJson, json.getBytes(StandardCharsets.UTF_8));
      System.out.println("\nSaved " + outJson);
    }
  }

  static Query makeQuery(String method, float[] vector, int ef) {
    return method.equals("patience")
        ? CountingQueries.patience(FIELD, vector, ef)
        : CountingQueries.baseline(FIELD, vector, ef);
  }

  /** recall@k against the exact top-k; order ignored. */
  static double recallOf(TopDocs td, int[] docToRow, int[] goldenRow, int k) {
    Set<Integer> golden = new HashSet<>();
    for (int j = 0; j < k; j++) {
      golden.add(goldenRow[j]);
    }
    int hits = 0;
    for (int j = 0; j < Math.min(td.scoreDocs.length, k); j++) {
      if (golden.contains(docToRow[td.scoreDocs[j].doc])) {
        hits++;
      }
    }
    return hits / (double) k;
  }

  static String row(int k, String method, int ef, int delta, double recall,
      double visitAvg, double qps, double[] recalls, long[] visited) {
    StringJoiner r = new StringJoiner(",");
    for (double v : recalls) {
      r.add(String.format(Locale.ROOT, "%.4f", v));
    }
    StringJoiner v = new StringJoiner(",");
    for (long n : visited) {
      v.add(Long.toString(n));
    }
    return String.format(Locale.ROOT,
        "    {\"k\": %d, \"method\": \"%s\", \"ef\": %d, \"delta\": %d, "
        + "\"recall\": %.4f, \"visited_avg\": %.1f, \"qps\": %.1f, "
        + "\"recalls\": [%s], \"visited\": [%s]}",
        k, method, ef, delta, recall, visitAvg, qps, r, v);
  }

  static int[] loadDocToRow(DirectoryReader reader) throws IOException {
    StoredFields stored = reader.storedFields();
    int[] map = new int[reader.maxDoc()];
    for (int d = 0; d < reader.maxDoc(); d++) {
      map[d] = stored.document(d).getField("id").numericValue().intValue();
    }
    return map;
  }

  static float[][] readFloats(Path file, int rows) throws IOException {
    float[][] out = new float[rows][DIM];
    try (FileChannel ch = FileChannel.open(file)) {
      FloatBuffer fb = ch.map(FileChannel.MapMode.READ_ONLY, 0, (long) rows * DIM * 4)
          .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
      for (int i = 0; i < rows; i++) {
        fb.get(out[i]);
      }
    }
    return out;
  }

  static int[][] readInts(Path file, int rows, int cols) throws IOException {
    int[][] out = new int[rows][cols];
    try (FileChannel ch = FileChannel.open(file)) {
      IntBuffer ib = ch.map(FileChannel.MapMode.READ_ONLY, 0, (long) rows * cols * 4)
          .order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
      for (int i = 0; i < rows; i++) {
        ib.get(out[i]);
      }
    }
    return out;
  }
}
