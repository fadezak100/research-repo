/**
 * SIFT1M benchmark on Lucene's HNSW — the implementation the paper used.
 *
 * Compares standard HNSW search (KnnFloatVectorQuery) against the paper's
 * early-termination strategy, which the authors merged into Lucene as
 * PatienceKnnVectorQuery (Lucene >= 10.2).
 *
 * Usage:
 *   java LuceneHnswBenchmark index  <dataDir> <indexDir>
 *   java LuceneHnswBenchmark search <dataDir> <indexDir> <resultsJson>
 *
 * The flow mirrors faiss/baseline_hnsw.py exactly:
 *   load vectors -> build/load HNSW index -> sweep efSearch -> grade vs golden answers
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
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

// Lucene classes. "document" = how you feed data in, "index" = the stored
// structure, "search" = the query side, "store" = where files live on disk.
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.PatienceKnnVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class LuceneHnswBenchmark {

  // ---- experiment constants (same values as the Python/Faiss baseline) ----
  static final String FIELD = "vec";        // name of the vector field inside the index
  static final int DIM = 128;               // SIFT vectors are 128-dimensional
  static final int NTRAIN = 1_000_000;      // database size
  static final int NTEST = 10_000;          // number of queries
  static final int GT_COLS = 100;           // golden file stores top-100 true neighbors per query
  static final int K = 10;                  // we grade recall@10
  static final int[] EF_VALUES = {10, 20, 40, 80, 160, 320};  // efSearch sweep
  static final int WARMUP = 100;            // untimed queries before each timed run

  public static void main(String[] args) throws Exception {
    // First command-line argument picks the mode, like a subcommand.
    switch (args[0]) {
      case "index" -> buildIndex(Path.of(args[1]), Path.of(args[2]));
      case "search" -> search(Path.of(args[1]), Path.of(args[2]), Path.of(args[3]));
      default -> throw new IllegalArgumentException("mode must be 'index' or 'search'");
    }
  }

  // ==========================================================================
  // MODE 1: build the HNSW index (equivalent of faiss index.add(train))
  // ==========================================================================
  static void buildIndex(Path dataDir, Path indexDir) throws IOException {
    // Same caching idea as the .faiss file: if the index folder is non-empty,
    // assume it was built before and skip the ~4 minute build.
    if (Files.exists(indexDir) && Files.list(indexDir).findAny().isPresent()) {
      System.out.println("Index already exists at " + indexDir + ", skipping build.");
      return;
    }
    float[][] train = readFloats(dataDir.resolve("train.bin"), NTRAIN);
    System.out.println("Loaded train vectors: " + train.length + " x " + DIM);

    IndexWriterConfig iwc = new IndexWriterConfig();
    // Lucene normally flushes small "segments" (sub-indexes) to disk as RAM
    // fills up, each with its own HNSW graph. A big RAM buffer lets all 1M
    // vectors land in ONE segment = one graph, which is what we want to
    // benchmark. (Search over many small graphs would measure the wrong thing.)
    iwc.setRAMBufferSizeMB(1990);
    iwc.setUseCompoundFile(false);

    long t0 = System.nanoTime();
    try (Directory dir = FSDirectory.open(indexDir);
        IndexWriter writer = new IndexWriter(dir, iwc)) {
      for (int i = 0; i < NTRAIN; i++) {
        // Lucene indexes "documents". Each of ours has just two fields:
        //  - the vector itself (EUCLIDEAN = same distance metric as SIFT/Faiss)
        //  - the row number, stored so we can map results back to golden IDs
        Document doc = new Document();
        doc.add(new KnnFloatVectorField(FIELD, train[i], VectorSimilarityFunction.EUCLIDEAN));
        doc.add(new StoredField("id", i));
        // This call is where the HNSW insertion happens: the new vector is
        // wired into the graph (M=16 links, beamWidth=100 — Lucene defaults).
        writer.addDocument(doc);
        if ((i + 1) % 100_000 == 0) {
          System.out.printf(Locale.ROOT, "  %,d docs, %.0fs elapsed%n",
              i + 1, (System.nanoTime() - t0) / 1e9);
        }
      }
      // Safety net: if more than one segment was flushed anyway, merge them
      // into one. No-op when there is already a single segment.
      System.out.println("Merging to a single segment ...");
      writer.forceMerge(1);
      writer.commit();  // make everything durable on disk
    }
    System.out.printf(Locale.ROOT, "Index build took %.0fs%n", (System.nanoTime() - t0) / 1e9);
  }

  // ==========================================================================
  // MODE 2: the benchmark — sweep efSearch for baseline and patience
  // ==========================================================================
  static void search(Path dataDir, Path indexDir, Path resultsJson) throws IOException {
    float[][] test = readFloats(dataDir.resolve("test.bin"), NTEST);
    int[][] gt = readInts(dataDir.resolve("neighbors.bin"), NTEST, GT_COLS);

    try (Directory dir = FSDirectory.open(indexDir);
        DirectoryReader reader = DirectoryReader.open(dir)) {
      // IndexSearcher with no thread pool argument = single-threaded search.
      // Deliberate: stable timings, and it matches how the paper measures.
      IndexSearcher searcher = new IndexSearcher(reader);
      System.out.println("Segments: " + reader.leaves().size()
          + ", docs: " + reader.maxDoc());

      // Lucene identifies hits by internal docID, but the golden answers use
      // our row numbers. Build the docID -> row translation table up front.
      int[] docToRow = loadDocToRow(reader);

      List<String> rows = new ArrayList<>();
      System.out.printf(Locale.ROOT, "%n%-9s %8s %10s %10s %10s%n",
          "method", "ef", "recall@10", "QPS", "ms/query");

      // Two methods x six efSearch values = 12 timed runs of 10k queries.
      for (String method : new String[] {"baseline", "patience"}) {
        for (int ef : EF_VALUES) {
          // Warmup (untimed): first queries after opening an index are slow
          // for one-off reasons (files not yet in OS cache, JIT compilation).
          for (int q = 0; q < WARMUP; q++) {
            searcher.search(makeQuery(method, test[q], ef), K);
          }

          // Timed loop: one query at a time, collect the returned docIDs.
          int[][] found = new int[NTEST][];
          long t0 = System.nanoTime();
          for (int q = 0; q < NTEST; q++) {
            TopDocs td = searcher.search(makeQuery(method, test[q], ef), K);
            int[] ids = new int[td.scoreDocs.length];
            for (int j = 0; j < ids.length; j++) {
              ids[j] = td.scoreDocs[j].doc;   // internal docID of each hit
            }
            found[q] = ids;
          }
          double elapsed = (System.nanoTime() - t0) / 1e9;

          // Grading happens OUTSIDE the timed region on purpose — we measure
          // search speed, not bookkeeping.
          double recall = recallAt10(found, docToRow, gt);
          double qps = NTEST / elapsed;
          System.out.printf(Locale.ROOT, "%-9s %8d %10.4f %10.0f %10.3f%n",
              method, ef, recall, qps, 1000.0 * elapsed / NTEST);
          rows.add(String.format(Locale.ROOT,
              "    {\"method\": \"%s\", \"efSearch\": %d, \"recall@10\": %.4f, \"qps\": %.1f}",
              method, ef, recall, qps));
        }
      }

      // Write results as JSON (built by hand — not worth a library for this).
      String json = "{\n  \"dataset\": \"sift-128-euclidean\",\n"
          + "  \"engine\": \"lucene-10.5.0\",\n"
          + "  \"index\": {\"type\": \"HNSW\", \"M\": 16, \"beamWidth\": 100},\n"
          + "  \"k\": " + K + ",\n  \"threads\": 1,\n  \"sweep\": [\n"
          + String.join(",\n", rows) + "\n  ]\n}\n";
      Files.write(resultsJson, json.getBytes(StandardCharsets.UTF_8));
      System.out.println("\nSaved results to " + resultsJson);
    }
  }

  /**
   * THE HEART OF THE EXPERIMENT. Both methods run the exact same HNSW graph
   * walk with an exploration width of ef; the only difference is that
   * "patience" wraps the query in the authors' early-termination logic, which
   * watches the top-k candidates during traversal and stops once they stop
   * changing (saturation). This one-line wrapper IS the paper's contribution.
   */
  static Query makeQuery(String method, float[] vector, int ef) {
    KnnFloatVectorQuery knn = new KnnFloatVectorQuery(FIELD, vector, ef);
    return method.equals("patience") ? PatienceKnnVectorQuery.fromFloatQuery(knn) : knn;
  }

  /**
   * Same grading as the Python version: for each query, count how many of the
   * 10 returned IDs appear in the golden top-10, then divide by 10k * 10.
   * Order is ignored — recall only asks "did you find them".
   */
  static double recallAt10(int[][] found, int[] docToRow, int[][] gt) {
    long hits = 0;
    for (int q = 0; q < NTEST; q++) {
      Set<Integer> golden = new HashSet<>();
      for (int j = 0; j < K; j++) {
        golden.add(gt[q][j]);   // true top-10 for this query
      }
      for (int docId : found[q]) {
        if (golden.contains(docToRow[docId])) {  // translate docID -> row, then check
          hits++;
        }
      }
    }
    return hits / (double) (NTEST * K);
  }

  /**
   * Reads back the stored "id" field of every document to build the
   * docID -> row table. With our build (one thread, one segment) docIDs come
   * out equal to insertion order — the printed check confirms it — but we
   * never rely on that silently.
   */
  static int[] loadDocToRow(DirectoryReader reader) throws IOException {
    StoredFields stored = reader.storedFields();
    int[] map = new int[reader.maxDoc()];
    boolean identity = true;
    for (int d = 0; d < reader.maxDoc(); d++) {
      map[d] = stored.document(d).getField("id").numericValue().intValue();
      if (map[d] != d) {
        identity = false;
      }
    }
    System.out.println("docID == row order: " + identity);
    return map;
  }

  // ---- binary file readers ------------------------------------------------
  // The .bin files are raw little-endian numbers dumped by export_sift_bins.py
  // (Java can't read HDF5 directly). "Little-endian" matters: NumPy wrote the
  // bytes in that order, while Java's default is the opposite — without
  // .order(LITTLE_ENDIAN) every number would be garbage.

  static float[][] readFloats(Path file, int rows) throws IOException {
    float[][] out = new float[rows][DIM];
    try (FileChannel ch = FileChannel.open(file)) {
      // Memory-map the file (like np.fromfile) and view it as floats.
      FloatBuffer fb = ch.map(FileChannel.MapMode.READ_ONLY, 0, (long) rows * DIM * 4)
          .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
      for (int i = 0; i < rows; i++) {
        fb.get(out[i]);   // copy the next 128 floats = one vector
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
