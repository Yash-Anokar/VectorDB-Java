package vectordb;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** The 16-D demo index: one store, three search structures behind one lock. */
final class VectorDB {
    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw = new HNSW(16, 200);
    private int nextId = 1;
    final int dims;

    VectorDB(int dims) {
        this.dims = dims;
        this.kdt = new KDTree(dims);
    }

    synchronized int insert(String meta, String cat, float[] emb, Distance dist) {
        VectorItem v = new VectorItem(nextId++, meta, cat, emb);
        store.put(v.id(), v);
        bf.insert(v);
        kdt.insert(v);
        hnsw.insert(v, dist);
        return v.id();
    }

    synchronized boolean remove(int id) {
        if (store.remove(id) == null) return false;
        bf.remove(id);
        hnsw.remove(id);
        kdt.rebuild(store.values());
        return true;
    }

    record Hit(int id, String meta, String cat, float[] emb, float dist) {}
    record SearchOut(List<Hit> hits, long micros, String algo, String metric) {}

    synchronized SearchOut search(float[] q, int k, String metric, String algo) {
        Distance dfn = Distance.byName(metric);
        long t0 = System.nanoTime();
        List<Candidate> raw = switch (algo) {
            case "bruteforce" -> bf.knn(q, k, dfn);
            case "kdtree" -> kdt.knn(q, k, dfn);
            default -> hnsw.knn(q, k, 50, dfn);
        };
        long us = (System.nanoTime() - t0) / 1000;

        List<Hit> hits = new ArrayList<>();
        for (Candidate c : raw) {
            VectorItem v = store.get(c.id());
            if (v != null) hits.add(new Hit(v.id(), v.metadata(), v.category(), v.emb(), c.dist()));
        }
        return new SearchOut(hits, us, algo, metric);
    }

    record BenchOut(long bfUs, long kdUs, long hnswUs, int n) {}

    synchronized BenchOut benchmark(float[] q, int k, String metric) {
        Distance dfn = Distance.byName(metric);
        long bf1 = time(() -> bf.knn(q, k, dfn));
        long kd1 = time(() -> kdt.knn(q, k, dfn));
        long hn1 = time(() -> hnsw.knn(q, k, 50, dfn));
        return new BenchOut(bf1, kd1, hn1, store.size());
    }

    private static long time(Runnable r) {
        long t = System.nanoTime();
        r.run();
        return (System.nanoTime() - t) / 1000;
    }

    synchronized List<VectorItem> all() {
        return new ArrayList<>(store.values());
    }

    synchronized HNSW.GraphInfo hnswInfo() {
        return hnsw.getInfo();
    }

    synchronized int size() {
        return store.size();
    }
}
