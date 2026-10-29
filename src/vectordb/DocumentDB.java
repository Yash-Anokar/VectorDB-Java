package vectordb;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** HNSW over real Ollama embeddings (e.g. 768-D). Cosine distance only. */
final class DocumentDB {
    record DocItem(int id, String title, String text, float[] emb) {}
    record DocHit(float dist, DocItem item) {}

    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce(); // exact fallback for small sets
    private int nextId = 1;
    private int dims = 0; // set by the first inserted embedding

    synchronized int insert(String title, String text, float[] emb) {
        if (dims == 0) dims = emb.length;
        DocItem item = new DocItem(nextId++, title, text, emb);
        store.put(item.id(), item);
        VectorItem vi = new VectorItem(item.id(), title, "doc", emb);
        hnsw.insert(vi, Distance.COSINE);
        bf.insert(vi);
        return item.id();
    }

    /** Top-k chunks, dropping anything farther than maxDist (cosine distance). */
    synchronized List<DocHit> search(float[] q, int k, float maxDist) {
        if (store.isEmpty()) return List.of();
        List<Candidate> raw = store.size() < 10
            ? bf.knn(q, k, Distance.COSINE)
            : hnsw.knn(q, k, 50, Distance.COSINE);
        List<DocHit> out = new ArrayList<>();
        for (Candidate c : raw) {
            DocItem d = store.get(c.id());
            if (d != null && c.dist() <= maxDist) out.add(new DocHit(c.dist(), d));
        }
        return out;
    }

    List<DocHit> search(float[] q, int k) {
        return search(q, k, 0.7f);
    }

    synchronized boolean remove(int id) {
        if (store.remove(id) == null) return false;
        hnsw.remove(id);
        bf.remove(id);
        return true;
    }

    synchronized List<DocItem> all() {
        return new ArrayList<>(store.values());
    }

    synchronized int size() {
        return store.size();
    }

    synchronized int dims() {
        return dims;
    }
}
