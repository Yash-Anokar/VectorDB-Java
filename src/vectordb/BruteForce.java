package vectordb;

import java.util.ArrayList;
import java.util.List;

/** Exact linear scan. O(N*d). The baseline the other indexes are compared against. */
final class BruteForce {
    private final List<VectorItem> items = new ArrayList<>();

    void insert(VectorItem v) {
        items.add(v);
    }

    void remove(int id) {
        items.removeIf(v -> v.id() == id);
    }

    List<Candidate> knn(float[] q, int k, Distance dist) {
        List<Candidate> r = new ArrayList<>(items.size());
        for (VectorItem v : items) r.add(new Candidate(dist.between(q, v.emb()), v.id()));
        r.sort(null);
        return r.size() > k ? new ArrayList<>(r.subList(0, k)) : r;
    }
}
