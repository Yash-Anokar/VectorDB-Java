package vectordb;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Plain (unbalanced) k-d tree. Splits on axis depth % dims.
 * Pruning compares the single-axis gap to the current k-th best distance,
 * which is exact for Euclidean and only a heuristic for cosine/Manhattan (same as the C++ original).
 */
final class KDTree {
    private static final class Node {
        final VectorItem item;
        Node left, right;
        Node(VectorItem item) { this.item = item; }
    }

    private Node root;
    private final int dims;

    KDTree(int dims) {
        this.dims = dims;
    }

    void insert(VectorItem v) {
        root = insert(root, v, 0);
    }

    private Node insert(Node n, VectorItem v, int depth) {
        if (n == null) return new Node(v);
        int ax = depth % dims;
        if (v.emb()[ax] < n.item.emb()[ax]) n.left = insert(n.left, v, depth + 1);
        else n.right = insert(n.right, v, depth + 1);
        return n;
    }

    void rebuild(Collection<VectorItem> items) {
        root = null;
        for (VectorItem v : items) insert(v);
    }

    List<Candidate> knn(float[] q, int k, Distance dist) {
        PriorityQueue<Candidate> heap = new PriorityQueue<>(Collections.reverseOrder()); // max-heap
        search(root, q, k, 0, dist, heap);
        List<Candidate> r = new ArrayList<>(heap);
        r.sort(null);
        return r;
    }

    private void search(Node n, float[] q, int k, int depth, Distance dist, PriorityQueue<Candidate> heap) {
        if (n == null) return;
        float dn = dist.between(q, n.item.emb());
        if (heap.size() < k || dn < heap.peek().dist()) {
            heap.add(new Candidate(dn, n.item.id()));
            if (heap.size() > k) heap.poll();
        }
        int ax = depth % dims;
        float diff = q[ax] - n.item.emb()[ax];
        Node closer = diff < 0 ? n.left : n.right;
        Node farther = diff < 0 ? n.right : n.left;
        search(closer, q, k, depth + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().dist())
            search(farther, q, k, depth + 1, dist, heap);
    }
}
