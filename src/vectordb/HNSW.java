package vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

/**
 * Hierarchical Navigable Small World graph.
 * Each node gets a random top layer; layer 0 holds everything, higher layers are exponentially sparser.
 * Search drops from the top layer greedily, then widens to ef candidates at layer 0.
 */
final class HNSW {
    private static final class Node {
        final VectorItem item;
        final int maxLyr;
        final List<List<Integer>> nbrs = new ArrayList<>();

        Node(VectorItem item, int maxLyr) {
            this.item = item;
            this.maxLyr = maxLyr;
            for (int i = 0; i <= maxLyr; i++) nbrs.add(new ArrayList<>());
        }
    }

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int m, m0, efBuild;
    private final double mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng = new Random(42);

    HNSW(int m, int efBuild) {
        this.m = m;
        this.m0 = 2 * m;
        this.efBuild = efBuild;
        this.mL = 1.0 / Math.log(m);
    }

    private int randLevel() {
        double u = 1.0 - rng.nextDouble(); // (0,1], avoids log(0)
        return (int) Math.floor(-Math.log(u) * mL);
    }

    /** Beam search inside one layer. Returns up to ef nearest nodes, closest first. */
    private List<Candidate> searchLayer(float[] q, int ep, int ef, int layer, Distance dist) {
        Set<Integer> visited = new HashSet<>();
        PriorityQueue<Candidate> cands = new PriorityQueue<>();                       // min-heap
        PriorityQueue<Candidate> found = new PriorityQueue<>(Collections.reverseOrder()); // max-heap

        float d0 = dist.between(q, graph.get(ep).item.emb());
        visited.add(ep);
        cands.add(new Candidate(d0, ep));
        found.add(new Candidate(d0, ep));

        while (!cands.isEmpty()) {
            Candidate c = cands.poll();
            if (found.size() >= ef && c.dist() > found.peek().dist()) break;
            Node cn = graph.get(c.id());
            if (layer >= cn.nbrs.size()) continue;
            for (int nid : cn.nbrs.get(layer)) {
                Node nn = graph.get(nid);
                if (nn == null || !visited.add(nid)) continue;
                float nd = dist.between(q, nn.item.emb());
                if (found.size() < ef || nd < found.peek().dist()) {
                    cands.add(new Candidate(nd, nid));
                    found.add(new Candidate(nd, nid));
                    if (found.size() > ef) found.poll();
                }
            }
        }
        List<Candidate> res = new ArrayList<>(found);
        res.sort(null);
        return res;
    }

    private static List<Integer> selectNeighbors(List<Candidate> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        for (int i = 0; i < Math.min(cands.size(), maxM); i++) r.add(cands.get(i).id());
        return r;
    }

    void insert(VectorItem item, Distance dist) {
        int id = item.id();
        int lvl = randLevel();
        Node node = new Node(item, lvl);
        graph.put(id, node);

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        // greedy descent through layers above the new node's level
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < graph.get(ep).nbrs.size()) {
                List<Candidate> w = searchLayer(item.emb(), ep, 1, lc, dist);
                if (!w.isEmpty()) ep = w.get(0).id();
            }
        }
        // from the node's level down to 0: beam search, then connect both ways
        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<Candidate> w = searchLayer(item.emb(), ep, efBuild, lc, dist);
            int maxM = (lc == 0) ? m0 : m;
            List<Integer> sel = selectNeighbors(w, maxM);
            node.nbrs.set(lc, new ArrayList<>(sel));

            for (int nid : sel) {
                Node nb = graph.get(nid);
                if (nb == null) continue;
                while (nb.nbrs.size() <= lc) nb.nbrs.add(new ArrayList<>());
                List<Integer> conn = nb.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    List<Candidate> ds = new ArrayList<>();
                    for (int c : conn) {
                        Node cn = graph.get(c);
                        if (cn != null) ds.add(new Candidate(dist.between(nb.item.emb(), cn.item.emb()), c));
                    }
                    ds.sort(null);
                    conn.clear();
                    for (int i = 0; i < maxM && i < ds.size(); i++) conn.add(ds.get(i).id());
                }
            }
            if (!w.isEmpty()) ep = w.get(0).id();
        }
        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    List<Candidate> knn(float[] q, int k, int ef, Distance dist) {
        if (entryPt == -1) return List.of();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < graph.get(ep).nbrs.size()) {
                List<Candidate> w = searchLayer(q, ep, 1, lc, dist);
                if (!w.isEmpty()) ep = w.get(0).id();
            }
        }
        List<Candidate> w = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        return w.size() > k ? new ArrayList<>(w.subList(0, k)) : w;
    }

    void remove(int id) {
        if (!graph.containsKey(id)) return;
        for (Node nd : graph.values())
            for (List<Integer> layer : nd.nbrs) layer.removeIf(x -> x == id);
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : graph.keySet()) {
                if (nid != id) { entryPt = nid; break; }
            }
        }
        graph.remove(id);
    }

    int size() {
        return graph.size();
    }

    // ---- graph inspection for /hnsw-info ----

    record NodeView(int id, String metadata, String category, int maxLyr) {}
    record EdgeView(int src, int dst, int lyr) {}
    record GraphInfo(int topLayer, int nodeCount, int[] nodesPerLayer, int[] edgesPerLayer,
                     List<NodeView> nodes, List<EdgeView> edges) {}

    GraphInfo getInfo() {
        int maxL = Math.max(topLayer + 1, 1);
        int[] nodesPer = new int[maxL];
        int[] edgesPer = new int[maxL];
        List<NodeView> nodes = new ArrayList<>();
        List<EdgeView> edges = new ArrayList<>();
        for (Map.Entry<Integer, Node> e : graph.entrySet()) {
            int id = e.getKey();
            Node nd = e.getValue();
            nodes.add(new NodeView(id, nd.item.metadata(), nd.item.category(), nd.maxLyr));
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                nodesPer[lc]++;
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) { // count each undirected edge once
                            edgesPer[lc]++;
                            edges.add(new EdgeView(id, nid, lc));
                        }
                    }
                }
            }
        }
        return new GraphInfo(topLayer, graph.size(), nodesPer, edgesPer, nodes, edges);
    }
}
