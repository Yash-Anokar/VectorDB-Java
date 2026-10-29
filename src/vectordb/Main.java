package vectordb;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** HTTP server: demo-vector endpoints, document endpoints, and the RAG pipeline. */
public final class Main {
    static final int DIMS = 16; // demo vectors; document dims come from Ollama at runtime
    private static final Pattern DELETE_ID = Pattern.compile("^/delete/(\\d+)$");
    private static final Pattern DOC_DELETE_ID = Pattern.compile("^/doc/delete/(\\d+)$");

    private final VectorDB db = new VectorDB(DIMS);
    private final DocumentDB docDB = new DocumentDB();
    private final OllamaClient ollama = new OllamaClient();

    public static void main(String[] args) throws IOException {
        int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "8080"));
        new Main().run(port);
    }

    private void run(int port) throws IOException {
        DemoData.load(db);

        boolean up = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:" + port);
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (up ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (up) System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);

        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.setExecutor(Executors.newCachedThreadPool());
        server.createContext("/", this::handle);
        server.start();
    }

    // ---------------- routing ----------------

    private void handle(HttpExchange ex) throws IOException {
        try {
            String method = ex.getRequestMethod();
            String path = ex.getRequestURI().getPath();
            ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
            ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");

            if (method.equals("OPTIONS")) { send(ex, 204, "", "application/json"); return; }

            Map<String, String> q = query(ex.getRequestURI().getRawQuery());
            String body = method.equals("POST")
                ? new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8) : "";

            switch (method) {
                case "GET" -> {
                    switch (path) {
                        case "/" -> index(ex);
                        case "/search" -> search(ex, q);
                        case "/items" -> items(ex);
                        case "/benchmark" -> benchmark(ex, q);
                        case "/hnsw-info" -> hnswInfo(ex);
                        case "/doc/list" -> docList(ex);
                        case "/status" -> status(ex);
                        case "/stats" -> stats(ex);
                        default -> send(ex, 404, "{\"error\":\"not found\"}", "application/json");
                    }
                }
                case "POST" -> {
                    switch (path) {
                        case "/insert" -> insert(ex, body);
                        case "/doc/insert" -> docInsert(ex, body);
                        case "/doc/search" -> docSearch(ex, body);
                        case "/doc/ask" -> docAsk(ex, body);
                        default -> send(ex, 404, "{\"error\":\"not found\"}", "application/json");
                    }
                }
                case "DELETE" -> {
                    Matcher m = DELETE_ID.matcher(path);
                    Matcher dm = DOC_DELETE_ID.matcher(path);
                    if (m.matches()) json(ex, "{\"ok\":" + db.remove(Integer.parseInt(m.group(1))) + "}");
                    else if (dm.matches()) json(ex, "{\"ok\":" + docDB.remove(Integer.parseInt(dm.group(1))) + "}");
                    else send(ex, 404, "{\"error\":\"not found\"}", "application/json");
                }
                default -> send(ex, 405, "{\"error\":\"method not allowed\"}", "application/json");
            }
        } catch (Exception e) {
            e.printStackTrace();
            send(ex, 500, "{\"error\":" + Json.str(String.valueOf(e.getMessage())) + "}", "application/json");
        } finally {
            ex.close();
        }
    }

    // ---------------- demo vector endpoints ----------------

    private void search(HttpExchange ex, Map<String, String> q) throws IOException {
        float[] v = parseVec(q.getOrDefault("v", ""));
        if (v.length != DIMS) { json(ex, "{\"error\":\"need " + DIMS + "D vector\"}"); return; }
        int k = intOr(q.get("k"), 5);
        String metric = nonEmpty(q.get("metric"), "cosine");
        String algo = nonEmpty(q.get("algo"), "hnsw");

        VectorDB.SearchOut out = db.search(v, k, metric, algo);
        StringBuilder sb = new StringBuilder("{\"results\":[");
        for (int i = 0; i < out.hits().size(); i++) {
            if (i > 0) sb.append(',');
            VectorDB.Hit h = out.hits().get(i);
            sb.append("{\"id\":").append(h.id())
              .append(",\"metadata\":").append(Json.str(h.meta()))
              .append(",\"category\":").append(Json.str(h.cat()))
              .append(",\"distance\":").append(Json.num(h.dist(), 6))
              .append(",\"embedding\":").append(Json.vec(h.emb())).append('}');
        }
        sb.append("],\"latencyUs\":").append(out.micros())
          .append(",\"algo\":").append(Json.str(out.algo()))
          .append(",\"metric\":").append(Json.str(out.metric())).append('}');
        json(ex, sb.toString());
    }

    private void insert(HttpExchange ex, String body) throws IOException {
        Object j = Json.parseOrNull(body);
        String meta = Json.getStr(j, "metadata");
        String cat = Json.getStr(j, "category");
        float[] emb = Json.getFloats(j, "embedding");
        if (meta.isEmpty() || emb.length != DIMS) { json(ex, "{\"error\":\"invalid body\"}"); return; }
        int id = db.insert(meta, cat, emb, Distance.COSINE);
        json(ex, "{\"id\":" + id + "}");
    }

    private void items(HttpExchange ex) throws IOException {
        StringBuilder sb = new StringBuilder("[");
        List<VectorItem> items = db.all();
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(',');
            VectorItem v = items.get(i);
            sb.append("{\"id\":").append(v.id())
              .append(",\"metadata\":").append(Json.str(v.metadata()))
              .append(",\"category\":").append(Json.str(v.category()))
              .append(",\"embedding\":").append(Json.vec(v.emb())).append('}');
        }
        json(ex, sb.append(']').toString());
    }

    private void benchmark(HttpExchange ex, Map<String, String> q) throws IOException {
        float[] v = parseVec(q.getOrDefault("v", ""));
        if (v.length != DIMS) { json(ex, "{\"error\":\"need " + DIMS + "D vector\"}"); return; }
        int k = intOr(q.get("k"), 5);
        VectorDB.BenchOut b = db.benchmark(v, k, nonEmpty(q.get("metric"), "cosine"));
        json(ex, "{\"bruteforceUs\":" + b.bfUs() + ",\"kdtreeUs\":" + b.kdUs()
            + ",\"hnswUs\":" + b.hnswUs() + ",\"itemCount\":" + b.n() + "}");
    }

    private void hnswInfo(HttpExchange ex) throws IOException {
        HNSW.GraphInfo gi = db.hnswInfo();
        StringBuilder sb = new StringBuilder();
        sb.append("{\"topLayer\":").append(gi.topLayer())
          .append(",\"nodeCount\":").append(gi.nodeCount())
          .append(",\"nodesPerLayer\":").append(Json.ints(gi.nodesPerLayer()))
          .append(",\"edgesPerLayer\":").append(Json.ints(gi.edgesPerLayer()))
          .append(",\"nodes\":[");
        for (int i = 0; i < gi.nodes().size(); i++) {
            if (i > 0) sb.append(',');
            HNSW.NodeView n = gi.nodes().get(i);
            sb.append("{\"id\":").append(n.id())
              .append(",\"metadata\":").append(Json.str(n.metadata()))
              .append(",\"category\":").append(Json.str(n.category()))
              .append(",\"maxLyr\":").append(n.maxLyr()).append('}');
        }
        sb.append("],\"edges\":[");
        for (int i = 0; i < gi.edges().size(); i++) {
            if (i > 0) sb.append(',');
            HNSW.EdgeView e = gi.edges().get(i);
            sb.append("{\"src\":").append(e.src()).append(",\"dst\":").append(e.dst())
              .append(",\"lyr\":").append(e.lyr()).append('}');
        }
        json(ex, sb.append("]}").toString());
    }

    // ---------------- document + RAG endpoints ----------------

    private static final String OLLAMA_DOWN = "{\"error\":\"Ollama unavailable. Install from https://ollama.com "
        + "then run: ollama pull nomic-embed-text && ollama pull llama3.2\"}";

    /** POST /doc/insert {"title","text"}: chunk, embed each chunk, store. */
    private void docInsert(HttpExchange ex, String body) throws IOException {
        Object j = Json.parseOrNull(body);
        String title = Json.getStr(j, "title");
        String text = Json.getStr(j, "text");
        if (title.isEmpty() || text.isEmpty()) { json(ex, "{\"error\":\"need title and text\"}"); return; }

        List<String> chunks = TextChunker.chunk(text, 250, 30);
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb.length == 0) { json(ex, OLLAMA_DOWN); return; }
            String chunkTitle = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }
        json(ex, "{\"ids\":" + ids + ",\"chunks\":" + chunks.size() + ",\"dims\":" + docDB.dims() + "}");
    }

    private void docList(HttpExchange ex) throws IOException {
        StringBuilder sb = new StringBuilder("[");
        List<DocumentDB.DocItem> docs = docDB.all();
        for (int i = 0; i < docs.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocItem d = docs.get(i);
            String preview = d.text().length() > 120 ? d.text().substring(0, 120) + "…" : d.text();
            long words = d.text().chars().filter(c -> c == ' ').count() + 1;
            sb.append("{\"id\":").append(d.id())
              .append(",\"title\":").append(Json.str(d.title()))
              .append(",\"preview\":").append(Json.str(preview))
              .append(",\"words\":").append(words).append('}');
        }
        json(ex, sb.append(']').toString());
    }

    /** POST /doc/search {"question","k"}: retrieval only, used by the UI to highlight hits. */
    private void docSearch(HttpExchange ex, String body) throws IOException {
        Object j = Json.parseOrNull(body);
        String question = Json.getStr(j, "question");
        int k = Json.getInt(j, "k", 3);
        if (question.isEmpty()) { json(ex, "{\"error\":\"need question\"}"); return; }

        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) { json(ex, "{\"error\":\"Ollama unavailable\"}"); return; }

        List<DocumentDB.DocHit> hits = docDB.search(qEmb, k);
        StringBuilder sb = new StringBuilder("{\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocHit h = hits.get(i);
            sb.append("{\"id\":").append(h.item().id())
              .append(",\"title\":").append(Json.str(h.item().title()))
              .append(",\"distance\":").append(Json.num(h.dist(), 4)).append('}');
        }
        json(ex, sb.append("]}").toString());
    }

    /** POST /doc/ask {"question","k"}: embed, retrieve, generate. */
    private void docAsk(HttpExchange ex, String body) throws IOException {
        Object j = Json.parseOrNull(body);
        String question = Json.getStr(j, "question");
        int k = Json.getInt(j, "k", 3);
        if (question.isEmpty()) { json(ex, "{\"error\":\"need question\"}"); return; }

        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) { json(ex, "{\"error\":\"Ollama unavailable\"}"); return; }

        List<DocumentDB.DocHit> hits = docDB.search(qEmb, k);

        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            ctx.append('[').append(i + 1).append("] ").append(hits.get(i).item().title()).append(":\n")
               .append(hits.get(i).item().text()).append("\n\n");
        }
        String prompt =
            "You are a helpful assistant. Answer the user's question directly. "
            + "Use the provided context if it contains relevant information. "
            + "If it doesn't, just use your own general knowledge. "
            + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
            + "Just answer the question naturally.\n\n"
            + "Context:\n" + ctx + "Question: " + question + "\n\nAnswer:";

        String answer = ollama.generate(prompt);

        StringBuilder sb = new StringBuilder();
        sb.append("{\"answer\":").append(Json.str(answer))
          .append(",\"model\":").append(Json.str(ollama.genModel))
          .append(",\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) sb.append(',');
            DocumentDB.DocHit h = hits.get(i);
            sb.append("{\"id\":").append(h.item().id())
              .append(",\"title\":").append(Json.str(h.item().title()))
              .append(",\"text\":").append(Json.str(h.item().text()))
              .append(",\"distance\":").append(Json.num(h.dist(), 4)).append('}');
        }
        sb.append("],\"docCount\":").append(docDB.size()).append('}');
        json(ex, sb.toString());
    }

    private void status(HttpExchange ex) throws IOException {
        json(ex, "{\"ollamaAvailable\":" + ollama.isAvailable()
            + ",\"embedModel\":" + Json.str(ollama.embedModel)
            + ",\"genModel\":" + Json.str(ollama.genModel)
            + ",\"docCount\":" + docDB.size()
            + ",\"docDims\":" + docDB.dims()
            + ",\"demoDims\":" + DIMS
            + ",\"demoCount\":" + db.size() + "}");
    }

    private void stats(HttpExchange ex) throws IOException {
        json(ex, "{\"count\":" + db.size() + ",\"dims\":" + DIMS
            + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
            + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
    }

    private void index(HttpExchange ex) throws IOException {
        Path p = Path.of("index.html");
        if (!Files.exists(p)) { send(ex, 404, "", "text/html"); return; }
        send(ex, 200, Files.readString(p), "text/html; charset=utf-8");
    }

    // ---------------- helpers ----------------

    private static void json(HttpExchange ex, String body) throws IOException {
        send(ex, 200, body, "application/json");
    }

    private static void send(HttpExchange ex, int status, String body, String type) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", type.contains("charset") ? type : type + "; charset=utf-8");
        if (status == 204) { ex.sendResponseHeaders(204, -1); return; }
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    private static Map<String, String> query(String raw) {
        Map<String, String> m = new HashMap<>();
        if (raw == null || raw.isEmpty()) return m;
        for (String pair : raw.split("&")) {
            int eq = pair.indexOf('=');
            String k = eq < 0 ? pair : pair.substring(0, eq);
            String v = eq < 0 ? "" : pair.substring(eq + 1);
            m.put(URLDecoder.decode(k, StandardCharsets.UTF_8), URLDecoder.decode(v, StandardCharsets.UTF_8));
        }
        return m;
    }

    private static float[] parseVec(String s) {
        List<Float> out = new ArrayList<>();
        for (String t : s.split(",")) {
            try { out.add(Float.parseFloat(t.trim())); } catch (NumberFormatException ignored) {}
        }
        float[] v = new float[out.size()];
        for (int i = 0; i < v.length; i++) v[i] = out.get(i);
        return v;
    }

    private static int intOr(String s, int def) {
        try { return Integer.parseInt(s); } catch (Exception e) { return def; }
    }

    private static String nonEmpty(String s, String def) {
        return s == null || s.isEmpty() ? def : s;
    }
}
