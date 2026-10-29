package vectordb;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/** Thin client for a local Ollama server: embeddings and text generation. */
final class OllamaClient {
    private final String baseUrl;
    private final HttpClient http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(3))
        .proxy(java.net.ProxySelector.of(null)) // Ollama is local: never route through a system proxy
        .build();

    String embedModel = "nomic-embed-text";
    String genModel = "llama3.2";

    OllamaClient() {
        this(System.getenv().getOrDefault("OLLAMA_URL", "http://127.0.0.1:11434"));
    }

    OllamaClient(String baseUrl) {
        this.baseUrl = baseUrl;
    }

    boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/api/tags"))
                .timeout(Duration.ofSeconds(2)).GET().build();
            return http.send(req, HttpResponse.BodyHandlers.ofString()).statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    /** Returns an empty array if Ollama is down or the model is missing. */
    float[] embed(String text) {
        String body = "{\"model\":" + Json.str(embedModel) + ",\"prompt\":" + Json.str(text) + "}";
        String resp = post("/api/embeddings", body, 30);
        if (resp == null) return new float[0];
        return Json.getFloats(Json.parseOrNull(resp), "embedding");
    }

    /** Returns an error string if Ollama is unavailable. */
    String generate(String prompt) {
        String body = "{\"model\":" + Json.str(genModel) + ",\"prompt\":" + Json.str(prompt)
            + ",\"stream\":false}";
        String resp = post("/api/generate", body, 180); // LLMs can be slow
        if (resp == null) return "ERROR: Ollama unavailable. Run: ollama serve";
        return Json.getStr(Json.parseOrNull(resp), "response");
    }

    private String post(String path, String json, int readTimeoutSec) {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + path))
                .timeout(Duration.ofSeconds(readTimeoutSec))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
            HttpResponse<String> res = http.send(req, HttpResponse.BodyHandlers.ofString());
            return res.statusCode() == 200 ? res.body() : null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        } catch (Exception e) {
            return null;
        }
    }
}
