# VectorDB (Java)

A vector database built from scratch in Java 17+, with a web UI and a local-LLM RAG pipeline.
This is a port of the C++ original. Same algorithms, same REST API, same `index.html`.

- Search: HNSW, KD-Tree, Brute Force, with cosine, Euclidean or Manhattan distance
- 20 demo vectors (16D) in 4 categories, shown on a live 2D PCA plot
- Documents: paste text, it is chunked (250 words, 30 overlap), embedded by Ollama `nomic-embed-text`, and indexed in HNSW
- RAG: your question is embedded, the top chunks are retrieved, `llama3.2` writes the answer

No libraries, no Maven needed. It uses only the JDK (`com.sun.net.httpserver` and `java.net.http`), which replaces `httplib.h`.

## Run

You need a JDK (17 or newer) and, for the document and RAG tabs, [Ollama](https://ollama.com).

```
ollama pull nomic-embed-text
ollama pull llama3.2
```

Linux / macOS:
```
./run.sh
```

Windows:
```
run.bat
```

Then open http://localhost:8080. The demo search tab works without Ollama.

Manual build:
```
javac -d out src/vectordb/*.java
java -cp out vectordb.Main
```

Run it from this folder, because the server reads `index.html` from the current directory.

Optional environment variables: `PORT` (default 8080), `OLLAMA_URL` (default http://127.0.0.1:11434).
Models are set in `OllamaClient.java` (`embedModel`, `genModel`). For a faster LLM, use `llama3.2:1b`.

## Files

```
src/vectordb/
  Main.java          HTTP server, routes, RAG endpoints
  VectorDB.java      16D demo index (brute force + KD-tree + HNSW behind one lock)
  DocumentDB.java    HNSW over real Ollama embeddings
  HNSW.java          multilayer small-world graph
  KDTree.java        k-d tree
  BruteForce.java    exact linear scan
  Distance.java      cosine, Euclidean, Manhattan
  OllamaClient.java  /api/embeddings and /api/generate
  TextChunker.java   overlapping word windows
  Json.java          small JSON reader/writer
  DemoData.java      the 20 demo vectors
  Candidate.java, VectorItem.java   small data types
index.html           frontend (unchanged from the C++ version)
```

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/search?v=f1,...,f16&k=5&metric=cosine&algo=hnsw` | k-NN search (`algo`: hnsw, kdtree, bruteforce) |
| POST | `/insert` | insert a demo vector `{"metadata","category","embedding":[16 floats]}` |
| DELETE | `/delete/:id` | delete a demo vector |
| GET | `/items` | list demo vectors |
| GET | `/benchmark?v=...&k=5&metric=cosine` | time all three algorithms |
| GET | `/hnsw-info` | graph layers, nodes, edges |
| GET | `/stats` | database stats |
| POST | `/doc/insert` | `{"title","text"}`: chunk, embed, store |
| GET | `/doc/list` | stored chunks |
| DELETE | `/doc/delete/:id` | delete a chunk |
| POST | `/doc/search` | `{"question","k"}`: retrieval only |
| POST | `/doc/ask` | `{"question","k"}`: retrieve and generate |
| GET | `/status` | Ollama status and models |

## Differences from the C++ version

- JSON is parsed properly (nesting, `\uXXXX` escapes) instead of string scanning.
- HNSW levels use Java's `Random(42)`, so the upper-layer shape can differ from the C++ build. Search results are identical on the demo data.
- Compiles and runs the same on Windows, Linux and macOS. No `-lws2_32` flag.
- Like the original, data lives in memory and is lost on restart.
