package vectordb;

/** One stored vector with its label and category. */
record VectorItem(int id, String metadata, String category, float[] emb) {}
