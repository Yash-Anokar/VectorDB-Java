package vectordb;

/** Distance metrics. Smaller is closer for all three. */
@FunctionalInterface
interface Distance {
    float between(float[] a, float[] b);

    Distance EUCLIDEAN = (a, b) -> {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    };

    /** Cosine distance = 1 - cosine similarity. Zero vectors count as maximally far (1). */
    Distance COSINE = (a, b) -> {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / ((float) Math.sqrt(na) * (float) Math.sqrt(nb));
    };

    Distance MANHATTAN = (a, b) -> {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    };

    /** Unknown names fall back to Euclidean, same as the C++ version. */
    static Distance byName(String name) {
        return switch (name) {
            case "cosine" -> COSINE;
            case "manhattan" -> MANHATTAN;
            default -> EUCLIDEAN;
        };
    }
}
