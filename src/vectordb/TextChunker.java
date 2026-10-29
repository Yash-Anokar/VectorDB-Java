package vectordb;

import java.util.ArrayList;
import java.util.List;

/** Splits text into overlapping word windows for embedding. */
final class TextChunker {
    private TextChunker() {}

    static List<String> chunk(String text, int chunkWords, int overlapWords) {
        String trimmed = text.trim();
        if (trimmed.isEmpty()) return List.of();
        String[] words = trimmed.split("\\s+");
        if (words.length <= chunkWords) return List.of(text); // short text stays whole

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            chunks.add(String.join(" ", java.util.Arrays.copyOfRange(words, i, end)));
            if (end == words.length) break;
        }
        return chunks;
    }
}
