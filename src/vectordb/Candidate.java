package vectordb;

import java.util.Comparator;

/** A (distance, id) pair. Orders by distance, then id, like std::pair in the C++ version. */
record Candidate(float dist, int id) implements Comparable<Candidate> {
    private static final Comparator<Candidate> ORDER =
        Comparator.comparingDouble(Candidate::dist).thenComparingInt(Candidate::id);

    @Override
    public int compareTo(Candidate o) {
        return ORDER.compare(this, o);
    }
}
