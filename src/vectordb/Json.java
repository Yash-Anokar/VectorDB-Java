package vectordb;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Small JSON reader/writer so the project needs no libraries.
 * parse() returns Map, List, String, Double, Boolean or null.
 * Unlike the C++ string-scanning version, this handles nesting and \\uXXXX escapes properly.
 */
final class Json {
    private Json() {}

    // ---------- writing ----------

    static String str(String s) {
        StringBuilder o = new StringBuilder(s.length() + 2).append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> o.append("\\\"");
                case '\\' -> o.append("\\\\");
                case '\n' -> o.append("\\n");
                case '\r' -> o.append("\\r");
                case '\t' -> o.append("\\t");
                default -> {
                    if (c < 0x20) o.append(String.format("\\u%04x", (int) c));
                    else o.append(c);
                }
            }
        }
        return o.append('"').toString();
    }

    /** Vector as a JSON array with 4 decimals. */
    static String vec(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.ROOT, "%.4f", v[i]));
        }
        return sb.append(']').toString();
    }

    static String num(float f, int decimals) {
        return String.format(Locale.ROOT, "%." + decimals + "f", f);
    }

    static String ints(int[] a) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(a[i]);
        }
        return sb.append(']').toString();
    }

    // ---------- reading ----------

    /** Returns null for empty or malformed input. */
    static Object parseOrNull(String s) {
        try {
            return parse(s);
        } catch (RuntimeException e) {
            return null;
        }
    }

    static Object parse(String s) {
        Reader r = new Reader(s);
        r.ws();
        Object v = r.value();
        r.ws();
        if (r.i != s.length()) throw new IllegalArgumentException("trailing data");
        return v;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> obj(Object o) {
        return o instanceof Map ? (Map<String, Object>) o : Map.of();
    }

    static String getStr(Object o, String key) {
        Object v = obj(o).get(key);
        return v instanceof String s ? s : "";
    }

    static int getInt(Object o, String key, int def) {
        Object v = obj(o).get(key);
        return v instanceof Double d ? d.intValue() : def;
    }

    /** Reads a JSON array of numbers; empty array if missing or malformed. */
    static float[] getFloats(Object o, String key) {
        Object v = obj(o).get(key);
        if (!(v instanceof List<?> list)) return new float[0];
        float[] out = new float[list.size()];
        for (int i = 0; i < out.length; i++) {
            if (!(list.get(i) instanceof Double d)) return new float[0];
            out[i] = d.floatValue();
        }
        return out;
    }

    private static final class Reader {
        final String s;
        int i = 0;

        Reader(String s) { this.s = s; }

        void ws() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        }

        char peek() {
            if (i >= s.length()) throw new IllegalArgumentException("unexpected end");
            return s.charAt(i);
        }

        void expect(char c) {
            if (peek() != c) throw new IllegalArgumentException("expected " + c + " at " + i);
            i++;
        }

        Object value() {
            ws();
            char c = peek();
            if (c == '{') return object();
            if (c == '[') return array();
            if (c == '"') return string();
            if (s.startsWith("true", i)) { i += 4; return Boolean.TRUE; }
            if (s.startsWith("false", i)) { i += 5; return Boolean.FALSE; }
            if (s.startsWith("null", i)) { i += 4; return null; }
            return number();
        }

        Map<String, Object> object() {
            Map<String, Object> m = new LinkedHashMap<>();
            expect('{');
            ws();
            if (peek() == '}') { i++; return m; }
            while (true) {
                ws();
                String k = string();
                ws();
                expect(':');
                m.put(k, value());
                ws();
                if (peek() == ',') { i++; continue; }
                expect('}');
                return m;
            }
        }

        List<Object> array() {
            List<Object> l = new ArrayList<>();
            expect('[');
            ws();
            if (peek() == ']') { i++; return l; }
            while (true) {
                l.add(value());
                ws();
                if (peek() == ',') { i++; continue; }
                expect(']');
                return l;
            }
        }

        String string() {
            expect('"');
            StringBuilder sb = new StringBuilder();
            while (true) {
                char c = peek();
                i++;
                if (c == '"') return sb.toString();
                if (c != '\\') { sb.append(c); continue; }
                char e = peek();
                i++;
                switch (e) {
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> {
                        sb.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                    }
                    default -> sb.append(e); // \" \\ \/
                }
            }
        }

        Double number() {
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            if (start == i) throw new IllegalArgumentException("bad value at " + i);
            return Double.valueOf(s.substring(start, i));
        }
    }
}
