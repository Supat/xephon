import Foundation

/// R3 of the fuzzy-search audit: a phonetic equivalence key over the
/// canonical romaji space, targeting the CONFUSION CLASSES Japanese
/// ASR actually produces — long↔short vowels (東京 toukyou / tokyo),
/// geminate presence (kitte / kite), voiced↔unvoiced onsets
/// (k/g, s/z, t/d, h/b/p), and n/m assimilation. Two strings share a
/// key iff they differ only by those classes, so EXACT key equality
/// is a high-precision "sounds the same-ish" tier sitting between
/// exact-normalized matching and generic Levenshtein — it catches
/// systematic ASR variance without spending the edit budget that
/// generic fuzziness needs (and without its false positives), and it
/// has no minimum-length floor, recovering the short words the
/// Levenshtein tier excludes.
///
/// The key is aggressive BY DESIGN within its classes and must only
/// ever be compared for equality — substring/containment over keys
/// would compound the collapses into noise.
enum PhoneticKey {
    /// Key for arbitrary text (normalizes first).
    static func key(for text: String) -> String {
        key(fromCanonicalRomaji: JapaneseSearchNormalizer.normalize(text))
    }

    /// Key for text already in the normalizer's canonical romaji.
    static func key(fromCanonicalRomaji s: String) -> String {
        guard !s.isEmpty else { return "" }
        // 1. Collapse Hepburn digraphs onto base consonants so the
        //    devoicing step sees single letters (sh→s, ch/ts→t).
        var t = s
        for (from, to) in [("sh", "s"), ("ch", "t"), ("ts", "t")] where t.contains(from) {
            t = t.replacingOccurrences(of: from, with: to)
        }
        // 2. Per-character class fold: devoice (g→k, z→s, d→t,
        //    b/p→h), j onto s (voiced sh), f onto h (ふ row),
        //    m onto n (assimilation). Non-letters drop.
        var mapped = ""
        mapped.reserveCapacity(t.count)
        for c in t {
            switch c {
            case "g": mapped.append("k")
            case "z", "j": mapped.append("s")
            case "d": mapped.append("t")
            case "b", "p", "f": mapped.append("h")
            case "m": mapped.append("n")
            case "a"..."z": mapped.append(c)
            default: break
            }
        }
        // 3. Length collapse: doubled characters (long vowels AND
        //    geminates) fold to one; the kana long-vowel spellings
        //    ou → o and ei → e fold with them.
        var out = ""
        out.reserveCapacity(mapped.count)
        for c in mapped {
            if let last = out.last {
                if c == last { continue }
                if last == "o" && c == "u" { continue }
                if last == "e" && c == "i" { continue }
            }
            out.append(c)
        }
        return out
    }
}
