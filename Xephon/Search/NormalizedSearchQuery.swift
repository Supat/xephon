import Foundation

/// A search query reduced to the form the matcher actually compares
/// against. Two mutually-exclusive modes:
///
/// - **Substring** — ordinary multi-character queries. Matches when
///   the query's normalized (Hepburn-romaji) form appears anywhere in
///   the transcript's normalized form. This is the historical
///   cross-script behavior.
///
/// - **Boundary forms** — single Latin-letter queries, expanded via
///   `LatinLetterReadings` into the spoken letter-name reading(s) plus
///   the bare letter. Each is matched as a *token-boundary-aligned
///   run*: it matches only when some contiguous run of transcript
///   tokens joins to exactly that normalized form. This is what makes
///   "G" surface a standalone ジー / じー or a literal "G" without
///   firing on every romaji "g" buried inside an unrelated word (が,
///   ジーパン, Google).
///
/// Boundary-aligned run matching (rather than comparing fixed token
/// *sequences*) is deliberate: `CFStringTokenizer` splits the same
/// reading differently depending on its neighbours — エイチ is one
/// "eichi" token in a sentence but ["ei","chi"] in isolation. Joining
/// a run before comparing is invariant to that internal re-splitting,
/// so it matches in both cases. It still can't see through a reading
/// the tokenizer fuses into a larger lexical unit (ジーパン, the glued
/// NHK acronym エヌエイチケー, or a hiragana letter-name that bonds to a
/// preceding kana as in サイズはじー); those stay excluded, which is the
/// intended "whole-token" tightening.
///
/// Both products derive from the same `JapaneseSearchNormalizer` pass,
/// so a caller that has cached a transcript's normalized string and
/// its ordered token list can serve either mode from the cache.
struct NormalizedSearchQuery: Sendable, Hashable {
    /// Normalized substring to find anywhere. Empty in boundary-form
    /// mode and for empty input.
    let substring: String
    /// Normalized forms matched at token boundaries; the transcript
    /// matches when ANY of them equals the join of some contiguous
    /// run of its tokens. Empty in substring mode.
    let boundaryForms: [String]

    var isEmpty: Bool { substring.isEmpty && boundaryForms.isEmpty }

    /// Build from raw user input. Trims, then routes a single Latin
    /// letter into boundary-form mode (reading expansion) and
    /// everything else into substring mode.
    static func build(from raw: String) -> NormalizedSearchQuery {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return NormalizedSearchQuery(substring: "", boundaryForms: [])
        }
        if let letter = LatinLetterReadings.singleLetter(in: trimmed) {
            return NormalizedSearchQuery(
                substring: "",
                boundaryForms: boundaryForms(for: letter)
            )
        }
        return NormalizedSearchQuery(
            substring: JapaneseSearchNormalizer.normalize(trimmed),
            boundaryForms: []
        )
    }

    /// Normalized boundary forms a single letter expands to: the bare
    /// letter (so a literal isolated "G"/"g" still surfaces) plus the
    /// spoken katakana and hiragana readings. Deduped; empties
    /// dropped. Exposed so the find-and-replace surface can reuse the
    /// reading forms while excluding the bare letter (which it
    /// handles via its own raw-substring pass).
    static func boundaryForms(for letter: Character) -> [String] {
        var forms: [String] = []
        func add(_ form: String) {
            guard !form.isEmpty, !forms.contains(form) else { return }
            forms.append(form)
        }
        add(JapaneseSearchNormalizer.normalize(String(letter)))
        for reading in LatinLetterReadings.readingForms(for: letter) {
            add(JapaneseSearchNormalizer.normalize(reading))
        }
        return forms
    }

    /// Reading-only boundary forms (the bare letter excluded). Used by
    /// find-and-replace, where a literal letter is matched and
    /// replaced through the raw-substring path and only the spoken
    /// readings need the surface-but-don't-replace treatment.
    static func readingBoundaryForms(for letter: Character) -> [String] {
        var forms: [String] = []
        for reading in LatinLetterReadings.readingForms(for: letter) {
            let n = JapaneseSearchNormalizer.normalize(reading)
            if !n.isEmpty, !forms.contains(n) { forms.append(n) }
        }
        return forms
    }

    /// Match against a transcript supplied as both its normalized
    /// substring form and its ordered per-token normalized forms
    /// (both from one `JapaneseSearchNormalizer` pass, so callers
    /// cache them together).
    func matches(normalized: String, tokens: [String]) -> Bool {
        if !substring.isEmpty {
            return normalized.contains(substring)
        }
        for form in boundaryForms where Self.containsTokenAlignedRun(tokens, form) {
            return true
        }
        return false
    }

    /// True when some contiguous run of `tokens` joins to exactly
    /// `form`. Empty `form` never matches. Stops extending a run once
    /// its join can no longer equal `form` (length only grows), so the
    /// inner loop is bounded by the run length, not the token count.
    static func containsTokenAlignedRun(_ tokens: [String], _ form: String) -> Bool {
        guard !form.isEmpty else { return false }
        let targetCount = form.unicodeScalars.count
        for start in tokens.indices {
            var acc = ""
            var end = start
            while end < tokens.count {
                acc += tokens[end]
                if acc == form { return true }
                if acc.unicodeScalars.count >= targetCount { break }
                end += 1
            }
        }
        return false
    }
}
