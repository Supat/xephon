import Foundation
import Fusion

/// One suspected mis-transcription of a user keyword: a span of an
/// utterance's transcript that is NOT a raw match of the keyword
/// but reads like one — same reading, different surface form
/// (homophone tier: 資料 transcribed as 飼料), or a near reading
/// within the fuzzy-search thresholds (fuzzy tier). Detected by
/// `KeywordReviewModel`, surfaced as a warning chip on the keyword
/// row, adjudicated in `KeywordReviewSheet`.
struct KeywordSuspect: Identifiable, Hashable, Sendable {
    let keywordID: UUID
    let utteranceID: UUID
    /// The suspect span's text as it appears in the transcript.
    let surface: String
    /// Transcript snapshot at detection time — context display and
    /// the staleness check at commit (a row edited since detection
    /// must not be blind-patched at stale offsets).
    let transcript: String
    /// UTF-16 offsets of the span within `transcript`.
    let rangeStartUTF16: Int
    let rangeEndUTF16: Int
    /// true = exact normalized-reading match (homophone class,
    /// high confidence); false = fuzzy near-reading match.
    let isHomophone: Bool

    /// Deterministic identity so List rows and per-row edit state
    /// survive memo rebuilds; doubles as the rejection key.
    var id: String {
        "\(keywordID.uuidString)|\(utteranceID.uuidString)|\(rangeStartUTF16)|\(surface)"
    }
}

/// Detection + rejection state for the keyword mis-transcription
/// review. Owned by ControlPaneView via @State (cheap init, no side
/// effects); the Keywords card reads per-keyword counts for its
/// warning chips and the review sheet reads the full suspect lists.
///
/// Matching reuses the fuzzy-search stack: readings come from
/// `JapaneseSearchNormalizer` (kanji↔kana↔romaji collapse to one
/// normalized space, so a homophone mis-transcription is an EXACT
/// normalized match that raw search missed), and the near-reading
/// tier uses `FuzzySubstringMatcher` with the same length-scaled
/// thresholds the transcript search's "Include similar" mode uses —
/// one notion of "sounds close" across the app.
///
/// Rejections are session-scoped by design: a rejected claim is the
/// user saying "this row is correct", and a *newly analyzed* session
/// deserves fresh eyes; persisting rejections into .xph would also
/// mean persisting stale offsets.
@MainActor
@Observable
final class KeywordReviewModel {
    /// Rejected suspect ids (`KeywordSuspect.id`). Observed so chip
    /// counts and the sheet update the instant a claim is rejected.
    private(set) var rejectedIDs: Set<String> = []

    /// The session UndoManager (the recorder's), assigned by
    /// ControlPaneView before the review sheet presents. Weak —
    /// this model must not keep the controller's manager alive,
    /// and a nil manager just means rejections aren't undoable
    /// (they still work).
    @ObservationIgnored weak var undoManager: UndoManager?

    /// Monotonic mutation counter for the memo key. A plain
    /// rejection COUNT collides across undo cycles — reject A,
    /// undo, reject B leaves the count at 1 twice with two
    /// different sets, and the memo would happily serve the
    /// {A}-generation result for the {B} state.
    @ObservationIgnored private var rejectionGeneration = 0

    @ObservationIgnored private var memoKey: MemoKey?
    @ObservationIgnored private var memoSuspects: [UUID: [KeywordSuspect]] = [:]

    private struct MemoKey: Equatable {
        let utterancesVersion: Int
        let utteranceCount: Int
        let keywordSignature: [String]
        let rejectionCount: Int
    }

    /// Cap per keyword so a pathological keyword (e.g. a particle)
    /// can't flood the sheet or the scan budget.
    private static let maxSuspectsPerKeyword = 20

    /// Reject a suspect, undoably. Same inverse-inside-the-closure
    /// pattern as RecordingController's step system: re-registering
    /// the opposite operation while an undo invocation is running
    /// makes UndoManager route it to the redo stack automatically,
    /// so one method yields the full undo/redo cycle. Undoing a
    /// rejection resurfaces the suspect (chip count and open sheet
    /// both update via observation).
    func reject(_ suspect: KeywordSuspect) {
        applyRejection(id: suspect.id, rejected: true)
        undoManager?.setActionName(String(localized: "undo.keywords.reject"))
    }

    private func applyRejection(id: String, rejected: Bool) {
        if rejected {
            rejectedIDs.insert(id)
        } else {
            rejectedIDs.remove(id)
        }
        rejectionGeneration += 1
        undoManager?.registerUndo(withTarget: self) { target in
            target.applyRejection(id: id, rejected: !rejected)
        }
    }

    /// keywordID → suspects, memoized on (utterances, keyword texts,
    /// rejections). Rebuilds tokenize every transcript once per
    /// generation — the same budget the keyword occurrence counter
    /// already pays.
    func suspects(
        in recorder: RecordingController
    ) -> [UUID: [KeywordSuspect]] {
        let keywords = recorder.keywords.keywords
        let key = MemoKey(
            utterancesVersion: recorder.utterancesVersion,
            utteranceCount: recorder.utterances.count,
            keywordSignature: keywords.map { "\($0.id.uuidString)|\($0.text)" },
            rejectionCount: rejectionGeneration
        )
        if memoKey == key { return memoSuspects }
        memoKey = key
        memoSuspects = Self.detect(
            keywords: keywords,
            utterances: recorder.utterances,
            rejectedIDs: rejectedIDs
        )
        return memoSuspects
    }

    func suspects(
        forKeyword keywordID: UUID,
        in recorder: RecordingController
    ) -> [KeywordSuspect] {
        suspects(in: recorder)[keywordID] ?? []
    }

    // MARK: - Detection

    private struct PreparedKeyword {
        let keyword: Keyword
        let normalized: String
        let phoneticKey: String
        let levThreshold: Int
        let fuzzyEligible: Bool
    }

    private static func detect(
        keywords: [Keyword],
        utterances: [UtteranceEstimate],
        rejectedIDs: Set<String>
    ) -> [UUID: [KeywordSuspect]] {
        // Prepare each keyword's normalized form + thresholds once.
        // Single Latin letters are excluded outright — the letter-
        // name search feature owns that space and one-character
        // readings match half the language.
        let prepared: [PreparedKeyword] = keywords.compactMap { kw in
            let trimmed = kw.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.count >= 2 else { return nil }
            let normalized = JapaneseSearchNormalizer.normalize(trimmed)
            guard normalized.count >= 2 else { return nil }
            let fuzzy = normalized.count >= SearchReplaceCoordinator.minQueryLengthForSimilar
            return PreparedKeyword(
                keyword: kw,
                normalized: normalized,
                phoneticKey: PhoneticKey.key(fromCanonicalRomaji: normalized),
                levThreshold: SearchReplaceCoordinator.similarMatchThreshold(for: normalized.count),
                fuzzyEligible: fuzzy
            )
        }
        guard !prepared.isEmpty else { return [:] }

        var result: [UUID: [KeywordSuspect]] = [:]
        for utterance in utterances {
            let transcript = utterance.transcript
            guard !transcript.isEmpty else { continue }
            // Tokenize once per utterance, shared across keywords.
            let tokens = JapaneseSearchNormalizer.tokens(transcript)
            guard !tokens.isEmpty else { continue }
            var starts: [Int] = []
            starts.reserveCapacity(tokens.count)
            var total = 0
            for t in tokens {
                starts.append(total)
                total += t.normalized.count
            }
            let joined = tokens.map(\.normalized).joined()

            for pk in prepared {
                if (result[pk.keyword.id]?.count ?? 0) >= maxSuspectsPerKeyword {
                    continue
                }
                // Raw (case-insensitive) matches are CORRECT usages,
                // not suspects; any candidate overlapping one is
                // excluded — same discipline as the search
                // highlighter.
                let rawRanges = SearchReplaceCoordinator.rawMatches(
                    in: transcript, term: pk.keyword.text
                )
                var claimed: [Range<String.Index>] = rawRanges

                func appendSuspect(range: Range<String.Index>, isHomophone: Bool) {
                    guard !claimed.contains(where: { $0.overlaps(range) }) else { return }
                    claimed.append(range)
                    let surface = String(transcript[range])
                    // A surface identical to the keyword slipped past
                    // the raw check only via case folding — not a
                    // mis-transcription.
                    guard surface.lowercased() != pk.keyword.text.lowercased() else { return }
                    let ns = NSRange(range, in: transcript)
                    let suspect = KeywordSuspect(
                        keywordID: pk.keyword.id,
                        utteranceID: utterance.id,
                        surface: surface,
                        transcript: transcript,
                        rangeStartUTF16: ns.location,
                        rangeEndUTF16: ns.location + ns.length,
                        isHomophone: isHomophone
                    )
                    guard !rejectedIDs.contains(suspect.id) else { return }
                    result[pk.keyword.id, default: []].append(suspect)
                }

                // Tier 1 — homophone: the keyword's exact reading
                // appears in normalized space but the surface didn't
                // raw-match. Map the normalized hit back through the
                // tokens' original ranges (chunk granularity at the
                // edges, same as the search highlighter).
                var searchStart = joined.startIndex
                while searchStart < joined.endIndex,
                      let hit = joined.range(of: pk.normalized, range: searchStart..<joined.endIndex) {
                    let s = joined.distance(from: joined.startIndex, to: hit.lowerBound)
                    let e = joined.distance(from: joined.startIndex, to: hit.upperBound)
                    searchStart = hit.upperBound
                    guard let firstTok = tokens.indices.last(where: { starts[$0] <= s }) else { continue }
                    var lastTok = firstTok
                    while lastTok + 1 < tokens.count,
                          starts[lastTok] + tokens[lastTok].normalized.count < e {
                        lastTok += 1
                    }
                    appendSuspect(
                        range: tokens[firstTok].originalRange.lowerBound
                            ..< tokens[lastTok].originalRange.upperBound,
                        isHomophone: true
                    )
                }

                // Tier 1.5 — phonetic-key equality over token runs:
                // catches the STRUCTURED ASR confusion classes
                // (long/short vowel, geminates, voicing, n/m) that
                // generic edit distance spends its whole budget on.
                // No length floor, so short keywords the fuzzy tier
                // excludes still get coverage. Exact key equality
                // only — keys are aggressive within their classes
                // and containment would compound the collapses.
                if !pk.phoneticKey.isEmpty {
                    for start in tokens.indices {
                        if (result[pk.keyword.id]?.count ?? 0) >= maxSuspectsPerKeyword { break }
                        var acc = ""
                        var end = start
                        while end < tokens.count {
                            acc += tokens[end].normalized
                            // A run whose canonical form already
                            // overshoots the keyword's by more than
                            // the collapse classes can absorb can't
                            // key-match; stop extending.
                            if acc.count > pk.normalized.count + 4 { break }
                            if PhoneticKey.key(fromCanonicalRomaji: acc) == pk.phoneticKey {
                                appendSuspect(
                                    range: tokens[start].originalRange.lowerBound
                                        ..< tokens[end].originalRange.upperBound,
                                    isHomophone: true
                                )
                                break
                            }
                            end += 1
                        }
                    }
                }

                // Tier 2 — Levenshtein near-reading over the JOINED
                // normalized text (semi-global matcher), so a near
                // miss spanning tokenizer chunk boundaries is
                // visible; the hit maps back to the covering token
                // run. The LCS tier is deliberately ABSENT here
                // (unlike the opt-in "Include similar" search):
                // shared 3-char romaji runs are ubiquitous and every
                // suspect demands user attention, so precision wins
                // (fuzzy-search audit R5).
                if pk.fuzzyEligible,
                   (result[pk.keyword.id]?.count ?? 0) < maxSuspectsPerKeyword,
                   let hit = FuzzySubstringMatcher.findSimilarSubstring(
                       query: pk.normalized, in: joined, threshold: pk.levThreshold
                   ),
                   let firstTok = tokens.indices.last(where: { starts[$0] <= hit.lowerBound }) {
                    var lastTok = firstTok
                    while lastTok + 1 < tokens.count,
                          starts[lastTok] + tokens[lastTok].normalized.count < hit.upperBound {
                        lastTok += 1
                    }
                    appendSuspect(
                        range: tokens[firstTok].originalRange.lowerBound
                            ..< tokens[lastTok].originalRange.upperBound,
                        isHomophone: false
                    )
                }
            }
        }
        return result
    }
}
