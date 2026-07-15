import Foundation
import Fusion
import XephonPluginKit

/// Pure extraction machinery: candidate-row selection, the
/// deterministic stated-score pass, per-item prompt/schema
/// construction, lenient response parsing, and the merge policy.
/// Everything here is synchronous and value-in/value-out so the
/// whole pipeline is unit-testable without a model; the async
/// generate loop lives in `EvalFormModel`.
public enum EvalFormExtractor {

    // MARK: - Candidate rows

    /// 1-based session-wide row numbers whose transcript mentions
    /// any of the item's vocabulary surfaces. Numbering is global
    /// (not per-item) so evidence rows resolve against the session
    /// regardless of which item cited them.
    public static func candidateRowNumbers(
        for item: EvalFormTemplate.Item,
        utterances: [UtteranceEstimate]
    ) -> [Int] {
        let needles = item.vocabulary.map {
            $0.precomposedStringWithCompatibilityMapping.lowercased()
        }
        return utterances.enumerated().compactMap { idx, u in
            let hay = u.transcript
                .precomposedStringWithCompatibilityMapping
                .lowercased()
            return needles.contains(where: { hay.contains($0) }) ? idx + 1 : nil
        }
    }

    /// The item's mention rows expanded with ±`window` context
    /// rows. ASR splits Japanese judgments across segments — the
    /// onomatopoeia lands in one row and the verdict (often the
    /// SPOKEN SCORE) in the next — so both the deterministic pass
    /// and the LLM prompt need the neighbourhood, not just the
    /// mention.
    ///
    /// Attribution guard: a context row is attached to the item
    /// whose mention is NEAREST (ties attach to every tied item —
    /// the merge conflict machinery covers the rare ambiguity).
    /// Another item's own mention row sits at distance 0 of that
    /// item and therefore can never be captured as context here,
    /// which is what keeps a ±2 window from cross-attributing
    /// neighbouring items' remarks.
    public static func contextExpandedRows(
        for item: EvalFormTemplate.Item,
        template: EvalFormTemplate,
        utterances: [UtteranceEstimate],
        window: Int = 2
    ) -> [Int] {
        let allCores = template.items.map {
            ($0.id, candidateRowNumbers(for: $0, utterances: utterances))
        }
        guard let mine = allCores.first(where: { $0.0 == item.id })?.1,
              !mine.isEmpty
        else { return [] }
        let otherCores = allCores
            .filter { $0.0 != item.id }
            .map(\.1)
            .filter { !$0.isEmpty }

        var rows = Set(mine)
        for core in mine {
            let lo = max(1, core - window)
            let hi = min(utterances.count, core + window)
            guard lo <= hi else { continue }
            for neighbor in lo...hi where !rows.contains(neighbor) {
                let myDistance = mine.map { abs($0 - neighbor) }.min() ?? .max
                let otherDistance = otherCores
                    .compactMap { cores in cores.map { abs($0 - neighbor) }.min() }
                    .min()
                if let otherDistance, otherDistance < myDistance { continue }
                rows.insert(neighbor)
            }
        }
        return rows.sorted()
    }

    // MARK: - Road sections

    /// Segment the session by road callouts: a row matching any of
    /// a road's callout surfaces opens that road's segment, which
    /// runs until the row before the next different road's callout
    /// (or the session end). Surfaces come from the template's
    /// `effectiveRoadCallouts` — explicit per-road aliases when the
    /// pack defines them ("E3" for E3路), else the exact derived
    /// labels. Different surfaces of the SAME road are one road:
    /// consecutive re-mentions don't open a new segment. Repeated
    /// visits get numbered titles ("F路 (2)") so proposals stay
    /// unique. Rows before the first callout belong to no road.
    public static func roadSectionProposals(
        utterances: [UtteranceEstimate],
        callouts roadCallouts: [EvalFormTemplate.RoadCallout]
    ) -> [PluginSectionProposal] {
        // Fold + lowercase both sides so full-width/Latin-case
        // ASR variance can't break a match.
        let lexicon: [(road: String, surfaces: [String])] = roadCallouts
            .map { callout in
                (
                    road: callout.road,
                    surfaces: callout.surfaces
                        .map {
                            $0.precomposedStringWithCompatibilityMapping
                                .lowercased()
                        }
                        .filter { !$0.isEmpty }
                )
            }
            .filter { !$0.surfaces.isEmpty }
        guard !utterances.isEmpty, !lexicon.isEmpty else { return [] }
        // (start index, road) per callout, chronological.
        var callouts: [(index: Int, road: String)] = []
        for (index, u) in utterances.enumerated() {
            let hay = u.transcript
                .precomposedStringWithCompatibilityMapping
                .lowercased()
            if let road = lexicon.first(where: { entry in
                entry.surfaces.contains { calloutSurfaceMatches($0, in: hay) }
            })?.road {
                // Consecutive rows re-mentioning the CURRENT road
                // don't open a new segment.
                if callouts.last?.road != road {
                    callouts.append((index, road))
                }
            }
        }
        guard !callouts.isEmpty else { return [] }
        var visits: [String: Int] = [:]
        return numberedProposals(callouts: callouts, utterances: utterances, visits: &visits)
    }

    /// Substring match, except single Latin letters must stand
    /// alone: neither neighbour may be a Latin alphanumeric, so a
    /// bare "D" surface matches "Dに入ります" but never 4WD / HD.
    /// Both sides arrive folded + lowercased.
    static func calloutSurfaceMatches(_ surface: String, in hay: String) -> Bool {
        guard surface.count == 1,
              let letter = surface.first,
              letter.isASCII, letter.isLetter
        else { return hay.contains(surface) }
        var searchStart = hay.startIndex
        while let found = hay.range(of: surface, range: searchStart..<hay.endIndex) {
            let beforeOK = found.lowerBound == hay.startIndex
                || !isLatinAlphanumeric(hay[hay.index(before: found.lowerBound)])
            let afterOK = found.upperBound == hay.endIndex
                || !isLatinAlphanumeric(hay[found.upperBound])
            if beforeOK && afterOK { return true }
            searchStart = found.upperBound
        }
        return false
    }

    private static func isLatinAlphanumeric(_ c: Character) -> Bool {
        c.isASCII && (c.isLetter || c.isNumber)
    }

    private static func numberedProposals(
        callouts: [(index: Int, road: String)],
        utterances: [UtteranceEstimate],
        visits: inout [String: Int]
    ) -> [PluginSectionProposal] {
        return callouts.enumerated().map { calloutIdx, callout in
            let visit = (visits[callout.road] ?? 0) + 1
            visits[callout.road] = visit
            let title = visit == 1 ? callout.road : "\(callout.road) (\(visit))"
            let endIndex = calloutIdx + 1 < callouts.count
                ? callouts[calloutIdx + 1].index - 1
                : utterances.count - 1
            return PluginSectionProposal(
                title: title,
                startUtteranceID: utterances[callout.index].id,
                endUtteranceID: utterances[max(callout.index, endIndex)].id
            )
        }
    }

    // MARK: - Deterministic pass

    public struct DeterministicFindings: Equatable, Sendable {
        /// (row number, quantized value), chronological.
        public var statedScores: [(row: Int, value: Double)]
        public var statedPreference: (row: Int, value: Int)?

        public static func == (l: Self, r: Self) -> Bool {
            l.statedScores.elementsEqual(r.statedScores, by: ==)
                && l.statedPreference?.row == r.statedPreference?.row
                && l.statedPreference?.value == r.statedPreference?.value
        }
    }

    /// Run the spoken-score grammar over the item's candidate rows.
    public static func deterministicFindings(
        candidateRows: [Int],
        utterances: [UtteranceEstimate],
        template: EvalFormTemplate
    ) -> DeterministicFindings {
        var scores: [(row: Int, value: Double)] = []
        var preference: (row: Int, value: Int)?
        let allowed = template.strengthScale.allowedValues
        for row in candidateRows {
            guard row >= 1, row <= utterances.count else { continue }
            let text = utterances[row - 1].transcript
            for hit in SpokenScoreParser.statedStrengths(in: text, allowedValues: allowed) {
                scores.append((row: row, value: hit.value))
            }
            if preference == nil,
               let p = SpokenScoreParser.statedPreference(
                   in: text,
                   minimum: template.preferenceScale.minimum,
                   maximum: template.preferenceScale.maximum
               ) {
                preference = (row: row, value: p)
            }
        }
        return DeterministicFindings(
            statedScores: scores,
            statedPreference: preference
        )
    }

    // MARK: - Prompt + schema

    /// JSON Schema for the per-item extraction response. Nullable
    /// everywhere — emptiness must be cheap or the model fills
    /// fields to please (research doc §3).
    public static let itemSchemaJSON = """
    {"type":"object","properties":{
      "statedScore":{"type":["number","null"],"description":"relative strength score the evaluator explicitly said, verbatim; null unless spoken"},
      "inferredScore":{"type":["number","null"],"description":"score suggested from qualitative wording only; null when statedScore present or nothing inferable"},
      "likeDislike":{"type":["integer","null"],"description":"1-9 preference the evaluator explicitly said; null unless spoken"},
      "comment":{"type":["string","null"],"description":"one-to-two sentence Japanese distillation of what was said about this item; null when not discussed"},
      "evidenceRows":{"type":"array","items":{"type":"integer"},"description":"row numbers supporting the fields above"}
    },"required":["statedScore","inferredScore","likeDislike","comment","evidenceRows"]}
    """

    /// Build the per-item extraction prompt over the candidate
    /// rows. Row lines carry the SESSION-global number so evidence
    /// resolves session-wide. The deterministic findings are given
    /// to the model as ground truth it must not contradict.
    public static func extractionPrompt(
        item: EvalFormTemplate.Item,
        template: EvalFormTemplate,
        rows: [(number: Int, speakerID: String, transcript: String)],
        deterministic: DeterministicFindings
    ) -> String {
        var lines: [String] = []
        lines.append("You are filling ONE row of a Japanese vehicle ride-quality evaluation sheet from test-drive utterances.")
        lines.append("Sheet: \(template.name)")
        lines.append("Item \(item.number): \(item.titleJa) — \(item.definition)")
        lines.append("Strength scale: \(template.strengthScale.minimum) (strong) to +\(template.strengthScale.maximum) (weak) relative to the baseline spec, in steps of \(template.strengthScale.step). Preference scale: \(template.preferenceScale.minimum) (嫌い) to \(template.preferenceScale.maximum) (好き).")
        lines.append("POLARITY: negative = the sensation is STRONGER than the baseline (強い・増えた side); positive = WEAKER (弱い・減った・なくなった side). The score's sign MUST agree with the direction the evaluator described.")
        if let anchors = template.strengthScale.anchors, !anchors.isEmpty {
            let rubric = anchors
                .map { "±\(String(format: "%g", $0.magnitude)) = \($0.meaning)" }
                .joined(separator: "; ")
            lines.append("Magnitude rubric (the sheet's own calibration — apply it when choosing inferredScore): \(rubric).")
        }
        lines.append("STRICT RULES:")
        lines.append("- statedScore / likeDislike: ONLY values the evaluator explicitly SAID. Never convert qualitative wording into a stated value. When nothing was said, use null.")
        lines.append("- inferredScore: your suggestion from qualitative wording, only when statedScore is null and the wording clearly implies a direction; must be one of the scale's quantized steps; otherwise null.")
        lines.append("- comment: short Japanese distillation of what was actually said about THIS item; null when the rows don't discuss it.")
        lines.append("- evidenceRows: the [n] numbers of the rows each filled field rests on. Only numbers that appear below. A claim without a row does not belong on the sheet.")
        if !deterministic.statedScores.isEmpty {
            let stated = deterministic.statedScores
                .map { "[\($0.row)] → \($0.value)" }
                .joined(separator: ", ")
            lines.append("Pre-verified stated scores (regex-captured; treat as ground truth): \(stated)")
        }
        lines.append("")
        lines.append("Utterances (numbered [n] speaker: text; the rows mentioning the item plus their immediate neighbours — a verdict or score often lands in the row AFTER the mention):")
        for row in rows {
            lines.append("[\(row.number)] \(row.speakerID): \(row.transcript)")
        }
        lines.append("")
        lines.append("Return ONLY the JSON object. The FIRST character of your output MUST be `{`.")
        return lines.joined(separator: "\n")
    }

    // MARK: - Response parsing

    public struct ItemWire: Equatable, Sendable {
        public var statedScore: Double?
        public var inferredScore: Double?
        public var likeDislike: Int?
        public var comment: String?
        public var evidenceRows: [Int]?

        public init(
            statedScore: Double?,
            inferredScore: Double?,
            likeDislike: Int?,
            comment: String?,
            evidenceRows: [Int]?
        ) {
            self.statedScore = statedScore
            self.inferredScore = inferredScore
            self.likeDislike = likeDislike
            self.comment = comment
            self.evidenceRows = evidenceRows
        }
    }

    /// Lenient parse: strip code fences / think blocks by slicing
    /// from the first `{`, strict-decode the widest `{…}` window,
    /// and fall back to truncation repair (close an unterminated
    /// string, drop a dangling comma/colon, balance brackets) —
    /// small quantized models routinely type numbers as strings
    /// and run past the output cap mid-string. Nil on anything
    /// unusable — the caller records an extraction failure rather
    /// than guessing.
    public static func parseItemResponse(_ raw: String) -> ItemWire? {
        parseJSONObject(raw)
    }

    /// The lenient-slice + repair decode shared by every wire shape.
    static func parseJSONObject<T: Decodable>(_ raw: String) -> T? {
        guard let start = raw.firstIndex(of: "{") else { return nil }
        let decoder = JSONDecoder()
        // Widest well-formed window first.
        if let end = raw.lastIndex(of: "}"), end >= start,
           let ok = try? decoder.decode(
               T.self, from: Data(String(raw[start...end]).utf8)
           ) {
            return ok
        }
        // Truncation repair over the whole tail.
        if let repaired = repairTruncatedJSON(String(raw[start...])),
           let ok = try? decoder.decode(T.self, from: Data(repaired.utf8)) {
            return ok
        }
        return nil
    }

    /// Close a JSON object cut off mid-generation: terminate an
    /// open string, drop a dangling comma or `key:` tail, then
    /// close open brackets in reverse order. Returns nil when the
    /// input is already balanced (repair can't help) or brackets
    /// mismatch (garbage, not truncation).
    static func repairTruncatedJSON(_ s: String) -> String? {
        var closers: [Character] = []
        var inString = false
        var escaped = false
        for ch in s {
            if inString {
                if escaped { escaped = false }
                else if ch == "\\" { escaped = true }
                else if ch == "\"" { inString = false }
            } else {
                switch ch {
                case "\"": inString = true
                case "{": closers.append("}")
                case "[": closers.append("]")
                case "}", "]":
                    guard closers.last == ch else { return nil }
                    closers.removeLast()
                default: break
                }
            }
        }
        guard inString || !closers.isEmpty else { return nil }
        var repaired = s
        if inString { repaired += "\"" }
        while let last = repaired.last, last.isWhitespace {
            repaired.removeLast()
        }
        if repaired.hasSuffix(",") {
            repaired.removeLast()
        } else if repaired.hasSuffix(":") {
            repaired += "null"
        }
        repaired += String(closers.reversed())
        return repaired
    }

    // MARK: Lenient scalar decoding

    /// "-0.25" / "−0.25" / 7 / "null" tolerance — quantized models
    /// under a prompt contract frequently type numbers as strings.
    static func looseNumber(_ s: String) -> Double? {
        let folded = s.precomposedStringWithCompatibilityMapping
            .replacingOccurrences(of: "−", with: "-")
            .replacingOccurrences(of: "＋", with: "+")
            .trimmingCharacters(in: .whitespaces)
        guard !folded.isEmpty, folded.lowercased() != "null" else { return nil }
        return Double(folded)
    }

    fileprivate static func lenientDouble<K: CodingKey>(
        _ c: KeyedDecodingContainer<K>, _ key: K
    ) -> Double? {
        if let d = try? c.decodeIfPresent(Double.self, forKey: key) { return d }
        if let s = try? c.decodeIfPresent(String.self, forKey: key) {
            return looseNumber(s)
        }
        return nil
    }

    fileprivate static func lenientInt<K: CodingKey>(
        _ c: KeyedDecodingContainer<K>, _ key: K
    ) -> Int? {
        if let i = try? c.decodeIfPresent(Int.self, forKey: key) { return i }
        return lenientDouble(c, key).flatMap {
            $0.truncatingRemainder(dividingBy: 1) == 0 ? Int($0) : nil
        }
    }

    fileprivate static func lenientIntArray<K: CodingKey>(
        _ c: KeyedDecodingContainer<K>, _ key: K
    ) -> [Int]? {
        if let ints = try? c.decodeIfPresent([Int].self, forKey: key) { return ints }
        if let doubles = try? c.decodeIfPresent([Double].self, forKey: key) {
            return doubles.map { Int($0) }
        }
        if let strings = try? c.decodeIfPresent([String].self, forKey: key) {
            return strings.compactMap { looseNumber($0).map { Int($0) } }
        }
        return nil
    }

    // MARK: - Supplementary comment (補足コメント)

    /// Rows for the 補足コメント pass: substantive utterances no
    /// item's vocabulary claimed. "Substantive" is a cheap length
    /// floor — backchannels (うん, そう) carry nothing a sheet
    /// comment needs. Chronological, capped.
    public static func supplementaryCandidateRows(
        utterances: [UtteranceEstimate],
        claimedRows: Set<Int>,
        limit: Int = 40
    ) -> [Int] {
        var rows: [Int] = []
        for (idx, u) in utterances.enumerated() {
            let row = idx + 1
            guard !claimedRows.contains(row) else { continue }
            let trimmed = u.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.count >= 8 else { continue }
            rows.append(row)
            if rows.count >= limit { break }
        }
        return rows
    }

    public static let supplementarySchemaJSON = """
    {"type":"object","properties":{
      "comment":{"type":["string","null"],"description":"two-to-four short Japanese sentences of observations not covered by the sheet items; null when nothing substantive remains"},
      "evidenceRows":{"type":"array","items":{"type":"integer"}}
    },"required":["comment","evidenceRows"]}
    """

    public struct SupplementaryWire: Equatable, Sendable {
        public var comment: String?
        public var evidenceRows: [Int]?

        public init(comment: String?, evidenceRows: [Int]?) {
            self.comment = comment
            self.evidenceRows = evidenceRows
        }
    }

    public static func supplementaryPrompt(
        template: EvalFormTemplate,
        rows: [(number: Int, speakerID: String, transcript: String)]
    ) -> String {
        var lines: [String] = []
        lines.append("You are writing the 補足コメント (supplementary comment) block of a Japanese vehicle ride-quality evaluation sheet.")
        lines.append("Sheet: \(template.name). The per-item rows are handled separately — below are ONLY the utterances none of the sheet items claimed.")
        lines.append("Distill the observations worth recording (vehicle behaviours, test conditions, caveats) into two to four short Japanese sentences. Ignore small talk. When nothing is worth recording, use null.")
        lines.append("evidenceRows: the [n] numbers the comment rests on; only numbers that appear below.")
        lines.append("")
        lines.append("Utterances (numbered [n] speaker: text):")
        for row in rows {
            lines.append("[\(row.number)] \(row.speakerID): \(row.transcript)")
        }
        lines.append("")
        lines.append("Return ONLY the JSON object. The FIRST character of your output MUST be `{`.")
        return lines.joined(separator: "\n")
    }

    public static func parseSupplementaryResponse(_ raw: String) -> SupplementaryWire? {
        parseJSONObject(raw)
    }
}

// MARK: - Lenient Codable conformances

extension EvalFormExtractor.ItemWire: Codable {
    private enum CodingKeys: String, CodingKey {
        case statedScore, inferredScore, likeDislike, comment, evidenceRows
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        statedScore = EvalFormExtractor.lenientDouble(c, .statedScore)
        inferredScore = EvalFormExtractor.lenientDouble(c, .inferredScore)
        likeDislike = EvalFormExtractor.lenientInt(c, .likeDislike)
        comment = try? c.decodeIfPresent(String.self, forKey: .comment)
        evidenceRows = EvalFormExtractor.lenientIntArray(c, .evidenceRows)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(statedScore, forKey: .statedScore)
        try c.encodeIfPresent(inferredScore, forKey: .inferredScore)
        try c.encodeIfPresent(likeDislike, forKey: .likeDislike)
        try c.encodeIfPresent(comment, forKey: .comment)
        try c.encodeIfPresent(evidenceRows, forKey: .evidenceRows)
    }
}

extension EvalFormExtractor.SupplementaryWire: Codable {
    private enum CodingKeys: String, CodingKey {
        case comment, evidenceRows
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        comment = try? c.decodeIfPresent(String.self, forKey: .comment)
        evidenceRows = EvalFormExtractor.lenientIntArray(c, .evidenceRows)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(comment, forKey: .comment)
        try c.encodeIfPresent(evidenceRows, forKey: .evidenceRows)
    }
}

extension EvalFormExtractor {
    // MARK: - Metadata candidates

    /// Rows for the header-metadata pass: the session opening plus
    /// any row mentioning a metadata cue (weather changes mid-
    /// drive, specs restated at an absorber swap). Sorted, capped.
    public static func metadataCandidateRows(
        utterances: [UtteranceEstimate],
        cues: [String]?,
        openingCount: Int = 20,
        limit: Int = 30
    ) -> [Int] {
        var rows = Set(1...min(openingCount, max(utterances.count, 1)))
        if let cues, !cues.isEmpty {
            let needles = cues.map {
                $0.precomposedStringWithCompatibilityMapping.lowercased()
            }
            for (idx, u) in utterances.enumerated() {
                let hay = u.transcript
                    .precomposedStringWithCompatibilityMapping
                    .lowercased()
                if needles.contains(where: { hay.contains($0) }) {
                    rows.insert(idx + 1)
                }
            }
        }
        return rows.filter { $0 <= utterances.count }.sorted().prefix(limit).map { $0 }
    }

    // MARK: - Merge policy

    /// Direction words for the polarity sanity check. On the
    /// sheet, negative = stronger than baseline — an inferred
    /// score whose sign contradicts the evidence rows' direction
    /// words gets a review flag (never a silent sign flip; the
    /// heuristic can't see negation like 強くない).
    private static let strongerWords = ["強", "増え", "増加", "大きく", "悪化"]
    private static let weakerWords = ["弱", "減っ", "減り", "少なく", "なくなっ", "小さく", "改善"]

    /// Combine the deterministic findings with the model's wire
    /// output. Deterministic wins every disagreement (it cannot
    /// hallucinate); disagreements, multi-value captures, and
    /// polarity contradictions become conflict notes instead of
    /// silent picks. `transcriptForRow` feeds the polarity check
    /// with the cited rows' text; the default disables the check
    /// (callers without row access lose only the flag).
    public static func merge(
        item: EvalFormTemplate.Item,
        template: EvalFormTemplate,
        deterministic: DeterministicFindings,
        wire: ItemWire?,
        validRowNumbers: Set<Int>,
        transcriptForRow: (Int) -> String? = { _ in nil }
    ) -> EvalFormDraft.ItemResult {
        var result = EvalFormDraft.ItemResult(itemID: item.id)
        var evidence = Set<Int>()
        let allowed = template.strengthScale.allowedValues

        // Stated strength: deterministic first.
        let distinctStated = Array(Set(deterministic.statedScores.map(\.value)))
        if distinctStated.count == 1 {
            result.strengthScore = distinctStated[0]
            evidence.formUnion(deterministic.statedScores.map(\.row))
        } else if distinctStated.count > 1 {
            // Evaluators revise; keep the chronologically LAST and
            // flag the revision trail for review.
            result.strengthScore = deterministic.statedScores.last?.value
            evidence.formUnion(deterministic.statedScores.map(\.row))
            let trail = deterministic.statedScores
                .map { "\($0.value)@[\($0.row)]" }
                .joined(separator: " → ")
            result.conflicts.append("複数の発話スコア: \(trail)（最後の値を採用）")
        }

        if let wire {
            // Model-extracted stated score is accepted only when
            // the deterministic pass found nothing AND the value is
            // a legal scale step.
            if result.strengthScore == nil,
               let modelStated = wire.statedScore,
               allowed.contains(where: { abs($0 - modelStated) < 0.0005 }) {
                result.strengthScore = modelStated
            } else if let modelStated = wire.statedScore,
                      let deterministicStated = result.strengthScore,
                      abs(modelStated - deterministicStated) > 0.0005 {
                result.conflicts.append(
                    "モデル抽出スコア \(modelStated) が発話キャプチャ \(deterministicStated) と不一致（後者を採用）"
                )
            }
            // Inferred: only meaningful when nothing was stated,
            // and only on a legal step.
            if result.strengthScore == nil,
               let inferred = wire.inferredScore,
               allowed.contains(where: { abs($0 - inferred) < 0.0005 }) {
                result.strengthScoreInferred = inferred
            }
            result.comment = wire.comment?.isEmpty == true ? nil : wire.comment
            evidence.formUnion(
                (wire.evidenceRows ?? []).filter { validRowNumbers.contains($0) }
            )
        }

        // Preference: deterministic wins; wire fills the gap when
        // in range.
        if let p = deterministic.statedPreference {
            result.likeDislike = p.value
            evidence.insert(p.row)
        } else if let wp = wire?.likeDislike,
                  (template.preferenceScale.minimum...template.preferenceScale.maximum)
                      .contains(wp) {
            result.likeDislike = wp
        }

        result.evidenceRows = evidence.sorted()

        // Polarity sanity check on the INFERRED score only — a
        // stated score is the evaluator's own words. Compare the
        // sign against the direction words in the cited rows; a
        // clear contradiction (opposing hits, zero supporting)
        // becomes a review flag.
        if let inferred = result.strengthScoreInferred, inferred != 0 {
            var strongerHits = 0
            var weakerHits = 0
            for row in result.evidenceRows {
                guard let text = transcriptForRow(row) else { continue }
                strongerHits += Self.strongerWords.filter { text.contains($0) }.count
                weakerHits += Self.weakerWords.filter { text.contains($0) }.count
            }
            // inferred > 0 = weaker-than-baseline side.
            let contradicted = inferred > 0
                ? (strongerHits > 0 && weakerHits == 0)
                : (weakerHits > 0 && strongerHits == 0)
            if contradicted {
                let direction = inferred > 0 ? "「強い」" : "「弱い・減少」"
                result.conflicts.append(
                    "推定スコアの極性要確認（根拠発話は\(direction)方向、スコアは逆側）"
                )
            }
        }
        return result
    }
}
