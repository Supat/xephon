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

    // MARK: - Road sections

    /// Segment the session by road callouts: a row mentioning a
    /// road name opens that road's segment, which runs until the
    /// row before the next callout (or the session end). Repeated
    /// visits to the same road get numbered titles ("F路 (2)") so
    /// proposals stay unique. Rows before the first callout belong
    /// to no road.
    public static func roadSectionProposals(
        utterances: [UtteranceEstimate],
        roadNames: [String]
    ) -> [PluginSectionProposal] {
        guard !utterances.isEmpty, !roadNames.isEmpty else { return [] }
        // (start index, road) per callout, chronological.
        var callouts: [(index: Int, road: String)] = []
        for (index, u) in utterances.enumerated() {
            let hay = u.transcript.precomposedStringWithCompatibilityMapping
            if let road = roadNames.first(where: { hay.contains($0) }) {
                // Consecutive rows re-mentioning the CURRENT road
                // don't open a new segment.
                if callouts.last?.road != road {
                    callouts.append((index, road))
                }
            }
        }
        guard !callouts.isEmpty else { return [] }
        var visits: [String: Int] = [:]
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
        lines.append("Utterances (numbered [n] speaker: text):")
        for row in rows {
            lines.append("[\(row.number)] \(row.speakerID): \(row.transcript)")
        }
        lines.append("")
        lines.append("Return ONLY the JSON object. The FIRST character of your output MUST be `{`.")
        return lines.joined(separator: "\n")
    }

    // MARK: - Response parsing

    public struct ItemWire: Codable, Equatable, Sendable {
        public var statedScore: Double?
        public var inferredScore: Double?
        public var likeDislike: Int?
        public var comment: String?
        public var evidenceRows: [Int]?
    }

    /// Lenient parse: strip code fences / think blocks by slicing
    /// the first `{` … last `}` window, then strict-decode. Nil on
    /// anything unusable — the caller records an extraction
    /// failure for the item rather than guessing.
    public static func parseItemResponse(_ raw: String) -> ItemWire? {
        guard let start = raw.firstIndex(of: "{"),
              let end = raw.lastIndex(of: "}"),
              start <= end
        else { return nil }
        let slice = String(raw[start...end])
        return try? JSONDecoder().decode(ItemWire.self, from: Data(slice.utf8))
    }

    // MARK: - Merge policy

    /// Combine the deterministic findings with the model's wire
    /// output. Deterministic wins every disagreement (it cannot
    /// hallucinate); disagreements and multi-value captures become
    /// conflict notes instead of silent picks.
    public static func merge(
        item: EvalFormTemplate.Item,
        template: EvalFormTemplate,
        deterministic: DeterministicFindings,
        wire: ItemWire?,
        validRowNumbers: Set<Int>
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
        return result
    }
}
