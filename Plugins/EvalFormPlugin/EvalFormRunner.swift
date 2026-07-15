import Foundation
import Fusion
import XephonPluginKit
import XephonLogging

/// Headless fill pipeline: per-item deterministic + LLM extraction,
/// the 補足コメント pass, and the header-metadata pass, over any
/// `InferenceService`. Extracted from `EvalFormModel` so the SAME
/// code path serves the plugin page and the eval harness — the
/// harness runs this against synthetic known-answer sessions
/// without the app, the model wraps it with phase/persistence.
///
/// Throws only `CancellationError`; a failed generate on one item
/// degrades to that item's deterministic captures plus a conflict
/// note, and failed supplementary/metadata passes degrade to
/// empty — an almost-finished run is worth keeping.
public enum EvalFormRunner {

    public static func fill(
        template: EvalFormTemplate,
        utterances: [UtteranceEstimate],
        utterancesVersion: Int?,
        inference: any InferenceService,
        onProgress: (@Sendable (String) -> Void)? = nil
    ) async throws -> EvalFormDraft {
        // One batch for the whole fill: 6–8 generate calls share a
        // single model load/unload cycle on hosts that pay one.
        try await inference.withBatch {
            try await fillInBatch(
                template: template,
                utterances: utterances,
                utterancesVersion: utterancesVersion,
                inference: inference,
                onProgress: onProgress
            )
        }
    }

    private static func fillInBatch(
        template: EvalFormTemplate,
        utterances: [UtteranceEstimate],
        utterancesVersion: Int?,
        inference: any InferenceService,
        onProgress: (@Sendable (String) -> Void)?
    ) async throws -> EvalFormDraft {
        var draft = EvalFormDraft(
            templateID: template.id,
            generatedAtUtterancesVersion: utterancesVersion
        )
        // Rows any item claims — the complement feeds 補足コメント.
        var claimedRows = Set<Int>()
        for (index, item) in template.items.enumerated() {
            onProgress?("\(item.titleJa) (\(index + 1)/\(template.items.count))")
            // Mention rows ±2 context — ASR splits the verdict
            // (often the spoken score) into the row AFTER the
            // onomatopoeia; both tiers read the neighbourhood.
            // Empty iff the item was never mentioned.
            let candidates = EvalFormExtractor.contextExpandedRows(
                for: item, template: template, utterances: utterances
            )
            guard !candidates.isEmpty else {
                draft.items.append(.init(itemID: item.id))
                continue
            }
            claimedRows.formUnion(candidates)
            let deterministic = EvalFormExtractor.deterministicFindings(
                candidateRows: candidates,
                utterances: utterances,
                template: template
            )
            // Cap the prompt rows — a pathological vocabulary hit
            // ("フラット" in unrelated talk) times the ±2 window
            // must not blow the context budget. 60 rows ≈ 2k
            // prompt tokens on typical utterance lengths.
            let promptRows = candidates.prefix(60).map { row in
                (
                    number: row,
                    speakerID: utterances[row - 1].speakerID,
                    transcript: utterances[row - 1].transcript
                )
            }
            let prompt = EvalFormExtractor.extractionPrompt(
                item: item,
                template: template,
                rows: Array(promptRows),
                deterministic: deterministic
            )
            // Two attempts per item: quantized models occasionally
            // emit unparseable output once and clean JSON on the
            // retry; a failed item costs one extra generate, a
            // succeeded one costs nothing. Parse failures log the
            // raw tail — without it a failed item leaves no
            // evidence to troubleshoot (the first field-trial
            // lesson).
            var wire: EvalFormExtractor.ItemWire?
            for attempt in 1...2 where wire == nil {
                do {
                    let raw = try await inference.generate(
                        prompt: prompt,
                        schemaJSON: EvalFormExtractor.itemSchemaJSON,
                        maxOutputTokens: 512
                    )
                    wire = EvalFormExtractor.parseItemResponse(raw)
                    if wire == nil {
                        AppLog.app.warning(
                            "EvalForm item \(item.id, privacy: .public) parse failed (attempt \(attempt, privacy: .public)); raw tail: \(String(raw.suffix(240)), privacy: .public)"
                        )
                    }
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    AppLog.app.warning(
                        "EvalForm item \(item.id, privacy: .public) generate failed (attempt \(attempt, privacy: .public)): \(String(describing: error), privacy: .public)"
                    )
                }
            }
            var merged = EvalFormExtractor.merge(
                item: item,
                template: template,
                deterministic: deterministic,
                wire: wire,
                validRowNumbers: Set(candidates)
            )
            if wire == nil {
                merged.conflicts.append(
                    String(localized: "evalform.conflict.llmFailed", bundle: .module)
                )
            }
            draft.items.append(merged)
        }

        try Task.checkCancellation()
        onProgress?(String(localized: "evalform.running.supplementary", bundle: .module))
        let supplementary = await extractSupplementary(
            template: template,
            utterances: utterances,
            claimedRows: claimedRows,
            inference: inference
        )
        draft.supplementaryComment = supplementary?.comment
        draft.supplementaryEvidenceRows = supplementary?.evidenceRows

        try Task.checkCancellation()
        onProgress?(String(localized: "evalform.running.metadata", bundle: .module))
        draft.metadata = await extractMetadata(
            template: template,
            utterances: utterances,
            inference: inference
        )
        return draft
    }

    /// 補足コメント over the substantive rows no item claimed.
    /// Failures degrade to "no supplementary comment" — the field
    /// is additive, never worth sinking a finished run.
    private static func extractSupplementary(
        template: EvalFormTemplate,
        utterances: [UtteranceEstimate],
        claimedRows: Set<Int>,
        inference: any InferenceService
    ) async -> (comment: String, evidenceRows: [Int])? {
        let candidates = EvalFormExtractor.supplementaryCandidateRows(
            utterances: utterances,
            claimedRows: claimedRows
        )
        guard !candidates.isEmpty else { return nil }
        let rows = candidates.map { row in
            (
                number: row,
                speakerID: utterances[row - 1].speakerID,
                transcript: utterances[row - 1].transcript
            )
        }
        let prompt = EvalFormExtractor.supplementaryPrompt(
            template: template,
            rows: rows
        )
        do {
            let raw = try await inference.generate(
                prompt: prompt,
                schemaJSON: EvalFormExtractor.supplementarySchemaJSON,
                maxOutputTokens: 384
            )
            guard let wire = EvalFormExtractor.parseSupplementaryResponse(raw),
                  let comment = wire.comment, !comment.isEmpty
            else { return nil }
            let valid = Set(candidates)
            return (
                comment: comment,
                evidenceRows: (wire.evidenceRows ?? []).filter(valid.contains).sorted()
            )
        } catch {
            AppLog.app.warning(
                "EvalForm supplementary extract failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Header metadata, stated-only and null-first — same policy as
    /// items. Candidate rows are the session opening PLUS any row
    /// hitting a metadata cue, so specs restated mid-session (an
    /// absorber swap, a weather change) are visible to the pass.
    private static func extractMetadata(
        template: EvalFormTemplate,
        utterances: [UtteranceEstimate],
        inference: any InferenceService
    ) async -> [String: String] {
        let candidates = EvalFormExtractor.metadataCandidateRows(
            utterances: utterances,
            cues: template.metadataCues
        )
        guard !candidates.isEmpty else { return [:] }
        var lines: [String] = []
        lines.append("Extract the evaluation-sheet header fields from utterances of a Japanese test-drive session (the opening plus rows that mention conditions/specs).")
        lines.append("Fields: \(template.metadataFields.joined(separator: ", ")).")
        lines.append("Return a JSON object with exactly these fields as keys; value = the stated value as a short string, or null when not stated. NEVER guess. When a field is restated later (e.g. a spec swap), the LATEST statement wins.")
        lines.append("")
        for row in candidates {
            lines.append("[\(row)] \(utterances[row - 1].speakerID): \(utterances[row - 1].transcript)")
        }
        lines.append("")
        lines.append("Return ONLY the JSON object. The FIRST character of your output MUST be `{`.")
        let properties = template.metadataFields
            .map { "\"\($0)\":{\"type\":[\"string\",\"null\"]}" }
            .joined(separator: ",")
        let schema = "{\"type\":\"object\",\"properties\":{\(properties)}}"
        do {
            let raw = try await inference.generate(
                prompt: lines.joined(separator: "\n"),
                schemaJSON: schema,
                maxOutputTokens: 384
            )
            guard let start = raw.firstIndex(of: "{"),
                  let end = raw.lastIndex(of: "}"), start <= end,
                  let object = try? JSONSerialization.jsonObject(
                      with: Data(String(raw[start...end]).utf8)
                  ) as? [String: Any]
            else { return [:] }
            var result: [String: String] = [:]
            for field in template.metadataFields {
                if let value = object[field] as? String, !value.isEmpty {
                    result[field] = value
                }
            }
            return result
        } catch {
            AppLog.app.warning(
                "EvalForm metadata extract failed: \(String(describing: error), privacy: .public)"
            )
            return [:]
        }
    }
}
