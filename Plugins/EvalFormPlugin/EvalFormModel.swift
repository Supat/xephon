import Foundation
import SwiftUI
import Fusion
import XephonPluginKit
import XephonLogging

/// The plugin's live state: run orchestration (per-item generate
/// loop over the pure `EvalFormExtractor` machinery), draft
/// persistence through the `.xph` payload, and the review page's
/// bindings. One instance per activation.
@MainActor
@Observable
public final class EvalFormModel {
    public enum Phase: Equatable {
        case idle
        case running(String)
        case failed(String)
    }

    /// The active sheet definition: an imported pack when one is
    /// stored, else the embedded A-1 default. Cross-session
    /// (persistent storage), not per-session.
    public private(set) var template: EvalFormTemplate = .a1StraightRoad
    private let host: any PluginHost

    public private(set) var phase: Phase = .idle
    public private(set) var draft: EvalFormDraft?
    public private(set) var lastExport: String?
    /// Feedback line after a road-section detection run.
    public private(set) var lastSectionDetection: Int?

    private static let templatePackKey = "templatePack"

    /// Per-item candidate counts for the pre-run coverage readout
    /// ("ヒョコヒョコ: 12 rows") so the user can see what a run
    /// would chew on before spending minutes of inference.
    public var candidateCounts: [(item: EvalFormTemplate.Item, count: Int)] {
        let utterances = host.session.snapshot().utterances
        return template.items.map { item in
            (item, EvalFormExtractor.candidateRowNumbers(
                for: item, utterances: utterances
            ).count)
        }
    }

    public var inferenceAvailability: InferenceAvailability {
        host.inference.availability
    }

    /// Draft staleness vs the live session (summarizer pattern).
    public var draftIsStale: Bool {
        guard let draft, let version = draft.generatedAtUtterancesVersion
        else { return false }
        return version != host.session.snapshot().utterancesVersion
    }

    public init(host: any PluginHost) {
        self.host = host
        if let packData = storage.persistentData(forKey: Self.templatePackKey),
           let pack = try? EvalFormTemplate.decode(packData),
           !pack.items.isEmpty {
            template = pack
        }
        restoreDraft()
    }

    public func handle(_ event: SessionEvent) {
        switch event {
        case .sessionLoaded:
            restoreDraft()
        case .sessionCleared:
            draft = nil
            phase = .idle
        case .utterancesChanged:
            break  // staleness is derived, not evented
        }
    }

    private var storage: any PluginStorage {
        host.storage(for: EvalFormPlugin.self)
    }

    private func restoreDraft() {
        if let data = storage.sessionPayloadData {
            // Version-aware restore: v1 payloads migrate (empty
            // review state); newer-than-us payloads stay untouched
            // in the bundle and we show no draft.
            draft = EvalFormDraft.restore(
                data: data,
                storedVersion: storage.sessionPayloadVersion
            )
        } else {
            draft = nil
        }
        phase = .idle
    }

    private func persistDraft() {
        storage.sessionPayloadData = try? draft?.encoded()
    }

    // MARK: - Run

    /// Fill the sheet: deterministic pass per item, then one LLM
    /// call per item with candidate rows, then a metadata call.
    /// Items with zero candidate rows stay empty without a model
    /// call. Errors on one item don't sink the run — the item gets
    /// a conflict note and the loop continues.
    public func run() async {
        guard phase != .running("") else { return }
        let snapshot = host.session.snapshot()
        guard !snapshot.utterances.isEmpty else {
            phase = .failed(String(localized: "evalform.error.empty", bundle: .module))
            return
        }
        if case .unavailable(let reason) = host.inference.availability {
            phase = .failed(reason)
            return
        }
        var newDraft = EvalFormDraft(
            templateID: template.id,
            generatedAtUtterancesVersion: snapshot.utterancesVersion
        )
        let utterances = snapshot.utterances
        // Rows any item claims — the complement feeds 補足コメント.
        var claimedRows = Set<Int>()
        for (index, item) in template.items.enumerated() {
            phase = .running("\(item.titleJa) (\(index + 1)/\(template.items.count))")
            let candidates = EvalFormExtractor.candidateRowNumbers(
                for: item, utterances: utterances
            )
            guard !candidates.isEmpty else {
                newDraft.items.append(.init(itemID: item.id))
                continue
            }
            claimedRows.formUnion(candidates)
            let deterministic = EvalFormExtractor.deterministicFindings(
                candidateRows: candidates,
                utterances: utterances,
                template: template
            )
            // Cap the prompt at the first 40 candidate rows — a
            // pathological vocabulary hit ("フラット" in unrelated
            // talk) must not blow the context window.
            let promptRows = candidates.prefix(40).map { row in
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
            var wire: EvalFormExtractor.ItemWire?
            do {
                let raw = try await host.inference.generate(
                    prompt: prompt,
                    schemaJSON: EvalFormExtractor.itemSchemaJSON,
                    maxOutputTokens: 512
                )
                wire = EvalFormExtractor.parseItemResponse(raw)
            } catch is CancellationError {
                phase = .idle
                return
            } catch {
                AppLog.app.warning(
                    "EvalForm item \(item.id, privacy: .public) generate failed: \(String(describing: error), privacy: .public)"
                )
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
            newDraft.items.append(merged)
        }
        phase = .running(String(localized: "evalform.running.supplementary", bundle: .module))
        let supplementary = await extractSupplementary(
            utterances: utterances,
            claimedRows: claimedRows
        )
        newDraft.supplementaryComment = supplementary?.comment
        newDraft.supplementaryEvidenceRows = supplementary?.evidenceRows
        phase = .running(String(localized: "evalform.running.metadata", bundle: .module))
        newDraft.metadata = await extractMetadata(utterances: utterances)
        draft = newDraft
        persistDraft()
        phase = .idle
    }

    /// 補足コメント over the substantive rows no item claimed.
    /// Failures degrade to "no supplementary comment" — the field
    /// is additive, never worth sinking a finished run.
    private func extractSupplementary(
        utterances: [UtteranceEstimate],
        claimedRows: Set<Int>
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
            let raw = try await host.inference.generate(
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
    private func extractMetadata(utterances: [UtteranceEstimate]) async -> [String: String] {
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
            let raw = try await host.inference.generate(
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

    // MARK: - Template packs

    /// Whether the current draft was generated against the ACTIVE
    /// template — a pack swap orphans the old draft's item ids.
    public var draftMatchesTemplate: Bool {
        draft.map { $0.templateID == template.id } ?? true
    }

    /// Import a JSON template pack through the root picker,
    /// persist it as the active sheet, and seed its vocabulary
    /// into the keyword bank.
    public func importTemplatePack() {
        host.imports.presentImport(contentTypes: [.json]) { [weak self] outcome in
            guard let self else { return }
            switch outcome {
            case .loaded(let data):
                guard let pack = try? EvalFormTemplate.decode(data),
                      !pack.items.isEmpty else {
                    self.phase = .failed(
                        String(localized: "evalform.error.badPack", bundle: .module)
                    )
                    return
                }
                self.storage.setPersistentData(data, forKey: Self.templatePackKey)
                self.template = pack
                self.seedKeywords()
                self.phase = .idle
            case .cancelled:
                break
            case .failed(let reason):
                self.phase = .failed(reason)
            }
        }
    }

    /// Drop an imported pack and return to the embedded A-1 sheet.
    public func resetTemplateToDefault() {
        storage.setPersistentData(nil, forKey: Self.templatePackKey)
        template = .a1StraightRoad
        seedKeywords()
    }

    public var usesImportedTemplate: Bool {
        storage.persistentData(forKey: Self.templatePackKey) != nil
    }

    /// Seed the ACTIVE template's vocabulary. Called on activation
    /// and after a pack swap; idempotent by the host contract.
    public func seedKeywords() {
        host.annotations.contributeKeywords(
            template.items.flatMap(\.vocabulary).map(PluginKeywordSeed.init),
            groupName: template.name
        )
    }

    // MARK: - Road sections

    /// Detect road-callout segments and propose them as sections.
    public func detectRoadSections() {
        let proposals = EvalFormExtractor.roadSectionProposals(
            utterances: host.session.snapshot().utterances,
            roadNames: template.roadNames
        )
        lastSectionDetection = host.annotations.proposeSections(proposals)
    }

    // MARK: - Review state

    public func isReviewed(_ itemID: String) -> Bool {
        draft?.reviewedItemIDs.contains(itemID) ?? false
    }

    /// Toggle an item's reviewer-confirmed mark (payload v2 state).
    public func toggleReviewed(_ itemID: String) {
        guard var draft else { return }
        if let idx = draft.reviewedItemIDs.firstIndex(of: itemID) {
            draft.reviewedItemIDs.remove(at: idx)
        } else {
            draft.reviewedItemIDs.append(itemID)
        }
        self.draft = draft
        persistDraft()
    }

    // MARK: - Row helpers + export

    /// Toggle playback of a 1-based session row (evidence chip tap).
    public func playRow(_ row: Int) {
        let utterances = host.session.snapshot().utterances
        guard row >= 1, row <= utterances.count else { return }
        host.requestPlayback(utteranceID: utterances[row - 1].id)
    }

    public func transcript(forRow row: Int) -> String? {
        let utterances = host.session.snapshot().utterances
        guard row >= 1, row <= utterances.count else { return nil }
        return utterances[row - 1].transcript
    }

    /// True when there is a draft to export against the ACTIVE
    /// template — gates both the card buttons and the File-menu
    /// item.
    public var canExport: Bool {
        draft != nil && draftMatchesTemplate
    }

    public func exportMarkdown() {
        guard let draft, draftMatchesTemplate else { return }
        let markdown = EvalFormMarkdown.render(
            draft: draft,
            template: template,
            sessionTitle: host.session.snapshot().title
        )
        host.export.presentExport(
            data: Data(markdown.utf8),
            contentType: .plainText,
            suggestedFilename: "eval-form-\(template.id).md"
        ) { [weak self] outcome in
            self?.lastExport = String(describing: outcome)
        }
    }

    public func exportCSV() {
        guard let draft, draftMatchesTemplate else { return }
        let csv = EvalFormCSV.render(draft: draft, template: template)
        host.export.presentExport(
            data: Data(csv.utf8),
            contentType: .commaSeparatedText,
            suggestedFilename: "eval-form-\(template.id).csv"
        ) { [weak self] outcome in
            self?.lastExport = String(describing: outcome)
        }
    }
}
