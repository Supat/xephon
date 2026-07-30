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
        host.storage(for: VehiclePerformanceMetricsA_1_StraightPlugin.self)
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

    /// Fill the sheet via the headless `EvalFormRunner` — the same
    /// pipeline the eval harness runs. The model owns the gates
    /// (availability, empty session), the phase readout, and draft
    /// persistence; the runner owns the extraction.
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
        do {
            let newDraft = try await EvalFormRunner.fill(
                template: template,
                utterances: snapshot.utterances,
                utterancesVersion: snapshot.utterancesVersion,
                inference: host.inference,
                onProgress: { [weak self] step in
                    Task { @MainActor in self?.phase = .running(step) }
                }
            )
            draft = newDraft
            persistDraft()
            phase = .idle
        } catch {
            // Runner throws only on cancellation.
            phase = .idle
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
            callouts: template.effectiveRoadCallouts
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
