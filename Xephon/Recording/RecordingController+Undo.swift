import Foundation
import Fusion
import Diarization
import SERText

// Undo/redo plumbing for RecordingController. Lives here (not on the
// main file) to keep the controller readable and so the snapshot types
// + apply switch live next to each other.
//
// Model
// -----
// `RecordingController.undoManager` is the canonical, owned `UndoManager`
// (configured in `init` with `levelsOfUndo = 50` and
// `groupsByEvent = false`). Every user-initiated edit path captures a
// `UndoStep` representing the pre-edit slice of state and calls
// `registerUndoStep(_:actionName:)`. Undo / redo are routed via
// `Foundation.UndoManager`'s standard mechanism: when an undo fires,
// `apply(_:)` recomputes the inverse (current state) and registers it,
// which `UndoManager` auto-routes to the redo stack.
//
// Granularity
// -----------
// Single-property edits (rename a speaker, drag a slider, edit the
// session title) carry only their delta (`previous: T`) so the stack
// stays cheap. Three paths legitimately need the whole utterance batch
// — hand-edit (may split 1→N), apply-reevaluation (replaces a row
// + side maps + may reorder), revert-reevaluation (drops sibling rows
// produced by a multi-sentence split). For those, we capture a
// `UtteranceBatchSnapshot`.
//
// Out of scope
// ------------
// Speaker corrections that teach the FluidAudio diarizer DB
// (`correctUtteranceSpeaker`, `affirmUtteranceSpeaker`,
// `promoteUtteranceToNewSpeaker`) are deliberately excluded — there is
// no public "unteach" hook on `SpeakerManager`, so undoing the on-screen
// mutation would leave the diarizer's internal centroids out of sync
// with what the user sees. See MARK comments in
// `RecordingController+SpeakerEditing.swift`.

// MARK: - Step

@MainActor
enum UndoStep {
    case renameSpeaker(stored: String, previousOverride: String?)
    case reassignSpeaker(utteranceID: UUID, previousSpeakerID: String)
    case sessionTitle(previous: String)
    case fusionAcousticWeight(previous: Float)
    case fusionTextWeightFloor(previous: Float)
    case diarizerClusteringThreshold(previous: Float)
    case utteranceBatch(UtteranceBatchSnapshot)
    case sectionsBatch(previous: [ConversationSection])
    case glossary(GlossarySnapshot)
    case keywords(KeywordsSnapshot)
}

/// Wholesale snapshot of every piece of session-bound state that the
/// hand-edit, apply-reevaluation, and revert-reevaluation paths touch.
/// All component types are value types (or codable), so a struct copy
/// here is a full deep clone.
struct UtteranceBatchSnapshot: Sendable {
    var utterances: [UtteranceEstimate]
    var diarizationTimeline: [DiarizedSegment]
    var preReevaluationSnapshots: [UUID: UtteranceEstimate]
    var handEditChildren: [UUID: [UUID]]
    var utteranceEmbeddings: [UUID: [Float]]
    var utteranceObservationSegmentIDs: [UUID: UUID]
}

/// Snapshot of `GlossaryStore`'s user-visible state. Restored via
/// `replaceContents(with:)` so the store's `didMutate` fires exactly
/// once (single persistence write + one `onChange` callback).
struct GlossarySnapshot: Sendable {
    var entries: [LexiconBiasEntry]
    var isEnabled: Bool
    var isASRHintEnabled: Bool
}

/// Snapshot of `KeywordStore`'s persisted state plus the in-memory
/// selection set. Restored via `replaceContents(with:)` for atomic
/// didMutate firing; the selection set is reassigned separately
/// because it's deliberately not persisted.
struct KeywordsSnapshot: Sendable {
    var keywords: [Keyword]
    var groups: [KeywordGroup]
    var selectedKeywordIDs: Set<UUID>
}

// MARK: - Register / apply

extension RecordingController {

    // Single entry point every editor path funnels through. Captures
    // nothing on its own — the caller supplies the pre-edit step —
    // and stamps the localized action name so the menu reads
    // "Undo Edit Transcript" etc.
    func registerUndoStep(_ step: UndoStep, actionName: String) {
        // groupsByEvent = false means nothing auto-opens a group, so a
        // bare registerUndo throws "must begin a group". Wrap each step
        // in its own group. When the caller already opened one (search/
        // replace commitAll), this nests harmlessly — undoing the outer
        // group recurses into ours, and setActionName here names only
        // our nested group, leaving the outer batch name intact.
        undoManager.beginUndoGrouping()
        undoManager.registerUndo(withTarget: self) { target in
            target.apply(step)
        }
        undoManager.setActionName(actionName)
        undoManager.endUndoGrouping()
    }

    // Applied via the closure UndoManager invokes on undo (or redo).
    // The inverse step is computed from current state and registered
    // BEFORE the mutation — UndoManager auto-routes that registration
    // to the redo stack because we're already inside an undo
    // invocation. Result: bidirectional cycle without bespoke redo
    // bookkeeping.
    fileprivate func apply(_ step: UndoStep) {
        let inverse = captureInverse(of: step)
        undoManager.registerUndo(withTarget: self) { target in
            target.apply(inverse)
        }
        // Keep the action name stable across undo/redo cycles by
        // re-stamping with the current top of stack's name (no-op when
        // there isn't one; harmless when there is).
        applyStep(step)
    }

    private func captureInverse(of step: UndoStep) -> UndoStep {
        switch step {
        case .renameSpeaker(let stored, _):
            return .renameSpeaker(stored: stored, previousOverride: speakerNameOverrides[stored])
        case .reassignSpeaker(let utteranceID, _):
            let current = utterances.first(where: { $0.id == utteranceID })?.speakerID ?? ""
            return .reassignSpeaker(utteranceID: utteranceID, previousSpeakerID: current)
        case .sessionTitle:
            return .sessionTitle(previous: sessionTitle)
        case .fusionAcousticWeight:
            return .fusionAcousticWeight(previous: fusionAcousticWeight)
        case .fusionTextWeightFloor:
            return .fusionTextWeightFloor(previous: fusionTextWeightFloor)
        case .diarizerClusteringThreshold:
            return .diarizerClusteringThreshold(previous: diarizerClusteringThreshold)
        case .utteranceBatch:
            return .utteranceBatch(captureUtteranceBatchSnapshot())
        case .sectionsBatch:
            return .sectionsBatch(previous: sections.sections)
        case .glossary:
            return .glossary(captureGlossarySnapshot())
        case .keywords:
            return .keywords(captureKeywordsSnapshot())
        }
    }

    private func applyStep(_ step: UndoStep) {
        switch step {
        case .renameSpeaker(let stored, let previousOverride):
            // Same mutation shape as `renameSpeaker` (clear-on-empty
            // is collapsed into the optional), then bump version so
            // ContentView's filter memo invalidates and the display
            // name re-renders everywhere it appears.
            if let previousOverride, !previousOverride.isEmpty {
                speakerNameOverrides[stored] = previousOverride
            } else {
                speakerNameOverrides.removeValue(forKey: stored)
            }
            utterancesVersion &+= 1

        case .reassignSpeaker(let utteranceID, let previousSpeakerID):
            guard let idx = utterances.firstIndex(where: { $0.id == utteranceID }) else {
                return
            }
            utterances[idx] = utterances[idx].withSpeakerID(previousSpeakerID)
            commitUtteranceChanges()

        case .sessionTitle(let previous):
            sessionTitle = previous

        case .fusionAcousticWeight(let previous):
            // Route through the existing setter so UserDefaults
            // persistence + pipeline push happen exactly as they
            // would for a user gesture.
            setFusionAcousticWeight(previous)

        case .fusionTextWeightFloor(let previous):
            setFusionTextWeightFloor(previous)

        case .diarizerClusteringThreshold(let previous):
            // Async setter (pushes the threshold into the FluidAudio
            // diarizer actor); fire-and-forget. The undo bookkeeping
            // (inverse registration) already happened in `apply`
            // before this dispatch, so UndoManager's stack accounting
            // is correct regardless of the actual setter completion.
            Task { @MainActor in
                await setDiarizerClusteringThreshold(previous)
            }

        case .utteranceBatch(let snapshot):
            applyUtteranceBatch(snapshot)

        case .sectionsBatch(let previous):
            // `replaceAll` is a single assignment, so SwiftUI sees one
            // `@Observable` change instead of N adds. Matches the
            // session-load path.
            sections.replaceAll(previous)

        case .glossary(let snapshot):
            applyGlossarySnapshot(snapshot)

        case .keywords(let snapshot):
            applyKeywordsSnapshot(snapshot)
        }
    }

    // MARK: - Snapshot capture

    func captureUtteranceBatchSnapshot() -> UtteranceBatchSnapshot {
        UtteranceBatchSnapshot(
            utterances: utterances,
            diarizationTimeline: diarizationTimeline,
            preReevaluationSnapshots: preReevaluationSnapshots,
            handEditChildren: handEditChildren,
            utteranceEmbeddings: utteranceEmbeddings,
            utteranceObservationSegmentIDs: utteranceObservationSegmentIDs
        )
    }

    /// Restore the entire utterance batch in one shot and fire the
    /// single commit boundary that handles `utterancesVersion`,
    /// `conversationSummary` rebuild, `knownSpeakerIDs` refresh, and
    /// the speaker auto-demote sweep. Note: the auto-demote sweep is
    /// idempotent in this direction — if a restore re-introduces a
    /// speaker id, `current` will contain it and `removed` won't, so
    /// nothing is dropped.
    func applyUtteranceBatch(_ snap: UtteranceBatchSnapshot) {
        utterances = snap.utterances
        diarizationTimeline = snap.diarizationTimeline
        preReevaluationSnapshots = snap.preReevaluationSnapshots
        handEditChildren = snap.handEditChildren
        utteranceEmbeddings = snap.utteranceEmbeddings
        utteranceObservationSegmentIDs = snap.utteranceObservationSegmentIDs
        commitUtteranceChanges()
    }

    func captureGlossarySnapshot() -> GlossarySnapshot {
        GlossarySnapshot(
            entries: glossary.entries,
            isEnabled: glossary.isEnabled,
            isASRHintEnabled: glossary.isASRHintEnabled
        )
    }

    func applyGlossarySnapshot(_ snap: GlossarySnapshot) {
        // Set the stored properties directly to preserve UUIDs on
        // each entry. `replaceContents(with:)` re-keys for the import
        // flow's collision-safety guarantee — wrong here, since the
        // undo target is the exact prior entry identity. Each
        // assignment fires its own `didSet` → `didMutate` (3 writes,
        // 3 onChange callbacks) but `LexiconBiasEntry` is small and
        // the `didMutate` work is debounced internally to one disk
        // hit per runloop tick.
        glossary.entries = snap.entries
        glossary.isEnabled = snap.isEnabled
        glossary.isASRHintEnabled = snap.isASRHintEnabled
    }

    func captureKeywordsSnapshot() -> KeywordsSnapshot {
        KeywordsSnapshot(
            keywords: keywords.keywords,
            groups: keywords.groups,
            selectedKeywordIDs: keywords.selectedKeywordIDs
        )
    }

    func applyKeywordsSnapshot(_ snap: KeywordsSnapshot) {
        // Direct assignment, same reasoning as `applyGlossarySnapshot`
        // — `replaceContents` re-keys for import safety, which would
        // break referential identity for the undo target. Assign
        // groups first so that any keyword whose groupID references
        // a soon-to-be-restored group sees it resolved on read.
        keywords.groups = snap.groups
        keywords.keywords = snap.keywords
        // Selection isn't part of the persisted document; restore
        // separately. `selectedKeywordIDs` is deliberately not routed
        // through `didMutate`, so this assignment is observation-only.
        keywords.selectedKeywordIDs = snap.selectedKeywordIDs
    }

    // MARK: - Convenience registrations

    /// Capture the entire utterance batch and push it under the given
    /// localized action name. Used by `commitHandEdit`,
    /// `finalizeReevaluation` (immediately before `applyReevaluation`),
    /// and `revertReevaluation`. Call BEFORE mutating.
    func registerUtteranceBatchUndo(actionName: String) {
        let snap = captureUtteranceBatchSnapshot()
        registerUndoStep(.utteranceBatch(snap), actionName: actionName)
    }

    /// Capture the sections list and push under the action name. Call
    /// at the call site (SectionsCard, SectionEditorSheet) before
    /// invoking the matching `SectionStore` mutator.
    func registerSectionsUndo(actionName: String) {
        registerUndoStep(.sectionsBatch(previous: sections.sections), actionName: actionName)
    }

    /// Capture the glossary's persisted state and push under the
    /// action name. Call at the call site before mutating
    /// `glossary.entries` / `isEnabled` / `isASRHintEnabled`.
    func registerGlossaryUndo(actionName: String) {
        registerUndoStep(.glossary(captureGlossarySnapshot()), actionName: actionName)
    }

    /// Capture the keyword store's state and push under the action
    /// name. Call at the call site before mutating any keyword /
    /// group property.
    func registerKeywordsUndo(actionName: String) {
        registerUndoStep(.keywords(captureKeywordsSnapshot()), actionName: actionName)
    }
}
