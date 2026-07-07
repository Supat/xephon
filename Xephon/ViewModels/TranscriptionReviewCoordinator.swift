import Foundation
import SwiftUI
import Fusion
import Summarizer

/// Owns every piece of inline-edit state for the transcription
/// review sheet: per-issue typed-in text overrides, the set of
/// already-re-evaluated issue IDs, and the snapshot driving the
/// nested `EditUtteranceSheet` when the user reaches for "Range…".
///
/// Mutating UI flags (`editingUtterance`, `editingIssueID`) stay
/// default-observed so the sheet binding reacts; the inline-edit
/// dictionary + the re-evaluated-id set are
/// `@ObservationIgnored` because the row already observes
/// `recorder.utterances` and re-renders on its own — observing the
/// dict here too would double-invalidate every keystroke.
@MainActor
@Observable
final class TranscriptionReviewCoordinator {
    /// Snapshot of the utterance whose full `EditUtteranceSheet`
    /// is raised from inside this review sheet via "Range…".
    /// Non-nil drives the child `.sheet(item:)` presentation;
    /// cleared on commit / cancel. The snapshot carries any in-
    /// progress inline text edit so the user doesn't lose work
    /// when switching to the full panel.
    var editingUtterance: UtteranceEstimate?
    /// Which issue raised the current Range… edit sheet. After a
    /// commit we drop the issue — the row has been re-evaluated
    /// so the flag is stale.
    var editingIssueID: UUID?

    /// Per-issue inline transcript edits. Keyed by `issue.id` so
    /// a row that's been edited but not yet committed survives
    /// view re-renders (the controller's utterance list refreshes
    /// constantly under live SER work). The value is whatever the
    /// user has typed into the inline TextEditor; missing means
    /// "no edit yet, fall through to the row's current
    /// transcript".
    @ObservationIgnored
    private var edits: [UUID: String] = [:]

    /// Issues whose Re-evaluate button has already run
    /// successfully this session. The pipeline updates the row's
    /// stored transcript in place, so by the time the user sees
    /// the refreshed text the inline diff (user-typed vs. row's
    /// current) reads "no change" and Commit would be disabled.
    /// Flagging the issue id here force-enables Commit so the
    /// user can dismiss the (now-resolved) flag with a single tap
    /// instead of retyping or hunting for the dismiss button.
    @ObservationIgnored
    private var reevaluatedIssueIDs: Set<UUID> = []

    // MARK: - Inline TextEditor binding

    /// Binding that reads `edits[issue.id] ?? original` and writes
    /// back into the dict. Lets the TextEditor pre-populate from
    /// the row's current transcript on first paint without us
    /// mutating @State during view body, and preserves the in-
    /// progress edit across re-renders that the live SER pipeline
    /// constantly triggers.
    func textBinding(
        for issue: TranscriptionIssue,
        original: String
    ) -> Binding<String> {
        Binding(
            get: { self.edits[issue.id] ?? original },
            set: { self.edits[issue.id] = $0 }
        )
    }

    /// Commit enabled when either (a) the user has typed
    /// something different from the row's current text — the
    /// inline-edit case — or (b) the row was just re-evaluated,
    /// in which case Commit is the user's "acknowledge + dismiss"
    /// affordance and runs no further edit pass.
    func isCommitEnabled(
        for issue: TranscriptionIssue,
        original: String
    ) -> Bool {
        if reevaluatedIssueIDs.contains(issue.id) { return true }
        let current = (edits[issue.id] ?? original)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedOriginal = original
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return !current.isEmpty && current != trimmedOriginal
    }

    // MARK: - Per-issue actions

    func dismissIssue(_ issue: TranscriptionIssue, recorder: RecordingController) {
        recorder.dismissTranscriptionIssue(id: issue.id)
        edits.removeValue(forKey: issue.id)
    }

    /// Hand the row's *unchanged* audio back to the pipeline:
    /// offline ASR, SER, fusion, the works. Useful when the LLM
    /// flagged the row but the user trusts a fresh model pass more
    /// than their own inline correction. The issue stays in the
    /// list afterward (so the user can see what changed); Commit
    /// becomes the affordance to acknowledge + dismiss.
    func startReevaluate(
        _ issue: TranscriptionIssue,
        utterance: UtteranceEstimate,
        recorder: RecordingController
    ) {
        recorder.stopPlayback()
        let issueID = issue.id
        Task {
            await recorder.reevaluate(utterance)
            // Drop any stale inline edit so the TextEditor
            // re-binds to the refreshed row transcript.
            edits.removeValue(forKey: issueID)
            reevaluatedIssueIDs.insert(issueID)
        }
    }

    /// Open the nested `EditUtteranceSheet` carrying any in-
    /// progress inline edit so the user doesn't lose work when
    /// switching to the full panel.
    func openRangeEdit(
        for issue: TranscriptionIssue,
        utterance: UtteranceEstimate,
        recorder: RecordingController
    ) {
        recorder.stopPlayback()
        let original = utterance.transcript
        editingIssueID = issue.id
        editingUtterance = utterance.withTranscript(edits[issue.id] ?? original)
    }

    /// Range-edit commit path. Capture everything into locals
    /// BEFORE the writes that dismiss the sheet — `snapshot` is
    /// the content-builder parameter and once `editingUtterance`
    /// is cleared the SwiftUI hosting context can tear down.
    /// Sourcing utteranceID + issueID off `snapshot` /
    /// `editingIssueID` first means the Task owns its own copies
    /// and can't observe a torn-down state.
    func commitRangeEdit(
        snapshot: UtteranceEstimate,
        newText: String,
        newStart: TimeInterval,
        newEnd: TimeInterval,
        recorder: RecordingController
    ) {
        let utteranceID = snapshot.id
        let issueID = editingIssueID
        let committedText = newText
        let committedStart = newStart
        let committedEnd = newEnd
        recorder.stopPlayback()
        Task {
            await recorder.commitHandEdit(
                utteranceID: utteranceID,
                newText: committedText,
                newStart: committedStart,
                newEnd: committedEnd
            )
            if let issueID {
                recorder.dismissTranscriptionIssue(id: issueID)
                edits.removeValue(forKey: issueID)
            }
        }
        editingUtterance = nil
        editingIssueID = nil
    }

    func cancelRangeEdit(recorder: RecordingController) {
        recorder.stopPlayback()
        editingUtterance = nil
        editingIssueID = nil
    }

    /// Inline-Commit path. Handles three cases:
    ///
    ///   1. Re-eval with no further inline edit — nothing left to
    ///      commit, just dismiss. A second `commitHandEdit` pass
    ///      on the same text would burn SER + fusion cycles and
    ///      would clobber `wasReevaluated` with `wasHandEdited`.
    ///   2. Pure inline edit — commit the typed-in text.
    ///   3. Re-eval THEN inline edit — commit the typed-in text
    ///      on top of the re-evaluated transcript, so a pending
    ///      multi-sentence split doesn't drop on the floor.
    func commitInlineEdit(
        issue: TranscriptionIssue,
        utterance: UtteranceEstimate,
        original: String,
        recorder: RecordingController
    ) {
        recorder.stopPlayback()
        let edited = (edits[issue.id] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        // Look up the live (possibly post-reeval) transcript so we
        // can tell whether the user's typed-in edit still differs.
        // Without this, a Commit after Re-evaluate would drop a
        // pending multi-sentence edit on the floor — the row was
        // refreshed to the re-eval's ASR result, but the user's
        // typed-in split-worthy text in `edits[issue.id]` never
        // reached `commitHandEdit`, so the split path was skipped.
        let liveTranscript = recorder.utterances
            .first(where: { $0.id == utterance.id })?
            .transcript
            .trimmingCharacters(in: .whitespacesAndNewlines)
            ?? original.trimmingCharacters(in: .whitespacesAndNewlines)
        let wasReevaluated = reevaluatedIssueIDs.contains(issue.id)
        let hasPendingEdit = !edited.isEmpty && edited != liveTranscript

        if wasReevaluated && !hasPendingEdit {
            recorder.dismissTranscriptionIssue(id: issue.id)
            edits.removeValue(forKey: issue.id)
            reevaluatedIssueIDs.remove(issue.id)
            return
        }

        // Capture into locals before the @State mutations dismiss
        // the row — the Task body shouldn't dereference `issue` /
        // `utterance` whose hosting view may already be gone.
        let issueID = issue.id
        let utteranceID = utterance.id
        let newStart = utterance.start
        let newEnd = utterance.end
        let newText = hasPendingEdit ? edited : (edits[issue.id] ?? original)
        Task {
            await recorder.commitHandEdit(
                utteranceID: utteranceID,
                newText: newText,
                newStart: newStart,
                newEnd: newEnd
            )
            recorder.dismissTranscriptionIssue(id: issueID)
            edits.removeValue(forKey: issueID)
            reevaluatedIssueIDs.remove(issueID)
        }
    }
}
