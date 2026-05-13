import SwiftUI
import Fusion
import Summarizer

/// Modal sheet that surfaces the on-device transcription review
/// — a list of LLM-flagged rows, each with a kind chip, optional
/// confidence, an inline transcript editor, the model's reason,
/// and per-row Commit / Dismiss / (Range…) affordances. The
/// inline editor handles the common case (a homophone / particle
/// fix); "Range…" hands the in-progress text off to the full
/// `EditUtteranceSheet` for the rarer case that needs time-range
/// surgery too.
///
/// Three states, top-to-bottom:
///   1. Content area: issue list, the "reviewing" spinner, or an
///      empty state ("no issues found" / "tap Review to start").
///   2. Action bar: Review / Re-review button gated on
///      `summarizerReady`.
///
/// The caller auto-runs `onReview` once when the sheet is first
/// opened with no cached issues AND the summarizer is ready,
/// matching the summary sheet's auto-on-first-open behavior.
struct TranscriptionReviewSheet: View {
    let recorder: RecordingController
    let issues: [TranscriptionIssue]
    let isReviewing: Bool
    let onReview: () -> Void
    let onDismiss: () -> Void

    /// Snapshot of the utterance whose full `EditUtteranceSheet` is
    /// raised from inside this review sheet via "Range…". Non-nil
    /// drives the child `.sheet(item:)` presentation; cleared on
    /// commit / cancel. The snapshot carries any in-progress inline
    /// text edit so the user doesn't lose work when switching to
    /// the full panel.
    @State private var editingUtterance: UtteranceEstimate?
    /// Which issue raised the current Range… edit sheet. After a
    /// commit we drop the issue — the row has been re-evaluated so
    /// the flag is stale.
    @State private var editingIssueID: UUID?
    /// Per-issue inline transcript edits. Keyed by `issue.id` so a
    /// row that's been edited but not yet committed survives view
    /// re-renders (the controller's utterance list refreshes
    /// constantly under live SER work). The value is whatever the
    /// user has typed into the inline TextEditor; missing means
    /// "no edit yet, fall through to the row's current transcript".
    @State private var edits: [UUID: String] = [:]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                content
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                Divider()
                actionBar
            }
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "review.title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done"), action: onDismiss)
                }
            }
            .sheet(item: $editingUtterance) { snapshot in
                EditUtteranceSheet(
                    utterance: snapshot,
                    maxDuration: recorder.fileTotalAudioDuration,
                    audioEditingEnabled: recorder.playbackSourceURL != nil,
                    onPlayRange: { start, end in
                        recorder.playRange(start: start, end: end)
                    },
                    onStopRange: { recorder.stopPlayback() },
                    isPreviewPlaying: recorder.isPreviewPlaying,
                    onCommit: { newText, newStart, newEnd in
                        recorder.stopPlayback()
                        let issueID = editingIssueID
                        editingUtterance = nil
                        editingIssueID = nil
                        Task {
                            await recorder.commitHandEdit(
                                utteranceID: snapshot.id,
                                newText: newText,
                                newStart: newStart,
                                newEnd: newEnd
                            )
                            if let issueID {
                                recorder.dismissTranscriptionIssue(id: issueID)
                                edits.removeValue(forKey: issueID)
                            }
                        }
                    },
                    onCancel: {
                        recorder.stopPlayback()
                        editingUtterance = nil
                        editingIssueID = nil
                    }
                )
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        if isReviewing && issues.isEmpty {
            reviewingView
        } else if issues.isEmpty {
            emptyView
        } else {
            issueList
        }
    }

    @ViewBuilder
    private var issueList: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 12) {
                ForEach(issues) { issue in
                    issueCard(issue)
                }
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private func issueCard(_ issue: TranscriptionIssue) -> some View {
        // Resolve the live transcript at render time — if the user
        // edited the row through another path since the review ran,
        // we want to surface the current state, not a stale snapshot.
        // Missing means the row was deleted entirely; render nothing.
        if let utterance = recorder.utterances.first(where: { $0.id == issue.utteranceID }) {
            let original = utterance.transcript
            let audioEditingEnabled = recorder.playbackSourceURL != nil
            let isThisPlaying = recorder.isPreviewPlaying
                && recorder.playingUtteranceID == utterance.id
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    kindChip(issue.kind)
                    if let confidence = issue.confidence {
                        Text(String(format: "%.0f%%", confidence * 100))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.tertiary)
                    }
                    Spacer(minLength: 4)
                    if audioEditingEnabled {
                        Button {
                            if isThisPlaying {
                                recorder.stopPlayback()
                            } else {
                                recorder.playRange(
                                    start: utterance.start,
                                    end: utterance.end,
                                    owner: utterance.id
                                )
                            }
                        } label: {
                            Image(systemName: isThisPlaying
                                ? "stop.circle.fill"
                                : "play.circle.fill"
                            )
                                .font(.title2)
                                .symbolRenderingMode(.hierarchical)
                                .foregroundStyle(Color.accentColor)
                        }
                        .buttonStyle(.borderless)
                    }
                }
                TextEditor(text: textBinding(for: issue, original: original))
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 72)
                    .padding(8)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color(uiColor: .secondarySystemBackground))
                    )
                Text(issue.reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 8) {
                    Spacer()
                    Button {
                        recorder.dismissTranscriptionIssue(id: issue.id)
                        edits.removeValue(forKey: issue.id)
                    } label: {
                        Label(
                            String(localized: "review.dismiss"),
                            systemImage: "xmark"
                        )
                        .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.bordered)
                    if audioEditingEnabled {
                        Button(String(localized: "review.range")) {
                            recorder.stopPlayback()
                            editingIssueID = issue.id
                            editingUtterance = utterance.withTranscript(
                                edits[issue.id] ?? original
                            )
                        }
                        .buttonStyle(.bordered)
                    }
                    Button {
                        commitInlineEdit(
                            issue: issue,
                            utterance: utterance,
                            original: original
                        )
                    } label: {
                        Label(
                            String(localized: "review.commit"),
                            systemImage: "checkmark"
                        )
                        .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!isCommitEnabled(for: issue, original: original))
                }
                .controlSize(.small)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
    }

    /// Binding that reads `edits[issue.id] ?? original` and writes
    /// back into the dict. Lets the TextEditor pre-populate from
    /// the row's current transcript on first paint without us
    /// mutating @State during view body, and preserves the in-
    /// progress edit across re-renders that the live SER pipeline
    /// constantly triggers.
    private func textBinding(
        for issue: TranscriptionIssue,
        original: String
    ) -> Binding<String> {
        Binding(
            get: { edits[issue.id] ?? original },
            set: { edits[issue.id] = $0 }
        )
    }

    /// Commit enabled when the user has typed something different
    /// from the row's current text. Identical-to-original commits
    /// would spend SER + fusion cycles on a no-op; if the user
    /// thinks the row is fine after all, the right affordance is
    /// Dismiss, not Commit.
    private func isCommitEnabled(
        for issue: TranscriptionIssue,
        original: String
    ) -> Bool {
        let current = (edits[issue.id] ?? original)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedOriginal = original
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return !current.isEmpty && current != trimmedOriginal
    }

    private func commitInlineEdit(
        issue: TranscriptionIssue,
        utterance: UtteranceEstimate,
        original: String
    ) {
        recorder.stopPlayback()
        let newText = edits[issue.id] ?? original
        Task {
            await recorder.commitHandEdit(
                utteranceID: utterance.id,
                newText: newText,
                newStart: utterance.start,
                newEnd: utterance.end
            )
            recorder.dismissTranscriptionIssue(id: issue.id)
            edits.removeValue(forKey: issue.id)
        }
    }

    @ViewBuilder
    private func kindChip(_ kind: TranscriptionIssue.Kind) -> some View {
        Text(kindLabel(for: kind))
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(kindColor(for: kind).opacity(0.18))
            )
            .foregroundStyle(kindColor(for: kind))
    }

    private func kindLabel(for kind: TranscriptionIssue.Kind) -> String {
        switch kind {
        case .homophone:  return String(localized: "review.kind.homophone")
        case .contextual: return String(localized: "review.kind.contextual")
        case .grammar:    return String(localized: "review.kind.grammar")
        case .other:      return String(localized: "review.kind.other")
        }
    }

    private func kindColor(for kind: TranscriptionIssue.Kind) -> Color {
        switch kind {
        case .homophone:  return .orange
        case .contextual: return .purple
        case .grammar:    return .blue
        case .other:      return .gray
        }
    }

    @ViewBuilder
    private var reviewingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)
            Text(String(localized: "review.inProgress"))
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var emptyView: some View {
        VStack(spacing: 12) {
            Image(systemName: "text.magnifyingglass")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text(String(localized: "review.empty"))
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private var actionBar: some View {
        // Label flips between first-run ("Review") and re-run
        // ("Re-review") so the affordance reads correctly in either
        // state. Disabled while the LLM is running or when the
        // summarizer isn't configured / ready.
        let label = issues.isEmpty && !isReviewing
            ? String(localized: "review.start")
            : String(localized: "review.rerun")
        Button {
            onReview()
        } label: {
            HStack(spacing: 8) {
                if isReviewing {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: "text.magnifyingglass")
                }
                Text(label)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .disabled(isReviewing || !recorder.summarizerReady)
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }
}
