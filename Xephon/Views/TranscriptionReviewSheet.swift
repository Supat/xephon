import SwiftUI
import Fusion
import Summarizer

/// Modal sheet that surfaces the on-device transcription review
/// — a list of LLM-flagged rows, each with a kind chip, optional
/// confidence, the row's current transcript for context, and the
/// model's one-sentence reason. Acting on an issue raises the same
/// `EditUtteranceSheet` a long-press on the row would: the human
/// is responsible for the actual correction, the model is only
/// flagging where to look. Dismiss drops the issue from the list.
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

    /// Snapshot of the utterance whose `EditUtteranceSheet` is
    /// currently raised from inside this review sheet. Non-nil
    /// drives the child `.sheet(item:)` presentation; cleared on
    /// commit / cancel / dismiss. Held as the full struct (not the
    /// id) so the edit sheet's initial state populates from a
    /// stable snapshot even if a parallel re-eval mutates the row.
    @State private var editingUtterance: UtteranceEstimate?
    /// The issue id whose edit panel is currently up. After a
    /// commit we drop that issue from the cached list — the row's
    /// been touched, so the flag is stale regardless of whether
    /// the user actually changed the text.
    @State private var editingIssueID: UUID?

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
                            // Whether or not the user actually
                            // changed anything, the row has been
                            // re-evaluated — drop the issue so the
                            // user isn't nagged about it again.
                            if let issueID {
                                recorder.dismissTranscriptionIssue(id: issueID)
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
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    kindChip(issue.kind)
                    if let confidence = issue.confidence {
                        Text(String(format: "%.0f%%", confidence * 100))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.tertiary)
                    }
                    Spacer(minLength: 4)
                    Button {
                        editingIssueID = issue.id
                        editingUtterance = utterance
                    } label: {
                        Label(
                            String(localized: "review.edit"),
                            systemImage: "pencil"
                        )
                        .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    Button {
                        recorder.dismissTranscriptionIssue(id: issue.id)
                    } label: {
                        Label(
                            String(localized: "review.dismiss"),
                            systemImage: "xmark"
                        )
                        .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                Text(utterance.transcript)
                    .font(.body)
                    .foregroundStyle(.primary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(issue.reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
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
