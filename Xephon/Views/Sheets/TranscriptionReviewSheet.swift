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
///
/// All inline-edit state + per-issue actions live on
/// `TranscriptionReviewCoordinator`; this view is binding +
/// render.
struct TranscriptionReviewSheet: View {
    let recorder: RecordingController
    let issues: [TranscriptionIssue]
    let isReviewing: Bool
    /// True when utterances were edited after this issue list's
    /// input was read — drives the stale-warning banner.
    let isStale: Bool
    let onReview: () -> Void
    /// Explicit inference cancel — only rendered while reviewing.
    let onCancel: () -> Void
    let onDismiss: () -> Void

    @State private var coord = TranscriptionReviewCoordinator()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // See SessionSummarySheet's stale banner comment.
                // Eager by design: dismissing/editing an issue FROM
                // this sheet mutates utterances, so the remaining
                // issues correctly read as computed-against-an-
                // older-transcript.
                if isStale, !isReviewing, !issues.isEmpty {
                    StaleContentBanner(
                        message: String(localized: "review.stale")
                    )
                }
                // See SummaryResultView for the rationale —
                // the backend badge makes on-device vs remote
                // unambiguous before the user reads any flagged
                // rows.
                HStack {
                    LLMBackendBadge(recorder: recorder)
                    Spacer()
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 6)
                content
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                Divider()
                actionBar
            }
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "review.title"))
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                // Close-vs-cancel split — see SessionSummarySheet.
                if isReviewing {
                    ToolbarItem(placement: .cancellationAction) {
                        Button(
                            String(localized: "inference.cancel"),
                            role: .destructive,
                            action: onCancel
                        )
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done"), action: onDismiss)
                }
            }
            .sheet(item: $coord.editingUtterance) { snapshot in
                EditUtteranceSheet(
                    utterance: snapshot,
                    maxDuration: recorder.fileTotalAudioDuration,
                    audioEditingEnabled: recorder.playbackSourceURL != nil,
                    onPlayRange: { start, end in
                        recorder.playRange(start: start, end: end)
                    },
                    onStopRange: { recorder.stopPlayback() },
                    onTranscribeRange: { start, end in
                        await recorder.transcribeRange(start: start, end: end)
                    },
                    isPreviewPlaying: recorder.isPreviewPlaying,
                    onCommit: { newText, newStart, newEnd in
                        coord.commitRangeEdit(
                            snapshot: snapshot,
                            newText: newText,
                            newStart: newStart,
                            newEnd: newEnd,
                            recorder: recorder
                        )
                    },
                    onCancel: {
                        coord.cancelRangeEdit(recorder: recorder)
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
                TextEditor(text: coord.textBinding(for: issue, original: original))
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 72)
                    .padding(8)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color(uiColor: .secondarySystemBackground))
                    )
                // Validated locator: the exact span the model
                // flagged (guaranteed to appear in the transcript
                // by ReviewIssueValidator). Helps the user find the
                // problem without re-reading the whole row.
                if let excerpt = issue.excerpt, !excerpt.isEmpty {
                    Text(verbatim: "「\(excerpt)」")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.orange)
                }
                Text(issue.reason)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                HStack(spacing: 8) {
                    Spacer()
                    Button {
                        coord.dismissIssue(issue, recorder: recorder)
                    } label: {
                        Label(
                            String(localized: "review.dismiss"),
                            systemImage: "xmark"
                        )
                        .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.bordered)
                    if audioEditingEnabled {
                        // Each row's spinner / disabled state is
                        // independent — the controller still
                        // serializes overlapping re-eval requests via
                        // `reevaluatingUtteranceID`, but we don't
                        // disable the other rows' buttons visually so
                        // the user sees clearly which row is
                        // currently running.
                        let isThisRunning =
                            recorder.reevaluatingUtteranceID == utterance.id
                        Button {
                            coord.startReevaluate(
                                issue, utterance: utterance, recorder: recorder
                            )
                        } label: {
                            if isThisRunning {
                                ProgressView()
                                    .controlSize(.small)
                            } else {
                                Label(
                                    String(localized: "review.reevaluate"),
                                    systemImage: "arrow.clockwise"
                                )
                                .labelStyle(.iconOnly)
                            }
                        }
                        .buttonStyle(.bordered)
                        .disabled(isThisRunning)
                        Button(String(localized: "review.range")) {
                            coord.openRangeEdit(
                                for: issue, utterance: utterance, recorder: recorder
                            )
                        }
                        .buttonStyle(.bordered)
                    }
                    Button {
                        coord.commitInlineEdit(
                            issue: issue,
                            utterance: utterance,
                            original: original,
                            recorder: recorder
                        )
                    } label: {
                        Label(
                            String(localized: "review.commit"),
                            systemImage: "checkmark"
                        )
                        .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!coord.isCommitEnabled(for: issue, original: original))
                }
                .controlSize(.small)
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
            if let start = recorder.transcriptionReviewStart {
                ElapsedTimeLabel(start: start)
                    .font(.title3.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            LLMLiveProgressLabel(progress: recorder.summarizerLiveProgress)
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
                    if let start = recorder.transcriptionReviewStart {
                        ElapsedTimeLabel(start: start)
                            .foregroundStyle(.secondary)
                    }
                    // Re-reviewing with issues already listed renders
                    // this bar, not `reviewingView` — the live
                    // progress line must appear in both states.
                    LLMLiveProgressLabel(progress: recorder.summarizerLiveProgress)
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
