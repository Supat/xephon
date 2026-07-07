import SwiftUI
import Fusion

/// Adjudication sheet for one keyword's suspected mis-transcriptions
/// (see KeywordReviewModel). Each suspect row shows the utterance
/// with the suspect span highlighted and offers three verdicts:
///
/// - Not an Error: rejects the claim (session-scoped; the chip count
///   drops immediately).
/// - Replace: swaps the span for the keyword's text and commits
///   through the same `commitHandEdit` flow the find-and-replace
///   sheet uses (undo step, SER re-run, possible sentence split).
///   The row then stops being a suspect naturally — its surface now
///   raw-matches the keyword.
/// - Edit: manual correction in a TextEditor seeded with the current
///   transcript, committed through the same flow.
///
/// Corrections require session audio (commitHandEdit re-analyzes the
/// edited row); without it the action buttons disable and a footnote
/// explains — rejection stays available. Rows edited after detection
/// are marked stale rather than blind-patched at dead offsets.
struct KeywordReviewSheet: View {
    let recorder: RecordingController
    let model: KeywordReviewModel
    let keyword: Keyword

    @Environment(\.dismiss) private var dismiss
    /// Suspect id whose manual editor is open, plus its draft.
    @State private var editingSuspectID: String?
    @State private var editText: String = ""
    /// Suspect ids with a commit in flight — disables their buttons
    /// so a slow SER re-run can't be double-committed.
    @State private var committing: Set<String> = []

    private var canCommit: Bool { recorder.playbackSourceURL != nil }

    var body: some View {
        let suspects = model.suspects(forKeyword: keyword.id, in: recorder)
        NavigationStack {
            Group {
                if suspects.isEmpty {
                    ContentUnavailableView(
                        String(localized: "keywords.review.empty"),
                        systemImage: "checkmark.circle"
                    )
                } else {
                    List(suspects) { suspect in
                        suspectRow(suspect)
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle(String(localized: "keywords.review.title"))
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done")) { dismiss() }
                }
            }
            .safeAreaInset(edge: .bottom) {
                if !canCommit, !suspects.isEmpty {
                    Text(String(localized: "keywords.review.noAudio"))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.bar)
                }
            }
        }
    }

    @ViewBuilder
    private func suspectRow(_ suspect: KeywordSuspect) -> some View {
        let current = recorder.utterances.first { $0.id == suspect.utteranceID }
        let isStale = current?.transcript != suspect.transcript
        let isBusy = committing.contains(suspect.id)
        VStack(alignment: .leading, spacing: 8) {
            Text(highlightedContext(suspect))
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 6) {
                Text(verbatim: "「\(suspect.surface)」→「\(keyword.text)」")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.orange)
                if suspect.isHomophone {
                    // Same-reading badge: the strongest signal class.
                    Image(systemName: "waveform")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                // Replay the utterance's audio — hearing the span is
                // usually how the user decides between the three
                // verdicts. Same playRange(owner:) pattern as the
                // transcription-review sheet's cards; only rendered
                // when the session has audio (same gate as commits)
                // and the row still exists.
                if let current, canCommit {
                    let isThisPlaying = recorder.isPreviewPlaying
                        && recorder.playingUtteranceID == current.id
                    Button {
                        if isThisPlaying {
                            recorder.stopPlayback()
                        } else {
                            recorder.playRange(
                                start: current.start,
                                end: current.end,
                                owner: current.id
                            )
                        }
                    } label: {
                        Image(systemName: isThisPlaying
                            ? "stop.circle.fill"
                            : "play.circle.fill")
                            .font(.title3)
                            .symbolRenderingMode(.hierarchical)
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel(String(localized: isThisPlaying
                        ? "keywords.review.stop"
                        : "keywords.review.play"))
                }
            }
            Text(String(
                format: String(localized: "keywords.review.proposal"),
                keyword.text
            ))
            .font(.caption)
            .foregroundStyle(.secondary)
            if isStale {
                Text(String(localized: "keywords.review.stale"))
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if editingSuspectID == suspect.id {
                TextEditor(text: $editText)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 64)
                    .padding(6)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color(uiColor: .secondarySystemBackground))
                    )
            }
            HStack(spacing: 10) {
                Button(String(localized: "keywords.review.reject"), role: .destructive) {
                    model.reject(suspect)
                }
                .buttonStyle(.bordered)
                Spacer(minLength: 0)
                if editingSuspectID == suspect.id {
                    Button(String(localized: "keywords.review.cancel")) {
                        editingSuspectID = nil
                    }
                    .buttonStyle(.bordered)
                    Button(String(localized: "keywords.review.commit")) {
                        commit(suspect, newText: editText, current: current)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canCommit || isBusy
                        || editText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                } else {
                    Button(String(localized: "keywords.review.edit")) {
                        editText = current?.transcript ?? suspect.transcript
                        editingSuspectID = suspect.id
                    }
                    .buttonStyle(.bordered)
                    .disabled(!canCommit || isBusy || isStale)
                    Button(String(localized: "keywords.review.replace")) {
                        commit(suspect, newText: replacementText(suspect), current: current)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canCommit || isBusy || isStale)
                }
            }
            .controlSize(.small)
        }
        .padding(.vertical, 6)
        .opacity(isBusy ? 0.5 : 1)
    }

    /// Suspect transcript with the span highlighted. Offsets are
    /// valid against the DETECTION-TIME snapshot; the stale marker
    /// covers the edited-since case.
    private func highlightedContext(_ suspect: KeywordSuspect) -> AttributedString {
        var attributed = AttributedString(suspect.transcript)
        let ns = NSRange(
            location: suspect.rangeStartUTF16,
            length: suspect.rangeEndUTF16 - suspect.rangeStartUTF16
        )
        if let stringRange = Range(ns, in: suspect.transcript),
           let range = Range(stringRange, in: attributed) {
            attributed[range].backgroundColor = Color.orange.opacity(0.28)
            attributed[range].font = .body.weight(.semibold)
        }
        return attributed
    }

    /// The snapshot transcript with the suspect span swapped for the
    /// keyword's text. Only used when the row isn't stale, so the
    /// snapshot IS the current transcript.
    private func replacementText(_ suspect: KeywordSuspect) -> String {
        let ns = NSRange(
            location: suspect.rangeStartUTF16,
            length: suspect.rangeEndUTF16 - suspect.rangeStartUTF16
        )
        guard let range = Range(ns, in: suspect.transcript) else {
            return suspect.transcript
        }
        var text = suspect.transcript
        text.replaceSubrange(range, with: keyword.text)
        return text
    }

    /// Commit through the hand-edit flow (undo step, SER re-run,
    /// possible split — identical to the find-and-replace commit).
    /// On completion the utterancesVersion bump recomputes the
    /// suspect memo: a successful replacement raw-matches the
    /// keyword and drops out of the list on its own.
    private func commit(
        _ suspect: KeywordSuspect,
        newText: String,
        current: UtteranceEstimate?
    ) {
        guard let current, !committing.contains(suspect.id) else { return }
        committing.insert(suspect.id)
        editingSuspectID = nil
        Task {
            await recorder.commitHandEdit(
                utteranceID: current.id,
                newText: newText,
                newStart: current.start,
                newEnd: current.end
            )
            committing.remove(suspect.id)
        }
    }
}
