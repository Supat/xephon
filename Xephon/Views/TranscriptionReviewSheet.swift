import SwiftUI
import Summarizer

/// Modal sheet that surfaces the on-device transcription review
/// — a list of LLM-flagged rows with original / suggested diff,
/// reason text, and per-row accept / reject affordances. Routes
/// accept actions through `RecordingController.acceptTranscriptionSuggestion`
/// (which calls `commitHandEdit`, so SER + fusion get re-run on the
/// corrected transcript); rejects simply drop the suggestion.
///
/// Three states, top-to-bottom:
///   1. Content area: suggestion list, the "reviewing" spinner, or
///      an empty state ("no issues found" / "tap Review to start").
///   2. Action bar: Review / Re-review button gated on
///      `summarizerReady`.
///
/// The caller auto-runs `onReview` once when the sheet is first
/// opened with no cached suggestions AND the summarizer is ready,
/// matching the summary sheet's auto-on-first-open behavior.
struct TranscriptionReviewSheet: View {
    let recorder: RecordingController
    let suggestions: [TranscriptionSuggestion]
    let isReviewing: Bool
    let onReview: () -> Void
    let onDismiss: () -> Void

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
        }
    }

    @ViewBuilder
    private var content: some View {
        if isReviewing && suggestions.isEmpty {
            reviewingView
        } else if suggestions.isEmpty {
            emptyView
        } else {
            suggestionList
        }
    }

    @ViewBuilder
    private var suggestionList: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 12) {
                ForEach(suggestions) { suggestion in
                    suggestionCard(suggestion)
                }
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private func suggestionCard(_ suggestion: TranscriptionSuggestion) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                kindChip(suggestion.kind)
                if let confidence = suggestion.confidence {
                    Text(String(format: "%.0f%%", confidence * 100))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
                Spacer(minLength: 4)
                if !suggestion.suggestedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button {
                        Task {
                            await recorder.acceptTranscriptionSuggestion(id: suggestion.id)
                        }
                    } label: {
                        Label(
                            String(localized: "review.accept"),
                            systemImage: "checkmark"
                        )
                        .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                }
                Button {
                    recorder.dismissTranscriptionSuggestion(id: suggestion.id)
                } label: {
                    Label(
                        String(localized: "review.reject"),
                        systemImage: "xmark"
                    )
                    .labelStyle(.iconOnly)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            diffRow(
                label: String(localized: "review.original"),
                text: suggestion.originalText,
                style: .secondary
            )
            if !suggestion.suggestedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                diffRow(
                    label: String(localized: "review.suggested"),
                    text: suggestion.suggestedText,
                    style: .primary
                )
            }
            Text(suggestion.reason)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private enum DiffStyle { case primary, secondary }

    @ViewBuilder
    private func diffRow(label: String, text: String, style: DiffStyle) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
            Text(text)
                .font(style == .primary ? .body.weight(.semibold) : .body)
                .foregroundStyle(style == .primary ? .primary : .secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func kindChip(_ kind: TranscriptionSuggestion.Kind) -> some View {
        Text(kindLabel(for: kind))
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                Capsule().fill(kindColor(for: kind).opacity(0.18))
            )
            .foregroundStyle(kindColor(for: kind))
    }

    private func kindLabel(for kind: TranscriptionSuggestion.Kind) -> String {
        switch kind {
        case .homophone:  return String(localized: "review.kind.homophone")
        case .contextual: return String(localized: "review.kind.contextual")
        case .grammar:    return String(localized: "review.kind.grammar")
        case .other:      return String(localized: "review.kind.other")
        }
    }

    private func kindColor(for kind: TranscriptionSuggestion.Kind) -> Color {
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
        let label = suggestions.isEmpty && !isReviewing
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
