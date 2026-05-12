import SwiftUI
import Summarizer

/// Modal sheet that surfaces the on-device session summary. Two
/// states:
///
///   - `isGenerating` true and `summary` nil → progress UI.
///     Inference runs on-device and can take tens of seconds to
///     minutes on a 7B model, so we show a spinner + an explanatory
///     line rather than blocking the parent on a synchronous call.
///   - `summary` non-nil → result UI: topic / overall mood / per-
///     speaker arcs. Static content; user dismisses to return.
///
/// We don't take a callback to "regenerate" because the cost is
/// non-trivial and the inputs (utterance list) don't change while
/// the sheet is up. Re-running is invoked by re-tapping the
/// toolbar button after dismissal.
struct SessionSummarySheet: View {
    let summary: SessionSummary?
    let isGenerating: Bool
    let onDismiss: () -> Void

    var body: some View {
        NavigationStack {
            Group {
                if let summary {
                    resultView(for: summary)
                } else if isGenerating {
                    generatingView
                } else {
                    emptyView
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "summary.title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done"), action: onDismiss)
                }
            }
        }
    }

    @ViewBuilder
    private func resultView(for summary: SessionSummary) -> some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 18) {
                section(
                    header: String(localized: "summary.section.topic"),
                    body: summary.topic
                )
                section(
                    header: String(localized: "summary.section.overallMood"),
                    body: summary.overallMood
                )
                if !summary.perSpeaker.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        sectionHeader(String(localized: "summary.section.perSpeaker"))
                        ForEach(summary.perSpeaker, id: \.speakerID) { entry in
                            speakerCard(entry)
                        }
                    }
                }
                footer(model: summary.model, generatedAt: summary.generatedAt)
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private var generatingView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .controlSize(.large)
            Text(String(localized: "summary.generating"))
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
            Image(systemName: "text.book.closed")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text(String(localized: "summary.empty"))
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func section(header: String, body: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            sectionHeader(header)
            Text(body.isEmpty ? "—" : body)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder
    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.secondary)
            .textCase(.uppercase)
    }

    @ViewBuilder
    private func speakerCard(_ entry: SessionSummary.SpeakerSummary) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(displayName(for: entry))
                    .font(.callout.bold())
                    .foregroundStyle(speakerTint(for: entry.speakerID))
                Spacer(minLength: 6)
                Text(entry.dominantMood)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(
                        Capsule().fill(speakerTint(for: entry.speakerID).opacity(0.18))
                    )
                    .foregroundStyle(speakerTint(for: entry.speakerID))
            }
            Text(entry.summary.isEmpty ? "—" : entry.summary)
                .font(.body)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func displayName(for entry: SessionSummary.SpeakerSummary) -> String {
        if let name = entry.speakerName?.trimmingCharacters(in: .whitespacesAndNewlines),
           !name.isEmpty {
            return name
        }
        return entry.speakerID
    }

    @ViewBuilder
    private func footer(model: String, generatedAt: Date) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(String(format: String(localized: "summary.footer.model"), model))
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Text(generatedAt.formatted(date: .abbreviated, time: .shortened))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.top, 8)
    }
}
