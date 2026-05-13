import SwiftUI
import Summarizer

/// Modal sheet that surfaces the on-device session summary.
/// Strictly a result viewer — summarizer configuration (enable
/// toggle, backend picker, install / Remove-model affordance) lives
/// in `SummarizerCard` on the left pane's 4th page so the sheet
/// doesn't need to grow secondary chrome for state the user
/// configured once.
///
/// Vertical layout, top-to-bottom:
///   1. Content area: result paragraphs, the "generating" spinner,
///      or the empty-state placeholder, depending on state.
///   2. Regenerate bar (only when a summary already exists, one is
///      currently generating, or the summarizer is ready).
///
/// The caller auto-runs `onRegenerate` once when the sheet is first
/// opened with no cached summary AND the summarizer is ready;
/// otherwise the user explicitly initiates generation via the
/// regenerate bar after configuring the summarizer card.
struct SessionSummarySheet: View {
    let recorder: RecordingController
    let summary: SessionSummary?
    let isGenerating: Bool
    let onRegenerate: () -> Void
    let onDismiss: () -> Void

    /// Holds the URL of a freshly written Markdown export so the
    /// `.sheet(item:)` modifier can present `UIActivityViewController`
    /// over the summary sheet. Conforms to Identifiable via a URL
    /// extension elsewhere in the app.
    @State private var markdownExportURL: URL?
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
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

                // Regenerate bar only makes sense once we have a
                // summary or one is in flight — before that the user
                // is configuring the summarizer below, not deciding
                // whether to re-run. The first generation is kicked
                // off either by the toolbar Summarize button's auto-
                // fire (when the summarizer is already ready) or by
                // the user toggling the bottom controls into a ready
                // state, which lights up this bar.
                if summary != nil || isGenerating || recorder.summarizerReady {
                    Divider()
                    regenerateBar
                }
            }
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "summary.title"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        exportMarkdown()
                    } label: {
                        Label(
                            String(localized: "summary.exportMarkdown"),
                            systemImage: "square.and.arrow.up"
                        )
                    }
                    .disabled(summary == nil)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done"), action: onDismiss)
                }
            }
            .sheet(item: $markdownExportURL) { url in
                ShareSheet(items: [url])
            }
        }
    }

    /// Serialize the current summary to a temp `.md` file and surface
    /// the system share sheet. Filename is timestamped to match the
    /// JSON-export naming convention so multi-format exports of the
    /// same session sort together in Files.
    private func exportMarkdown() {
        guard let summary else { return }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("xephon-summary-\(Int(Date().timeIntervalSince1970)).md")
        do {
            try summary.toMarkdown().write(to: url, atomically: true, encoding: .utf8)
            markdownExportURL = url
        } catch {
            // Surfacing a banner here would require a parent binding;
            // failing silently is acceptable because (a) writing to
            // the temp dir is essentially infallible on iOS and (b)
            // the share sheet not appearing is the user-visible
            // signal that something went wrong.
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
    private var regenerateBar: some View {
        // Label flips between first-run ("Summarize") and rerun
        // ("Regenerate") so the affordance makes sense whether the
        // user is generating for the first time from inside the
        // sheet or re-running over an existing result.
        let label = summary == nil
            ? String(localized: "summary.summarize")
            : String(localized: "summary.regenerate")
        Button {
            onRegenerate()
        } label: {
            HStack(spacing: 8) {
                if isGenerating {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: "arrow.clockwise")
                }
                Text(label)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .disabled(isGenerating || !recorder.summarizerReady)
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
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
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(String(format: String(localized: "summary.footer.model"), model))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Text(generatedAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            Spacer(minLength: 12)
            VStack(alignment: .trailing, spacing: 2) {
                // `sparkles` is Apple's house glyph for AI-generated
                // content (Apple Intelligence affordances all use
                // it), so it reads as "this came from a model"
                // without needing explanatory text.
                HStack(spacing: 4) {
                    Image(systemName: "sparkles")
                    Text(String(localized: "summary.footer.aiGenerated"))
                }
                .font(.caption2)
                .foregroundStyle(.tertiary)
                Text(String(localized: "summary.footer.aiCaveat"))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.top, 8)
    }

}
