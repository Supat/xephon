import SwiftUI
import Summarizer

/// Per-section variant of `SessionSummarySheet`. Renders the
/// same `SessionSummary` body (topic / overall mood / per-
/// speaker / footer) but is scoped to a single
/// `ConversationSection` — the summary only describes the
/// utterances inside that section's bounds. Each complete
/// section has its own sheet and its own cached summary,
/// stamped onto the section's `cachedSummary` field via the
/// summarizer coordinator so it survives sheet close + reopen
/// AND a `.xph` save / load round trip.
///
/// Vertical layout matches `SessionSummarySheet`:
///   1. Content area — cached summary, "generating" spinner,
///      or empty placeholder depending on state.
///   2. Regenerate bar — only when a summary exists, one is
///      generating, or the summarizer is ready.
///
/// The two sheets are kept as separate files (rather than
/// generalized into one) because (a) the auto-fire policy and
/// inflight task slot are per-presentation, (b) the title /
/// export filename / a few minor labels differ, and (c) the
/// section sheet reads its `summary` / `isGenerating` out of
/// the section + coordinator pair rather than out of the
/// recorder's top-level `lastSessionSummary`. Sharing the
/// content view via a sub-view buys little and obscures the
/// per-presentation state plumbing.
struct SectionSummarySheet: View {
    let recorder: RecordingController
    let section: ConversationSection
    let summary: SessionSummary?
    let isGenerating: Bool
    let onRegenerate: () -> Void
    let onDismiss: () -> Void

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

                if summary != nil || isGenerating || recorder.summarizerReady {
                    Divider()
                    regenerateBar
                }
            }
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(displayTitle)
            .toolbarTitleDisplayMode(.inline)
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

    /// Heading shown in the nav bar. Falls back to the
    /// untitled placeholder so the bar isn't blank for
    /// sections the user hasn't named.
    private var displayTitle: String {
        let trimmed = section.title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty
            ? String(localized: "sections.untitled")
            : trimmed
    }

    /// Serialize the current summary to a temp `.md` file and
    /// surface the system share sheet. Filename includes a
    /// section-id slug so multi-section exports from the same
    /// session don't clobber each other in Files / Quick Look.
    private func exportMarkdown() {
        guard let summary else { return }
        let slug = String(section.id.uuidString.prefix(8))
        let stamp = Int(Date().timeIntervalSince1970)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("xephon-section-\(slug)-\(stamp).md")
        do {
            try summary.toMarkdown().write(to: url, atomically: true, encoding: .utf8)
            markdownExportURL = url
        } catch {
            // Same failure-silent rationale as `SessionSummarySheet`:
            // temp-dir writes are essentially infallible on iOS and
            // the share sheet not appearing is the user-visible
            // signal that something went wrong.
        }
    }

    @ViewBuilder
    private func resultView(for summary: SessionSummary) -> some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 18) {
                if let setting = summary.inferredSetting?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                   !setting.isEmpty {
                    section(
                        header: String(localized: "summary.section.setting"),
                        body: setting
                    )
                }
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
                footer(
                    model: summary.model,
                    mode: summary.mode,
                    generatedAt: summary.generatedAt
                )
            }
            .padding(20)
        }
    }

    @ViewBuilder
    private var regenerateBar: some View {
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
                    if let start = recorder.summarizerInferenceStart {
                        ElapsedTimeLabel(start: start)
                            .foregroundStyle(.secondary)
                    }
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
            Text(String(localized: "sections.summary.generating"))
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            if let start = recorder.summarizerInferenceStart {
                ElapsedTimeLabel(start: start)
                    .font(.title3.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
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
            Text(String(localized: "sections.summary.empty"))
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
    private func footer(
        model: String,
        mode: SummarizeMode?,
        generatedAt: Date
    ) -> some View {
        let modelLine: String = {
            let base = String(format: String(localized: "summary.footer.model"), model)
            guard let mode else { return base }
            let modeLabel: String
            switch mode {
            case .fast:      modeLabel = String(localized: "summary.footer.mode.fast")
            case .heuristic: modeLabel = String(localized: "summary.footer.mode.heuristic")
            case .deep:      modeLabel = String(localized: "summary.footer.mode.deep")
            }
            return "\(base) · \(modeLabel)"
        }()
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(modelLine)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                Text(generatedAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            Spacer(minLength: 12)
            VStack(alignment: .trailing, spacing: 2) {
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
