import SwiftUI
import Summarizer

/// Modal sheet that surfaces the on-device session summary, and
/// hosts the summarizer's own configuration controls along the
/// bottom edge — toggle, backend picker, install / progress state
/// — so the user can enable, swap backends, or reclaim disk for
/// the model without leaving the sheet they came to see results in.
///
/// Vertical layout, top-to-bottom:
///   1. Content area: result paragraphs, the "generating" spinner,
///      or the empty-state placeholder, depending on state.
///   2. Regenerate bar (only when a summary already exists or one
///      is currently generating).
///   3. Summarizer controls — always pinned at the very bottom so
///      the user can find them in any state.
///
/// The caller auto-runs `onRegenerate` once when the sheet is first
/// opened with no cached summary AND the summarizer is ready;
/// otherwise the user explicitly initiates generation via the
/// regenerate bar after configuring the bottom controls.
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
    /// Drives the destructive-action confirmation dialog for the
    /// Remove-model glyph. Deleting the Qwen weights forces a
    /// ~4.6 GB re-download, so a tap-confirm gate avoids the
    /// "oops I meant to tap something else" failure mode that
    /// borderless icon buttons are especially prone to.
    @State private var showingRemoveModelConfirm: Bool = false

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

                Divider()
                summarizerSection
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

    // MARK: - Summarizer controls (pinned to the bottom of the sheet)

    /// On-device session summarizer controls. The toggle drives
    /// `RecordingController.setSummarizerEnabled` which kicks off
    /// the on-demand model download via `ModelStore.ensureOptional`;
    /// inline progress / install state renders under it. A "Remove
    /// model" affordance lets the user reclaim the ~4 GB working
    /// set without disabling the feature. Strictly on-device — the
    /// download fetches from the pinned Hugging Face release,
    /// inference runs locally on MLX.
    @ViewBuilder
    private var summarizerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Text(String(localized: "settings.summarizer.enable"))
                    .font(.callout)
                Spacer(minLength: 0)
                if recorder.summarizerEnabled {
                    summarizerBackendPicker
                }
                Toggle(
                    "",
                    isOn: Binding(
                        get: { recorder.summarizerEnabled },
                        set: { newValue in
                            Task { await recorder.setSummarizerEnabled(newValue) }
                        }
                    )
                )
                .labelsHidden()
                .toggleStyle(.switch)
            }
            if recorder.summarizerEnabled {
                summarizerStatusLine
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    @ViewBuilder
    private var summarizerBackendPicker: some View {
        Picker(
            String(localized: "settings.summarizer.backend"),
            selection: Binding(
                get: { recorder.summarizerBackend },
                set: { newValue in
                    Task { await recorder.setSummarizerBackend(newValue) }
                }
            )
        ) {
            Text(String(localized: "settings.summarizer.backend.appleFM"))
                .tag(SummarizerBackend.appleFM)
            Text(String(localized: "settings.summarizer.backend.qwen"))
                .tag(SummarizerBackend.qwen)
        }
        .pickerStyle(.menu)
        .labelsHidden()
    }

    @ViewBuilder
    private var summarizerStatusLine: some View {
        switch recorder.summarizerBackend {
        case .appleFM:
            if recorder.summarizerAppleFMAvailable {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.green)
                    Text(String(localized: "settings.summarizer.ready"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                Text(String(localized: "settings.summarizer.appleFM.unavailable"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .qwen:
            if recorder.summarizerDownloading {
                HStack(spacing: 8) {
                    CircularDownloadProgress(
                        downloaded: recorder.modelDownload.downloadedBytes,
                        total: recorder.modelDownload.totalBytes
                    )
                    Text(downloadProgressText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if recorder.summarizerModelInstalled {
                HStack(spacing: 6) {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.green)
                    Text(String(localized: "settings.summarizer.ready"))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 6)
                    Button {
                        showingRemoveModelConfirm = true
                    } label: {
                        Label(
                            String(localized: "settings.summarizer.remove"),
                            systemImage: "trash"
                        )
                        .labelStyle(.iconOnly)
                        .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .accessibilityLabel(Text(String(localized: "settings.summarizer.remove")))
                    .confirmationDialog(
                        String(localized: "settings.summarizer.removeConfirm.title"),
                        isPresented: $showingRemoveModelConfirm,
                        titleVisibility: .visible
                    ) {
                        Button(
                            String(localized: "settings.summarizer.removeConfirm.confirm"),
                            role: .destructive
                        ) {
                            Task { await recorder.removeSummarizerModel() }
                        }
                        Button(
                            String(localized: "settings.summarizer.removeConfirm.cancel"),
                            role: .cancel
                        ) {}
                    } message: {
                        Text(String(localized: "settings.summarizer.removeConfirm.message"))
                    }
                }
            } else {
                Text(String(localized: "settings.summarizer.notInstalled"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// "Downloading Qwen3 · 412 MB of 4.6 GB" or similar — the
    /// fraction comes from the circular indicator next to it, so
    /// the text is byte counts, not a percent.
    private var downloadProgressText: String {
        let downloaded = recorder.modelDownload.downloadedBytes
        let total = recorder.modelDownload.totalBytes
        if total > 0, downloaded > 0 {
            return String(
                format: String(localized: "settings.summarizer.downloading.bytes"),
                Self.formatBytes(downloaded),
                Self.formatBytes(total)
            )
        }
        return String(localized: "settings.summarizer.downloading")
    }

    /// `1.7 GB`, `412 MB`, etc. — tracks Apple's convention for
    /// human-readable byte sizes (decimal SI).
    private static func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = bytes >= 1_000_000_000 ? [.useGB] : [.useMB]
        return formatter.string(fromByteCount: bytes)
    }
}
