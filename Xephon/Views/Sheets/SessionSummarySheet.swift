import SwiftUI
import Summarizer

/// Modal sheet that surfaces the on-device session summary.
/// Strictly a result viewer — summarizer configuration (enable
/// toggle, backend picker, install / Remove-model affordance) lives
/// in `SummarizerCard` on the left pane's 4th page so the sheet
/// doesn't need to grow secondary chrome for state the user
/// configured once.
///
/// Body rendering (result / generating / empty + the regenerate
/// bar) lives in the shared `SummaryResultView`, which is also
/// used by `SectionSummarySheet`. This wrapper owns the per-
/// presentation chrome: navigation title, toolbar (Markdown
/// export + Done), and the Markdown share-sheet plumbing.
///
/// The caller auto-runs `onRegenerate` once when the sheet is
/// first opened with no cached summary AND the summarizer is
/// ready; otherwise the user explicitly initiates generation
/// via the regenerate bar after configuring the summarizer
/// card.
struct SessionSummarySheet: View {
    let recorder: RecordingController
    let summary: SessionSummary?
    let isGenerating: Bool
    let onRegenerate: () -> Void
    /// Explicit inference cancel — only rendered while generating.
    let onCancel: () -> Void
    let onDismiss: () -> Void

    /// Holds the URL of a freshly written Markdown export so the
    /// `.sheet(item:)` modifier can present `UIActivityViewController`
    /// over the summary sheet. Conforms to Identifiable via a URL
    /// extension elsewhere in the app.
    @State private var markdownExportURL: URL?

    var body: some View {
        NavigationStack {
            SummaryResultView(
                recorder: recorder,
                summary: summary,
                isGenerating: isGenerating,
                generatingMessage: String(localized: "summary.generating"),
                emptyMessage: String(localized: "summary.empty"),
                onRegenerate: onRegenerate
            )
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "summary.title"))
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
                // Done ONLY closes the sheet — a running generation
                // continues in the background (toolbar spinner keeps
                // showing; the result lands via writeback). Explicit
                // cancellation is the leading Cancel button, present
                // only while a pass is in flight.
                if isGenerating {
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
}
