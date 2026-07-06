import SwiftUI
import Summarizer

/// Per-section variant of `SessionSummarySheet`. Same body
/// (topic / overall mood / per-speaker / footer) but scoped
/// to a single `ConversationSection` — the summary only
/// describes the utterances inside that section's bounds.
/// Each complete section has its own sheet and its own cached
/// summary, stamped onto the section's `cachedSummary` field
/// via the summarizer coordinator so it survives sheet close
/// + reopen AND a `.xph` save / load round trip.
///
/// Body rendering lives in the shared `SummaryResultView`
/// (also used by `SessionSummarySheet`); this wrapper owns
/// the per-section chrome: navigation title (the section
/// name), toolbar, and Markdown export with a section-id
/// filename slug so multi-section exports from the same
/// session don't clobber each other.
struct SectionSummarySheet: View {
    let recorder: RecordingController
    let section: ConversationSection
    let summary: SessionSummary?
    let isGenerating: Bool
    let onRegenerate: () -> Void
    /// Explicit inference cancel — only rendered while generating.
    let onCancel: () -> Void
    let onDismiss: () -> Void

    @State private var markdownExportURL: URL?

    var body: some View {
        NavigationStack {
            SummaryResultView(
                recorder: recorder,
                summary: summary,
                isGenerating: isGenerating,
                generatingMessage: String(localized: "sections.summary.generating"),
                emptyMessage: String(localized: "sections.summary.empty"),
                onRegenerate: onRegenerate
            )
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
                // Close-vs-cancel split — see SessionSummarySheet.
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
}
