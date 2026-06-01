import SwiftUI
import Summarizer

/// Read-only catalog of every on-device LLM prompt the
/// pipeline uses. Lives on the Summarizer page right after
/// `ModelsCard` so users can see exactly what each model is
/// being asked to do, without having to crack open the source.
///
/// Each prompt is a tap target that raises a modal sheet
/// holding the prompt's full body. The card surface itself
/// stays compact — even on iPad portrait where the summarizer
/// column is ~1/3 width, the row labels read fine and the
/// sheet uses the full screen for prompt content. Editing is
/// intentionally not wired (would require runtime prompt
/// override plumbing in every spec).
///
/// Source of truth: `PromptCatalog` in the Summarizer module.
/// MLX entries are produced by calling the real
/// `buildPrompt(...)` on the live spec types with two canned
/// sample utterances, so the structure (system header /
/// utterance lines / footer / instruction sandwich) is exactly
/// what reaches the model. Apple FM entries surface the
/// static instruction constants the same path uses.
struct PromptsCard: View {
    /// Groups loaded once on appearance. Computing prompts is
    /// cheap (string concatenation against a 2-utterance
    /// sample) but doing it every body re-render would be
    /// pointless churn.
    @State private var summarizerEntries: [PromptCatalog.PromptEntry] = []
    @State private var reviewerEntries: [PromptCatalog.PromptEntry] = []
    @State private var textSEREntries: [PromptCatalog.PromptEntry] = []
    /// Currently-presented prompt. Non-nil drives the
    /// `.sheet(item:)` modal. Tapping a row sets it; the
    /// sheet's Done button clears it. `PromptEntry` already
    /// conforms to `Identifiable` via its stable `id` field.
    @State private var presentedPrompt: PromptCatalog.PromptEntry?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(String(localized: "prompts.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
            }
            Text(String(localized: "prompts.subtitle"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)

            section(
                heading: String(localized: "prompts.section.summarizer"),
                entries: summarizerEntries
            )
            section(
                heading: String(localized: "prompts.section.reviewer"),
                entries: reviewerEntries
            )
            section(
                heading: String(localized: "prompts.section.textSER"),
                entries: textSEREntries
            )
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .task {
            if summarizerEntries.isEmpty {
                summarizerEntries = PromptCatalog.summarizerPrompts()
                reviewerEntries = PromptCatalog.reviewerPrompts()
                textSEREntries = PromptCatalog.textSERPrompts()
            }
        }
        .sheet(item: $presentedPrompt) { entry in
            PromptDetailSheet(entry: entry) {
                presentedPrompt = nil
            }
        }
    }

    @ViewBuilder
    private func section(
        heading: String,
        entries: [PromptCatalog.PromptEntry]
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(heading)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
                .padding(.top, 6)
            ForEach(entries) { entry in
                Button {
                    presentedPrompt = entry
                } label: {
                    HStack(spacing: 8) {
                        Text(entry.title)
                            .font(.caption)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        Spacer(minLength: 8)
                        Image(systemName: "chevron.right")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .contentShape(Rectangle())
                    .padding(.vertical, 4)
                }
                .buttonStyle(.plain)
                if entry.id != entries.last?.id {
                    Divider()
                }
            }
        }
    }
}

/// Modal sheet that displays one prompt's full body in
/// monospaced selectable text. Hosts its own NavigationStack
/// so it gets a title bar with Copy + Done buttons — the
/// card stays out of the modal's chrome, and the prompt text
/// gets the full screen.
private struct PromptDetailSheet: View {
    let entry: PromptCatalog.PromptEntry
    let onDismiss: () -> Void

    /// Flips true for ~1.2 s right after the user taps the
    /// Copy toolbar button, swapping the icon to a checkmark
    /// so the otherwise-silent system pasteboard write gets
    /// visible confirmation. SwiftUI handles cross-fade via
    /// the implicit animation on the `systemImage`.
    @State private var justCopied = false
    @State private var copyResetTask: Task<Void, Never>?

    var body: some View {
        NavigationStack {
            ScrollView(.vertical, showsIndicators: true) {
                Text(entry.body)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .foregroundStyle(.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            }
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(entry.title)
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        copyToClipboard()
                    } label: {
                        Label(
                            String(
                                localized: justCopied
                                    ? "prompts.copied"
                                    : "prompts.copy"
                            ),
                            systemImage: justCopied
                                ? "checkmark"
                                : "doc.on.doc"
                        )
                    }
                    .accessibilityLabel(Text(String(localized: "prompts.copy")))
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done"), action: onDismiss)
                }
            }
        }
    }

    /// Write the prompt body to the system pasteboard and flip
    /// `justCopied` for 1.2 s. The reset is scheduled on a
    /// detached Task that we cancel on each new tap so rapid
    /// re-taps don't cause the checkmark to flicker back early.
    private func copyToClipboard() {
        UIPasteboard.general.string = entry.body
        copyResetTask?.cancel()
        withAnimation { justCopied = true }
        copyResetTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            guard !Task.isCancelled else { return }
            withAnimation { justCopied = false }
        }
    }
}
