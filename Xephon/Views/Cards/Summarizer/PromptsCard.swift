import SwiftUI
import Summarizer

/// Read-only catalog of every on-device LLM prompt the
/// pipeline uses. Lives on the Summarizer page right after
/// `ModelsCard` so users can see exactly what each model is
/// being asked to do, without having to crack open the source.
///
/// Each prompt is rendered inside a `DisclosureGroup` to keep
/// the card compact — many prompts are several hundred lines
/// of instructions and would overwhelm the scroll if all
/// expanded. The body text is monospaced + selectable so a
/// curious user can copy a prompt into their notes; editing
/// is intentionally not wired (would require runtime prompt
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
    }

    @ViewBuilder
    private func section(
        heading: String,
        entries: [PromptCatalog.PromptEntry]
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(heading)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.tertiary)
                .textCase(.uppercase)
                .padding(.top, 6)
            ForEach(entries) { entry in
                DisclosureGroup(entry.title) {
                    Text(entry.body)
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(Color.secondary.opacity(0.08))
                        )
                        .padding(.top, 4)
                }
                .font(.caption)
            }
        }
    }
}
