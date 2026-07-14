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
    /// Live session state. Passed into the detail sheet so the
    /// "Generate Real Prompt" button can build the actual
    /// prompt the model would receive (with the session's
    /// utterances + speaker names) and present a share sheet
    /// for export.
    let recorder: RecordingController

    /// Groups loaded once on appearance. Computing prompts is
    /// cheap (string concatenation against a 2-utterance
    /// sample) but doing it every body re-render would be
    /// pointless churn.
    @State private var summarizerEntries: [PromptCatalog.PromptEntry] = []
    @State private var reviewerEntries: [PromptCatalog.PromptEntry] = []
    @State private var textSEREntries: [PromptCatalog.PromptEntry] = []
    /// Per-model Summary + Reviewer prompts built from the live
    /// session's full utterance list with no per-spec cap and no
    /// chunking. Rebuilt whenever `recorder.utterancesVersion`
    /// flips so the body reflects the current transcript.
    @State private var debugEntries: [PromptCatalog.PromptEntry] = []
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
            if !debugEntries.isEmpty {
                section(
                    heading: String(localized: "prompts.section.debug"),
                    entries: debugEntries
                )
            }
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
            refreshDebugEntries()
        }
        .onChange(of: recorder.utterancesVersion) { _, _ in
            refreshDebugEntries()
        }
        .sheet(item: $presentedPrompt) { entry in
            PromptDetailSheet(
                entry: entry,
                recorder: recorder,
                onDismiss: { presentedPrompt = nil }
            )
        }
    }

    private func refreshDebugEntries() {
        let language: ReviewLanguage = recorder.sessionLanguage == .japanese
            ? .japanese
            : .english
        debugEntries = PromptCatalog.debugPrompts(
            utterances: recorder.utterances,
            speakerNames: recorder.speakerNameOverrides,
            language: language
        )
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
/// so it gets a title bar with Copy + Real Prompt + Done
/// buttons — the card stays out of the modal's chrome, and the
/// prompt text gets the full screen.
private struct PromptDetailSheet: View {
    let entry: PromptCatalog.PromptEntry
    let recorder: RecordingController
    let onDismiss: () -> Void

    /// Flips true for ~1.2 s right after the user taps the
    /// Copy toolbar button, swapping the icon to a checkmark
    /// so the otherwise-silent system pasteboard write gets
    /// visible confirmation. SwiftUI handles cross-fade via
    /// the implicit animation on the `systemImage`.
    @State private var justCopied = false
    @State private var copyResetTask: Task<Void, Never>?
    /// Set when the user taps Real Prompt and the catalog
    /// returns nil (entry can't be fully realized — e.g.
    /// deep-merge needs live model intermediates). Drives an
    /// alert so the user understands why the export didn't fire.
    @State private var unsupportedAlert = false
    /// Temp-file URL for the real-prompt `.txt` export. Set
    /// when the Export Real Prompt button writes the prompt
    /// to the temp dir; the nested `.sheet(item:)` below
    /// raises a `ShareSheet` over it.
    ///
    /// `ShareSheet` rather than `FilePickerCoordinator.presentExport`
    /// because the file exporter modifier lives at
    /// `ContentView`'s root and SwiftUI can't reliably raise it
    /// while a child sheet (this one) is already presented —
    /// the singleton file picker is reserved for top-level
    /// chrome flows (Save Session, Open, Export JSON). Same
    /// precedent as `SessionSummarySheet`'s Markdown export.
    @State private var realPromptExportURL: URL?

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
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        exportRealPrompt()
                    } label: {
                        Label(
                            String(localized: "prompts.generateReal"),
                            systemImage: "square.and.arrow.up.on.square"
                        )
                    }
                    // Empty utterance list means nothing to
                    // inject — disable so the user doesn't get
                    // an export with just the static
                    // instructions (which they could already
                    // copy via the Copy button).
                    .disabled(recorder.utterances.isEmpty)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done"), action: onDismiss)
                }
            }
            .alert(
                String(localized: "prompts.generateReal.unsupported.title"),
                isPresented: $unsupportedAlert
            ) {
                Button(String(localized: "summary.done")) {
                    unsupportedAlert = false
                }
            } message: {
                Text(String(localized: "prompts.generateReal.unsupported.message"))
            }
            .sheet(item: $realPromptExportURL) { url in
                ShareSheet(items: [url])
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

    /// Build the real prompt(s) for this entry using the live
    /// session's utterances + speaker names, concatenate when
    /// the live path would send multiple (chunked reviewer,
    /// per-window deep summarizer), write a `.txt` into the
    /// temp dir, and raise a `ShareSheet` (system "Save to
    /// Files" / share targets) over the prompt sheet. Surfaces
    /// an alert when the catalog returns nil (deep-merge
    /// entries can't be fully realized without running the live
    /// model first).
    private func exportRealPrompt() {
        // Debug entries are already built from the live session's
        // full utterance list with no cap and no chunking, so
        // `entry.body` IS the real prompt — skip the catalog
        // round-trip (which would fall through to `default: nil`
        // and falsely raise the deep-merge unsupported alert).
        let prompts: [String]
        if entry.id.hasPrefix("debug.") {
            prompts = [entry.body]
        } else {
            let language: ReviewLanguage = recorder.sessionLanguage == .japanese
                ? .japanese
                : .english
            guard let real = PromptCatalog.realPrompts(
                forEntryID: entry.id,
                utterances: recorder.utterances,
                speakerNames: recorder.speakerNameOverrides,
                language: language,
                glossaryTerms: recorder.summarizer.meetingGlossaryTerms()
            ) else {
                unsupportedAlert = true
                return
            }
            prompts = real
        }
        let body: String
        if prompts.count == 1 {
            body = prompts[0]
        } else {
            body = prompts.enumerated().map { idx, prompt in
                "########## Prompt \(idx + 1) of \(prompts.count) ##########\n\n\(prompt)"
            }.joined(separator: "\n\n")
        }
        let slug = entry.id.replacingOccurrences(of: ".", with: "-")
        let stamp = Int(Date().timeIntervalSince1970)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("xephon-prompt-\(slug)-\(stamp).txt")
        do {
            try body.write(to: url, atomically: true, encoding: .utf8)
            realPromptExportURL = url
        } catch {
            // Temp-dir writes are essentially infallible on
            // iOS; if it fails the share sheet not appearing
            // is the user-visible signal that something went
            // wrong, same as `SessionSummarySheet.exportMarkdown`.
        }
    }
}
