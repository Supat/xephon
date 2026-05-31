import SwiftUI
import Fusion

/// Sheet for creating or editing a `ConversationSection`.
/// Reused by the `SectionsCard` for both the "Add Section"
/// flow (`section: nil`) and the edit-existing flow
/// (`section: existing`). Picks start / end utterances from
/// the session's `utterances` via menu-style Pickers.
///
/// Validates: at least one of (start, end) must be set
/// (sections with neither bound are meaningless). When both
/// are set, start must occur at or before end (single-
/// utterance sections are allowed). Title may be empty —
/// the card renders a placeholder ("Untitled section") in
/// that case rather than rejecting blank submissions.
///
/// Incomplete sections (exactly one bound) are valid and
/// useful: the user can bookmark a starting moment during
/// recording and fill in the end later (or vice versa).
struct SectionEditorSheet: View {
    let utterances: [UtteranceEstimate]
    let speakerNames: [String: String]
    /// Existing section being edited, or nil for a fresh
    /// section. When non-nil, `id` is preserved on save so
    /// callers can route the result through `store.update`
    /// instead of double-adding.
    let section: ConversationSection?
    let onSave: (ConversationSection) -> Void
    let onCancel: () -> Void

    @State private var title: String = ""
    @State private var startID: UUID? = nil
    @State private var endID: UUID? = nil
    @State private var errorMessage: String? = nil

    var body: some View {
        NavigationStack {
            Form {
                Section(String(localized: "sections.editor.title")) {
                    TextField(
                        String(localized: "sections.editor.titlePlaceholder"),
                        text: $title
                    )
                }
                Section(String(localized: "sections.editor.start")) {
                    boundaryPicker(selection: $startID)
                }
                Section(String(localized: "sections.editor.end")) {
                    boundaryPicker(selection: $endID)
                }
                if let errorMessage {
                    Section {
                        Text(errorMessage)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle(navigationTitle)
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(
                        String(localized: "sections.editor.cancel"),
                        action: onCancel
                    )
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button(
                        String(localized: "sections.editor.save"),
                        action: attemptSave
                    )
                    .disabled(!canSave)
                }
            }
            .onAppear {
                if let section {
                    title = section.title
                    startID = section.startUtteranceID
                    endID = section.endUtteranceID
                }
            }
        }
    }

    @ViewBuilder
    private func boundaryPicker(selection: Binding<UUID?>) -> some View {
        Picker("", selection: selection) {
            Text(String(localized: "sections.editor.unset"))
                .tag(UUID?.none)
            ForEach(Array(utterances.enumerated()), id: \.element.id) {
                idx, u in
                Text(optionLabel(for: u, index: idx + 1))
                    .tag(u.id as UUID?)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
    }

    private var navigationTitle: String {
        section == nil
            ? String(localized: "sections.editor.newTitle")
            : String(localized: "sections.editor.editTitle")
    }

    /// Save enabled when at least one bound is set.
    /// Incomplete sections (exactly one bound) are valid;
    /// only "neither bound set" is rejected here.
    private var canSave: Bool {
        startID != nil || endID != nil
    }

    /// "#3 [0:12] Alice: hello there…" — minimal label that
    /// lets the user pick the right utterance from a long
    /// list. Truncates transcript to the first 30 chars.
    private func optionLabel(
        for u: UtteranceEstimate,
        index: Int
    ) -> String {
        let speaker = speakerNames[u.speakerID]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .nilIfEmpty
            ?? u.speakerID
        let preview = String(u.transcript.prefix(30))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let ellipsis = u.transcript.count > 30 ? "…" : ""
        return "#\(index) [\(formatTime(u.start))] \(speaker): \(preview)\(ellipsis)"
    }

    private func attemptSave() {
        // At least one bound must be set — `canSave` already
        // gates the button, but a belt-and-braces guard here
        // keeps the contract local to the call site.
        guard startID != nil || endID != nil else { return }
        // Validate ordering only when BOTH bounds are set;
        // incomplete sections have nothing to compare against.
        // Equal IDs = single-utterance section, allowed.
        if let startID, let endID {
            guard
                let startIdx = utterances.firstIndex(where: { $0.id == startID }),
                let endIdx = utterances.firstIndex(where: { $0.id == endID })
            else {
                errorMessage = String(localized: "sections.editor.error.invalidRange")
                return
            }
            guard startIdx <= endIdx else {
                errorMessage = String(localized: "sections.editor.error.startAfterEnd")
                return
            }
        }
        let trimmedTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let result = ConversationSection(
            id: section?.id ?? UUID(),
            title: trimmedTitle,
            startUtteranceID: startID,
            endUtteranceID: endID
        )
        onSave(result)
    }

    private func formatTime(_ t: Double) -> String {
        let minutes = Int(t) / 60
        let seconds = Int(t) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
