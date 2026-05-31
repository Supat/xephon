import SwiftUI
import Fusion

/// Card hosting the user-defined `ConversationSection` list.
/// Lives on the Sections page in `ControlPaneView`'s TabView
/// between Speakers and Keywords. Each row shows the
/// section's title + start-end time range; a tap opens the
/// editor sheet pre-filled, a trash button removes the row.
/// "Add Section" at the bottom raises the editor for a fresh
/// section.
///
/// Sections live in memory only (`SectionStore`) — they
/// reference per-session utterance IDs, so the card disables
/// the Add button when the session is empty and surfaces a
/// "no sections yet" empty state otherwise.
struct SectionsCard: View {
    let recorder: RecordingController
    let store: SectionStore

    @State private var editing: ConversationSection? = nil
    @State private var showingNewSection: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(String(localized: "sections.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                if !store.sections.isEmpty {
                    Text("\(store.sections.count)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            if recorder.utterances.isEmpty {
                emptyStateNoUtterances
            } else if store.sections.isEmpty {
                emptyStateNoSections
            } else {
                ForEach(store.sections) { section in
                    SectionsCardRow(
                        section: section,
                        utterances: recorder.utterances,
                        onTap: { editing = section },
                        onDelete: { store.remove(id: section.id) }
                    )
                    if section.id != store.sections.last?.id {
                        Divider()
                    }
                }
            }

            Button {
                showingNewSection = true
            } label: {
                Label(
                    String(localized: "sections.add"),
                    systemImage: "plus.circle.fill"
                )
                .font(.callout)
            }
            .buttonStyle(.borderedProminent)
            .disabled(recorder.utterances.isEmpty)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.top, 4)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .sheet(isPresented: $showingNewSection) {
            SectionEditorSheet(
                utterances: recorder.utterances,
                speakerNames: recorder.speakerNameOverrides,
                section: nil,
                onSave: { newSection in
                    store.add(newSection)
                    showingNewSection = false
                },
                onCancel: { showingNewSection = false }
            )
        }
        .sheet(item: $editing) { section in
            SectionEditorSheet(
                utterances: recorder.utterances,
                speakerNames: recorder.speakerNameOverrides,
                section: section,
                onSave: { updated in
                    store.update(updated)
                    editing = nil
                },
                onCancel: { editing = nil }
            )
        }
    }

    @ViewBuilder
    private var emptyStateNoUtterances: some View {
        Text(String(localized: "sections.emptyNoUtterances"))
            .font(.caption)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
    }

    @ViewBuilder
    private var emptyStateNoSections: some View {
        Text(String(localized: "sections.empty"))
            .font(.caption)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
    }
}

/// One row in the sections list. Title + range on the left,
/// inline trash button on the right. Tap the row body to
/// open the editor; tap trash to raise a confirmation
/// dialog (delete is one-tap on a small target, and the
/// list lives next to the transcript where a stray touch
/// is plausible).
private struct SectionsCardRow: View {
    let section: ConversationSection
    let utterances: [UtteranceEstimate]
    let onTap: () -> Void
    let onDelete: () -> Void

    @State private var showingDeleteConfirm: Bool = false

    var body: some View {
        HStack(spacing: 8) {
            Button(action: onTap) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(displayTitle)
                            .font(.callout)
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        if !section.isComplete {
                            // Distinctive glyph for the
                            // incomplete state so the user can
                            // tell at a glance which sections
                            // still need their other bound
                            // filled in.
                            Image(systemName: "exclamationmark.circle")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                                .accessibilityLabel(Text(
                                    String(localized: "sections.incomplete.a11y")
                                ))
                        }
                    }
                    Text(rangeDescription)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Button {
                showingDeleteConfirm = true
            } label: {
                Image(systemName: "trash")
                    .font(.caption)
                    .foregroundStyle(.red.opacity(0.7))
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(Text(String(localized: "sections.delete")))
            .confirmationDialog(
                String(format: String(localized: "sections.deleteConfirm.title"), displayTitle),
                isPresented: $showingDeleteConfirm,
                titleVisibility: .visible
            ) {
                Button(
                    String(localized: "sections.deleteConfirm.confirm"),
                    role: .destructive,
                    action: onDelete
                )
                Button(
                    String(localized: "sections.deleteConfirm.cancel"),
                    role: .cancel
                ) {}
            } message: {
                Text(String(localized: "sections.deleteConfirm.message"))
            }
        }
        .padding(.vertical, 4)
    }

    private var displayTitle: String {
        let trimmed = section.title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty
            ? String(localized: "sections.untitled")
            : trimmed
    }

    /// Three rendering paths:
    /// - Both bounds set → "0:12 → 1:45   1:33 total"
    /// - Start only → "Starts at 0:12 — end not set"
    /// - End only → "Ends at 1:45 — start not set"
    /// (The neither-bound-set case can't occur — the editor
    /// rejects it — but `rangeMissing` is the safe fallback.)
    private var rangeDescription: String {
        let startTime = section.startUtteranceID.flatMap { id in
            utterances.first(where: { $0.id == id })?.start
        }
        let endTime = section.endUtteranceID.flatMap { id in
            utterances.first(where: { $0.id == id })?.end
        }
        switch (startTime, endTime) {
        case (let s?, let e?):
            return String(
                format: String(localized: "sections.rangeFormat"),
                formatTime(s),
                formatTime(e),
                formatTime(max(0, e - s))
            )
        case (let s?, nil):
            return String(
                format: String(localized: "sections.rangeStartOnly"),
                formatTime(s)
            )
        case (nil, let e?):
            return String(
                format: String(localized: "sections.rangeEndOnly"),
                formatTime(e)
            )
        case (nil, nil):
            return String(localized: "sections.rangeMissing")
        }
    }

    private func formatTime(_ t: Double) -> String {
        let minutes = Int(t) / 60
        let seconds = Int(t) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }
}
