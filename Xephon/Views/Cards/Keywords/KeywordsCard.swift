import SwiftUI
import UniformTypeIdentifiers

/// User-managed keyword bank, rendered inline on the keywords
/// page (not a sheet). Header carries an overflow Menu with
/// Import / Export / Add Group / Remove All — same three-dot
/// disclosure pattern the glossary sheet uses, kept inside the
/// card so the page chrome stays thin.
///
/// Layout: one section per `KeywordGroup` plus an implicit
/// Ungrouped section. Group headers are themselves selection
/// affordances — tap to select every keyword in the group at
/// once (OR-filters the transcript across all of them); tap again
/// to clear. Per-keyword taps still toggle that single keyword as
/// before; both kinds of selection share the same
/// `selectedKeywordIDs` set on the store.
///
/// All mutations write back to the bound `KeywordStore`
/// immediately; the store persists on the next MainActor tick
/// and pings its `onChange` hook for any downstream consumer
/// that opts in.
struct KeywordsCard: View {
    @Bindable var store: KeywordStore
    /// App-level pickup. Import/Export route through this rather
    /// than via local `.fileImporter` / `.fileExporter` modifiers
    /// to avoid the multi-presentation hazard that swallowed
    /// Export in earlier iterations.
    let filePicker: FilePickerCoordinator
    /// Per-keyword tally — `count[keyword.id]` is the number of
    /// utterances whose normalized transcript contains that
    /// keyword's normalized form. Computed by the parent against
    /// the full utterance list (no other filter applied) so the
    /// number reads as "how prevalent is this keyword" regardless
    /// of the search field / label chip state. Map omits keys
    /// with zero matches; the row treats missing as 0.
    let keywordCounts: [UUID: Int]

    /// Single-source-of-truth for the in-card alerts +
    /// confirmation dialogs (everything that isn't the file
    /// picker — that's the coordinator's job now). Same
    /// rationale: drive every binding from one enum so at most
    /// one modal is presented at a time by construction. The
    /// per-row keyword-delete dialog deliberately stays separate
    /// (attached to each × button so iPad popovers anchor to the
    /// tapped glyph).
    private enum Presentation: Equatable {
        case none
        case errorAlert(String)
        case addGroup
        case renameGroup(KeywordGroup)
        case deleteGroup(KeywordGroup)
    }

    @State private var presentation: Presentation = .none
    @State private var newKeyword: String = ""
    @State private var addGroupText: String = ""
    @State private var renameGroupText: String = ""
    /// Drives the per-row delete confirmation dialog. Non-nil means
    /// the user tapped a row's × and we're awaiting their confirm
    /// / cancel; the actual `store.remove(id:)` only fires inside
    /// the Delete button's action so a tap-then-tap-cancel leaves
    /// the keyword untouched.
    @State private var pendingDeleteKeyword: Keyword?
    @FocusState private var newKeywordFocused: Bool

    // MARK: - Presentation bindings

    /// Each modifier reads its own Bool binding derived from the
    /// shared `presentation` enum. Setting `false` from the system
    /// (e.g. dismiss tap, swipe-down) routes back to
    /// `presentation = .none` so the next presentation starts from
    /// a clean state.
    private var errorAlertBinding: Binding<Bool> {
        Binding(
            get: { if case .errorAlert = presentation { return true }; return false },
            set: { if !$0 { presentation = .none } }
        )
    }
    private var errorMessage: String? {
        if case .errorAlert(let msg) = presentation { return msg }
        return nil
    }
    private var addGroupAlertBinding: Binding<Bool> {
        Binding(
            get: { if case .addGroup = presentation { return true }; return false },
            set: { if !$0 { presentation = .none } }
        )
    }
    private var renameGroupAlertBinding: Binding<Bool> {
        Binding(
            get: { renamingGroup != nil },
            set: { if !$0 { presentation = .none } }
        )
    }
    private var renamingGroup: KeywordGroup? {
        if case .renameGroup(let g) = presentation { return g }
        return nil
    }
    private var deleteGroupDialogBinding: Binding<Bool> {
        Binding(
            get: { deleteGroupTarget != nil },
            set: { if !$0 { presentation = .none } }
        )
    }
    private var deleteGroupTarget: KeywordGroup? {
        if case .deleteGroup(let g) = presentation { return g }
        return nil
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            inputRow
            Divider()
            content
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .alert(
            String(localized: "keywords.error.title"),
            isPresented: errorAlertBinding,
            presenting: errorMessage
        ) { _ in
            Button(String(localized: "keywords.error.ok"), role: .cancel) {
                presentation = .none
            }
        } message: { msg in
            Text(msg)
        }
        .alert(
            String(localized: "keywords.group.add.title"),
            isPresented: addGroupAlertBinding
        ) {
            TextField(
                String(localized: "keywords.group.add.placeholder"),
                text: $addGroupText
            )
            Button(String(localized: "keywords.group.add.create")) {
                store.addGroup(name: addGroupText)
                addGroupText = ""
                presentation = .none
            }
            Button(String(localized: "keywords.delete.confirm.cancel"), role: .cancel) {
                addGroupText = ""
                presentation = .none
            }
        }
        .alert(
            String(localized: "keywords.group.rename.title"),
            isPresented: renameGroupAlertBinding,
            presenting: renamingGroup
        ) { group in
            TextField(
                String(localized: "keywords.group.add.placeholder"),
                text: $renameGroupText
            )
            Button(String(localized: "keywords.group.rename.save")) {
                store.renameGroup(group.id, to: renameGroupText)
                renameGroupText = ""
                presentation = .none
            }
            Button(String(localized: "keywords.delete.confirm.cancel"), role: .cancel) {
                renameGroupText = ""
                presentation = .none
            }
        }
        .confirmationDialog(
            deleteGroupTarget.map {
                String(
                    format: String(localized: "keywords.group.delete.confirm.title"),
                    $0.name
                )
            } ?? "",
            isPresented: deleteGroupDialogBinding,
            titleVisibility: .visible,
            presenting: deleteGroupTarget
        ) { group in
            Button(
                String(localized: "keywords.delete.confirm.delete"),
                role: .destructive
            ) {
                store.removeGroup(group.id)
                presentation = .none
            }
            Button(
                String(localized: "keywords.delete.confirm.cancel"),
                role: .cancel
            ) {
                presentation = .none
            }
        } message: { _ in
            Text(String(localized: "keywords.group.delete.confirm.message"))
        }
    }

    @ViewBuilder
    private var header: some View {
        HStack {
            Text(String(localized: "keywords.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            Spacer()
            Text("\(store.keywords.count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
            Menu {
                Button {
                    addGroupText = ""
                    presentation = .addGroup
                } label: {
                    Label(
                        String(localized: "keywords.group.add"),
                        systemImage: "folder.badge.plus"
                    )
                }
                Divider()
                Button {
                    filePicker.presentImport(allowedTypes: [.json]) { result in
                        handleImport(result: result)
                    }
                } label: {
                    Label(
                        String(localized: "keywords.import"),
                        systemImage: "square.and.arrow.down"
                    )
                }
                Button {
                    do {
                        let data = try store.exportJSONData()
                        filePicker.presentExport(
                            data: data,
                            contentType: .json,
                            defaultFilename: "keywords"
                        ) { result in
                            if case .failure(let error) = result {
                                presentation = .errorAlert(String(describing: error))
                            }
                        }
                    } catch {
                        presentation = .errorAlert(String(describing: error))
                    }
                } label: {
                    Label(
                        String(localized: "keywords.export"),
                        systemImage: "square.and.arrow.up"
                    )
                }
                .disabled(store.keywords.isEmpty)
                Divider()
                Button(role: .destructive) {
                    store.removeAll()
                } label: {
                    Label(
                        String(localized: "keywords.removeAll"),
                        systemImage: "trash"
                    )
                }
                .disabled(store.keywords.isEmpty)
            } label: {
                Image(systemName: "ellipsis.circle")
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var inputRow: some View {
        HStack(spacing: 6) {
            TextField(
                String(localized: "keywords.add.placeholder"),
                text: $newKeyword
            )
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .focused($newKeywordFocused)
            .onSubmit(submitNewKeyword)
            Button(action: submitNewKeyword) {
                Image(systemName: "plus.circle.fill")
                    .font(.title3)
                    .foregroundStyle(submitDisabled ? Color.secondary : Color.accentColor)
                    .symbolRenderingMode(.hierarchical)
            }
            .buttonStyle(.borderless)
            .disabled(submitDisabled)
        }
    }

    @ViewBuilder
    private var content: some View {
        if store.keywords.isEmpty && store.groups.isEmpty {
            emptyState
        } else {
            sectionedList
        }
    }

    @ViewBuilder
    private var emptyState: some View {
        VStack(spacing: 4) {
            Text(String(localized: "keywords.empty.title"))
                .font(.callout)
                .foregroundStyle(.secondary)
            Text(String(localized: "keywords.empty.hint"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
    }

    /// Sectioned layout: Ungrouped first (when populated), then
    /// each named group in store order. Groups with zero keywords
    /// still render their header so the user has a target for
    /// rename / delete and a destination for the per-keyword move
    /// menu.
    @ViewBuilder
    private var sectionedList: some View {
        VStack(alignment: .leading, spacing: 12) {
            let ungrouped = store.keywords.filter { $0.groupID == nil }
            if !ungrouped.isEmpty {
                section(
                    groupID: nil,
                    title: String(localized: "keywords.group.ungrouped"),
                    keywords: ungrouped,
                    isNamedGroup: false
                )
            }
            ForEach(store.groups) { group in
                let inGroup = store.keywords.filter { $0.groupID == group.id }
                section(
                    groupID: group.id,
                    title: group.name,
                    keywords: inGroup,
                    isNamedGroup: true,
                    group: group
                )
            }
        }
    }

    @ViewBuilder
    private func section(
        groupID: UUID?,
        title: String,
        keywords: [Keyword],
        isNamedGroup: Bool,
        group: KeywordGroup? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            groupHeader(
                groupID: groupID,
                title: title,
                keywordCount: keywords.count,
                isNamedGroup: isNamedGroup,
                group: group
            )
            ForEach(keywords) { keyword in
                keywordRow(keyword)
            }
        }
    }

    @ViewBuilder
    private func groupHeader(
        groupID: UUID?,
        title: String,
        keywordCount: Int,
        isNamedGroup: Bool,
        group: KeywordGroup?
    ) -> some View {
        let isFullySelected = !store.keywords
            .filter { $0.groupID == groupID }
            .isEmpty
            && store.isGroupFullySelected(groupID)
        groupHeaderBody(
            groupID: groupID,
            title: title,
            keywordCount: keywordCount,
            isNamedGroup: isNamedGroup,
            group: group,
            isFullySelected: isFullySelected
        )
        // Group header is a drop destination — dropping a keyword
        // here moves it to the END of this section and adopts the
        // group assignment. Natural target for filling an empty
        // group, or appending past the last existing row.
        .dropDestination(for: String.self) { droppedIDs, _ in
            return handleDropOnGroupHeader(
                droppedIDStrings: droppedIDs,
                groupID: groupID
            )
        }
    }

    @ViewBuilder
    private func groupHeaderBody(
        groupID: UUID?,
        title: String,
        keywordCount: Int,
        isNamedGroup: Bool,
        group: KeywordGroup?,
        isFullySelected: Bool
    ) -> some View {
        HStack(spacing: 6) {
            Button {
                store.toggleGroupSelection(groupID: groupID)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isFullySelected
                        ? "checkmark.circle.fill"
                        : "folder.fill"
                    )
                    .font(.caption)
                    .foregroundStyle(isFullySelected ? Color.accentColor : Color.secondary)
                    .symbolRenderingMode(.hierarchical)
                    Text(title)
                        .font(.caption.bold())
                        .foregroundStyle(isFullySelected ? Color.accentColor : Color.secondary)
                        .textCase(.uppercase)
                    Text(String(
                        format: String(localized: "keywords.group.count"),
                        keywordCount
                    ))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(keywordCount == 0)
            .accessibilityLabel(
                String(
                    format: String(
                        localized: isFullySelected
                            ? "keywords.group.deselect.a11y"
                            : "keywords.group.select.a11y"
                    ),
                    title
                )
            )
            Spacer()
            if isNamedGroup, let group {
                Menu {
                    Button {
                        renameGroupText = group.name
                        presentation = .renameGroup(group)
                    } label: {
                        Label(
                            String(localized: "keywords.group.rename"),
                            systemImage: "pencil"
                        )
                    }
                    Button(role: .destructive) {
                        presentation = .deleteGroup(group)
                    } label: {
                        Label(
                            String(localized: "keywords.group.delete"),
                            systemImage: "trash"
                        )
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(.top, 4)
    }

    @ViewBuilder
    private func keywordRow(_ keyword: Keyword) -> some View {
        let isSelected = store.selectedKeywordIDs.contains(keyword.id)
        HStack(spacing: 6) {
            Button {
                store.toggleSelection(keyword)
            } label: {
                HStack(spacing: 6) {
                    if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.tint)
                            .symbolRenderingMode(.hierarchical)
                    }
                    Text(keyword.text)
                        .font(.body)
                        .foregroundStyle(isSelected ? Color.accentColor : Color.primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(
                String(
                    format: String(
                        localized: isSelected
                            ? "keywords.select.deselect.a11y"
                            : "keywords.select.select.a11y"
                    ),
                    keyword.text
                )
            )
            // Utterance-occurrence badge. Sits immediately in
            // front of the group picker so the row reads as
            // "[text] · [count] · [group ▼] · [×]". Always
            // rendered (even when 0) so the layout doesn't
            // jitter as keywords are added or the session
            // grows; missing map entries are treated as 0.
            Text("\(keywordCounts[keyword.id] ?? 0)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .accessibilityLabel(
                    String(
                        format: String(localized: "keywords.count.a11y"),
                        keyword.text,
                        keywordCounts[keyword.id] ?? 0
                    )
                )
            groupAssignMenu(for: keyword)
            Button {
                pendingDeleteKeyword = keyword
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
                    .symbolRenderingMode(.hierarchical)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(
                String(
                    format: String(localized: "keywords.remove.a11y"),
                    keyword.text
                )
            )
            .confirmationDialog(
                String(
                    format: String(localized: "keywords.delete.confirm.title"),
                    keyword.text
                ),
                isPresented: Binding(
                    get: { pendingDeleteKeyword?.id == keyword.id },
                    set: { if !$0 { pendingDeleteKeyword = nil } }
                ),
                titleVisibility: .visible
            ) {
                Button(
                    String(localized: "keywords.delete.confirm.delete"),
                    role: .destructive
                ) {
                    store.remove(id: keyword.id)
                    pendingDeleteKeyword = nil
                }
                Button(
                    String(localized: "keywords.delete.confirm.cancel"),
                    role: .cancel
                ) {
                    pendingDeleteKeyword = nil
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(
                    isSelected
                        ? Color.accentColor.opacity(0.18)
                        : Color(uiColor: .secondarySystemBackground)
                )
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(
                    isSelected ? Color.accentColor.opacity(0.55) : Color.clear,
                    lineWidth: 1
                )
        )
        // Drag-to-reorder. Payload is the keyword's UUID string;
        // String already conforms to Transferable so we don't have
        // to define a custom representation. The preview view
        // shows the keyword text so the drag affordance carries
        // meaningful identity instead of a placeholder rectangle.
        .draggable(keyword.id.uuidString) {
            Text(keyword.text)
                .font(.body)
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )
        }
        // Drop-on-row inserts the source BEFORE this row in the
        // flat keywords array, AND adopts this row's group. Same
        // store call handles within-section reorder and cross-
        // section move uniformly.
        .dropDestination(for: String.self) { droppedIDs, _ in
            return handleDropOnRow(
                droppedIDStrings: droppedIDs,
                targetKeyword: keyword
            )
        }
    }

    private func handleDropOnRow(
        droppedIDStrings: [String],
        targetKeyword: Keyword
    ) -> Bool {
        var didMove = false
        for raw in droppedIDStrings {
            guard let id = UUID(uuidString: raw), id != targetKeyword.id else {
                continue
            }
            store.move(id, beforeKeywordWithID: targetKeyword.id)
            didMove = true
        }
        return didMove
    }

    private func handleDropOnGroupHeader(
        droppedIDStrings: [String],
        groupID: UUID?
    ) -> Bool {
        var didMove = false
        for raw in droppedIDStrings {
            guard let id = UUID(uuidString: raw) else { continue }
            store.move(id, toEndOfGroup: groupID)
            didMove = true
        }
        return didMove
    }

    /// Per-row group picker. Tappable chip shows the current
    /// group's name (or — for ungrouped) and expands into a Menu
    /// listing every existing group plus an "Ungrouped" option.
    /// Hidden when no groups exist at all — the menu would only
    /// offer the Ungrouped option, which is also where the
    /// keyword already is.
    @ViewBuilder
    private func groupAssignMenu(for keyword: Keyword) -> some View {
        if !store.groups.isEmpty {
            Menu {
                Button {
                    store.assignKeyword(keyword.id, toGroup: nil)
                } label: {
                    if keyword.groupID == nil {
                        Label(
                            String(localized: "keywords.group.assign.none"),
                            systemImage: "checkmark"
                        )
                    } else {
                        Text(String(localized: "keywords.group.assign.none"))
                    }
                }
                Divider()
                ForEach(store.groups) { group in
                    Button {
                        store.assignKeyword(keyword.id, toGroup: group.id)
                    } label: {
                        if keyword.groupID == group.id {
                            Label(group.name, systemImage: "checkmark")
                        } else {
                            Text(group.name)
                        }
                    }
                }
            } label: {
                HStack(spacing: 2) {
                    Text(currentGroupName(for: keyword))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Image(systemName: "chevron.down")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(
                    Capsule()
                        .fill(Color(uiColor: .tertiarySystemBackground))
                )
            }
            .menuStyle(.borderlessButton)
            .accessibilityLabel(String(localized: "keywords.group.assign"))
        }
    }

    private func currentGroupName(for keyword: Keyword) -> String {
        guard let id = keyword.groupID,
              let group = store.groups.first(where: { $0.id == id }) else {
            return "—"
        }
        return group.name
    }

    private var submitDisabled: Bool {
        newKeyword.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func submitNewKeyword() {
        let trimmed = newKeyword.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        store.add(trimmed)
        newKeyword = ""
        // Keep focus so the user can chain additions without
        // re-tapping the field.
        newKeywordFocused = true
    }

    private func handleImport(result: Result<URL, any Error>) {
        switch result {
        case .success(let url):
            let didStart = url.startAccessingSecurityScopedResource()
            defer {
                if didStart { url.stopAccessingSecurityScopedResource() }
            }
            do {
                let data = try Data(contentsOf: url)
                try store.importJSONData(data)
            } catch {
                presentation = .errorAlert(String(describing: error))
            }
        case .failure(let error):
            presentation = .errorAlert(String(describing: error))
        }
    }
}

