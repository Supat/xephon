import SwiftUI
import Fusion

/// Modal sheet that does in-place find-and-replace across the
/// session's utterance transcripts. Modeled on
/// `TranscriptionReviewSheet`: search + replace fields at the top,
/// a scrollable list of matching rows beneath, each row showing the
/// transcript with the search term highlighted plus per-row Replace
/// and Commit buttons.
///
/// Two-stage workflow per row:
///   1. **Replace** stages the substitution into
///      `coord.staged(for:)`. The text field reflects the staged
///      result, with the post-replace term tinted green so the
///      user can verify the edit before committing.
///   2. **Commit** writes the staged text to the row via
///      `recorder.commitHandEdit`, which re-runs the text SER + late
///      fusion. The row drops out of the match list afterward
///      (assuming the replace term doesn't itself contain the
///      search term).
///
/// Matching is cross-script via `JapaneseSearchNormalizer`, the
/// same normalizer the transcript list's search field uses —
///渋谷, しぶや, シブヤ, and Shibuya all surface the same rows. The
/// normalizer collapses every form to a Hepburn-romaji key so the
/// query and the transcript compare on neutral ground. Empty
/// search term yields no results — we don't list every utterance.
///
/// **Replace / highlight caveat.** Normalization is one-way (kana →
/// romaji), so a normalized hit can't reliably be mapped back to a
/// character range in the original transcript. Highlighting +
/// Replace therefore still operate on raw case-insensitive
/// substring matches. When a row matched cross-script but the raw
/// transcript doesn't contain the literal search term, the card
/// still shows up so the user knows it exists, but the Replace
/// button is disabled and a small note explains why.
///
/// All staged-edit / selection / debounced-search state lives on
/// `SearchReplaceCoordinator`; this view is binding + render.
struct SearchReplaceSheet: View {
    let recorder: RecordingController
    let onDismiss: () -> Void

    @State private var coord = SearchReplaceCoordinator()
    /// Tracks which card's TextEditor currently has focus, by
    /// utterance id. Drives the per-card Annotate button's enabled
    /// state — Annotate only fires when its OWN editor is focused
    /// so the cursor-position insertion lands in the right row.
    @FocusState private var focusedEditorID: UUID?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                searchFields
                Divider()
                content
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            }
            .background(Color(uiColor: .systemBackground))
            .navigationTitle(String(localized: "searchReplace.title"))
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(String(localized: "summary.done"), action: onDismiss)
                }
            }
            // Each highlighted match is wired into an attributed-
            // string `.link` pointing at `xephon-match://m?u=…&i=…`.
            // Override the system openURL handler so those taps
            // toggle the per-row selection set instead of leaving
            // the app. `.systemAction` is returned for any URL we
            // don't recognize so the sheet can still host real
            // links if one ever shows up here.
            .environment(\.openURL, OpenURLAction { url in
                guard let parsed = SearchReplaceMatchURL.parse(url) else {
                    return .systemAction
                }
                coord.toggleSelection(
                    matchIndex: parsed.index,
                    for: parsed.utteranceID
                )
                return .handled
            })
            .onAppear {
                // Pre-fill from the keyword selection ONLY when
                // exactly one keyword is selected. Multi-selection
                // (and group-selection, which selects all keywords
                // in the group) doesn't have an obvious single
                // string to seed the field with, and joining them
                // would produce a query that matches nothing.
                // The empty-field guards mean typed-then-cleared
                // fields stay cleared within one sheet lifetime;
                // re-opening rebuilds the coordinator (it's
                // @State) and re-checks the selection.
                //
                // Both Find AND Replace seed to the same keyword
                // text: the most common single-keyword workflow is
                // "I picked this term to focus on; now fix one of
                // its appearances," where the user edits the
                // Replace field down from the search term rather
                // than typing the prefix again. Seeding Replace
                // to the same value is a no-op until the user
                // changes it — Replace's stage button is gated on
                // `searchTerm != replaceTerm` implicitly because
                // identical replace text leaves no diff to stage.
                if let text = recorder.keywords.singleSelectedKeyword?.text,
                   !text.isEmpty {
                    if coord.searchTerm.isEmpty {
                        coord.searchTerm = text
                    }
                    if coord.replaceTerm.isEmpty {
                        coord.replaceTerm = text
                    }
                }
                coord.scheduleSearch(in: recorder)
            }
            .onDisappear { coord.cancelSearch() }
            .onChange(of: coord.searchTerm) { _, _ in coord.scheduleSearch(in: recorder) }
            .onChange(of: coord.includeSimilar) { _, _ in coord.scheduleSearch(in: recorder) }
            .onChange(of: recorder.utterancesVersion) { _, _ in coord.scheduleSearch(in: recorder) }
            .onChange(of: recorder.utterances.count) { _, _ in coord.scheduleSearch(in: recorder) }
        }
    }

    @ViewBuilder
    private var searchFields: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .frame(width: 18)
                TextField(
                    String(localized: "searchReplace.search.placeholder"),
                    text: $coord.searchTerm
                )
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                // Fuzzy "Include similar" toggle. Sits on the search
                // row (not the replace row) because it modifies what
                // counts as a match — the replace term is unaffected.
                // `.fixedSize()` keeps it from stealing field width;
                // `.controlSize(.mini)` matches the compact density
                // of the surrounding chrome.
                Toggle(
                    String(localized: "searchReplace.includeSimilar"),
                    isOn: $coord.includeSimilar
                )
                .toggleStyle(.switch)
                .controlSize(.mini)
                .labelsHidden()
                .accessibilityLabel(
                    String(localized: "searchReplace.includeSimilar.a11y")
                )
                Image(systemName: "arrow.up.left.and.down.right.magnifyingglass")
                    .foregroundStyle(coord.includeSimilar ? Color.accentColor : Color.secondary)
                    .font(.caption)
                    .accessibilityHidden(true)
            }
            HStack(spacing: 8) {
                Image(systemName: "arrow.right.circle")
                    .foregroundStyle(.secondary)
                    .frame(width: 18)
                TextField(
                    String(localized: "searchReplace.replace.placeholder"),
                    text: $coord.replaceTerm
                )
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            }
            HStack {
                Text(String.localizedStringWithFormat(
                    String(localized: "searchReplace.matchCount"),
                    coord.matches.count
                ))
                .font(.caption)
                .foregroundStyle(.secondary)
                Spacer()
                // Sheet-level commit. Fires `commitHandEdit` for
                // every row currently in `stagedReplacements`,
                // serially, then refreshes the match list. Gated
                // on at least one staged row.
                Button {
                    coord.commitAll(recorder: recorder)
                } label: {
                    Label(
                        String(localized: "searchReplace.commitAll"),
                        systemImage: "checkmark.circle.fill"
                    )
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .disabled(!coord.canCommitAll)
            }
        }
        .padding(20)
    }

    @ViewBuilder
    private var content: some View {
        if coord.matches.isEmpty {
            emptyView
        } else {
            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(coord.matches, id: \.id) { utterance in
                        MatchCard(
                            utterance: utterance,
                            recorder: recorder,
                            coord: coord,
                            focusedEditorID: $focusedEditorID
                        )
                    }
                }
                .padding(20)
            }
        }
    }

    @ViewBuilder
    private var emptyView: some View {
        VStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text(emptyMessage)
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyMessage: String {
        if coord.trimmedSearch.isEmpty {
            return String(localized: "searchReplace.empty.noTerm")
        }
        return String(localized: "searchReplace.empty.noMatch")
    }

}

/// One row in the find-and-replace match list. Extracted from
/// `SearchReplaceSheet` so it can own per-row `@State` for the
/// TextEditor's selection — without per-instance state the
/// Annotate button couldn't reliably read the cursor position
/// for the right editor.
private struct MatchCard: View {
    let utterance: UtteranceEstimate
    let recorder: RecordingController
    let coord: SearchReplaceCoordinator
    var focusedEditorID: FocusState<UUID?>.Binding

    /// Two-way binding for the inline TextEditor's cursor /
    /// selection. iOS 18+ `TextEditor(text:selection:)` keeps this
    /// in sync with the actual native UITextView selection so the
    /// Annotate button can read the current cursor position and
    /// insert at the right offset.
    @State private var textSelection: TextSelection?

    var body: some View {
        let staged = coord.staged(for: utterance.id)
        let displayedText = staged ?? utterance.transcript
        let matchRanges = coord.rawMatches(in: displayedText)
        let selected = coord.selection(for: utterance.id)
        let audioEditingEnabled = recorder.playbackSourceURL != nil
        let isThisPlaying = recorder.isPreviewPlaying
            && recorder.playingUtteranceID == utterance.id
        let original = utterance.transcript
        let similarRanges = staged == nil
            ? coord.similarMatchRanges(for: utterance)
            : []
        VStack(alignment: .leading, spacing: 10) {
            cardHeader(
                staged: staged,
                audioEditingEnabled: audioEditingEnabled,
                isThisPlaying: isThisPlaying
            )
            // Read-only Text on top with raw matches in
            // yellow/green (tappable links + selection) and any
            // similar / cross-script regions in purple. Purple
            // highlights only show pre-stage — once the user
            // stages a change, the original-text indices no
            // longer line up with the displayed text, so we
            // suppress them to avoid pointing at the wrong
            // characters.
            Text(SearchReplaceHighlighter.attributed(
                utteranceID: utterance.id,
                text: displayedText,
                replaceTerm: coord.replaceTerm,
                matchRanges: matchRanges,
                similarRanges: similarRanges,
                selectedIndices: selected,
                replaced: staged != nil
            ))
            .font(.body)
            .tint(.primary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color(uiColor: .secondarySystemBackground))
            )
            .textSelection(.enabled)
            // Editable TextEditor — bound through
            // `setManualStaged`, always present so the user can
            // manually fix anything the Replace button can't
            // (including regions inside a raw-matched row that
            // need word-level surgery rather than substring swap).
            // `selection:` binding feeds the Annotate insertion
            // point; `focused` ties this editor to the sheet's
            // shared `focusedEditorID` so the Annotate button only
            // fires on the row whose editor currently holds focus.
            let editBinding = Binding<String>(
                get: { coord.staged(for: utterance.id) ?? original },
                set: { coord.setManualStaged($0, for: utterance.id, original: original) }
            )
            TextEditor(text: editBinding, selection: $textSelection)
                .focused(focusedEditorID, equals: utterance.id)
                .font(.body)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 72)
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color(uiColor: .secondarySystemBackground))
                )
            if matchRanges.count > 1 && staged == nil {
                selectionControls(
                    matchCount: matchRanges.count,
                    selectedCount: selected.count
                )
            }
            actionRow(
                staged: staged,
                matchCount: matchRanges.count,
                selectedCount: selected.count
            )
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    @ViewBuilder
    private func cardHeader(
        staged: String?,
        audioEditingEnabled: Bool,
        isThisPlaying: Bool
    ) -> some View {
        HStack(spacing: 8) {
            Text(utterance.speakerID)
                .font(.caption.monospaced())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(
                    speakerTint(for: utterance.speakerID).opacity(0.15),
                    in: Capsule()
                )
                .foregroundStyle(speakerTint(for: utterance.speakerID))
            Text(String(format: "%.1fs", utterance.start))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
            Spacer(minLength: 4)
            if staged != nil {
                Text(String(localized: "searchReplace.staged"))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.green)
            } else if coord.isSimilarMatch(utterance) {
                Text(String(localized: "searchReplace.similar"))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.purple)
            } else if !coord.hasRawMatch(utterance) {
                Text(String(localized: "searchReplace.crossScript"))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.orange)
            }
            if audioEditingEnabled {
                Button {
                    if isThisPlaying {
                        recorder.stopPlayback()
                    } else {
                        recorder.playRange(
                            start: utterance.start,
                            end: utterance.end,
                            owner: utterance.id
                        )
                    }
                } label: {
                    Image(systemName: isThisPlaying
                        ? "stop.circle.fill"
                        : "play.circle.fill"
                    )
                        .font(.title2)
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(Color.accentColor)
                }
                .buttonStyle(.borderless)
            }
        }
    }

    @ViewBuilder
    private func selectionControls(
        matchCount: Int,
        selectedCount: Int
    ) -> some View {
        HStack(spacing: 8) {
            Text(String.localizedStringWithFormat(
                String(localized: "searchReplace.selectedCount"),
                selectedCount,
                matchCount
            ))
            .font(.caption)
            .foregroundStyle(.secondary)
            Spacer()
            Button(String(localized: "searchReplace.selectAll")) {
                coord.setSelection(Set(0..<matchCount), for: utterance.id)
            }
            .disabled(selectedCount == matchCount)
            Button(String(localized: "searchReplace.clear")) {
                coord.setSelection([], for: utterance.id)
            }
            .disabled(selectedCount == 0)
        }
        .controlSize(.mini)
        .buttonStyle(.borderless)
    }

    @ViewBuilder
    private func actionRow(
        staged: String?,
        matchCount: Int,
        selectedCount: Int
    ) -> some View {
        // Replace splits into two modes depending on what kind of
        // match this row carries:
        //
        // - Raw substring present → standard "Replace" / "Replace all"
        //   / "Replace selected (n)" against the raw matches in the
        //   displayed text. Enabled gate is `canStageReplace`.
        //
        // - No raw substring but the per-token similar pass found
        //   at least one chunk → "Replace Anyway" replaces those
        //   chunks in the ORIGINAL transcript with the replace
        //   term, regardless of any current staging. Lets the user
        //   one-shot a swap on a cross-script / fuzzy hit instead
        //   of editing the transcript by hand.
        let similarRanges = coord.similarMatchRanges(for: utterance)
        let canRaw = coord.canStageReplace(for: utterance)
        let useAnyway = !canRaw && !similarRanges.isEmpty
        let replaceLabel: String = {
            if useAnyway {
                return String(localized: "searchReplace.replaceAnyway")
            }
            if selectedCount > 0 {
                return String.localizedStringWithFormat(
                    String(localized: "searchReplace.replaceSelected"),
                    selectedCount
                )
            }
            if matchCount > 1 {
                return String(localized: "searchReplace.replaceAll")
            }
            return String(localized: "searchReplace.replace")
        }()
        // Annotate enabled iff THIS row's editor currently holds
        // focus AND a search term exists — the action inserts
        // "(searchTerm)" at the cursor of the focused editor, so
        // both conditions are required for a meaningful insert.
        let annotateEnabled = focusedEditorID.wrappedValue == utterance.id
            && !coord.trimmedSearch.isEmpty
        HStack(spacing: 8) {
            Spacer()
            Button {
                annotate()
            } label: {
                Label(
                    String(localized: "searchReplace.annotate"),
                    systemImage: "text.insert"
                )
            }
            .buttonStyle(.bordered)
            .disabled(!annotateEnabled)
            Button {
                if useAnyway {
                    coord.stageReplaceAnyway(for: utterance)
                } else {
                    coord.stageReplace(for: utterance)
                }
            } label: {
                Label(replaceLabel, systemImage: "arrow.triangle.2.circlepath")
            }
            .buttonStyle(.bordered)
            .disabled(!(canRaw || useAnyway))
            Button {
                coord.commit(for: utterance, recorder: recorder)
            } label: {
                Label(
                    String(localized: "searchReplace.commit"),
                    systemImage: "checkmark"
                )
                .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderedProminent)
            .disabled(staged == nil)
        }
        .controlSize(.small)
    }

    /// Wrap the current search term in parentheses and insert it
    /// at the editor's cursor position. When the user has a
    /// non-empty selection inside the editor, the insertion
    /// replaces that range — the standard "type to overwrite
    /// selection" behavior. Falls back to appending at the end
    /// when `textSelection` doesn't have a usable single range
    /// (e.g. a multi-selection or nil). Cursor advances to right
    /// after the inserted text.
    private func annotate() {
        let term = coord.trimmedSearch
        guard !term.isEmpty else { return }
        let insertion = "(\(term))"
        let original = utterance.transcript
        var text = coord.staged(for: utterance.id) ?? original

        let insertAt: Range<String.Index>
        if case .selection(let range) = textSelection?.indices {
            // Clamp the cached range to the current text — if the
            // user typed since the selection was captured, the
            // indices may point past the new end.
            let lower = min(range.lowerBound, text.endIndex)
            let upper = min(range.upperBound, text.endIndex)
            insertAt = lower..<upper
        } else {
            insertAt = text.endIndex..<text.endIndex
        }
        text.replaceSubrange(insertAt, with: insertion)
        coord.setManualStaged(text, for: utterance.id, original: original)
        // Position the cursor right after the inserted text.
        // `offsetBy: insertion.count` is in Character count which
        // matches both String.Index advancement and TextSelection's
        // String.Index domain.
        let newCursor = text.index(
            insertAt.lowerBound,
            offsetBy: insertion.count,
            limitedBy: text.endIndex
        ) ?? text.endIndex
        textSelection = TextSelection(insertionPoint: newCursor)
    }
}
