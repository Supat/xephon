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
            .onAppear { coord.scheduleSearch(in: recorder) }
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
                        matchCard(utterance)
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

    @ViewBuilder
    private func matchCard(_ utterance: UtteranceEstimate) -> some View {
        let staged = coord.staged(for: utterance.id)
        let displayedText = staged ?? utterance.transcript
        let matchRanges = coord.rawMatches(in: displayedText)
        let selected = coord.selection(for: utterance.id)
        let audioEditingEnabled = recorder.playbackSourceURL != nil
        let isThisPlaying = recorder.isPreviewPlaying
            && recorder.playingUtteranceID == utterance.id
        VStack(alignment: .leading, spacing: 10) {
            cardHeader(
                utterance: utterance,
                staged: staged,
                audioEditingEnabled: audioEditingEnabled,
                isThisPlaying: isThisPlaying
            )
            // Non-editable transcript with each match highlighted
            // and individually tappable. The custom `xephon-match`
            // URL scheme on each match routes the tap through the
            // sheet's openURL handler to toggle selection.
            // Single uniform layout per card so a row can carry
            // BOTH replaceable (raw substring) AND editable
            // (cross-script / similar) matches without the UI
            // forcing one mode:
            //
            // - Read-only Text on top with raw matches in
            //   yellow/green (tappable links + selection) and any
            //   similar / cross-script regions in purple. Purple
            //   highlights only show pre-stage — once the user
            //   stages a change, the original-text indices no
            //   longer line up with the displayed text, so we
            //   suppress them to avoid pointing at the wrong
            //   characters.
            //
            // - Editable TextEditor below, bound through
            //   `setManualStaged`. Always present so the user can
            //   manually fix anything the Replace button can't —
            //   including regions inside a raw-matched row that
            //   need word-level surgery rather than substring swap.
            //
            // Replace + Commit work as before: Replace stages a
            // raw-substring substitution; Commit pushes whatever's
            // currently staged through `commitHandEdit`.
            let original = utterance.transcript
            let similarRanges = staged == nil
                ? coord.similarMatchRanges(for: utterance)
                : []
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
            let editBinding = Binding<String>(
                get: { coord.staged(for: utterance.id) ?? original },
                set: { coord.setManualStaged($0, for: utterance.id, original: original) }
            )
            TextEditor(text: editBinding)
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
                    utterance: utterance,
                    matchCount: matchRanges.count,
                    selectedCount: selected.count
                )
            }
            actionRow(
                utterance: utterance,
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
        utterance: UtteranceEstimate,
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
                // Fuzzy match — neither raw substring nor exact
                // cross-script. Distinct purple tint so the user
                // can tell at a glance that this row matched via
                // the "Include similar" pass (and that Replace
                // therefore won't work on it for the same reason
                // it doesn't on cross-script rows).
                Text(String(localized: "searchReplace.similar"))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.purple)
            } else if !coord.hasRawMatch(utterance) {
                Text(String(localized: "searchReplace.crossScript"))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.orange)
            }
            // Inline play affordance — mirrors the per-row button in
            // `TranscriptionReviewSheet`. File-mode only (mic-mode
            // sessions have no source audio to slice); `owner` pins
            // the per-row "playing" state so neighbouring cards know
            // to render the idle glyph rather than the stop glyph.
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
        utterance: UtteranceEstimate,
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
        utterance: UtteranceEstimate,
        staged: String?,
        matchCount: Int,
        selectedCount: Int
    ) -> some View {
        // Label flips between "Replace all" and "Replace selected (n)"
        // so the user knows up front what the button will do; the
        // implicit "no selection ⇒ replace all" fallback is the
        // less surprising default but worth labelling.
        let replaceLabel: String = {
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
        HStack(spacing: 8) {
            Spacer()
            Button {
                coord.stageReplace(for: utterance)
            } label: {
                Label(replaceLabel, systemImage: "arrow.triangle.2.circlepath")
            }
            .buttonStyle(.bordered)
            .disabled(!coord.canStageReplace(for: utterance))
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
}
