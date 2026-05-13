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
///   1. **Replace** stages the substitution into `stagedReplacements[id]`.
///      The text field reflects the staged result, with the post-
///      replace term tinted green so the user can verify the edit
///      before committing.
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
struct SearchReplaceSheet: View {
    let recorder: RecordingController
    let onDismiss: () -> Void

    @State private var searchTerm: String = ""
    @State private var replaceTerm: String = ""
    /// Staged replacement text per utterance. Populated by tapping
    /// Replace on a row; consumed by Commit which calls
    /// `commitHandEdit` and clears the entry. Survives view re-
    /// renders so the SER pipeline updating the row underneath
    /// doesn't blow away pending stages.
    @State private var stagedReplacements: [UUID: String] = [:]
    /// Per-utterance set of match indices the user has picked for
    /// replacement. Empty (or absent) means "no explicit selection"
    /// — the Replace button treats that as "replace every match in
    /// this row", which is what most users want when there's only
    /// one match anyway. Indices reset after each Replace pass
    /// because the staged text's match positions no longer line up
    /// with the pre-staging ones.
    @State private var selectedMatches: [UUID: Set<Int>] = [:]

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
            .navigationBarTitleDisplayMode(.inline)
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
                handleMatchURL(url)
            })
        }
    }

    private func handleMatchURL(_ url: URL) -> OpenURLAction.Result {
        guard url.scheme == Self.matchURLScheme,
              let components = URLComponents(
                url: url, resolvingAgainstBaseURL: false
              ),
              let utteranceIDString = components.queryItems?
                .first(where: { $0.name == "u" })?.value,
              let utteranceID = UUID(uuidString: utteranceIDString),
              let indexString = components.queryItems?
                .first(where: { $0.name == "i" })?.value,
              let index = Int(indexString) else {
            return .systemAction
        }
        var set = selectedMatches[utteranceID] ?? []
        if set.contains(index) {
            set.remove(index)
        } else {
            set.insert(index)
        }
        selectedMatches[utteranceID] = set
        return .handled
    }

    private static let matchURLScheme = "xephon-match"

    @ViewBuilder
    private var searchFields: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .frame(width: 18)
                TextField(
                    String(localized: "searchReplace.search.placeholder"),
                    text: $searchTerm
                )
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            }
            HStack(spacing: 8) {
                Image(systemName: "arrow.right.circle")
                    .foregroundStyle(.secondary)
                    .frame(width: 18)
                TextField(
                    String(localized: "searchReplace.replace.placeholder"),
                    text: $replaceTerm
                )
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            }
            HStack {
                Text(String.localizedStringWithFormat(
                    String(localized: "searchReplace.matchCount"),
                    matches.count
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
        if matches.isEmpty {
            emptyView
        } else {
            ScrollView(.vertical, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(matches, id: \.id) { utterance in
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
        if trimmedSearch.isEmpty {
            return String(localized: "searchReplace.empty.noTerm")
        }
        return String(localized: "searchReplace.empty.noMatch")
    }

    @ViewBuilder
    private func matchCard(_ utterance: UtteranceEstimate) -> some View {
        let staged = stagedReplacements[utterance.id]
        let displayedText = staged ?? utterance.transcript
        let matchRanges = rawMatches(in: displayedText, term: trimmedSearch)
        let selected = selectedMatches[utterance.id] ?? []
        VStack(alignment: .leading, spacing: 10) {
            cardHeader(utterance: utterance, staged: staged)
            // Non-editable transcript with each match highlighted
            // and individually tappable. The custom `xephon-match`
            // URL scheme on each match routes the tap through the
            // sheet's openURL handler to toggle selection.
            Text(highlightedTranscript(
                utterance: utterance,
                text: displayedText,
                replaced: staged != nil,
                matchRanges: matchRanges,
                selectedIndices: selected
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
        staged: String?
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
            } else if !hasRawMatch(utterance) {
                Text(String(localized: "searchReplace.crossScript"))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.orange)
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
                selectedMatches[utterance.id] = Set(0..<matchCount)
            }
            .disabled(selectedCount == matchCount)
            Button(String(localized: "searchReplace.clear")) {
                selectedMatches[utterance.id] = []
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
                stageReplace(for: utterance)
            } label: {
                Label(replaceLabel, systemImage: "arrow.triangle.2.circlepath")
            }
            .buttonStyle(.bordered)
            .disabled(!canStageReplace(for: utterance))
            Button {
                commit(for: utterance)
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

    // MARK: - Matching + staging

    private var trimmedSearch: String {
        searchTerm.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Utterances that match the query under the same cross-script
    /// normalization the transcript list uses — kanji ↔ kana ↔
    /// romaji all collapse to the same Hepburn key, so typing
    /// "shibuya" finds 渋谷 and vice versa. Recomputed from
    /// `recorder.utterances`, so a row that's been committed drops
    /// out automatically once its transcript no longer matches.
    private var matches: [UtteranceEstimate] {
        let term = trimmedSearch
        guard !term.isEmpty else { return [] }
        let normalizedQuery = JapaneseSearchNormalizer.normalize(term)
        guard !normalizedQuery.isEmpty else { return [] }
        return recorder.utterances.filter { u in
            // Raw substring is the fast path and covers the
            // common case where the user types the same script
            // the transcript is in. Normalized fallback covers
            // cross-script hits.
            if u.transcript.localizedStandardRange(of: term) != nil {
                return true
            }
            return JapaneseSearchNormalizer
                .normalize(u.transcript)
                .contains(normalizedQuery)
        }
    }

    /// True when this utterance contains the search term as a raw
    /// case-insensitive substring (the path that supports
    /// highlighting + Replace). False when the only reason the row
    /// surfaced is the cross-script normalized match — in which
    /// case Replace is disabled and the card shows a hint.
    private func hasRawMatch(_ utterance: UtteranceEstimate) -> Bool {
        let term = trimmedSearch
        guard !term.isEmpty else { return false }
        return utterance.transcript.localizedStandardRange(of: term) != nil
    }

    private func canStageReplace(for utterance: UtteranceEstimate) -> Bool {
        guard !trimmedSearch.isEmpty else { return false }
        let displayed = stagedReplacements[utterance.id] ?? utterance.transcript
        // Two reasons Replace stays disabled: (1) the displayed
        // text no longer contains the search term as a literal
        // substring (already replaced, or the row only matched via
        // cross-script normalization so we never had a raw range
        // to operate on); (2) trivially, the search is empty.
        return displayed.localizedStandardRange(of: trimmedSearch) != nil
    }

    private func stageReplace(for utterance: UtteranceEstimate) {
        let term = trimmedSearch
        guard !term.isEmpty else { return }
        let source = stagedReplacements[utterance.id] ?? utterance.transcript
        let matches = rawMatches(in: source, term: term)
        guard !matches.isEmpty else { return }
        let selected = selectedMatches[utterance.id] ?? []
        // Empty selection = "replace every match", which matches
        // the no-friction default the user expects when there's
        // only one match. Non-empty selection = swap only those.
        let indicesToReplace: Set<Int> = selected.isEmpty
            ? Set(0..<matches.count)
            : selected.intersection(0..<matches.count)
        guard !indicesToReplace.isEmpty else { return }
        // Replace from highest index to lowest so each earlier
        // range stays valid as we mutate — `String.Index` would
        // otherwise dangle past an in-place mutation.
        var result = source
        for idx in indicesToReplace.sorted(by: >) {
            result.replaceSubrange(matches[idx], with: replaceTerm)
        }
        guard result != source else { return }
        stagedReplacements[utterance.id] = result
        // Clear the selection — its indices no longer correspond
        // to anything in the staged text. The next render will
        // recompute matches against the new source.
        selectedMatches[utterance.id] = []
    }

    private func commit(for utterance: UtteranceEstimate) {
        guard let staged = stagedReplacements[utterance.id] else { return }
        let id = utterance.id
        let start = utterance.start
        let end = utterance.end
        Task {
            await recorder.commitHandEdit(
                utteranceID: id,
                newText: staged,
                newStart: start,
                newEnd: end
            )
            stagedReplacements.removeValue(forKey: id)
        }
    }

    /// Enumerate every case-insensitive occurrence of `term` inside
    /// `text` as raw `String.Index` ranges. The cursor advances by
    /// `range.upperBound` so overlapping matches are skipped (which
    /// also stops the loop from spinning when `term` is empty
    /// inside `replaceSubrange`).
    private func rawMatches(
        in text: String,
        term: String
    ) -> [Range<String.Index>] {
        guard !term.isEmpty else { return [] }
        var out: [Range<String.Index>] = []
        var cursor = text.startIndex
        while cursor < text.endIndex,
              let range = text.range(
                of: term,
                options: .caseInsensitive,
                range: cursor..<text.endIndex
              ) {
            out.append(range)
            cursor = range.upperBound
        }
        return out
    }

    // MARK: - Highlight rendering

    /// Build an AttributedString tinting each match by selection
    /// state. Pre-replace renders attach a `xephon-match://` link
    /// to each match so the openURL handler can toggle selection
    /// on tap; post-replace renders skip the links because the
    /// staged text is shown read-only with the inserted replace
    /// term highlighted in green.
    private func highlightedTranscript(
        utterance: UtteranceEstimate,
        text: String,
        replaced: Bool,
        matchRanges: [Range<String.Index>],
        selectedIndices: Set<Int>
    ) -> AttributedString {
        var attributed = AttributedString(text)
        // Post-replace path: highlight every occurrence of the
        // replace term so the user sees what landed.
        if replaced {
            let needle = replaceTerm
            guard !needle.isEmpty else { return attributed }
            let bg = Color.green.opacity(0.35)
            var cursor = text.startIndex
            while cursor < text.endIndex,
                  let r = text.range(
                    of: needle,
                    options: .caseInsensitive,
                    range: cursor..<text.endIndex
                  ) {
                if let attrRange = Range(r, in: attributed) {
                    attributed[attrRange].backgroundColor = bg
                }
                cursor = r.upperBound
            }
            return attributed
        }
        // Pre-replace path: each search-term match gets a tappable
        // link. Selected matches read green; unselected stay yellow.
        for (idx, range) in matchRanges.enumerated() {
            guard let attrRange = Range(range, in: attributed) else { continue }
            let isSelected = selectedIndices.contains(idx)
            attributed[attrRange].backgroundColor = isSelected
                ? Color.green.opacity(0.55)
                : Color.yellow.opacity(0.55)
            attributed[attrRange].link = Self.matchURL(
                utteranceID: utterance.id, index: idx
            )
            // Override link foreground so it doesn't paint blue —
            // we want the text to stay primary-tinted, with only
            // the background telling the user this is a hot zone.
            attributed[attrRange].foregroundColor = .primary
        }
        return attributed
    }

    /// Build the `xephon-match://m?u=<uuid>&i=<index>` URL the
    /// attributed-string link points at. Encoded via
    /// `URLComponents` rather than a string-concat so the UUID's
    /// hyphens (and any future query item additions) stay
    /// percent-safe.
    private static func matchURL(utteranceID: UUID, index: Int) -> URL? {
        var components = URLComponents()
        components.scheme = matchURLScheme
        components.host = "m"
        components.queryItems = [
            URLQueryItem(name: "u", value: utteranceID.uuidString),
            URLQueryItem(name: "i", value: String(index)),
        ]
        return components.url
    }
}
