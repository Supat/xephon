import SwiftUI
import Fusion

/// Two-row filter header above the transcript list: a search field
/// plus a label-filter menu, with a speaker chip strip beneath
/// (visible only when ≥2 distinct speakers exist).
///
/// All filter state lives on `TranscriptFilterModel`; this view
/// just binds the model's `@Observable` properties via `@Bindable`
/// and routes the search-field focus binding back to ContentView
/// so the ⌘F menu command can move focus into the field.
struct TranscriptFilterBar: View {
    let recorder: RecordingController
    @Bindable var model: TranscriptFilterModel
    var searchFieldFocused: FocusState<Bool>.Binding

    var body: some View {
        VStack(spacing: 0) {
            filterBar
            speakerChipBar
        }
    }

    // MARK: - Filter bar

    /// Inline filter row: a free-text search field plus a label
    /// dropdown. Both filters AND together so the user can scope by
    /// "happy utterances containing 楽しい" etc. The bar always shows
    /// when the full list is non-empty so the controls don't appear
    /// then disappear as utterances arrive.
    @ViewBuilder
    private var filterBar: some View {
        HStack(spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField(
                    String(localized: "filter.search.placeholder"),
                    text: $model.searchText
                )
                .textFieldStyle(.plain)
                .font(.callout)
                .submitLabel(.search)
                .focused(searchFieldFocused)
                // Esc inside the search field returns focus to the
                // surrounding view so subsequent ⌘ shortcuts and
                // arrow-key list navigation work without an extra tap.
                // Returning `.handled` keeps the keystroke from
                // bubbling to the List's own Esc handler (which would
                // otherwise also clear the row selection).
                .onKeyPress(.escape) {
                    if searchFieldFocused.wrappedValue {
                        searchFieldFocused.wrappedValue = false
                        return .handled
                    }
                    return .ignored
                }
                if !model.searchText.isEmpty {
                    Button {
                        model.searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))

            // Label picker. Pulled into a Menu so the chevron looks
            // native and tinted-by-label rows read at a glance. Tap
            // expands to a list of labels with their per-label counts;
            // "All labels" clears the filter.
            Menu {
                Button {
                    model.selectedLabelFilter = nil
                } label: {
                    if model.selectedLabelFilter == nil {
                        Label(String(localized: "filter.label.all"), systemImage: "checkmark")
                    } else {
                        Text(String(localized: "filter.label.all"))
                    }
                }
                let labels = model.availableLabels(in: recorder.utterances)
                if !labels.isEmpty {
                    Divider()
                }
                ForEach(labels, id: \.self) { label in
                    Button {
                        model.selectedLabelFilter = label
                    } label: {
                        let displayed = label.capitalized(with: Locale(identifier: "en_US"))
                        let count = recorder.conversationSummary.labelCounts[label] ?? 0
                        if model.selectedLabelFilter == label {
                            Label("\(displayed) (\(count))", systemImage: "checkmark")
                        } else {
                            Text("\(displayed) (\(count))")
                        }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "line.3.horizontal.decrease.circle")
                    Text(model.filterLabelDisplay)
                        .lineLimit(1)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2)
                }
                .font(.callout)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .foregroundStyle(model.selectedLabelFilter.map(emotionTint(for:)) ?? Color.primary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - Speaker chip bar

    /// Horizontal chip row of every speaker seen this session, plus
    /// an "All" chip on the left that clears the speaker filter.
    /// Tapping a speaker chip toggles it as the active speaker
    /// filter — tapping the same chip again returns to "All". Hidden
    /// when only one speaker has been detected (no point filtering
    /// when there's nothing to choose between).
    @ViewBuilder
    private var speakerChipBar: some View {
        let speakers = model.availableSpeakers(in: recorder.utterances)
        if speakers.count >= 2 {
            HStack(spacing: 8) {
                ScrollView(.horizontal, showsIndicators: false) {
                    // Wrap the chip strip in a `GlassEffectContainer`
                    // so adjacent glass capsules merge their blurs
                    // and the morph feels fluid when a speaker is
                    // added/removed.
                    GlassEffectContainer(spacing: 8) {
                        HStack(spacing: 8) {
                            speakerChip(
                                text: String(localized: "filter.speaker.all"),
                                tint: .secondary,
                                isSelected: model.selectedSpeakerFilter == nil
                                    && !model.showingMismatchOnly
                            ) {
                                model.selectedSpeakerFilter = nil
                                model.showingMismatchOnly = false
                            }
                            // Mismatch chip — visible only when the
                            // cumulative timeline disagrees with at
                            // least one stored speaker. Mutually
                            // exclusive with the per-speaker chips
                            // since "all speakers with mismatches" is
                            // the meaningful narrowing; a single
                            // speaker's mismatched rows would be a
                            // narrower-still slice nobody asked for.
                            if !model.mismatchedUtteranceIDs(in: recorder).isEmpty {
                                speakerChip(
                                    text: String(localized: "filter.speaker.mismatch"),
                                    tint: .orange,
                                    isSelected: model.showingMismatchOnly
                                ) {
                                    if model.showingMismatchOnly {
                                        model.showingMismatchOnly = false
                                    } else {
                                        model.showingMismatchOnly = true
                                        model.selectedSpeakerFilter = nil
                                    }
                                }
                            }
                            ForEach(speakers, id: \.self) { id in
                                let label = formatSpeakerLabel(
                                    id,
                                    customName: recorder.speakerDisplayName(forStored: id)
                                )
                                let tint = speakerTint(for: id)
                                speakerChip(
                                    text: label,
                                    tint: tint,
                                    isSelected: model.selectedSpeakerFilter == id
                                ) {
                                    model.selectedSpeakerFilter =
                                        (model.selectedSpeakerFilter == id) ? nil : id
                                    model.showingMismatchOnly = false
                                }
                            }
                        }
                    }
                    .padding(.leading, 12)
                    .padding(.vertical, 6)
                }
                // Distinct speaker count, pinned to the trailing edge
                // so the chip strip can scroll horizontally beneath it
                // without pushing the count out of view.
                Label("\(speakers.count)", systemImage: "person.2.fill")
                    .font(.caption.monospacedDigit())
                    .labelStyle(.titleAndIcon)
                    .foregroundStyle(.secondary)
                    .padding(.trailing, 12)
            }
        }
    }

    @ViewBuilder
    private func speakerChip(
        text: String,
        tint: Color,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Text(text)
                .font(.caption.bold())
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .foregroundStyle(tint)
                .glassEffect(
                    .regular
                        .tint(tint.opacity(isSelected ? 0.55 : 0.2))
                        .interactive(),
                    in: Capsule()
                )
        }
        .buttonStyle(.plain)
    }
}

/// "No matches" placeholder shown in place of the transcript list
/// when every utterance has been filtered out. Lives next to the
/// filter bar because its "Clear filters" button writes the same
/// filter knobs the bar drives.
struct TranscriptNoMatchesView: View {
    @Bindable var model: TranscriptFilterModel
    /// Keyword selection is a filter layer too (the OR filter the
    /// Keywords card drives); the clear button must release it or a
    /// keyword-only no-match state can't be escaped from here.
    let keywords: KeywordStore

    var body: some View {
        ContentUnavailableView {
            Label(
                String(localized: "filter.noMatches.title"),
                systemImage: "line.3.horizontal.decrease.circle"
            )
        } description: {
            Text(String(localized: "filter.noMatches.subtitle"))
        } actions: {
            Button(String(localized: "filter.noMatches.clear")) {
                model.clearFilters(keywords: keywords)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
