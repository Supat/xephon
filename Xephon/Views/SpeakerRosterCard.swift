import SwiftUI
import Diarization
import SERAcoustic
import Fusion

/// Read-only directory of every speaker the session knows about and
/// the user-supplied display name (if any) attached to each one.
/// Lives on the left pane's cluster-diagnostics page below the
/// heatmap so the three views read together: "this is the cloud,
/// this is how close each pair is, this is who each id maps to."
///
/// Roster is the union of (a) ids referenced by at least one
/// utterance and (b) ids in the diarizer's internal speaker DB —
/// the latter covers entries FluidAudio retained from the
/// streaming pass that haven't been claimed by a row yet, so the
/// user can see them before deciding what to do with them.
struct SpeakerRosterCard: View {
    let recorder: RecordingController
    let cluster: SpeakerClusterSnapshot
    /// Speaker id of the currently-focused utterance, or nil.
    /// Drives a soft background tint on the matching row so the
    /// roster, cluster scatter, and heatmap all light up in sync
    /// when the user is inspecting a row.
    var highlightedSpeakerID: String?
    /// Speaker ids that are referenced by at least one utterance.
    /// Drives the "Linked only" toggle: when enabled, rows for
    /// speaker ids absent from this set (diarizer-DB orphans,
    /// promoted-but-unclaimed entries) are hidden so the roster
    /// reads as the conversation's active cast. Nil disables the
    /// toggle — nothing meaningful to filter against.
    var linkedSpeakerIDs: Set<String>?

    /// Header toggle: hide speaker rows whose id isn't in
    /// `linkedSpeakerIDs`. Styled to match the cluster card's
    /// matching toggle.
    @State private var hideUnreferencedSpeakers: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text(String(localized: "roster.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Spacer(minLength: 4)
                // Same caption2 link-icon button the cluster card
                // uses. Only surfaces when there's at least one
                // unreferenced speaker the toggle would actually
                // hide — otherwise it'd be a no-op control.
                if let linked = linkedSpeakerIDs,
                   allSpeakerIDs.contains(where: { !linked.contains($0) }) {
                    Button {
                        hideUnreferencedSpeakers.toggle()
                    } label: {
                        Label(
                            String(localized: "cluster.scatter.linkedOnly"),
                            systemImage: hideUnreferencedSpeakers
                                ? "link.circle.fill"
                                : "link.circle"
                        )
                        .font(.caption2)
                        .labelStyle(.titleAndIcon)
                        .lineLimit(1)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(
                        hideUnreferencedSpeakers
                            ? AnyShapeStyle(Color.accentColor)
                            : AnyShapeStyle(HierarchicalShapeStyle.secondary)
                    )
                }
                if !speakerIDs.isEmpty {
                    Text("\(speakerIDs.count)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }
            if speakerIDs.isEmpty {
                Text(String(localized: "roster.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(speakerIDs, id: \.self) { id in
                        row(for: id)
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    /// Sorted union of utterance-roster ids and cluster-DB ids,
    /// before any "Linked only" filtering. Sort is alphabetical,
    /// which for `S0N` ids is also numeric; no point overthinking
    /// ordering when the keys are zero-padded 2-digit suffixes.
    /// The header's toggle-visibility check reads this so it can
    /// decide whether anything would actually get hidden.
    private var allSpeakerIDs: [String] {
        var seen: Set<String> = []
        var ids: [String] = []
        for id in recorder.knownSpeakerIDs() where seen.insert(id).inserted {
            ids.append(id)
        }
        for spk in cluster.speakers where seen.insert(spk.id).inserted {
            ids.append(spk.id)
        }
        return ids.sorted()
    }

    /// What the body actually renders. Applies the "Linked only"
    /// filter when the toggle is on and the controller has supplied
    /// a referenced-id set; otherwise hands back the full union.
    private var speakerIDs: [String] {
        guard hideUnreferencedSpeakers,
              let linked = linkedSpeakerIDs else { return allSpeakerIDs }
        return allSpeakerIDs.filter { linked.contains($0) }
    }

    @ViewBuilder
    private func row(for id: String) -> some View {
        let demo = demographics[id]
        let isHighlighted = highlightedSpeakerID == id
        HStack(spacing: 8) {
            Text(id)
                .font(.caption.monospaced())
                .foregroundStyle(speakerTint(for: id))
                .frame(width: 40, alignment: .leading)
            if let name = recorder.speakerDisplayName(forStored: id),
               !name.isEmpty {
                Text(name)
                    .font(.caption)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            } else {
                Text("—")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            if let gender = demo?.majorityGender {
                genderChip(gender)
            }
            if let range = demo?.ageRangeYears {
                Text(Self.formatAgeRange(range))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
            // Per-speaker utterance count anchored to the trailing
            // edge so the roster reads "id · name · demographics …
            // n" left-to-right. Zero counts surface speakers that
            // are in the diarizer DB but haven't been claimed by a
            // row yet — useful diagnostic before the user decides
            // whether to prune them.
            Text("\(utteranceCounts[id, default: 0])")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 2)
        .background(
            // Tinted background only when the row's speaker matches
            // the currently-focused utterance. Uses that speaker's
            // own tint at low opacity so the highlight reads in the
            // same color story the rest of the cluster diagnostics
            // pages use.
            RoundedRectangle(cornerRadius: 4, style: .continuous)
                .fill(
                    isHighlighted
                        ? speakerTint(for: id).opacity(0.18)
                        : Color.clear
                )
        )
    }

    /// Utterance count per speaker id, computed once per render
    /// from `recorder.utterances`. Speakers not present in any
    /// utterance read as 0 — useful for spotting diarizer-DB
    /// orphans that the user can prune.
    private var utteranceCounts: [String: Int] {
        var counts: [String: Int] = [:]
        for utt in recorder.utterances {
            counts[utt.speakerID, default: 0] += 1
        }
        return counts
    }

    /// Per-row demographics from the W2V2 age-gender model. Built
    /// once per render by walking `recorder.utterances`. Speakers
    /// with no age-gender data (model not loaded, or only too-short
    /// clips) are absent from the map — the row falls back to id +
    /// name only.
    private var demographics: [String: SpeakerDemographics] {
        var votes: [String: [AgeGenderEstimate.Gender: Int]] = [:]
        var ages: [String: (min: Float, max: Float)] = [:]
        for utt in recorder.utterances {
            guard let ag = utt.ageGender else { continue }
            let speaker = utt.speakerID
            if let top = ag.topGender {
                votes[speaker, default: [:]][top, default: 0] += 1
            }
            let years = ag.ageYears
            if let existing = ages[speaker] {
                ages[speaker] = (
                    min: min(existing.min, years),
                    max: max(existing.max, years)
                )
            } else {
                ages[speaker] = (min: years, max: years)
            }
        }
        var out: [String: SpeakerDemographics] = [:]
        let allSpeakers = Set(votes.keys).union(ages.keys)
        for speaker in allSpeakers {
            // Plurality vote, with CaseIterable order as the
            // deterministic tiebreaker so the chip doesn't flicker
            // between re-renders when two classes share the lead.
            let majority = votes[speaker].flatMap { tally in
                AgeGenderEstimate.Gender.allCases.max { a, b in
                    (tally[a] ?? 0) < (tally[b] ?? 0)
                }.flatMap { winner in
                    (tally[winner] ?? 0) > 0 ? winner : nil
                }
            }
            let range: ClosedRange<Float>? = ages[speaker].map { $0.min ... $0.max }
            out[speaker] = SpeakerDemographics(
                majorityGender: majority,
                ageRangeYears: range
            )
        }
        return out
    }

    /// "32" if min==max (single observation or zero spread), else
    /// "32–48" with an en-dash. Years are rounded to the nearest
    /// integer because the model's regression resolution is much
    /// coarser than a decimal point would imply.
    private static func formatAgeRange(_ range: ClosedRange<Float>) -> String {
        let lo = Int(range.lowerBound.rounded())
        let hi = Int(range.upperBound.rounded())
        if lo == hi {
            return "\(lo)"
        }
        return "\(lo)\u{2013}\(hi)"
    }

    @ViewBuilder
    private func genderChip(_ gender: AgeGenderEstimate.Gender) -> some View {
        let tint = Self.genderTint(gender)
        Text(Self.genderInitial(gender))
            .font(.caption2.monospaced().bold())
            .foregroundStyle(tint)
            .frame(width: 14, height: 14)
            .background(tint.opacity(0.18), in: Circle())
            .accessibilityLabel(Text(Self.genderAccessibilityLabel(gender)))
    }

    /// Single-letter label so the chip stays compact at the right
    /// edge of a narrow row. Localized initials live in
    /// `Localizable.strings` so the chip reads ja/en consistently.
    private static func genderInitial(_ gender: AgeGenderEstimate.Gender) -> String {
        switch gender {
        case .female: return String(localized: "roster.gender.female.initial")
        case .male: return String(localized: "roster.gender.male.initial")
        case .child: return String(localized: "roster.gender.child.initial")
        }
    }

    private static func genderAccessibilityLabel(_ gender: AgeGenderEstimate.Gender) -> String {
        switch gender {
        case .female: return String(localized: "roster.gender.female")
        case .male: return String(localized: "roster.gender.male")
        case .child: return String(localized: "roster.gender.child")
        }
    }

    /// Distinct hues per class. Female / male use the conventional
    /// pink/blue (legible cross-culturally and accessible-enough
    /// against the card background); child uses a neutral teal to
    /// dodge the gender-binary implication.
    private static func genderTint(_ gender: AgeGenderEstimate.Gender) -> Color {
        switch gender {
        case .female: return Color(red: 0.93, green: 0.30, blue: 0.55)
        case .male: return Color(red: 0.20, green: 0.45, blue: 0.95)
        case .child: return Color(red: 0.10, green: 0.65, blue: 0.55)
        }
    }
}

private struct SpeakerDemographics {
    let majorityGender: AgeGenderEstimate.Gender?
    let ageRangeYears: ClosedRange<Float>?
}
