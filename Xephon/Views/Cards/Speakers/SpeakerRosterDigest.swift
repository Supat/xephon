import Foundation
import Diarization
import SERAcoustic

/// Pre-computed roster bundle for `SpeakerRosterCard`: the sorted
/// union of utterance-roster + diarizer-DB speaker ids, the per-
/// speaker utterance count, and the per-speaker age/gender
/// demographics distilled from `UtteranceEstimate.ageGender`.
///
/// Built once per render inside the card body (the card's `recorder`
/// + `cluster` inputs already drive re-renders on change). The
/// previous shape — three independent computed properties on the
/// view — re-folded the entire utterance list every time the per-row
/// ForEach indexed back into them, producing O(N²) work per render.
/// This struct does one pass in `init`.
struct SpeakerRosterDigest {
    /// Sorted union of utterance-roster ids and cluster-DB ids,
    /// before any "Linked only" filtering. Alphabetical sort is
    /// also numeric for the `S0N` ids the diarizer hands out.
    let allSpeakerIDs: [String]
    /// Utterance count per speaker id. Speakers not present in any
    /// utterance read as 0 — useful for spotting diarizer-DB
    /// orphans that the user can prune.
    let utteranceCounts: [String: Int]
    /// Per-speaker plurality-voted gender + observed age range,
    /// computed from `UtteranceEstimate.ageGender`. Speakers with
    /// no age-gender data (model not loaded, or only too-short
    /// clips) are absent from the map — the row falls back to
    /// id + name only.
    let demographics: [String: SpeakerDemographics]

    struct SpeakerDemographics {
        let majorityGender: AgeGenderEstimate.Gender?
        let ageRangeYears: ClosedRange<Float>?
    }

    @MainActor
    init(recorder: RecordingController, cluster: SpeakerClusterSnapshot) {
        var counts: [String: Int] = [:]
        var votes: [String: [AgeGenderEstimate.Gender: Int]] = [:]
        var ages: [String: (min: Float, max: Float)] = [:]
        for utt in recorder.utterances {
            counts[utt.speakerID, default: 0] += 1
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
        var demographics: [String: SpeakerDemographics] = [:]
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
            demographics[speaker] = SpeakerDemographics(
                majorityGender: majority,
                ageRangeYears: range
            )
        }
        var seen: Set<String> = []
        var ids: [String] = []
        for id in recorder.knownSpeakerIDs() where seen.insert(id).inserted {
            ids.append(id)
        }
        for spk in cluster.speakers where seen.insert(spk.id).inserted {
            ids.append(spk.id)
        }
        self.allSpeakerIDs = ids.sorted()
        self.utteranceCounts = counts
        self.demographics = demographics
    }
}
