import Foundation
import Fusion
import SERAcoustic

/// Per-speaker demographic digest — majority gender + observed age
/// range — built from each utterance's `ageGender` field. Both
/// summarizer backends (MLX/Qwen and Apple Foundation Models)
/// prepend a rendered version of this to their prompts so the LLM
/// can ground per-speaker descriptions in something more concrete
/// than just an opaque `S0N` id.
///
/// Identical reduction to `SpeakerRosterCard.demographics` —
/// keeping the two in sync matters: the user reads the roster card
/// and the summary side-by-side, and divergent demographics would
/// be confusing. If you change the aggregation rule here, change it
/// there too (or factor both onto this type).
public struct SpeakerDemographicsDigest: Sendable {
    public struct Entry: Sendable {
        public let speakerID: String
        public let majorityGender: AgeGenderEstimate.Gender?
        /// Observed age range in years (min ... max). Nil if no
        /// utterance for this speaker carried an `ageGender` block.
        public let ageRangeYears: ClosedRange<Float>?
    }

    public let entries: [String: Entry]

    /// Build the digest from a flat utterance list. Speakers absent
    /// from the result either don't appear in `utterances` at all,
    /// or had no row carrying age-gender output (model not loaded /
    /// clips too short to score).
    public static func build(
        from utterances: [UtteranceEstimate]
    ) -> SpeakerDemographicsDigest {
        var votes: [String: [AgeGenderEstimate.Gender: Int]] = [:]
        var ages: [String: (min: Float, max: Float)] = [:]
        for utt in utterances {
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
        var out: [String: Entry] = [:]
        let allSpeakers = Set(votes.keys).union(ages.keys)
        for speaker in allSpeakers {
            // Plurality vote with CaseIterable order as the
            // deterministic tiebreaker — matches SpeakerRosterCard.
            let majority = votes[speaker].flatMap { tally in
                AgeGenderEstimate.Gender.allCases.max { a, b in
                    (tally[a] ?? 0) < (tally[b] ?? 0)
                }.flatMap { winner in
                    (tally[winner] ?? 0) > 0 ? winner : nil
                }
            }
            let range: ClosedRange<Float>? = ages[speaker].map { $0.min ... $0.max }
            out[speaker] = Entry(
                speakerID: speaker,
                majorityGender: majority,
                ageRangeYears: range
            )
        }
        return SpeakerDemographicsDigest(entries: out)
    }

    /// Render a compact, LLM-friendly roster block. Ordered by
    /// `speakerIDs` so the model sees the same sequence the
    /// utterance list uses. Returns an empty string when no speaker
    /// has demographic data — callers can append unconditionally
    /// and the block disappears when the W2V2 age-gender model
    /// wasn't loaded.
    ///
    /// Format:
    /// ```
    /// Speaker demographics (rough — W2V2 age-gender estimates):
    /// - S01 (Alice): female, age ~30–45
    /// - S02: male, age ~25
    /// ```
    public func renderForPrompt(
        speakerIDs: [String],
        speakerNames: [String: String]
    ) -> String {
        let rows = speakerIDs.compactMap { id -> String? in
            guard let entry = entries[id] else { return nil }
            var parts: [String] = []
            if let name = speakerNames[id], !name.isEmpty {
                parts.append("\(id) (\(name))")
            } else {
                parts.append(id)
            }
            var detail: [String] = []
            if let gender = entry.majorityGender {
                detail.append(label(for: gender))
            }
            if let range = entry.ageRangeYears {
                detail.append(formatAgeRange(range))
            }
            guard !detail.isEmpty else { return nil }
            return "- " + parts.joined() + ": " + detail.joined(separator: ", ")
        }
        guard !rows.isEmpty else { return "" }
        var lines: [String] = []
        lines.append("Speaker demographics (rough — W2V2 age-gender estimates; use as soft hints, not facts):")
        lines.append(contentsOf: rows)
        return lines.joined(separator: "\n")
    }

    private func label(for gender: AgeGenderEstimate.Gender) -> String {
        switch gender {
        case .female: return "female"
        case .male: return "male"
        case .child: return "child"
        }
    }

    private func formatAgeRange(_ range: ClosedRange<Float>) -> String {
        let lo = Int(range.lowerBound.rounded())
        let hi = Int(range.upperBound.rounded())
        if lo == hi {
            return "age ~\(lo)"
        }
        return "age ~\(lo)\u{2013}\(hi)"
    }
}
