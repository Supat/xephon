import Foundation
import Testing
import Fusion
@testable import Summarizer

/// Pins the meeting-mode evidence contract: prompt row numbers in
/// the model's `evidence` arrays are validated against the numbered
/// row list and mapped to utterance IDs — invalid citations are
/// STRIPPED (never guessed), duplicates dedup, and claims keep
/// their text either way. This is the accuracy floor the meeting
/// summary's citations stand on; keep it green across prompt /
/// parser changes.
@Suite("Meeting evidence citations")
struct MeetingEvidenceTests {

    private func row(_ speaker: String, _ text: String, at t: Double) -> UtteranceEstimate {
        UtteranceEstimate(
            speakerID: speaker,
            start: t,
            end: t + 1,
            transcript: text,
            asrConfidence: 0.9,
            dimensional: nil,
            acousticCategorical: nil,
            plutchik: nil,
            fusedValence: nil,
            fusedArousal: nil,
            fusedDominance: nil,
            fusedTopLabel: nil
        )
    }

    @Test("Valid citations map to utterance IDs in order")
    func validCitationsMap() throws {
        let rows = [row("S01", "a", at: 0), row("S02", "b", at: 1), row("S01", "c", at: 2)]
        // Topic-level evidence arrives as an array (merge-pass shape);
        // position evidence as the single-number contract.
        let raw = """
        {"topic":"t","topics":[{"title":"T1","raisedBy":"S01","evidence":[3,1],
        "positions":[{"speaker":"S02","stance":"agrees","evidence":2}]}],
        "perSpeaker":[{"speakerID":"S01","talkingPoints":["p"]},{"speakerID":"S02","talkingPoints":["q"]}]}
        """
        let summary = try MLXLLMSummarizerCore.parseMeeting(
            raw: raw,
            rows: rows,
            speakerNames: [:],
            modelIdentifier: "test",
            expectedSpeakerIDs: ["S01", "S02"]
        )
        let topic = try #require(summary.topics?.first)
        #expect(topic.evidenceUtteranceIDs == [rows[2].id, rows[0].id])
        #expect(topic.positions.first?.evidenceUtteranceIDs == [rows[1].id])
    }

    @Test("Invented and out-of-range citations strip; claims survive")
    func invalidCitationsStrip() throws {
        let rows = [row("S01", "a", at: 0), row("S02", "b", at: 1)]
        let raw = """
        {"topic":"t","topics":[{"title":"T1","raisedBy":null,"evidence":[0,99,-3,1,1],
        "positions":[{"speaker":"S01","stance":"claim with no valid rows","evidence":42}]}],
        "perSpeaker":[{"speakerID":"S01","talkingPoints":[]},{"speakerID":"S02","talkingPoints":[]}]}
        """
        let summary = try MLXLLMSummarizerCore.parseMeeting(
            raw: raw,
            rows: rows,
            speakerNames: [:],
            modelIdentifier: "test",
            expectedSpeakerIDs: ["S01", "S02"]
        )
        let topic = try #require(summary.topics?.first)
        // 0, 99, -3 stripped; duplicate 1 deduped → exactly one id.
        #expect(topic.evidenceUtteranceIDs == [rows[0].id])
        // Entirely-invalid evidence → empty array, but the claim's
        // text is kept: validation polices citations, not content.
        #expect(topic.positions.first?.evidenceUtteranceIDs == [])
        #expect(topic.positions.first?.stance == "claim with no valid rows")
    }

    @Test("Missing evidence arrays decode as nil (older outputs)")
    func missingEvidenceIsNil() throws {
        let rows = [row("S01", "a", at: 0)]
        let raw = """
        {"topic":"t","topics":[{"title":"T1","raisedBy":"S01",
        "positions":[{"speaker":"S01","stance":"s"}]}],
        "perSpeaker":[{"speakerID":"S01","talkingPoints":["p"]}]}
        """
        let summary = try MLXLLMSummarizerCore.parseMeeting(
            raw: raw,
            rows: rows,
            speakerNames: [:],
            modelIdentifier: "test",
            expectedSpeakerIDs: ["S01"]
        )
        let topic = try #require(summary.topics?.first)
        #expect(topic.evidenceUtteranceIDs == nil)
        #expect(topic.positions.first?.evidenceUtteranceIDs == nil)
    }
}

/// Pins the truncation salvage: output that hit the token cap
/// mid-`topics` (no perSpeaker key — observed on-device with the
/// 4096-token cap) recovers the complete topic entries instead of
/// failing the whole summary.
@Suite("Meeting truncation salvage")
struct MeetingTruncationTests {
    @Test("Cap-truncated topics prefix salvages complete entries")
    func truncatedTopicsSalvage() throws {
        // One complete topic entry, then a second cut mid-string —
        // the shape a token-capped run produces.
        let raw = """
        {"topic":"overview","topics":[
        {"title":"T1","raisedBy":"S01","evidence":[1],
        "positions":[{"speaker":"S02","stance":"agrees","evidence":[1]}]},
        {"title":"T2","raisedBy":"S0
        """
        let wire = try #require(MLXLLMSummarizerCore.decodeMeetingWire(raw: raw))
        #expect(wire.topics?.count == 1)
        #expect(wire.topics?.first?.title == "T1")
        #expect(wire.perSpeaker?.isEmpty == true)
    }

    @Test("Properly closed meeting JSON is untouched by salvage")
    func completeJSONNotSalvaged() throws {
        let raw = """
        {"topic":"t","topics":[{"title":"T1","raisedBy":null,
        "positions":[]}],"perSpeaker":[{"speakerID":"S01","talkingPoints":["p"]}]}
        """
        let wire = try #require(MLXLLMSummarizerCore.decodeMeetingWire(raw: raw))
        #expect(wire.topics?.count == 1)
        #expect(wire.perSpeaker?.count == 1)
    }
}

/// Pins the brace-inversion guard: a stray `}` in prose BEFORE the
/// JSON opens, with generation cut off before any closing brace,
/// previously trapped in the ClosedRange subscript (audit finding,
/// crash-class). Must return nil, not crash.
@Suite("Meeting parser brace inversion")
struct MeetingBraceInversionTests {
    @Test("Inverted braces return nil instead of trapping")
    func invertedBracesSafe() {
        let raw = "reasoning: score 8} then the JSON: {\"topic\":\"cut off"
        #expect(MLXLLMSummarizerCore.decodeMeetingWire(raw: raw) == nil)
    }
}
