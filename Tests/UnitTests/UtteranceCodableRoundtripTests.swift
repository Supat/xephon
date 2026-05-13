import Foundation
import Testing
@testable import Fusion
@testable import SERAcoustic
@testable import SERText
@testable import Export

/// Pinning test for the per-utterance Codable contract. Every field
/// added to `UtteranceEstimate` should be exercised here so the JSON
/// export and the `.xph` (binary plist) save path both keep round-
/// tripping. New model integrations land here first; a missing field
/// in this suite means external tooling could stop seeing data on
/// the next refactor.
@Suite("UtteranceEstimate Codable round-trip")
struct UtteranceCodableRoundtripTests {

    /// Reference fixture exercising every optional field, including
    /// the recently-added age-gender output from the W2V2 model.
    private func sample() -> UtteranceEstimate {
        UtteranceEstimate(
            id: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            speakerID: "S01",
            speakerName: "Alice",
            start: 12.34,
            end: 14.71,
            transcript: "今日は本当に楽しかった",
            asrConfidence: 0.87,
            dimensional: VADScore(valence: 0.78, arousal: 0.62, dominance: 0.55),
            acousticCategorical: CategoricalEmotion(
                probabilities: [
                    .happy: 0.71, .neutral: 0.18, .surprised: 0.06,
                    .sad: 0.02, .angry: 0.01, .fearful: 0.01,
                    .disgusted: 0.005, .other: 0.005, .unknown: 0.0,
                ]
            ),
            ageGender: AgeGenderEstimate(
                age: 0.32,
                genderProbabilities: [.female: 0.95, .male: 0.03, .child: 0.02]
            ),
            plutchik: PlutchikScore(
                probabilities: [
                    .joy: 0.81, .trust: 0.34, .anticipation: 0.22,
                    .surprise: 0.09, .sadness: 0.04, .fear: 0.02,
                    .anger: 0.02, .disgust: 0.01,
                ]
            ),
            textBackend: "deberta",
            speechBoost: true,
            wasReevaluated: false,
            wasHandEdited: false,
            fusedValence: 0.79,
            fusedArousal: 0.61,
            fusedDominance: 0.55,
            fusedTopLabel: "joy"
        )
    }

    @Test func jsonRoundTripPreservesAgeGender() throws {
        let original = sample()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(UtteranceEstimate.self, from: data)
        try assertSameSurface(original, decoded)
        // Spot-check the JSON literal so a future refactor can't
        // silently rename the `ageGender` key without breaking the
        // contract documented in `docs/output_schema.md`.
        let raw = String(data: data, encoding: .utf8)!
        #expect(raw.contains("\"ageGender\""))
        #expect(raw.contains("\"female\""))
    }

    @Test func plistRoundTripPreservesAgeGender() throws {
        let original = sample()
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        let data = try encoder.encode(original)
        let decoded = try PropertyListDecoder().decode(
            UtteranceEstimate.self, from: data
        )
        try assertSameSurface(original, decoded)
    }

    @Test func sessionBundleRoundTripCarriesAgeGender() throws {
        let original = sample()
        let doc = SessionDocument(
            sourceKind: .file,
            audioFilename: "fixture.m4a",
            audio: nil,
            utterances: [original]
        )
        let bytes = try SessionBundle.encode(doc)
        let restored = try SessionBundle.decode(bytes)
        let row = try #require(restored.utterances.first)
        try assertSameSurface(original, row)
    }

    /// Compare two `UtteranceEstimate`s field-by-field, picking
    /// approximate equality for floats so JSON's lossy decimal-to-
    /// double conversion doesn't false-positive a regression.
    private func assertSameSurface(
        _ a: UtteranceEstimate,
        _ b: UtteranceEstimate
    ) throws {
        #expect(a.id == b.id)
        #expect(a.speakerID == b.speakerID)
        #expect(a.speakerName == b.speakerName)
        #expect(approxEqual(a.start, b.start))
        #expect(approxEqual(a.end, b.end))
        #expect(a.transcript == b.transcript)
        #expect(approxEqualOpt(a.asrConfidence, b.asrConfidence))
        #expect(a.textBackend == b.textBackend)
        #expect(a.speechBoost == b.speechBoost)
        #expect(a.wasReevaluated == b.wasReevaluated)
        #expect(a.wasHandEdited == b.wasHandEdited)
        #expect(a.fusedTopLabel == b.fusedTopLabel)
        #expect(approxEqualOpt(a.fusedValence, b.fusedValence))
        #expect(approxEqualOpt(a.fusedArousal, b.fusedArousal))
        #expect(approxEqualOpt(a.fusedDominance, b.fusedDominance))

        let ag = try #require(a.ageGender)
        let bg = try #require(b.ageGender)
        #expect(approxEqual(ag.age, bg.age))
        for label in AgeGenderEstimate.Gender.allCases {
            #expect(approxEqualOpt(
                ag.genderProbabilities[label],
                bg.genderProbabilities[label]
            ))
        }
    }

    private func approxEqual<T: BinaryFloatingPoint>(_ a: T, _ b: T) -> Bool {
        abs(a - b) < 1e-5
    }
    private func approxEqual(_ a: TimeInterval, _ b: TimeInterval) -> Bool {
        abs(a - b) < 1e-9
    }
    private func approxEqualOpt<T: BinaryFloatingPoint>(_ a: T?, _ b: T?) -> Bool {
        switch (a, b) {
        case (nil, nil): return true
        case let (x?, y?): return abs(x - y) < 1e-5
        default: return false
        }
    }
}
