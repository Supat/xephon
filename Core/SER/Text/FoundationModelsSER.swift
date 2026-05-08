import Foundation
import FoundationModels
import XephonLogging

// Optional second-opinion classifier using Apple Foundation Models (iPadOS 26+, ~3B).
// Use for structured V/A or open-ended affect description. NOT the classifier of record —
// expect lower quality than fine-tuned DeBERTa-WRIME per Takenaka 2025.
public actor FoundationModelsSER: TextSER {
    // Each classify() call constructs a fresh LanguageModelSession so the
    // 4096-token context window doesn't fill up across utterances. Reusing a
    // session accumulates conversation history and eventually overflows
    // (LanguageModelSession.GenerationError.exceededContextWindowSize).
    private static let instructions = """
        You are a Japanese-language affect annotator. Read the utterance and
        estimate how strongly each Plutchik emotion is present, as independent
        probabilities in [0, 1] (they do NOT need to sum to 1).

        Use the full range. Most utterances carry mild traces of multiple
        emotions — do NOT default to all-zeros. Calibration guide:
          0.0  emotion is clearly absent
          0.2  faint trace, possibly inferred
          0.5  noticeable but not dominant
          0.7  clearly present
          0.9+ dominant / intense

        For neutral or ambiguous utterances, prefer values around 0.2–0.4
        across multiple emotions rather than zeros across the board. Always
        return all eight fields.
        """

    public init() {}

    public func classify(_ text: String) async throws -> PlutchikScore {
        guard SystemLanguageModel.default.isAvailable else {
            throw TextSERError.foundationModelsUnavailable
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return PlutchikScore(probabilities: [:])
        }

        let session = LanguageModelSession(instructions: Self.instructions)
        do {
            let response = try await session.respond(
                to: "Utterance: \(trimmed)",
                generating: GenerablePlutchik.self,
                includeSchemaInPrompt: true
            )
            let g = response.content
            AppLog.serText.debug("FoundationModelsSER returned \(trimmed.count, privacy: .public) char input")
            return PlutchikScore(probabilities: [
                .joy:          Float(g.joy),
                .sadness:      Float(g.sadness),
                .anticipation: Float(g.anticipation),
                .surprise:     Float(g.surprise),
                .anger:        Float(g.anger),
                .fear:         Float(g.fear),
                .disgust:      Float(g.disgust),
                .trust:        Float(g.trust),
            ])
        } catch {
            throw TextSERError.underlying(error)
        }
    }
}

@Generable
private struct GenerablePlutchik {
    @Guide(description: "Probability of joy in [0.0, 1.0]")
    var joy: Double
    @Guide(description: "Probability of sadness in [0.0, 1.0]")
    var sadness: Double
    @Guide(description: "Probability of anticipation in [0.0, 1.0]")
    var anticipation: Double
    @Guide(description: "Probability of surprise in [0.0, 1.0]")
    var surprise: Double
    @Guide(description: "Probability of anger in [0.0, 1.0]")
    var anger: Double
    @Guide(description: "Probability of fear in [0.0, 1.0]")
    var fear: Double
    @Guide(description: "Probability of disgust in [0.0, 1.0]")
    var disgust: Double
    @Guide(description: "Probability of trust in [0.0, 1.0]")
    var trust: Double
}
