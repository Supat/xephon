import Foundation

public enum TextSERError: Error, Sendable {
    case notImplemented
    case modelUnavailable(reason: String)
    case foundationModelsUnavailable
    /// Apple FoundationModels declined to respond because the input or
    /// the structured output tripped its safety classifier. Distinguished
    /// from `underlying` so callers can stamp a "guardrail" marker on
    /// the utterance instead of treating it as a generic failure —
    /// guardrail trips are common on emotional content and aren't a
    /// model-availability problem.
    case guardrailViolation
    case underlying(any Error)
}
