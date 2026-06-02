import Foundation

/// Typed errors raised by `LMStudioClient` and the LM Studio
/// summarizer / reviewer adapters. Mirrors `SummarizerError` and
/// `TranscriptionReviewError` in granularity so the coordinator's
/// existing error-surfacing path (`parent.errorMessage = String(describing: error)`)
/// produces a useful banner without per-case translation.
public enum LMStudioError: Error, CustomStringConvertible {
    /// Settings is missing host / port / model — the picker
    /// should have prevented this, but a transient
    /// flip-during-inference race could still hit it. Caller
    /// should treat as "nothing to retry, ask the user."
    case notConfigured(reason: String)
    /// `baseURL` resolved but the OS couldn't connect (server
    /// down, wrong IP, firewall). `reason` is the underlying
    /// URLError description for the logs / banner.
    case connectionFailed(reason: String)
    /// Server responded with a non-2xx HTTP status. Body is
    /// the raw response text (truncated to a small budget for
    /// logging) — LM Studio puts useful error detail in the
    /// JSON body when the model isn't loaded or the request is
    /// malformed.
    case httpError(status: Int, body: String)
    /// Request exceeded `requestTimeoutSeconds`. Separate case
    /// from `connectionFailed` so the banner can suggest
    /// raising the timeout slider rather than "check the
    /// server".
    case timeout
    /// Response was 200 but the JSON body didn't have the
    /// shape we expect (`choices[0].message.content` missing
    /// or non-string). Rare with LM Studio's standard
    /// OpenAI-compatible server but possible if the user is
    /// pointing at a proxy that mangles the shape.
    case invalidResponse(reason: String)

    public var description: String {
        switch self {
        case .notConfigured(let r):       return "LM Studio not configured: \(r)"
        case .connectionFailed(let r):    return "LM Studio connection failed: \(r)"
        case .httpError(let s, let body):
            let preview = body.prefix(200)
            return "LM Studio HTTP \(s): \(preview)"
        case .timeout:                    return "LM Studio request timed out"
        case .invalidResponse(let r):     return "LM Studio response shape unexpected: \(r)"
        }
    }
}
