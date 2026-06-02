import Foundation
import XephonLogging

/// HTTP client for LM Studio's OpenAI-compatible server. One
/// surface method (`chat`) plus a lightweight reachability probe
/// (`listModels`) used by the Settings card's Test Connection
/// button. Actor-isolated because each `RecordingController` holds
/// one instance and we don't want overlapping requests sharing
/// the same URLSession's underlying connection pool to surface
/// race-condition decoding bugs.
///
/// The client deliberately doesn't own any UI state; the caller
/// (`LMStudioSummarizer` / `LMStudioTranscriptionReviewer`) gets
/// back a raw string or throws an `LMStudioError`. Per
/// CLAUDE.md's no-silent-fallback policy, errors bubble up so the
/// banner surfaces a real diagnosis rather than vanishing into a
/// retry on a different backend.
public actor LMStudioClient {
    /// Snapshot of the settings at construction time. Plain
    /// value-type capture rather than holding a reference to
    /// the `@Observable` MainActor settings object — the actor
    /// shouldn't touch MainActor state mid-request, and a
    /// settings change mid-flight wouldn't take effect for an
    /// in-progress generate anyway. The coordinator builds a
    /// fresh client per inference call so the snapshot is
    /// always fresh.
    public struct Configuration: Sendable {
        public let baseURL: URL
        public let modelID: String
        public let requestTimeoutSeconds: TimeInterval

        public init(
            baseURL: URL,
            modelID: String,
            requestTimeoutSeconds: TimeInterval
        ) {
            self.baseURL = baseURL
            self.modelID = modelID
            self.requestTimeoutSeconds = requestTimeoutSeconds
        }
    }

    private let configuration: Configuration

    public init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Process-wide shared session. Earlier versions built a
    /// per-call `URLSession`, which on iOS retained an internal
    /// delegate operation queue (and its backing thread) until
    /// the session was fully torn down — and actor-deinit
    /// timing under Swift 6 made that teardown happen on
    /// MainActor at unpredictable points, surfacing as a
    /// post-failure UI freeze. `URLSession.shared` doesn't have
    /// this lifecycle problem; per-request timeouts ride on
    /// `URLRequest.timeoutInterval` instead of the session
    /// configuration.
    private var urlSession: URLSession { .shared }

    /// POST `/v1/chat/completions` with one user message and
    /// return the assistant's response text. `temperature`
    /// defaults to 0.2 to match the on-device MLX path.
    /// `maxTokens` is wired through so the Summarizer +
    /// Reviewer can match each path's existing cap; nil sends
    /// no cap and lets LM Studio's per-model default decide.
    /// `responseFormatJSON` is the pre-encoded JSON bytes of
    /// an OpenAI `response_format` object, or nil to omit the
    /// field entirely — kept as raw bytes so the caller can
    /// pass a fully-typed `LMStudioResponseFormat` without
    /// dragging the Encodable type onto the actor boundary.
    public func chat(
        userMessage: String,
        temperature: Double = 0.2,
        maxTokens: Int? = nil,
        responseFormatJSON: Data? = nil
    ) async throws -> String {
        let endpoint = configuration.baseURL
            .appendingPathComponent("v1")
            .appendingPathComponent("chat")
            .appendingPathComponent("completions")
        let body = try makeRequestBody(
            userMessage: userMessage,
            temperature: temperature,
            maxTokens: maxTokens,
            responseFormatJSON: responseFormatJSON
        )
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = body
        request.timeoutInterval = configuration.requestTimeoutSeconds

        let (data, response) = try await sendRequest(request)
        try ensureSuccessful(response: response, body: data)

        let decoded: LMStudioChatPayload.Response
        do {
            decoded = try JSONDecoder().decode(
                LMStudioChatPayload.Response.self,
                from: data
            )
        } catch {
            throw LMStudioError.invalidResponse(
                reason: "decode failed: \(error)"
            )
        }
        guard let first = decoded.choices.first,
              let content = first.message?.content else {
            throw LMStudioError.invalidResponse(
                reason: "no choices[0].message.content in response"
            )
        }
        if let usage = decoded.usage {
            AppLog.app.info(
                "LMStudio chat finished: prompt=\(usage.promptTokens ?? -1, privacy: .public) completion=\(usage.completionTokens ?? -1, privacy: .public) total=\(usage.totalTokens ?? -1, privacy: .public)"
            )
        }
        return content
    }

    /// GET `/v1/models` — used by the Test Connection button
    /// to verify the server is reachable and to confirm the
    /// configured `modelID` is loaded. Returns the list of
    /// model identifiers so the Settings card can show "found
    /// 3 models; configured model is loaded" vs "found 3
    /// models, but X is not among them — check the LM Studio
    /// model dropdown".
    public func listModels() async throws -> [String] {
        let endpoint = configuration.baseURL
            .appendingPathComponent("v1")
            .appendingPathComponent("models")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = configuration.requestTimeoutSeconds
        let (data, response) = try await sendRequest(request)
        try ensureSuccessful(response: response, body: data)
        do {
            let decoded = try JSONDecoder().decode(
                LMStudioChatPayload.ModelsResponse.self,
                from: data
            )
            return decoded.data.map(\.id)
        } catch {
            throw LMStudioError.invalidResponse(
                reason: "models decode failed: \(error)"
            )
        }
    }

    /// Build the JSON body for `/v1/chat/completions`. When
    /// `responseFormatJSON` is nil we encode the typed Request
    /// directly. When it's non-nil we need to splice the
    /// already-encoded `response_format` blob into the body —
    /// done by encoding the Request first, then merging the
    /// `response_format` key via JSONSerialization. Keeps the
    /// hot path (no schema) one JSON pass; pays the
    /// round-trip cost only when the user opted into strict
    /// schemas.
    private func makeRequestBody(
        userMessage: String,
        temperature: Double,
        maxTokens: Int?,
        responseFormatJSON: Data?
    ) throws -> Data {
        let request = LMStudioChatPayload.Request(
            model: configuration.modelID,
            messages: [
                LMStudioChatPayload.Message(role: "user", content: userMessage)
            ],
            temperature: temperature,
            maxTokens: maxTokens,
            // `responseFormat` is set via the merge below when
            // `responseFormatJSON` is non-nil; nil here means the
            // field is omitted from the encoded body entirely.
            responseFormat: nil
        )
        let baseData: Data
        do {
            baseData = try JSONEncoder().encode(request)
        } catch {
            throw LMStudioError.invalidResponse(
                reason: "request encoding failed: \(error)"
            )
        }
        guard let responseFormatJSON else { return baseData }
        do {
            guard var dict = try JSONSerialization.jsonObject(with: baseData) as? [String: Any] else {
                throw LMStudioError.invalidResponse(
                    reason: "request body wasn't a JSON object"
                )
            }
            let formatObj = try JSONSerialization.jsonObject(with: responseFormatJSON)
            dict["response_format"] = formatObj
            return try JSONSerialization.data(withJSONObject: dict)
        } catch let e as LMStudioError {
            throw e
        } catch {
            throw LMStudioError.invalidResponse(
                reason: "response_format merge failed: \(error)"
            )
        }
    }

    // MARK: - Internals

    /// Bridge `URLSession.data(for:)` errors into typed
    /// `LMStudioError`s so the calling layer doesn't have to
    /// switch on `URLError.Code` to tell a timeout from a
    /// connection refusal. CancellationErrors pass through —
    /// the coordinator already handles them as user-initiated
    /// stops without painting a banner.
    private func sendRequest(_ request: URLRequest) async throws -> (Data, URLResponse) {
        do {
            return try await urlSession.data(for: request)
        } catch is CancellationError {
            throw CancellationError()
        } catch let urlError as URLError {
            switch urlError.code {
            case .timedOut:
                throw LMStudioError.timeout
            case .cannotConnectToHost,
                 .cannotFindHost,
                 .networkConnectionLost,
                 .notConnectedToInternet,
                 .dnsLookupFailed:
                throw LMStudioError.connectionFailed(
                    reason: urlError.localizedDescription
                )
            default:
                throw LMStudioError.connectionFailed(
                    reason: urlError.localizedDescription
                )
            }
        } catch {
            throw LMStudioError.connectionFailed(
                reason: String(describing: error)
            )
        }
    }

    private func ensureSuccessful(response: URLResponse, body: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw LMStudioError.invalidResponse(
                reason: "non-HTTP response"
            )
        }
        guard (200..<300).contains(http.statusCode) else {
            let bodyText = String(data: body, encoding: .utf8) ?? ""
            throw LMStudioError.httpError(status: http.statusCode, body: bodyText)
        }
    }
}
