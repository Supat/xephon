import Foundation

/// Codable wire types for LM Studio's OpenAI-compatible
/// `/v1/chat/completions` endpoint. Kept narrow — only the fields
/// the summarizer + reviewer actually send / read. LM Studio
/// accepts a much wider request surface (stream, tools, response
/// format, …) but we deliberately don't expose any of it from the
/// client surface until the feature is actually needed; less to
/// mis-encode, easier to evolve.
internal enum LMStudioChatPayload {
    /// Request body. `model` is allowed to be empty (LM Studio
    /// falls back to whichever model is currently loaded in that
    /// case). `temperature` defaults to 0.2 to match the on-device
    /// MLX path so output behaviour reads consistently across
    /// backends. `responseFormat` is nil unless the user enables
    /// the "Strict JSON schema" toggle — OpenAI-spec
    /// `response_format` with a JSON schema body, encoded only
    /// when non-nil so a server that doesn't recognize the field
    /// still gets a clean request.
    struct Request: Encodable {
        let model: String
        let messages: [Message]
        let temperature: Double
        let maxTokens: Int?
        let responseFormat: LMStudioResponseFormat?

        enum CodingKeys: String, CodingKey {
            case model, messages, temperature
            case maxTokens = "max_tokens"
            case responseFormat = "response_format"
        }
    }

    /// One chat message. The summarizer + reviewer send a single
    /// `.user` message containing the full `MLXLLMSpec.buildPrompt`
    /// output; LM Studio applies the loaded model's chat template
    /// server-side just like MLX-LM does on-device. Keeping the
    /// system slot empty avoids a double-wrap with whatever
    /// system message the model's template already injects.
    struct Message: Encodable {
        let role: String   // "system" | "user" | "assistant"
        let content: String
    }

    /// Response body. Only `choices[0].message.content` is
    /// load-bearing; everything else (id, usage, finish_reason)
    /// is informational for the logs and intentionally optional
    /// so a leaner proxy response still decodes.
    struct Response: Decodable {
        struct Choice: Decodable {
            struct ResponseMessage: Decodable {
                let role: String?
                let content: String?
            }
            let index: Int?
            let message: ResponseMessage?
            let finishReason: String?

            enum CodingKeys: String, CodingKey {
                case index, message
                case finishReason = "finish_reason"
            }
        }
        struct Usage: Decodable {
            let promptTokens: Int?
            let completionTokens: Int?
            let totalTokens: Int?

            enum CodingKeys: String, CodingKey {
                case promptTokens = "prompt_tokens"
                case completionTokens = "completion_tokens"
                case totalTokens = "total_tokens"
            }
        }
        let id: String?
        let model: String?
        let choices: [Choice]
        let usage: Usage?
    }

    /// `/v1/models` response — used by the Test Connection
    /// affordance in Settings to verify (a) the server is
    /// reachable and (b) the configured `modelID` is loaded.
    struct ModelsResponse: Decodable {
        struct Entry: Decodable {
            let id: String
        }
        let data: [Entry]
    }
}
