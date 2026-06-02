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

    /// Response body. `choices[0].message.content` is the
    /// canonical OpenAI shape, but LM Studio servers using
    /// strict `response_format: json_schema` mode can route
    /// the structured output through one of two alternate
    /// fields depending on version: `tool_calls[].function.arguments`
    /// (legacy strict-mode pipe) or `reasoning_content`
    /// (reasoning-model variants). `LMStudioClient.chat`
    /// resolves across all three; the decoder just needs the
    /// fields to be present in the type. Everything else
    /// (id, usage, finish_reason) is informational for the
    /// logs and stays optional so a leaner proxy response
    /// still decodes.
    struct Response: Decodable {
        struct Choice: Decodable {
            struct ResponseMessage: Decodable {
                let role: String?
                let content: String?
                let toolCalls: [ToolCall]?
                let reasoningContent: String?

                enum CodingKeys: String, CodingKey {
                    case role, content
                    case toolCalls = "tool_calls"
                    case reasoningContent = "reasoning_content"
                }
            }
            let index: Int?
            let message: ResponseMessage?
            let finishReason: String?

            enum CodingKeys: String, CodingKey {
                case index, message
                case finishReason = "finish_reason"
            }
        }
        struct ToolCall: Decodable {
            let id: String?
            let type: String?
            let function: Function?

            struct Function: Decodable {
                let name: String?
                /// JSON string body of the tool-call arguments.
                /// For LM Studio's strict-json_schema route this
                /// contains the same JSON object that would
                /// normally land in `message.content`.
                let arguments: String?
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
