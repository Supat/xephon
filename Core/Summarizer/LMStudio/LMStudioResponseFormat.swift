import Foundation

/// OpenAI-format `response_format` payload — wraps a JSON schema
/// describing the expected response shape. LM Studio servers
/// that support constrained decoding use this to guide token
/// sampling toward valid JSON; servers that don't recognize it
/// either ignore it or 400 (the toggle defaults off in
/// `LMStudioSettings` for that reason).
///
/// Only the `json_schema` variant is modeled here — `json_object`
/// (loose JSON mode) and `text` (the default) are equivalent to
/// just omitting the field, which we do by passing `nil`.
internal struct LMStudioResponseFormat: Encodable, Sendable {
    let type: String  // always "json_schema"
    let jsonSchema: SchemaWrapper

    enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }

    struct SchemaWrapper: Encodable, Sendable {
        let name: String
        let schema: JSONSchemaNode
        let strict: Bool
    }

    static func jsonSchema(
        name: String,
        schema: JSONSchemaNode,
        strict: Bool = true
    ) -> LMStudioResponseFormat {
        LMStudioResponseFormat(
            type: "json_schema",
            jsonSchema: SchemaWrapper(name: name, schema: schema, strict: strict)
        )
    }
}

/// Minimal typed JSON Schema representation covering the shapes
/// the LM Studio summarizer + reviewer need:
/// `object` (with properties + required), `array` (of items),
/// `string` (optionally enum-constrained), `number`.
///
/// Indirect because nested objects + array items are common
/// (`{"perSpeaker": {"type": "array", "items": {"type": "object", … }}}`).
internal indirect enum JSONSchemaNode: Encodable, Sendable {
    /// `{ "type": "object", "properties": …, "required": …,
    ///    "additionalProperties": false }`. additionalProperties
    /// hard-set to false because LM Studio servers that honor
    /// strict-mode reject any extra fields, which is what we
    /// want — the parser only reads the named ones.
    case object(properties: [(name: String, node: JSONSchemaNode)], required: [String])
    case array(items: JSONSchemaNode)
    /// Optional `enum` constraint pinning the string to a fixed
    /// vocabulary (e.g. issue `kind` ∈ {homophone, contextual,
    /// grammar, other}). nil = any string.
    case string(allowedValues: [String]?)
    case number

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .object(let props, let required):
            try container.encode("object", forKey: .type)
            // `properties` is an ordered list of (name, node)
            // pairs so the emitted JSON schema preserves
            // declaration order. JSON objects aren't ordered per
            // spec, but ordered output reads better when the
            // schema is inspected (e.g. in the Prompts card's
            // debug entry) and avoids per-launch reorders by
            // the dictionary's randomized iteration.
            var propsContainer = container.nestedContainer(
                keyedBy: DynamicKey.self,
                forKey: .properties
            )
            for (name, node) in props {
                guard let key = DynamicKey(stringValue: name) else { continue }
                try propsContainer.encode(node, forKey: key)
            }
            try container.encode(required, forKey: .required)
            try container.encode(false, forKey: .additionalProperties)
        case .array(let items):
            try container.encode("array", forKey: .type)
            try container.encode(items, forKey: .items)
        case .string(let allowed):
            try container.encode("string", forKey: .type)
            if let allowed { try container.encode(allowed, forKey: .enumValues) }
        case .number:
            try container.encode("number", forKey: .type)
        }
    }

    private enum CodingKeys: String, CodingKey {
        case type, properties, required, items
        case additionalProperties
        case enumValues = "enum"
    }

    /// Dynamic key used to encode a property bag whose keys are
    /// runtime strings (the `properties` map). JSON-tolerant: any
    /// non-empty string is a valid property name.
    private struct DynamicKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }
}

/// JSON Schema descriptions for the two response shapes the LM
/// Studio adapters consume. Built as static lets so the encoder
/// pays the construction cost once per launch, not per request.
internal enum LMStudioSchemas {
    /// Mirrors `MLXLLMSummarizerCore.parse`'s `Wire` struct:
    /// `{ setting, topic, overallMood, perSpeaker[] }`, with
    /// each `perSpeaker` entry being `{ speakerID, summary,
    /// dominantMood }`. **Every property listed here is also
    /// in `required`** — OpenAI's strict-mode JSON-schema
    /// contract demands it (any optional property silently
    /// fails strict validation, causing the server to either
    /// 400 the request or fall back to freeform text that
    /// can't be parsed). The parser tolerates empty strings
    /// for `setting`, so requiring it costs nothing on the
    /// emit side.
    static let summarySchema: JSONSchemaNode = .object(
        properties: [
            ("setting", .string(allowedValues: nil)),
            ("topic", .string(allowedValues: nil)),
            ("overallMood", .string(allowedValues: nil)),
            ("perSpeaker", .array(items: .object(
                properties: [
                    ("speakerID", .string(allowedValues: nil)),
                    ("summary", .string(allowedValues: nil)),
                    ("dominantMood", .string(allowedValues: nil)),
                ],
                required: ["speakerID", "summary", "dominantMood"]
            )))
        ],
        required: ["setting", "topic", "overallMood", "perSpeaker"]
    )

    /// Mirrors `MLXLLMReviewerCore.parse`'s `Wire` struct:
    /// `{ issues: [{ rowIndex, kind, reason, confidence }] }`.
    /// `kind` is enum-constrained to the four values the parser
    /// maps onto `TranscriptionIssue.Kind`. `confidence` is
    /// in `required` for strict-mode compliance — the parser's
    /// Wire decoder accepts whatever number the model emits,
    /// including 0 when it has no confidence to report.
    static let reviewSchema: JSONSchemaNode = .object(
        properties: [
            ("issues", .array(items: .object(
                properties: [
                    ("rowIndex", .number),
                    ("kind", .string(allowedValues: [
                        "homophone", "contextual", "grammar", "other"
                    ])),
                    ("reason", .string(allowedValues: nil)),
                    ("confidence", .number),
                ],
                required: ["rowIndex", "kind", "reason", "confidence"]
            )))
        ],
        required: ["issues"]
    )
}
