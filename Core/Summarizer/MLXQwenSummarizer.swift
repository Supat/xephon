import Foundation
import Fusion
import XephonLogging
import MLXLLM
import MLXLMCommon

/// MLX-backed summarizer using a Qwen2.5-Instruct 4-bit model
/// hydrated under `ModelStore`'s install directory. All inference
/// runs on-device — no network calls, no cloud fallback.
///
/// Lifecycle:
///   1. Construct with the resolved model directory + identifier.
///   2. Call `load()` (or let `summarize(...)` lazy-load on first
///      use) to bring the weights into memory. Loading 4-bit 7B is
///      ~5–10 s on M4 iPad Pro.
///   3. Call `summarize(utterances:speakerNames:)` to generate.
///   4. Call `unload()` to release the ~4 GB working set when the
///      summarize UI dismisses — keeping the model resident across
///      a recording session would push memory pressure too hard
///      next to W2V2 + emotion2vec + DeBERTa.
public actor MLXQwenSummarizer: SessionSummarizer {
    public let modelIdentifier: String
    private let modelDirectory: URL
    private var container: ModelContainer?

    public init(modelIdentifier: String, modelDirectory: URL) {
        self.modelIdentifier = modelIdentifier
        self.modelDirectory = modelDirectory
    }

    public var isReady: Bool {
        container != nil
    }

    /// Bring the weights into memory. Idempotent. Surfaces
    /// `SummarizerError.modelLoadFailed` on any underlying MLX
    /// error (corrupted weights, format mismatch, etc.).
    public func load() async throws {
        if container != nil { return }
        AppLog.app.info(
            "MLXQwenSummarizer loading from \(self.modelDirectory.path, privacy: .public)"
        )
        do {
            // MLXLMCommon resolves a directory containing
            // `config.json`, `tokenizer.json`, and the safetensors
            // shards into a `ModelContainer` that owns the loaded
            // weights + tokenizer for the duration of the actor.
            let configuration = ModelConfiguration(
                directory: modelDirectory
            )
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: configuration
            )
            AppLog.app.info("MLXQwenSummarizer loaded")
        } catch {
            throw SummarizerError.modelLoadFailed(
                reason: String(describing: error)
            )
        }
    }

    public func unload() {
        container = nil
        AppLog.app.info("MLXQwenSummarizer unloaded")
    }

    public func summarize(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) async throws -> SessionSummary {
        try await load()
        guard let container else {
            throw SummarizerError.modelNotInstalled
        }
        guard !utterances.isEmpty else {
            return SessionSummary(
                topic: "",
                overallMood: "",
                perSpeaker: [],
                model: modelIdentifier,
                generatedAt: Date()
            )
        }

        let prompt = Self.buildPrompt(
            utterances: utterances,
            speakerNames: speakerNames
        )
        let raw: String
        do {
            raw = try await container.perform { context in
                let input = UserInput(prompt: prompt)
                let prepared = try await context.processor.prepare(
                    input: input
                )
                var collected = ""
                _ = try MLXLMCommon.generate(
                    input: prepared,
                    parameters: GenerateParameters(
                        // Deterministic for reproducibility — the
                        // summary is a derived artifact of the
                        // session, not creative writing. We can
                        // raise temperature later if user feedback
                        // says outputs feel mechanical.
                        temperature: 0.2
                    ),
                    context: context
                ) { tokens in
                    collected = context.tokenizer.decode(tokens: tokens)
                    return .more
                }
                return collected
            }
        } catch let error as SummarizerError {
            throw error
        } catch {
            throw SummarizerError.inferenceFailed(
                reason: String(describing: error)
            )
        }

        return try Self.parse(
            raw: raw,
            speakerNames: speakerNames,
            modelIdentifier: modelIdentifier
        )
    }

    // MARK: - Prompt + parser

    /// Order distinct speaker ids in their first-appearance order,
    /// matching the chip-bar ordering in the UI so the LLM's
    /// per-speaker arc reads in the same order the user is reading
    /// the rows in.
    private static func orderedSpeakerIDs(
        from utterances: [UtteranceEstimate]
    ) -> [String] {
        var seen: Set<String> = []
        var ordered: [String] = []
        for u in utterances where !seen.contains(u.speakerID) {
            seen.insert(u.speakerID)
            ordered.append(u.speakerID)
        }
        return ordered
    }

    /// Build the chat-style prompt the model sees. Compact JSON-ish
    /// per-utterance lines keep the token budget bounded — every
    /// numerical score is preserved (the whole point of going
    /// beyond Apple FM was to keep these) but we elide the V/A/D
    /// detail when it's nil and round floats to 2 decimals.
    private static func buildPrompt(
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]
    ) -> String {
        let speakers = orderedSpeakerIDs(from: utterances)
        var lines: [String] = []
        lines.reserveCapacity(utterances.count + 10)
        lines.append("You are an analyst summarizing a multi-speaker conversation.")
        lines.append("Read every utterance below and produce a JSON object with three fields:")
        lines.append("  \"topic\" — one or two sentences on what the conversation is about.")
        lines.append("  \"overallMood\" — one paragraph on the session's overall emotional tone.")
        lines.append("  \"perSpeaker\" — array, one entry per speaker id in this list: \(speakers.joined(separator: ", ")).")
        lines.append("Each perSpeaker entry has: { \"speakerID\": <id>, \"summary\": <one paragraph>, \"dominantMood\": <one short phrase> }.")
        lines.append("Use the numerical V/A/D and label probabilities to judge confidence and weight your reasoning.")
        lines.append("Return ONLY valid JSON, no prose before or after.")
        lines.append("")
        lines.append("Utterances:")
        for u in utterances {
            lines.append(compactLine(for: u, speakerNames: speakerNames))
        }
        return lines.joined(separator: "\n")
    }

    /// One per-utterance line. Keep order stable (speaker, time,
    /// label, V/A, text) so the model sees consistent positional
    /// cues across rows.
    private static func compactLine(
        for u: UtteranceEstimate,
        speakerNames: [String: String]
    ) -> String {
        var fields: [String] = []
        fields.append("speaker=\(u.speakerID)")
        if let name = speakerNames[u.speakerID], !name.isEmpty {
            fields.append("name=\(name)")
        }
        fields.append(String(format: "t=%.1fs", u.start))
        if let label = u.fusedTopLabel { fields.append("label=\(label)") }
        if let v = u.fusedValence { fields.append(String(format: "V=%.2f", v)) }
        if let a = u.fusedArousal { fields.append(String(format: "A=%.2f", a)) }
        if let d = u.fusedDominance { fields.append(String(format: "D=%.2f", d)) }
        let escaped = u.transcript
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        fields.append("text=\"\(escaped)\"")
        return "- " + fields.joined(separator: " ")
    }

    /// Decode the LLM's JSON output. Robust against the common
    /// failure modes of small-LLM output: leading / trailing prose
    /// outside the JSON block, fenced code blocks, or partial
    /// trailing fragments.
    private static func parse(
        raw: String,
        speakerNames: [String: String],
        modelIdentifier: String
    ) throws -> SessionSummary {
        let stripped = stripCodeFence(raw)
        guard let braceStart = stripped.firstIndex(of: "{"),
              let braceEnd = stripped.lastIndex(of: "}") else {
            throw SummarizerError.decodeFailed(reason: "no JSON object found")
        }
        let jsonString = String(stripped[braceStart...braceEnd])
        guard let data = jsonString.data(using: .utf8) else {
            throw SummarizerError.decodeFailed(reason: "non-utf8 output")
        }
        struct Wire: Decodable {
            struct PerSpeaker: Decodable {
                let speakerID: String
                let summary: String
                let dominantMood: String
            }
            let topic: String
            let overallMood: String
            let perSpeaker: [PerSpeaker]
        }
        let decoded: Wire
        do {
            decoded = try JSONDecoder().decode(Wire.self, from: data)
        } catch {
            throw SummarizerError.decodeFailed(
                reason: String(describing: error)
            )
        }
        let perSpeaker = decoded.perSpeaker.map { entry in
            SessionSummary.SpeakerSummary(
                speakerID: entry.speakerID,
                speakerName: speakerNames[entry.speakerID],
                summary: entry.summary,
                dominantMood: entry.dominantMood
            )
        }
        return SessionSummary(
            topic: decoded.topic,
            overallMood: decoded.overallMood,
            perSpeaker: perSpeaker,
            model: modelIdentifier,
            generatedAt: Date()
        )
    }

    /// Strip ```json ... ``` fences a chat-tuned model might emit
    /// even after being told "JSON only." Leaves bare JSON alone.
    private static func stripCodeFence(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            if let firstNewline = s.firstIndex(of: "\n") {
                s = String(s[s.index(after: firstNewline)...])
            }
        }
        if s.hasSuffix("```") {
            s = String(s.dropLast(3)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }
}
