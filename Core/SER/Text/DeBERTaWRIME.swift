import Foundation
import CoreML
import Tokenizers
import Hub
import XephonLogging

// Fine-tuned Japanese DeBERTa-v3-large on WRIME → 8-Plutchik.
// Per Takenaka 2025: fine-tuned encoder beats prompt-only LLMs on this task.
//
// Caveat per CLAUDE.md: WRIME-trained classifiers under-detect strong affect
// when the speaker uses 敬語 (politeness register).
//
// Bundle contract:
//   <subdir>/model.mlpackage      Core ML conversion of the WRIME classifier
//   <subdir>/tokenizer.json       HuggingFace tokenizer.json (SentencePiece for DeBERTa-v3)
//   <subdir>/tokenizer_config.json
//
// Conversion sketch (Python, off-device):
//   from transformers import AutoModelForSequenceClassification, AutoTokenizer
//   import coremltools as ct, torch
//   tok = AutoTokenizer.from_pretrained(MODEL_ID)
//   m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
//   ex = tok("テスト", return_tensors="pt", padding="max_length", max_length=128)
//   traced = torch.jit.trace(m, (ex.input_ids, ex.attention_mask))
//   mlpkg = ct.convert(traced, source="pytorch",
//                      inputs=[ct.TensorType(name="input_ids", shape=ex.input_ids.shape, dtype=int32),
//                              ct.TensorType(name="attention_mask", shape=ex.attention_mask.shape, dtype=int32)])
//   mlpkg.save("model.mlpackage")
public actor DeBERTaWRIME: TextSER {
    private static let LABEL_ORDER: [PlutchikScore.Label] = [
        // WRIME ordering (Plutchik 8 — the project's canonical ordering for the
        // converted classifier head). Adjust to match `id2label` in your conversion.
        .joy, .sadness, .anticipation, .surprise,
        .anger, .fear, .disgust, .trust
    ]

    // MLModel isn't Sendable — bypass strict concurrency since the actor
    // serializes all calls through this single instance.
    private nonisolated(unsafe) let model: MLModel
    private let tokenizer: any Tokenizer
    private let maxTokens: Int

    public init(modelURL: URL, tokenizerDirectory: URL, maxTokens: Int = 128) async throws {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            self.model = try MLModel(contentsOf: modelURL, configuration: config)

            let tokenizerConfigURL = tokenizerDirectory.appendingPathComponent("tokenizer_config.json")
            let tokenizerDataURL   = tokenizerDirectory.appendingPathComponent("tokenizer.json")
            let configData = try Data(contentsOf: tokenizerConfigURL)
            let tokenizerData = try Data(contentsOf: tokenizerDataURL)
            let configDict = try JSONSerialization.jsonObject(with: configData) as? [NSString: Any] ?? [:]
            let dataDict   = try JSONSerialization.jsonObject(with: tokenizerData) as? [NSString: Any] ?? [:]
            self.tokenizer = try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
            self.maxTokens = maxTokens
            AppLog.serText.info("DeBERTaWRIME loaded: \(modelURL.lastPathComponent, privacy: .public)")
        } catch {
            throw TextSERError.modelUnavailable(reason: String(describing: error))
        }
    }

    /// Convenience: load from app bundle. Looks for a `deberta-wrime` subdirectory.
    public init(maxTokens: Int = 128) async throws {
        guard let modelURL = Bundle.main.url(
            forResource: "model",
            withExtension: "mlpackage",
            subdirectory: "deberta-wrime"
        ),
        let resourceDir = Bundle.main.url(
            forResource: "tokenizer",
            withExtension: "json",
            subdirectory: "deberta-wrime"
        )?.deletingLastPathComponent()
        else {
            throw TextSERError.modelUnavailable(
                reason: "deberta-wrime model.mlpackage + tokenizer.json not in app bundle (see source comment for the conversion sketch)"
            )
        }
        try await self.init(modelURL: modelURL, tokenizerDirectory: resourceDir, maxTokens: maxTokens)
    }

    public func classify(_ text: String) async throws -> PlutchikScore {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return PlutchikScore(probabilities: [:])
        }

        // Tokenize + pad/truncate to fixed length so the Core ML graph (static
        // shapes) is happy.
        var ids = tokenizer.encode(text: trimmed, addSpecialTokens: true)
        if ids.count > maxTokens { ids = Array(ids.prefix(maxTokens)) }
        var mask = Array(repeating: Int32(1), count: ids.count)
        let padId = Int32(tokenizer.unknownTokenId ?? 0)
        while ids.count < maxTokens {
            ids.append(Int(padId))
            mask.append(0)
        }
        let inputIds = ids.map { Int32($0) }

        let inputArr = try MLMultiArray(shape: [1, NSNumber(value: maxTokens)], dataType: .int32)
        let maskArr  = try MLMultiArray(shape: [1, NSNumber(value: maxTokens)], dataType: .int32)
        for i in 0..<maxTokens {
            inputArr[i] = NSNumber(value: inputIds[i])
            maskArr[i]  = NSNumber(value: mask[i])
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids":      MLFeatureValue(multiArray: inputArr),
            "attention_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let output: MLFeatureProvider
        do {
            output = try await model.prediction(from: provider)
        } catch {
            throw TextSERError.underlying(error)
        }

        guard let logitsName = output.featureNames.first(where: { $0.contains("logits") || $0 == "output" }),
              let logitsArr = output.featureValue(for: logitsName)?.multiArrayValue else {
            throw TextSERError.modelUnavailable(reason: "no logits in DeBERTa output")
        }

        let count = min(logitsArr.count, Self.LABEL_ORDER.count)
        var raw: [Float] = []
        raw.reserveCapacity(count)
        for i in 0..<count { raw.append(logitsArr[i].floatValue) }
        let probs = Self.sigmoid(raw) // WRIME is multi-label; sigmoid per emotion.

        var dict: [PlutchikScore.Label: Float] = [:]
        for (i, label) in Self.LABEL_ORDER.prefix(count).enumerated() {
            dict[label] = probs[i]
        }
        return PlutchikScore(probabilities: dict)
    }

    private static func sigmoid(_ x: [Float]) -> [Float] {
        x.map { 1.0 / (1.0 + Foundation.exp(-$0)) }
    }
}
