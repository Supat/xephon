import Foundation
import OnnxRuntimeBindings
import Tokenizers
import Hub
import SERRuntime
import XephonLogging
import XephonUtilities

// On-device WRIME text-emotion classifier. Despite the type name (kept for
// historical continuity), the bundled artifact today is a fine-tuned
// Japanese RoBERTa-base regressor (MuneK/roberta-base-japanese-finetuned-wrime,
// ~110M params) — Takenaka 2025's DeBERTa-v3-large is the long-term target
// once a usable Core ML / ONNX export is published.
//
// Caveat per CLAUDE.md: WRIME-trained classifiers under-detect strong affect
// when the speaker uses 敬語 (politeness register).
//
// Bundle contract:
//   <subdir>/model.onnx           ONNX export of the fine-tuned model
//   <subdir>/tokenizer.json       HuggingFace fast-tokenizer file
//   <subdir>/tokenizer_config.json
//
// Conversion (off-device, requires `pip install optimum[onnxruntime]`):
//   optimum-cli export onnx \
//       --model MuneK/roberta-base-japanese-finetuned-wrime \
//       --task text-classification \
//       Models/wrime-roberta/
//
// Output schema (verified after export):
//   inputs:  input_ids       [batch, seq]  Int64
//            attention_mask  [batch, seq]  Int64
//   outputs: logits          [batch, 8]    Float32   regression intensities
//                                                    in roughly [0, 3]
public actor DeBERTaWRIME: TextSER, BackgroundAwareSER {
    private static let LABEL_ORDER: [PlutchikScore.Label] = [
        // Canonical WRIME-ver2 ordering for the 8-emotion intensity vector.
        .joy, .sadness, .anticipation, .surprise,
        .anger, .fear, .disgust, .trust
    ]

    private var session: ORTSession
    private let modelURL: URL
    /// User's CoreML preference, frozen at init. Distinguishes intent
    /// from `usingCoreML` (the live session's EP, toggled by the
    /// foreground/background observer).
    private let coreMLAllowed: Bool
    private var usingCoreML: Bool
    private let tokenizer: any Tokenizer
    private let maxTokens: Int
    private let intensityScale: Float

    public init(
        modelURL: URL,
        tokenizerDirectory: URL,
        maxTokens: Int = 128,
        intensityScale: Float = 3.0,
        useCoreML: Bool = true
    ) async throws {
        do {
            self.modelURL = modelURL
            // Try the user's preferred EP first; if CoreML init throws
            // — observed on macOS "Designed for iPad" where the EP
            // refuses this specific FP16 WRIME export — fall back to
            // a CPU-only session and freeze `coreMLAllowed` to false
            // so the foreground/background swap doesn't keep re-
            // attempting the doomed CoreML init. iPad keeps the
            // CoreML speedup; macOS keeps a working text-SER path.
            let resolvedSession: ORTSession
            let resolvedCoreMLAllowed: Bool
            if useCoreML {
                if let session = try? Self.makeSession(modelURL: modelURL, useCoreML: true) {
                    resolvedSession = session
                    resolvedCoreMLAllowed = true
                } else {
                    AppLog.serText.warning(
                        "DeBERTa CoreML EP init failed; retrying on CPU (likely macOS-via-Designed-for-iPad EP rejection)"
                    )
                    resolvedSession = try Self.makeSession(modelURL: modelURL, useCoreML: false)
                    resolvedCoreMLAllowed = false
                }
            } else {
                resolvedSession = try Self.makeSession(modelURL: modelURL, useCoreML: false)
                resolvedCoreMLAllowed = false
            }
            self.session = resolvedSession
            self.coreMLAllowed = resolvedCoreMLAllowed
            let coreMLActive = resolvedCoreMLAllowed && ORTIsCoreMLExecutionProviderAvailable()
            self.usingCoreML = coreMLActive

            let tokenizerConfigURL = tokenizerDirectory.appendingPathComponent("tokenizer_config.json")
            let tokenizerDataURL   = tokenizerDirectory.appendingPathComponent("tokenizer.json")
            let configData = try Data(contentsOf: tokenizerConfigURL)
            let tokenizerData = try Data(contentsOf: tokenizerDataURL)
            var configDict = try JSONSerialization.jsonObject(with: configData) as? [NSString: Any] ?? [:]
            let dataDict   = try JSONSerialization.jsonObject(with: tokenizerData) as? [NSString: Any] ?? [:]

            // swift-transformers `knownTokenizers` does not register
            // `AlbertTokenizer`, but the on-disk fast tokenizer is actually a
            // SentencePiece-Unigram + Metaspace tokenizer — identical in
            // shape to XLMRoberta. Re-tag the class so AutoTokenizer routes to
            // `UnigramTokenizer` instead of throwing `unsupportedTokenizer`.
            if let cls = configDict["tokenizer_class" as NSString] as? String,
               cls.contains("Albert") {
                configDict["tokenizer_class" as NSString] = "XLMRobertaTokenizer" as NSString
            }
            self.tokenizer = try AutoTokenizer.from(
                tokenizerConfig: Config(configDict),
                tokenizerData: Config(dataDict)
            )
            self.maxTokens = maxTokens
            self.intensityScale = intensityScale
            AppLog.serText.info(
                "DeBERTaWRIME ONNX loaded: \(modelURL.lastPathComponent, privacy: .public) (CoreML EP: \(coreMLActive, privacy: .public))"
            )
        } catch {
            throw TextSERError.modelUnavailable(reason: String(describing: error))
        }
    }

    /// Foreground/background lifecycle hook. Mirrors the acoustic
    /// SER actors: when the app moves out of `.active`, rebuild the
    /// ORT session on CPU so the next `classify(_:)` doesn't trip
    /// the CoreML EP "Insufficient Permission (to submit GPU work
    /// from background)" wall; rebuild on the user's preferred
    /// backend on return. No-op when the actor was constructed
    /// CPU-only (the current default in
    /// `AnalysisPipeline.autoConfigured`).
    public func setBackgroundMode(_ inBackground: Bool) async {
        let targetCoreML = !inBackground && coreMLAllowed
        guard targetCoreML != usingCoreML else { return }
        do {
            session = try Self.makeSession(modelURL: modelURL, useCoreML: targetCoreML)
            usingCoreML = targetCoreML && ORTIsCoreMLExecutionProviderAvailable()
            AppLog.serText.info(
                "DeBERTa session → \(self.usingCoreML ? "CoreML" : "CPU", privacy: .public) (inBackground=\(inBackground, privacy: .public))"
            )
        } catch {
            AppLog.serText.warning(
                "DeBERTa session swap failed (inBackground=\(inBackground, privacy: .public)): \(String(describing: error), privacy: .public); keeping current session"
            )
        }
    }

    private static func makeSession(modelURL: URL, useCoreML: Bool) throws -> ORTSession {
        // Process-wide ORTEnv. See `SERRuntime.ORTRuntime` for why.
        let env = ORTRuntime.sharedEnv
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        // See W2V2DimensionalSER for rationale: the default `All` tier
        // contains a layer-norm fusion that conflicts with FP16-conversion
        // Cast nodes. Cap at `Extended` for safety across all SER models.
        try options.setGraphOptimizationLevel(.extended)
        if useCoreML, ORTIsCoreMLExecutionProviderAvailable() {
            let coreml = ORTCoreMLExecutionProviderOptions()
            coreml.enableOnSubgraphs = true
            try? options.appendCoreMLExecutionProvider(with: coreml)
        }
        return try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
    }

    /// Convenience: load from the app bundle. Looks for a `wrime-roberta` subdir.
    public init(maxTokens: Int = 128, intensityScale: Float = 3.0) async throws {
        guard let modelURL = Bundle.main.url(
            forResource: "model",
            withExtension: "onnx",
            subdirectory: "wrime-roberta"
        ),
        let tokenizerJSON = Bundle.main.url(
            forResource: "tokenizer",
            withExtension: "json",
            subdirectory: "wrime-roberta"
        )
        else {
            throw TextSERError.modelUnavailable(
                reason: "wrime-roberta model.onnx + tokenizer.json not in app bundle (run optimum-cli export onnx — see source comment)"
            )
        }
        try await self.init(
            modelURL: modelURL,
            tokenizerDirectory: tokenizerJSON.deletingLastPathComponent(),
            maxTokens: maxTokens,
            intensityScale: intensityScale
        )
    }

    public func classify(_ text: String) async throws -> PlutchikScore {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return PlutchikScore(probabilities: [:])
        }

        // Tokenize, pad/truncate to fixed length (static shape ONNX graph).
        var ids = tokenizer.encode(text: trimmed, addSpecialTokens: true)
        if ids.count > maxTokens { ids = Array(ids.prefix(maxTokens)) }
        let realCount = ids.count
        let padId = tokenizer.unknownTokenId ?? 0
        while ids.count < maxTokens { ids.append(padId) }
        var mask: [Int64] = Array(repeating: 0, count: maxTokens)
        for i in 0..<realCount { mask[i] = 1 }
        let inputIds = ids.map { Int64($0) }

        let inputData = inputIds.withUnsafeBufferPointer { src -> NSMutableData in
            NSMutableData(bytes: src.baseAddress, length: maxTokens * MemoryLayout<Int64>.size)
        }
        let maskData = mask.withUnsafeBufferPointer { src -> NSMutableData in
            NSMutableData(bytes: src.baseAddress, length: maxTokens * MemoryLayout<Int64>.size)
        }
        let shape: [NSNumber] = [1, NSNumber(value: maxTokens)]
        let inputIdsValue = try ORTValue(tensorData: inputData, elementType: .int64, shape: shape)
        let maskValue     = try ORTValue(tensorData: maskData,  elementType: .int64, shape: shape)

        let inputs = ["input_ids": inputIdsValue, "attention_mask": maskValue]
        let outputs: [String: ORTValue]
        do {
            outputs = try session.run(withInputs: inputs, outputNames: ["logits"], runOptions: nil)
        } catch {
            // Defense in depth: the proactive `setBackgroundMode(_:)`
            // swap should have moved the session to CPU before iOS
            // revoked GPU access. If that swap missed (race, or its
            // own rebuild threw and left the session on CoreML),
            // this rebuilds on CPU and retries. The next foreground
            // `setBackgroundMode(false)` will restore CoreML — this
            // is not a permanent pin.
            guard usingCoreML else {
                throw TextSERError.underlying(error)
            }
            AppLog.serText.warning(
                "DeBERTa CoreML EP failed (\(String(describing: error), privacy: .public)); rebuilding session on CPU"
            )
            do {
                session = try Self.makeSession(modelURL: modelURL, useCoreML: false)
                usingCoreML = false
                outputs = try session.run(withInputs: inputs, outputNames: ["logits"], runOptions: nil)
            } catch {
                throw TextSERError.underlying(error)
            }
        }
        guard let logits = outputs["logits"] else {
            throw TextSERError.modelUnavailable(reason: "no logits in WRIME output")
        }
        let outData = try logits.tensorData() as Data
        let raw = outData.withUnsafeBytes { ptr -> [Float] in
            Array(ptr.bindMemory(to: Float.self))
        }

        let count = min(raw.count, Self.LABEL_ORDER.count)
        guard count > 0 else {
            throw TextSERError.modelUnavailable(reason: "empty logits")
        }
        var dict: [PlutchikScore.Label: Float] = [:]
        for i in 0..<count {
            // The model is a regression head trained on WRIME intensities in
            // [0, 3]. Normalize to [0, 1] for our PlutchikScore convention.
            let normalized = (raw[i] / intensityScale).clamped(to: 0...1)
            dict[Self.LABEL_ORDER[i]] = normalized
        }
        return PlutchikScore(probabilities: dict)
    }
}
