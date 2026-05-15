import Foundation
import OnnxRuntimeBindings
import Audio
import SERRuntime
import XephonLogging

// audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim → V/A/D in [0, 1].
// Distributed as ONNX on Zenodo (doi:10.5281/zenodo.6221127). Run via
// onnxruntime-swift-package-manager with the CoreML execution provider, which
// targets ANE on M-series silicon and falls back to CPU otherwise.
//
// Model schema (verified against the Zenodo .onnx):
//   inputs:  signal  [1, time]   Float32   (raw 16 kHz mono)
//   outputs: logits  [1, 3]      Float32   order = arousal, dominance, valence
//            hidden_states [1, 1024] Float32  (unused here; useful for fusion)
public actor W2V2DimensionalSER: DimensionalAcousticSER, BackgroundAwareSER {
    private var session: ORTSession
    private let modelURL: URL
    /// User's CoreML preference, frozen at init. See
    /// `Emotion2VecCategoricalSER.coreMLAllowed` for the rationale —
    /// distinguishes intent from the live session's EP.
    private let coreMLAllowed: Bool
    private var usingCoreML: Bool

    /// Initialize from an explicit `.onnx` URL.
    public init(modelURL: URL, useCoreML: Bool = true) throws {
        self.modelURL = modelURL
        self.coreMLAllowed = useCoreML
        self.session = try Self.makeSession(modelURL: modelURL, useCoreML: useCoreML)
        let coreMLActive = useCoreML && ORTIsCoreMLExecutionProviderAvailable()
        self.usingCoreML = coreMLActive
        AppLog.serAcoustic.info(
            "W2V2 ONNX loaded: \(modelURL.lastPathComponent, privacy: .public) (CoreML EP: \(coreMLActive, privacy: .public))"
        )
    }

    /// Same lifecycle hook as Emotion2VecCategoricalSER. No-op
    /// when the actor was constructed CPU-only (current default in
    /// `AnalysisPipeline.autoConfigured`).
    public func setBackgroundMode(_ inBackground: Bool) async {
        let targetCoreML = !inBackground && coreMLAllowed
        guard targetCoreML != usingCoreML else { return }
        do {
            session = try Self.makeSession(modelURL: modelURL, useCoreML: targetCoreML)
            usingCoreML = targetCoreML && ORTIsCoreMLExecutionProviderAvailable()
            AppLog.serAcoustic.info(
                "W2V2 V/A/D session → \(self.usingCoreML ? "CoreML" : "CPU", privacy: .public) (inBackground=\(inBackground, privacy: .public))"
            )
        } catch {
            AppLog.serAcoustic.warning(
                "W2V2 V/A/D session swap failed (inBackground=\(inBackground, privacy: .public)): \(String(describing: error), privacy: .public); keeping current session"
            )
        }
    }

    private static func makeSession(modelURL: URL, useCoreML: Bool) throws -> ORTSession {
        // Process-wide ORTEnv. See `SERRuntime.ORTRuntime` for why.
        let env = ORTRuntime.sharedEnv
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        // Cap graph optimization at `Extended`. The default `All` tier
        // includes the SimplifiedLayerNormFusion pass, which doesn't know
        // how to walk past the auto-inserted `InsertedPrecisionFreeCast_*`
        // nodes the FP16 converter places around blocked ops — it crashes
        // session init with "name does not exist". `Extended` keeps every
        // useful optimization (constant folding, common subexpression
        // elimination, layer-norm-without-fusion) but skips the strict
        // tier that conflicts with our quantization layout.
        try options.setGraphOptimizationLevel(.extended)
        if useCoreML, ORTIsCoreMLExecutionProviderAvailable() {
            let coreml = ORTCoreMLExecutionProviderOptions()
            coreml.enableOnSubgraphs = true
            try? options.appendCoreMLExecutionProvider(with: coreml)
        }
        return try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
    }

    /// Convenience: load from the app bundle.
    /// Looks for `model.onnx` under either a `w2v2-msp-dim` subdirectory or at the bundle root.
    public init(useCoreML: Bool = true) throws {
        let url = Bundle.main.url(
            forResource: "model",
            withExtension: "onnx",
            subdirectory: "w2v2-msp-dim"
        ) ?? Bundle.main.url(forResource: "model", withExtension: "onnx")
        guard let modelURL = url else {
            throw AcousticSERError.modelUnavailable(
                reason: "audeering W2V2 model.onnx not in app bundle (run scripts/fetch_models.sh and add Models/w2v2-msp-dim/model.onnx to the app target)"
            )
        }
        try self.init(modelURL: modelURL, useCoreML: useCoreML)
    }

    public func score(_ buffer: AudioChunk) async throws -> VADScore {
        let n = buffer.samples.count
        guard n > 0 else {
            throw AcousticSERError.onnxRuntimeFailure(reason: "empty audio")
        }
        let scalars = try runInference(samples: buffer.samples, frameCount: n)
        guard scalars.count >= 3 else {
            throw AcousticSERError.onnxRuntimeFailure(reason: "expected 3 logits, got \(scalars.count)")
        }
        // model.yaml: labels = [arousal, dominance, valence]
        return VADScore(
            valence: scalars[2],
            arousal: scalars[0],
            dominance: scalars[1]
        )
    }

    /// One-shot inference with reactive CPU fallback. Defense in
    /// depth: the proactive `setBackgroundMode(_:)` swap should
    /// have already moved the session to CPU before iOS revokes
    /// GPU access, but if that swap missed (race with a sudden
    /// occlusion, or the rebuild itself threw and the session
    /// stayed on CoreML), this catch rebuilds on CPU and retries
    /// the call. The next foreground transition's
    /// `setBackgroundMode(false)` will move the session back to
    /// CoreML, so this isn't a permanent pin.
    private func runInference(samples: [Float], frameCount n: Int) throws -> [Float] {
        do {
            return try invoke(samples: samples, frameCount: n)
        } catch {
            guard usingCoreML else { throw error }
            AppLog.serAcoustic.warning(
                "W2V2 CoreML EP failed (\(String(describing: error), privacy: .public)); rebuilding session on CPU"
            )
            session = try Self.makeSession(modelURL: modelURL, useCoreML: false)
            usingCoreML = false
            return try invoke(samples: samples, frameCount: n)
        }
    }

    private func invoke(samples: [Float], frameCount n: Int) throws -> [Float] {
        // audeering preprocessor_config.json sets `do_normalize: true`;
        // feeding raw audio shifts the input distribution off training.
        let normalized = Wav2Vec2Preprocess.normalize(samples)
        let bytes = n * MemoryLayout<Float>.size
        let inputData = normalized.withUnsafeBufferPointer { src -> NSMutableData in
            NSMutableData(bytes: src.baseAddress, length: bytes)
        }
        let inputValue = try ORTValue(
            tensorData: inputData,
            elementType: .float,
            shape: [1 as NSNumber, NSNumber(value: n)]
        )
        let outputs = try session.run(
            withInputs: ["signal": inputValue],
            outputNames: ["logits"],
            runOptions: nil
        )
        guard let logits = outputs["logits"] else {
            throw AcousticSERError.onnxRuntimeFailure(reason: "no logits output")
        }
        let outData = try logits.tensorData() as Data
        return outData.withUnsafeBytes { ptr -> [Float] in
            Array(ptr.bindMemory(to: Float.self))
        }
    }
}
