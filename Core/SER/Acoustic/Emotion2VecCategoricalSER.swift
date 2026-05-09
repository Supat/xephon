import Foundation
import OnnxRuntimeBindings
import Audio
import XephonLogging

// emotion2vec_plus_large → 9-class softmax (angry, disgusted, fearful, happy,
// neutral, other, sad, surprised, unknown). Cross-lingual; useful as a second
// opinion alongside W2V2 dimensional, especially given Japanese pitch-accent
// vs. emotional-prosody confounds.
//
// The upstream weights (`Models/emotion2vec-plus-large/model.pt`) are FunASR
// PyTorch checkpoints — not directly loadable here. Convert to ONNX once via:
//
//   python -m funasr.bin.export \
//       --model emotion2vec_plus_large \
//       --type onnx \
//       --output Models/emotion2vec-plus-large/
//
// Expected ONNX schema (matches FunASR's emotion2vec export):
//   inputs:  speech         [1, samples]  Float32   (raw 16 kHz mono)
//            speech_lengths [1]           Int32
//   outputs: logits         [1, 9]        Float32   order = LABEL_ORDER
public actor Emotion2VecCategoricalSER: CategoricalAcousticSER {
    private static let LABEL_ORDER: [CategoricalEmotion.Label] = [
        .angry, .disgusted, .fearful, .happy, .neutral,
        .other, .sad, .surprised, .unknown
    ]

    private var session: ORTSession
    private let modelURL: URL
    private var usingCoreML: Bool

    public init(modelURL: URL, useCoreML: Bool = true) throws {
        self.modelURL = modelURL
        self.session = try Self.makeSession(modelURL: modelURL, useCoreML: useCoreML)
        let coreMLActive = useCoreML && ORTIsCoreMLExecutionProviderAvailable()
        self.usingCoreML = coreMLActive
        AppLog.serAcoustic.info(
            "emotion2vec ONNX loaded: \(modelURL.lastPathComponent, privacy: .public) (CoreML EP: \(coreMLActive, privacy: .public))"
        )
    }

    private static func makeSession(modelURL: URL, useCoreML: Bool) throws -> ORTSession {
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        if useCoreML, ORTIsCoreMLExecutionProviderAvailable() {
            let coreml = ORTCoreMLExecutionProviderOptions()
            coreml.enableOnSubgraphs = true
            try? options.appendCoreMLExecutionProvider(with: coreml)
        }
        return try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
    }

    public init(useCoreML: Bool = true) throws {
        // Bundle layout produced by scripts/export_emotion2vec_onnx.py:
        //   Xephon.app/emotion2vec_onnx/model.onnx (graph)
        //   Xephon.app/emotion2vec_onnx/model.data (external weights)
        // Both files must live in the same directory so ORT's external-data
        // loader can resolve the relative reference.
        let url = Bundle.main.url(
            forResource: "model",
            withExtension: "onnx",
            subdirectory: "emotion2vec_onnx"
        )
        guard let modelURL = url else {
            throw AcousticSERError.modelUnavailable(
                reason: "emotion2vec_onnx/model.onnx not in app bundle (run scripts/fetch_models.sh, which calls scripts/export_emotion2vec_onnx.py)"
            )
        }
        try self.init(modelURL: modelURL, useCoreML: useCoreML)
    }

    public func score(_ buffer: AudioChunk) async throws -> CategoricalEmotion {
        let n = buffer.samples.count
        guard n > 0 else {
            throw AcousticSERError.onnxRuntimeFailure(reason: "empty audio")
        }
        let raw = try runInference(samples: buffer.samples, frameCount: n)
        guard raw.count >= Self.LABEL_ORDER.count else {
            throw AcousticSERError.onnxRuntimeFailure(
                reason: "expected \(Self.LABEL_ORDER.count) classes, got \(raw.count)"
            )
        }
        let probs = Self.softmax(Array(raw.prefix(Self.LABEL_ORDER.count)))
        var dict: [CategoricalEmotion.Label: Float] = [:]
        for (idx, label) in Self.LABEL_ORDER.enumerated() {
            dict[label] = probs[idx]
        }
        return CategoricalEmotion(probabilities: dict)
    }

    /// One-shot inference with automatic CPU fallback on first CoreML EP
    /// failure. See W2V2DimensionalSER.runInference for rationale.
    private func runInference(samples: [Float], frameCount n: Int) throws -> [Float] {
        do {
            return try invoke(samples: samples, frameCount: n)
        } catch {
            guard usingCoreML else { throw error }
            AppLog.serAcoustic.warning(
                "emotion2vec CoreML EP failed (\(String(describing: error), privacy: .public)); rebuilding session on CPU"
            )
            session = try Self.makeSession(modelURL: modelURL, useCoreML: false)
            usingCoreML = false
            return try invoke(samples: samples, frameCount: n)
        }
    }

    private func invoke(samples: [Float], frameCount n: Int) throws -> [Float] {
        let bytes = n * MemoryLayout<Float>.size
        let inputData = samples.withUnsafeBufferPointer { src -> NSMutableData in
            NSMutableData(bytes: src.baseAddress, length: bytes)
        }
        let inputValue = try ORTValue(
            tensorData: inputData,
            elementType: .float,
            shape: [1 as NSNumber, NSNumber(value: n)]
        )
        // The exported graph takes only `speech` — utterance-level layer
        // norm + encoder + mean-pool + linear head are all baked in.
        let outputs = try session.run(
            withInputs: ["speech": inputValue],
            outputNames: ["logits"],
            runOptions: nil
        )
        guard let logits = outputs["logits"] else {
            throw AcousticSERError.onnxRuntimeFailure(reason: "no logits output")
        }
        let info = try logits.tensorTypeAndShapeInfo()
        let outData = try logits.tensorData() as Data
        let raw = outData.withUnsafeBytes { ptr -> [Float] in
            Array(ptr.bindMemory(to: Float.self))
        }
        // FunASR exports may emit [batch, time, classes] (per-frame logits
        // pre-pooling) or already-pooled [batch, classes]. Mean-pool over
        // the time axis when 3D so callers always see a single 9-vector.
        return Self.poolToUtteranceLogits(raw: raw, shape: info.shape.map { $0.intValue })
    }

    private static func poolToUtteranceLogits(raw: [Float], shape: [Int]) -> [Float] {
        guard shape.count >= 2 else { return raw }
        // 2-D: already utterance-level. Drop leading batch dim.
        if shape.count == 2 {
            let classes = shape[1]
            return Array(raw.prefix(classes))
        }
        // 3-D [batch, time, classes]: average over time within batch 0.
        let time = shape[1]
        let classes = shape[2]
        guard time > 0, classes > 0, raw.count >= time * classes else { return raw }
        var out = [Float](repeating: 0, count: classes)
        for t in 0..<time {
            let base = t * classes
            for c in 0..<classes { out[c] += raw[base + c] }
        }
        let inv = 1.0 / Float(time)
        for c in 0..<classes { out[c] *= inv }
        return out
    }

    private static func softmax(_ x: [Float]) -> [Float] {
        guard let m = x.max() else { return x }
        let exps = x.map { Foundation.exp($0 - m) }
        let s = exps.reduce(0, +)
        guard s > 0 else { return Array(repeating: 0, count: x.count) }
        return exps.map { $0 / s }
    }
}
