import Foundation
import OnnxRuntimeBindings
import Audio
import SERRuntime
import XephonLogging

/// audeering/wav2vec2-large-robust-6-ft-age-gender → continuous age
/// regression + 3-class gender softmax. Same W2V2 backbone as
/// `W2V2DimensionalSER`; we ship our own ONNX export (built by
/// `scripts/export_w2v2_age_gender_onnx.py`) because the upstream
/// HF repo only publishes PyTorch weights with a custom head.
///
/// Model schema:
///   inputs:  speech         [1, time]  Float32   raw 16 kHz mono,
///                                                zero-mean / unit-var
///                                                normalized (the
///                                                checkpoint's
///                                                preprocessor_config.json
///                                                sets do_normalize: true)
///   outputs: logits_age     [1, 1]     Float32   regression in [0, 1]
///            logits_gender  [1, 3]     Float32   softmax in
///                                                [female, male, child]
///                                                order. Authoritative
///                                                source: the audeering
///                                                tutorial notebook at
///                                                github.com/audeering/
///                                                w2v2-age-gender-how-to,
///                                                cell 5, which states
///                                                "logits_gender …
///                                                expresses the
///                                                confidence for being
///                                                female, male or
///                                                child", and the
///                                                shipped config.json
///                                                id2label
///                                                {0: female, 1: male,
///                                                2: child}. The
///                                                Hugging Face model
///                                                card's prose ("child,
///                                                female, or male") and
///                                                its example output
///                                                column header are
///                                                simply wrong.
public actor W2V2AgeGenderSER: AgeGenderSER, BackgroundAwareSER {
    private var session: ORTSession
    private let modelURL: URL
    /// User's CoreML preference, frozen at init. See
    /// `Emotion2VecCategoricalSER.coreMLAllowed` for the rationale.
    private let coreMLAllowed: Bool
    private var usingCoreML: Bool

    /// Fixed output order of the gender softmax — must mirror the
    /// order of the export script's `gender` head columns. Used to
    /// turn the flat `[3]` ORT output into a label-keyed dict.
    private static let genderOrder: [AgeGenderEstimate.Gender] = [
        .female, .male, .child,
    ]

    public init(modelURL: URL, useCoreML: Bool = true) throws {
        self.modelURL = modelURL
        self.coreMLAllowed = useCoreML
        self.session = try Self.makeSession(modelURL: modelURL, useCoreML: useCoreML)
        let coreMLActive = useCoreML && ORTIsCoreMLExecutionProviderAvailable()
        self.usingCoreML = coreMLActive
        AppLog.serAcoustic.info(
            "W2V2 age-gender loaded: \(modelURL.lastPathComponent, privacy: .public) (CoreML EP: \(coreMLActive, privacy: .public))"
        )
    }

    /// Same lifecycle hook as the other acoustic actors. No-op when
    /// constructed CPU-only (current default in
    /// `AnalysisPipeline.autoConfigured`).
    public func setBackgroundMode(_ inBackground: Bool) async {
        let targetCoreML = !inBackground && coreMLAllowed
        guard targetCoreML != usingCoreML else { return }
        do {
            session = try Self.makeSession(modelURL: modelURL, useCoreML: targetCoreML)
            usingCoreML = targetCoreML && ORTIsCoreMLExecutionProviderAvailable()
            AppLog.serAcoustic.info(
                "W2V2 age-gender session → \(self.usingCoreML ? "CoreML" : "CPU", privacy: .public) (inBackground=\(inBackground, privacy: .public))"
            )
        } catch {
            AppLog.serAcoustic.warning(
                "W2V2 age-gender session swap failed (inBackground=\(inBackground, privacy: .public)): \(String(describing: error), privacy: .public); keeping current session"
            )
        }
    }

    private static func makeSession(modelURL: URL, useCoreML: Bool) throws -> ORTSession {
        let env = ORTRuntime.sharedEnv
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        // Same `Extended` cap rationale as `W2V2DimensionalSER` —
        // the strict `All` tier's SimplifiedLayerNormFusion pass
        // doesn't walk past the precision-free cast nodes the FP16
        // converter inserts, and that breaks session load.
        try options.setGraphOptimizationLevel(.extended)
        if useCoreML, ORTIsCoreMLExecutionProviderAvailable() {
            let coreml = ORTCoreMLExecutionProviderOptions()
            coreml.enableOnSubgraphs = true
            try? options.appendCoreMLExecutionProvider(with: coreml)
        }
        return try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
    }

    public func estimate(_ buffer: AudioChunk) async throws -> AgeGenderEstimate {
        let n = buffer.samples.count
        guard n > 0 else {
            throw AcousticSERError.onnxRuntimeFailure(reason: "empty audio")
        }
        let (age, gender) = try runInference(samples: buffer.samples, frameCount: n)
        guard gender.count == Self.genderOrder.count else {
            throw AcousticSERError.onnxRuntimeFailure(
                reason: "expected \(Self.genderOrder.count) gender logits, got \(gender.count)"
            )
        }
        var probs: [AgeGenderEstimate.Gender: Float] = [:]
        for (i, label) in Self.genderOrder.enumerated() {
            probs[label] = gender[i]
        }
        return AgeGenderEstimate(age: age, genderProbabilities: probs)
    }

    /// One-shot inference with the same CoreML→CPU fallback the
    /// V/A/D adapter uses; the failure mode (GPU access revoked
    /// in background) is identical so the recovery is too.
    private func runInference(
        samples: [Float],
        frameCount n: Int
    ) throws -> (age: Float, gender: [Float]) {
        do {
            return try invoke(samples: samples, frameCount: n)
        } catch {
            guard usingCoreML else { throw error }
            AppLog.serAcoustic.warning(
                "W2V2 age-gender CoreML EP failed (\(String(describing: error), privacy: .public)); rebuilding session on CPU"
            )
            session = try Self.makeSession(modelURL: modelURL, useCoreML: false)
            usingCoreML = false
            return try invoke(samples: samples, frameCount: n)
        }
    }

    private func invoke(
        samples: [Float],
        frameCount n: Int
    ) throws -> (age: Float, gender: [Float]) {
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
            withInputs: ["speech": inputValue],
            outputNames: ["logits_age", "logits_gender"],
            runOptions: nil
        )
        guard let ageOut = outputs["logits_age"],
              let genderOut = outputs["logits_gender"] else {
            throw AcousticSERError.onnxRuntimeFailure(
                reason: "missing age-gender outputs"
            )
        }
        let ageData = try ageOut.tensorData() as Data
        let ageScalars = ageData.withUnsafeBytes { ptr -> [Float] in
            Array(ptr.bindMemory(to: Float.self))
        }
        let genderData = try genderOut.tensorData() as Data
        let genderScalars = genderData.withUnsafeBytes { ptr -> [Float] in
            Array(ptr.bindMemory(to: Float.self))
        }
        guard let age = ageScalars.first else {
            throw AcousticSERError.onnxRuntimeFailure(reason: "no age scalar")
        }
        return (age: age, gender: genderScalars)
    }
}
