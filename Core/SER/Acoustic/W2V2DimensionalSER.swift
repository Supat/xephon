import Foundation
import OnnxRuntimeBindings
import Audio
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
public actor W2V2DimensionalSER: DimensionalAcousticSER {
    private let session: ORTSession

    /// Initialize from an explicit `.onnx` URL.
    public init(modelURL: URL, useCoreML: Bool = true) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        if useCoreML, ORTIsCoreMLExecutionProviderAvailable() {
            let coreml = ORTCoreMLExecutionProviderOptions()
            coreml.enableOnSubgraphs = true
            // Best-effort: if the CoreML EP can't claim a subgraph it
            // transparently falls back to the CPU EP for that node.
            try? options.appendCoreMLExecutionProvider(with: coreml)
        }
        self.session = try ORTSession(
            env: env,
            modelPath: modelURL.path,
            sessionOptions: options
        )
        AppLog.serAcoustic.info(
            "W2V2 ONNX loaded: \(modelURL.lastPathComponent, privacy: .public) (CoreML EP: \(useCoreML && ORTIsCoreMLExecutionProviderAvailable(), privacy: .public))"
        )
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

        let bytes = n * MemoryLayout<Float>.size
        let inputData = buffer.samples.withUnsafeBufferPointer { src -> NSMutableData in
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
        let scalars = outData.withUnsafeBytes { ptr -> [Float] in
            Array(ptr.bindMemory(to: Float.self))
        }
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
}
