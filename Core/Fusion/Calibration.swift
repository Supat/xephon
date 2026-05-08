import Foundation

// Per-target calibration applied after late fusion. Mapping from raw model
// outputs to the project's reporting scale lives here.
public struct Calibration: Sendable, Hashable {
    // Linear remap of [0,1] dimensional outputs onto the project's reporting scale.
    public let valenceScale: Float
    public let valenceOffset: Float
    public let arousalScale: Float
    public let arousalOffset: Float
    public let dominanceScale: Float
    public let dominanceOffset: Float

    public static let identity = Calibration(
        valenceScale: 1, valenceOffset: 0,
        arousalScale: 1, arousalOffset: 0,
        dominanceScale: 1, dominanceOffset: 0
    )

    public init(
        valenceScale: Float,
        valenceOffset: Float,
        arousalScale: Float,
        arousalOffset: Float,
        dominanceScale: Float,
        dominanceOffset: Float
    ) {
        self.valenceScale = valenceScale
        self.valenceOffset = valenceOffset
        self.arousalScale = arousalScale
        self.arousalOffset = arousalOffset
        self.dominanceScale = dominanceScale
        self.dominanceOffset = dominanceOffset
    }
}
