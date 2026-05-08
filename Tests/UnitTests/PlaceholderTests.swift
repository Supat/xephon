import Testing
@testable import Fusion

@Suite("Fusion placeholder")
struct FusionPlaceholderTests {
    @Test func calibrationIdentityIsLossless() async throws {
        let c = Calibration.identity
        #expect(c.valenceScale == 1)
        #expect(c.valenceOffset == 0)
        #expect(c.arousalScale == 1)
        #expect(c.arousalOffset == 0)
        #expect(c.dominanceScale == 1)
        #expect(c.dominanceOffset == 0)
    }
}
