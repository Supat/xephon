import Testing
@testable import ASR

// Eval tests run against real audio fixtures (Tests/Fixtures/, LFS-tracked).
// SpeechTranscriber does not run on iOS Simulator — gate physical-device-only tests.
@Suite("ASR eval (placeholder)")
struct ASREvalPlaceholderTests {
    @Test
    func transcriberConformanceCompiles() async throws {
        let t: any Transcriber = SpeechAnalyzerTranscriber()
        await #expect(t.locale.identifier == "ja_JP")
    }
}
