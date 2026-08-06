import Testing
@testable import Summarizer

/// Pins the reuse arithmetic of the MLX prompt-prefix KV cache —
/// the pure part of `MLXPromptPrefixCache`. The invariant under
/// test: the reported count never exceeds (a) the verified common
/// prefix of the two token arrays, (b) what the KV cache actually
/// holds (`cacheOffset` — a cancelled prefill leaves it short of
/// the recorded tokens), or (c) `full.count - 1` (the iterator
/// must always keep at least one prompt token to process — the
/// verbatim-retry case would otherwise leave it none).
@Suite("MLX prompt-prefix cache reuse arithmetic")
struct MLXPromptPrefixCacheTests {

    @Test func divergentSuffixReusesOnlyTheCommonPrefix() {
        // Item call then preference call: same head, different tail.
        let count = MLXPromptPrefixCache.reusableTokenCount(
            full: [1, 2, 3, 9, 9],
            stored: [1, 2, 3, 4, 5, 6],
            cacheOffset: 6
        )
        #expect(count == 3)
    }

    @Test func verbatimRetryLeavesOneTokenForTheIterator() {
        // Parse-failure retry re-sends the identical prompt; the
        // cache may additionally hold generated tokens (offset >
        // stored count) — both are trimmed down to full.count - 1.
        let count = MLXPromptPrefixCache.reusableTokenCount(
            full: [1, 2, 3, 4],
            stored: [1, 2, 3, 4],
            cacheOffset: 10
        )
        #expect(count == 3)
    }

    @Test func cancelledPrefillClampsToWhatTheCacheHolds() {
        // Recorded tokens ran ahead of the KV state (prefill was
        // cancelled between chunks) — only the physically-held
        // prefix is reusable.
        let count = MLXPromptPrefixCache.reusableTokenCount(
            full: [1, 2, 3, 4, 5],
            stored: [1, 2, 3, 4, 5],
            cacheOffset: 2
        )
        #expect(count == 2)
    }

    @Test func coldCacheAndDisjointPromptsReuseNothing() {
        #expect(MLXPromptPrefixCache.reusableTokenCount(
            full: [1, 2, 3], stored: [], cacheOffset: 0
        ) == 0)
        #expect(MLXPromptPrefixCache.reusableTokenCount(
            full: [7, 8], stored: [1, 2, 3], cacheOffset: 3
        ) == 0)
    }

    @Test func emptyFullPromptNeverGoesNegative() {
        #expect(MLXPromptPrefixCache.reusableTokenCount(
            full: [], stored: [1, 2], cacheOffset: 2
        ) == 0)
    }

    @Test func fullPromptShorterThanStoredIsClampedToItsOwnTail() {
        // Next prompt is a strict prefix of the previous one —
        // reuse everything except the final token.
        let count = MLXPromptPrefixCache.reusableTokenCount(
            full: [1, 2, 3],
            stored: [1, 2, 3, 4, 5],
            cacheOffset: 5
        )
        #expect(count == 2)
    }
}
