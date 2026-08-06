import Foundation
import MLX
import MLXLMCommon

/// Reusable KV-cache state for consecutive MLX generation calls
/// whose prompts share a leading token run.
///
/// The plugin fill pipeline is prefill-dominated (4–9 s per call
/// on-device). Today the win is the A1 Eval parse-failure retry,
/// which re-sends its prompt verbatim and reuses all but the tail
/// token. A rows-first prompt layout that also let the Tier 3a
/// preference call share the item call's rows (measured 3.4×
/// faster on that call in the Mac harness) regressed extraction
/// quality in the field and was reverted — if a future prompt
/// redesign reintroduces a shared head, gate it on the
/// ground-truth eval harness; this cache needs no code change to
/// benefit.
///
/// Reuse note: a warm prefill starts mid-sequence, so its Metal
/// reduction orders differ from a cold run's — greedy outputs can
/// diverge at near-ties (observed: one synonym swap late in a
/// 256-token generation). Equivalent quality, not bit-identical.
///
/// Reuse is verified, never assumed (the discipline is borrowed
/// from TurboFieldfare's verified-prefix prompt cache): the number
/// of skipped tokens is the exact longest common prefix of the two
/// calls' token arrays, additionally clamped to what the KV cache
/// physically holds (`offset` — a cancelled prefill leaves it short
/// of the recorded tokens), and the cache is trimmed back to that
/// point before reuse. Any state that can't be verified or trimmed
/// is discarded in favor of a fresh prefill — worst case is the
/// old behavior, never a corrupt one.
final class MLXPromptPrefixCache: @unchecked Sendable {
    // @unchecked: an instance is owned by a single summarizer actor
    // whose generate calls are serialized, and is only touched
    // inside `ModelContainer.perform` while that actor awaits the
    // result — there is no concurrent access path. (KVCache /
    // MLXArray are not Sendable, so the compiler can't see this.)

    /// Token ids of the last call's full prompt. The KV cache holds
    /// at least `min(storedTokens.count, offset)` of these, plus
    /// whatever the model generated after them (trimmed off before
    /// the next reuse).
    private var storedTokens: [Int] = []
    private var kv: [KVCache] = []

    /// Drop all cached state (model unload, batch end).
    func reset() {
        storedTokens = []
        kv = []
    }

    /// Hand back a KV cache primed with `reused` leading tokens of
    /// `fullTokens` — either the previous call's cache trimmed to
    /// the verified shared prefix, or a fresh empty cache when
    /// nothing (safe) is reusable. Records `fullTokens` as the new
    /// stored prompt either way.
    func adopt(
        fullTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters
    ) -> (cache: [KVCache], reused: Int) {
        let offset = kv.first?.offset ?? 0
        let reusable = Self.reusableTokenCount(
            full: fullTokens,
            stored: storedTokens,
            cacheOffset: offset
        )
        storedTokens = fullTokens
        if reusable > 0, !kv.isEmpty, canTrimPromptCache(kv) {
            let excess = offset - reusable
            if excess > 0 {
                // Per-layer, NOT MLXLMCommon.trimPromptCache: in the
                // pinned 2.29.1 that helper trims only cache.first
                // (upstream bug — the Python mlx_lm loops over all
                // layers), which left layers 1…N-1 untrimmed, failed
                // the verify below, and silently disabled reuse on
                // every call (found via the Mac repro harness).
                for layerCache in kv {
                    _ = layerCache.trim(excess)
                }
            }
            // Trust the cache's own account of what it holds, not
            // ours — a cache type that under-trims is discarded.
            if kv.allSatisfy({ $0.offset == reusable }) {
                return (kv, reusable)
            }
        }
        kv = model.newCache(parameters: parameters)
        return (kv, 0)
    }

    /// Leading tokens of `full` that a cache holding `stored` up to
    /// `cacheOffset` can stand in for. Clamped to `full.count - 1`
    /// so the iterator always has at least one prompt token left to
    /// process (a verbatim retry would otherwise leave it none).
    static func reusableTokenCount(
        full: [Int],
        stored: [Int],
        cacheOffset: Int
    ) -> Int {
        var common = 0
        while common < full.count, common < stored.count,
              full[common] == stored[common] {
            common += 1
        }
        return max(0, min(common, cacheOffset, full.count - 1))
    }
}
