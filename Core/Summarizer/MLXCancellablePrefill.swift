import Foundation
import MLX
import MLXLMCommon

/// Cancellable prompt prefill for the MLX summarize / review paths.
///
/// `MLXLMCommon.generate` prefills the prompt inside
/// `TokenIterator.init` → `model.prepare`, whose chunk loop has NO
/// cancellation points — a meeting-mode prompt (10k+ tokens) runs
/// seconds of GPU work that `Task.cancel()` cannot stop. That is
/// fatal on iOS backgrounding: the scenePhase watcher cancels the
/// inference Task, but the prefill keeps submitting Metal command
/// buffers and the first one submitted after the transition aborts
/// the process (`kIOGPUCommandBufferCallbackErrorBackground-
/// ExecutionNotPermitted`, uncaught C++ throw — observed on-device).
///
/// This helper mirrors `LLMModel.prepare`'s chunk loop with a
/// `Task.checkCancellation()` between chunks, then hands the primed
/// KV cache to a `TokenIterator` whose own prepare sees only the
/// ≤ one-chunk remainder. The un-cancellable window shrinks from
/// "the whole prompt" to one chunk (~tens of ms at step 128).
///
/// Text-only by design — the summarizer / reviewer prompts carry no
/// image/video input, and the VLM prepare path differs.
enum MLXCancellablePrefill {
    /// Prefill `input`'s prompt into a KV cache, checking for
    /// Task cancellation between chunks, and return the iterator to
    /// hand to `generate(input:context:iterator:didGenerate:)`
    /// together with the remainder input it was built from.
    ///
    /// When `prefixCache` is given, the leading tokens the prompt
    /// verifiably shares with the previous call on that cache are
    /// skipped instead of re-prefilled (see `MLXPromptPrefixCache`);
    /// `reusedTokens` reports how many were skipped (0 = cold).
    static func primedIterator(
        input: LMInput,
        context: ModelContext,
        parameters: GenerateParameters,
        prefixCache: MLXPromptPrefixCache? = nil
    ) throws -> (remaining: LMInput, iterator: TokenIterator, reusedTokens: Int) {
        let cache: [KVCache]
        let reused: Int
        if let prefixCache {
            (cache, reused) = prefixCache.adopt(
                fullTokens: input.text.tokens.asArray(Int.self),
                model: context.model,
                parameters: parameters
            )
        } else {
            cache = context.model.newCache(parameters: parameters)
            reused = 0
        }
        let step = parameters.prefillStepSize
        var text = input.text
        if reused > 0 {
            text = text[reused...]
        }
        while text.tokens.size > step {
            try Task.checkCancellation()
            _ = context.model(
                text[.newAxis, ..<step],
                cache: cache.isEmpty ? nil : cache,
                state: nil
            )
            eval(cache)
            text = text[step...]
        }
        try Task.checkCancellation()
        let remaining = LMInput(text: text)
        let iterator = try TokenIterator(
            input: remaining,
            model: context.model,
            cache: cache,
            parameters: parameters
        )
        return (remaining, iterator, reused)
    }
}
