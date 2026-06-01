import Foundation

/// Per-utterance informativeness scorer. Outputs a 0..1
/// relative score for each utterance in a session, where
/// higher = more semantically distinctive (less likely to be
/// a backchannel / filler / short acknowledgment).
///
/// Computed via TF-IDF over the session's own transcripts,
/// modulated by a Japanese-aware backchannel/filler penalty.
/// Pure deterministic compute — no LLM, no network, runs in
/// milliseconds even on hundreds of utterances.
///
/// **What "informative" means here.** Scores reflect
/// **distinctive vocabulary**, not **semantic importance**.
/// A verbose speaker who casually drops "辞めることにしました"
/// ("I quit my job") once between paragraphs of small talk
/// will see that line underranked because their own session-
/// level IDF is washed out by everything else they said. For
/// semantic ranking you'd add an LLM re-rank pass over the
/// top-N TF-IDF candidates.
///
/// **Design choice — not a field on `UtteranceEstimate`.**
/// Scores are kept as a separate `[UUID: Double]` dict rather
/// than stamped onto `UtteranceEstimate.salience` for three
/// reasons: (1) score is purely derivative, can be recomputed
/// in milliseconds; (2) score is session-relative — adding or
/// removing one utterance changes every other utterance's
/// score, so the field would constantly go stale; (3)
/// persisting in the `.xph` bundle / JSON export would bloat
/// the schema with a value that should be recomputed at read
/// time anyway. Callers that want to render scores cache the
/// dict themselves alongside whatever state already drives a
/// re-render.
///
/// **Two natural uses in this app:**
/// - Transcript UI: sort/highlight by score, or a "show top
///   N most informative" filter for skimming long sessions.
/// - Summarizer prompt selection: feed the LLM the top-N
///   most informative rows instead of the trailing-window
///   truncation. Worth especially on Llama where prompt
///   token budget is the dominant cost.
public enum Informativeness {
    /// Score every utterance and return a `[utteranceID:
    /// score]` map where score ∈ [0, 1]. Score is relative to
    /// the input — the highest-scoring utterance in the
    /// session gets 1.0, everything else is proportional.
    /// Returns an empty dictionary when `utterances` is
    /// empty; returns `{id: 0}` for every id when no token
    /// has any IDF mass (e.g. a single utterance with all-
    /// backchannel content).
    ///
    /// `boostedIDs` — utterance IDs the caller has flagged as
    /// containing one or more user-curated keywords (matched
    /// per the caller's preferred normalizer). Each scored
    /// utterance in this set is multiplied by `keywordBoost`
    /// (default 4.0) BEFORE the session-relative `[0,1]`
    /// normalization, so keyword hits routinely dominate the
    /// top of the ranking. Pre-computed by the caller so this
    /// module stays free of locale-specific normalization
    /// (cross-script matching, Hepburn romaji, etc. live in
    /// the caller — `Xephon/Search/JapaneseSearchNormalizer`
    /// is the canonical one).
    public static func score(
        utterances: [UtteranceEstimate],
        boostedIDs: Set<UUID> = []
    ) -> [UUID: Double] {
        guard !utterances.isEmpty else { return [:] }
        let tokenizations: [(UUID, [String])] = utterances.map {
            ($0.id, tokenize($0.transcript))
        }
        let idf = computeIDF(tokenizations: tokenizations.map { $0.1 })
        var raw: [UUID: Double] = [:]
        raw.reserveCapacity(utterances.count)
        for (id, tokens) in tokenizations {
            var s = rawScore(tokens: tokens, idf: idf)
            if boostedIDs.contains(id) {
                s *= Self.keywordBoost
            }
            raw[id] = s
        }
        guard let maxScore = raw.values.max(), maxScore > 0 else {
            return Dictionary(uniqueKeysWithValues: raw.keys.map { ($0, 0.0) })
        }
        return raw.mapValues { $0 / maxScore }
    }

    /// Multiplier applied to an utterance's raw score when its
    /// ID appears in the caller's `boostedIDs` set (i.e. it
    /// contains a user-curated keyword). 4.0 is "heavy" — a
    /// keyword-hit utterance typically jumps above all non-
    /// keyword rows in the session-normalized output. The
    /// factor is large enough to dominate but small enough
    /// that a session full of keyword hits still distinguishes
    /// the most informative among them by their underlying
    /// TF-IDF mass (vs. a flat 0/1 boost which would lose all
    /// quality ordering inside the boosted subset).
    private static let keywordBoost: Double = 4.0

    /// Return the set of utterance IDs in the top-N by
    /// informativeness. The set is unordered (matches callers
    /// that want to filter while preserving the input's
    /// chronological order via `utterances.filter { ids.contains($0.id) }`).
    /// Returns an empty set when `n <= 0` or `utterances` is
    /// empty.
    public static func topN(
        _ n: Int,
        utterances: [UtteranceEstimate],
        boostedIDs: Set<UUID> = []
    ) -> Set<UUID> {
        guard n > 0, !utterances.isEmpty else { return [] }
        let scores = score(utterances: utterances, boostedIDs: boostedIDs)
        let sortedByScore = scores
            .sorted { $0.value > $1.value }
            .prefix(n)
        return Set(sortedByScore.map { $0.key })
    }

    /// Speaker-balanced top-N selector. Guarantees each speaker
    /// gets at least one slot (when `n >= speakerCount`), then
    /// distributes the remaining budget proportionally by each
    /// speaker's total informativeness contribution (sum of
    /// their utterances' scores — captures both volume AND
    /// quality in one weight, vs. e.g. raw utterance count
    /// which over-rewards verbose-but-low-information speakers).
    /// Caps at `maxSpeakers` for defensive scaling; if the input
    /// has more speakers than the cap, drops the lowest-total-
    /// weight ones first.
    ///
    /// Returned IDs are unordered — callers restore chronology
    /// via `utterances.filter { ids.contains($0.id) }` (matches
    /// the `topN` convention).
    ///
    /// **Allocation algorithm.** Largest-remainder (Hamilton)
    /// method on `n - speakerCount` after seeding 1 per speaker:
    /// 1. Each active speaker: 1 floor slot.
    /// 2. Hamilton over remaining budget by total-informativeness
    ///    weight: exact share = weight / Σweight × remaining;
    ///    each speaker floor(share); leftover distributed by
    ///    descending fractional remainder.
    /// 3. Cap each speaker at their actual utterance count; if
    ///    capped, mark saturated and redistribute the surplus
    ///    among non-saturated speakers (re-run Hamilton on
    ///    surplus). Iterate to a fixed point (max
    ///    `maxSpeakers` rounds since each round saturates ≥ 1
    ///    speaker or exits).
    /// 4. Per speaker, pick top-K by per-utterance
    ///    informativeness.
    ///
    /// Fallback: when `n < speakerCount`, the "include every
    /// speaker" goal is impossible — degenerate to `topN(n:)`
    /// (pure global ranking, no balancing).
    public static func topNBalancedBySpeaker(
        _ n: Int,
        utterances: [UtteranceEstimate],
        boostedIDs: Set<UUID> = [],
        maxSpeakers: Int = 10
    ) -> Set<UUID> {
        guard n > 0, !utterances.isEmpty else { return [] }
        if n >= utterances.count {
            return Set(utterances.map { $0.id })
        }
        let scores = score(utterances: utterances, boostedIDs: boostedIDs)
        let bySpeaker = Dictionary(grouping: utterances, by: \.speakerID)
        var speakers = utterances.orderedSpeakerIDs
        if speakers.count > maxSpeakers {
            // Defensive cap. In normal sessions speaker count is
            // well under 10; this branch is for the rare case
            // where diarization over-segments.
            let weights = speakerWeights(
                speakers: speakers,
                bySpeaker: bySpeaker,
                scores: scores
            )
            let kept = Set(
                weights
                    .sorted { $0.value > $1.value }
                    .prefix(maxSpeakers)
                    .map { $0.key }
            )
            speakers = speakers.filter { kept.contains($0) }
        }
        // Can't give every speaker a slot — give up on
        // balancing and use the pure global ranker. Caller's
        // selection sizing is too tight for the speaker count;
        // the speaker-balancing promise is impossible here.
        guard n >= speakers.count else {
            return topN(n, utterances: utterances, boostedIDs: boostedIDs)
        }
        // Seed: floor of 1 per speaker.
        var allocation: [String: Int] = [:]
        for speaker in speakers {
            allocation[speaker] = 1
        }
        var remaining = n - speakers.count
        var activeSpeakers = speakers
        // Iterative redistribution: a low-volume speaker can be
        // allocated more than they have. Cap at their count,
        // freed surplus goes back to non-saturated speakers'
        // next allocation round. At most `maxSpeakers`
        // iterations because each round either saturates ≥ 1
        // speaker or exits when nothing more can be assigned.
        while remaining > 0, !activeSpeakers.isEmpty {
            let activeWeights = speakerWeights(
                speakers: activeSpeakers,
                bySpeaker: bySpeaker,
                scores: scores
            )
            let extras = hamiltonAllocate(
                budget: remaining,
                speakers: activeSpeakers,
                weights: activeWeights
            )
            var distributedThisRound = 0
            var newlySaturated: [String] = []
            for speaker in activeSpeakers {
                let utteranceCount = bySpeaker[speaker]?.count ?? 0
                let extra = extras[speaker] ?? 0
                let current = allocation[speaker] ?? 0
                let target = current + extra
                let capped = min(target, utteranceCount)
                let actualAdded = capped - current
                allocation[speaker] = capped
                distributedThisRound += actualAdded
                if capped >= utteranceCount {
                    newlySaturated.append(speaker)
                }
            }
            remaining -= distributedThisRound
            if newlySaturated.isEmpty || distributedThisRound == 0 {
                // No saturation triggered this round AND budget
                // remains — every active speaker is already at
                // their natural cap, can't do better. Bail.
                break
            }
            activeSpeakers.removeAll { newlySaturated.contains($0) }
        }
        // Per speaker, take their top-K by informativeness.
        var picks: Set<UUID> = []
        for speaker in speakers {
            let count = allocation[speaker] ?? 0
            guard count > 0, let candidates = bySpeaker[speaker] else { continue }
            let sorted = candidates.sorted {
                (scores[$0.id] ?? 0) > (scores[$1.id] ?? 0)
            }
            picks.formUnion(sorted.prefix(count).map { $0.id })
        }
        return picks
    }

    /// Sum of per-utterance informativeness scores per speaker.
    /// Used as the allocation weight in `topNBalancedBySpeaker`
    /// because it folds two desirable signals into one number:
    /// (1) more utterances → larger sum (rewards contribution
    /// volume), (2) more distinctive utterances → larger
    /// individual terms (rewards quality). Mean would only
    /// capture quality; raw count would only capture volume.
    private static func speakerWeights(
        speakers: [String],
        bySpeaker: [String: [UtteranceEstimate]],
        scores: [UUID: Double]
    ) -> [String: Double] {
        var weights: [String: Double] = [:]
        weights.reserveCapacity(speakers.count)
        for speaker in speakers {
            let utts = bySpeaker[speaker] ?? []
            let sum = utts.reduce(0.0) { $0 + (scores[$1.id] ?? 0) }
            weights[speaker] = sum
        }
        return weights
    }

    /// Largest-remainder (Hamilton) integer allocation. Given a
    /// budget and per-bucket real-valued weights, returns an
    /// integer allocation that sums to `budget` and approximates
    /// each bucket's proportional share with the minimum
    /// rounding distortion. Standard apportionment algorithm.
    /// When all weights are zero (e.g. all-backchannel speakers
    /// in a session where every row was downweighted), falls
    /// back to as-even-as-possible distribution.
    private static func hamiltonAllocate(
        budget: Int,
        speakers: [String],
        weights: [String: Double]
    ) -> [String: Int] {
        guard budget > 0, !speakers.isEmpty else { return [:] }
        let total = weights.values.reduce(0, +)
        if total == 0 {
            let base = budget / speakers.count
            let remainder = budget % speakers.count
            var allocation: [String: Int] = [:]
            for (idx, speaker) in speakers.enumerated() {
                allocation[speaker] = base + (idx < remainder ? 1 : 0)
            }
            return allocation
        }
        var shares: [(speaker: String, share: Double)] = []
        shares.reserveCapacity(speakers.count)
        for speaker in speakers {
            let w = weights[speaker] ?? 0
            shares.append((speaker, w / total * Double(budget)))
        }
        var allocation: [String: Int] = [:]
        var distributed = 0
        for (speaker, share) in shares {
            let floorVal = Int(share)
            allocation[speaker] = floorVal
            distributed += floorVal
        }
        var leftover = budget - distributed
        let byFraction = shares.sorted {
            ($0.share - Double(Int($0.share))) > ($1.share - Double(Int($1.share)))
        }
        for entry in byFraction {
            if leftover <= 0 { break }
            allocation[entry.speaker, default: 0] += 1
            leftover -= 1
        }
        return allocation
    }

    /// Tokenize a Japanese (or mixed) transcript via Apple's
    /// built-in `String.enumerateSubstrings(... options: .byWords)`,
    /// which delegates to `CFStringTokenizer`. Not as morpho-
    /// logically aware as Mecab, but ships with the OS and
    /// segments Japanese acceptably for TF-IDF (over- and
    /// under-segmentation tend to wash out across the
    /// session's IDF aggregate). ASCII tokens are lowercased;
    /// CJK is preserved as-is. Empty / whitespace-only
    /// tokens are skipped.
    private static func tokenize(_ text: String) -> [String] {
        var tokens: [String] = []
        text.enumerateSubstrings(
            in: text.startIndex..<text.endIndex,
            options: .byWords
        ) { substring, _, _, _ in
            guard let s = substring else { return }
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return }
            tokens.append(trimmed.lowercased())
        }
        return tokens
    }

    /// Standard smoothed IDF across the per-utterance token
    /// lists:
    ///     idf(t) = log((1 + N) / (1 + df(t))) + 1
    /// where N = number of utterances and df(t) = number of
    /// utterances containing token t. The `+1` shift keeps
    /// every observed token at idf ≥ 1 (so a high-frequency
    /// content word still gets weighted) while the `1+`
    /// smoothing avoids `log(0)` on singletons.
    private static func computeIDF(
        tokenizations: [[String]]
    ) -> [String: Double] {
        var docFreq: [String: Int] = [:]
        for tokens in tokenizations {
            for token in Set(tokens) {
                docFreq[token, default: 0] += 1
            }
        }
        let n = Double(tokenizations.count)
        var idf: [String: Double] = [:]
        idf.reserveCapacity(docFreq.count)
        for (token, df) in docFreq {
            idf[token] = log((1.0 + n) / (1.0 + Double(df))) + 1.0
        }
        return idf
    }

    /// Score one utterance's tokens. Three pieces:
    /// 1. Sum the tokens' IDF values (content-word mass).
    /// 2. Normalize by `sqrt(tokenCount)` — sub-linear
    ///    length penalty. Linear normalization would over-
    ///    penalize natural-length utterances; no
    ///    normalization would let "はい はい はい そう そう"
    ///    outscore a short content utterance just by
    ///    repetition. `sqrt` is the BM25 / classical
    ///    information-retrieval middle ground.
    /// 3. Multiply by a backchannel penalty in `[0.1, 1.0]`
    ///    proportional to how much of the row is composed of
    ///    tokens in `backchannelTokens`. 100% backchannel
    ///    → 0.1 (kept above zero so a row composed entirely
    ///    of acknowledgments still gets a positive score —
    ///    callers can decide to threshold).
    private static func rawScore(
        tokens: [String],
        idf: [String: Double]
    ) -> Double {
        guard !tokens.isEmpty else { return 0 }
        let summed = tokens.reduce(into: 0.0) { acc, token in
            acc += idf[token] ?? 0
        }
        let normalized = summed / sqrt(Double(tokens.count))
        let bcHits = tokens.filter { backchannelTokens.contains($0) }.count
        let bcRatio = Double(bcHits) / Double(tokens.count)
        let penalty = 1.0 - 0.9 * bcRatio
        return normalized * penalty
    }

    /// Common Japanese conversational backchannels, fillers,
    /// and short acknowledgments. Anything in this set down-
    /// weights the utterance proportional to its share of the
    /// row's tokens. Lowercased for parity with `tokenize`
    /// (which lowercases ASCII); CJK characters are preserved
    /// verbatim and matched by case-insensitive equality.
    ///
    /// List drawn from common JP conversational corpora (CSJ,
    /// CallHome, OGVC) backchannel inventories. Kept inline
    /// here because it's small and rarely changes; lifting to
    /// a resource file later is straightforward if it grows.
    private static let backchannelTokens: Set<String> = [
        // Affirmation / acknowledgment
        "うん", "ううん", "うんうん", "はい", "ええ", "そう",
        "そうそう", "そうなんだ", "そうですね", "そうだね",
        "なるほど", "へえ", "ふーん", "ふん", "ほー", "ほう",
        "おお", "おー", "あー", "あ", "あっ", "あぁ",
        // Hesitation / filler
        "えーと", "えーっと", "えっと", "えっとー", "あの",
        "あのー", "その", "そのー", "まあ", "まー", "まぁ",
        "ちょっと", "なんか", "なんていうか",
        // Yes / no
        "うんん", "いえ", "いいえ", "やー", "やあ",
        // Discourse markers commonly used as standalone
        // backchannels in conversational JP
        "ね", "よ", "ねえ", "やっぱり", "やっぱ",
        // Polite acknowledgments / very common short
        // openings that read as backchannel in dialogue
        "わかった", "わかりました", "了解", "確かに",
    ]
}
