import Foundation

/// The filled-form draft — what the plugin persists into the `.xph`
/// payload and what the review page renders. Every field follows
/// the stated-vs-inferred / null-first / evidence policy from
/// docs/eval_form_autofill_research.md §3: `strengthScore` is
/// non-nil ONLY when the evaluator stated it; model suggestions
/// live in the separate `strengthScoreInferred` and are visually
/// marked in the UI.
public struct EvalFormDraft: Codable, Sendable, Equatable {
    public struct ItemResult: Codable, Sendable, Equatable {
        public var itemID: String
        /// Stated relative-strength score (deterministic capture or
        /// LLM-extracted verbatim). Nil = not stated.
        public var strengthScore: Double?
        /// Model-suggested score from qualitative language. Kept
        /// apart from `strengthScore` by design.
        public var strengthScoreInferred: Double?
        /// Stated 1–9 preference. Nil = not stated.
        public var likeDislike: Int?
        /// Distilled per-item comment (ja). Nil = item not discussed.
        public var comment: String?
        /// 1-based row numbers (into the extraction's numbered row
        /// list) supporting the fields above.
        public var evidenceRows: [Int]
        /// Human-readable conflict notes (two different stated
        /// scores, deterministic/LLM disagreement, …).
        public var conflicts: [String]

        public init(
            itemID: String,
            strengthScore: Double? = nil,
            strengthScoreInferred: Double? = nil,
            likeDislike: Int? = nil,
            comment: String? = nil,
            evidenceRows: [Int] = [],
            conflicts: [String] = []
        ) {
            self.itemID = itemID
            self.strengthScore = strengthScore
            self.strengthScoreInferred = strengthScoreInferred
            self.likeDislike = likeDislike
            self.comment = comment
            self.evidenceRows = evidenceRows
            self.conflicts = conflicts
        }
    }

    public var templateID: String
    /// `utterancesVersion` the draft was generated against — a
    /// staleness stamp, same pattern the summarizer uses.
    public var generatedAtUtterancesVersion: Int?
    public var items: [ItemResult]
    /// Session-level 補足コメント distilled from substantive rows no
    /// item claimed.
    public var supplementaryComment: String?
    /// Extracted header metadata (field name → value), stated-only.
    public var metadata: [String: String]

    public init(
        templateID: String,
        generatedAtUtterancesVersion: Int? = nil,
        items: [ItemResult] = [],
        supplementaryComment: String? = nil,
        metadata: [String: String] = [:]
    ) {
        self.templateID = templateID
        self.generatedAtUtterancesVersion = generatedAtUtterancesVersion
        self.items = items
        self.supplementaryComment = supplementaryComment
        self.metadata = metadata
    }

    public func encoded() throws -> Data {
        try JSONEncoder().encode(self)
    }

    public static func decode(_ data: Data) throws -> EvalFormDraft {
        try JSONDecoder().decode(EvalFormDraft.self, from: data)
    }
}
