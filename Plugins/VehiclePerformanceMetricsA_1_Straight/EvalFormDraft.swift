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
        /// Model-suggested 1–9 preference from wording that
        /// expresses liking/disliking. Kept apart from
        /// `likeDislike` by design — a suggestion, never a sheet
        /// entry. Optional-tolerant addition within payload v2
        /// (missing key decodes nil).
        public var likeDislikeInferred: Int?
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
            likeDislikeInferred: Int? = nil,
            comment: String? = nil,
            evidenceRows: [Int] = [],
            conflicts: [String] = []
        ) {
            self.itemID = itemID
            self.strengthScore = strengthScore
            self.strengthScoreInferred = strengthScoreInferred
            self.likeDislike = likeDislike
            self.likeDislikeInferred = likeDislikeInferred
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
    /// Evidence rows behind `supplementaryComment`. Optional-
    /// tolerant addition within payload v2 (missing key decodes
    /// nil) — no version bump needed, same rule SessionDocument
    /// uses for new optional fields.
    public var supplementaryEvidenceRows: [Int]?
    /// Extracted header metadata (field name → value), stated-only.
    public var metadata: [String: String]
    /// Item ids the reviewer has marked confirmed (payload v2 —
    /// the review-workflow state that motivated the version bump).
    public var reviewedItemIDs: [String]
    /// Road provenance for CITED rows, frozen at fill time (row →
    /// road label from the template's callout segmentation). Nil
    /// when the session had no callouts. Optional-tolerant v2
    /// addition — missing key decodes nil.
    public var roadByRow: [Int: String]?

    public init(
        templateID: String,
        generatedAtUtterancesVersion: Int? = nil,
        items: [ItemResult] = [],
        supplementaryComment: String? = nil,
        supplementaryEvidenceRows: [Int]? = nil,
        metadata: [String: String] = [:],
        reviewedItemIDs: [String] = [],
        roadByRow: [Int: String]? = nil
    ) {
        self.templateID = templateID
        self.generatedAtUtterancesVersion = generatedAtUtterancesVersion
        self.items = items
        self.supplementaryComment = supplementaryComment
        self.supplementaryEvidenceRows = supplementaryEvidenceRows
        self.metadata = metadata
        self.reviewedItemIDs = reviewedItemIDs
        self.roadByRow = roadByRow
    }

    /// Distinct roads where the item's cited evidence appeared —
    /// canonical full labels, first-appearance order over the
    /// (sorted) evidence rows. Empty when the draft carries no
    /// road provenance or the item cites nothing.
    public func roadsForItem(_ itemID: String) -> [String] {
        guard let roadByRow, !roadByRow.isEmpty,
              let result = items.first(where: { $0.itemID == itemID })
        else { return [] }
        var seen = Set<String>()
        var ordered: [String] = []
        for row in result.evidenceRows {
            if let road = roadByRow[row], seen.insert(road).inserted {
                ordered.append(road)
            }
        }
        return ordered
    }

    public func encoded() throws -> Data {
        try JSONEncoder().encode(self)
    }

    public static func decode(_ data: Data) throws -> EvalFormDraft {
        try JSONDecoder().decode(EvalFormDraft.self, from: data)
    }

    // MARK: - Payload migration

    /// Version-aware restore — THE migration seam
    /// (docs/plugin_architecture.md §3: `payloadVersion` rides on
    /// every stored payload; the plugin migrates on read). v1
    /// payloads (before `reviewedItemIDs`) upgrade with an empty
    /// review state; payloads from a NEWER plugin than this build
    /// return nil rather than a lossy guess — the draft stays
    /// untouched in the bundle for the newer build that wrote it.
    public static func restore(data: Data, storedVersion: Int?) -> EvalFormDraft? {
        switch storedVersion {
        case 2:
            return try? decode(data)
        case 1, nil:
            guard let v1 = try? JSONDecoder().decode(DraftV1.self, from: data)
            else { return nil }
            return EvalFormDraft(
                templateID: v1.templateID,
                generatedAtUtterancesVersion: v1.generatedAtUtterancesVersion,
                items: v1.items,
                supplementaryComment: v1.supplementaryComment,
                metadata: v1.metadata,
                reviewedItemIDs: []
            )
        default:
            return nil
        }
    }

    /// The payload-v1 wire shape, frozen. `ItemResult` is shared —
    /// it did not change between v1 and v2.
    private struct DraftV1: Codable {
        var templateID: String
        var generatedAtUtterancesVersion: Int?
        var items: [ItemResult]
        var supplementaryComment: String?
        var metadata: [String: String]
    }
}
