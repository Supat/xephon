import Foundation

/// What the extraction did NOT find — the reviewer's to-fill list.
/// The report renders this alongside the detections so the sheet's
/// remaining manual work is explicit instead of implied by dashes:
/// missing header fields, and per item which sheet targets (評点 /
/// 好き嫌い / コメント) have no detected value, distinguishing
/// "mentioned but only an inferred score" from "never mentioned."
public enum EvalFormCoverage {

    public struct ItemGaps: Equatable, Sendable {
        public let itemID: String
        /// False when nothing at all was detected for the item —
        /// no fields, no evidence. Rendered as "言及なし".
        public let mentioned: Bool
        /// No STATED score. `inferredScorePresent` qualifies it:
        /// an inferred suggestion exists but needs confirmation.
        public let statedScoreMissing: Bool
        public let inferredScorePresent: Bool
        public let preferenceMissing: Bool
        public let commentMissing: Bool

        public var hasGaps: Bool {
            !mentioned || statedScoreMissing || preferenceMissing || commentMissing
        }
    }

    public struct Undetected: Equatable, Sendable {
        public let missingHeaderFields: [String]
        /// One entry per template item WITH gaps, template order.
        public let items: [ItemGaps]

        public var isEmpty: Bool {
            missingHeaderFields.isEmpty && items.isEmpty
        }
    }

    public static func undetected(
        draft: EvalFormDraft,
        template: EvalFormTemplate
    ) -> Undetected {
        let missingHeader = template.metadataFields.filter {
            draft.metadata[$0] == nil
        }
        let items: [ItemGaps] = template.items.compactMap { item in
            let result = draft.items.first { $0.itemID == item.id }
            let mentioned = result.map {
                $0.strengthScore != nil
                    || $0.strengthScoreInferred != nil
                    || $0.likeDislike != nil
                    || $0.comment != nil
                    || !$0.evidenceRows.isEmpty
            } ?? false
            let gaps = ItemGaps(
                itemID: item.id,
                mentioned: mentioned,
                statedScoreMissing: result?.strengthScore == nil,
                inferredScorePresent: result?.strengthScoreInferred != nil,
                preferenceMissing: result?.likeDislike == nil,
                commentMissing: result?.comment == nil
            )
            return gaps.hasGaps ? gaps : nil
        }
        return Undetected(missingHeaderFields: missingHeader, items: items)
    }

    /// Compact per-item gap phrase shared by both renderers:
    /// "言及なし" for unmentioned items, else the missing targets
    /// ("評点（推定のみ・要確認）" when a suggestion exists).
    public static func gapPhrase(_ gaps: ItemGaps) -> String {
        guard gaps.mentioned else { return "言及なし" }
        var parts: [String] = []
        if gaps.statedScoreMissing {
            parts.append(gaps.inferredScorePresent ? "評点（推定のみ・要確認）" : "評点")
        }
        if gaps.preferenceMissing { parts.append("好き嫌い") }
        if gaps.commentMissing { parts.append("コメント") }
        return parts.joined(separator: ", ")
    }
}
