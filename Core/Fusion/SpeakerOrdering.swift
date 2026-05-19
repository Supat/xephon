import Foundation

/// Distinct speaker IDs in the order they first appear in the
/// supplied utterance list. Callers decide whether to sort first:
///
/// - Pass the list as-is when the order is already meaningful
///   (e.g. ASR-emission order from `RecordingController.utterances`,
///   or a passage handed to the summarizer).
/// - Pre-sort by `start` (`utterances.sorted(by: { $0.start < $1.start })`)
///   when you need strict chronological first-appearance — the
///   per-card matrices use this so the row/column order matches
///   the chip-bar above the transcript.
extension Array where Element == UtteranceEstimate {
    public var orderedSpeakerIDs: [String] {
        var seen: Set<String> = []
        var ordered: [String] = []
        for u in self where !seen.contains(u.speakerID) {
            seen.insert(u.speakerID)
            ordered.append(u.speakerID)
        }
        return ordered
    }
}

/// Hashable directed (or undirected) pair of speaker IDs, used as
/// the dictionary key for every "row × column" speaker matrix in
/// the app: leadership, interruptions, response latency, etc.
///
/// Field names are generic (`row`/`col`) so the same key works
/// regardless of which side carries the "actor" semantics for a
/// given matrix (leader/follower, interrupter/victim, responder/
/// partner, …). Callers wrap the named pair at the call site.
public struct SpeakerPairKey: Hashable, Sendable {
    public let row: String
    public let col: String

    public init(_ row: String, _ col: String) {
        self.row = row
        self.col = col
    }
}
