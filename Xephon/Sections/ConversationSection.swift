import Foundation
import Summarizer

/// One user-defined section of the conversation — a named
/// portion bounded by a start and / or end utterance. Named
/// `ConversationSection` (not `Section`) to avoid clashing
/// with SwiftUI's `Section` view builder at call sites that
/// import both modules.
///
/// Bounds are stored as utterance IDs (not indices / times)
/// so a section stays anchored across re-evaluation, hand
/// edits, and import / load cycles — anything that changes
/// the surrounding rows without touching the bounded
/// utterances themselves keeps the section intact.
///
/// **Complete vs. incomplete state.** A section is *complete*
/// when both `startUtteranceID` and `endUtteranceID` are
/// set; *incomplete* when exactly one is set (typical use:
/// bookmarking an interesting moment during a recording and
/// filling in the other bound later). A section with neither
/// bound set is meaningless and rejected at the editor
/// layer — but at the data layer both `Optional<UUID>` so
/// callers can construct work-in-progress instances.
///
/// `Codable` so a future commit can persist sections in the
/// `.xph` bundle alongside the utterances they reference; the
/// initial implementation keeps the store in memory only
/// (sections are tied to per-session utterance IDs and don't
/// survive a session swap).
public struct ConversationSection: Sendable, Hashable, Identifiable, Codable {
    public let id: UUID
    /// User-supplied title for the section. May be empty;
    /// the UI shows a fallback placeholder when blank rather
    /// than rejecting empty titles outright (users sometimes
    /// want a quick anchor without naming it).
    public var title: String
    /// First utterance in the section. Inclusive. Nil when
    /// the user has only marked the end so far (incomplete
    /// state).
    public var startUtteranceID: UUID?
    /// Last utterance in the section. Inclusive — a single-
    /// utterance section is `startUtteranceID == endUtteranceID`.
    /// Nil when the user has only marked the start so far
    /// (incomplete state).
    public var endUtteranceID: UUID?
    /// Cached on-device summary scoped to this section's
    /// utterance range. Only populated on complete sections
    /// after the user taps the row's summary button; nil
    /// means "never generated yet" (or invalidated by a
    /// re-encode that dropped the field — codable-optional so
    /// older `.xph` bundles decode cleanly). Persisted as
    /// part of the section blob so the per-section sheet
    /// reopens with the same result across save / load.
    public var cachedSummary: SessionSummary?

    public init(
        id: UUID = UUID(),
        title: String,
        startUtteranceID: UUID?,
        endUtteranceID: UUID?,
        cachedSummary: SessionSummary? = nil
    ) {
        self.id = id
        self.title = title
        self.startUtteranceID = startUtteranceID
        self.endUtteranceID = endUtteranceID
        self.cachedSummary = cachedSummary
    }

    /// True iff both bounds are set. Incomplete sections
    /// (exactly one bound) render with a distinct caption in
    /// the card so the user knows there's something left to
    /// finish.
    public var isComplete: Bool {
        startUtteranceID != nil && endUtteranceID != nil
    }
}
