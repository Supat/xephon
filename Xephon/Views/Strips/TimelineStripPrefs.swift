import Foundation

/// UserDefaults keys for the transcript pane's timeline-strip
/// visibility toggles. Shared between MainToolbar's Timelines menu
/// (writes) and TranscriptPaneView's strip gates (reads) via
/// `@AppStorage` on both sides — UserDefaults-backed bindings keep
/// the two views in sync without threading state through
/// ContentView. `bool(forKey:)`-style defaults don't apply here:
/// both `@AppStorage` declarations default to `true` (all strips
/// visible), matching the pre-toggle behavior for existing users.
enum TimelineStripPrefs {
    static let showDiarizationKey = "timelineStrip.showDiarization"
    static let showEmotionKey = "timelineStrip.showEmotion"
    static let showFusionKey = "timelineStrip.showFusion"
    static let showKeywordKey = "timelineStrip.showKeyword"
}
