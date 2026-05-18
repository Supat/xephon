import Foundation
import ActivityKit
import XephonLogging

/// Lock-Screen / Dynamic Island integration for the recording session.
/// Owns the activity ID, request/update/end lifecycle, and the
/// coalescing logic that prevents fast-pace segment finalization from
/// flooding ActivityKit with updates faster than the system can apply
/// them.
///
/// Stored as the activity's *ID* (Sendable `String`) rather than the
/// `Activity<T>` reference itself — `Activity<T>` is a non-Sendable
/// class and storing it as MainActor state risks data races under
/// Swift 6 strict concurrency. `update`/`end` re-resolve the activity
/// by id inside `nonisolated` static helpers, so the reference never
/// crosses an isolation boundary.
///
/// All Activity I/O is local — no APNs / push channel.
@MainActor
final class LiveActivityController {
    private var activityId: String?
    /// Coalesced-update task. While running, additional
    /// `scheduleUpdate(_:)` calls just refresh `pendingState` and the
    /// task picks them up on its next loop iteration. Without this
    /// coalescing, fast-pace segment finalization queues
    /// `Activity.update` awaits faster than the system rate-limits
    /// can drain — the queue itself contributes to MainActor congestion.
    private var updateTask: Task<Void, Never>?
    private var pendingState: XephonActivityAttributes.ContentState?

    /// Begin a Live Activity for the session. Failures are logged and
    /// non-fatal — the session keeps running, the Lock Screen just
    /// shows nothing. No-op if one is already running or the user has
    /// disabled Live Activities globally.
    func start(sourceLabel: String) {
        guard activityId == nil else { return }
        guard ActivityAuthorizationInfo().areActivitiesEnabled else {
            AppLog.app.info("Live Activities disabled by user; skipping")
            return
        }
        let attrs = XephonActivityAttributes(
            sessionStartedAt: Date(),
            sourceLabel: sourceLabel
        )
        let initial = XephonActivityAttributes.ContentState(
            elapsedSeconds: 0,
            utteranceCount: 0,
            topLabel: nil,
            valence: nil,
            arousal: nil,
            isAnalyzing: false
        )
        do {
            let activity = try Activity.request(
                attributes: attrs,
                content: ActivityContent(state: initial, staleDate: nil)
            )
            activityId = activity.id
        } catch {
            AppLog.app.warning("Live Activity start failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Latest-wins update. If a coalesced task is already in flight,
    /// this just refreshes `pendingState` and returns; the running
    /// task picks up the new state on its next iteration. Otherwise
    /// spawns a new task to drain `pendingState` until empty.
    func scheduleUpdate(_ state: XephonActivityAttributes.ContentState) {
        guard activityId != nil else { return }
        pendingState = state
        guard updateTask == nil else { return }
        updateTask = Task { @MainActor [weak self] in
            guard let self else { return }
            while let next = self.pendingState {
                self.pendingState = nil
                guard let id = self.activityId else { break }
                await Self.update(activityId: id, state: next)
            }
            self.updateTask = nil
        }
    }

    /// End the activity with a final state. Cancels any queued
    /// coalesced update and awaits the in-flight one before posting
    /// the final state, so the Lock Screen lingers on the right
    /// content. Dismisses after a 2-minute window so the user can
    /// glance at the final state before it disappears.
    func end(finalState: XephonActivityAttributes.ContentState) async {
        guard let id = activityId else { return }
        activityId = nil
        pendingState = nil
        await updateTask?.value
        updateTask = nil
        await Self.end(activityId: id, finalState: finalState)
    }

    // Nonisolated static helpers — the `Activity<T>` reference never
    // leaves these functions, so it never crosses an isolation
    // boundary.

    nonisolated private static func update(
        activityId: String,
        state: XephonActivityAttributes.ContentState
    ) async {
        guard let activity = Activity<XephonActivityAttributes>.activities
            .first(where: { $0.id == activityId }) else { return }
        await activity.update(ActivityContent(state: state, staleDate: nil))
    }

    nonisolated private static func end(
        activityId: String,
        finalState: XephonActivityAttributes.ContentState
    ) async {
        guard let activity = Activity<XephonActivityAttributes>.activities
            .first(where: { $0.id == activityId }) else { return }
        await activity.end(
            ActivityContent(state: finalState, staleDate: nil),
            dismissalPolicy: .after(Date().addingTimeInterval(120))
        )
    }
}
