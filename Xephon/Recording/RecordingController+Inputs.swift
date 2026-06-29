import Foundation
@preconcurrency import AVFoundation
import Audio
import XephonLogging

// Audio input enumeration / selection split out of
// RecordingController.swift to keep the controller's source file
// readable. Same cross-file extension pattern as other splits.
// Covers `refreshInputs` (`session.availableInputs` query with
// the category-swap dance), `handleAudioRouteChange` (the
// AVAudioSession `routeChangeNotification` reaction), `selectInput`
// (user pick), and the `effectiveInputUID` computed property that
// the picker label binds to.
//
// The unrelated "configure capture / pipeline" setters
// (`setSpeechBoostEnabled`, `setTextSERBackend`, `setBackgroundMode`,
// `setSessionLanguage`) that historically lived under the same
// MARK section stay in the main file — they're capture/pipeline
// configuration, not input selection.
extension RecordingController {



    func refreshInputs() async {
        #if os(iOS) || targetEnvironment(macCatalyst)
        // `session.availableInputs` filters by the current category.
        // Under `.playback` (or default) it returns only the built-in
        // mic; Bluetooth inputs like AirPods only show up under
        // `.record` / `.playAndRecord` with bluetooth options. To
        // make the input picker reflect a freshly-connected AirPods
        // without committing to mic recording, briefly set the
        // category to `.playAndRecord` for the query, then restore
        // the previous declared category.
        //
        // We never call `setActive(true)`, so this is metadata-only
        // — the hardware route doesn't change, which is what kept
        // the playback-silence bug from coming back. Gated on idle
        // + no active playback because changing category on an
        // already-active session leaves the route latched (see
        // docs/playback_silence_postmortem.md).
        //
        // ALSO skip the swap when the session is already in a
        // recording-class category: (1) the inputs are already
        // visible without a swap, and (2) `setCategory` fires
        // `.categoryChange` + `.routeConfigurationChange`
        // notifications even when "switching" between equivalent
        // categories, which both disrupts an active engine on this
        // process *and* re-triggers `handleAudioRouteChange` →
        // `refreshInputs` → `setCategory` for every other
        // RecordingController that's still subscribed. Without this
        // guard, a stale idle controller (see the instanceTag doc)
        // category-thrashes the singleton session while another
        // controller is mid-recording, and the recording captures no
        // audio.
        let session = AVAudioSession.sharedInstance()
        let alreadyExposesInputs = session.category == .record
            || session.category == .playAndRecord
        let canReconfigure = phase == .idle
            && playbackPlayer == nil
            && !alreadyExposesInputs
        let priorCategory = session.category
        let priorMode = session.mode
        let priorOptions = session.categoryOptions
        if canReconfigure {
            try? session.setCategory(
                .playAndRecord,
                mode: .default,
                options: [.allowBluetoothHFP, .defaultToSpeaker]
            )
        }
        #endif
        let inputs = await capture.availableInputs()
        let current = await capture.currentInput()
        #if os(iOS) || targetEnvironment(macCatalyst)
        if canReconfigure {
            try? session.setCategory(priorCategory, mode: priorMode, options: priorOptions)
        }
        #endif
        AppLog.app.info("refreshInputs[\(self.instanceTag, privacy: .public)]: phase=\(String(describing: self.phase), privacy: .public) canReconfigure=\(canReconfigure, privacy: .public) inputs.count=\(inputs.count, privacy: .public) current=\(current?.uid ?? "<nil>", privacy: .public)")
        // Only commit the refreshed inputs when the query had a chance
        // to enumerate all ports. Without `canReconfigure` we read with
        // whatever category the session happens to be in (mid-record,
        // mid-deactivate, etc.), and `session.availableInputs` can
        // return a transient subset — or [] when the session was just
        // deactivated. Overwriting `availableInputs` with that stale
        // snapshot leaves the picker disabled (count ≤ 1) after the
        // first recording, since route-change notifications from the
        // bindPreferredInput deactivate/reactivate cycle all fire
        // while phase is still .recording. Keep the prior list when
        // we couldn't query authoritatively; the next idle refresh
        // (or the post-stop refresh) updates it cleanly. Also keep
        // the prior list when an idle query came back empty — that's
        // typically a deactivation-transient (the OS hasn't re-
        // enumerated the USB device yet) rather than a legitimate
        // "no inputs connected" state.
        if !inputs.isEmpty {
            // Commit only on an actual change so the idle input poll
            // (which calls this every ~2 s) doesn't fire @Observable
            // churn on every tick when nothing's plugged or unplugged.
            if inputs != self.availableInputs {
                self.availableInputs = inputs
            }
            if current?.uid != self.currentInputUID {
                self.currentInputUID = current?.uid
            }
        } else {
            AppLog.app.info("refreshInputs[\(self.instanceTag, privacy: .public)]: empty query result; keeping prior list (count=\(self.availableInputs.count, privacy: .public))")
        }
    }

    /// React to `AVAudioSession.routeChangeNotification`. Refreshes
    /// the visible input list and, when we're idle AND the change
    /// reason is `.oldDeviceUnavailable`, fully deactivates the shared
    /// session so the next playback or record activation rebinds
    /// against the current route.
    ///
    /// The reason filter is load-bearing. `AVAudioSession` is a
    /// process-wide singleton: deactivating it affects every
    /// controller in the app, including one that's mid-recording.
    /// Route changes fire constantly during normal operation —
    /// `.categoryChange` when our own `setCategory` runs,
    /// `.routeConfigurationChange` on USB clock renegotiation,
    /// `.newDeviceAvailable` when a device plugs in. Deactivating on
    /// any of those stomps an active recording. The original AirPods
    /// scenario this method handles is specifically the
    /// `.oldDeviceUnavailable` case (a device we were using
    /// disappeared, leaving the session bound to a dead route);
    /// gating on that is enough to fix AirPods without breaking
    /// another controller's session.
    ///
    /// Active playback gets a hard stop on any route change — Apple's
    /// human-interface guidance is that an unplugged or disconnected
    /// output should pause / stop playback rather than silently
    /// continue through whatever the OS falls back to.
    func handleAudioRouteChange(reasonRaw: UInt? = nil, previousInputUID: String? = nil) async {
        #if os(iOS) || targetEnvironment(macCatalyst)
        let reason = (AVAudioSession.sharedInstance().currentRoute.inputs.first?.portName ?? "<none>")
        AppLog.app.info("handleAudioRouteChange[\(self.instanceTag, privacy: .public)] fired; currentInput=\(reason, privacy: .public) phase=\(String(describing: self.phase), privacy: .public) reasonRaw=\(reasonRaw ?? 99, privacy: .public)")
        let parsedReason = reasonRaw.flatMap(AVAudioSession.RouteChangeReason.init(rawValue:))
        // `.categoryChange` is overwhelmingly our own `setCategory`
        // echoing back. Treat it as a no-op so the refresh path
        // doesn't loop into another `setCategory` and re-fire the
        // notification, then fire another, etc. Without this guard,
        // any second-instance controller's idle `refreshInputs`
        // becomes a self-perpetuating route-change storm that starves
        // the recording controller's engine.
        if parsedReason == .categoryChange { return }
        // USB / external mic unplugged mid-recording: the OS falls back
        // to the built-in mic and capture continues, so we DON'T stop —
        // we just surface a banner so the user knows the source changed.
        // Keyed on the previous route's *input* port (not just the
        // reason) so an output-only change — e.g. unplugging headphones
        // while on the built-in mic — doesn't trigger it. Falls through
        // to refreshInputs below so the picker updates to the new route.
        if phase == .recording,
           parsedReason == .oldDeviceUnavailable,
           let lostInputUID = previousInputUID,
           !AVAudioSession.sharedInstance().currentRoute.inputs.contains(where: { $0.uid == lostInputUID }) {
            AppLog.app.info("input \(lostInputUID, privacy: .public) disconnected mid-recording; falling back to built-in, keeping recording")
            errorMessage = String(localized: "error.capture.inputDisconnectedFallback")
        }
        #endif
        await refreshInputs()
        if playbackPlayer != nil {
            stopPlayback()
        }
        guard phase == .idle else { return }
        #if os(iOS) || targetEnvironment(macCatalyst)
        guard parsedReason == .oldDeviceUnavailable else { return }
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
        #endif
    }

    func selectInput(uid: String?) async {
        AppLog.app.info("selectInput called with uid=\(uid ?? "<nil>", privacy: .public)")
        // Record the user's intent immediately so the picker label
        // reflects it even if the OS route doesn't actually switch
        // (or flickers back). This is the picker's source of truth;
        // `currentInputUID` is only used as diagnostic.
        self.selectedInputUID = uid
        do {
            try await capture.setPreferredInput(uid)
            await refreshInputs()
            AppLog.app.info("selectInput post-refresh: selectedInputUID=\(self.selectedInputUID ?? "<nil>", privacy: .public) currentInputUID=\(self.currentInputUID ?? "<nil>", privacy: .public)")
        } catch {
            errorMessage = String(describing: error)
            AppLog.app.error("setPreferredInput failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// The input that the next recording will actually use. Matches
    /// the bindPreferredInput fallback in `AudioCapture.start()`:
    /// explicit user pick → that UID; no pick → built-in mic UID
    /// from the current available-inputs list. The picker label
    /// binds to this so what the user sees is what they get.
    var effectiveInputUID: String? {
        if let selectedInputUID { return selectedInputUID }
        return availableInputs.first { $0.kind == .builtInMic }?.uid
    }

}
