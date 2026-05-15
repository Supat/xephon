import Foundation

/// Lifecycle hook so on-device inference actors can swap their
/// backend when the app moves between foreground and background.
/// iOS / Mac "Designed for iPad" revoke ANE/GPU access while the
/// app isn't `.active`, which would otherwise trip the CoreML EP
/// at the next inference call.
///
/// `setBackgroundMode(true)` is called on transition away from
/// `.active` and `setBackgroundMode(false)` on return, letting each
/// actor proactively rebuild on CPU / CoreML rather than wait for a
/// failure and recover reactively. The runtime catch-shims that
/// each ONNX-backed actor still ships act as defense in depth for
/// any path the proactive swap doesn't cover.
///
/// Default conformance is a no-op so test doubles and pure-CPU
/// implementations don't need to do anything. Lives in the
/// `SERRuntime` module so both `SERAcoustic` and `SERText` actors
/// (which already depend on it) can conform without duplication.
public protocol BackgroundAwareSER: Actor {
    func setBackgroundMode(_ inBackground: Bool) async
}

public extension BackgroundAwareSER {
    func setBackgroundMode(_ inBackground: Bool) async {}
}
