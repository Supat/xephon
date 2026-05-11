# Silent per-utterance playback post-mortem

## Symptom

After enabling per-utterance audio playback for file-analysis sessions
(tap a row's play button → hear `[utterance.start, utterance.end]` from
the source file), playback worked the first time but went silent on
subsequent runs when a microphone session had happened first.

Diagnostic logs showed everything looking healthy:

```
playback session: category=AVAudioSessionCategoryPlayback
                  mode=AVAudioSessionModeDefault
                  outputs=Speaker
togglePlayback: play() returned true, duration=245s, seek=11.52s
togglePlayback: playingUtteranceID set to ...
```

`AVAudioPlayer.play()` returned `true`. The session reported the
`.playback` category. The route reported `Speaker`. No errors anywhere.
And yet — no sound from the speaker.

## Wrong hypotheses, in order

### 1. "Security-scoped resource access expired."

The first failure mode looked like a permission issue (`permErr -54`
during `AVAudioPlayer(contentsOf:)`). The picker hands out
security-scoped URLs whose grant lapses if no one calls
`startAccessingSecurityScopedResource()` before `AudioFileCapture`
releases its own ref at the end of analysis.

This *was* a real bug — `RecordingController` now acquires its own
ref via `setPlaybackSourceURL(url)` at the top of `startFromFile`, and
holds it for the lifetime of `playbackSourceURL`. ContentView also
pins scope inside the file-picker success callback so the URL survives
the discard+pacing dialog hop. That fixed the `-54` open failure.

But — after fixing it — `AVAudioPlayer` opened the file fine, `play()`
returned `true`, and playback was still silent. So scope wasn't the
remaining issue.

### 2. "Audio session is stuck in `.record` mode."

When microphone recording is active, `AVAudioEngineCapture.start()`
configures the shared `AVAudioSession` to
`.record / .measurement / [.allowBluetoothHFP]`. `AVAudioPlayer.play()`
won't produce audio under `.record`. So the next playback path needs to
switch the category to `.playback` first.

Adding `setCategory(.playback) + setActive(true)` to `togglePlayback`
fixed the *first* playback after a mic session. But on subsequent file
analyses, the silent symptom returned.

### 3. "The session needs an explicit deactivate before the category change."

Tried wrapping the category change with
`setActive(false, options: .notifyOthersOnDeactivation)` first, then
`setCategory(.playback)`, then `setActive(true)`. This actually made
things worse — it broke even the *first* playback. Deactivating an
active session and immediately reactivating it leaves the system in a
"category-says-playback-but-no-audio" state, presumably because the
hardware route hasn't finished tearing down before the activation
re-attaches it.

### 4. "Use `.playAndRecord + .defaultToSpeaker + overrideOutputAudioPort(.speaker)`."

`.playAndRecord` is route-compatible with the prior `.record` config,
so changing category shouldn't require a full route teardown.
`.defaultToSpeaker` is meant to force output to the built-in speaker
even when prior options had latched it elsewhere.
`overrideOutputAudioPort(.speaker)` is the explicit override.

The diagnostic log confirmed `category=PlayAndRecord, mode=Default,
outputs=Speaker` after this — but `play()` still produced silence.
The route *name* was right; the route *behavior* wasn't.

### 5. "Skip `refreshInputs()` at the end of a file analysis."

`refreshInputs()` ends up calling `AudioCapture.availableInputs()`,
which calls `configureCategoryIfNeeded(session)`, which **forces** the
session to `.record / .measurement / [.allowBluetoothHFP]`. Doing
that after every file analysis tramples the session even when no mic
is involved.

Removing this call fixed the consecutive-file case (no mic recording
needed beforehand → no `.record` trample → `.playback` worked
cleanly). But it didn't address the original mic→file→playback path,
because the mic session itself had still left the session active in
`.record` mode after its `stop()`.

## The real cause

`AVAudioSession` doesn't reset its hardware route configuration just
because we change `category`. If the session is **still active** when
we change the category, the OS reconfigures the route graph as a
delta on top of the previous active config — and that delta can leave
the output detached from the speaker even when `currentRoute.outputs`
truthfully reports `Speaker`.

The route name is what's *requested*; whether audio actually reaches
the speaker depends on whether the *underlying audio graph* has
settled in a state compatible with playback.

`AVAudioEngineCapture.start()` activates the session for `.record`.
Its `stop()` was tearing down the engine and its taps, but **not**
deactivating the session itself. The session stayed active in
`.record / .measurement / [.allowBluetoothHFP]` indefinitely, until
the next code path "changed" the category — at which point the OS
made a delta change rather than a clean re-init, and the output
silently routed nowhere.

## The fix

Two coordinated changes:

1. **`AVAudioEngineCapture.stop()` deactivates the session** explicitly:

   ```swift
   try? AVAudioSession.sharedInstance().setActive(
       false,
       options: .notifyOthersOnDeactivation
   )
   ```

   This forces the OS to fully tear down the route graph for our
   session. The next `setActive(true)` from any code path starts from
   scratch.

2. **`RecordingController.stop()` no longer calls `refreshInputs()`
   when transitioning out of file mode.** `AudioCapture.availableInputs()`
   has the side effect of stamping `.record / .measurement` on the
   shared session, which defeats the cleanup from change #1 in the
   file→playback flow. The inputs list will refresh the next time the
   user actually starts a mic session (`AVAudioEngineCapture.start()`
   configures the category explicitly anyway).

With both changes, `togglePlayback`'s simple
`setCategory(.playback) + setActive(true)` works in every order of
operations: file→playback, mic→file→playback, file→mic→file→playback,
etc.

## Lessons

- **`session.category == .playback` is not the same as "the session
  is in playback configuration."** The category name reports what
  the app *asked for*; the actual hardware state depends on whether
  the session has been deactivated since the last config change.

- **`currentRoute.outputs` reports the OS's route bookkeeping, not
  whether audio is reaching that port.** Both can be `Speaker` while
  the speaker is silent. Don't trust the route as proof of audibility
  — confirm with your ears or a level meter.

- **Always pair `setActive(true)` with a `setActive(false)` when the
  category will change later.** A capture/recording subsystem that
  activates the session in its `start()` must deactivate in its
  `stop()`. Leaving the session active across category transitions is
  the canonical recipe for the silent-`play()` failure mode.

- **Don't have view-level helpers re-configure the audio session.**
  `availableInputs()` looked like a pure query; it wasn't. Hidden
  category writes inside read-shaped helpers are exactly the kind of
  side effect that survives every fix attempt until you find them.

- **Diagnostic logs that print the *requested* state are deceptive
  when the bug is in the *applied* state.** The first two months of
  fix attempts here all printed `category=Playback, outputs=Speaker`
  and matched expectations. The bug only became visible when we
  realized "the log line is the question, not the answer."

## Related code

- `Core/Audio/AudioCapture.swift` — `AVAudioEngineCapture.stop()` is
  the activation/deactivation pair point.
- `Xephon/RecordingController.swift` — `togglePlayback(for:)`
  configures the session for playback; `stop()` no longer calls
  `refreshInputs()` on the file→idle transition;
  `setPlaybackSourceURL(_:)` manages the security-scoped resource
  ref independently of `AudioFileCapture`'s own ref.
- `Xephon/ContentView.swift` — file-picker callback pins scope on
  `pendingFileURL`; `startFromFileAndReleaseScope` and
  `releasePendingFileScope` close that ref once the recorder has
  taken its own.
