# Xephon User Manual

Xephon is an iPadOS research app that listens to Japanese conversational
audio and tells you, per utterance, **who** spoke, **what** they said, and
**how they felt** — both as an emotion label (Plutchik 8) and as a point
on the valence / arousal / dominance axes. Everything runs on-device. Your
audio never leaves the iPad.

This manual walks you through setup, the main workflows, and the parts of
the UI that aren't self-explanatory.

---

## 1. Before you start

**Requirements**

- iPad Pro with an M4 or M5 chip (the 16-core Neural Engine SKU). Other
  iPads can install the app but the primary speech recognizer won't run
  on them.
- iPadOS 26 or later.
- Microphone permission (for live recording) and Speech Recognition
  permission (for transcription). The app will prompt the first time
  each is needed.

**First launch**

On first launch the app downloads its ML models (~ 1 GB total) over
Wi-Fi. This takes a few minutes. The download is resumable — if you close
the app mid-download, it picks up where it left off on next launch.

> **Figure 1: First-launch model download progress.**
> Centered modal showing a circular progress ring at ~ 45 %, with the
> active model name ("WRIME RoBERTa · tokenizer.json") and a "1 of 9
> models" subtitle. Cancel button at the bottom.

When the ring closes you land on the main screen.

---

## 2. The main screen

Xephon's main screen is a two-pane layout: a **control pane** on the left
and a **transcript pane** on the right. In portrait the control pane
collapses to a sidebar drawer.

> **Figure 2: Main screen, landscape, idle state.**
> Left pane: input picker ("iPad Microphone ⌄"), a large "Start
> Recording" button, an "Open Audio File" icon button next to it, then
> stacked cards: Settings, Pipeline, (optional) Summarizer. Right pane:
> a "Search utterances" field at top, a "All Labels ⌄" filter chip on
> the right, a horizontal diarization strip below them, then the
> transcript list (empty state: "No utterances yet — tap Start
> Recording to begin"). Top toolbar carries (left → right) Save, Find,
> Search, Export buttons.

The control pane has four cards you'll use most:

| Card           | What it does                                                       |
|----------------|--------------------------------------------------------------------|
| **Settings**   | Language, text-SER backend, speech-boost toggle, diarizer slider, Custom Glossary |
| **Pipeline**   | Live "is each stage working?" indicators while recording           |
| **Summarizer** | Generates an LLM summary of the session (optional, off by default) |
| **Models**     | Tap "Manage Models" if you ever need to re-download / delete model weights |

---

## 3. Recording from the microphone

1. Pick your input device from the picker at the top of the control
   pane. If a USB-C mic is plugged in, it shows up there alongside
   "iPad Microphone".
2. (Optional) Toggle **Speech Boost** in the Settings card if your input
   is quiet or distant. It boosts the 1–4 kHz speech band on the
   recognizer's feed only; the acoustic emotion models still see the
   original audio.
3. Tap **Start Recording**. The button turns into a "Stop" pill and an
   elapsed-time counter starts running.

> **Figure 3: Mid-recording state.**
> Left pane top: red "Stop · 00:34" pill replacing the blue Start
> button, a horizontal level meter below it pulsing. Pipeline card
> shows the six stages (Capture, ASR (Live), Diarizer, Acoustic SER,
> Text SER, Fusion) each with a green check; ASR card shows "1 snt"
> count. Right pane: diarization strip filling left-to-right with
> alternating colored bands per speaker; transcript list shows two
> finalized utterances (gray "Speaker 1" chip, transcript text,
> emotion label badge on the right) and a third volatile-text row in
> italic gray.

While you're recording, **volatile text** (the recognizer's
in-progress hypothesis) appears in italic gray at the bottom of the
list and updates 5× per second. When the recognizer commits a
sentence boundary, the row solidifies into a finalized utterance.

To stop, tap the **Stop** pill. The app finishes processing any
in-flight utterances and returns to idle.

---

## 4. Analyzing a recorded audio file

1. Tap the **Open Audio File** icon (📂) next to the Start Recording
   button.
2. Pick a file in the system file picker. Supported: anything
   AVFoundation can decode (MP3, M4A, WAV, AAC, FLAC, ...).
3. Tap **Start Recording** (now labeled the same way; mode is shown
   above the button as "File: filename.mp3").

> **Figure 4: File-mode picker.**
> System document picker open over Xephon. Files browser shows two
> selectable audio files highlighted in blue.

File-mode analysis runs faster than real time (the pump reads as fast
as the analyzer can keep up). A typical 10-minute MP3 finishes in
~ 3 minutes on M4. While it runs, the same Pipeline card shows
throughput and the transcript fills in progressively.

When the file is exhausted the app automatically stops, finalizes the
last utterances, and returns to idle.

---

## 5. Reading the transcript

Each row in the transcript is one utterance.

> **Figure 5: One utterance row, annotated.**
> A single row with callouts pointing at each part:
>   • Speaker chip "S01" (tinted to that speaker's color) — bottom-left
>   • Play button (▶) — leading edge
>   • Re-evaluate button (↻) — directly below play
>   • Backend badge "WRIME" — top-right of the metadata strip
>   • Modality badge "≠ Modalities" — next to backend badge when text
>     and acoustic disagree
>   • Glossary chip "Glossary 2" (purple) — when glossary terms fired
>   • Hand-edited badge "Edited" — when the user committed a hand-edit
>   • Top-line emotion label "Happy" (color-coded) — top-right
>   • V / A readout "V +0.07  A +0.10" — second line right
>   • Transcript text — center, the largest text element

Tapping the row expands it into a detail panel:

> **Figure 6: Expanded detail panel.**
> Same row, expanded. Shows three columns:
>   • Top fused labels (bar chart, top 3 emotions with probabilities)
>   • V/A pulls scatter mini (110 × 110): a small square plot with
>     three dots — acoustic (orange), text (purple), fused (green) —
>     and translucent connecting lines
>   • Acoustic SER (emotion2vec) bars on the left side, Text SER
>     (WRIME) bars on the right
> Below the columns: ASR confidence, Acoustic V/A/D readout, Fused
> V/A/D readout, Fusion-weight summary ("Text 72% · Acoustic 28%"),
> and an Age + Gender readout when available.

### Gestures and quick actions

- **Tap the transcript text** for ~ 0.5 s → **Edit Utterance** sheet
  (see § 7).
- **Tap the speaker chip** → reassign popover (pick a different
  speaker, rename the current one, or "Teach diarizer this is
  Speaker N").
- **Tap the ▶ button** → play this utterance's audio from the source
  file. Tap again to stop. The button shows a stop icon while
  playing.
- **Tap the ↻ button** → re-evaluate (see § 8).
- **Long-press the "Edited" badge for 2 s** → revert to the
  pre-edit row.

---

## 6. Filtering the transcript

Above the list you have:

- A **search field** that does Japanese-aware substring matching across
  all transcripts. Search is normalized (hiragana/katakana
  interchangeable, full-width and half-width digits/Latin treated as
  equivalent).
- A **label filter** chip ("All Labels ⌄") for narrowing by fused
  emotion.
- A **speaker filter row** that shows every speaker as a tappable chip
  (tap to include only that speaker; "All Speakers" to clear). A
  "Mismatch" chip filters to rows where acoustic and text SER
  disagree.

> **Figure 7: Filter bar with two speakers selected.**
> Top row: search field on the left with placeholder text, label
> filter chip on the right. Second row of pill-shaped speaker chips:
> "All Speakers" unselected, "S01" and "S03" highlighted (filled),
> "S02" and "S05" outlined.

When no rows match, an empty state appears with a "Clear filters"
button.

---

## 7. Editing an utterance

Long-press the transcript text of any row to raise the **Edit
Utterance** sheet.

> **Figure 8: Edit Utterance sheet.**
> Modal sheet. Top: "Transcript" header, then a multi-line text editor
> filled with the row's text. Below: "Audio Range" header, a
> horizontal control row with a large play button on the left and a
> stacked Start/End time control on the right (each with a numeric
> field flanked by -/+ buttons). Below that: a full-width "Transcribe"
> button. Toolbar: Cancel (top-left), Commit (top-right).

You can:

- **Edit the transcript text** directly in the editor.
- **Adjust the time range** with the -/+ buttons or by typing
  `m:ss.s` into either field.
- **Play the current range** to hear what you've selected.
- **Tap Transcribe** to re-run offline ASR on the current range and
  drop the result into the text field. If you've enabled the
  glossary's ASR Hint pathway (see § 9), this run uses your hinted
  terms; otherwise it uses the default offline recognizer. The text
  editor locks while transcription is running.
- **Commit** to apply your edits. The row's transcript, range, and
  affect scores are all re-computed.

Hand-edited rows get an "Edited" badge. Long-press it for 2 s to
revert.

---

## 8. Re-evaluating a row

Tap the ↻ button on any row to re-run ASR + SER + fusion on that
utterance's audio with padded boundaries (so the recognizer has
prosodic context the streaming pass didn't see). Useful when the
streaming transcript looks cut off or wrong.

> **Figure 9: Re-evaluation in progress.**
> The target row's re-evaluate button replaced by a small spinner.
> Underneath the row, in italic gray, the offline recognizer's
> volatile text crawls left-to-right as it processes.

When the re-eval finishes the row is replaced in place with the new
result, stamped with a "Re-evaluated" indicator. Like hand-edits,
you can revert via long-press.

---

## 9. Custom Glossary

The Custom Glossary is a per-user list of terms that can do two
things:

1. **Bias** the text-SER Plutchik distribution toward a specific
   emotion when that term appears in a transcript (e.g. "残業" →
   tilt toward sadness).
2. **Hint** the SFSpeechRecognizer with a list of vocabulary terms
   on the Transcribe pathway so domain words / proper nouns are
   more likely to be transcribed correctly.

Each entry can do either, both, or neither.

Open the sheet via **Settings → Custom Glossary**.

> **Figure 10: Custom Glossary sheet.**
> Modal sheet, top to bottom:
>   • Two master toggles, each with a hint footer:
>       - "📕 Apply Bias" (purple book icon)
>       - "🎙 ASR Hint" (mic icon)
>   • Scrollable list of entry cards. Each card shows:
>       - Top row (left → right): colored emotion chip ("Joy"),
>         spacer, purple book toggle (filled when useAsBias on),
>         blue mic toggle (filled when useAsASRHint on), red trash
>         button
>       - Term TextField below (e.g. "残業")
>       - Bottom row: Emotion picker ("Joy ⌄"), spacer, "WEIGHT"
>         label + numeric readout (e.g. "0.7"), -/+ stepper
>   • Bottom: full-width "+ Add Entry" button
> Toolbar: Done (top-left), ⋯ menu (top-right) with Import / Export.

### Per-entry toggles

- **📕 Book** — when filled (purple), this entry contributes to
  text-SER bias. When outlined, the entry is excluded from bias
  even if the master "Apply Bias" is on. The emotion label chip on
  the left greys out when this is off — that's your visual
  preview that the row won't actually tilt the distribution.
- **🎙 Mic** — when filled (blue), this entry's term is forwarded
  to SFSpeechRecognizer's `contextualStrings` on the next
  Transcribe call. When outlined, the entry is excluded.

Both per-entry toggles dim when their respective master toggle is
off, so you can see at a glance that they're inert.

### Adding entries

Tap **+ Add Entry** at the bottom. A new empty row appears at the
end of the list with the keyboard already focused on its term
field. Type your term, pick an emotion, set the weight (0.0 = no
effect, 1.0 = strong tilt). Both per-entry toggles default to on.

### Editing weights

Weight controls how strongly the bias tilts the matched class. The
math is in logit space, so weight 1.0 roughly multiplies that
class's odds-ratio by 7×; weight 0.5 by 2.7×.

> **Figure 11: Weight stepper close-up.**
> A horizontal control: small label "WEIGHT" in gray, monospaced
> numeric readout "0.6", and a circular -/+ stepper. Below: an
> implicit slider trail showing 0.0 ─── 0.6 ─── 1.0.

### Deleting entries

Tap the trash button on any card. The row is removed immediately;
there's no undo, so be deliberate.

### Import / Export

Tap **⋯ → Export…** to save your glossary as a JSON file. Tap
**⋯ → Import…** to load one. Use this to share glossaries between
devices or sessions.

### When edits take effect

Every glossary change persists immediately (write-through JSON in
Application Support; survives relaunch). Bias / hint behavior on
**new** utterances applies right away. To re-bias **existing**
utterances with the current glossary, tap **Done** to dismiss the
sheet — the app re-applies bias to every stored row and re-runs
fusion in the background. This is fast (no model calls; pure math).

> **Figure 12: Done re-apply.**
> Just after tapping Done, the transcript list shows several
> previously-purple "Glossary N" chips updating in place, with
> some fused labels visibly changing color. No spinner — it's
> sub-second on a typical session.

---

## 10. Speaker management

Xephon's diarizer identifies speakers automatically by their voice
embedding. They get default labels "S01", "S02", etc.

> **Figure 13: Diarization strip.**
> The horizontal bar at the top of the transcript pane. Solid
> colored segments per speaker stacked into a single horizontal
> band, with a thin time-cursor line showing the current playback
> position. Speaker colors match the row chips.

### Renaming a speaker

Tap any speaker chip in a row → popover → **Rename Speaker…** →
type a name (e.g. "Alice"). The rename applies everywhere
immediately.

### Reassigning a row

Tap the chip → tap another speaker's chip in the popover. That
row's speaker is changed.

### Teaching the diarizer

If you reassign a row AND want the diarizer to learn from your
correction for future utterances, flip **Teach diarizer** on in
the popover before tapping the target speaker. The new audio gets
folded into that speaker's centroid.

### Affirming a speaker

If the diarizer's guess is right but you want to reinforce its
confidence (useful for "Speaker 3" rows that are actually Speaker 1
but the diarizer wasn't sure), tap **Affirm Speaker** in the
popover. Requires source audio (file mode or imported session).

### Speaker sensitivity

The **Speaker Sensitivity** slider in Settings controls how eagerly
the diarizer splits voices into distinct speakers. Drag right →
more distinct speakers; drag left → speakers merge more. Double-tap
the label to reset to the default.

> **Figure 14: Speaker Sensitivity slider.**
> A horizontal slider labeled "👥 Speaker Sensitivity   0.60" with
> "Merge" on the left end and "Split" on the right. Below: a
> single-line hint "Higher = more distinct speakers. Double-tap
> label to reset."

---

## 11. Session summary

If you've enabled the summarizer (Settings → Summarizer card), Xephon
can produce a one-page LLM summary of the whole session.

Backends:

- **Apple Foundation Models** (built into iOS 26, no download
  needed). Default when available.
- **Qwen3 8B** (MLX). Downloads ~ 4.6 GB on first use.

Open the summary sheet from the toolbar (📖 icon).

> **Figure 15: Session Summary sheet.**
> Modal sheet. Sections, top to bottom:
>   • Setting (e.g. "Casual conversation, two friends")
>   • Topic (e.g. "Weekend plans and a recent work incident")
>   • Overall Mood (e.g. "Mostly positive, briefly tense around
>     11:30")
>   • Per Speaker (one paragraph per speaker on their affect arc)
> Footer: "Model: Apple Foundation Models · AI Generated · May
> contain errors". Toolbar: Done (top-right), Regenerate
> (bottom-left).

The summary auto-generates the first time you open the sheet (when
the summarizer is ready). Tap **Regenerate** to redo it after
edits.

---

## 12. Transcription review

The Transcription Review sheet asks the same on-device LLM to scan
your transcript for likely transcription errors (homophones, grammar,
context).

> **Figure 16: Transcription Review sheet.**
> Modal sheet. List of issue cards, each showing:
>   • Kind chip ("Homophone" / "Contextual" / "Grammar" / "Other")
>     with a confidence percentage
>   • Inline text editor with the current transcript (editable)
>   • Reason text underneath in gray
>   • Action row: Dismiss (✕), Re-evaluate (↻), Range… (→ opens
>     Edit Utterance sheet), Commit (✓)
> Bottom action bar: "Review" / "Re-review" full-width button.

Use the inline editor for quick fixes; the Range… button hands the
in-progress edit off to the full Edit Utterance sheet when you need
to adjust the time range too.

---

## 13. Saving and sharing

### Save a session

Toolbar 💾 icon → standard system save sheet. The session is written
as a `.xph` bundle (binary plist) containing the utterance list,
source audio (file mode only), diarizer state, summary, and
metadata. Save anywhere — Files app, iCloud Drive, etc.

### Open a saved session

Toolbar 📂 icon in the toolbar, or use the Files app and tap a
`.xph` file. The session loads with full state (including playback if
the source audio was bundled).

### Export JSON

Toolbar share icon → **Export JSON**. Writes the per-utterance
rows as JSON (see `docs/output_schema.md`). Use this to bring
results into external tooling.

> **Figure 17: Export menu.**
> Action sheet from the share icon with three options stacked: "Save
> Session (.xph)", "Export JSON", "Share Audio File…" (only when the
> session has a bundled source).

---

## 14. Settings reference

> **Figure 18: Settings card, expanded.**
> The full Settings card with every control labeled:
>   • Language picker (🇯🇵 Japanese ⌄)
>   • Text SER picker (WRIME ⌄)
>   • Speech Boost toggle (mic mode only)
>   • Speaker Sensitivity slider (👥 ─── value)
>   • Custom Glossary navigation row (📕 Custom Glossary  N >)

| Control               | When to change                                          |
|-----------------------|---------------------------------------------------------|
| Language              | Recording in a language other than Japanese.            |
| Text SER              | Pick WRIME for fastest Japanese inference, or Apple FM for richer (slower) reasoning. Auto-falls back when one is unavailable. |
| Speech Boost          | Quiet or distant input. Affects ASR only.              |
| Speaker Sensitivity   | Diarizer is splitting one person into many speakers (drag left), or merging two people into one (drag right). |
| Custom Glossary       | Open to bias text-SER or seed ASR with your vocabulary. |

---

## 15. Pipeline card — what each indicator means

> **Figure 19: Pipeline card, all stages active.**
> Vertical list of six stages, each with an icon, label, throughput
> number, and a green-check status pill. Stages:
>   • 🎙 Capture
>   • 🔊 ASR (Live)        — "5 snt"
>   • 👥 Diarizer          — "2 spk"
>   • 🎚 Acoustic SER      — "1.84 s"
>   • 💬 Text SER          — "120 ms"
>   • ✨ Fusion            — "5 utts"
>   • 📤 Export

The status pill is green when the stage is healthy, yellow when
degraded (e.g. running on CPU because the ANE is busy), and red when
unavailable.

A diagnostics banner appears at the top of the screen if any model
fails to load. Tap the banner for details.

---

## 16. Troubleshooting

| Symptom                                              | Try                                                                                                                                                                                                          |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "Some models didn't load" banner on launch           | Settings → Models card → Manage Models → tap Re-download for the failing model. Network required.                                                                                                            |
| Live recording shows transcript but no emotions      | The acoustic SER models may have failed to load. See the Pipeline card. Re-download via Models card.                                                                                                         |
| One person is being split into multiple "speakers"   | Drag the Speaker Sensitivity slider left. Then re-evaluate the affected rows, or reassign them and toggle "Teach diarizer" so the merge sticks.                                                              |
| Two people are being merged into one speaker        | Drag the Speaker Sensitivity slider right. Reassign rows and use Teach diarizer for corrections.                                                                                                              |
| Transcribe button gives garbled output                | If glossary's ASR Hint is on, the SFSpeechRecognizer pathway may be biasing toward your hint terms incorrectly. Turn the master ASR Hint toggle off and try again to compare with the default offline path. |
| Glossary entries with weight set but no chip on rows | The master "Apply Bias" toggle is off, OR the entry's 📕 toggle is off, OR the term doesn't substring-match the transcript (case-insensitive). Check the emotion label chip on the entry — if greyed, the entry is inert. |
| "Empty audio" or missing acoustic on late rows      | The audio source ended before ASR finished. The app now re-reads from the source file for file-mode sessions, so this should be rare. Re-evaluate the affected row.                                          |
| Apple FM never produces text                        | The app is backgrounded or another app has the ANE. Foreground Xephon; it will retry the Foundation Models call on the next utterance.                                                                       |

---

## 17. Privacy

- **All audio processing is on-device.** Recording, transcription,
  diarization, acoustic and text emotion analysis all run on the
  iPad's Neural Engine / GPU / CPU. No audio, no transcript, no
  embedding ever leaves the device.
- **No analytics, no telemetry.** Xephon doesn't phone home.
- **Model weights** are downloaded from the public release the first
  time they're needed. After that, everything is local.
- **Saved sessions** (`.xph` files) live wherever you save them —
  Files app, iCloud Drive, USB, etc. Treat them like any other
  sensitive document.
- **Glossary entries** persist in the app's private Application
  Support directory. They're not bundled with `.xph` sessions, so
  sharing a session doesn't share your glossary; use the glossary
  sheet's Export to move it deliberately.

See `docs/privacy.md` for the formal privacy statement.

---

## 18. Keyboard shortcuts (external keyboard)

| Shortcut       | Action                              |
|----------------|-------------------------------------|
| ⌘ R            | Start / Stop recording              |
| ⌘ O            | Open audio file                     |
| ⌘ S            | Save session                        |
| ⌘ F            | Focus search field                  |
| ⌘ E            | Export JSON                         |
| ⌘ ⇧ S          | Open session summary sheet          |
| ⌘ ⇧ R          | Open transcription review sheet     |
| Space          | Play / pause selected row's audio   |
| Esc            | Dismiss the active sheet            |

---

## 19. Getting help

Xephon is research software — there's no support desk. The
documentation in `docs/` covers the architecture, model selection,
and known limitations. The `eval_log.md` file in particular tracks
known CER / WER / SER metrics on the project's held-out calibration
set so you can sanity-check your own results.

For bug reports or feature requests, contact the developer directly.
