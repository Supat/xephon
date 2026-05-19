import Foundation
@preconcurrency import AVFoundation
import Audio
import ASR
import Diarization
import Fusion
import Export
import Summarizer
import XephonLogging

// `.xph` session bundle save/load split out of
// RecordingController.swift to keep the controller's source file
// readable. Same cross-file `extension RecordingController`
// pattern as `+HandEdit.swift` / `+Reevaluation.swift` — no own
// observable state, just snapshot-to-bundle and bundle-to-state
// translation. `loadSession` writes to a lot of controller state
// (utterances, embeddings, timeline, speaker DB, source-mode,
// etc.); the corresponding properties had to relax their
// `private(set)` / `private` modifiers (see annotations at each
// declaration site).
extension RecordingController {



    /// Build a `SessionDocument` snapshot for the current state. For
    /// file-mode sessions, embeds the source audio bytes inline so
    /// playback round-trips after import; mic-mode sessions skip
    /// the audio block per the schema's no-playback-for-mic contract.
    ///
    /// Throws when the audio file can't be read (e.g. the picker's
    /// scope expired). Callers should ensure `playbackSourceURL` is
    /// fresh before calling — `togglePlayback`-tested URLs are good.
    func makeSessionDocument() async throws -> SessionDocument {
        let utts = utterances
        let names = speakerNameOverrides.isEmpty ? nil : speakerNameOverrides
        // Snapshot the diarizer's SpeakerManager so re-diarization
        // after Open Session lands on the same session-stable IDs
        // the original recording assigned. Nil when the diarizer
        // wasn't engaged (mic mode with no speech) or hasn't loaded
        // its models yet — both are normal and don't surface to the
        // user. Awaiting the pipeline here is what forced this
        // function to become async; callers run it from a Task.
        let speakerDB = await pipelineForExport()?.exportSpeakerDatabase()
        // Carry the pre-edit revert state alongside the utterances
        // so a long-press revert on a row that was hand-edited or
        // re-evaluated keeps working after Save → Open. Filter
        // snapshots whose target is no longer in `utterances` —
        // those would be orphans and the revert path would no-op
        // for them anyway. Empty maps round-trip as nil so v1-shaped
        // bundles (no revert state) stay byte-identical on save.
        let liveIDs = Set(utts.map(\.id))
        let snapshotsForExport = preReevaluationSnapshots.filter { liveIDs.contains($0.key) }
        let childrenForExport = handEditChildren.filter { liveIDs.contains($0.key) }
        let snapshots = snapshotsForExport.isEmpty ? nil : snapshotsForExport
        let children = childrenForExport.isEmpty ? nil : childrenForExport
        // Serialize the diarizer timeline so the per-session
        // visualization strip in the transcript pane survives
        // Save → Open. JSON-encoded so the Export layer doesn't
        // need to import the Diarization module's segment type.
        // Nil when no diarization ran (mic-mode pre-roll on a save
        // with no audio, or an analysis path that never engaged
        // the diarizer).
        let timelineBlob: Data? = {
            guard !diarizationTimeline.isEmpty else { return nil }
            return try? JSONEncoder().encode(diarizationTimeline)
        }()
        // Persist the last LLM summary alongside the utterances so
        // reopening a `.xph` shows the cached result without
        // re-running the multi-second LLM pass. JSON-encoded so the
        // Export layer doesn't need to depend on Summarizer (which
        // would drag MLX into a module that has no business with
        // it). Nil when no summary has been produced this session.
        let summaryBlob: Data? = {
            guard let summary = lastSessionSummary else { return nil }
            return try? JSONEncoder().encode(summary)
        }()
        // Persist the LLM's transcription review issues alongside
        // the utterances so reopening a `.xph` shows the same
        // flagged rows without re-running the multi-second review
        // pass. Pair them with a per-utterance transcript snapshot
        // so a stale issue (target row was edited since save) can
        // be filtered out on load. Snapshot only the utterances
        // that have an active issue — no point recording text for
        // rows that aren't flagged.
        let issuesBlob: Data? = {
            guard !transcriptionIssues.isEmpty else { return nil }
            return try? JSONEncoder().encode(transcriptionIssues)
        }()
        let issueSnapshots: [UUID: String]? = {
            guard !transcriptionIssues.isEmpty else { return nil }
            let flagged = Set(transcriptionIssues.map(\.utteranceID))
            let map = utts.reduce(into: [UUID: String]()) { acc, u in
                if flagged.contains(u.id) { acc[u.id] = u.transcript }
            }
            return map.isEmpty ? nil : map
        }()
        // Persist the per-utterance speaker embeddings so the
        // cluster scatter's tap-to-scroll (and per-observation
        // focus arrow) survives Save → Open. Filter to live
        // utterance ids — a stale entry whose row has since been
        // dropped would just bloat the file. Empty round-trips as
        // nil so sessions that never engaged the diarizer don't
        // grow a useless field.
        let embeddingsForExport = utteranceEmbeddings.filter { liveIDs.contains($0.key) }
        let embeddings = embeddingsForExport.isEmpty ? nil : embeddingsForExport
        // Persist the pinned observation segment id per utterance
        // so the cluster scatter's tap-to-scroll keeps doing exact
        // id matching across Save → Open instead of falling back
        // to embedding-distance argmin (lossier on overlapping
        // clouds). Filtered to live utterance ids and dropped to
        // nil when empty for the same byte-cleanliness reason as
        // the other optional maps.
        let segmentIDsForExport = utteranceObservationSegmentIDs.filter { liveIDs.contains($0.key) }
        let segmentIDs = segmentIDsForExport.isEmpty ? nil : segmentIDsForExport
        if let url = playbackSourceURL {
            let stillScoped = url.startAccessingSecurityScopedResource()
            defer {
                if stillScoped { url.stopAccessingSecurityScopedResource() }
            }
            do {
                let audioData = try Data(contentsOf: url)
                return SessionDocument(
                    sourceKind: .file,
                    audioFilename: url.lastPathComponent,
                    audio: audioData,
                    utterances: utts,
                    speakerNames: names,
                    speakerDatabase: speakerDB,
                    originalSnapshots: snapshots,
                    handEditChildren: children,
                    diarizationTimeline: timelineBlob,
                    sessionSummary: summaryBlob,
                    transcriptionIssues: issuesBlob,
                    transcriptionIssueTranscriptSnapshots: issueSnapshots,
                    utteranceEmbeddings: embeddings,
                    utteranceObservationSegmentIDs: segmentIDs
                )
            } catch {
                throw SessionBundle.BundleError.ioFailure(
                    "audio read failed: \(error.localizedDescription)"
                )
            }
        }
        return SessionDocument(
            sourceKind: .microphone,
            audioFilename: nil,
            audio: nil,
            utterances: utts,
            speakerNames: names,
            speakerDatabase: speakerDB,
            originalSnapshots: snapshots,
            handEditChildren: children,
            diarizationTimeline: timelineBlob,
            sessionSummary: summaryBlob,
            transcriptionIssues: issuesBlob,
            transcriptionIssueTranscriptSnapshots: issueSnapshots,
            utteranceEmbeddings: embeddings,
            utteranceObservationSegmentIDs: segmentIDs
        )
    }

    /// Read-only access to the existing pipeline, without forcing
    /// initialization. `makeSessionDocument` uses it to skip the
    /// diarizer DB snapshot when no pipeline ever spun up (e.g.
    /// saving an imported session that was never re-analyzed).
    private func pipelineForExport() -> AnalysisPipeline? {
        pipeline
    }

    /// Replace the current session state with the contents of a
    /// previously-saved bundle. No-op when not idle so we never
    /// clobber an in-flight recording. Extracts the bundle's audio
    /// (if any) to a sandboxed temp file and wires it as the new
    /// playback source.
    func loadSession(_ document: SessionDocument) async throws {
        guard phase == .idle else { return }
        stopPlayback()
        // Bump the session token so ContentView's `@State` keyed
        // by UUID (`visibleUtteranceIDs`, `expandedUtteranceIDs`,
        // `selectedUtteranceID`, `scrollRequestUtteranceID`,
        // `normalizedTranscriptCache`) is dropped before the new
        // utterance list takes over. Without this, stale view
        // state from the previous file leaks into the loaded one
        // and timeline strips (which mix `recorder.utterances`
        // with the view-side visibility set) can render wrong.
        sessionToken = UUID()
        utterances = document.utterances
        // Restore the pre-edit revert state from the bundle so a
        // long-press on a row's Edited / completed marker after
        // Open Session still rolls the row back to its original
        // streaming-pass record. `removeAll` first to drop any
        // leftover state from a prior session that wasn't cleared
        // (no recording started yet).
        preReevaluationSnapshots.removeAll()
        handEditChildren.removeAll()
        // Drop the prior session's per-utterance embeddings before
        // adopting the new bundle's map — leftover entries keyed by
        // UUIDs from the previous session would either land on no
        // row at all or, worse, collide with a freshly assigned
        // UUID and mis-target the cluster-tap → scroll lookup.
        utteranceEmbeddings.removeAll()
        utteranceObservationSegmentIDs.removeAll()
        diarizationTimeline = []
        speakerCluster = SpeakerClusterSnapshot(speakers: [])
        lastKnownSpeakerIDs = []
        // The cached summary was generated against the previous
        // session's utterances. Default to clearing; restore below
        // if the loaded `.xph` carries a persisted one.
        let restoredSummary: SessionSummary? = document.sessionSummary
            .flatMap { try? JSONDecoder().decode(SessionSummary.self, from: $0) }
        summarizer.restore(summary: restoredSummary)
        // Restore LLM-flagged issues, filtering against the saved
        // transcript snapshots so a row edited between save and
        // re-open doesn't surface a flag pointing at text that no
        // longer exists. Bundles without the issue blob (v1 / no
        // review run) reset to empty.
        let restoredIssues: [TranscriptionIssue]
        if let blob = document.transcriptionIssues,
           let candidates = try? JSONDecoder().decode([TranscriptionIssue].self, from: blob) {
            let snapshots = document.transcriptionIssueTranscriptSnapshots ?? [:]
            let liveByID = Dictionary(
                uniqueKeysWithValues: document.utterances.map { ($0.id, $0) }
            )
            restoredIssues = candidates.filter { issue in
                guard let live = liveByID[issue.utteranceID] else { return false }
                // No snapshot means we can't prove staleness; keep
                // the issue rather than silently dropping it.
                guard let snapshot = snapshots[issue.utteranceID] else { return true }
                return snapshot == live.transcript
            }
        } else {
            restoredIssues = []
        }
        summarizer.restore(issues: restoredIssues)
        if let saved = document.originalSnapshots {
            preReevaluationSnapshots = saved
        }
        if let savedChildren = document.handEditChildren {
            handEditChildren = savedChildren
        }
        if let savedEmbeddings = document.utteranceEmbeddings {
            utteranceEmbeddings = savedEmbeddings
        }
        if let savedSegmentIDs = document.utteranceObservationSegmentIDs {
            utteranceObservationSegmentIDs = savedSegmentIDs
        }
        if let timelineBlob = document.diarizationTimeline,
           let restored = try? JSONDecoder().decode([DiarizedSegment].self, from: timelineBlob) {
            diarizationTimeline = restored
        }
        speakerNameOverrides = document.speakerNames ?? [:]
        // Same-length imports would otherwise hit the filter memo;
        // bump defensively so the cache rebuilds for any load.
        commitUtteranceChanges()
        lastChunkSpeakerCount = 0
        lastChunkSentenceCount = 0
        lastAcousticDuration = nil
        lastTextDuration = nil
        lastSegmentTotal = nil
        lastASRFinalizeLatency = nil
        // Imported sessions act like a finished file analysis: in
        // the .microphone source mode (so Record starts fresh) with
        // a playback URL pointing at the extracted audio (so the
        // per-row play button works).
        sourceMode = .microphone
        capture = micCapture
        if let audioData = document.audio, !audioData.isEmpty {
            let filename = document.audioFilename ?? "audio"
            let tempDir = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "xephon-session-\(UUID().uuidString)",
                    isDirectory: true
                )
            do {
                try FileManager.default.createDirectory(
                    at: tempDir,
                    withIntermediateDirectories: true
                )
            } catch {
                throw SessionBundle.BundleError.ioFailure(
                    "temp dir create failed: \(error.localizedDescription)"
                )
            }
            let destination = tempDir.appendingPathComponent(filename)
            do {
                try audioData.write(to: destination, options: .atomic)
            } catch {
                throw SessionBundle.BundleError.ioFailure(
                    "audio extract failed: \(error.localizedDescription)"
                )
            }
            setPlaybackSourceURL(destination)
            fileTotalAudioDuration = {
                guard let f = try? AVAudioFile(forReading: destination) else { return nil }
                let rate = f.processingFormat.sampleRate
                guard rate > 0, f.length > 0 else { return nil }
                return TimeInterval(Double(f.length) / rate)
            }()
        } else {
            setPlaybackSourceURL(nil)
            fileTotalAudioDuration = nil
        }
        // Restore the FluidAudio speaker DB when the saved bundle
        // carries one. Without this, a re-diarize on a hand-edited
        // slice would cluster against an empty SpeakerManager and
        // assign brand-new IDs that don't correspond to the
        // utterance rows' speaker labels. We swallow throws — a
        // stale blob (e.g. FluidAudio's `Speaker` schema changed
        // across versions) shouldn't prevent the user from opening
        // their session; they just lose the diarizer-restore
        // benefit and re-diarize starts from scratch.
        if let blob = document.speakerDatabase, !blob.isEmpty {
            do {
                try await ensurePipeline().importSpeakerDatabase(blob)
                // Seed the cluster snapshot synchronously so the
                // scatter / heatmap render with data the moment the
                // user navigates to the cluster page. Without this,
                // `speakerCluster` stays at the empty value set
                // earlier in loadSession until the TabView's 1 Hz
                // refresh task fires its first tick — up to a full
                // second of empty state on a freshly loaded session.
                await refreshClusterSnapshot()
            } catch {
                AppLog.app.warning(
                    "loadSession: speaker DB restore failed: \(String(describing: error), privacy: .public)"
                )
            }
        }
    }

}
