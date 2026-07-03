import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { fetchIceServers, fetchVoiceList, postRenegotiate } from "./api/rtc.js";
import { Info, Listbox, ToggleRow, MiniSlider } from "./components/Controls.jsx";
import { PreflightModal, VisionSourceModal, FrameModal } from "./components/Modals.jsx";
import { Badge, Flow, Level, RailColumn, Row, RTTGraph, Scope, TelemetryCell, VuMeter } from "./components/Telemetry.jsx";
import { Icon } from "./components/icons.jsx";
import {
  ADHERENCE_MODES,
  DEFAULTS,
  DEFAULT_VISION_PROMPT,
  EXPRESSION_MODES,
  HEARTBEAT_INTERVAL_MS,
  HEARTBEAT_MAX_PENDING,
  HEARTBEAT_MISSED_LIMIT,
  HEARTBEAT_STALE_AFTER_MS,
  JITTER_BUFFER_SMOOTH_SEC,
  PERSONA_PRESETS,
  RECONNECT_GRACE_MS,
  RENEGOTIATE_MAX_ATTEMPTS,
  RENEGOTIATE_RETRY_DELAY_MS,
  SESSION_PROFILES,
  VISION_MOTION_THRESHOLD,
  VISION_PER_CALL_USD,
  VOICES,
} from "./data/dashboardData.jsx";
import { useStoredState } from "./hooks/useStoredState.js";
import { useToast } from "./hooks/useToast.js";
import { rmsFromAnalyser } from "./utils/audio.js";
import { cls, fmt, fmtGb } from "./utils/format.js";

function parseStoredArray(value) {
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function parseStoredObject(value) {
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function storedProfileId() {
  return `user_${globalThis.crypto?.randomUUID?.() || Date.now().toString(36)}`;
}

function formatOffset(ms = 0) {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function formatDiffValue(value) {
  if (typeof value === "boolean") return value ? "on" : "off";
  if (value === null || value === undefined || value === "") return "off";
  return String(value);
}

// Stable keys for the fixed-length decorative voice-row waveform bars.
const GLYPH_BARS = Array.from({ length: 11 }, (_, i) => `glyph-${i}`);

function App() {
  const toast = useToast();
  const [phase, setPhase] = useState("idle");
  // User toggle to peek at the frozen config column while a session runs.
  const [sideExpanded, setSideExpanded] = useState(false);
  // Collapsible tuning rack. Starts open on tall viewports and collapsed on
  // short ones so the transcript keeps its room; the stored preference wins
  // on later loads.
  const [railOpen, setRailOpen] = useStoredState(
    "pp_railOpen",
    typeof window !== "undefined" ? window.innerHeight > 700 : true,
    (v) => v === "1",
    (v) => (v ? "1" : "0"),
  );
  const [stageMessage, setStageMessage] = useState("Standby");
  const [connectionIssue, setConnectionIssue] = useState(null);
  const [elapsedSec, setElapsedSec] = useState(0);
  // Labelled snapshot bookmarks for jump-back, newest first: {id, label, atSec}.
  // Session-scoped runtime state (not persisted); reset when a session starts.
  const [bookmarks, setBookmarks] = useState([]);
  const [latencyMs, setLatencyMs] = useState(0);
  const [tailLatencyMs, setTailLatencyMs] = useState(0);
  const [rttSamples, setRttSamples] = useState([]);
  const [reconnecting, setReconnecting] = useState(false);
  // Transport telemetry sampled from getStats(): quality is a 0-100 score
  // derived from jitter and loss; candidate is the selected pair's relay
  // type. All best-effort; zeros until a live candidate pair exists.
  const [netStats, setNetStats] = useState({ quality: 0, jitterMs: 0, lossPct: 0, candidate: "" });
  // Client-side jitter-buffer bias. "latency" keeps playout tight;
  // "smooth" raises the receiver's playoutDelayHint to ride out jitter.
  const [jitterBuffer, setJitterBuffer] = useStoredState("pp_jitterBuffer", "latency");
  const [levels, setLevels] = useState({ mic: 0, ai: 0 });
  const [speaking, setSpeaking] = useState(null);
  const [interrupting, setInterrupting] = useState(false);
  const [_transcriptText, setTranscriptText] = useState("");
  // AI transcript split into per-turn segments for chronological rendering.
  // Each entry is { id, at, text }; a new segment opens on the same
  // turn-boundary signal that drives the session timeline (a >1600 ms gap
  // between text chunks). The flat accumulator is only read inside its own
  // setter, where per-turn word and rate accounting slices it.
  const [aiTurns, setAiTurns] = useState([]);
  // User-side transcript turns. Each entry is { id, audioOnly, text }. A
  // turn is created from the local speaking-state transition (mic spoke,
  // assistant resumed) with audioOnly true and no text; the optional
  // server-side recognizer upgrades it in place via a user_text message
  // when it produces words. With the recognizer off the turns stay
  // audio-only, matching the "spoke · audio only" marker.
  const [userTurns, setUserTurns] = useState([]);
  const [notices, setNotices] = useState([]);
  const [sessionTimeline, setSessionTimeline] = useState([]);
  const [assistantRate, setAssistantRate] = useState({ words: 0, seconds: 0, wpm: 0 });
  const [recordingUrl, setRecordingUrl] = useState(null);
  const [recordingMime, setRecordingMime] = useState("audio/webm");
  // Optional server-side recording status. Stays null unless the server
  // emits a recording event, so the UI is unchanged when the feature is off.
  const [serverRecording, setServerRecording] = useState(null);

  const [presetId, setPresetId] = useState("teacher");
  const [sessionProfileId, setSessionProfileId] = useState("custom");
  const [profileName, setProfileName] = useStoredState("pp_profileName", "My profile");
  const [customProfiles, setCustomProfiles] = useStoredState("pp_customSessionProfiles", [], parseStoredArray, JSON.stringify);
  const [pinnedTuning, setPinnedTuning] = useStoredState("pp_pinnedTuningProfile", null, parseStoredObject, JSON.stringify);
  const [textPrompt, setTextPrompt] = useStoredState("pp_textPrompt", PERSONA_PRESETS[0].prompt);
  const [visionPrompt, setVisionPrompt] = useStoredState("pp_visionPrompt", DEFAULT_VISION_PROMPT);
  const [voice, setVoice] = useStoredState("pp_voicePrompt", "NATF1");
  const [voiceGender, setVoiceGender] = useState("F");
  const [voiceList, setVoiceList] = useState(VOICES);
  // Id of the preset voice whose sample is currently being fetched/played.
  // Holds at most one at a time, so starting a new preview supersedes any
  // in-flight one. Drives the row's play/stop glyph and waveform recolor.
  const [previewing, setPreviewing] = useState(null);
  const [adherenceMode, setAdherenceMode] = useStoredState("pp_adherenceMode", "balanced");
  const [expressionMode, setExpressionMode] = useStoredState("pp_expressionMode", "natural");
  const [uploadedVoiceFilename, setUploadedVoiceFilename] = useState("");
  const [uploadedVoiceLabel, setUploadedVoiceLabel] = useState("");
  const [uploadedVoiceMeta, setUploadedVoiceMeta] = useState(null);
  const [uploadedVoicePreviewUrl, setUploadedVoicePreviewUrl] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadKind, setUploadKind] = useState("");
  // How strongly an uploaded clip conditions the timbre, as an integer
  // 0..100 for the UI; the payload sends the 0..1 float. Only meaningful
  // with a clip uploaded. Connect-time only, like the rest of the prefix.
  const [cloneStrength, setCloneStrength] = useStoredState("pp_cloneStrength", 70, Number);
  // Optional second voice mixed into the prefix. blendMix is the secondary
  // share as an integer 0..100 for the UI; the payload sends the 0..1 float.
  // Connect-time only, like the rest of the voice prefix.
  const [voiceBlend, setVoiceBlend] = useStoredState("pp_voiceBlend", false, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [voiceB, setVoiceB] = useStoredState("pp_voiceB", "NATM0");
  const [blendMix, setBlendMix] = useStoredState("pp_blendMix", 50, Number);

  const [textTemp, setTextTemp] = useStoredState("pp_textTempSlider", DEFAULTS.textTemp, Number);
  const [textTopk, setTextTopk] = useStoredState("pp_textTopkSlider", DEFAULTS.textTopk, Number);
  const [audioTemp, setAudioTemp] = useStoredState("pp_audioTempSlider", DEFAULTS.audioTemp, Number);
  const [audioTopk, setAudioTopk] = useStoredState("pp_audioTopkSlider", DEFAULTS.audioTopk, Number);
  const [repPenalty, setRepPenalty] = useStoredState("pp_repPenaltySlider", DEFAULTS.repPenalty, Number);
  const [repContext, setRepContext] = useStoredState("pp_repContextSlider", DEFAULTS.repContext, Number);
  const [padBonus, setPadBonus] = useStoredState("pp_padBonusSlider", DEFAULTS.padBonus, Number);
  const [maxTurn, setMaxTurn] = useStoredState("pp_maxTurnSlider", DEFAULTS.maxTurn, Number);
  const [echoCancel, setEchoCancel] = useStoredState("pp_echoCancel", DEFAULTS.echoCancel, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [noiseSupp, setNoiseSupp] = useStoredState("pp_noiseSupp", DEFAULTS.noiseSupp, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [autoGain, setAutoGain] = useStoredState("pp_autoGain", DEFAULTS.autoGain, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [visionInTranscript, setVisionInTranscript] = useStoredState("pp_visionInTranscript", false, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [reinforceInSilences, setReinforceInSilences] = useStoredState("pp_reinforceInSilences", false, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [seedRandom, setSeedRandom] = useStoredState("pp_seedRandom", true, (v) => v === "1", (v) => (v ? "1" : "0"));
  const [seed, setSeed] = useStoredState("pp_seedValue", DEFAULTS.seed, Number);
  const [idleTimeout, setIdleTimeout] = useStoredState("pp_idleTimeout", 0, Number); // minutes; 0 = off

  const [serverInfo, setServerInfo] = useState({ gpuName: "", vramTotal: 0, serverBuild: "" });
  const [gpuStat, setGpuStat] = useState({ vramUsed: 0, gpuUtil: null });
  // Server-measured real-time factor: compute time per audio frame divided
  // by that frame's audio duration. Below 1 means inference keeps up; at or
  // above 1 it is falling behind. 0 when not live (no measurement).
  const [rtf, setRtf] = useState(0);

  const [visionOn, setVisionOn] = useState(false);
  const [visionPaused, setVisionPaused] = useState(false);
  const [visionEnabledFromServer, setVisionEnabledFromServer] = useState(true);
  const [visionInjecting, setVisionInjecting] = useState(false);
  const [visionFramesSent, setVisionFramesSent] = useState(0);
  const [visionFramesGated, setVisionFramesGated] = useState(0);
  const [visionLastSentAt, setVisionLastSentAt] = useState(0);
  const [visionClockMs, setVisionClockMs] = useState(0);
  const [visionIntervalMs, setVisionIntervalMs] = useStoredState("pp_visionIntervalMs", DEFAULTS.visionIntervalMs, Number);
  const [visionCostLimitUsd, setVisionCostLimitUsd] = useStoredState("pp_visionCostLimitUsd", 0, Number);
  const [visionBudgetTripped, setVisionBudgetTripped] = useState(false);
  const [currentCaption, setCurrentCaption] = useState("");
  const [captionEntries, setCaptionEntries] = useState([]);
  const [inspectFrame, setInspectFrame] = useState(null);
  const [visionSourceOpen, setVisionSourceOpen] = useState(false);

  const [preflightOpen, setPreflightOpen] = useState(false);
  const [preflight, setPreflight] = useState({ mic: "idle", out: "idle", turn: "idle" });
  const [preflightDone, setPreflightDone] = useState(false);
  const [audioOutputs, setAudioOutputs] = useState([]);
  const [outputDeviceId, setOutputDeviceId] = useStoredState("pp_outputDeviceId", "default");
  const [connectHoldPct, setConnectHoldPct] = useState(0);

  const aiAudioRef = useRef(null);
  const visionVideoRef = useRef(null);
  const pcRef = useRef(null);
  const controlRef = useRef(null);
  const sessionIdRef = useRef(null);
  const pendingCandidatesRef = useRef([]);
  const candidateStreamRef = useRef(null);
  const micStreamRef = useRef(null);
  const aiStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const aiSourceRef = useRef(null);
  const micSourceRef = useRef(null);
  const aiAnalyserRef = useRef(null);
  const micAnalyserRef = useRef(null);
  const recordingDestinationRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const recordingUrlRef = useRef(null);
  const cloneFileRef = useRef(null);
  const visionStreamRef = useRef(null);
  const visionIntervalRef = useRef(null);
  const visionStatusTickRef = useRef(null);
  const visionLastFrameDataRef = useRef(null);
  // performance.now() of the last frame actually sent, mirrored from the
  // visionLastSentAt state so the capture interval can read it without a
  // stale closure.
  const visionLastSentAtRef = useRef(0);
  const lastFramePreviewRef = useRef(null);
  const lastFrameMetaRef = useRef(null);
  const pendingVisionFramesRef = useRef(new Map());
  const pendingDetailFrameRef = useRef(null);
  const visionFrameSeqRef = useRef(0);
  const heartbeatTimerRef = useRef(null);
  const pingSeqRef = useRef(0);
  const pendingPingsRef = useRef(new Map());
  const lastPongAtRef = useRef(0);
  const missedPongRef = useRef(0);
  const heartbeatWarnedRef = useRef(false);
  const lastRewindClickRef = useRef(0);
  const lastBookmarkClickRef = useRef(0);
  const lastInterruptClickRef = useRef(0);
  const liveConfigPendingRef = useRef({});
  const liveConfigTimerRef = useRef(null);
  // The connect-time config payload as it was sent on channel open, so the
  // ready handler can diff it against current state.
  const sentConfigRef = useRef(null);
  const interruptTimerRef = useRef(null);
  const reconnectGraceTimerRef = useRef(null);
  // Holds the latest reconnect callback so the one-time PC state handlers
  // installed at connect can trigger a restart with current phase/refs.
  const reconnectRef = useRef(null);
  const connectHoldTimerRef = useRef(null);
  const connectHoldTickRef = useRef(null);
  const assistantIdleTimerRef = useRef(null);
  const configFileRef = useRef(null);
  const profileLibraryFileRef = useRef(null);
  const voicePreviewAudioRef = useRef(null);
  const uploadedVoicePreviewUrlRef = useRef("");
  // Object URL for the synthesized preset-voice sample blob. Revoked when a
  // new preview supersedes it or playback ends, so blobs don't accumulate.
  const voicePreviewObjectUrlRef = useRef("");
  const lastServerEventRef = useRef({ text: "", at: 0 });
  const assistantTurnRef = useRef({ startedAt: 0, startLength: 0, lastChunkAt: 0, lastLength: 0, words: 0 });
  const transcriptLengthRef = useRef(0);
  const sessionStartedAtRef = useRef(0);
  // Tracks the id of the user turn currently awaiting recognized words, so
  // a user_text message can upgrade the right audio-only row. Null when no
  // user turn is open.
  const userTurnOpenRef = useRef(null);
  // Id of the assistant transcript segment currently receiving text chunks.
  const aiTurnOpenRef = useRef(null);
  const recordingPlaybackRef = useRef(null);
  const stateRef = useRef({});
  const bargeActiveRef = useRef(false);
  // Latches once the microphone channel registers speech, so a user turn is
  // recorded when the assistant next resumes. Cleared after the turn is
  // pushed. Drives the audio-only transcript marker.
  const userSpokeRef = useRef(false);
  // performance.now() at the tick the latched speech first registered; used
  // as the user turn's timestamp so it sorts ahead of the assistant reply
  // that answers it (whose segment opens before the meter detects resumed
  // assistant audio).
  const userSpokeAtRef = useRef(0);

  stateRef.current = { visionOn, visionPaused, visionInjecting, phase, interrupting, jitterBuffer };

  // Latest live-tunable sampling values, refreshed every render. The rail
  // sliders stay interactive during connecting/warmup while sendLiveConfig
  // drops updates, so the ready handler diffs these against the
  // connect-time payload and resends whatever moved in that window.
  const liveTuningRef = useRef({});
  liveTuningRef.current = {
    text_temperature: Number(textTemp),
    audio_temperature: Number(audioTemp),
    text_topk: Number.parseInt(textTopk, 10),
    audio_topk: Number.parseInt(audioTopk, 10),
    repetition_penalty: Number(repPenalty),
    repetition_penalty_context: Number.parseInt(repContext, 10),
    padding_bonus: Number(padBonus),
    max_turn_text_tokens: Number.parseInt(maxTurn, 10),
  };

  const isLive = phase === "live";
  const cfgLocked = phase === "connecting" || phase === "warmup" || phase === "live";
  // While locked, the config column collapses to a rail by default; the user
  // can peek at the frozen settings (sideExpanded) and re-collapse. Never
  // unfreezes the controls.
  const sideCollapsed = cfgLocked && !sideExpanded;
  const isBusy = connectionIssue === "busy";
  const isTurnFailed = connectionIssue === "turn";

  const addNotice = useCallback((level, text, kind = "event", extra = {}) => {
    const ts = new Date().toTimeString().slice(0, 8);
    const offsetMs = sessionStartedAtRef.current ? Math.max(0, performance.now() - sessionStartedAtRef.current) : 0;
    const noticeId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    setNotices((items) => [{ id: noticeId, ts, level, text }, ...items].slice(0, 20));
    setSessionTimeline((items) => [
      { id: `${Date.now()}-${items.length}`, ts, offsetMs, level, kind, label: text, ...extra },
      ...items,
    ].slice(0, 80));
  }, []);

  const recordRttSample = useCallback((rtt) => {
    if (!(rtt > 0)) return;
    const value = Math.round(rtt);
    setLatencyMs(value);
    setRttSamples((samples) => {
      const next = [...samples.slice(-79), value];
      const tail = [...next.slice(-20)].sort((a, b) => a - b);
      const percentileIndex = Math.min(tail.length - 1, Math.ceil(tail.length * 0.95) - 1);
      setTailLatencyMs(tail[Math.max(0, percentileIndex)] || 0);
      return next;
    });
  }, []);

  const pulseInterrupt = useCallback(() => {
    setInterrupting(true);
    if (interruptTimerRef.current) clearTimeout(interruptTimerRef.current);
    interruptTimerRef.current = window.setTimeout(() => {
      setInterrupting(false);
      interruptTimerRef.current = null;
    }, 1800);
  }, []);

  const clearUploadedVoice = useCallback(() => {
    voicePreviewAudioRef.current?.pause?.();
    voicePreviewAudioRef.current = null;
    if (uploadedVoicePreviewUrlRef.current) {
      URL.revokeObjectURL(uploadedVoicePreviewUrlRef.current);
      uploadedVoicePreviewUrlRef.current = "";
    }
    if (voicePreviewObjectUrlRef.current) {
      URL.revokeObjectURL(voicePreviewObjectUrlRef.current);
      voicePreviewObjectUrlRef.current = "";
    }
    setPreviewing(null);
    setUploadedVoiceFilename("");
    setUploadedVoiceLabel("");
    setUploadedVoiceMeta(null);
    setUploadedVoicePreviewUrl("");
    setUploadStatus("");
    setUploadKind("");
  }, []);

  const getMicConstraints = useCallback(
    () => ({
      echoCancellation: echoCancel,
      noiseSuppression: noiseSupp,
      autoGainControl: autoGain,
    }),
    [echoCancel, noiseSupp, autoGain],
  );

  const refreshAudioOutputs = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const outputs = devices
        .filter((device) => device.kind === "audiooutput")
        .map((device, index) => ({
          id: device.deviceId || `output-${index}`,
          label: device.label || `Output ${index + 1}`,
        }));
      setAudioOutputs(outputs);
    } catch (error) {
      addNotice("warn", `Could not list outputs: ${error.message || error}`);
    }
  }, [addNotice]);

  useEffect(() => {
    const track = micStreamRef.current?.getAudioTracks?.()[0];
    if (!track) return;
    track.applyConstraints(getMicConstraints()).catch(() => {
      addNotice("warn", "Mic constraints will apply next session");
    });
  }, [getMicConstraints, addNotice]);

  useEffect(() => {
    refreshAudioOutputs();
  }, [refreshAudioOutputs]);

  useEffect(() => {
    let cancelled = false;
    fetchVoiceList().then((ids) => {
      if (!cancelled && ids) setVoiceList(ids);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Keep the blend's secondary voice valid: it must exist in the list and
  // differ from the primary, so a voice never blends with itself. Snaps to
  // the first other voice when the primary moves onto the secondary or the
  // secondary falls out of the list.
  useEffect(() => {
    if (voiceB !== voice && voiceList.includes(voiceB)) return;
    const next = voiceList.find((item) => item !== voice);
    if (next && next !== voiceB) setVoiceB(next);
  }, [voice, voiceB, voiceList, setVoiceB]);

  const canRouteOutput = typeof HTMLMediaElement !== "undefined" && "setSinkId" in HTMLMediaElement.prototype;
  const audioOutputOptions = useMemo(() => {
    const options = [
      {
        value: "default",
        label: "System default",
        desc: canRouteOutput ? "Browser default route" : "Browser controlled",
      },
      ...audioOutputs
        .filter((device) => device.id && device.id !== "default")
        .map((device) => ({
          value: device.id,
          label: device.label,
          desc: device.id === "communications" ? "Communications route" : "Detected output",
        })),
    ];
    if (outputDeviceId && outputDeviceId !== "default" && !options.some((option) => option.value === outputDeviceId)) {
      options.push({
        value: outputDeviceId,
        label: "Saved output",
        desc: "Unavailable until permission refresh",
      });
    }
    return options;
  }, [audioOutputs, canRouteOutput, outputDeviceId]);

  const visionCostUsd = visionFramesSent * VISION_PER_CALL_USD;
  const visionCostLimitActive = Number(visionCostLimitUsd) > 0;
  const visionCostRemaining = Math.max(0, Number(visionCostLimitUsd || 0) - visionCostUsd);
  const timelinePreview = useMemo(() => sessionTimeline.slice(0, 14).reverse(), [sessionTimeline]);
  const timelineDurationMs = useMemo(
    () => Math.max(1000, elapsedSec * 1000, ...sessionTimeline.map((item) => item.offsetMs || 0)),
    [elapsedSec, sessionTimeline],
  );

  useEffect(() => {
    const audio = aiAudioRef.current;
    if (!audio || !outputDeviceId) return;
    if (!audio.setSinkId) {
      if (outputDeviceId !== "default") {
        addNotice("warn", "Output routing unsupported in this browser");
      }
      return;
    }
    audio.setSinkId(outputDeviceId).catch((error) => {
      addNotice("warn", `Could not switch output: ${error.message || error}`);
    });
  }, [addNotice, outputDeviceId]);

  const allSessionProfiles = useMemo(() => [...SESSION_PROFILES, ...customProfiles], [customProfiles]);
  const selectedCustomProfile = useMemo(
    () => customProfiles.find((profile) => profile.id === sessionProfileId) || null,
    [customProfiles, sessionProfileId],
  );
  const currentProfileSnapshot = useMemo(() => {
    const label = profileName.trim() || "My profile";
    const adherence = ADHERENCE_MODES.find((item) => item.id === adherenceMode) || ADHERENCE_MODES[0];
    const expression = EXPRESSION_MODES.find((item) => item.id === expressionMode) || EXPRESSION_MODES[0];
    return {
      custom: true,
      label,
      desc: `${adherence.label} · ${expression.label} · ${uploadedVoiceFilename ? "uploaded voice" : voice}`,
      presetId,
      textPrompt,
      voice,
      adherenceMode,
      expressionMode,
      textTemp: Number(textTemp),
      textTopk: Number(textTopk),
      audioTemp: Number(audioTemp),
      audioTopk: Number(audioTopk),
      repPenalty: Number(repPenalty),
      repContext: Number(repContext),
      padBonus: Number(padBonus),
      maxTurn: Number(maxTurn),
      echoCancel: !!echoCancel,
      noiseSupp: !!noiseSupp,
      autoGain: !!autoGain,
      visionInTranscript: !!visionInTranscript,
      reinforceInSilences: !!reinforceInSilences,
      visionPrompt,
      visionIntervalMs: Number(visionIntervalMs),
      visionCostLimitUsd: Number(visionCostLimitUsd),
      seedRandom: !!seedRandom,
      seed: Number(seed),
    };
  }, [
    adherenceMode,
    audioTemp,
    audioTopk,
    autoGain,
    echoCancel,
    expressionMode,
    maxTurn,
    noiseSupp,
    padBonus,
    presetId,
    profileName,
    reinforceInSilences,
    repContext,
    repPenalty,
    seed,
    seedRandom,
    textPrompt,
    textTemp,
    textTopk,
    uploadedVoiceFilename,
    visionInTranscript,
    visionCostLimitUsd,
    visionIntervalMs,
    visionPrompt,
    voice,
  ]);

  const applyPreset = (id) => {
    const preset = PERSONA_PRESETS.find((item) => item.id === id);
    if (!preset) return;
    setSessionProfileId("custom");
    setPresetId(id);
    setTextPrompt(preset.prompt);
  };

  const applySessionProfileData = useCallback((profile) => {
    if (!profile) return;
    const preset = PERSONA_PRESETS.find((item) => item.id === profile.presetId);
    setSessionProfileId(profile.id);
    setProfileName(profile.custom ? profile.label : "My profile");
    if (typeof profile.textPrompt === "string") {
      setPresetId(preset ? preset.id : "custom");
      setTextPrompt(profile.textPrompt);
    } else if (preset) {
      setPresetId(preset.id);
      setTextPrompt(preset.prompt);
    }
    setVoice(profile.voice || "NATF1");
    setVoiceGender("all");
    setVoiceBlend(false);
    clearUploadedVoice();
    setAdherenceMode(profile.adherenceMode || "balanced");
    setExpressionMode(profile.expressionMode || "natural");
    setTextTemp(Number.isFinite(Number(profile.textTemp)) ? Number(profile.textTemp) : DEFAULTS.textTemp);
    setTextTopk(Number.isFinite(Number(profile.textTopk)) ? Number(profile.textTopk) : DEFAULTS.textTopk);
    setAudioTemp(Number.isFinite(Number(profile.audioTemp)) ? Number(profile.audioTemp) : DEFAULTS.audioTemp);
    setAudioTopk(Number.isFinite(Number(profile.audioTopk)) ? Number(profile.audioTopk) : DEFAULTS.audioTopk);
    setRepPenalty(Number.isFinite(Number(profile.repPenalty)) ? Number(profile.repPenalty) : DEFAULTS.repPenalty);
    setRepContext(Number.isFinite(Number(profile.repContext)) ? Number(profile.repContext) : DEFAULTS.repContext);
    setPadBonus(Number.isFinite(Number(profile.padBonus)) ? Number(profile.padBonus) : DEFAULTS.padBonus);
    setMaxTurn(Number.isFinite(Number(profile.maxTurn)) ? Number(profile.maxTurn) : DEFAULTS.maxTurn);
    setEchoCancel(typeof profile.echoCancel === "boolean" ? profile.echoCancel : DEFAULTS.echoCancel);
    setNoiseSupp(typeof profile.noiseSupp === "boolean" ? profile.noiseSupp : DEFAULTS.noiseSupp);
    setAutoGain(typeof profile.autoGain === "boolean" ? profile.autoGain : DEFAULTS.autoGain);
    setVisionInTranscript(!!profile.visionInTranscript);
    setReinforceInSilences(!!profile.reinforceInSilences);
    setVisionPrompt(profile.visionPrompt || DEFAULT_VISION_PROMPT);
    setVisionIntervalMs(Number.isFinite(Number(profile.visionIntervalMs)) ? Number(profile.visionIntervalMs) : DEFAULTS.visionIntervalMs);
    setVisionCostLimitUsd(Number.isFinite(Number(profile.visionCostLimitUsd)) ? Number(profile.visionCostLimitUsd) : 0);
    setSeedRandom(!!profile.seedRandom);
    if (Number.isFinite(Number(profile.seed))) setSeed(Number(profile.seed));
    addNotice("ok", `Profile loaded: ${profile.label}`);
  }, [
    addNotice,
    clearUploadedVoice,
    setAdherenceMode,
    setAudioTemp,
    setAudioTopk,
    setAutoGain,
    setEchoCancel,
    setExpressionMode,
    setMaxTurn,
    setNoiseSupp,
    setPadBonus,
    setProfileName,
    setRepContext,
    setRepPenalty,
    setSeed,
    setSeedRandom,
    setTextPrompt,
    setTextTemp,
    setTextTopk,
    setVisionInTranscript,
    setReinforceInSilences,
    setVisionIntervalMs,
    setVisionPrompt,
    setVisionCostLimitUsd,
    setVoice,
    setVoiceBlend,
  ]);

  const applySessionProfile = (id) => {
    applySessionProfileData(allSessionProfiles.find((item) => item.id === id));
  };

  const composeTextPrompt = useCallback(() => {
    const adherence = ADHERENCE_MODES.find((item) => item.id === adherenceMode) || ADHERENCE_MODES[0];
    const expression = EXPRESSION_MODES.find((item) => item.id === expressionMode) || EXPRESSION_MODES[0];
    return [textPrompt || "", adherence.instruction, expression.instruction]
      .filter(Boolean)
      .join("\n\n");
  }, [adherenceMode, expressionMode, textPrompt]);

  const buildConfigPayload = useCallback(() => {
    const selectedVoice = uploadedVoiceFilename || (voice ? `${voice}.pt` : "");
    // Blend is built-in voices only: an uploaded clip has no per-frame
    // embedding sequence to align, so the second voice is sent only when
    // no clip is selected, the two ids differ, and the mix is nonzero.
    const blendOn = voiceBlend && !uploadedVoiceFilename && voiceB && voiceB !== voice && blendMix > 0;
    return {
      voice_prompt: selectedVoice,
      voice_prompt_b: blendOn ? `${voiceB}.pt` : "",
      voice_blend_mix: blendOn ? blendMix / 100 : 0,
      // Only an uploaded clip has a strength to scale; preset and blended
      // prompts send 1.0 so the contract stays uniform and the server's
      // preset path ignores it.
      clone_strength: uploadedVoiceFilename ? Number(cloneStrength) / 100 : 1.0,
      text_prompt: composeTextPrompt(),
      vision_prompt: visionPrompt || "",
      vision_in_transcript: !!visionInTranscript,
      reinforce_in_silences: !!reinforceInSilences,
      audio_temperature: Number(audioTemp),
      text_temperature: Number(textTemp),
      text_topk: Number.parseInt(textTopk, 10),
      audio_topk: Number.parseInt(audioTopk, 10),
      repetition_penalty: Number(repPenalty),
      repetition_penalty_context: Number.parseInt(repContext, 10),
      padding_bonus: Number(padBonus),
      max_turn_text_tokens: Number.parseInt(maxTurn, 10),
      seed: seedRandom ? -1 : Number.parseInt(seed, 10),
      session_timeout_sec: Number(idleTimeout) > 0 ? Number(idleTimeout) * 60 : 0,
      vision_cost_limit_usd: Number(visionCostLimitUsd) || 0,
      vision_cost_per_call_usd: VISION_PER_CALL_USD,
    };
  }, [
    uploadedVoiceFilename,
    voice,
    voiceBlend,
    voiceB,
    blendMix,
    cloneStrength,
    composeTextPrompt,
    visionPrompt,
    visionInTranscript,
    reinforceInSilences,
    audioTemp,
    textTemp,
    textTopk,
    audioTopk,
    repPenalty,
    repContext,
    padBonus,
    maxTurn,
    seedRandom,
    seed,
    idleTimeout,
    visionCostLimitUsd,
  ]);

  const buildConfigProfile = useCallback(() => ({
    version: 1,
    exported_at: new Date().toISOString(),
    session_profile_id: sessionProfileId === "custom" ? "" : sessionProfileId,
    preset_id: presetId,
    adherence_mode: adherenceMode,
    expression_mode: expressionMode,
    voice_filter: { gender: voiceGender },
    uploaded_voice_label: uploadedVoiceLabel,
    uploaded_voice_meta: uploadedVoiceMeta,
    config: { ...buildConfigPayload(), text_prompt: textPrompt || "" },
    resolved_text_prompt: composeTextPrompt(),
    mic: {
      echo_cancellation: !!echoCancel,
      noise_suppression: !!noiseSupp,
      auto_gain: !!autoGain,
      output_device_id: outputDeviceId,
    },
    vision: {
      interval_ms: Number(visionIntervalMs),
      cost_limit_usd: Number(visionCostLimitUsd),
    },
  }), [
    presetId,
    sessionProfileId,
    adherenceMode,
    expressionMode,
    voiceGender,
    uploadedVoiceLabel,
    uploadedVoiceMeta,
    buildConfigPayload,
    composeTextPrompt,
    textPrompt,
    echoCancel,
    noiseSupp,
    autoGain,
    outputDeviceId,
    visionIntervalMs,
    visionCostLimitUsd,
  ]);

  const applyConfigProfile = useCallback((profile) => {
    const config = profile?.config && typeof profile.config === "object" ? profile.config : profile;
    if (!config || typeof config !== "object") {
      throw new Error("config JSON must be an object");
    }
    const readNumber = (value, fallback) => {
      const next = Number(value);
      return Number.isFinite(next) ? next : fallback;
    };
    const text = typeof config.text_prompt === "string" ? config.text_prompt : textPrompt;
    const preset = PERSONA_PRESETS.find((item) => item.id === profile?.preset_id || item.prompt === text);
    setSessionProfileId(allSessionProfiles.some((item) => item.id === profile?.session_profile_id) ? profile.session_profile_id : "custom");
    setPresetId(preset ? preset.id : "custom");
    setTextPrompt(text);
    if (ADHERENCE_MODES.some((item) => item.id === profile?.adherence_mode)) {
      setAdherenceMode(profile.adherence_mode);
    }
    if (EXPRESSION_MODES.some((item) => item.id === profile?.expression_mode)) {
      setExpressionMode(profile.expression_mode);
    }
    if (typeof config.vision_prompt === "string") setVisionPrompt(config.vision_prompt);
    setVisionInTranscript(!!config.vision_in_transcript);
    setReinforceInSilences(!!config.reinforce_in_silences);
    setAudioTemp(readNumber(config.audio_temperature, DEFAULTS.audioTemp));
    setTextTemp(readNumber(config.text_temperature, DEFAULTS.textTemp));
    setTextTopk(readNumber(config.text_topk, DEFAULTS.textTopk));
    setAudioTopk(readNumber(config.audio_topk, DEFAULTS.audioTopk));
    setRepPenalty(readNumber(config.repetition_penalty, DEFAULTS.repPenalty));
    setRepContext(readNumber(config.repetition_penalty_context, DEFAULTS.repContext));
    setPadBonus(readNumber(config.padding_bonus, DEFAULTS.padBonus));
    setMaxTurn(readNumber(config.max_turn_text_tokens, DEFAULTS.maxTurn));
    // Stored as seconds; the stepper edits minutes in 5-step increments.
    // Snap so a hand-edited off-grid value still lands on a reachable step.
    const timeoutMin = readNumber(config.session_timeout_sec, 0) / 60;
    setIdleTimeout(Math.max(0, Math.min(60, Math.round(timeoutMin / 5) * 5)));
    const nextSeed = Number(config.seed);
    setSeedRandom(!Number.isFinite(nextSeed) || nextSeed === -1);
    if (Number.isFinite(nextSeed) && nextSeed !== -1) setSeed(nextSeed);

    const voicePrompt = typeof config.voice_prompt === "string" ? config.voice_prompt : "";
    if (voicePrompt.startsWith("upload:")) {
      clearUploadedVoice();
      setUploadStatus("Config references an uploaded clip. Re-upload the audio to use it.");
      setUploadKind("error");
    } else if (voicePrompt.endsWith(".pt")) {
      const voiceName = voicePrompt.slice(0, -3);
      if (VOICES.includes(voiceName)) setVoice(voiceName);
      clearUploadedVoice();
    }

    // Repopulate the strength slider so a re-uploaded clip resumes at the
    // saved value. The clip itself can't be restored from the config alone.
    const cloneStrengthFloat = readNumber(config.clone_strength, cloneStrength / 100);
    setCloneStrength(Math.max(0, Math.min(100, Math.round(cloneStrengthFloat * 100))));

    const voicePromptB = typeof config.voice_prompt_b === "string" ? config.voice_prompt_b : "";
    const blendMixFloat = readNumber(config.voice_blend_mix, 0);
    if (voicePromptB.endsWith(".pt") && blendMixFloat > 0) {
      const voiceBName = voicePromptB.slice(0, -3);
      if (VOICES.includes(voiceBName)) setVoiceB(voiceBName);
      setBlendMix(Math.max(0, Math.min(100, Math.round(blendMixFloat * 100))));
      setVoiceBlend(true);
    } else {
      setVoiceBlend(false);
    }

    const filter = profile?.voice_filter || {};
    if (["all", "F", "M"].includes(filter.gender)) setVoiceGender(filter.gender);
    const mic = profile?.mic || {};
    if (typeof mic.echo_cancellation === "boolean") setEchoCancel(mic.echo_cancellation);
    if (typeof mic.noise_suppression === "boolean") setNoiseSupp(mic.noise_suppression);
    if (typeof mic.auto_gain === "boolean") setAutoGain(mic.auto_gain);
    if (typeof mic.output_device_id === "string") setOutputDeviceId(mic.output_device_id || "default");
    const interval = readNumber(profile?.vision?.interval_ms, visionIntervalMs);
    if (interval >= 1000 && interval <= 30000) setVisionIntervalMs(interval);
    setVisionCostLimitUsd(Math.max(0, readNumber(profile?.vision?.cost_limit_usd, visionCostLimitUsd)));
  }, [allSessionProfiles, clearUploadedVoice, cloneStrength, textPrompt, visionCostLimitUsd, visionIntervalMs, setAdherenceMode, setExpressionMode, setAudioTemp, setTextTemp, setTextTopk, setAudioTopk, setRepPenalty, setRepContext, setPadBonus, setMaxTurn, setSeedRandom, setSeed, setIdleTimeout, setTextPrompt, setVisionPrompt, setVisionInTranscript, setReinforceInSilences, setVoice, setVoiceBlend, setVoiceB, setBlendMix, setCloneStrength, setEchoCancel, setNoiseSupp, setAutoGain, setOutputDeviceId, setVisionIntervalMs, setVisionCostLimitUsd]);

  const exportConfig = useCallback(() => {
    const profile = JSON.stringify(buildConfigProfile(), null, 2);
    const blob = new Blob([profile], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "personaplex-config.json";
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    addNotice("ok", "Config exported");
  }, [addNotice, buildConfigProfile]);

  const importConfig = useCallback(
    async (file) => {
      if (!file) return;
      try {
        const profile = JSON.parse(await file.text());
        applyConfigProfile(profile);
        addNotice("ok", "Config imported");
      } catch (error) {
        addNotice("err", `Config import failed: ${error.message || error}`);
      }
    },
    [addNotice, applyConfigProfile],
  );

  const saveCustomProfile = useCallback(() => {
    const label = profileName.trim() || "My profile";
    const now = new Date().toISOString();
    const profile = {
      ...currentProfileSnapshot,
      id: storedProfileId(),
      label,
      createdAt: now,
      updatedAt: now,
    };
    setCustomProfiles((items) => [profile, ...items].slice(0, 24));
    setSessionProfileId(profile.id);
    setProfileName(label);
    addNotice("ok", `Profile saved: ${label}`);
  }, [addNotice, currentProfileSnapshot, profileName, setCustomProfiles, setProfileName]);

  const duplicateCurrentProfile = useCallback(() => {
    const baseLabel = profileName.trim() || selectedCustomProfile?.label || "My profile";
    const label = `${baseLabel} copy`.slice(0, 48);
    const now = new Date().toISOString();
    const profile = {
      ...currentProfileSnapshot,
      id: storedProfileId(),
      label,
      createdAt: now,
      updatedAt: now,
    };
    setCustomProfiles((items) => [profile, ...items].slice(0, 24));
    setSessionProfileId(profile.id);
    setProfileName(label);
    addNotice("ok", `Profile duplicated: ${label}`);
  }, [addNotice, currentProfileSnapshot, profileName, selectedCustomProfile, setCustomProfiles, setProfileName]);

  const updateCustomProfile = useCallback(() => {
    if (!selectedCustomProfile) {
      saveCustomProfile();
      return;
    }
    const label = profileName.trim() || selectedCustomProfile.label || "My profile";
    const updated = {
      ...currentProfileSnapshot,
      id: selectedCustomProfile.id,
      label,
      createdAt: selectedCustomProfile.createdAt || new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setCustomProfiles((items) => items.map((item) => (item.id === selectedCustomProfile.id ? updated : item)));
    setSessionProfileId(updated.id);
    setProfileName(label);
    addNotice("ok", `Profile updated: ${label}`);
  }, [addNotice, currentProfileSnapshot, profileName, saveCustomProfile, selectedCustomProfile, setCustomProfiles, setProfileName]);

  const deleteCustomProfile = useCallback(() => {
    if (!selectedCustomProfile) return;
    const label = selectedCustomProfile.label || "profile";
    if (!globalThis.confirm?.(`Delete ${label}?`)) return;
    setCustomProfiles((items) => items.filter((item) => item.id !== selectedCustomProfile.id));
    setSessionProfileId("custom");
    setProfileName("My profile");
    addNotice("warn", `Profile deleted: ${label}`);
  }, [addNotice, selectedCustomProfile, setCustomProfiles, setProfileName]);

  const exportProfileLibrary = useCallback(() => {
    const payload = JSON.stringify({
      version: 1,
      exported_at: new Date().toISOString(),
      profiles: customProfiles,
    }, null, 2);
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "personaplex-profiles.json";
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    addNotice("ok", "Profiles exported");
  }, [addNotice, customProfiles]);

  const importProfileLibrary = useCallback(async (file) => {
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      const profiles = Array.isArray(payload?.profiles) ? payload.profiles : Array.isArray(payload) ? payload : [];
      const imported = profiles
        .filter((profile) => profile && typeof profile === "object" && typeof profile.label === "string")
        .map((profile) => ({
          ...profile,
          id: storedProfileId(),
          custom: true,
          label: profile.label.slice(0, 48),
          desc: typeof profile.desc === "string" ? profile.desc.slice(0, 120) : "Imported card",
          importedAt: new Date().toISOString(),
        }));
      if (!imported.length) throw new Error("no profiles found");
      setCustomProfiles((items) => [...imported, ...items].slice(0, 24));
      addNotice("ok", `Profiles imported: ${imported.length}`);
    } catch (error) {
      addNotice("err", `Profile import failed: ${error.message || error}`);
    }
  }, [addNotice, setCustomProfiles]);

  const pinCurrentTuning = useCallback(() => {
    const label = sessionProfileId === "custom"
      ? profileName.trim() || "Current tuning"
      : allSessionProfiles.find((item) => item.id === sessionProfileId)?.label || "Current tuning";
    setPinnedTuning({
      label,
      savedAt: new Date().toISOString(),
      profile: { ...currentProfileSnapshot, label },
    });
    addNotice("ok", `Pinned tuning: ${label}`);
  }, [addNotice, allSessionProfiles, currentProfileSnapshot, profileName, sessionProfileId, setPinnedTuning]);

  const applyPinnedTuning = useCallback(() => {
    if (!pinnedTuning?.profile) return;
    applySessionProfileData({
      ...pinnedTuning.profile,
      id: "custom",
      custom: true,
      label: pinnedTuning.label || pinnedTuning.profile.label || "Pinned tuning",
    });
  }, [applySessionProfileData, pinnedTuning]);

  const tuningDiffs = useMemo(() => {
    const pinned = pinnedTuning?.profile;
    if (!pinned) return [];
    const fields = [
      ["Prompt", currentProfileSnapshot.textPrompt, pinned.textPrompt],
      ["Voice", currentProfileSnapshot.voice, pinned.voice],
      ["Adherence", currentProfileSnapshot.adherenceMode, pinned.adherenceMode],
      ["Expression", currentProfileSnapshot.expressionMode, pinned.expressionMode],
      ["Text t", currentProfileSnapshot.textTemp, pinned.textTemp],
      ["Text k", currentProfileSnapshot.textTopk, pinned.textTopk],
      ["Audio t", currentProfileSnapshot.audioTemp, pinned.audioTemp],
      ["Audio k", currentProfileSnapshot.audioTopk, pinned.audioTopk],
      ["Rep", currentProfileSnapshot.repPenalty, pinned.repPenalty],
      ["Rep ctx", currentProfileSnapshot.repContext, pinned.repContext],
      ["Pad", currentProfileSnapshot.padBonus, pinned.padBonus],
      ["Max turn", currentProfileSnapshot.maxTurn, pinned.maxTurn],
      ["Echo", currentProfileSnapshot.echoCancel, pinned.echoCancel],
      ["Noise", currentProfileSnapshot.noiseSupp, pinned.noiseSupp],
      ["AGC", currentProfileSnapshot.autoGain, pinned.autoGain],
      ["Vision cadence", currentProfileSnapshot.visionIntervalMs, pinned.visionIntervalMs],
      ["Vision budget", currentProfileSnapshot.visionCostLimitUsd || "off", pinned.visionCostLimitUsd || "off"],
      ["Seed", currentProfileSnapshot.seedRandom ? "random" : currentProfileSnapshot.seed, pinned.seedRandom ? "random" : pinned.seed],
    ];
    return fields
      .filter(([, current, previous]) => String(current ?? "") !== String(previous ?? ""))
      .map(([label, current, previous]) => ({ label, current, previous }));
  }, [currentProfileSnapshot, pinnedTuning]);

  const postCandidate = useCallback(async (candidate) => {
    if (!sessionIdRef.current) return;
    try {
      await fetch("/api/rtc/candidate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          candidate: candidate ? candidate.candidate : null,
          sdpMid: candidate ? candidate.sdpMid : null,
          sdpMLineIndex: candidate ? candidate.sdpMLineIndex : null,
        }),
      });
    } catch (error) {
      console.warn("candidate POST failed:", error);
    }
  }, []);

  const flushPendingCandidates = useCallback(async () => {
    const buffered = pendingCandidatesRef.current;
    pendingCandidatesRef.current = [];
    for (const candidate of buffered) {
      await postCandidate(candidate);
    }
  }, [postCandidate]);

  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current.addEventListener("statechange", () => {
        const context = audioContextRef.current;
        if (!context) return;
        if (context.state === "suspended" || context.state === "interrupted") {
          context.resume().catch(() => {});
        }
      });
    }
    if (audioContextRef.current.state === "suspended") {
      await audioContextRef.current.resume();
    }
  }, []);

  const attachAudioGraph = useCallback(() => {
    const context = audioContextRef.current;
    if (!context) return;
    if (!recordingDestinationRef.current) {
      recordingDestinationRef.current = context.createMediaStreamDestination();
    }
    if (aiStreamRef.current && !aiSourceRef.current) {
      aiSourceRef.current = context.createMediaStreamSource(aiStreamRef.current);
      aiAnalyserRef.current = context.createAnalyser();
      aiAnalyserRef.current.fftSize = 256;
      aiAnalyserRef.current.smoothingTimeConstant = 0.85;
      aiSourceRef.current.connect(aiAnalyserRef.current);
      aiSourceRef.current.connect(recordingDestinationRef.current);
    }
    if (micStreamRef.current && !micSourceRef.current) {
      micSourceRef.current = context.createMediaStreamSource(micStreamRef.current);
      micAnalyserRef.current = context.createAnalyser();
      micAnalyserRef.current.fftSize = 256;
      micAnalyserRef.current.smoothingTimeConstant = 0.85;
      micSourceRef.current.connect(micAnalyserRef.current);
      micSourceRef.current.connect(recordingDestinationRef.current);
    }
  }, []);

  const startRecording = useCallback(() => {
    recordedChunksRef.current = [];
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
    recordingUrlRef.current = null;
    setRecordingUrl(null);
    if (!recordingDestinationRef.current || !window.MediaRecorder) return;
    try {
      const recorder = new MediaRecorder(recordingDestinationRef.current.stream);
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data?.size) recordedChunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        if (!recordedChunksRef.current.length) return;
        const type = recorder.mimeType || "audio/webm";
        const blob = new Blob(recordedChunksRef.current, { type });
        const url = URL.createObjectURL(blob);
        recordingUrlRef.current = url;
        setRecordingMime(type);
        setRecordingUrl(url);
      };
      recorder.start();
    } catch (error) {
      addNotice("warn", "Session recording unavailable");
      console.warn("MediaRecorder unavailable:", error);
    }
  }, [addNotice]);

  const stopRecording = useCallback((showDownload) => {
    const recorder = mediaRecorderRef.current;
    if (!showDownload) recordedChunksRef.current = [];
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        // Ignore recorder shutdown failures.
      }
    }
    mediaRecorderRef.current = null;
  }, []);

  const stopVision = useCallback(() => {
    if (visionIntervalRef.current) clearInterval(visionIntervalRef.current);
    if (visionStatusTickRef.current) clearInterval(visionStatusTickRef.current);
    visionIntervalRef.current = null;
    visionStatusTickRef.current = null;
    visionStreamRef.current?.getTracks?.().forEach((track) => {
      track.stop();
    });
    visionStreamRef.current = null;
    visionLastFrameDataRef.current = null;
    lastFramePreviewRef.current = null;
    if (visionVideoRef.current) visionVideoRef.current.srcObject = null;
    setVisionOn(false);
    setVisionPaused(false);
    setVisionInjecting(false);
    setCurrentCaption("");
    visionLastSentAtRef.current = 0;
    setVisionLastSentAt(0);
  }, []);

  const cleanup = useCallback(
    (options = {}) => {
      const { showDownload = false, keepPhase = false } = options;
      stopRecording(showDownload);
      if (candidateStreamRef.current) {
        try {
          candidateStreamRef.current.close();
        } catch {
          // Ignore stream close failures.
        }
        candidateStreamRef.current = null;
      }
      sessionIdRef.current = null;
      pendingCandidatesRef.current = [];
      if (controlRef.current) {
        try {
          controlRef.current.close();
        } catch {
          // Ignore channel close failures.
        }
        controlRef.current = null;
      }
      if (pcRef.current) {
        try {
          pcRef.current.ontrack = null;
          pcRef.current.onconnectionstatechange = null;
          pcRef.current.oniceconnectionstatechange = null;
          pcRef.current.onicecandidate = null;
          pcRef.current.close();
        } catch {
          // Ignore peer close failures.
        }
        pcRef.current = null;
      }
      if (aiAudioRef.current) {
        try {
          aiAudioRef.current.pause();
          aiAudioRef.current.srcObject = null;
        } catch {
          // Ignore audio cleanup failures.
        }
      }
      for (const nodeRef of [aiSourceRef, micSourceRef, aiAnalyserRef, micAnalyserRef, recordingDestinationRef]) {
        try {
          nodeRef.current?.disconnect?.();
        } catch {
          // Ignore graph disconnect failures.
        }
        nodeRef.current = null;
      }
      micStreamRef.current?.getTracks?.().forEach((track) => {
        track.stop();
      });
      micStreamRef.current = null;
      aiStreamRef.current = null;
      if (interruptTimerRef.current) clearTimeout(interruptTimerRef.current);
      interruptTimerRef.current = null;
      if (reconnectGraceTimerRef.current) clearTimeout(reconnectGraceTimerRef.current);
      reconnectGraceTimerRef.current = null;
      setReconnecting(false);
      setNetStats({ quality: 0, jitterMs: 0, lossPct: 0, candidate: "" });
      if (liveConfigTimerRef.current) clearTimeout(liveConfigTimerRef.current);
      liveConfigTimerRef.current = null;
      liveConfigPendingRef.current = {};
      if (heartbeatTimerRef.current) clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
      pendingPingsRef.current.clear();
      missedPongRef.current = 0;
      heartbeatWarnedRef.current = false;
      lastPongAtRef.current = 0;
      if (connectHoldTimerRef.current) clearTimeout(connectHoldTimerRef.current);
      if (connectHoldTickRef.current) clearInterval(connectHoldTickRef.current);
      if (assistantIdleTimerRef.current) clearTimeout(assistantIdleTimerRef.current);
      connectHoldTimerRef.current = null;
      connectHoldTickRef.current = null;
      assistantIdleTimerRef.current = null;
      setConnectHoldPct(0);
      setInterrupting(false);
      stopVision();
      if (navigator.mediaSession) navigator.mediaSession.playbackState = "none";
      setLevels({ mic: 0, ai: 0 });
      setSpeaking(null);
      setGpuStat({ vramUsed: 0, gpuUtil: null });
      setRtf(0);
      // The server's finalize event usually can't reach a closing data
      // channel, so on a real session end mark an active server recording
      // ready here; the file exists once the session has ended.
      if (showDownload) {
        setServerRecording((prev) =>
          prev?.url ? { ...prev, active: false, ready: true } : prev,
        );
      }
      if (!keepPhase) setPhase(showDownload ? "ended" : "idle");
    },
    [stopRecording, stopVision],
  );

  // biome-ignore lint/correctness/useExhaustiveDependencies: captureFrame is declared after this hook (temporal dead zone); it is referentially stable, so omitting it is safe
  const handleControlMessage = useCallback(
    (message) => {
      if (message.type === "ready") {
        setPhase("live");
        setStageMessage("Connected");
        setConnectionIssue(null);
        setServerInfo((info) => ({
          gpuName: typeof message.gpu_name === "string" ? message.gpu_name : info.gpuName,
          vramTotal: Number.isFinite(message.vram_total) ? message.vram_total : info.vramTotal,
          serverBuild: typeof message.server_build === "string" ? message.server_build : info.serverBuild,
        }));
        addNotice("ok", "Warmup complete, session live");
        toast("Connected");
        if (navigator.mediaSession) {
          try {
            if (window.MediaMetadata) {
              navigator.mediaSession.metadata = new MediaMetadata({
                title: "PersonaPlex Conversation",
                artist: "PersonaPlex",
              });
            }
            navigator.mediaSession.playbackState = "playing";
          } catch {
            // Media Session metadata is best-effort browser polish.
          }
        }
        attachAudioGraph();
        startRecording();
        // Resend any live-tunable value the user moved while the phase was
        // connecting/warmup: those slider changes updated React state but
        // sendLiveConfig drops updates until the session is live.
        const sentConfig = sentConfigRef.current || {};
        const drifted = {};
        for (const [key, val] of Object.entries(liveTuningRef.current)) {
          if (sentConfig[key] !== val) drifted[key] = val;
        }
        if (Object.keys(drifted).length && controlRef.current?.readyState === "open") {
          controlRef.current.send(JSON.stringify({ type: "update_config", ...drifted }));
        }
      } else if (message.type === "text") {
        const chunk = message.v || "";
        if (!chunk) return;
        const now = performance.now();
        if (!assistantTurnRef.current.startedAt || now - (assistantTurnRef.current.lastChunkAt || 0) > 1600) {
          assistantTurnRef.current = {
            startedAt: now,
            startLength: transcriptLengthRef.current,
            lastChunkAt: now,
            lastLength: transcriptLengthRef.current,
            words: 0,
          };
          const ts = new Date().toTimeString().slice(0, 8);
          const offsetMs = sessionStartedAtRef.current ? Math.max(0, now - sessionStartedAtRef.current) : 0;
          setSessionTimeline((items) => [
            { id: `${Date.now()}-ai`, ts, offsetMs, level: "ok", kind: "assistant", label: "Assistant turn started" },
            ...items,
          ].slice(0, 80));
          const aiTurnId = `${Date.now()}-ai-${Math.random().toString(36).slice(2, 7)}`;
          aiTurnOpenRef.current = aiTurnId;
          setAiTurns((turns) => [...turns, { id: aiTurnId, at: now, text: "" }].slice(-60));
        } else {
          assistantTurnRef.current.lastChunkAt = now;
        }
        if (assistantIdleTimerRef.current) clearTimeout(assistantIdleTimerRef.current);
        setTranscriptText((text) => {
          const next = text + chunk;
          const turnText = next.slice(assistantTurnRef.current.startLength).trim();
          const words = turnText ? turnText.split(/\s+/).filter(Boolean).length : 0;
          const seconds = Math.max(0.1, (now - assistantTurnRef.current.startedAt) / 1000);
          transcriptLengthRef.current = next.length;
          assistantTurnRef.current.lastLength = next.length;
          assistantTurnRef.current.words = words;
          setAssistantRate({ words, seconds, wpm: Math.round((words / seconds) * 60) });
          return next;
        });
        setAiTurns((turns) =>
          turns.map((turn) =>
            turn.id === aiTurnOpenRef.current ? { ...turn, text: turn.text + chunk } : turn,
          ),
        );
        assistantIdleTimerRef.current = window.setTimeout(() => {
          const turn = assistantTurnRef.current;
          if (!turn.startedAt) return;
          const endedAt = performance.now();
          const ts = new Date().toTimeString().slice(0, 8);
          const offsetMs = sessionStartedAtRef.current ? Math.max(0, endedAt - sessionStartedAtRef.current) : 0;
          const durationMs = Math.max(0, endedAt - turn.startedAt);
          setSessionTimeline((items) => [
            {
              id: `${Date.now()}-ai-end`,
              ts,
              offsetMs,
              durationMs,
              level: "ok",
              kind: "assistant-end",
              label: `Assistant turn closed: ${turn.words || 0} words`,
            },
            ...items,
          ].slice(0, 80));
          assistantTurnRef.current = {
            startedAt: 0,
            startLength: transcriptLengthRef.current,
            lastChunkAt: 0,
            lastLength: transcriptLengthRef.current,
            words: 0,
          };
          aiTurnOpenRef.current = null;
          assistantIdleTimerRef.current = null;
        }, 1600);
      } else if (message.type === "user_text") {
        // Optional server-side recognition of the user's speech. Additive:
        // absent on servers without ASR, in which case user turns stay
        // audio-only. A non-final message updates the in-progress turn; a
        // final message closes it. The text replaces (not appends to) the
        // turn body, since the server sends the growing turn text each time.
        const userText = (message.v || "").trim();
        const userFinal = !!message.final;
        if (!userText && !userFinal) return;
        const openId = userTurnOpenRef.current;
        // A new turn id is needed only when there is no open turn to
        // upgrade. Compute it (and update the open-turn ref) here so the
        // state updater stays a pure function of its input.
        const freshId =
          openId == null
            ? `${Date.now()}-you-${Math.random().toString(36).slice(2, 7)}`
            : null;
        // Stamp a fresh turn with the time the mic registered the speech
        // (when the latch has one) so it sorts ahead of the reply.
        const freshAt = userSpokeAtRef.current || performance.now();
        if (freshId !== null) userTurnOpenRef.current = userFinal ? null : freshId;
        else if (userFinal) userTurnOpenRef.current = null;
        if (userFinal) {
          // The recognizer closed this utterance; drop the local latch so
          // the speaking-transition effect does not append a duplicate
          // audio-only row for the same speech.
          userSpokeRef.current = false;
          userSpokeAtRef.current = 0;
        }
        setUserTurns((turns) => {
          if (openId != null) {
            const idx = turns.findIndex((t) => t.id === openId);
            if (idx !== -1) {
              const next = turns.slice();
              next[idx] = { ...next[idx], text: userText, audioOnly: userText.length === 0 };
              return next;
            }
            // The open turn scrolled out of the capped window; nothing to
            // upgrade, so leave the list unchanged.
            return turns;
          }
          // No open turn (recognition outran the local speaking
          // transition): record a fresh turn so the words are not dropped.
          return [
            ...turns,
            { id: freshId, audioOnly: userText.length === 0, text: userText, at: freshAt },
          ].slice(-40);
        });
      } else if (message.type === "vision_caption") {
        const text = message.text || "";
        const ts = new Date().toTimeString().slice(0, 8);
        const frameId = message.frame_id || "";
        const pendingFrame = frameId ? pendingVisionFramesRef.current.get(frameId) : null;
        const frame = pendingFrame?.frame || lastFramePreviewRef.current;
        const meta = pendingFrame?.meta || lastFrameMetaRef.current;
        if (frameId) pendingVisionFramesRef.current.delete(frameId);
        const entryId = frameId || `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        setCurrentCaption(text);
        setCaptionEntries((entries) => [{ id: entryId, ts, text, frame, meta, frameId }, ...entries].slice(0, 14));
        const offsetMs = sessionStartedAtRef.current ? Math.max(0, performance.now() - sessionStartedAtRef.current) : 0;
        setSessionTimeline((items) => [
          {
            id: `${Date.now()}-vision`,
            ts,
            offsetMs,
            level: "info",
            kind: "vision",
            label: text || "Vision caption",
            frame,
            meta,
            frameId,
          },
          ...items,
        ].slice(0, 80));
        if (frameId && pendingDetailFrameRef.current === frameId) {
          // A detail re-request's richer caption returned; refresh the
          // open inspector in place without reopening it.
          pendingDetailFrameRef.current = null;
          setInspectFrame((current) =>
            current && current.frameId === frameId
              ? { ...current, text, detailPending: false }
              : current,
          );
        }
      } else if (message.type === "vision_status") {
        setVisionEnabledFromServer(!!message.enabled);
        if (!message.enabled) {
          // A server-driven disable can be an error auto-disable, the spend
          // ceiling, or a missing key. Stop local capture, but do not infer a
          // budget trip here: that flag is owned by the client-side cost
          // effect (and the server's own spend notice surfaces the budget
          // case), so error-disables are no longer mislabeled "budget hit".
          if (visionStreamRef.current) {
            stopVision();
          }
          addNotice("warn", "Vision disabled by server for this session");
        }
      } else if (message.type === "stat") {
        setGpuStat((stat) => ({
          vramUsed: Number.isFinite(message.vram_used) ? message.vram_used : stat.vramUsed,
          gpuUtil: Number.isFinite(message.gpu_util) ? message.gpu_util : stat.gpuUtil,
        }));
        // Gate on field presence: the stat envelope is shared, so each
        // consumer reads only the fields it knows.
        if (Number.isFinite(message.rtf)) setRtf(message.rtf);
      } else if (message.type === "pong") {
        const sentAt = typeof message.t === "number" ? message.t : null;
        if (sentAt !== null) {
          if (typeof message.seq === "number") pendingPingsRef.current.delete(message.seq);
          const rtt = performance.now() - sentAt;
          lastPongAtRef.current = performance.now();
          missedPongRef.current = 0;
          if (heartbeatWarnedRef.current) {
            heartbeatWarnedRef.current = false;
            addNotice("ok", "Connection responsive");
          }
          recordRttSample(rtt);
        }
      } else if (message.type === "request_vision_frame") {
        if (stateRef.current.visionOn && !stateRef.current.visionPaused) {
          captureFrame(false, false);
        }
      } else if (message.type === "vision_inject") {
        setVisionInjecting(!!message.active);
        addNotice(message.active ? "info" : "ok", message.active ? "Inject window opened, audio gated" : "Inject window closed", "inject");
      } else if (message.type === "interrupted") {
        pulseInterrupt();
      } else if (message.type === "event") {
        const text = message.text || message.kind || "Server event";
        if (message.kind === "recording") {
          const data = message.data || {};
          setServerRecording({
            active: !!data.active,
            ready: !!data.ready,
            url: typeof data.url === "string" ? data.url : null,
          });
        }
        const kindRaw = String(message.kind || "");
        const kind = kindRaw.includes("rewind") || kindRaw.includes("bookmark")
          ? "rewind"
          : kindRaw.includes("inject")
            ? "inject"
            : "event";
        lastServerEventRef.current = { text, at: performance.now() };
        addNotice(message.level || "info", text, kind);
      } else if (message.type === "notice") {
        const text = message.text || "Server notice";
        const recentEvent = lastServerEventRef.current;
        if (recentEvent.text !== text || performance.now() - recentEvent.at > 500) {
          addNotice("info", text);
        }
        toast(text);
      } else if (message.type === "error") {
        addNotice("err", message.reason || "Server error");
        toast(message.reason || "Server error");
        cleanup({ keepPhase: true });
        setPhase("idle");
      } else if (message.type === "end") {
        addNotice("info", "Server ended session");
        cleanup({ showDownload: true });
        setStageMessage("Session complete");
      }
    },
    [addNotice, attachAudioGraph, cleanup, pulseInterrupt, recordRttSample, startRecording, stopVision, toast],
  );

  const startCandidateStream = useCallback(
    (sessionId) => {
      const stream = new EventSource(`/api/rtc/candidates?session_id=${encodeURIComponent(sessionId)}`);
      candidateStreamRef.current = stream;
      stream.onmessage = (event) => {
        try {
          const candidate = JSON.parse(event.data);
          pcRef.current?.addIceCandidate(candidate).catch((error) => {
            console.warn("addIceCandidate failed:", error);
          });
        } catch (error) {
          console.warn("bad candidate JSON:", error);
        }
      };
      stream.addEventListener("done", () => {
        stream.close();
        if (candidateStreamRef.current === stream) candidateStreamRef.current = null;
      });
      stream.onerror = () => {
        stream.close();
        if (candidateStreamRef.current === stream) candidateStreamRef.current = null;
      };
      flushPendingCandidates();
    },
    [flushPendingCandidates],
  );

  const reconnect = useCallback(async () => {
    const pc = pcRef.current;
    const sessionId = sessionIdRef.current;
    if (phase !== "live" || !pc || !sessionId || reconnecting) return;
    if (reconnectGraceTimerRef.current) {
      clearTimeout(reconnectGraceTimerRef.current);
      reconnectGraceTimerRef.current = null;
    }
    setReconnecting(true);
    addNotice("warn", "ICE restart, renegotiating transport");
    try {
      pc.restartIce();
      const offer = await pc.createOffer({ iceRestart: true });
      await pc.setLocalDescription(offer);
      // The renegotiate POST rides the same network that just dropped, so
      // early attempts can fail while the outage is still in progress.
      // Retry a few times before declaring the session lost.
      let answer = null;
      for (let attempt = 1; attempt <= RENEGOTIATE_MAX_ATTEMPTS; attempt += 1) {
        try {
          answer = await postRenegotiate(sessionId, pc.localDescription);
          break;
        } catch (error) {
          if (attempt === RENEGOTIATE_MAX_ATTEMPTS) throw error;
          addNotice("warn", `Renegotiate failed, retrying (${attempt}/${RENEGOTIATE_MAX_ATTEMPTS})`);
          await new Promise((resolve) => {
            setTimeout(resolve, RENEGOTIATE_RETRY_DELAY_MS);
          });
          // The session may have been torn down while we waited (e.g. the
          // server's end message ran cleanup); stop retrying quietly.
          if (pcRef.current !== pc) return;
        }
      }
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
      // The session_id is unchanged, so the existing candidate stream keeps
      // serving fresh server candidates. Only re-open it if the blip closed
      // the EventSource (onerror/done null it out).
      if (!candidateStreamRef.current) startCandidateStream(sessionId);
      addNotice("ok", "Reconnected, session and model state preserved");
      toast("Reconnected");
    } catch (error) {
      // Terminal: the session is gone, but the local capture is not. Land
      // on the ended screen with the recording downloadable and say what
      // actually happened.
      addNotice("err", error.message || "ICE restart failed");
      addNotice("warn", "Connection lost, session ended; recording kept");
      toast("Connection lost");
      cleanup({ showDownload: true });
      setStageMessage("Connection lost");
    } finally {
      setReconnecting(false);
    }
  }, [phase, reconnecting, addNotice, cleanup, startCandidateStream, toast]);

  reconnectRef.current = reconnect;

  const startConversation = useCallback(async () => {
    if (phase === "connecting" || phase === "warmup" || phase === "live") return;
    cleanup({ keepPhase: true });
    setConnectionIssue(null);
    setSideExpanded(false);
    setPhase("connecting");
    setStageMessage("Requesting microphone");
    setTranscriptText("");
    transcriptLengthRef.current = 0;
    setAiTurns([]);
    aiTurnOpenRef.current = null;
    setUserTurns([]);
    userTurnOpenRef.current = null;
    userSpokeRef.current = false;
    userSpokeAtRef.current = 0;
    setNotices([]);
    setSessionTimeline([]);
    setServerRecording(null);
    setAssistantRate({ words: 0, seconds: 0, wpm: 0 });
    assistantTurnRef.current = { startedAt: 0, startLength: 0, lastChunkAt: 0, lastLength: 0, words: 0 };
    setCaptionEntries([]);
    setCurrentCaption("");
    pendingVisionFramesRef.current.clear();
    setVisionFramesSent(0);
    setVisionFramesGated(0);
    setVisionBudgetTripped(false);
    setBookmarks([]);
    setElapsedSec(0);
    sessionStartedAtRef.current = performance.now();
    addNotice("info", "Requesting microphone access");

    try {
      micStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: getMicConstraints(),
      });
      await refreshAudioOutputs();
      await initAudioContext();
      setStageMessage("Fetching TURN credentials");
      const iceServers = await fetchIceServers();
      addNotice("info", "Creating peer connection");
      const pc = new RTCPeerConnection({ iceServers, iceCandidatePoolSize: 1 });
      pcRef.current = pc;

      pc.ontrack = (event) => {
        aiStreamRef.current = event.streams?.[0] || new MediaStream([event.track]);
        if (aiAudioRef.current) {
          aiAudioRef.current.srcObject = aiStreamRef.current;
          aiAudioRef.current.play().catch((error) => {
            console.warn("AI audio autoplay blocked:", error);
          });
        }
        // Bias the freshly available receiver's jitter buffer to the current
        // preference. Best-effort: not every browser exposes a writable hint.
        const receiver = event.receiver;
        if (receiver && "playoutDelayHint" in receiver) {
          try {
            receiver.playoutDelayHint =
              stateRef.current.jitterBuffer === "smooth" ? JITTER_BUFFER_SMOOTH_SEC : 0;
          } catch {
            // Ignore browsers that reject the assignment.
          }
        }
        attachAudioGraph();
      };

      pc.onconnectionstatechange = () => {
        if (!pcRef.current) return;
        const state = pcRef.current.connectionState;
        const live = stateRef.current.phase === "live";
        if (state === "connected") {
          if (reconnectGraceTimerRef.current) {
            clearTimeout(reconnectGraceTimerRef.current);
            reconnectGraceTimerRef.current = null;
          }
          if (live) setStageMessage("Live");
        } else if (state === "failed") {
          if (live) {
            // Try an in-place restart first; reconnect's catch path runs the
            // terminal teardown if the restart itself fails.
            reconnectRef.current?.();
          } else {
            addNotice("err", "Connection failed");
            cleanup({ keepPhase: true });
            setPhase("idle");
            setStageMessage("Connection failed");
          }
        } else if (state === "disconnected") {
          setStageMessage("Reconnecting");
          if (live && !reconnectGraceTimerRef.current) {
            // Give ICE a grace window to self-recover before forcing a
            // restart, so a normal blip is not preempted needlessly.
            reconnectGraceTimerRef.current = window.setTimeout(() => {
              reconnectGraceTimerRef.current = null;
              if (pcRef.current && pcRef.current.connectionState !== "connected") {
                reconnectRef.current?.();
              }
            }, RECONNECT_GRACE_MS);
          }
        }
      };

      pc.oniceconnectionstatechange = () => {
        if (!pcRef.current) return;
        const state = pcRef.current.iceConnectionState;
        const live = stateRef.current.phase === "live";
        if (!live) {
          if (state === "checking") setStageMessage("Connecting peers");
          if (state === "connected" || state === "completed") setStageMessage("Opening control channel");
          if (state === "failed") {
            addNotice("err", "ICE failed, TURN may be unreachable");
            cleanup({ keepPhase: true });
            setPhase("idle");
          }
          return;
        }
        if (state === "failed") {
          // Attempt an ICE restart in place; the conversation and model
          // state are preserved on success, terminal teardown on failure.
          reconnectRef.current?.();
        }
      };

      const control = pc.createDataChannel("control");
      controlRef.current = control;
      control.onopen = () => {
        const payload = buildConfigPayload();
        control.send(JSON.stringify({ type: "config", ...payload }));
        sentConfigRef.current = payload;
        setPhase("warmup");
        setStageMessage("Loading model and warming audio");
        addNotice("info", "Config sent, waiting for server warmup");
      };
      control.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        try {
          handleControlMessage(JSON.parse(event.data));
        } catch (error) {
          console.warn("bad control JSON:", error);
        }
      };

      pc.onicecandidate = (event) => {
        if (sessionIdRef.current) postCandidate(event.candidate);
        else pendingCandidatesRef.current.push(event.candidate);
      };

      micStreamRef.current.getAudioTracks().forEach((track) => {
        pc.addTrack(track, micStreamRef.current);
      });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      setStageMessage("Negotiating session");
      const res = await fetch("/api/rtc/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      if (res.status === 409) {
        const error = new Error("Pod busy. Another client is already connected.");
        error.code = "session_busy";
        throw error;
      }
      if (!res.ok) {
        let detail = "";
        try {
          detail = (await res.json()).error || "";
        } catch {
          // Keep empty detail.
        }
        throw new Error(`Server returned ${res.status}${detail ? `: ${detail}` : ""}`);
      }

      const answer = await res.json();
      sessionIdRef.current = answer.session_id || null;
      await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
      if (sessionIdRef.current) startCandidateStream(sessionIdRef.current);
    } catch (error) {
      console.error("startConversation failed:", error);
      if (error.code === "session_busy") {
        setConnectionIssue("busy");
        addNotice("err", "Connect denied, session busy");
      } else if (error.code === "turn_unavailable") {
        setConnectionIssue("turn");
        addNotice("err", "TURN provisioning failed");
      } else if (error.name === "NotAllowedError") {
        addNotice("err", "Microphone access denied");
      } else {
        addNotice("err", error.message || "Failed to start conversation");
      }
      toast(error.message || "Failed to start conversation");
      cleanup({ keepPhase: true });
      setPhase("idle");
      setStageMessage("Standby");
    }
  }, [
    addNotice,
    attachAudioGraph,
    buildConfigPayload,
    cleanup,
    getMicConstraints,
    handleControlMessage,
    initAudioContext,
    phase,
    postCandidate,
    refreshAudioOutputs,
    startCandidateStream,
    toast,
  ]);

  const clearConnectHold = useCallback(() => {
    if (connectHoldTimerRef.current) clearTimeout(connectHoldTimerRef.current);
    if (connectHoldTickRef.current) clearInterval(connectHoldTickRef.current);
    connectHoldTimerRef.current = null;
    connectHoldTickRef.current = null;
    setConnectHoldPct(0);
  }, []);

  const beginConnectHold = useCallback((event) => {
    if (phase !== "idle" && phase !== "ended") return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    clearConnectHold();
    const startedAt = performance.now();
    connectHoldTickRef.current = window.setInterval(() => {
      setConnectHoldPct(Math.min(100, ((performance.now() - startedAt) / 700) * 100));
    }, 30);
    connectHoldTimerRef.current = window.setTimeout(() => {
      clearConnectHold();
      startConversation();
    }, 700);
  }, [clearConnectHold, phase, startConversation]);

  const keyConnect = useCallback((event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    clearConnectHold();
    startConversation();
  }, [clearConnectHold, startConversation]);

  const stopConversation = useCallback(() => {
    addNotice("info", "Session ended, recording available");
    cleanup({ showDownload: true });
    setPhase("ended");
    setStageMessage("Session complete");
  }, [addNotice, cleanup]);

  const newConversation = () => {
    cleanup();
    setTranscriptText("");
    transcriptLengthRef.current = 0;
    setAiTurns([]);
    aiTurnOpenRef.current = null;
    setUserTurns([]);
    userTurnOpenRef.current = null;
    userSpokeRef.current = false;
    userSpokeAtRef.current = 0;
    setCaptionEntries([]);
    setCurrentCaption("");
    pendingVisionFramesRef.current.clear();
    setNotices([]);
    setSessionTimeline([]);
    setAssistantRate({ words: 0, seconds: 0, wpm: 0 });
    assistantTurnRef.current = { startedAt: 0, startLength: 0, lastChunkAt: 0, lastLength: 0, words: 0 };
    sessionStartedAtRef.current = 0;
    setVisionBudgetTripped(false);
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
    recordingUrlRef.current = null;
    setRecordingUrl(null);
    setServerRecording(null);
    setBookmarks([]);
    setElapsedSec(0);
    setPhase("idle");
    setStageMessage("Standby");
  };

  const sendVisionFrame = useCallback((dataUrl, meta, detail = false) => {
    if (!dataUrl || !controlRef.current || controlRef.current.readyState !== "open") return null;
    const base64 = dataUrl.split(",")[1] || "";
    if (!base64) return null;
    const frameId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${visionFrameSeqRef.current++}`;
    const nextMeta = {
      ...meta,
      detail: !!detail,
      bytes: meta?.bytes || Math.round((base64.length * 3) / 4),
      sent_at: new Date().toISOString(),
    };
    lastFramePreviewRef.current = dataUrl;
    lastFrameMetaRef.current = nextMeta;
    pendingVisionFramesRef.current.set(frameId, { frame: dataUrl, meta: nextMeta });
    if (pendingVisionFramesRef.current.size > 20) {
      const oldest = pendingVisionFramesRef.current.keys().next().value;
      pendingVisionFramesRef.current.delete(oldest);
    }
    controlRef.current.send(
      JSON.stringify({ type: "vision_frame", frame_id: frameId, data: base64, detail: !!detail }),
    );
    setVisionFramesSent((count) => count + 1);
    const now = performance.now();
    visionLastSentAtRef.current = now;
    setVisionLastSentAt(now);
    setVisionClockMs(now);
    return frameId;
  }, []);

  const captureFrame = useCallback(
    async (detail = false, force = false) => {
      if (!visionStreamRef.current || !visionVideoRef.current) return false;
      if (controlRef.current?.readyState !== "open") return false;
      const video = visionVideoRef.current;
      if (!video.videoWidth || !video.videoHeight) return false;
      const divisor = detail ? 1 : 2;
      const quality = detail ? 0.8 : 0.55;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(160, Math.floor(video.videoWidth / divisor));
      canvas.height = Math.max(90, Math.floor(video.videoHeight / divisor));
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      if (!detail && !force) {
        const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
        if (
          visionLastFrameDataRef.current &&
          visionLastFrameDataRef.current.length === frame.data.length
        ) {
          let diff = 0;
          // Sample luma across R, G, and B per stride. Stepping by 16 (a
          // multiple of 4) used to land only on the red channel, so motion
          // that moved green/blue while holding red could fall under the
          // threshold and be gated out.
          for (let i = 0; i + 2 < frame.data.length; i += 16) {
            diff +=
              Math.abs(frame.data[i] - visionLastFrameDataRef.current[i]) +
              Math.abs(frame.data[i + 1] - visionLastFrameDataRef.current[i + 1]) +
              Math.abs(frame.data[i + 2] - visionLastFrameDataRef.current[i + 2]);
          }
          const meanDelta = diff / (frame.data.length / 16) / 3 / 255;
          if (meanDelta < VISION_MOTION_THRESHOLD) {
            setVisionFramesGated((count) => count + 1);
            return false;
          }
        }
        visionLastFrameDataRef.current = new Uint8ClampedArray(frame.data);
      }

      const dataUrl = canvas.toDataURL("image/jpeg", quality);
      const base64 = dataUrl.split(",")[1] || "";
      const meta = {
        width: canvas.width,
        height: canvas.height,
        bytes: Math.round((base64.length * 3) / 4),
        detail: !!detail,
        quality,
        source: detail ? "detail" : force ? "forced" : "motion",
      };
      return sendVisionFrame(dataUrl, meta, detail);
    },
    [sendVisionFrame],
  );

  const startVisionSource = useCallback(async (source) => {
    if (!isLive) return;
    if (!visionEnabledFromServer) {
      addNotice("warn", "Vision is unavailable for this session");
      toast("Vision unavailable");
      return;
    }
    setVisionSourceOpen(false);
    try {
      const useCamera = source === "camera";
      const stream = useCamera
        ? await navigator.mediaDevices.getUserMedia({ video: true })
        : await navigator.mediaDevices.getDisplayMedia({ video: true });
      visionStreamRef.current = stream;
      // The browser can end the tracks outside the app UI (the floating
      // "Stop sharing" bar, a camera unplug); tear vision down and say so
      // instead of keeping a frozen "Cam · Live" badge capturing stale
      // frames.
      stream.getTracks().forEach((track) => {
        track.addEventListener("ended", () => {
          if (visionStreamRef.current !== stream) return;
          stopVision();
          addNotice("warn", "Vision source ended by the browser or device", "vision");
        });
      });
      setVisionOn(true);
      setVisionPaused(false);
      setVisionFramesSent(0);
      setVisionFramesGated(0);
      setVisionBudgetTripped(false);
      setCaptionEntries([]);
      addNotice("info", useCamera ? "Vision camera started" : "Vision screen share started", "vision");
      visionStatusTickRef.current = setInterval(() => {
        setVisionClockMs(performance.now());
      }, 1000);
    } catch (error) {
      addNotice("err", `Could not start vision: ${error.message || error}`);
    }
  }, [
    addNotice,
    isLive,
    stopVision,
    toast,
    visionEnabledFromServer,
  ]);

  const startVision = useCallback(() => {
    if (!isLive) return;
    if (!visionEnabledFromServer) {
      addNotice("warn", "Vision is unavailable for this session");
      toast("Vision unavailable");
      return;
    }
    if (visionStreamRef.current) {
      stopVision();
      addNotice("info", "Vision stopped", "vision");
      return;
    }
    setVisionSourceOpen(true);
  }, [addNotice, isLive, stopVision, toast, visionEnabledFromServer]);

  useEffect(() => {
    if (!visionOn || !visionCostLimitActive || visionBudgetTripped) return;
    if (visionCostUsd < Number(visionCostLimitUsd)) return;
    setVisionBudgetTripped(true);
    stopVision();
    addNotice("warn", `Vision cost ceiling reached at $${visionCostUsd.toFixed(4)}`, "budget");
  }, [addNotice, stopVision, visionBudgetTripped, visionCostLimitActive, visionCostLimitUsd, visionCostUsd, visionOn]);

  useEffect(() => {
    if (visionOn && visionVideoRef.current && visionStreamRef.current) {
      visionVideoRef.current.srcObject = visionStreamRef.current;
      visionVideoRef.current.play().catch(() => {});
    }
  }, [visionOn]);

  useEffect(() => {
    if (!visionOn || !visionStreamRef.current) return undefined;
    if (visionIntervalRef.current) clearInterval(visionIntervalRef.current);
    // Clamp at point of use: a stale or corrupt stored value (0, negative,
    // NaN) would otherwise coerce setInterval to the browser minimum and fire
    // captures (real Gemini calls) many times a second.
    const periodMs = Math.min(
      30000,
      Math.max(1000, Number(visionIntervalMs) || DEFAULTS.visionIntervalMs),
    );
    const intervalId = setInterval(() => {
      if (stateRef.current.visionPaused) return;
      // Fallback only: server-requested and forced captures cover an
      // active scene, so skip the tick when a frame already went out
      // within the current period.
      if (performance.now() - visionLastSentAtRef.current < periodMs) return;
      captureFrame(false, false);
    }, periodMs);
    visionIntervalRef.current = intervalId;
    return () => {
      clearInterval(intervalId);
      if (visionIntervalRef.current === intervalId) visionIntervalRef.current = null;
    };
  }, [captureFrame, visionIntervalMs, visionOn]);

  const forceCapture = () => {
    if (!visionOn) return;
    captureFrame(true, true);
    addNotice("info", "Detail frame captured, bypassed motion gate", "vision");
  };

  const closeInspectFrame = () => {
    pendingDetailFrameRef.current = null;
    setInspectFrame(null);
  };

  const requestFrameDetail = (entry) => {
    if (!entry?.frame) {
      forceCapture();
      closeInspectFrame();
      return;
    }
    const meta = {
      ...(entry.meta || {}),
      source: "history-detail",
      detail: true,
    };
    const detailFrameId = sendVisionFrame(entry.frame, meta, true);
    if (!detailFrameId) return;
    // The re-send mints a fresh frame id; the richer caption reconciles
    // by that id, so the inspector must track the re-send's id (not the
    // original) to update in place when the reply lands.
    pendingDetailFrameRef.current = detailFrameId;
    setInspectFrame((current) =>
      current ? { ...current, meta, frameId: detailFrameId, detailPending: true } : current,
    );
    addNotice("info", "Re-requested detail frame", "vision", {
      frame: entry.frame,
      meta,
      frameId: detailFrameId,
    });
  };

  const activateTimelinePoint = useCallback((item) => {
    if (item?.kind === "vision" && item.frame) {
      setInspectFrame({
        ts: item.ts,
        text: item.label,
        frame: item.frame,
        meta: item.meta,
        frameId: item.frameId,
      });
      return;
    }
    const playback = recordingPlaybackRef.current;
    if (!playback || !recordingUrl) return;
    playback.currentTime = Math.max(0, (item.offsetMs || 0) / 1000);
    playback.play?.().catch(() => {});
  }, [recordingUrl]);

  const toggleVisionPause = () => {
    setVisionPaused((paused) => {
      addNotice("info", paused ? "Vision resumed" : "Vision paused", "vision");
      return !paused;
    });
  };

  const rewind = () => {
    const now = performance.now();
    if (now - lastRewindClickRef.current < 1000) return;
    lastRewindClickRef.current = now;
    if (controlRef.current?.readyState === "open") {
      controlRef.current.send(JSON.stringify({ type: "rewind" }));
      addNotice("info", "Rewind requested", "rewind");
    }
  };

  const addBookmark = () => {
    if (phase !== "live") return;
    const now = performance.now();
    if (now - lastBookmarkClickRef.current < 1000) return;
    lastBookmarkClickRef.current = now;
    if (controlRef.current?.readyState !== "open") return;
    const id = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const label = `Mark ${bookmarks.length + 1}`;
    const atSec = elapsedSec;
    setBookmarks((prev) => [{ id, label, atSec }, ...prev].slice(0, 6));
    controlRef.current.send(JSON.stringify({ type: "bookmark", id, label, at_sec: atSec }));
    addNotice("ok", `Bookmarked snapshot at ${formatOffset(atSec * 1000)}`, "rewind");
    toast("Snapshot bookmarked");
  };

  const jumpBookmark = (bm) => {
    if (phase !== "live" || !bm) return;
    const now = performance.now();
    if (now - lastRewindClickRef.current < 1000) return;
    lastRewindClickRef.current = now;
    if (controlRef.current?.readyState !== "open") return;
    controlRef.current.send(JSON.stringify({ type: "rewind", id: bm.id }));
    addNotice("warn", `Restored snapshot · ${bm.label}`, "rewind");
    toast(`Jumped to ${bm.label}`);
  };

  const sendLiveConfig = useCallback((partial) => {
    if (!isLive) return;
    // Coalesce the in-flight fields and flush on the trailing edge so a
    // fast slider drag sends one update per tick instead of flooding the
    // control channel; the final value still lands on release.
    Object.assign(liveConfigPendingRef.current, partial);
    if (liveConfigTimerRef.current) return;
    liveConfigTimerRef.current = setTimeout(() => {
      liveConfigTimerRef.current = null;
      const fields = liveConfigPendingRef.current;
      liveConfigPendingRef.current = {};
      if (controlRef.current?.readyState === "open" && Object.keys(fields).length) {
        controlRef.current.send(JSON.stringify({ type: "update_config", ...fields }));
      }
    }, 150);
  }, [isLive]);

  const interruptResponse = useCallback(
    (reason = "manual") => {
      const now = performance.now();
      if (now - lastInterruptClickRef.current < 900) return;
      lastInterruptClickRef.current = now;
      if (controlRef.current?.readyState === "open") {
        controlRef.current.send(JSON.stringify({ type: "interrupt", reason }));
        pulseInterrupt();
        addNotice(
          reason === "barge_in" ? "warn" : "info",
          reason === "barge_in" ? "Barge-in sent, stopping assistant audio" : "Stop response requested",
          "interrupt",
        );
      }
    },
    [addNotice, pulseInterrupt],
  );

  const inspectVoiceClip = useCallback((file, url) => new Promise((resolve) => {
    const audio = new Audio();
    let settled = false;
    let timer = null;
    const done = (duration = 0) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      audio.removeAttribute("src");
      resolve({
        name: file.name,
        size: file.size,
        type: file.type || "audio",
        duration,
      });
    };
    timer = window.setTimeout(() => done(0), 1500);
    audio.onloadedmetadata = () => done(Number.isFinite(audio.duration) ? audio.duration : 0);
    audio.onerror = () => done(0);
    audio.src = url;
  }), []);

  const previewUploadedVoice = useCallback(() => {
    if (!uploadedVoicePreviewUrl) return;
    voicePreviewAudioRef.current?.pause?.();
    const audio = new Audio(uploadedVoicePreviewUrl);
    voicePreviewAudioRef.current = audio;
    audio.play().catch((error) => {
      addNotice("err", `Preview failed: ${error.message || error}`);
    });
  }, [addNotice, uploadedVoicePreviewUrl]);

  // Synthesize and play a short sample of a preset voice. The server holds
  // the single GPU, so a preview is only honored when no session is live; it
  // returns 409 otherwise. Holds at most one preview at a time: a new press
  // stops and supersedes any in-flight one.
  const previewVoice = useCallback(
    async (id) => {
      if (!id) return;
      // Mirror the server's reject-while-live policy in the UI. The sidebar
      // is locked during a session anyway; this is the fast local path.
      if (isLive) {
        addNotice("warn", "Voice preview unavailable during a live session");
        return;
      }
      // Pressing the stop glyph (same voice already previewing) stops it.
      const alreadyPreviewing = voicePreviewAudioRef.current && previewing === id;
      // Supersede any in-flight preview: stop its audio and free its blob.
      voicePreviewAudioRef.current?.pause?.();
      voicePreviewAudioRef.current = null;
      if (voicePreviewObjectUrlRef.current) {
        URL.revokeObjectURL(voicePreviewObjectUrlRef.current);
        voicePreviewObjectUrlRef.current = "";
      }
      if (alreadyPreviewing) {
        setPreviewing(null);
        return;
      }
      setPreviewing(id);
      try {
        const res = await fetch("/api/voice-preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ voice: id }),
        });
        if (res.status === 409) {
          addNotice("warn", "Voice preview unavailable during a live session");
          setPreviewing((current) => (current === id ? null : current));
          return;
        }
        if (!res.ok) {
          const json = await res.json().catch(() => null);
          throw new Error(json?.error || `preview failed (${res.status})`);
        }
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        voicePreviewObjectUrlRef.current = objectUrl;
        const audio = new Audio(objectUrl);
        voicePreviewAudioRef.current = audio;
        const clear = () => {
          if (voicePreviewObjectUrlRef.current === objectUrl) {
            URL.revokeObjectURL(objectUrl);
            voicePreviewObjectUrlRef.current = "";
          }
          setPreviewing((current) => (current === id ? null : current));
        };
        audio.onended = clear;
        await audio.play();
      } catch (error) {
        if (voicePreviewObjectUrlRef.current) {
          URL.revokeObjectURL(voicePreviewObjectUrlRef.current);
          voicePreviewObjectUrlRef.current = "";
        }
        addNotice("err", `Preview failed: ${error.message || error}`);
        setPreviewing((current) => (current === id ? null : current));
      }
    },
    [addNotice, isLive, previewing],
  );

  const uploadVoice = async (file) => {
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) {
      setUploadStatus("File too large. Max 20 MB.");
      setUploadKind("error");
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    const meta = await inspectVoiceClip(file, previewUrl);
    if (meta.duration > 60) {
      URL.revokeObjectURL(previewUrl);
      setUploadStatus(`Clip too long (${fmt(meta.duration, 1)} s). Max 60 s.`);
      setUploadKind("error");
      return;
    }
    setUploadStatus(`Uploading ${file.name}`);
    setUploadKind("uploading");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/voice-upload", { method: "POST", body: form });
      const json = await res.json().catch(() => null);
      if (!res.ok) throw new Error(json?.error || `upload failed (${res.status})`);
      if (!json?.filename) throw new Error("server returned no filename");
      voicePreviewAudioRef.current?.pause?.();
      voicePreviewAudioRef.current = null;
      if (uploadedVoicePreviewUrlRef.current) URL.revokeObjectURL(uploadedVoicePreviewUrlRef.current);
      uploadedVoicePreviewUrlRef.current = previewUrl;
      setSessionProfileId("custom");
      setUploadedVoiceFilename(json.filename);
      setUploadedVoiceLabel(file.name);
      setUploadedVoiceMeta(meta);
      setUploadedVoicePreviewUrl(previewUrl);
      const detail = `${meta.duration ? `${fmt(meta.duration, 1)} s · ` : ""}${fmt(meta.size / (1024 * 1024), 1)} MB`;
      setUploadStatus(`Using uploaded voice: ${file.name} (${detail})`);
      setUploadKind("success");
      addNotice("ok", "Voice reference uploaded");
    } catch (error) {
      URL.revokeObjectURL(previewUrl);
      clearUploadedVoice();
      setUploadStatus(`Upload failed: ${error.message || error}`);
      setUploadKind("error");
    }
  };

  const runPreflight = async () => {
    setPreflightOpen(true);
    setPreflightDone(false);
    setPreflight({ mic: "checking", out: "idle", turn: "idle" });
    let failed = false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: getMicConstraints() });
      stream.getTracks().forEach((track) => {
        track.stop();
      });
      await refreshAudioOutputs();
      setPreflight((state) => ({ ...state, mic: "ok", out: "checking" }));
    } catch {
      failed = true;
      setPreflight((state) => ({ ...state, mic: "fail", out: "checking" }));
    }
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      gain.gain.value = 0.04;
      osc.frequency.value = 440;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      await new Promise((resolve) => setTimeout(resolve, 180));
      osc.stop();
      await ctx.close();
      setPreflight((state) => ({ ...state, out: "ok", turn: "checking" }));
    } catch {
      failed = true;
      setPreflight((state) => ({ ...state, out: "fail", turn: "checking" }));
    }
    try {
      await fetchIceServers();
      setPreflight((state) => ({ ...state, turn: "ok" }));
    } catch {
      failed = true;
      setPreflight((state) => ({ ...state, turn: "fail" }));
    }
    setPreflightDone(true);
    addNotice(failed ? "warn" : "ok", failed ? "Pre-flight found an issue" : "Pre-flight checks passed");
  };

  useEffect(() => {
    if (phase !== "live") return undefined;
    const id = setInterval(() => setElapsedSec((value) => value + 1), 1000);
    return () => clearInterval(id);
  }, [phase]);

  useEffect(() => {
    if (phase !== "live") {
      setLevels({ mic: 0, ai: 0 });
      setSpeaking(null);
      return undefined;
    }
    let overlapTicks = 0;
    const id = setInterval(() => {
      const mic = rmsFromAnalyser(micAnalyserRef.current);
      const ai = visionInjecting ? 0 : rmsFromAnalyser(aiAnalyserRef.current);
      const micBars = Math.min(10, Math.round(mic * 10));
      const aiBars = Math.min(10, Math.round(ai * 10));
      setLevels({ mic: micBars, ai: aiBars });
      if (micBars > 2 && aiBars > 2) {
        overlapTicks += 1;
        setSpeaking("both");
        if (overlapTicks === 3 && !bargeActiveRef.current) {
          bargeActiveRef.current = true;
          interruptResponse("barge_in");
        }
      } else {
        overlapTicks = 0;
        bargeActiveRef.current = false;
        if (aiBars > 2) setSpeaking("ai");
        else if (micBars > 2) setSpeaking("you");
        else setSpeaking(null);
      }
    }, 100);
    return () => clearInterval(id);
  }, [interruptResponse, phase, visionInjecting]);

  // Record a user turn from the local speaking transition: the mic channel
  // registered speech, then the assistant resumed. This is the only honest
  // user-side signal without recognition, so the turn starts as audio-only
  // and never carries fabricated words; the optional server recognizer
  // upgrades it later. Reset the latch when leaving the live phase.
  useEffect(() => {
    if (phase !== "live") {
      userSpokeRef.current = false;
      userSpokeAtRef.current = 0;
      userTurnOpenRef.current = null;
      return;
    }
    if (speaking === "you" || speaking === "both") {
      if (!userSpokeRef.current) userSpokeAtRef.current = performance.now();
      userSpokeRef.current = true;
    } else if (speaking === "ai" && userSpokeRef.current) {
      // Assistant resumed after the user spoke: close the user turn,
      // stamped with the time the speech started.
      userSpokeRef.current = false;
      const at = userSpokeAtRef.current || performance.now();
      userSpokeAtRef.current = 0;
      const id = `${Date.now()}-you-${Math.random().toString(36).slice(2, 7)}`;
      userTurnOpenRef.current = id;
      setUserTurns((turns) => [...turns, { id, audioOnly: true, text: "", at }].slice(-40));
    }
  }, [speaking, phase]);

  useEffect(() => {
    if (phase !== "live") {
      setLatencyMs(0);
      setTailLatencyMs(0);
      setRttSamples([]);
      setNetStats({ quality: 0, jitterMs: 0, lossPct: 0, candidate: "" });
      return undefined;
    }
    // Loss is reported over the interval between polls rather than
    // cumulatively, so an early burst does not pin the readout for the
    // whole session. These hold the previous poll's running counters.
    let prevReceived = 0;
    let prevLost = 0;
    const id = setInterval(async () => {
      const pc = pcRef.current;
      if (!pc) return;
      try {
        const stats = await pc.getStats();
        let rtt = 0;
        let jitterMs = 0;
        let received = 0;
        let lost = 0;
        let selectedLocalId = "";
        const localCandidates = new Map();
        stats.forEach((report) => {
          if (report.type === "inbound-rtp" && report.kind === "audio") {
            if (typeof report.jitter === "number") jitterMs = report.jitter * 1000;
            if (typeof report.packetsReceived === "number") received = report.packetsReceived;
            if (typeof report.packetsLost === "number") lost = report.packetsLost;
          } else if (
            report.type === "candidate-pair" &&
            (report.nominated || report.selected)
          ) {
            if (typeof report.currentRoundTripTime === "number") {
              rtt = Math.round(report.currentRoundTripTime * 1000);
            }
            selectedLocalId = report.localCandidateId || selectedLocalId;
          } else if (report.type === "local-candidate") {
            localCandidates.set(report.id, report);
          }
        });
        const local = selectedLocalId ? localCandidates.get(selectedLocalId) : null;
        let candidate = "";
        if (local) {
          const type =
            local.candidateType === "relay"
              ? "TURN"
              : local.candidateType === "srflx"
                ? "STUN"
                : local.candidateType === "host"
                  ? "HOST"
                  : (local.candidateType || "").toUpperCase();
          const proto = (local.relayProtocol || local.protocol || "").toUpperCase();
          candidate = proto ? `${type} · ${proto}` : type;
        }
        const deltaReceived = Math.max(0, received - prevReceived);
        const deltaLost = Math.max(0, lost - prevLost);
        prevReceived = received;
        prevLost = lost;
        const denom = deltaReceived + deltaLost;
        const lossPct = denom > 0 ? (deltaLost / denom) * 100 : 0;
        // Quality penalizes loss heavily and jitter mildly; clamped 0-100.
        const quality = Math.max(0, Math.min(100, Math.round(100 - lossPct * 8 - jitterMs * 0.8)));
        setNetStats({
          quality,
          jitterMs: Math.round(jitterMs),
          lossPct: Math.round(lossPct * 10) / 10,
          candidate,
        });
        // App-level pongs drive the RTT readout when fresh; the transport
        // round-trip only fills the gap while measured RTT is stale.
        if (performance.now() - lastPongAtRef.current >= HEARTBEAT_STALE_AFTER_MS) {
          recordRttSample(rtt);
        }
      } catch {
        // Stats are best-effort; no UI error needed.
      }
    }, 1000);
    return () => clearInterval(id);
  }, [phase, recordRttSample]);

  useEffect(() => {
    const pc = pcRef.current;
    if (phase !== "live" || !pc?.getReceivers) return;
    const hint = jitterBuffer === "smooth" ? JITTER_BUFFER_SMOOTH_SEC : 0;
    pc.getReceivers().forEach((receiver) => {
      if (receiver.track?.kind === "audio" && "playoutDelayHint" in receiver) {
        try {
          receiver.playoutDelayHint = hint;
        } catch {
          // Not all browsers expose a writable hint; ignore.
        }
      }
    });
  }, [jitterBuffer, phase]);

  useEffect(() => {
    if (phase !== "live") {
      if (heartbeatTimerRef.current) clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
      pendingPingsRef.current.clear();
      pingSeqRef.current = 0;
      lastPongAtRef.current = 0;
      missedPongRef.current = 0;
      heartbeatWarnedRef.current = false;
      return undefined;
    }
    const id = setInterval(() => {
      const control = controlRef.current;
      if (control?.readyState !== "open") return;
      const pending = pendingPingsRef.current;
      // A ping still in the map at the next tick never got a pong.
      const now = performance.now();
      for (const [seq, sentAt] of pending) {
        if (now - sentAt < HEARTBEAT_INTERVAL_MS) continue;
        pending.delete(seq);
        missedPongRef.current += 1;
      }
      if (missedPongRef.current >= HEARTBEAT_MISSED_LIMIT && !heartbeatWarnedRef.current) {
        heartbeatWarnedRef.current = true;
        addNotice("warn", "Connection unresponsive");
      }
      const seq = pingSeqRef.current++;
      const t = performance.now();
      pending.set(seq, t);
      if (pending.size > HEARTBEAT_MAX_PENDING) {
        const oldest = pending.keys().next().value;
        pending.delete(oldest);
      }
      try {
        control.send(JSON.stringify({ type: "ping", t, seq }));
      } catch {
        // Send is best-effort; a closed channel is handled by teardown paths.
      }
    }, HEARTBEAT_INTERVAL_MS);
    heartbeatTimerRef.current = id;
    return () => {
      clearInterval(id);
      if (heartbeatTimerRef.current === id) heartbeatTimerRef.current = null;
    };
  }, [addNotice, phase]);

  useEffect(() => () => cleanup(), [cleanup]);

  const filteredVoices = voiceList.filter((item) => {
    if (voiceGender !== "all" && item[3] !== voiceGender) return false;
    return true;
  });

  const elapsedStr = useMemo(() => {
    const minutes = Math.floor(elapsedSec / 60);
    const seconds = elapsedSec % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }, [elapsedSec]);

  const phaseIdx = { idle: 0, connecting: 1, warmup: 2, live: 3, ended: 4 }[phase] ?? 0;
  const phaseProgress = { idle: 0, connecting: 25, warmup: 55, live: 82, ended: 100 }[phase] ?? 0;
  // Interleave assistant segments and user turns into one chronological list
  // so the transcript reads as a back-and-forth instead of one AI blob.
  const transcriptTurns = [
    ...aiTurns
      .filter((turn) => turn.text.trim())
      .map((turn) => ({ id: turn.id, role: "ai", at: turn.at || 0, text: turn.text })),
    ...userTurns.map((turn) => ({
      id: turn.id,
      role: "you",
      at: turn.at || 0,
      audioOnly: turn.audioOnly,
      text: turn.text,
    })),
  ].sort((a, b) => a.at - b.at);
  const blendActive = voiceBlend && !uploadedVoiceFilename && voiceB && voiceB !== voice && blendMix > 0;
  const voiceDisplay = uploadedVoiceFilename
    ? uploadedVoiceLabel || "uploaded"
    : blendActive
      ? `${voice}+${voiceB}`
      : voice;
  const visionAge = visionLastSentAt
    ? Math.max(0, Math.round(((visionClockMs || performance.now()) - visionLastSentAt) / 1000))
    : null;

  const gpuLabel = serverInfo.gpuName ? `GPU · ${serverInfo.gpuName}` : "GPU";
  const gpuValue = (() => {
    if (!isLive) return "idle";
    const parts = [];
    if (gpuStat.vramUsed > 0) {
      parts.push(serverInfo.vramTotal > 0 ? `${fmtGb(gpuStat.vramUsed)}/${fmtGb(serverInfo.vramTotal)} GB` : `${fmtGb(gpuStat.vramUsed)} GB`);
    }
    if (Number.isFinite(gpuStat.gpuUtil)) parts.push(`${gpuStat.gpuUtil}% util`);
    return parts.length ? parts.join(" · ") : "live";
  })();

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <svg viewBox="0 0 12 12" aria-hidden="true" focusable="false">
              <rect className="b1" x="0" y="3.5" width="1.6" height="5" rx="0.4" />
              <rect className="b1" x="2.4" y="1" width="1.6" height="10" rx="0.4" />
              <rect className="b2" x="4.8" y="4" width="1.6" height="4" rx="0.4" />
              <rect className="b1" x="7.2" y="2" width="1.6" height="8" rx="0.4" />
              <rect className="b2" x="9.6" y="5" width="1.6" height="2" rx="0.4" />
            </svg>
          </div>
          <div>
            <div className="brand-name">
              PersonaPlex<span>Studio</span>
            </div>
            <div className="brand-tag">runpod</div>
          </div>
        </div>
        <div className="phaseline" style={{ "--progress": `${phaseProgress}%` }}>
          {["Ready", "Connect", "Warmup", "Live", "Complete"].map((label, index) => (
            <span key={label} className={cls("phase", index < phaseIdx && "done", index === phaseIdx && "active")}>
              <span className="dot" />
              {label}
            </span>
          ))}
        </div>
        <div className="pills">
          <div className="pill">
            <span className="l">GPU</span>
            <span className="v">{serverInfo.gpuName || "·"}</span>
          </div>
          <div className="pill">
            <span className="l">ICE</span>
            <span className={cls("v", isLive && !reconnecting && "live", reconnecting && "warn")}>
              {reconnecting ? "···" : isLive ? "TURN" : "·"}
            </span>
          </div>
          <div className="pill">
            <span className="l">RTT</span>
            <span className="v">
              {latencyMs || "·"}
              <span style={{ color: "var(--ink-4)", marginLeft: 2 }}>ms</span>
            </span>
          </div>
        </div>
      </header>

      <div className={cls("body", sideCollapsed && "side-collapsed")}>
        <aside className="side" aria-label="Persona and voice settings">
          {sideCollapsed && (
            <button
              type="button"
              className="side-rail"
              aria-label="Show configuration (locked during session)"
              onClick={() => setSideExpanded(true)}
            >
              <span className="side-rail-x" aria-hidden="true">›</span>
              <span className="side-rail-label">Configuration</span>
              <span className="side-rail-lock" aria-hidden="true">{Icon.lock}</span>
            </button>
          )}
          <div className={cls("side-scroll", cfgLocked && "cfg-locked")}>
            {cfgLocked && (
              <div className="cfg-lock-note" role="status">
                <span className="cfg-lock-glyph" aria-hidden="true">{Icon.lock}</span>
                <span className="lk">Locked</span>
                <span>Settings are fixed for this session. Changes apply on the next connect.</span>
                <button
                  type="button"
                  className="cfg-collapse"
                  aria-label="Collapse configuration"
                  onClick={() => setSideExpanded(false)}
                >
                  ‹
                </button>
              </div>
            )}
            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">01 · PERSONA</div>
                  <div className="sect-title" style={{ display: "inline-flex", alignItems: "center" }}>
                    System prompt
                    <Info k="systemPrompt" />
                  </div>
                </div>
                <span className="sect-sub">{presetId === "custom" ? "custom" : "preset"}</span>
              </div>
              <div className="session-profile">
                <Listbox
                  label="Session profile"
                  caption="Session profile"
                  info="profile"
                  value={sessionProfileId}
                  options={[
                    ...allSessionProfiles.map((profile) => ({
                      value: profile.id,
                      label: profile.label,
                      desc: profile.custom ? profile.desc || "Saved card" : profile.desc,
                    })),
                    ...(sessionProfileId === "custom"
                      ? [{ value: "custom", label: "Custom", desc: "Current settings." }]
                      : []),
                  ]}
                  onChange={(value) => {
                    if (value !== "custom") applySessionProfile(value);
                  }}
                />
                <div className="profile-tools">
                  <input
                    type="text"
                    aria-label="Profile name"
                    value={profileName}
                    maxLength={48}
                    onChange={(event) => setProfileName(event.target.value)}
                  />
                  <div className="profile-actions">
                    <button className="btn ghost" type="button" onClick={saveCustomProfile}>Save</button>
                    <button className="btn ghost" type="button" disabled={!selectedCustomProfile} onClick={updateCustomProfile}>Update</button>
                    <button className="btn ghost" type="button" disabled={!selectedCustomProfile} onClick={deleteCustomProfile}>Delete</button>
                  </div>
                  <div className="profile-library-actions">
                    <button className="btn ghost" type="button" onClick={duplicateCurrentProfile}>Duplicate</button>
                    <button className="btn ghost" type="button" disabled={!customProfiles.length} onClick={exportProfileLibrary}>Export</button>
                    <button className="btn ghost" type="button" onClick={() => profileLibraryFileRef.current?.click()}>Import</button>
                  </div>
                  <div className="profile-compare">
                    <div className="profile-compare-copy">
                      <span className="k">PINNED BASELINE</span>
                      <span className="v">
                        {pinnedTuning
                          ? `${pinnedTuning.label || "Baseline"} · ${tuningDiffs.length ? `${tuningDiffs.length} changed` : "matched"}`
                          : "none"}
                      </span>
                    </div>
                    {/* biome-ignore lint/a11y/useAriaPropsSupportedByRole: aria-label names this readout region for assistive tech; visible per-row labels are also present */}
                    <div className="profile-diffs" aria-label="Pinned baseline differences">
                      {pinnedTuning ? (
                        tuningDiffs.length ? (
                          <>
                            {tuningDiffs.slice(0, 4).map((diff) => (
                              <div className="profile-diff" key={diff.label}>
                                <span className="diff-label">{diff.label}</span>
                                <span className="diff-pair">
                                  <span>{formatDiffValue(diff.previous)}</span>
                                  <span>to</span>
                                  <span>{formatDiffValue(diff.current)}</span>
                                </span>
                              </div>
                            ))}
                            {tuningDiffs.length > 4 && <span className="profile-diff-more">+{tuningDiffs.length - 4}</span>}
                          </>
                        ) : (
                          <span className="profile-empty">matched</span>
                        )
                      ) : (
                        <span className="profile-empty">pin current controls</span>
                      )}
                    </div>
                    <div className="profile-compare-actions">
                      <button className="btn ghost" type="button" onClick={pinCurrentTuning}>Pin</button>
                      <button className="btn ghost" type="button" disabled={!pinnedTuning} onClick={applyPinnedTuning}>Apply</button>
                      <button className="btn ghost" type="button" disabled={!pinnedTuning} onClick={() => setPinnedTuning(null)}>Clear</button>
                    </div>
                  </div>
                </div>
              </div>
              <div style={{ height: 8 }} />
              <Listbox
                label="Persona preset"
                caption="Persona preset"
                info="persona"
                value={presetId}
                options={[
                  ...PERSONA_PRESETS.map((preset) => ({
                    value: preset.id,
                    label: preset.label,
                    desc: preset.prompt,
                  })),
                  ...(presetId === "custom"
                    ? [{ value: "custom", label: "Custom", desc: "Your edited prompt." }]
                    : []),
                ]}
                onChange={(value) => {
                  if (value === "custom") setPresetId("custom");
                  else applyPreset(value);
                }}
              />
              <div style={{ height: 8 }} />
              <textarea
                aria-label="System prompt"
                value={textPrompt}
                maxLength={2000}
                onChange={(event) => {
                  setTextPrompt(event.target.value);
                  setPresetId("custom");
                  setSessionProfileId("custom");
                }}
              />
              <div className="field-meta">
                <span>Wrapped in &lt;system&gt;</span>
                <span>{textPrompt.length} / 2000</span>
              </div>
              <div className="prompt-modes">
                <Listbox
                  label="Adherence"
                  caption="Adherence"
                  info="adherence"
                  value={adherenceMode}
                  options={ADHERENCE_MODES.map((mode) => ({
                    value: mode.id,
                    label: mode.label,
                    desc: mode.desc,
                  }))}
                  onChange={(value) => {
                    setAdherenceMode(value);
                    setSessionProfileId("custom");
                  }}
                />
                <Listbox
                  label="Expression"
                  caption="Expression"
                  info="expression"
                  value={expressionMode}
                  options={EXPRESSION_MODES.map((mode) => ({
                    value: mode.id,
                    label: mode.label,
                    desc: mode.desc,
                  }))}
                  onChange={(value) => {
                    setExpressionMode(value);
                    setSessionProfileId("custom");
                  }}
                />
              </div>
              <div className="opt-row">
                <div className="opt-l">
                  <span className="opt-n" style={{ display: "inline-flex", alignItems: "center" }}>
                    Reinforce in silences
                    <Info k="reinforce" />
                  </span>
                  <span className="opt-d">Re-assert the persona during pauses to fight long-session drift</span>
                </div>
                <button
                  type="button"
                  className={cls("switch", reinforceInSilences && "on")}
                  role="switch"
                  aria-checked={reinforceInSilences}
                  aria-label="Reinforce persona during silences"
                  onClick={() => {
                    setReinforceInSilences(!reinforceInSilences);
                    setSessionProfileId("custom");
                  }}
                />
              </div>
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">02 · VOICE</div>
                  <div className="sect-title" style={{ display: "inline-flex", alignItems: "center" }}>
                    Timbre &amp; prefix
                    <Info k="voice" />
                  </div>
                </div>
                <span className="sect-sub mono">{voiceDisplay}</span>
              </div>
              <div className="voice-filters">
                <div className="vf-row">
                  <span className="vf-l">Gender</span>
                  <div className="vf-seg">
                    {["F", "M", "all"].map((gender) => (
                      <button
                        key={gender}
                        type="button"
                        className={cls(voiceGender === gender && "on")}
                        onClick={() => setVoiceGender(gender)}
                      >
                        {gender === "all" ? "All" : gender}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="voice-list">
                {filteredVoices.map((item) => {
                  const seedValue = [...item].reduce((sum, char) => sum + char.charCodeAt(0), 0);
                  const heights = Array.from({ length: 11 }, (_, index) => 3 + ((seedValue * (index + 1) * 7) % 11));
                  const selectVoice = () => {
                    setSessionProfileId("custom");
                    setVoice(item);
                    clearUploadedVoice();
                  };
                  const isPreviewing = previewing === item;
                  return (
                    // biome-ignore lint/a11y: row is a voice selector with a nested preview button; Enter/Space keyboard handler and aria-pressed are provided; a semantic restructure is deferred pending visual QA
                    <div
                      key={item}
                      role="button"
                      tabIndex={0}
                      className={cls("voice", !uploadedVoiceFilename && voice === item && "active")}
                      aria-pressed={!uploadedVoiceFilename && voice === item}
                      aria-label={`Use voice ${item}`}
                      onClick={selectVoice}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          selectVoice();
                        }
                      }}
                    >
                      <button
                        type="button"
                        className={cls("play", isPreviewing && "playing")}
                        aria-label={isPreviewing ? `Stop preview of voice ${item}` : `Preview voice ${item}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          previewVoice(item);
                        }}
                      >
                        <svg viewBox="0 0 8 8" aria-hidden="true" focusable="false">
                          {isPreviewing ? (
                            <rect x="2" y="2" width="4" height="4" fill="currentColor" />
                          ) : (
                            <polygon points="2,1 7,4 2,7" fill="currentColor" />
                          )}
                        </svg>
                      </button>
                      <span className="name">{item}</span>
                      <span className="glyph">
                        {heights.map((height, index) => (
                          <i key={GLYPH_BARS[index]} style={{ height }} />
                        ))}
                      </span>
                    </div>
                  );
                })}
              </div>
              {!uploadedVoiceFilename && (
                <>
                  <ToggleRow
                    info="voiceBlend"
                    name="Blend a second voice"
                    desc="Interpolate two speaker embeddings"
                    value={voiceBlend}
                    onChange={(value) => {
                      setVoiceBlend(value);
                      setSessionProfileId("custom");
                    }}
                  />
                  {voiceBlend && (
                    <div className="blend">
                      <div className="blend-ends">
                        <span className="mono">{voice}</span>
                        <select
                          aria-label="Second voice to blend"
                          value={voiceB}
                          onChange={(event) => {
                            setVoiceB(event.target.value);
                            setSessionProfileId("custom");
                          }}
                        >
                          {voiceList
                            .filter((item) => item !== voice)
                            .map((item) => (
                              <option key={item} value={item}>
                                {item}
                              </option>
                            ))}
                        </select>
                      </div>
                      <input
                        className="blend-slider"
                        type="range"
                        min={0}
                        max={100}
                        step={1}
                        value={blendMix}
                        aria-label="Voice blend mix"
                        onChange={(event) => {
                          setBlendMix(Number(event.target.value));
                          setSessionProfileId("custom");
                        }}
                      />
                      <div className="blend-meta mono">
                        <span>{100 - blendMix}%</span>
                        <span>mix</span>
                        <span>{blendMix}%</span>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">03 · CLONE</div>
                  <div className="sect-title" style={{ display: "inline-flex", alignItems: "center" }}>
                    Reference clip
                    <Info k="clone" />
                  </div>
                </div>
                <span className="sect-sub">{uploadedVoiceFilename ? "active" : "optional"}</span>
              </div>
              <button
                type="button"
                className="drop"
                onClick={() => cloneFileRef.current?.click()}
              >
                <div className="t">{uploadedVoiceLabel || "Drop audio or click to upload"}</div>
                <div>10 to 60 s, one clean speaker, common audio formats</div>
              </button>
              <input
                ref={cloneFileRef}
                id="cloneFile"
                className="sr-only"
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.opus,.aac"
                aria-label="Upload voice reference clip"
                onChange={(event) => uploadVoice(event.target.files?.[0])}
              />
              {uploadStatus && <div className={cls("upload-status", uploadKind)}>{uploadStatus}</div>}
              {uploadedVoiceMeta && (
                <div className="clone-meta">
                  <span>{uploadedVoiceMeta.duration ? `${fmt(uploadedVoiceMeta.duration, 1)} s` : "duration unknown"}</span>
                  <span>{fmt(uploadedVoiceMeta.size / (1024 * 1024), 1)} MB</span>
                  <span>{uploadedVoiceMeta.type}</span>
                </div>
              )}
              {uploadedVoiceFilename && (
                <div className="clone-actions">
                  <button
                    className="btn ghost"
                    type="button"
                    disabled={!uploadedVoicePreviewUrl}
                    onClick={previewUploadedVoice}
                  >
                    Preview clip
                  </button>
                  <button
                    className="btn ghost"
                    type="button"
                    onClick={clearUploadedVoice}
                  >
                    Remove clone
                  </button>
                </div>
              )}
              {uploadedVoiceFilename && (
                <div className="clone-strength">
                  <div className="clone-strength-row">
                    <span className="l" style={{ display: "inline-flex", alignItems: "center" }}>
                      Clone strength
                      <Info k="cloneStrength" />
                    </span>
                    <span className="v mono">{cloneStrength}%</span>
                  </div>
                  <input
                    className="clone-strength-slider"
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={cloneStrength}
                    aria-label="Clone strength"
                    onChange={(event) => {
                      setCloneStrength(Number(event.target.value));
                      setSessionProfileId("custom");
                    }}
                  />
                  <div className="clone-strength-hint">How strongly the reference clip conditions timbre</div>
                </div>
              )}
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">04 · VISION</div>
                  <div className="sect-title" style={{ display: "inline-flex", alignItems: "center" }}>
                    Scene prompt
                    <Info k="visionPrompt" />
                  </div>
                </div>
                <span className="sect-sub">Gemini</span>
              </div>
              <textarea
                aria-label="Vision prompt"
                value={visionPrompt}
                maxLength={1000}
                onChange={(event) => {
                  setVisionPrompt(event.target.value);
                  setSessionProfileId("custom");
                }}
              />
              <div className="field-meta">
                <span>Sent with captured frames</span>
                <span>{visionPrompt.length} / 1000</span>
              </div>
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">05 · MIC</div>
                  <div className="sect-title">Capture input</div>
                </div>
                <span className="sect-sub">getUserMedia</span>
              </div>
              <ToggleRow info="echo" name="Echo cancellation" desc="Speaker bleed can loop the model" value={echoCancel} onChange={(value) => { setEchoCancel(value); setSessionProfileId("custom"); }} />
              <ToggleRow info="noise" name="Noise suppression" desc="Drops keyboard, fan, hiss" value={noiseSupp} onChange={(value) => { setNoiseSupp(value); setSessionProfileId("custom"); }} />
              <ToggleRow info="agc" name="Auto gain" desc="May swing the model input" value={autoGain} onChange={(value) => { setAutoGain(value); setSessionProfileId("custom"); }} />
              <div className="device-route">
                <div className="device-route-copy">
                  <div className="n" style={{ display: "inline-flex", alignItems: "center" }}>
                    Speaker output
                    <Info k="output" />
                  </div>
                  <div className="d">{canRouteOutput ? "Assistant playback route" : "Browser controlled"}</div>
                </div>
                {canRouteOutput ? (
                  <Listbox
                    value={audioOutputOptions.some((option) => option.value === outputDeviceId) ? outputDeviceId : "default"}
                    options={audioOutputOptions}
                    onChange={(value) => {
                      setOutputDeviceId(value);
                      setSessionProfileId("custom");
                    }}
                    placeholder="System default"
                    label="Speaker output"
                  />
                ) : (
                  <span className="device-route-status">not supported</span>
                )}
              </div>
              {canRouteOutput && (
                <button className="btn ghost block device-refresh" type="button" onClick={refreshAudioOutputs}>
                  Refresh outputs
                </button>
              )}
            </div>

            <div className="sect">
              <div className="sect-h">
                <div>
                  <div className="sect-num">06 · SEED</div>
                  <div className="sect-title" style={{ display: "inline-flex", alignItems: "center" }}>
                    Reproducibility
                    <Info k="seed" />
                  </div>
                </div>
                <span className="sect-sub">{seedRandom ? "random" : "fixed"}</span>
              </div>
              <div className="seed-row">
                <input
                  type="number"
                  aria-label="Seed value"
                  min={0}
                  max={2147483647}
                  value={seed}
                  disabled={seedRandom}
                  onChange={(event) => {
                    setSeed(Number.parseInt(event.target.value, 10) || 0);
                    setSessionProfileId("custom");
                  }}
                />
                <button className="btn ghost" type="button" onClick={() => {
                  setSeedRandom(!seedRandom);
                  setSessionProfileId("custom");
                }}>
                  {seedRandom ? "Lock" : "Random"}
                </button>
              </div>
              <div className="opt-row">
                <div className="opt-l">
                  <span className="opt-n" style={{ display: "inline-flex", alignItems: "center" }}>
                    Session limit
                    <Info k="idle" />
                  </span>
                  <span className="opt-d">Auto-end the session to release the live slot</span>
                </div>
                <div className="step-num">
                  <button
                    type="button"
                    aria-label="Decrease session limit"
                    onClick={() => {
                      setIdleTimeout((value) => Math.max(0, value - 5));
                      setSessionProfileId("custom");
                    }}
                  >
                    −
                  </button>
                  <span className="mono">{idleTimeout ? `${idleTimeout}m` : "off"}</span>
                  <button
                    type="button"
                    aria-label="Increase session limit"
                    onClick={() => {
                      setIdleTimeout((value) => Math.min(60, value + 5));
                      setSessionProfileId("custom");
                    }}
                  >
                    +
                  </button>
                </div>
              </div>
              <div className="opt-row">
                <div className="opt-l">
                  <span className="opt-n">Session config</span>
                  <span className="opt-d">Save or load the full connect-time setup as a file</span>
                </div>
                <div className="config-io">
                  <button className="btn ghost" type="button" onClick={exportConfig}>{Icon.dl} Export</button>
                  <button className="btn ghost" type="button" onClick={() => configFileRef.current?.click()}>{Icon.plus} Import</button>
                </div>
              </div>
            </div>
          </div>

          <div className="cta">
            {sideCollapsed ? (
              isLive ? (
                <button
                  className="btn danger lg block"
                  type="button"
                  aria-label="End session"
                  title="End session"
                  onClick={stopConversation}
                >
                  {Icon.stop}
                </button>
              ) : (
                <button
                  className="btn lg block"
                  type="button"
                  disabled
                  aria-label={phase === "warmup" ? "Warming up" : "Negotiating"}
                  title={phase === "warmup" ? "Warming up" : "Negotiating"}
                >
                  {Icon.mic}
                </button>
              )
            ) : phase === "idle" || phase === "ended" ? (
              <>
                <button
                  className="btn primary lg block hold-connect"
                  type="button"
                  onPointerDown={beginConnectHold}
                  onPointerUp={clearConnectHold}
                  onPointerCancel={clearConnectHold}
                  onPointerLeave={clearConnectHold}
                  onKeyDown={keyConnect}
                  aria-label="Hold to connect, or press Enter to connect"
                  style={{ "--hold": `${connectHoldPct}%` }}
                >
                  {Icon.mic} Hold to connect
                  <span className="hold-fill" aria-hidden="true" />
                </button>
                <button className="btn ghost block" type="button" style={{ marginTop: 6, fontSize: 11 }} onClick={runPreflight}>
                  {preflightDone ? (preflight.turn === "ok" ? "Devices tested" : "Re-test devices") : "Test devices"}
                </button>
              </>
            ) : phase === "connecting" ? (
              <button className="btn lg block" type="button" disabled>
                Negotiating
              </button>
            ) : phase === "warmup" ? (
              <button className="btn lg block" type="button" disabled>
                Warming up
              </button>
            ) : (
              <button className="btn danger lg block" type="button" onClick={stopConversation}>
                {Icon.stop} End session
              </button>
            )}
          </div>
        </aside>

        <main className="stage">
          <div className="stage-head">
            <div className="l">
              <div>
                <div className="h1">Conversation</div>
                <div className="sub">
                  {isBusy
                    ? "Pod busy, another client connected"
                    : isTurnFailed
                      ? "TURN provisioning failed"
                      : isLive
                        ? `Voice: ${voiceDisplay}${interrupting ? " · stopping response" : visionInjecting ? " · injecting context" : ""}`
                        : stageMessage}
                </div>
              </div>
            </div>
            <div className="r">
              {isBusy && <Badge kind="warn" label="Busy" />}
              {isTurnFailed && <Badge kind="warn" label="TURN failed" />}
              {!isBusy && !isTurnFailed && isLive && <Badge kind="live" label={`Live · ${elapsedStr}`} />}
              {!isBusy && !isTurnFailed && phase === "connecting" && <Badge kind="warn" label="Connecting" />}
              {!isBusy && !isTurnFailed && phase === "warmup" && <Badge kind="warn" label="Warmup" />}
              {!isBusy && !isTurnFailed && phase === "idle" && <Badge label="Ready" />}
              {!isBusy && !isTurnFailed && phase === "ended" && <Badge label={`Ended · ${elapsedStr}`} />}
            </div>
          </div>

          <div className="telem">
            <TelemetryCell label="Latency" value={latencyMs || "·"} unit="ms" fill={Math.min(100, (latencyMs / 300) * 100)} warn={latencyMs > 220} err={latencyMs > 280} />
            <TelemetryCell label="Tail · p95" value={tailLatencyMs || "·"} unit="ms" fill={Math.min(100, (tailLatencyMs / 380) * 100)} warn={tailLatencyMs > 260} err={tailLatencyMs > 340} />
            <TelemetryCell label="Turn buffer" value={assistantRate.words} unit={`/${maxTurn} ≈tok`} fill={Math.min(100, (assistantRate.words / Math.max(1, maxTurn)) * 100)} warn={assistantRate.words > maxTurn * 0.75} err={assistantRate.words > maxTurn * 0.9} />
            <TelemetryCell label="Response rate" value={assistantRate.wpm || "·"} unit="wpm" fill={Math.min(100, (assistantRate.wpm / 220) * 100)} warn={assistantRate.wpm > 170} err={assistantRate.wpm > 220} />
            <TelemetryCell label="Vision sent / gated" value={visionFramesSent} unit={`/${visionFramesSent + visionFramesGated || "·"}`} fill={(visionFramesSent / Math.max(1, visionFramesSent + visionFramesGated)) * 100} violet />
          </div>

          <div className="stage-main">
            <div
              className="deck"
              role="img"
              aria-label={
                isLive
                  ? `Signal meters. Outbound voice ${voiceDisplay}: ${visionInjecting ? "gated" : speaking === "ai" || speaking === "both" ? "active" : "idle"}. Inbound microphone: ${speaking === "you" || speaking === "both" ? "active" : "idle"}.`
                  : "Signal meters on standby."
              }
            >
              <div className={cls("chan", (speaking === "you" || speaking === "both") && "hot")}>
                <div className="chan-h">
                  <span className="chan-n">IN · YOU</span>
                  <span className="chan-led amber" />
                </div>
                <div className="chan-body">
                  <div className="chan-scale mono">
                    <span>0</span>
                    <span>-6</span>
                    <span>-18</span>
                    <span>-∞</span>
                  </div>
                  <VuMeter value={levels.mic} color="amber" peak={speaking === "you" || speaking === "both"} />
                </div>
                <div className="chan-s mono">
                  {speaking === "you" || speaking === "both" ? "SIG" : isLive ? "idle" : "—"}
                </div>
              </div>

              <div className="scope">
                <div className="scope-screen">
                  <Scope active={isLive && !visionInjecting} speaking={visionInjecting ? null : speaking} />
                </div>
                <div className="scope-corner tl">
                  <span className="dot green" />CH1 · AI OUT
                </div>
                <div className="scope-corner tr">
                  CH2 · YOU IN<span className="dot amber" />
                </div>
                <div className="scope-corner bl mono">24 kHz · MIMI</div>
                <div className="scope-corner br mono">
                  {isLive
                    ? `${(rtf || 0.6).toFixed(2)}× RTF`
                    : phase === "warmup"
                      ? "WARMUP"
                      : phase === "connecting"
                        ? "SYNC"
                        : "STANDBY"}
                </div>

                {visionInjecting && (
                  <div className="viz-inject">
                    <span className="d" /> Injecting context <span className="gate">audio gated</span>
                  </div>
                )}
                {interrupting && !visionInjecting && (
                  <div className="viz-inject interrupt">
                    <span className="d" /> Stop response <span className="gate">output cleared</span>
                  </div>
                )}
                {speaking === "both" && !visionInjecting && !interrupting && (
                  <div className="viz-inject barge">
                    <span className="d" /> Barge-in <span className="gate">user took the turn</span>
                  </div>
                )}
                {(isBusy || isTurnFailed || reconnecting || phase === "idle" || phase === "connecting" || phase === "warmup") && (
                  <div className={cls("viz-overlay", (reconnecting || phase === "connecting" || phase === "warmup") && "connecting", (isBusy || isTurnFailed) && "error")}>
                    <div className="stack">
                      <span className="label">
                        <span className="d" />
                        {isBusy
                          ? "Pod busy"
                          : isTurnFailed
                            ? "TURN provisioning failed"
                            : reconnecting
                              ? "ICE restart, re-establishing transport"
                              : phase === "idle"
                                ? "Standby. Connect to begin."
                                : stageMessage}
                      </span>
                      {(isBusy || isTurnFailed) && (
                        <span className="sub">
                          {isBusy
                            ? "Server enforces one live session. Try again when the current client disconnects."
                            : "Cloudflare TURN credentials could not be minted. Check TURN_KEY_ID and TURN_KEY_API_TOKEN."}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className={cls("chan", (speaking === "ai" || speaking === "both") && !visionInjecting && "hot")}>
                <div className="chan-h">
                  <span className="chan-led green" />
                  <span className="chan-n">OUT · AI</span>
                </div>
                <div className="chan-body">
                  <VuMeter value={visionInjecting ? 0 : levels.ai} color="green" peak={!visionInjecting && (speaking === "ai" || speaking === "both")} />
                  <div className="chan-scale mono">
                    <span>0</span>
                    <span>-6</span>
                    <span>-18</span>
                    <span>-∞</span>
                  </div>
                </div>
                <div className="chan-s mono">
                  {visionInjecting ? "GATE" : speaking === "ai" || speaking === "both" ? "SIG" : isLive ? "idle" : "—"}
                </div>
              </div>
            </div>

            <div className={cls("lower", visionOn && "with-vision")}>
              <div className="transcript">
                <div className="transcript-h">
                  <span className="l">Transcript</span>
                  <span className="r">{assistantRate.words ? `${assistantRate.wpm} wpm` : isLive ? "streaming" : phase}</span>
                </div>
                <div className="transcript-stream">
                  {transcriptTurns.length === 0 ? (
                    <div className="transcript-empty">
                      <div>
                        <div className="label">{isLive ? "Listening" : "No active transcript"}</div>
                        <div className="sub">{isLive ? "Speak into your microphone." : "Configure persona on the left, then connect."}</div>
                      </div>
                    </div>
                  ) : (
                    transcriptTurns.map((turn) =>
                      turn.role === "ai" ? (
                        <div key={turn.id} className="line ai">
                          <span className="who">AI</span>
                          <span className="text">{turn.text}</span>
                        </div>
                      ) : (
                        <div key={turn.id} className={cls("line you", turn.audioOnly && "audio-only")}>
                          <span className="who">You</span>
                          {turn.audioOnly ? (
                            <span className="text muted">spoke · audio only</span>
                          ) : (
                            <span className="text">{turn.text}</span>
                          )}
                        </div>
                      ),
                    )
                  )}
                </div>
                {/* biome-ignore lint/a11y/useAriaPropsSupportedByRole: aria-label names this readout region for assistive tech; visible per-row labels are also present */}
                <div className="turn-insights" aria-label="Conversation telemetry">
                  <div className="turn-insight">
                    <span className="k">Last turn</span>
                    <span className="v">{assistantRate.words ? `${assistantRate.words} words` : "waiting"}</span>
                  </div>
                  <div className="turn-insight">
                    <span className="k">Rate</span>
                    <span className="v">{assistantRate.words ? `${assistantRate.wpm} wpm` : "no sample"}</span>
                  </div>
                  <div className="turn-insight">
                    <span className="k">Timeline</span>
                    <span className="v">{sessionTimeline.length ? `${sessionTimeline.length} points` : "empty"}</span>
                  </div>
                </div>
                {timelinePreview.length > 0 && (
                  <div className="turn-ribbon-wrap">
                    <div className="turn-ribbon-head">
                      <span>Session timeline</span>
                      <span>{recordingUrl ? "scrub enabled" : phase === "ended" ? "recording unavailable" : "live capture"}</span>
                    </div>
                    <ul className="turn-ribbon" aria-label="Session timeline">
                      {timelinePreview.map((item) => {
                        const offset = formatOffset(item.offsetMs || 0);
                        const progress = Math.min(100, Math.max(0, ((item.offsetMs || 0) / timelineDurationMs) * 100));
                        return (
                          <li className="turn-mark-item" key={item.id}>
                            <button
                              type="button"
                              className={cls("turn-mark", item.kind, item.level)}
                              style={{ "--p": `${progress}%` }}
                              aria-label={`${item.kind || "event"} at ${offset}: ${item.label}`}
                              onClick={() => activateTimelinePoint(item)}
                            >
                              <span className="t">{offset}</span>
                              <span className="m">{item.label}</span>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                    {recordingUrl && (
                      // biome-ignore lint/a11y/useMediaCaption: synthesized/recorded conversational audio has no caption track
                      <audio
                        ref={recordingPlaybackRef}
                        className="timeline-audio"
                        controls
                        src={recordingUrl}
                        aria-label="Session recording playback"
                      />
                    )}
                  </div>
                )}
              </div>

              {visionOn && (
                <div className="vision">
                  <div className="vision-frame">
                    <video ref={visionVideoRef} className="vision-video" autoPlay playsInline muted />
                    <div className={cls("vision-rec", visionPaused && "paused")}>{visionPaused ? "Paused" : "Cam · Live"}</div>
                    <div className={cls("vision-caption", currentCaption && "visible")}>{currentCaption}</div>
                  </div>
                  <div className="vision-meta">
                    <span><b>{visionFramesSent}</b> sent</span>
                    <span><b>{visionFramesGated}</b> gated</span>
                    <span>~$<b>{visionCostUsd.toFixed(4)}</b></span>
                    {visionCostLimitActive && <span><b>${visionCostRemaining.toFixed(4)}</b> left</span>}
                    {visionBudgetTripped && <span className="warn">budget hit</span>}
                  </div>
                  <div className="vision-history">
                    {captionEntries.length === 0 ? (
                      <div className="v-entry" style={{ fontStyle: "italic", color: "var(--ink-5)", cursor: "default" }}>
                        Awaiting first description
                      </div>
                    ) : (
                      captionEntries.map((entry) => (
                        <button
                          type="button"
                          className="v-entry"
                          aria-label={`Inspect frame from ${entry.ts}`}
                          key={entry.id}
                          onClick={() => setInspectFrame(entry)}
                          title="Inspect source frame"
                        >
                          <span className="ts">{entry.ts}</span>
                          {entry.text}
                        </button>
                      ))
                    )}
                  </div>
                  <div className="vision-tune">
                    <div>
                      <div className="mini-row">
                        <span className="l" style={{ display: "inline-flex", alignItems: "center" }}>
                          Idle heartbeat
                          <Info k="heartbeat" />
                        </span>
                        <span className="v">every {visionIntervalMs / 1000} s</span>
                      </div>
                      <input
                        type="range"
                        min={1}
                        max={30}
                        step={1}
                        value={visionIntervalMs / 1000}
                        aria-label="Idle heartbeat interval"
                        onChange={(event) => {
                          setVisionIntervalMs(Number(event.target.value) * 1000);
                          setSessionProfileId("custom");
                        }}
                      />
                    </div>
                    <div>
                      <div className="mini-row">
                        <span className="l" style={{ display: "inline-flex", alignItems: "center" }}>
                          Cost ceiling
                          <Info k="visionBudget" />
                        </span>
                        <span className="v">{visionCostLimitActive ? `$${Number(visionCostLimitUsd).toFixed(2)}` : "off"}</span>
                      </div>
                      <input
                        className="vision-budget"
                        type="number"
                        min={0}
                        max={10}
                        step={0.05}
                        value={visionCostLimitUsd}
                        aria-label="Gemini vision cost ceiling"
                        onChange={(event) => {
                          const nextLimit = Math.max(0, Number(event.target.value) || 0);
                          setVisionCostLimitUsd(nextLimit);
                          setVisionBudgetTripped(false);
                          setSessionProfileId("custom");
                          // The server enforces its own copy of the ceiling;
                          // push the change so a mid-session raise takes
                          // effect there too.
                          sendLiveConfig({ vision_cost_limit_usd: nextLimit });
                        }}
                      />
                    </div>
                    <div className="mini-row" style={{ paddingTop: 4 }}>
                      <span className="l" style={{ display: "inline-flex", alignItems: "center" }}>
                        Echo in transcript
                        <Info k="vision" />
                      </span>
                      <button
                        type="button"
                        className={cls("switch", visionInTranscript && "on")}
                        role="switch"
                        aria-checked={visionInTranscript}
                        aria-label="Echo vision captions in transcript"
                        onClick={() => {
                          const nextEcho = !visionInTranscript;
                          setVisionInTranscript(nextEcho);
                          setSessionProfileId("custom");
                          // The echo is produced server-side at caption
                          // time; push the toggle so it applies mid-session.
                          sendLiveConfig({ vision_in_transcript: nextEcho });
                        }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className={cls("rack", cfgLocked && "cfg-locked")}>
            <button
              type="button"
              className="rack-bar"
              aria-expanded={railOpen}
              aria-controls="tuning-rail"
              onClick={() => setRailOpen((open) => !open)}
            >
              <span className="rack-t">Tuning · sampling &amp; behavior</span>
              <span className="rack-meta mono">
                {railOpen
                  ? "Hide"
                  : `t ${fmt(textTemp, 2)} · rep ${fmt(repPenalty, 2)} · ${maxTurn ? `${maxTurn} tok` : "no cap"}`}
              </span>
              <span className="rack-x mono" aria-hidden="true">{railOpen ? "▾" : "▸"}</span>
            </button>
            {railOpen && (
              <div className="rail" id="tuning-rail">
                <RailColumn title="TEXT" aggregate={`t ${fmt(textTemp, 2)} · k ${textTopk}`}>
                  <MiniSlider label="Temperature" info="txtTemp" value={textTemp} onChange={(value) => { setTextTemp(value); setSessionProfileId("custom"); sendLiveConfig({ text_temperature: Number(value) }); }} min={0.1} max={1.5} step={0.05} format={(v) => fmt(v, 2)} />
                  <MiniSlider label="Top-k" info="txtTopK" value={textTopk} onChange={(value) => { setTextTopk(value); setSessionProfileId("custom"); sendLiveConfig({ text_topk: Number.parseInt(value, 10) }); }} min={1} max={500} step={1} format={(v) => fmt(v, 0)} />
                </RailColumn>
                <RailColumn title="AUDIO" aggregate={`t ${fmt(audioTemp, 2)} · k ${audioTopk}`}>
                  <MiniSlider label="Temperature" info="audTemp" value={audioTemp} onChange={(value) => { setAudioTemp(value); setSessionProfileId("custom"); sendLiveConfig({ audio_temperature: Number(value) }); }} min={0.1} max={1.5} step={0.05} format={(v) => fmt(v, 2)} />
                  <MiniSlider label="Top-k" info="audTopK" value={audioTopk} onChange={(value) => { setAudioTopk(value); setSessionProfileId("custom"); sendLiveConfig({ audio_topk: Number.parseInt(value, 10) }); }} min={1} max={2048} step={1} format={(v) => fmt(v, 0)} />
                </RailColumn>
                <RailColumn title="REPETITION" aggregate={`${fmt(repPenalty, 2)} · ${repContext} tok`}>
                  <MiniSlider label="Penalty" info="repPen" value={repPenalty} onChange={(value) => { setRepPenalty(value); setSessionProfileId("custom"); sendLiveConfig({ repetition_penalty: Number(value) }); }} min={1} max={2} step={0.05} format={(v) => fmt(v, 2)} />
                  <MiniSlider label="Context" info="repCtx" value={repContext} onChange={(value) => { setRepContext(value); setSessionProfileId("custom"); sendLiveConfig({ repetition_penalty_context: Number.parseInt(value, 10) }); }} min={0} max={256} step={8} format={(v) => fmt(v, 0)} />
                </RailColumn>
                <RailColumn title="TURN" aggregate={`${maxTurn} tok · pad ${fmt(padBonus, 1)}`}>
                  <MiniSlider label="Padding bonus" info="padBonus" value={padBonus} onChange={(value) => { setPadBonus(value); setSessionProfileId("custom"); sendLiveConfig({ padding_bonus: Number(value) }); }} min={0} max={6} step={0.1} format={(v) => fmt(v, 1)} />
                  <MiniSlider label="Max length" info="maxTurn" value={maxTurn} onChange={(value) => { setMaxTurn(value); setSessionProfileId("custom"); sendLiveConfig({ max_turn_text_tokens: Number.parseInt(value, 10) }); }} min={0} max={2000} step={10} format={(v) => (v ? `${v}` : "off")} />
                </RailColumn>
              </div>
            )}
          </div>

          <div className="transport">
            <div className="levels">
              <Level label="MIC IN" value={levels.mic} you />
              <Level label="AI OUT" value={levels.ai} />
            </div>
            <div className="controls">
              {isLive && (
                <>
                  <button className={cls("btn", visionOn && "primary")} type="button" onClick={startVision}>
                    {Icon.eye} {visionOn ? "Stop vision" : "Add vision"}
                  </button>
                  {visionOn && (
                    <>
                      <button className="btn ghost" type="button" onClick={toggleVisionPause}>{Icon.pause} {visionPaused ? "Resume" : "Pause"}</button>
                      <button className="btn ghost" type="button" onClick={forceCapture}>{Icon.cam} Force capture</button>
                    </>
                  )}
                  <button className="btn ghost" type="button" onClick={rewind}>{Icon.rewind} Rewind</button>
                  <button className="btn ghost" type="button" onClick={addBookmark} title="Bookmark the current snapshot to jump back to it">{Icon.bookmark} Bookmark</button>
                  <button className="btn danger" type="button" onClick={() => interruptResponse("manual")}>{Icon.stop} Stop response</button>
                </>
              )}
              {phase === "ended" && (
                <>
                  {recordingUrl && (
                    <a className="btn primary" href={recordingUrl} download={`personaplex_conversation.${recordingMime.includes("ogg") ? "ogg" : "webm"}`}>
                      {Icon.dl} Download audio
                    </a>
                  )}
                  {serverRecording?.ready && serverRecording.url && (
                    <a
                      className={cls("btn", recordingUrl ? "ghost" : "primary")}
                      href={serverRecording.url}
                      download="conversation-audio.wav"
                    >
                      {Icon.dl} {recordingUrl ? "Server copy" : "Download audio"}
                    </a>
                  )}
                  <button className="btn" type="button" onClick={newConversation}>{Icon.plus} New</button>
                </>
              )}
              {phase === "idle" && <span className="mono" style={{ fontSize: 10, letterSpacing: "0.16em", color: "var(--ink-4)" }}>STANDBY</span>}
            </div>
          </div>
        </main>

        <aside className="cons" aria-label="Session diagnostics">
          <div className="cons-sect">
            <div className="cons-h">A · Session</div>
            <Row label="Status" value={phase} dot={isLive ? "ok" : phase === "connecting" || phase === "warmup" ? "warn" : ""} />
            <Row label="Uptime" value={elapsedStr} />
            <Row label="Voice" value={voiceDisplay} />
            <Row label="Vision" value={visionOn ? (visionPaused ? "paused" : `live · ${visionAge ?? "idle"} s`) : visionEnabledFromServer ? "available" : "disabled"} />
            {serverRecording && (
              <Row
                label="Server recording"
                value={serverRecording.active ? "active" : serverRecording.ready ? "saved" : "off"}
                dot={serverRecording.active ? "ok" : ""}
              />
            )}
            <Row
              label="Real-time factor"
              value={isLive ? `${rtf.toFixed(2)}×` : "—"}
              dot={isLive ? (rtf < 1 ? "ok" : "warn") : ""}
            />
            <div className="rttgraph">
              <div className="axis">RTT · 60 s</div>
              <RTTGraph samples={rttSamples} />
            </div>
          </div>

          <div className="cons-sect">
            <div className="cons-h">B · Network</div>
            <div className="row">
              <span className="k"><span className={cls("d", isLive && (netStats.quality >= 70 ? "ok" : netStats.quality >= 40 ? "warn" : "err"))} />Quality</span>
              <span className="v">{isLive ? `${netStats.quality}%` : "—"}</span>
            </div>
            <div className="netbar">
              <div
                className={cls("netbar-fill", netStats.quality < 40 && "err", netStats.quality >= 40 && netStats.quality < 70 && "warn")}
                style={{ width: `${isLive ? netStats.quality : 0}%` }}
              />
            </div>
            <Row label="Jitter" value={isLive ? `${netStats.jitterMs} ms` : "—"} />
            <Row label="Packet loss" value={isLive ? `${netStats.lossPct}%` : "—"} dot={isLive && netStats.lossPct > 1 ? "warn" : ""} />
            <Row label="Candidate" value={isLive && netStats.candidate ? netStats.candidate : "—"} dot={isLive ? "ok" : ""} />
            <div className="seg-mini-row">
              <span className="seg-mini-label" id="jitter-buffer-label" style={{ display: "inline-flex", alignItems: "center" }}>Jitter buffer<Info k="jitter" /></span>
              {/* biome-ignore lint/a11y/useSemanticElements: segmented toggle; role=group with aria-labelledby is correct here and <fieldset> would impose form-control styling */}
              <div className="seg-mini" role="group" aria-labelledby="jitter-buffer-label">
                {[
                  ["latency", "Latency"],
                  ["smooth", "Smooth"],
                ].map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    className={cls(jitterBuffer === id && "on")}
                    aria-pressed={jitterBuffer === id}
                    onClick={() => setJitterBuffer(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <button className="btn ghost block" type="button" disabled={!isLive || reconnecting} onClick={reconnect}>
              {reconnecting ? "Reconnecting…" : "ICE restart · reconnect"}
            </button>
            <div className="cons-note">
              Survives a transient drop without losing the session or model state.
            </div>
          </div>

          <div className="cons-sect">
            <div className="cons-h">C · Pipeline</div>
            <div className="flow">
              <Flow label="Peer connection" value={isLive ? "connected · turn" : phase === "connecting" ? "gathering ICE" : "idle"} active={isLive || phase === "connecting"} warn={phase === "connecting"} />
              <Flow label="Mimi codec" value={isLive || phase === "warmup" ? "24 kHz · 12.5 fps" : "idle"} active={isLive || phase === "warmup"} />
              <Flow label="LM · personaplex-7b" value={isLive ? `t ${fmt(textTemp, 2)} · k ${textTopk}${visionInjecting ? " · gated" : ""}` : phase === "warmup" ? "warming" : "idle"} active={isLive || phase === "warmup"} warn={visionInjecting} />
              {visionOn && <Flow label="Gemini vision" value={visionPaused ? "paused" : "frames active"} active={!visionPaused} warn={visionPaused} branch />}
              <Flow label="Audio graph" value={isLive ? "recording · analysers" : "idle"} active={isLive} />
              <Flow label={gpuLabel} value={gpuValue} active={isLive} />
            </div>
          </div>

          <div className="cons-sect">
            <div className="cons-h">D · Snapshots</div>
            {bookmarks.length > 0 ? (
              <div className="bm-list">
                {bookmarks.map((bm) => (
                  <button
                    key={bm.id}
                    className="bm"
                    type="button"
                    disabled={!isLive}
                    onClick={() => jumpBookmark(bm)}
                    title={`Jump back to ${bm.label}`}
                  >
                    {Icon.bookmark}
                    <span className="bm-l">{bm.label}</span>
                    <span className="bm-t mono">{formatOffset(bm.atSec * 1000)}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="cons-note">
                The <b>Bookmark</b> control on the transport bar labels a snapshot to jump back to. Auto-rewind also restores the latest snapshot after repeated safety-net trips.
              </div>
            )}
          </div>

          <div className="cons-sect">
            <div className="cons-h events-h">
              <span>E · Events</span>
              {notices.length > 0 && <button className="clear" type="button" onClick={() => setNotices([])}>Clear · {notices.length}</button>}
            </div>
            <div className="events" role="log" aria-live="polite" aria-relevant="additions text">
              {notices.map((notice) => (
                <div className={cls("ev", notice.level)} key={notice.id}>
                  <span className="d" />
                  <span className="ts">{notice.ts.slice(0, 5)}</span>
                  <span className="txt">{notice.text}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="cons-sect">
            <div className="cons-h">F · Build</div>
            <Row label="Model" value="personaplex-7b v1" />
            <Row label="Server" value={serverInfo.serverBuild || "·"} />
            <Row label="Client" value="React · Bun" />
            <Row label="License" value="NVIDIA OML" />
          </div>
        </aside>
      </div>
      {/* biome-ignore lint/a11y/useMediaCaption: synthesized/recorded conversational audio has no caption track */}
      <audio ref={aiAudioRef} autoPlay playsInline style={{ display: "none" }} />
      <input
        ref={configFileRef}
        type="file"
        accept="application/json,.json"
        style={{ display: "none" }}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          importConfig(file);
        }}
      />
      <input
        ref={profileLibraryFileRef}
        type="file"
        accept="application/json,.json"
        style={{ display: "none" }}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          importProfileLibrary(file);
        }}
      />
      {preflightOpen && (
        <PreflightModal
          preflight={preflight}
          done={preflightDone}
          onRun={runPreflight}
          onClose={() => setPreflightOpen(false)}
        />
      )}
      {visionSourceOpen && (
        <VisionSourceModal
          onClose={() => setVisionSourceOpen(false)}
          onCamera={() => startVisionSource("camera")}
          onScreen={() => startVisionSource("screen")}
        />
      )}
      {inspectFrame && (
        <FrameModal
          entry={inspectFrame}
          onClose={closeInspectFrame}
          onDetail={() => requestFrameDetail(inspectFrame)}
        />
      )}
    </div>
  );
}

export default App;
