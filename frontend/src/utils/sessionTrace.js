const DEFAULT_MAX_EVENTS = 2048;
const DEFAULT_MAX_BYTES = 1024 * 1024;
const MIN_MAX_BYTES = 1024;
const MAX_DEPTH = 5;
const MAX_KEYS = 64;
const MAX_ARRAY_ITEMS = 64;
const MAX_SAFE_STRING = 512;
const MAX_CONTENT_STRING = 8192;
const REDACTED = "[redacted]";

export const SESSION_TRACE_SCHEMA_VERSION = 1;

const PROMPT_FIELDS = new Set(["text_prompt", "system_prompt", "vision_prompt"]);

const BOOLEAN_FIELDS = new Set([
  "active",
  "auto_gain_control",
  "caption_cfg",
  "detail",
  "echo_cancellation",
  "enabled",
  "force",
  "historical_detail",
  "native_duplex_recommended",
  "noise_suppression",
  "ready",
  "reinforce_in_silences",
  "resumed",
  "vision_feed_model",
  "vision_ground_user_turns",
  "vision_in_transcript",
  "vision_prompt_replace",
]);

const NUMBER_FIELDS = new Set([
  "age_ms",
  "age_sec",
  "assistant_turns",
  "attempt",
  "audio_temperature",
  "audio_topk",
  "auto_recoveries",
  "bookmarks",
  "bytes",
  "caption_cfg_gamma",
  "chars",
  "chunk_count",
  "clipped_samples",
  "clone_strength",
  "count",
  "duration_ms",
  "errors",
  "frames",
  "generation",
  "gpu_util",
  "idle_rms",
  "inject_silence_rms",
  "inject_silence_streak",
  "interrupts",
  "jitter_ms",
  "latency_ms",
  "limit",
  "loss_pct",
  "max",
  "max_gpu_util",
  "max_rtf",
  "max_turn_text_tokens",
  "max_vram_used",
  "min",
  "network_drops",
  "network_quality",
  "offset_ms",
  "padding_bonus",
  "pcm_drop_events",
  "pcm_dropped_ms",
  "pcm_queue_capacity",
  "pcm_queue_depth",
  "pcm_queue_high_water",
  "peak",
  "quality",
  "queued",
  "reconnects",
  "remaining_tokens",
  "repetition_penalty",
  "repetition_penalty_context",
  "resume_legs",
  "rewinds",
  "rtf",
  "rtt_ms",
  "seed",
  "seq",
  "session_timeout_sec",
  "silence_streak",
  "source_generation",
  "system_prompt_chars",
  "text_prompt_chars",
  "text_prompt_tokens",
  "text_temperature",
  "text_topk",
  "tokens",
  "total",
  "user_turns",
  "vision_captions",
  "vision_cost_limit_usd",
  "vision_cost_per_call_usd",
  "vision_frames",
  "vision_frames_gated",
  "vision_frames_sent",
  "vision_prompt_chars",
  "voice_blend_mix",
  "vram_total",
  "vram_used",
  "width",
  "words",
  "outbound_buffer_ms",
  "outbound_drop_events",
  "outbound_dropped_ms",
  "outbound_flush_events",
  "outbound_flushed_ms",
  "outbound_high_water_ms",
]);

const CONTROLLED_STRING_FIELDS = new Set([
  "browser_family",
  "candidate_type",
  "end_reason",
  "error_code",
  "jitter_buffer",
  "mode",
  "reason",
  "source",
  "status",
  "transport",
  "turn_handling",
]);

// `reason` is often populated from exception paths. A lexical token check is
// insufficient there because a secret can itself be one opaque token. Keep
// only protocol reasons we deliberately emit and collapse everything else.
const SAFE_REASON_VALUES = new Set([
  "action_timeout",
  "barge_in",
  "cadence",
  "camera_refresh",
  "caption_feed",
  "control_closed",
  "manual",
  "regression",
  "regression_stop",
  "screen_refresh",
  "silence",
]);

// These fields are never written to a report, including when the user elects
// to include conversational content. They either carry binary media, network
// credentials/topology, a temporary bearer-like identifier, or a stable
// browser/device fingerprint.
const HARD_PRIVATE_FIELDS = new Set([
  "answer",
  "api_key",
  "audio_blob",
  "audio_data",
  "authorization",
  "base64",
  "blob",
  "candidate",
  "cookie",
  "credential",
  "device_id",
  "filename",
  "frame",
  "hf_token",
  "ice_candidate",
  "ice_servers",
  "image",
  "media_stream",
  "microphone_audio",
  "offer",
  "output_device_id",
  "password",
  "pcm",
  "recording",
  "recording_url",
  "resume_session_id",
  "screenshot",
  "sdp",
  "secret",
  "session_id",
  "token",
  "turn_credential",
  "turn_password",
  "url",
  "user_agent",
  "useragent",
  "vision_frame",
  "vision_frame_chunk",
  "voice_prompt",
  "voice_prompt_b",
  "wav",
]);

const CONTENT_FIELDS = new Set([
  "assistant_text",
  "caption",
  "content",
  "label",
  "message",
  "prompt",
  "system_prompt",
  "text",
  "text_prompt",
  "transcript",
  "user_text",
  "vision_prompt",
]);

const CONFIG_FIELDS = new Set([
  "audio_temperature",
  "audio_topk",
  "caption_cfg",
  "caption_cfg_gamma",
  "clone_strength",
  "inject_silence_rms",
  "inject_silence_streak",
  "max_turn_text_tokens",
  "padding_bonus",
  "reinforce_in_silences",
  "repetition_penalty",
  "repetition_penalty_context",
  "seed",
  "session_timeout_sec",
  "system_prompt_chars",
  "text_prompt_chars",
  "text_prompt_tokens",
  "text_temperature",
  "text_topk",
  "vision_cost_limit_usd",
  "vision_cost_per_call_usd",
  "vision_feed_model",
  "vision_ground_user_turns",
  "vision_in_transcript",
  "vision_prompt_chars",
  "vision_prompt_replace",
  "voice_blend_mix",
  "voice_fingerprint",
]);

const RUNTIME_FIELDS = new Set([
  "browser_family",
  "cuda_version",
  "gpu_name",
  "model_license",
  "model_repo",
  "model_revision",
  "model_variant",
  "native_duplex_recommended",
  "python_version",
  "server_build",
  "torch_version",
  "vision_model",
  "vram_total",
]);

const SESSION_FIELDS = new Set([
  "audio_constraints",
  "duration_ms",
  "echo_cancellation",
  "end_reason",
  "jitter_buffer",
  "noise_suppression",
  "auto_gain_control",
  "resume_legs",
  "started_at",
  "turn_handling",
]);

const SUMMARY_FIELDS = new Set([
  "assistant_turns",
  "auto_recoveries",
  "bookmarks",
  "errors",
  "interrupts",
  "max_gpu_util",
  "max_rtf",
  "max_vram_used",
  "network_drops",
  "pcm_drop_events",
  "pcm_dropped_ms",
  "outbound_drop_events",
  "outbound_dropped_ms",
  "outbound_flush_events",
  "outbound_flushed_ms",
  "outbound_high_water_ms",
  "reconnects",
  "rewinds",
  "user_turns",
  "vision_captions",
  "vision_frames",
]);

const EVENT_FIELDS = new Set([
  ...CONFIG_FIELDS,
  "active",
  "age_ms",
  "age_sec",
  "applied",
  "attempt",
  "bytes",
  "candidate_type",
  "chars",
  "chunk_count",
  "clipped_samples",
  "count",
  "detail",
  "duration_ms",
  "enabled",
  "end_reason",
  "error_code",
  "force",
  "frame_id",
  "frames",
  "generation",
  "gpu_util",
  "historical_detail",
  "idle_rms",
  "jitter_ms",
  "latency_ms",
  "limit",
  "loss_pct",
  "max",
  "min",
  "mode",
  "network_quality",
  "offset_ms",
  "pcm_drop_events",
  "pcm_dropped_ms",
  "pcm_queue_capacity",
  "pcm_queue_depth",
  "pcm_queue_high_water",
  "peak",
  "quality",
  "queued",
  "ready",
  "reason",
  "remaining_tokens",
  "resumed",
  "rtf",
  "rtt_ms",
  "seq",
  "silence_streak",
  "source",
  "source_generation",
  "status",
  "tokens",
  "total",
  "transport",
  "vision_frames_gated",
  "vision_frames_sent",
  "vram_used",
  "width",
  "words",
  "outbound_buffer_ms",
  "outbound_drop_events",
  "outbound_dropped_ms",
  "outbound_flush_events",
  "outbound_flushed_ms",
  "outbound_high_water_ms",
]);

const CONTAINER_FIELDS = new Set([
  "applied_config",
  "audio_constraints",
  "config",
  "data",
  "details",
  "metrics",
  "network",
  "requested_config",
  "runtime",
  "server",
  "summary",
]);

const SAFE_KEY_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;
const SAFE_TOKEN_PATTERN = /^[a-z0-9][a-z0-9_.:-]{0,95}$/i;
const EMAIL_PATTERN = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;
const IPV4_PATTERN = /\b(?:\d{1,3}\.){3}\d{1,3}\b/g;
const URL_PATTERN = /\b(?:https?|wss?):\/\/[^\s"']+/gi;
const BEARER_PATTERN = /\bBearer\s+[A-Za-z0-9._~+/=-]+/gi;
const JWT_PATTERN = /\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b/g;
const ASSIGNMENT_SECRET_PATTERN = /\b(api[_-]?key|token|secret|password|credential)\s*[:=]\s*[^\s,;]+/gi;
const SECRET_TOKEN_PATTERN = /\b(?:sk-[A-Za-z0-9_-]{8,}|AIza[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,})\b/g;
const UNIX_PATH_PATTERN = /(^|[\s(])\/(?:home|root|tmp|var|workspace)(?:\/[^\s),;:]*)?/g;
const WINDOWS_PATH_PATTERN = /\b[A-Za-z]:\\(?:[^\\\s]+\\)*[^\s,;:]*/g;

const textEncoder = new TextEncoder();

function byteLength(value) {
  return textEncoder.encode(value).byteLength;
}

function jsonByteLength(value) {
  return byteLength(JSON.stringify(value));
}

function monotonicNow() {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

function wallClockNow() {
  return new Date().toISOString();
}

function normalizeKey(key) {
  return String(key)
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2")
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase();
}

function isHardPrivateField(key) {
  if (HARD_PRIVATE_FIELDS.has(key)) return true;
  if (
    key.endsWith("_api_key") ||
    key.endsWith("_credential") ||
    key.endsWith("_password") ||
    key.endsWith("_secret")
  ) {
    return true;
  }
  return key.endsWith("_token") && !key.endsWith("_prompt_token");
}

function scrubString(value, maxLength = MAX_SAFE_STRING) {
  return String(value)
    .replace(BEARER_PATTERN, REDACTED)
    .replace(JWT_PATTERN, REDACTED)
    .replace(SECRET_TOKEN_PATTERN, REDACTED)
    .replace(ASSIGNMENT_SECRET_PATTERN, `$1=${REDACTED}`)
    .replace(URL_PATTERN, REDACTED)
    .replace(EMAIL_PATTERN, REDACTED)
    .replace(IPV4_PATTERN, REDACTED)
    .replace(UNIX_PATH_PATTERN, `$1${REDACTED}`)
    .replace(WINDOWS_PATH_PATTERN, REDACTED)
    .slice(0, maxLength);
}

function safeToken(value, fallback) {
  const token = scrubString(value ?? "", 96);
  if (token.includes(REDACTED)) return fallback;
  return SAFE_TOKEN_PATTERN.test(token) ? token : fallback;
}

function safePrimitive(value, maxString = MAX_SAFE_STRING) {
  if (typeof value === "string") return scrubString(value, maxString);
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (value === null) return null;
  return undefined;
}

function allowedFieldsForMode(mode) {
  if (mode === "config") return CONFIG_FIELDS;
  if (mode === "runtime") return RUNTIME_FIELDS;
  if (mode === "session") return SESSION_FIELDS;
  if (mode === "summary") return SUMMARY_FIELDS;
  return EVENT_FIELDS;
}

function childModeForKey(key, fallback) {
  if (key === "config" || key === "requested_config" || key === "applied_config") {
    return "config";
  }
  if (key === "runtime") return "runtime";
  if (key === "summary") return "summary";
  return fallback;
}

function sanitizeValue(value, options, depth, mode) {
  if (depth > MAX_DEPTH) return undefined;
  const primitive = safePrimitive(value);
  if (primitive !== undefined) return primitive;
  if (Array.isArray(value)) {
    const out = [];
    for (const item of value.slice(0, MAX_ARRAY_ITEMS)) {
      const sanitized = sanitizeValue(item, options, depth + 1, mode);
      if (sanitized !== undefined) out.push(sanitized);
    }
    return out;
  }
  if (!value || typeof value !== "object") return undefined;

  const out = {};
  const allowed = allowedFieldsForMode(mode);
  for (const [rawKey, rawValue] of Object.entries(value).slice(0, MAX_KEYS)) {
    const key = normalizeKey(rawKey);
    if (!key || !SAFE_KEY_PATTERN.test(key) || isHardPrivateField(key)) continue;

    const isContent = CONTENT_FIELDS.has(key) || PROMPT_FIELDS.has(key);
    if (isContent) {
      if (!options.includeContent || typeof rawValue !== "string") continue;
      out[key] = scrubString(rawValue, MAX_CONTENT_STRING);
      continue;
    }

    if (!allowed.has(key) && !CONTAINER_FIELDS.has(key)) continue;
    if (BOOLEAN_FIELDS.has(key)) {
      if (typeof rawValue === "boolean") out[key] = rawValue;
      continue;
    }
    if (NUMBER_FIELDS.has(key)) {
      if (typeof rawValue === "number" && Number.isFinite(rawValue)) {
        out[key] = rawValue;
      }
      continue;
    }
    if (key === "voice_fingerprint") {
      if (typeof rawValue === "string" && /^sha256:[a-f0-9]{64}$/i.test(rawValue)) {
        out[key] = rawValue.toLowerCase();
      }
      continue;
    }
    if (CONTROLLED_STRING_FIELDS.has(key)) {
      if (typeof rawValue === "string") {
        const token = safeToken(rawValue, "other");
        out[key] = key === "reason" && !SAFE_REASON_VALUES.has(token)
          ? "other"
          : token;
      }
      continue;
    }
    const sanitized = sanitizeValue(
      rawValue,
      options,
      depth + 1,
      childModeForKey(key, mode),
    );
    if (sanitized !== undefined) out[key] = sanitized;
  }
  return out;
}

function sanitizeSection(value, mode, includeContent = false) {
  return sanitizeValue(value ?? {}, { includeContent }, 0, mode) ?? {};
}

export function sanitizeTraceData(value, { includeContent = false } = {}) {
  return sanitizeSection(value, "event", includeContent);
}

export function sanitizeTraceConfig(value, { includeContent = false } = {}) {
  return sanitizeSection(value, "config", includeContent);
}

function hexDigest(buffer) {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function sha256TraceText(value) {
  if (typeof value !== "string" || value.length === 0) return null;
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = await subtle.digest("SHA-256", textEncoder.encode(value));
  return `sha256:${hexDigest(digest)}`;
}

export async function hashTracePrompts(config) {
  if (!config || typeof config !== "object") return {};
  const entries = await Promise.all(
    [...PROMPT_FIELDS].map(async (field) => {
      const digest = await sha256TraceText(config[field]);
      return digest ? [field, digest] : null;
    }),
  );
  return Object.fromEntries(entries.filter(Boolean));
}

function copyEventsWithinReportLimit(report, events, maxBytes) {
  const bounded = { ...report, events: events.slice() };
  while (bounded.events.length > 0 && jsonByteLength(bounded) > maxBytes) {
    bounded.events.shift();
  }
  if (jsonByteLength(bounded) <= maxBytes) return bounded;

  // A deliberately content-rich header can itself exceed a caller's small
  // cap. Preserve diagnostic identity/config and remove optional content
  // before giving up on the requested byte ceiling.
  bounded.config = {
    requested: sanitizeTraceConfig(bounded.config?.requested),
    applied: sanitizeTraceConfig(bounded.config?.applied),
    ...(bounded.config?.prompt_hashes
      ? { prompt_hashes: bounded.config.prompt_hashes }
      : {}),
  };
  bounded.privacy = { ...bounded.privacy, content_included: false };
  if (jsonByteLength(bounded) <= maxBytes) return bounded;

  // Runtime strings are useful, but unlike the revision/build identifiers
  // they are not worth violating the caller's hard export bound. Keep a
  // compact identity before progressively falling back to the schema shell.
  bounded.runtime = Object.fromEntries(
    Object.entries(bounded.runtime ?? {}).map(([key, value]) => [
      key,
      typeof value === "string" ? value.slice(0, 96) : value,
    ]),
  );
  bounded.summary = {};
  delete bounded.config.prompt_hashes;
  if (jsonByteLength(bounded) <= maxBytes) return bounded;

  bounded.runtime = {};
  bounded.config = { requested: {}, applied: {} };
  bounded.session = {
    ...(Number.isFinite(bounded.session?.duration_ms)
      ? { duration_ms: bounded.session.duration_ms }
      : {}),
    ...(typeof bounded.session?.end_reason === "string"
      ? { end_reason: bounded.session.end_reason }
      : {}),
  };
  if (jsonByteLength(bounded) <= maxBytes) return bounded;

  // MIN_MAX_BYTES is intentionally larger than this schema-only envelope.
  // Retain its shape so importers never need a second fallback format.
  return {
    schema_version: bounded.schema_version,
    generated_at: bounded.generated_at,
    privacy: bounded.privacy,
    runtime: {},
    session: {},
    config: { requested: {}, applied: {} },
    summary: {},
    events: [],
  };
}

export function createSessionTrace({
  maxEvents = DEFAULT_MAX_EVENTS,
  maxBytes = DEFAULT_MAX_BYTES,
  clock = monotonicNow,
  wallClock = wallClockNow,
} = {}) {
  const eventLimit = Math.max(1, Math.floor(Number(maxEvents) || DEFAULT_MAX_EVENTS));
  const byteLimit = Math.max(MIN_MAX_BYTES, Math.floor(Number(maxBytes) || DEFAULT_MAX_BYTES));
  const startedAt = clock();
  const generatedAt = wallClock();
  let nextSeq = 1;
  let eventBytes = 0;
  let finishedAt = null;
  let runtime = {};
  let session = {};
  let requestedConfig = {};
  let appliedConfig = {};
  let summary = {};
  let requestedPromptHashes = Promise.resolve({});
  let appliedPromptHashes = Promise.resolve({});
  const events = [];

  const boundEvents = () => {
    while (events.length > eventLimit || eventBytes > byteLimit) {
      const removed = events.shift();
      eventBytes -= jsonByteLength(removed);
    }
  };

  const trace = {
    record(kind, data = {}, { source = "client", level = "info" } = {}) {
      const at = Math.max(0, clock() - startedAt);
      const event = {
        seq: nextSeq,
        t_ms: Math.round(at * 10) / 10,
        source: safeToken(source, "client"),
        kind: safeToken(kind, "event"),
        level: safeToken(level, "info"),
        data: sanitizeTraceData(data, { includeContent: true }),
      };
      nextSeq += 1;
      events.push(event);
      eventBytes += jsonByteLength(event);
      boundEvents();
      return event.seq;
    },

    setRuntime(value) {
      runtime = sanitizeSection(value, "runtime", false);
    },

    setSession(value) {
      session = {
        ...session,
        ...sanitizeSection(value, "session", false),
      };
    },

    setRequestedConfig(value) {
      requestedPromptHashes = hashTracePrompts(value);
      requestedConfig = sanitizeTraceConfig(value, { includeContent: true });
    },

    setAppliedConfig(value) {
      appliedPromptHashes = hashTracePrompts(value);
      appliedConfig = sanitizeTraceConfig(value, { includeContent: true });
    },

    setSummary(value) {
      summary = {
        ...summary,
        ...sanitizeSection(value, "summary", false),
      };
    },

    finish(endReason = "ended") {
      if (finishedAt !== null) return;
      finishedAt = clock();
      session = {
        ...session,
        duration_ms: Math.max(0, Math.round((finishedAt - startedAt) * 10) / 10),
        end_reason: scrubString(endReason),
      };
    },

    toReport({ includeContent = false } = {}) {
      const durationEnd = finishedAt ?? clock();
      const report = {
        schema_version: SESSION_TRACE_SCHEMA_VERSION,
        generated_at: generatedAt,
        privacy: {
          content_included: Boolean(includeContent),
          audio_included: false,
          images_included: false,
          network_identifiers_included: false,
          session_identifiers_included: false,
          secrets_included: false,
        },
        runtime: sanitizeSection(runtime, "runtime", false),
        session: {
          ...sanitizeSection(session, "session", false),
          duration_ms: Math.max(
            0,
            Math.round((durationEnd - startedAt) * 10) / 10,
          ),
        },
        config: {
          requested: sanitizeTraceConfig(requestedConfig, { includeContent }),
          applied: sanitizeTraceConfig(appliedConfig, { includeContent }),
        },
        summary: sanitizeSection(summary, "summary", false),
        events: events.map((event) => ({
          ...event,
          data: sanitizeTraceData(event.data, { includeContent }),
        })),
      };
      return copyEventsWithinReportLimit(report, report.events, byteLimit);
    },

    async toReportAsync({ includeContent = false, includePromptHashes = true } = {}) {
      const report = trace.toReport({ includeContent });
      if (!includePromptHashes) return report;
      const [requested, applied] = await Promise.all([
        requestedPromptHashes,
        appliedPromptHashes,
      ]);
      if (Object.keys(requested).length || Object.keys(applied).length) {
        report.config.prompt_hashes = { requested, applied };
        report.privacy.prompt_hashes_included = true;
      }
      return copyEventsWithinReportLimit(report, report.events, byteLimit);
    },

    clear() {
      events.splice(0, events.length);
      eventBytes = 0;
      nextSeq = 1;
    },

    get size() {
      return events.length;
    },
  };

  return trace;
}

export function serializeSessionTraceReport(report, { pretty = true } = {}) {
  return JSON.stringify(report, null, pretty ? 2 : 0);
}

export async function prepareSessionTraceExport(
  trace,
  {
    includeContent = false,
    includePromptHashes = true,
    pretty = false,
    filename = "personaplex-bug-report.json",
  } = {},
) {
  if (!trace || typeof trace.toReportAsync !== "function") {
    throw new TypeError("prepareSessionTraceExport requires a session trace");
  }
  const report = await trace.toReportAsync({ includeContent, includePromptHashes });
  const json = serializeSessionTraceReport(report, { pretty });
  return {
    filename: String(filename || "personaplex-bug-report.json").replace(
      /[^a-zA-Z0-9._-]+/g,
      "-",
    ),
    report,
    json,
    blob: new Blob([json], { type: "application/json" }),
  };
}

export async function downloadSessionTrace(trace, options = {}) {
  if (typeof document === "undefined" || typeof URL?.createObjectURL !== "function") {
    throw new Error("session trace download requires a browser document");
  }
  const prepared = await prepareSessionTraceExport(trace, options);
  const url = URL.createObjectURL(prepared.blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = prepared.filename;
  anchor.click();
  globalThis.setTimeout(() => URL.revokeObjectURL(url), 0);
  return prepared.report;
}
