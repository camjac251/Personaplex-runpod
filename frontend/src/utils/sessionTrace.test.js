import { describe, expect, test } from "bun:test";

import {
  createSessionTrace,
  hashTracePrompts,
  prepareSessionTraceExport,
  sanitizeTraceData,
  sha256TraceText,
} from "./sessionTrace.js";

const SECRET = {
  api: "sk-secret-api-value",
  audio: "raw-audio-sentinel",
  caption: "private-camera-caption",
  candidate: "candidate:1 1 udp 1 192.0.2.44 8998 typ host",
  device: "private-device-id",
  image: "data:image/jpeg;base64,private-image-sentinel",
  prompt: "private-system-prompt",
  sdp: "v=0 private-sdp-sentinel",
  session: "private-session-id",
  transcript: "private-user-transcript",
  userAgent: "private-browser-fingerprint",
};

function deterministicClock() {
  let value = 100;
  return () => {
    value += 10;
    return value;
  };
}

describe("session trace privacy", () => {
  test("the default report excludes content, media, network identifiers, and secrets", async () => {
    const trace = createSessionTrace({
      clock: deterministicClock(),
      wallClock: () => "2026-07-13T00:00:00.000Z",
    });
    trace.setRuntime({
      server_build: "abc123",
      model_repo: "kyutai/personaplex-rl-seamless",
      model_revision: "revision-1",
      gpu_name: "RTX Test",
      userAgent: SECRET.userAgent,
    });
    trace.setSession({
      session_id: SECRET.session,
      output_device_id: SECRET.device,
      turn_handling: "native",
      echo_cancellation: true,
    });
    trace.setRequestedConfig({
      text_prompt: SECRET.prompt,
      vision_prompt: "private-vision-prompt",
      audio_temperature: 0.8,
      text_topk: 25,
      seed: 42,
    });
    trace.record(
      "vision.result",
      {
        caption: SECRET.caption,
        transcript: SECRET.transcript,
        frame: SECRET.image,
        image: SECRET.image,
        pcm: SECRET.audio,
        sdp: SECRET.sdp,
        candidate: SECRET.candidate,
        session_id: SECRET.session,
        api_key: SECRET.api,
        latency_ms: 123,
        frame_id: "frame-7",
        data: {
          text: "private-context-text",
          rtf: 0.42,
        },
      },
      { source: "server", level: "ok" },
    );

    const report = await trace.toReportAsync();
    const serialized = JSON.stringify(report);
    for (const secret of Object.values(SECRET)) {
      expect(serialized).not.toContain(secret);
    }
    expect(serialized).not.toContain("private-vision-prompt");
    expect(serialized).not.toContain("private-context-text");
    expect(report.runtime.server_build).toBe("abc123");
    expect(report.session.turn_handling).toBe("native");
    expect(report.config.requested.audio_temperature).toBe(0.8);
    expect(report.events[0].data.latency_ms).toBe(123);
    expect(report.events[0].data.data.rtf).toBe(0.42);
    expect(report.config.prompt_hashes.requested.text_prompt).toMatch(
      /^sha256:[a-f0-9]{64}$/,
    );
    expect(report.privacy.content_included).toBe(false);
    expect(report.privacy.audio_included).toBe(false);
    expect(report.privacy.images_included).toBe(false);
  });

  test("content inclusion is explicit and still never includes hard-private fields", async () => {
    const trace = createSessionTrace({ clock: deterministicClock() });
    trace.setRequestedConfig({
      text_prompt: SECRET.prompt,
      audio_temperature: 0.7,
      voice_prompt: "uploaded-private-name.wav",
    });
    trace.record("turn.closed", {
      transcript: SECRET.transcript,
      caption: SECRET.caption,
      frame: SECRET.image,
      pcm: SECRET.audio,
      sdp: SECRET.sdp,
      session_id: SECRET.session,
      words: 3,
    });

    const report = await trace.toReportAsync({ includeContent: true });
    const serialized = JSON.stringify(report);
    expect(report.config.requested.text_prompt).toBe(SECRET.prompt);
    expect(report.events[0].data.transcript).toBe(SECRET.transcript);
    expect(report.events[0].data.caption).toBe(SECRET.caption);
    expect(serialized).not.toContain(SECRET.image);
    expect(serialized).not.toContain(SECRET.audio);
    expect(serialized).not.toContain(SECRET.sdp);
    expect(serialized).not.toContain(SECRET.session);
    expect(serialized).not.toContain("uploaded-private-name.wav");
    expect(report.privacy.content_included).toBe(true);
  });

  test("unknown fields are denied and diagnostic strings are scrubbed", () => {
    const sanitized = sanitizeTraceData({
      arbitrary: "must-not-pass",
      reason:
        "failed at /workspace/private/file.py for user@example.com via https://private.example and 192.0.2.4 token=abcd",
      rtf: 0.5,
    });
    const serialized = JSON.stringify(sanitized);
    expect(serialized).not.toContain("must-not-pass");
    expect(serialized).not.toContain("/workspace/private/file.py");
    expect(serialized).not.toContain("user@example.com");
    expect(serialized).not.toContain("https://private.example");
    expect(serialized).not.toContain("192.0.2.4");
    expect(serialized).not.toContain("token=abcd");
    expect(sanitized.rtf).toBe(0.5);
  });

  test("free-form reasons are reduced to controlled tokens", () => {
    const privateReason = "Alice described her unreleased project during the call";
    const sanitized = sanitizeTraceData({ reason: privateReason });
    expect(JSON.stringify(sanitized)).not.toContain(privateReason);
    expect(sanitized.reason).toBe("other");
    expect(sanitizeTraceData({ reason: "action_timeout" }).reason).toBe(
      "action_timeout",
    );
    expect(sanitizeTraceData({ reason: "opaqueProjectPassword" }).reason).toBe(
      "other",
    );
  });

  test("typed fields cannot be used to smuggle string secrets", async () => {
    const trace = createSessionTrace({ clock: deterministicClock() });
    trace.setRequestedConfig({
      audio_temperature: SECRET.api,
      vision_feed_model: SECRET.api,
      voice_fingerprint: SECRET.api,
    });
    trace.record("stat", {
      rtf: SECRET.api,
      enabled: SECRET.api,
      error_code: SECRET.api,
    });
    const serialized = JSON.stringify(await trace.toReportAsync());
    expect(serialized).not.toContain(SECRET.api);
  });
});

describe("session trace bounds and exports", () => {
  test("transport health survives sanitizing and finish is idempotent", () => {
    let now = 0;
    const trace = createSessionTrace({ clock: () => now });
    now = 100;
    trace.record("stat", {
      pcm_queue_depth: 2,
      pcm_queue_capacity: 10,
      outbound_buffer_ms: 40,
      outbound_drop_events: 1,
      outbound_dropped_ms: 20,
    });
    trace.finish("ended");
    now = 500;
    trace.finish("later");

    const report = trace.toReport();
    expect(report.session.duration_ms).toBe(100);
    expect(report.session.end_reason).toBe("ended");
    expect(report.events[0].data).toMatchObject({
      pcm_queue_depth: 2,
      pcm_queue_capacity: 10,
      outbound_buffer_ms: 40,
      outbound_drop_events: 1,
      outbound_dropped_ms: 20,
    });
  });

  test("the event ring evicts oldest entries by count", () => {
    const trace = createSessionTrace({ maxEvents: 3, clock: deterministicClock() });
    for (let seq = 1; seq <= 5; seq += 1) {
      trace.record("stat", { seq, rtf: seq / 10 });
    }
    const report = trace.toReport();
    expect(trace.size).toBe(3);
    expect(report.events.map((event) => event.seq)).toEqual([3, 4, 5]);
    expect(report.events.map((event) => event.data.seq)).toEqual([3, 4, 5]);
  });

  test("the serialized report stays under its byte budget", async () => {
    const maxBytes = 4096;
    const trace = createSessionTrace({
      maxEvents: 100,
      maxBytes,
      clock: deterministicClock(),
    });
    for (let seq = 0; seq < 100; seq += 1) {
      trace.record("error", {
        seq,
        message: `${"diagnostic ".repeat(60)}${seq}`,
        error_code: `error-${seq}`,
      });
    }
    const report = await trace.toReportAsync({ includeContent: true });
    const bytes = new TextEncoder().encode(JSON.stringify(report)).byteLength;
    expect(bytes).toBeLessThanOrEqual(maxBytes);
    expect(report.events.length).toBeLessThan(100);
    expect(report.events.at(-1).data.seq).toBe(99);
  });

  test("a single oversized event is evicted from the in-memory ring", () => {
    const trace = createSessionTrace({
      maxBytes: 1024,
      clock: deterministicClock(),
    });
    trace.record("diagnostic", { message: "private detail ".repeat(1000) });
    expect(trace.size).toBe(0);
    expect(trace.toReport({ includeContent: true }).events).toEqual([]);
  });

  test("the hard byte budget also bounds oversized metadata", async () => {
    const maxBytes = 1024;
    const trace = createSessionTrace({ maxBytes, clock: deterministicClock() });
    trace.setRuntime({
      model_repo: "model-".repeat(200),
      model_revision: "revision-".repeat(200),
      model_license: "license-".repeat(200),
      server_build: "build-".repeat(200),
      gpu_name: "gpu-".repeat(200),
    });
    trace.setRequestedConfig({
      text_prompt: "prompt ".repeat(10_000),
      vision_prompt: "vision ".repeat(10_000),
      audio_temperature: 0.8,
    });
    const report = await trace.toReportAsync({ includeContent: true });
    const bytes = new TextEncoder().encode(JSON.stringify(report)).byteLength;
    expect(bytes).toBeLessThanOrEqual(maxBytes);
    expect(report.schema_version).toBe(1);
    expect(report.privacy.audio_included).toBe(false);
  });

  test("hashes are deterministic without retaining prompt text", async () => {
    expect(await sha256TraceText("abc")).toBe(
      "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    const hashes = await hashTracePrompts({
      text_prompt: SECRET.prompt,
      vision_prompt: "vision",
      unrelated: "ignored",
    });
    expect(Object.keys(hashes).sort()).toEqual(["text_prompt", "vision_prompt"]);
    expect(JSON.stringify(hashes)).not.toContain(SECRET.prompt);
  });

  test("the export helper returns a JSON blob with a safe filename", async () => {
    const trace = createSessionTrace({ clock: deterministicClock() });
    trace.record("ready", { resumed: false });
    const prepared = await prepareSessionTraceExport(trace, {
      filename: "PersonaPlex report / private.json",
    });
    expect(prepared.filename).toBe("PersonaPlex-report-private.json");
    expect(prepared.blob.type).toStartWith("application/json");
    expect(prepared.json).toContain('"schema_version":1');
    expect(JSON.parse(await prepared.blob.text())).toEqual(prepared.report);
  });

  test("the default export blob stays within the trace byte budget", async () => {
    const maxBytes = 2048;
    const trace = createSessionTrace({
      maxBytes,
      maxEvents: 100,
      clock: deterministicClock(),
    });
    for (let seq = 0; seq < 100; seq += 1) {
      trace.record("error", {
        seq,
        message: `diagnostic ${seq} ${"detail ".repeat(80)}`,
      });
    }
    const prepared = await prepareSessionTraceExport(trace, {
      includeContent: true,
    });
    expect(prepared.blob.size).toBeLessThanOrEqual(maxBytes);
  });
});
