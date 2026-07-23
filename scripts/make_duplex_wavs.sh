#!/usr/bin/env bash
# Synthesize every duplex regression scenario WAV (mono PCM16 48 kHz) at the
# exact timings the manifests in moshi/tests/fixtures/duplex/ document:
#
#   turn_taking.wav          one request ending at exactly 3000 ms
#   manual_stop.wav          long-answer request ending at exactly 3000 ms
#   max_turn_cap.wav         enumeration request ending at exactly 3000 ms
#   rapid_turns.wav          three turns ending at 1500, 4500, and 7500 ms
#   pause_mid_utterance.wav  speech 0-1800 ms, silence 1800-2600 ms,
#                            continuation 2600-4200 ms
#   long_session_soak.wav    twelve requests ending every 50 s through 600 s
#
# Usage: scripts/make_duplex_wavs.sh <output-dir>
#
# Requires espeak-ng, ffmpeg, and python3 (stdlib wave only). Every clip is
# right-aligned so speech ends exactly on its manifest boundary, and the
# script aborts with a clear error when a synthesized clip cannot fit inside
# its slot.
#
# espeak-ng keeps these fixtures reproducible, but synthetic speech sits far
# outside the conversational distribution PersonaPlex trained on. Human
# recordings that keep the same slot timings are drop-in replacements, and
# the model responds to them far more naturally than to espeak output.
set -euo pipefail

usage() {
  echo "usage: $0 <output-dir>" >&2
  exit 2
}

[ $# -eq 1 ] || usage
out=$1

for tool in espeak-ng ffmpeg python3; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool not found: $tool" >&2
    exit 1
  fi
done

mkdir -p "$out"
tmp=$(mktemp -d "$out/tmp.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

say() { # say <text> <out.wav> [words-per-minute]
  espeak-ng -v en-us -s "${3:-200}" -w "$tmp/raw.wav" "$1"
  ffmpeg -loglevel error -y -i "$tmp/raw.wav" -ar 48000 -ac 1 -sample_fmt s16 "$2"
}

# place <out.wav> <total_ms> [<in.wav> <slot_start_ms> <slot_end_ms>]...
# Right-align each clip so its speech ends exactly at slot_end_ms; abort when
# a clip is longer than its slot or a slot extends past the buffer.
place() {
  python3 - "$@" <<'PY'
import sys
import wave

RATE = 48_000
args = sys.argv[1:]
out_path, total_ms = args[0], int(args[1])
triples = args[2:]
if total_ms <= 0:
    sys.exit(f"place: total_ms must be positive, got {total_ms}")
if not triples or len(triples) % 3:
    sys.exit("place: arguments after <total_ms> must be "
             "<in.wav> <slot_start_ms> <slot_end_ms> triples")
buf = bytearray(2 * (total_ms * RATE // 1000))
for index in range(0, len(triples), 3):
    path = triples[index]
    slot_start_ms = int(triples[index + 1])
    slot_end_ms = int(triples[index + 2])
    if not 0 <= slot_start_ms < slot_end_ms:
        sys.exit(f"place: bad slot [{slot_start_ms}, {slot_end_ms}] for {path}")
    if slot_end_ms > total_ms:
        sys.exit(
            f"place: slot end {slot_end_ms} ms extends past the "
            f"{total_ms} ms buffer for {path}"
        )
    with wave.open(path, "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != RATE
        ):
            sys.exit(f"place: {path} is not mono PCM16 48 kHz")
        data = handle.readframes(handle.getnframes())
    duration_ms = len(data) / 2 * 1000.0 / RATE
    slot_ms = slot_end_ms - slot_start_ms
    if duration_ms > slot_ms:
        sys.exit(
            f"place: {path} runs {duration_ms:.0f} ms, longer than its "
            f"{slot_ms} ms slot ending at {slot_end_ms} ms"
        )
    end = 2 * (slot_end_ms * RATE // 1000)
    buf[end - len(data) : end] = data
with wave.open(out_path, "wb") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(RATE)
    handle.writeframes(bytes(buf))
PY
}

# turn_taking: one request ending at exactly 3000 ms.
say "Tell me a quick fact about the ocean." "$tmp/tt.wav"
place "$out/turn_taking.wav" 3000 "$tmp/tt.wav" 0 3000

# manual_stop: request for a long answer ending at exactly 3000 ms.
say "Explain the whole history of computers." "$tmp/ms.wav"
place "$out/manual_stop.wav" 3000 "$tmp/ms.wav" 0 3000

# max_turn_cap: request for a long enumeration ending at exactly 3000 ms.
say "List every animal you can think of." "$tmp/mc.wav"
place "$out/max_turn_cap.wav" 3000 "$tmp/mc.wav" 0 3000

# rapid_turns: three turns ending at 1500, 4500, and 7500 ms.
say "Hi, how are you?" "$tmp/r1.wav" 210
say "What's your favorite color?" "$tmp/r2.wav" 210
say "Thanks, goodbye now." "$tmp/r3.wav" 210
place "$out/rapid_turns.wav" 7500 \
  "$tmp/r1.wav" 0 1500 \
  "$tmp/r2.wav" 1500 4500 \
  "$tmp/r3.wav" 4500 7500

# pause_mid_utterance: speech 0-1800 ms, silence 1800-2600 ms, continuation
# 2600-4200 ms.
say "Could you please tell me" "$tmp/p1.wav" 190
say "what the weather is like?" "$tmp/p2.wav" 190
place "$out/pause_mid_utterance.wav" 4200 \
  "$tmp/p1.wav" 0 1800 \
  "$tmp/p2.wav" 2600 4200

# long_session_soak: twelve short requests, each ending exactly on a 50 s
# boundary, through 600 s.
soak_texts=(
  "Hey there, how is your day going so far?"
  "Tell me one interesting fact about space."
  "What is your favorite season, and why?"
  "Can you recommend a good book to read?"
  "Describe the sound of rain in one sentence."
  "What would you cook for a quick dinner?"
  "Give me a short tip for staying focused."
  "What is a good way to start the morning?"
  "Name a city you would love to visit."
  "How would you describe the color blue?"
  "Share a quick thought about music."
  "We are almost done. Any final thoughts?"
)
soak_args=()
for i in "${!soak_texts[@]}"; do
  clip="$tmp/soak_$i.wav"
  say "${soak_texts[$i]}" "$clip"
  soak_args+=("$clip" "$((i * 50000))" "$(((i + 1) * 50000))")
done
place "$out/long_session_soak.wav" 600000 "${soak_args[@]}"

python3 - "$out"/*.wav <<'PY'
import sys
import wave

for path in sys.argv[1:]:
    with wave.open(path, "rb") as handle:
        print(f"{path} {handle.getnframes() / handle.getframerate():.3f}s")
PY
