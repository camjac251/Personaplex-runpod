"""Bounds checks for the control-channel sampling clamps.

Run directly: ``uv run python moshi/tests/test_session_config.py``.
No pytest dependency to keep the project deps lean; assertions raise.
"""

from __future__ import annotations

import sys

# Allow running this script from inside the repo without installing.
sys.path.insert(0, "moshi")

from moshi.rtc_session import (  # noqa: E402
    AUDIO_TOPK_MAX,
    AUDIO_TOPK_MIN,
    CAPTION_CFG_GAMMA_MAX,
    CAPTION_CFG_GAMMA_MIN,
    INJECT_SILENCE_RMS_MAX,
    INJECT_SILENCE_RMS_MIN,
    INJECT_SILENCE_STREAK_MAX,
    INJECT_SILENCE_STREAK_MIN,
    MAX_TURN_TEXT_TOKENS_MAX,
    MAX_TURN_TEXT_TOKENS_MIN,
    PADDING_BONUS_MAX,
    PADDING_BONUS_MIN,
    REPETITION_PENALTY_CONTEXT_MAX,
    REPETITION_PENALTY_CONTEXT_MIN,
    REPETITION_PENALTY_MAX,
    REPETITION_PENALTY_MIN,
    SEED_MAX,
    SEED_MIN,
    SEED_RANDOM,
    SESSION_TIMEOUT_SEC_MAX,
    SESSION_TIMEOUT_SEC_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    TEXT_MIN_P_MAX,
    TEXT_MIN_P_MIN,
    TEXT_TOPK_MAX,
    TEXT_TOPK_MIN,
    VISION_COST_LIMIT_USD_MAX,
    VISION_COST_LIMIT_USD_MIN,
    VISION_COST_PER_CALL_USD_MAX,
    VISION_COST_PER_CALL_USD_MIN,
    SessionConfig,
    clamp_audio_topk,
    clamp_caption_cfg_gamma,
    clamp_inject_silence_rms,
    clamp_inject_silence_streak,
    clamp_max_turn_text_tokens,
    clamp_padding_bonus,
    clamp_repetition_penalty,
    clamp_repetition_penalty_context,
    clamp_seed,
    clamp_session_timeout_sec,
    clamp_temperature,
    clamp_text_min_p,
    clamp_text_topk,
    clamp_vision_cost_limit_usd,
    clamp_vision_cost_per_call_usd,
    parse_session_config,
)


def test_in_range_values_pass_through() -> None:
    assert clamp_temperature(0.7) == 0.7
    assert clamp_temperature(1) == 1.0
    assert clamp_temperature("0.7") == 0.7
    assert clamp_temperature(TEMPERATURE_MIN) == TEMPERATURE_MIN
    assert clamp_temperature(TEMPERATURE_MAX) == TEMPERATURE_MAX


def test_out_of_range_values_clamp_to_bounds() -> None:
    assert clamp_temperature(0.0) == TEMPERATURE_MIN
    assert clamp_temperature(-5.0) == TEMPERATURE_MIN
    assert clamp_temperature(100.0) == TEMPERATURE_MAX


def test_non_finite_values_are_rejected() -> None:
    # float() accepts "nan"/"inf" strings and json.loads accepts bare
    # NaN/Infinity literals, so the clamp is the last line of defense.
    for bad in ("nan", "inf", "-inf", float("nan"), float("inf"), float("-inf")):
        try:
            clamp_temperature(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_defaults_are_within_clamp_bounds() -> None:
    defaults = SessionConfig()
    assert clamp_temperature(defaults.audio_temperature) == defaults.audio_temperature
    assert clamp_temperature(defaults.text_temperature) == defaults.text_temperature
    assert (
        clamp_inject_silence_rms(defaults.inject_silence_rms)
        == defaults.inject_silence_rms
    )
    assert (
        clamp_inject_silence_streak(defaults.inject_silence_streak)
        == defaults.inject_silence_streak
    )
    assert defaults.vision_feed_model is False
    assert defaults.vision_ground_user_turns is False
    assert defaults.vision_prompt_replace is False


def test_defaults_match_stable_conversation_tuning() -> None:
    defaults = SessionConfig()
    assert defaults.text_temperature == 0.7
    assert defaults.text_topk == 25
    assert defaults.text_min_p == 0.0
    assert defaults.semantic_temp_cap == 0.7
    assert defaults.audio_temperature == 0.8
    assert defaults.audio_topk == 250
    assert defaults.repetition_penalty == 1.0
    assert defaults.repetition_penalty_context == 64
    assert defaults.padding_bonus == 0.0
    assert defaults.max_turn_text_tokens == 120


def test_inject_silence_clamps() -> None:
    # In-range passes through; out-of-range clamps to bounds.
    assert clamp_inject_silence_rms(0.01) == 0.01
    assert clamp_inject_silence_rms(0.0) == INJECT_SILENCE_RMS_MIN
    assert clamp_inject_silence_rms(1.0) == INJECT_SILENCE_RMS_MAX
    # Streak parses through float first, so a float string and an int both work.
    assert clamp_inject_silence_streak(6) == 6
    assert clamp_inject_silence_streak("8") == 8
    assert clamp_inject_silence_streak(0) == INJECT_SILENCE_STREAK_MIN
    assert clamp_inject_silence_streak(999) == INJECT_SILENCE_STREAK_MAX
    # Non-finite is rejected on both, including bare inf that would make
    # int(inf) raise OverflowError instead of the caught ValueError.
    for bad in ("nan", "inf", float("nan"), float("inf")):
        for fn in (clamp_inject_silence_rms, clamp_inject_silence_streak):
            try:
                fn(bad)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {fn.__name__}({bad!r})")


def test_all_numeric_hard_bounds() -> None:
    assert clamp_text_topk(-1) == TEXT_TOPK_MIN
    assert clamp_text_topk(10_000) == TEXT_TOPK_MAX
    assert clamp_text_min_p(-1) == TEXT_MIN_P_MIN
    assert clamp_text_min_p(2) == TEXT_MIN_P_MAX
    assert clamp_audio_topk(-1) == AUDIO_TOPK_MIN
    assert clamp_audio_topk(10_000) == AUDIO_TOPK_MAX
    assert clamp_repetition_penalty(-1) == REPETITION_PENALTY_MIN
    assert clamp_repetition_penalty(10) == REPETITION_PENALTY_MAX
    assert clamp_repetition_penalty_context(-1) == REPETITION_PENALTY_CONTEXT_MIN
    assert clamp_repetition_penalty_context(10_000) == REPETITION_PENALTY_CONTEXT_MAX
    assert clamp_padding_bonus(-1) == PADDING_BONUS_MIN
    assert clamp_padding_bonus(10) == PADDING_BONUS_MAX
    assert clamp_max_turn_text_tokens(-1) == MAX_TURN_TEXT_TOKENS_MIN
    assert clamp_max_turn_text_tokens(10_000) == MAX_TURN_TEXT_TOKENS_MAX
    assert clamp_session_timeout_sec(-1) == SESSION_TIMEOUT_SEC_MIN
    assert clamp_session_timeout_sec(10_000) == SESSION_TIMEOUT_SEC_MAX
    assert clamp_vision_cost_limit_usd(-1) == VISION_COST_LIMIT_USD_MIN
    assert clamp_vision_cost_limit_usd(100) == VISION_COST_LIMIT_USD_MAX
    assert clamp_vision_cost_per_call_usd(-1) == VISION_COST_PER_CALL_USD_MIN
    assert clamp_vision_cost_per_call_usd(100) == VISION_COST_PER_CALL_USD_MAX
    assert clamp_caption_cfg_gamma(0) == CAPTION_CFG_GAMMA_MIN
    assert clamp_caption_cfg_gamma(99) == CAPTION_CFG_GAMMA_MAX
    assert clamp_caption_cfg_gamma(2.5) == 2.5
    assert clamp_seed(None) is None
    assert clamp_seed(SEED_RANDOM) == SEED_RANDOM
    assert clamp_seed(-2) == SEED_MIN
    assert clamp_seed(SEED_MAX + 1) == SEED_MAX


def test_parse_session_config_defaults() -> None:
    assert parse_session_config({}) == SessionConfig()


def test_parse_session_config_clamps_finite_values() -> None:
    cfg = parse_session_config(
        {
            "audio_temperature": 0,
            "text_temperature": 99,
            "semantic_temp_cap": 99,
            "text_topk": -4,
            "audio_topk": 9999,
            "repetition_penalty": -3,
            "repetition_penalty_context": 9999,
            "padding_bonus": 99,
            "max_turn_text_tokens": -1,
            "session_timeout_sec": 99999,
            "vision_cost_limit_usd": -1,
            "vision_cost_per_call_usd": 99,
            "inject_silence_rms": 0,
            "inject_silence_streak": 999,
            "seed": SEED_MAX + 10,
        }
    )
    assert cfg.audio_temperature == TEMPERATURE_MIN
    assert cfg.text_temperature == TEMPERATURE_MAX
    assert cfg.semantic_temp_cap == TEMPERATURE_MAX
    assert cfg.text_topk == TEXT_TOPK_MIN
    assert cfg.audio_topk == AUDIO_TOPK_MAX
    assert cfg.repetition_penalty == REPETITION_PENALTY_MIN
    assert cfg.repetition_penalty_context == REPETITION_PENALTY_CONTEXT_MAX
    assert cfg.padding_bonus == PADDING_BONUS_MAX
    assert cfg.max_turn_text_tokens == MAX_TURN_TEXT_TOKENS_MIN
    assert cfg.session_timeout_sec == SESSION_TIMEOUT_SEC_MAX
    assert cfg.vision_cost_limit_usd == VISION_COST_LIMIT_USD_MIN
    assert cfg.vision_cost_per_call_usd == VISION_COST_PER_CALL_USD_MAX
    assert cfg.inject_silence_rms == INJECT_SILENCE_RMS_MIN
    assert cfg.inject_silence_streak == INJECT_SILENCE_STREAK_MAX
    assert cfg.seed == SEED_MAX


def test_parse_session_config_rejects_non_finite_values() -> None:
    float_fields = (
        "voice_blend_mix",
        "clone_strength",
        "audio_temperature",
        "text_temperature",
        "repetition_penalty",
        "padding_bonus",
        "text_min_p",
        "semantic_temp_cap",
        "vision_cost_limit_usd",
        "vision_cost_per_call_usd",
        "inject_silence_rms",
        "caption_cfg_gamma",
    )
    int_fields = (
        "seed",
        "text_topk",
        "audio_topk",
        "repetition_penalty_context",
        "max_turn_text_tokens",
        "session_timeout_sec",
        "inject_silence_streak",
    )
    for field in float_fields + int_fields:
        for bad in (float("nan"), float("inf"), "1e309"):
            try:
                parse_session_config({field: bad})
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {field}={bad!r}")


def test_parse_session_config_rejects_malformed_numeric_types() -> None:
    for field in (
        "seed",
        "text_topk",
        "repetition_penalty",
        "session_timeout_sec",
        "vision_cost_limit_usd",
    ):
        for bad in (None, [], {}, True):
            if field == "seed" and bad is None:
                continue
            try:
                parse_session_config({field: bad})
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {field}={bad!r}")


def test_parse_session_config_preserves_valid_wire_values() -> None:
    cfg = parse_session_config(
        {
            "voice_prompt": "voice.pt",
            "text_prompt": "You enjoy talking.",
            "vision_feed_model": True,
            "vision_prompt_replace": True,
            "seed": SEED_RANDOM,
            "text_topk": "64",
            "text_min_p": "0.08",
            "semantic_temp_cap": "0.85",
            "audio_topk": "420",
            "repetition_penalty": "1.08",
            "repetition_penalty_context": "128",
            "padding_bonus": "0.5",
            "max_turn_text_tokens": "240",
            "session_timeout_sec": "3600",
            "vision_cost_limit_usd": "2.5",
            "vision_cost_per_call_usd": "0.00012",
        }
    )
    assert cfg.voice_prompt == "voice.pt"
    assert cfg.text_prompt == "You enjoy talking."
    assert cfg.vision_feed_model is True
    assert cfg.vision_prompt_replace is True
    assert cfg.seed == SEED_RANDOM
    assert cfg.text_topk == 64
    assert cfg.text_min_p == 0.08
    assert cfg.semantic_temp_cap == 0.85
    assert cfg.audio_topk == 420
    assert cfg.repetition_penalty == 1.08
    assert cfg.repetition_penalty_context == 128
    assert cfg.padding_bonus == 0.5
    assert cfg.max_turn_text_tokens == 240
    assert cfg.session_timeout_sec == 3600
    assert cfg.vision_cost_limit_usd == 2.5
    assert cfg.vision_cost_per_call_usd == 0.00012


if __name__ == "__main__":
    print("test_in_range_values_pass_through ...")
    test_in_range_values_pass_through()
    print("  ok")
    print("test_out_of_range_values_clamp_to_bounds ...")
    test_out_of_range_values_clamp_to_bounds()
    print("  ok")
    print("test_non_finite_values_are_rejected ...")
    test_non_finite_values_are_rejected()
    print("  ok")
    print("test_defaults_are_within_clamp_bounds ...")
    test_defaults_are_within_clamp_bounds()
    print("  ok")
    print("test_defaults_match_stable_conversation_tuning ...")
    test_defaults_match_stable_conversation_tuning()
    print("  ok")
    print("test_inject_silence_clamps ...")
    test_inject_silence_clamps()
    print("  ok")
    print("test_all_numeric_hard_bounds ...")
    test_all_numeric_hard_bounds()
    print("  ok")
    print("test_parse_session_config_defaults ...")
    test_parse_session_config_defaults()
    print("  ok")
    print("test_parse_session_config_clamps_finite_values ...")
    test_parse_session_config_clamps_finite_values()
    print("  ok")
    print("test_parse_session_config_rejects_non_finite_values ...")
    test_parse_session_config_rejects_non_finite_values()
    print("  ok")
    print("test_parse_session_config_rejects_malformed_numeric_types ...")
    test_parse_session_config_rejects_malformed_numeric_types()
    print("  ok")
    print("test_parse_session_config_preserves_valid_wire_values ...")
    test_parse_session_config_preserves_valid_wire_values()
    print("  ok")
    print("all session config tests passed")
