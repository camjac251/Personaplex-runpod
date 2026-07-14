"""Checks for startup-time PersonaPlex checkpoint selection."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[2]
RESOLVER = ROOT / "docker" / "model-env.sh"
BOOTSTRAP = ROOT / "start.sh"
RL_REPO = "kyutai/personaplex-rl-seamless"
RL_REVISION = "3fa800309a4b743a8a6d764253eb45def0334afc"
BASE_REPO = "nvidia/personaplex-7b-v1"
BASE_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309f"


def _resolve(**values: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {
            "PERSONAPLEX_MODEL",
            "PERSONAPLEX_HF_REPO",
            "PERSONAPLEX_HF_REVISION",
        }
    }
    env.update(values)
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                f'source "{RESOLVER}"; '
                "personaplex_resolve_model || exit $?; "
                'printf "%s\\n%s\\n" "$PERSONAPLEX_SELECTED_HF_REPO" '
                '"$PERSONAPLEX_SELECTED_HF_REVISION"'
            ),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _resolve_runpod_snapshot_policy(
    value: str | None = None,
) -> subprocess.CompletedProcess[str]:
    assignment = next(
        (
            line
            for line in BOOTSTRAP.read_text(encoding="utf-8").splitlines()
            if line.startswith("export PERSONAPLEX_PERIODIC_SNAPSHOTS=")
        ),
        None,
    )
    assert assignment is not None, "RunPod bootstrap must set the snapshot policy"
    env = {
        key: current
        for key, current in os.environ.items()
        if key != "PERSONAPLEX_PERIODIC_SNAPSHOTS"
    }
    if value is not None:
        env["PERSONAPLEX_PERIODIC_SNAPSHOTS"] = value
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                f"{assignment}; "
                'printf "%s\\n" "$PERSONAPLEX_PERIODIC_SNAPSHOTS"'
            ),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_default_model_is_pinned_rl_seamless() -> None:
    result = _resolve()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [RL_REPO, RL_REVISION]


def test_runpod_disables_periodic_snapshots_by_default() -> None:
    result = _resolve_runpod_snapshot_policy()
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["0"]


def test_runpod_preserves_periodic_snapshot_override() -> None:
    result = _resolve_runpod_snapshot_policy("1")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["1"]


def test_base_alias_selects_pinned_nvidia_model() -> None:
    result = _resolve(PERSONAPLEX_MODEL="base")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [BASE_REPO, BASE_REVISION]


def test_known_repo_override_gets_its_matching_pin() -> None:
    result = _resolve(PERSONAPLEX_HF_REPO=BASE_REPO)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [BASE_REPO, BASE_REVISION]


def test_legacy_base_revision_does_not_override_new_default() -> None:
    result = _resolve(PERSONAPLEX_HF_REVISION=BASE_REVISION)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [RL_REPO, RL_REVISION]
    assert "ignoring legacy NVIDIA revision override" in result.stderr


def test_known_revision_cannot_be_crossed_with_explicit_model() -> None:
    rl_with_base_revision = _resolve(
        PERSONAPLEX_MODEL="rl-seamless",
        PERSONAPLEX_HF_REVISION=BASE_REVISION,
    )
    assert rl_with_base_revision.returncode == 64

    base_with_rl_revision = _resolve(
        PERSONAPLEX_MODEL="base",
        PERSONAPLEX_HF_REVISION=RL_REVISION,
    )
    assert base_with_rl_revision.returncode == 64


def test_custom_repo_requires_explicit_revision() -> None:
    result = _resolve(PERSONAPLEX_HF_REPO="example/custom")
    assert result.returncode == 64
    assert "requires PERSONAPLEX_HF_REVISION" in result.stderr


def test_custom_repo_and_revision_are_forwarded() -> None:
    revision = "e" * 40
    result = _resolve(
        PERSONAPLEX_HF_REPO="example/custom",
        PERSONAPLEX_HF_REVISION=revision,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["example/custom", revision]
