# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for scripts/vllm/integration/smoke_mamba_state_contamination.py."""

import importlib.util
import pathlib
import sys

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "vllm"
    / "integration"
    / "smoke_mamba_state_contamination.py"
)
_spec = importlib.util.spec_from_file_location(
    "smoke_mamba_state_contamination", _SCRIPT
)
assert _spec is not None and _spec.loader is not None
smoke_mamba_state_contamination = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke_mamba_state_contamination)


def _tag_of(prompt: str) -> str:
    return "ALPHA" if "Inventory ledger ALPHA." in prompt else "BRAVO"


def test_default_cold_reference_passes_when_warm_matches_cold(monkeypatch):
    seen: set[str] = set()

    def fake_complete(url, model, prompt, max_tokens, timeout, headers=None):
        tag = _tag_of(prompt)
        cached = 4480 if tag in seen else 0
        seen.add(tag)
        tokens = [1, 2] if tag == "ALPHA" else [3, 4]
        return {
            "tokens": tokens,
            "text": tag,
            "prompt_tokens": 4754,
            "cached_tokens": cached,
        }

    monkeypatch.setattr(smoke_mamba_state_contamination, "complete", fake_complete)
    monkeypatch.setattr(
        sys,
        "argv",
        ["smoke_mamba_state_contamination.py", "--model", "m", "--rounds", "3"],
    )
    assert smoke_mamba_state_contamination.main() == 0


def test_warm_reference_tolerates_cold_vs_warm_shift_and_checks_cross_slot(
    monkeypatch,
):
    seen: set[str] = set()
    slot = {"last": None}
    cross_slot_calls = 0

    def fake_complete(url, model, prompt, max_tokens, timeout, headers=None):
        nonlocal cross_slot_calls
        tag = _tag_of(prompt)
        cached = 4480 if tag in seen else 0
        if cached > 0 and slot["last"] != tag:
            cross_slot_calls += 1
        seen.add(tag)
        slot["last"] = tag
        # Cold prefill differs from warm checkpoint-resumed prefill by one
        # near-tied greedy token, while all warm hits with proper state restore
        # produce identical tokens regardless of the previous slot occupant.
        if tag == "ALPHA":
            tokens = [10, 1248] if cached == 0 else [10, 1103]
        else:
            tokens = [20, 30]
        return {
            "tokens": tokens,
            "text": tag,
            "prompt_tokens": 4754,
            "cached_tokens": cached,
        }

    monkeypatch.setattr(smoke_mamba_state_contamination, "complete", fake_complete)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_mamba_state_contamination.py",
            "--model",
            "m",
            "--rounds",
            "4",
            "--warm-reference",
        ],
    )
    assert smoke_mamba_state_contamination.main() == 0
    assert cross_slot_calls == 6


def test_warm_reference_catches_cross_slot_contamination(monkeypatch):
    seen: set[str] = set()
    slot = {"last": None}

    def fake_complete(url, model, prompt, max_tokens, timeout, headers=None):
        tag = _tag_of(prompt)
        cached = 4480 if tag in seen else 0
        # Simulate missing Mamba state restore: a cache hit inherits whichever
        # prefix last wrote the recurrent slot instead of restoring tag's state.
        effective_state = tag if cached == 0 else slot["last"]
        seen.add(tag)
        slot["last"] = tag
        tokens = [1, 2] if effective_state == "ALPHA" else [9, 9]
        return {
            "tokens": tokens,
            "text": tag,
            "prompt_tokens": 4754,
            "cached_tokens": cached,
        }

    monkeypatch.setattr(smoke_mamba_state_contamination, "complete", fake_complete)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_mamba_state_contamination.py",
            "--model",
            "m",
            "--rounds",
            "3",
            "--warm-reference",
        ],
    )
    assert smoke_mamba_state_contamination.main() == 1


def test_warm_reference_fails_when_reference_repeat_misses_cache(monkeypatch, capsys):
    def fake_complete(url, model, prompt, max_tokens, timeout, headers=None):
        return {
            "tokens": [1, 2],
            "text": "x",
            "prompt_tokens": 4754,
            "cached_tokens": 0,
        }

    monkeypatch.setattr(smoke_mamba_state_contamination, "complete", fake_complete)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_mamba_state_contamination.py",
            "--model",
            "m",
            "--rounds",
            "2",
            "--warm-reference",
        ],
    )
    assert smoke_mamba_state_contamination.main() == 1
    assert "STATE_CONTAMINATION_NO_HITS 1" in capsys.readouterr().out
