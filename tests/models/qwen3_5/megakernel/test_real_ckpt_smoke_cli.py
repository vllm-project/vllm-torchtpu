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
"""The real-checkpoint smoke's command line: its settings are flags, not
environment variables."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

import pytest

SMOKE_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "src/vllm_torchtpu/models/qwen3_5/megakernel/serving/run_real_ckpt_smoke.py"
)


def _load_smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_real_ckpt_smoke", SMOKE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_real_ckpt_smoke"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_defaults_serve_both_modes_from_the_hub_cache() -> None:
    args = _load_smoke().parse_args([])
    assert (args.mode, args.ckpt, args.megakernel, args.port) == (
        "both",
        None,
        "",
        8123,
    )


def test_flags_set_mode_checkpoint_settings_and_port() -> None:
    args = _load_smoke().parse_args(
        [
            "--mode",
            "mtp",
            "--ckpt",
            "/ckpt",
            "--megakernel",
            "profile=qwen35-baseline",
            "--port",
            "9000",
        ]
    )
    assert (args.mode, args.ckpt, args.megakernel, args.port) == (
        "mtp",
        "/ckpt",
        "profile=qwen35-baseline",
        9000,
    )


def test_an_unknown_mode_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _load_smoke().parse_args(["--mode", "prefill"])


def test_an_explicit_checkpoint_skips_the_hub_cache() -> None:
    assert _load_smoke()._checkpoint("/ckpt") == "/ckpt"
