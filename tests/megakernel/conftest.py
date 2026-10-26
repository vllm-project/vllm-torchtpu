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
"""Pytest configuration and shared fixtures for the TPU megakernel test suite."""

from __future__ import annotations

import os
from typing import Any

_xla_flags = os.environ.get("XLA_FLAGS", "")
for _flag_prefix, _flag in (
    (
        "--xla_force_host_platform_device_count",
        "--xla_force_host_platform_device_count=8",
    ),
    (
        "--xla_llvm_disable_expensive_passes",
        "--xla_llvm_disable_expensive_passes=true",
    ),
):
    if _flag_prefix not in _xla_flags:
        _xla_flags = f"{_xla_flags} {_flag}".strip()
os.environ["XLA_FLAGS"] = _xla_flags

import pytest  # noqa: E402

from tests.megakernel.megakernel_fakes import TINY_TILES  # noqa: E402
from vllm_torchtpu.megakernel.exec_spec import ExecSpec  # noqa: E402


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "slow: multi-rank or large-batch CPU interpret tests"
    )
    config.addinivalue_line("markers", "golden: bit-exact golden hash regression tests")


@pytest.fixture
def fake_ex() -> ExecSpec:
    return ExecSpec(B=4, T=1, R=2, ctx=256, tiles=TINY_TILES, interpret=True)
