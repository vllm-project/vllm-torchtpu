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
"""Shared test fixtures for the TPU megakernel test suite."""

from __future__ import annotations

import gc
import os
import sys
from typing import Any

import pytest

from tests.megakernel.megakernel_helpers import (  # noqa: F401
    EXACT_FLIPS,
    TINY_TILES,
    FakeAltMixerBlock,
    FakeFFNBlock,
    FakeGeometry,
    FakeMixerBlock,
    FakeMixerTunables,
    MiniCtx,
    _fake_ex_fixture,
    make_fake_model_spec,
    tile_field,
    with_tile_fields,
)


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers", "slow: multi-rank or large-batch CPU interpret tests"
    )
    config.addinivalue_line("markers", "golden: bit-exact golden hash regression tests")


@pytest.fixture(autouse=True)
def _reclaim_megakernel_cpu_jax_cache():
    """Release one-shot megakernel `make_step` closures after each CPU test."""
    yield
    if os.path.exists("/dev/vfio") or "jax" not in sys.modules:
        return
    from jax._src import xla_bridge

    if xla_bridge.backends_are_initialized():
        sys.modules["jax"].clear_caches()
        gc.collect()
