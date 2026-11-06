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
"""Pytest configuration for `tests/megakernel_runtime`."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.megakernel.conftest import register_megakernel_cpu_plugin

_RUNTIME = "vllm_torchtpu.megakernel_runtime."

# A built-in shorthand for tests: a stdlib-only template package on a temp
# path, entered in `plugin.SHORTHANDS` (the table itself may be empty).
FAKE_SHORTHAND = "mkfake"
FAKE_TEMPLATE = "mk_fake_template"
FAKE_ARCH = "MkFakeForCausalLM"
FAKE_MODEL_TYPE = "mk_fake"
_FAKE_SPEC = """
from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec

SPEC = TpuMonoSpec(
    "mkfake", "1", create=".adapter:create", top_level_options={"mode", "profile"}
)
"""


def pytest_configure(config: pytest.Config) -> None:
    register_megakernel_cpu_plugin(config)


@pytest.fixture(autouse=True)
def _std_megakernel_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route the lazy vLLM loggers of the imported runtime modules to
    `logging` when vLLM is absent (CPU)."""
    for name in sorted(n for n in sys.modules if n.startswith(_RUNTIME)):
        lazy = getattr(sys.modules[name], "logger", None)
        if not callable(getattr(lazy, "_resolve", None)):
            continue
        try:
            lazy._resolve()
        except ModuleNotFoundError:
            monkeypatch.setattr(lazy, "_logger", logging.getLogger(lazy._name))


@pytest.fixture
def mk_fake_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """`FAKE_SHORTHAND` in the shorthand table, for `FAKE_ARCH` /
    `FAKE_MODEL_TYPE`, naming the importable package `FAKE_TEMPLATE`."""
    from vllm_torchtpu.megakernel_runtime import plugin

    pkg = tmp_path / "fake_template" / FAKE_TEMPLATE
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(_FAKE_SPEC, encoding="utf-8")
    monkeypatch.syspath_prepend(str(pkg.parent))
    importlib.invalidate_caches()
    entry = plugin.Shorthand(FAKE_TEMPLATE, (FAKE_ARCH,), (FAKE_MODEL_TYPE,))
    monkeypatch.setitem(plugin.SHORTHANDS, FAKE_SHORTHAND, entry)
    yield FAKE_SHORTHAND
    sys.modules.pop(FAKE_TEMPLATE, None)
