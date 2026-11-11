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
"""Pytest configuration for `tests/megakernel/adapter`."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

# A built-in shorthand for tests: a stdlib-only template package on a temp
# path, entered in `plugin.SHORTHANDS` (the table itself may be empty).
FAKE_SHORTHAND = "mkfake"
FAKE_TEMPLATE = "mk_fake_template"
FAKE_ARCH = "MkFakeForCausalLM"
FAKE_MODEL_TYPE = "mk_fake"
_FAKE_SPEC = """
from vllm_torchtpu.megakernel.core.runtime.contract import TpuMonoSpec

SPEC = TpuMonoSpec(
    "mkfake", "1", create=".adapter:create", top_level_options={"mode", "profile"}
)
"""


@pytest.fixture
def mk_fake_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """`FAKE_SHORTHAND` in the shorthand table, for `FAKE_ARCH` /
    `FAKE_MODEL_TYPE`, naming the importable package `FAKE_TEMPLATE`."""
    from vllm_torchtpu.megakernel.adapter import plugin

    pkg = tmp_path / "fake_template" / FAKE_TEMPLATE
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(_FAKE_SPEC, encoding="utf-8")
    monkeypatch.syspath_prepend(str(pkg.parent))
    importlib.invalidate_caches()
    entry = plugin.Shorthand(FAKE_TEMPLATE, (FAKE_ARCH,), (FAKE_MODEL_TYPE,))
    monkeypatch.setitem(plugin.SHORTHANDS, FAKE_SHORTHAND, entry)
    yield FAKE_SHORTHAND
    sys.modules.pop(FAKE_TEMPLATE, None)
