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
"""The plugin test kit on the toy plugin and on small bad plugins."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm_torchtpu.megakernel_runtime import testkit
from vllm_torchtpu.megakernel_runtime.config import MegakernelConfigError

_INIT = '''
"""A plugin package."""
from vllm_torchtpu.megakernel_runtime.plugin import TpuMonoSpec
{extra}
SPEC = TpuMonoSpec("pkg", "1", create={create!r})
'''
_ADAPTER = "def create(view, options):\n    return None\n"

mk_fake_runner = testkit.mk_fake_runner
mk_toy_plugin = testkit.mk_toy_plugin


@pytest.fixture
def make_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(tmp_path))
    made: list[str] = []

    def make(name: str, *, create: str = ".adapter:create", extra: str = "") -> str:
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            _INIT.format(extra=extra, create=create), encoding="utf-8"
        )
        (pkg / "adapter.py").write_text(_ADAPTER, encoding="utf-8")
        importlib.invalidate_caches()
        made.append(name)
        return name

    yield make
    for name in made:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            sys.modules.pop(mod)


def test_check_spec_accepts_the_toy_and_a_lazy_package(
    mk_toy_plugin: Path, make_pkg: Any
) -> None:
    spec = testkit.check_spec(testkit.TOY_MODULE)
    assert spec.name == "toy" and spec is sys.modules["toy_mk_plugin"].SPEC
    assert testkit.check_spec(make_pkg("mk_ok_pkg")).create == ".adapter:create"
    assert testkit.check_spec(spec, lazy_create=False) is spec


def test_check_spec_rejects_eager_create_heavy_imports_and_bad_refs(
    mk_toy_plugin: Path, make_pkg: Any
) -> None:
    with pytest.raises(AssertionError, match="lazy 'module:attr'"):
        testkit.check_spec(testkit.TOY_MODULE, lazy_create=True)
    with pytest.raises(AssertionError, match="blocked"):
        testkit.check_spec(make_pkg("mk_heavy_pkg", extra="import torch"))
    with pytest.raises(MegakernelConfigError, match="can't resolve create"):
        testkit.check_spec(make_pkg("mk_bad_ref_pkg", create=".adapter:nope"))
    with pytest.raises(MegakernelConfigError, match="not a TpuMonoSpec"):
        testkit.check_spec(f"{testkit.TOY_MODULE}:ToyProgram")


def test_run_cpu_e2e_serves_the_toy(mk_toy_plugin: Path, mk_fake_runner: Any) -> None:
    ids = torch.arange(3, dtype=torch.int32)
    big = torch.arange(9, dtype=torch.int32)
    res = testkit.run_cpu_e2e(
        testkit.TOY_MODULE,
        mk_fake_runner,
        [(ids, torch.zeros(3)), (big, torch.zeros(9))],
        options={"scale": 3.0},
    )
    assert res.hits == [True, False] and res.outputs[1] is None
    assert torch.equal(res.outputs[0], ids.float() * 3.0 + 4)
    assert res.step_mix == {"mk_t1_b4": 1, "stock_big": 1}
    assert res.program.events[:2] == ["load", "bind"]
    assert res.runner._megakernel is res.driver


def test_run_cpu_e2e_steps_from_the_program(
    mk_toy_plugin: Path, mk_fake_runner: Any
) -> None:
    res = testkit.run_cpu_e2e(
        testkit.TOY_MODULE,
        mk_fake_runner(),
        lambda prog: [(torch.full((5,), prog.scale), torch.zeros(5))],
    )
    assert torch.equal(res.outputs[0], torch.full((5,), 2.0 * 2 + 8))


def test_run_cpu_e2e_names_a_startup_decline(
    mk_toy_plugin: Path, mk_fake_runner: Any
) -> None:
    with pytest.raises(AssertionError, match="toy_declined"):
        testkit.run_cpu_e2e(
            testkit.TOY_MODULE, mk_fake_runner, [], options={"decline": True}
        )


def test_assert_rank_uniform_declines_fails_when_both_ranks_serve(
    tmp_path: Path,
) -> None:
    src = (testkit.TOY_PLUGIN, testkit.TOY_PLUGIN)
    with pytest.raises(AssertionError):
        testkit.assert_rank_uniform_declines(tmp_path, sources=src)
