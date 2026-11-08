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

from tests.megakernel_runtime import toy_plugin
from tests.megakernel_runtime.toy_plugin import TOY_MODULE, TOY_PLUGIN
from vllm_torchtpu.megakernel_runtime import testkit
from vllm_torchtpu.megakernel_runtime.config import MegakernelConfigError

_INIT = '''
"""A plugin package."""
from vllm_torchtpu.megakernel.runtime.contract import TpuMonoSpec
{extra}
SPEC = TpuMonoSpec("pkg", "1", create={create!r}{spec_kw})
'''
_ADAPTER = "def create(view, options):\n    return None\n"

mk_fake_runner = testkit.mk_fake_runner
mk_toy_plugin = toy_plugin.mk_toy_plugin


@pytest.fixture
def make_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(tmp_path))
    made: list[str] = []

    def make(
        name: str,
        *,
        create: str = ".adapter:create",
        extra: str = "",
        spec_kw: str = "",
    ) -> str:
        pkg = tmp_path / name
        pkg.mkdir()
        init = _INIT.format(extra=extra, create=create, spec_kw=spec_kw)
        (pkg / "__init__.py").write_text(init, encoding="utf-8")
        (pkg / "adapter.py").write_text(_ADAPTER, encoding="utf-8")
        importlib.invalidate_caches()
        made.append(name)
        return name

    yield make
    for name in made:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            sys.modules.pop(mod)


def test_check_plugin_spec_accepts_the_toy_and_a_lazy_package(
    mk_toy_plugin: Path, make_pkg: Any
) -> None:
    spec = testkit.check_plugin_spec(TOY_MODULE)
    assert spec.name == "toy" and spec is sys.modules["toy_mk_plugin"].SPEC
    assert testkit.check_plugin_spec(make_pkg("mk_ok_pkg")).create == ".adapter:create"
    assert testkit.check_plugin_spec(spec, lazy_create=False) is spec


def test_check_plugin_spec_rejects_eager_create_heavy_imports_and_bad_refs(
    mk_toy_plugin: Path, make_pkg: Any
) -> None:
    with pytest.raises(AssertionError, match="lazy 'module:attr'"):
        testkit.check_plugin_spec(TOY_MODULE, lazy_create=True)
    with pytest.raises(AssertionError, match="blocked"):
        testkit.check_plugin_spec(make_pkg("mk_heavy_pkg", extra="import torch"))
    with pytest.raises(MegakernelConfigError, match="can't resolve create"):
        testkit.check_plugin_spec(make_pkg("mk_bad_ref_pkg", create=".adapter:nope"))
    with pytest.raises(MegakernelConfigError, match="not a TpuMonoSpec"):
        testkit.check_plugin_spec(f"{TOY_MODULE}:ToyProgram")


@pytest.mark.parametrize(
    ("spec_kw", "error", "match"),
    [
        ('decline_codes=("other",)', AssertionError, "bad decline_codes"),
        ('decline_codes=("a", "a")', AssertionError, "duplicate decline_codes"),
        ('decline_codes=("",)', AssertionError, "bad decline_codes"),
        ('layout_request=".adapter:nope"', MegakernelConfigError, "layout_request"),
    ],
    ids=["reserved_other", "duplicate", "empty", "unresolved_layout_request"],
)
def test_check_plugin_spec_rejects_bad_decline_codes_and_layout_request(
    make_pkg: Any, spec_kw: str, error: type[Exception], match: str
) -> None:
    pkg = make_pkg("mk_bad_spec_pkg", spec_kw=f", {spec_kw}")
    with pytest.raises(error, match=match):
        testkit.check_plugin_spec(pkg)


def test_run_cpu_e2e_serves_the_toy(mk_toy_plugin: Path, mk_fake_runner: Any) -> None:
    ids = torch.arange(3, dtype=torch.int32)
    big = torch.arange(9, dtype=torch.int32)
    res = testkit.run_cpu_e2e(
        TOY_MODULE,
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
        TOY_MODULE,
        mk_fake_runner(),
        lambda prog: [(torch.full((5,), prog.scale), torch.zeros(5))],
    )
    assert torch.equal(res.outputs[0], torch.full((5,), 2.0 * 2 + 8))


def test_run_cpu_e2e_names_a_startup_decline(
    mk_toy_plugin: Path, mk_fake_runner: Any
) -> None:
    with pytest.raises(AssertionError, match="toy_declined"):
        testkit.run_cpu_e2e(TOY_MODULE, mk_fake_runner, [], options={"decline": True})


def test_assert_rank_uniform_declines_fails_when_both_ranks_serve(
    tmp_path: Path,
) -> None:
    src = (TOY_PLUGIN, TOY_PLUGIN)
    # The serve check fails, not a crashed rank (run_two_ranks asserts that
    # each rank exits cleanly, with its log as the message).
    with pytest.raises(AssertionError, match=r"rank 0 served \(ProgramDriver\)"):
        testkit.assert_rank_uniform_declines(tmp_path, plugin=TOY_MODULE, sources=src)


# --- 2-rank gloo ----------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario", ["serve", "decline", "mismatch", "create_error", "import_error"]
)
def test_two_rank_gloo_agreement(tmp_path: Path, scenario: str) -> None:
    """Real 2-process gloo agreement: never hangs, always rank-uniform."""
    src = (TOY_PLUGIN, TOY_PLUGIN)
    options: dict[str, Any] = {}
    if scenario == "decline":
        options = {"decline_rank": 1}
    elif scenario == "mismatch":
        src = (TOY_PLUGIN, TOY_PLUGIN + "\n# skewed on rank 1\n")
    elif scenario == "create_error":
        options = {"raise_rank": 1}
    elif scenario == "import_error":
        src = (TOY_PLUGIN, 'raise ImportError("plugin missing on this host")\n')
    kw: dict[str, Any] = {"options": options, "sources": src}
    if scenario == "serve":
        r0, r1 = testkit.run_two_ranks(tmp_path, plugin=TOY_MODULE, **kw)
    else:
        r0, r1 = testkit.assert_rank_uniform_declines(tmp_path, plugin=TOY_MODULE, **kw)
    if scenario == "serve":
        buckets = [[1, 4], [1, 8]]
        assert r0 == r1 == {"type": "ProgramDriver", "code": None, "buckets": buckets}
    elif scenario == "decline":
        assert r0 == {"type": "DeclinedDriver", "code": "peer_declined:toy_declined"}
        assert r1 == {"type": "DeclinedDriver", "code": "toy_declined"}
    elif scenario == "mismatch":
        assert r0 == r1 == {"type": "DeclinedDriver", "code": "plugin_mismatch"}
    elif scenario == "create_error":
        assert r0 == {"type": "DeclinedDriver", "code": "peer_declined:create_error"}
        assert r1 == {"raised": "RuntimeError: toy create bug"}
    else:
        # Rank 1 votes before raising (token 0): rank 0 declines, no hang.
        assert r0 == {"type": "DeclinedDriver", "code": "plugin_mismatch"}
        assert r1["raised"].startswith("MegakernelConfigError")
        assert "plugin missing on this host" in r1["raised"]
