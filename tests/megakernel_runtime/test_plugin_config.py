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
"""The runtime config against the library's runtime-config parser, and the
driver with a missing shorthand template."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import vllm_torchtpu.megakernel_runtime.driver as driver_mod
from vllm_torchtpu.megakernel_runtime import config as cfg_mod
from vllm_torchtpu.megakernel_runtime import plugin as plugin_mod

# A built-in shorthand and a checkpoint arch it claims: conftest's
# `mk_fake_template` (its top-level options are the library runtime config's
# template keys), entered in the table for every test here.
_SHORTHAND = "mkfake"
_HF = SimpleNamespace(architectures=["MkFakeForCausalLM"])


@pytest.fixture(autouse=True)
def _fake_template(mk_fake_template: str) -> None:
    assert mk_fake_template == _SHORTHAND


# --- VLLM_TPU_MEGAKERNEL / additional_config matrix ---------------------------------

_TEMPLATE_KEYS = (
    "mode",
    "profile",
    "buckets",
    "verify_buckets",
    "buckets_explicit",
    "verify_buckets_explicit",
    "fused_pool_io",
    "fused_pool_io_max_b",
    "fused_pool_io_min_b_hi",
    "pool_gather",
    "compile_hlo_check",
    "explicit",
)

_LEGACY_ENV = (
    None,
    "",
    "0",
    "false",
    "no",
    "off",
    "none",
    "1",
    "true",
    "YES",
    "on",
    "ep",
    "tp",
    '{"profile": "qwen35-baseline"}',
    '{"enabled": false, "mode": "tp"}',
    '{"mode": "auto", "buckets": "8,16", "verify_buckets": [4, 8]}',
    "profile=qwen35-candidate30,buckets=8:16",
    "verify_buckets=4:8,fused_pool_io=auto,pool_gather=0",
    "ab=1,stats_period_s=5",
    "fused_pool_io_max_b=16,fused_pool_io_min_b_hi=48,compile_hlo_check=0",
)
_LEGACY_ADD = (
    None,
    True,
    False,
    "ep",
    "0",
    {"profile": "qwen35-candidate30-fused"},
    {"buckets": [8, 16], "ab": True},
    {"enabled": "0"},
    {"mode": "tp", "stats_period_s": 2},
)


@pytest.fixture
def _template_profile_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy inputs name the Qwen3.5 template's profiles; register
    stand-ins under those names (the runtime's tests import no model folder)."""
    from vllm_torchtpu.megakernel import Profile, register_profile
    from vllm_torchtpu.megakernel import profile as profile_mod

    monkeypatch.setattr(profile_mod, "_REGISTRY", dict(profile_mod._REGISTRY))
    for name in ("qwen35-baseline", "qwen35-candidate30", "qwen35-candidate30-fused"):
        if name not in profile_mod._REGISTRY:
            register_profile(
                Profile(name=name, numerics="approx", fused_pool_io="auto")
            )


def _enabled_by_rule(*sources: Any) -> bool:
    """`enabled` by the config's documented rule, independently of it: the
    last explicit on / off, else on when any source configures anything."""
    explicit: bool | None = None
    configured = False
    for src in sources:
        if src is None:
            continue
        if isinstance(src, bool):
            explicit = src
            continue
        if isinstance(src, str):
            low = src.strip().lower()
            if not low:
                continue
            if low in ("0", "false", "no", "off", "none"):
                explicit = False
                continue
            if low in ("1", "true", "yes", "on", "ep", "tp"):
                explicit = True
                continue
            if low.startswith("{"):
                src = json.loads(src)
            else:
                src = dict(kv.split("=", 1) for kv in low.split(","))
        configured = configured or any(v is not None for v in src.values())
        if src.get("enabled") is not None:
            explicit = cfg_mod.parse_bool(src["enabled"])
        elif src.get("plugin") is not None:
            explicit = True
    return configured if explicit is None else explicit


@pytest.mark.usefixtures("_template_profile_names")
@pytest.mark.parametrize("env", _LEGACY_ENV)
@pytest.mark.parametrize("add", _LEGACY_ADD)
def test_legacy_inputs_match_parse_runtime_config(env: Any, add: Any) -> None:
    """Legacy syntaxes resolve to what `parse_runtime_config` gives: the
    runtime keys directly, the template settings through `options`. `enabled`
    follows the runtime's rule instead: an earlier explicit off holds over a
    later source's settings (`parse_runtime_config` turns such a source on)."""
    from vllm_torchtpu.megakernel.runtime import parse_runtime_config

    old = parse_runtime_config(env, add)
    new = cfg_mod.parse_config(env, add, raw_env=env)
    assert new.enabled == _enabled_by_rule(env, add)
    assert new.ab == old["ab"] and new.stats_period_s == old["stats_period_s"]
    resolved = cfg_mod.resolve_plugin_ref(new, _HF)
    assert resolved is not None
    ref, options = resolved
    assert ref.shorthand == _SHORTHAND
    tmpl = parse_runtime_config(options)
    assert {k: tmpl[k] for k in _TEMPLATE_KEYS} == {k: old[k] for k in _TEMPLATE_KEYS}


def test_missing_shorthand_template_leaves_the_megakernel_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the shorthand's model folder installed, the driver logs it and
    stays off: no error, no collective."""
    import logging

    from vllm_torchtpu.megakernel_runtime import testing

    gone = plugin_mod.SHORTHANDS[_SHORTHAND]._replace(
        module="no_such_mk_folder.megakernel"
    )
    monkeypatch.setitem(plugin_mod.SHORTHANDS, _SHORTHAND, gone)
    hf = _HF
    ref, _ = cfg_mod.resolve_plugin_ref(cfg_mod.parse_config("1"), hf)
    msg = cfg_mod.missing_template(ref, hf)
    assert msg is not None

    def no_collective(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("must stay off before any collective")

    monkeypatch.setattr(driver_mod, "agree_on_decline", no_collective)
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    runner = testing.fake_runner(hf_config=hf, megakernel=True)
    with caplog.at_level(logging.WARNING):
        assert driver_mod.ProgramDriver.maybe_create(runner) is None
    assert msg in caplog.text
