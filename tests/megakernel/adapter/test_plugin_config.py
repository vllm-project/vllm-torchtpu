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
"""The adapter config's legacy inputs against its documented per-key rule (the
template settings reach the library's `parse_runtime_options` through
`options`), and the driver with a missing shorthand template."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

import vllm_torchtpu.megakernel.adapter.driver as driver_mod
from vllm_torchtpu.megakernel.adapter import config as cfg_mod
from vllm_torchtpu.megakernel.adapter import plugin as plugin_mod
from vllm_torchtpu.megakernel.adapter import testing
from vllm_torchtpu.megakernel.core import Profile
from vllm_torchtpu.megakernel.core import profile as profile_mod
from vllm_torchtpu.megakernel.core.runtime import (
    DEFAULT_BUCKETS,
    DEFAULT_FUSED_POOL_IO_MAX_B,
    DEFAULT_FUSED_POOL_IO_MIN_B_HI,
    DEFAULT_VERIFY_BUCKETS,
    parse_runtime_options,
)

# A built-in shorthand and a checkpoint arch it claims: conftest's
# `mk_fake_template` (its top-level options are the library runtime config's
# template keys), entered in the table for every test here.
_SHORTHAND = "mkfake"
_HF = SimpleNamespace(architectures=["MkFakeForCausalLM"])


@pytest.fixture(autouse=True)
def _fake_template(mk_fake_template: str) -> None:
    assert mk_fake_template == _SHORTHAND


# --- VLLM_TPU_MEGAKERNEL / additional_config matrix ---------------------------------

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


# Stand-ins for the Qwen3.5 template's profiles that the legacy inputs name
# (the adapter's tests import no model folder), each with its own non-default
# profile-owned values so the filled-in defaults are checked per profile.
_STAND_INS = {
    p.name: p
    for p in (
        Profile(
            name="qwen35-baseline",
            numerics="approx",
            fused_pool_io="auto",
            fused_pool_io_max_b=12,
            pool_gather=None,
        ),
        Profile(
            name="qwen35-candidate30",
            numerics="approx",
            fused_pool_io=True,
            fused_pool_io_min_b_hi=40,
            buckets=(16, 32),
            verify_buckets=(8, 16),
        ),
        Profile(
            name="qwen35-candidate30-fused",
            numerics="approx",
            fused_pool_io="auto",
            fused_pool_io_max_b=24,
            fused_pool_io_min_b_hi=56,
            pool_gather=True,
        ),
    )
}


@pytest.fixture
def _template_profile_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the stand-ins under the template's profile names."""
    monkeypatch.setattr(
        profile_mod, "_REGISTRY", {**profile_mod._REGISTRY, **_STAND_INS}
    )


def _word_bool(val: Any) -> bool:
    """A legacy on / off value (a bool, or one of the words the inputs use)."""
    if isinstance(val, bool):
        return val
    return {"0": False, "false": False, "1": True, "true": True}[str(val).lower()]


def _enabled_by_rule(*sources: Any) -> bool:
    """`enabled` by the config's documented rule, computed here without the
    config module: the last explicit on / off, else on when any source
    configures anything."""
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
            explicit = _word_bool(src["enabled"])
        elif src.get("plugin") is not None:
            explicit = True
    return configured if explicit is None else explicit


def _source_mapping(src: Any) -> dict[str, Any]:
    """One source's keys: none for an on / off word, `mode` for `ep` / `tp`."""
    if src is None or isinstance(src, bool):
        return {}
    if not isinstance(src, str):
        return dict(src)
    s = src.strip()
    if s.lower() in ("", "0", "false", "no", "off", "none", "1", "true", "yes", "on"):
        return {}
    if s.lower() in ("ep", "tp"):
        return {"mode": s.lower()}
    if s.startswith("{"):
        return json.loads(s)
    return dict(kv.split("=", 1) for kv in s.split(","))


def _keys_by_rule(*sources: Any) -> dict[str, Any]:
    """Every key by the documented rule, independently of the parser: the
    last source that sets a key wins."""
    merged: dict[str, Any] = {}
    for src in sources:
        merged.update((k, v) for k, v in _source_mapping(src).items() if v is not None)
    return merged


def _typed_by_rule(opts: dict[str, Any]) -> dict[str, Any]:
    """`parse_runtime_options(opts)` by its documented rule, computed from the
    inputs and the stand-in profile: set keys typed, unset profile-owned keys
    from the named profile (`pool_gather=None` as `"auto"`), else the library
    defaults."""
    prof = _STAND_INS[opts["profile"]] if "profile" in opts else None

    def rows(val: Any) -> tuple[int, ...]:
        if isinstance(val, str):
            return tuple(int(x) for x in val.replace(":", ",").split(","))
        return tuple(val)

    def owned(key: str, parse: Any, default: Any) -> Any:
        if key in opts:
            return parse(opts[key])
        return default if prof is None else getattr(prof, key)

    flag = {"auto": "auto", "0": False, "1": True}.__getitem__
    pool_gather = owned("pool_gather", flag, False)
    prof_buckets = None if prof is None else prof.buckets
    prof_verify = None if prof is None else prof.verify_buckets
    return {
        "mode": {"auto": ""}.get(opts.get("mode", ""), opts.get("mode", "")),
        "profile": None if prof is None else prof.name,
        "buckets": rows(opts["buckets"])
        if "buckets" in opts
        else prof_buckets or DEFAULT_BUCKETS,
        "verify_buckets": rows(opts["verify_buckets"])
        if "verify_buckets" in opts
        else prof_verify or DEFAULT_VERIFY_BUCKETS,
        "buckets_explicit": "buckets" in opts,
        "verify_buckets_explicit": "verify_buckets" in opts,
        "fused_pool_io": owned("fused_pool_io", flag, False),
        "fused_pool_io_max_b": owned(
            "fused_pool_io_max_b", int, DEFAULT_FUSED_POOL_IO_MAX_B
        ),
        "fused_pool_io_min_b_hi": owned(
            "fused_pool_io_min_b_hi", int, DEFAULT_FUSED_POOL_IO_MIN_B_HI
        ),
        "pool_gather": "auto" if pool_gather is None else pool_gather,
        "compile_hlo_check": flag(opts.get("compile_hlo_check", "0")),
        "explicit": frozenset(
            opts.keys()
            & {
                "fused_pool_io",
                "fused_pool_io_max_b",
                "fused_pool_io_min_b_hi",
                "pool_gather",
            }
        ),
    }


@pytest.mark.usefixtures("_template_profile_names")
@pytest.mark.parametrize("env", _LEGACY_ENV)
@pytest.mark.parametrize("add", _LEGACY_ADD)
def test_legacy_inputs_follow_the_per_key_rule(env: Any, add: Any) -> None:
    """Legacy syntaxes resolve per key, the later source winning: the runtime
    keys directly, the template settings into `options` (which the library's
    `parse_runtime_options` then types, checked against `_typed_by_rule`).
    `enabled` follows its own rule: an earlier explicit off holds over a later
    source's settings."""
    new = cfg_mod.parse_config(env, add, raw_env=env)
    assert new.enabled == _enabled_by_rule(env, add)
    want = _keys_by_rule(env, add)
    assert new.ab == _word_bool(want.get("ab", False))
    want_period = float(want.get("stats_period_s", cfg_mod.DEFAULT_STATS_PERIOD_S))
    assert new.stats_period_s == want_period
    resolved = cfg_mod.resolve_plugin_ref(new, _HF)
    assert resolved is not None
    ref, options = resolved
    assert ref.shorthand == _SHORTHAND
    want_opts = {k: v for k, v in want.items() if k not in cfg_mod.RUNTIME_KEYS}
    assert dict(options) == want_opts
    got, typed = parse_runtime_options(options), _typed_by_rule(want_opts)
    assert got == typed
    assert {k: type(v) for k, v in got.items()} == {
        k: type(v) for k, v in typed.items()
    }


def test_missing_shorthand_template_leaves_the_megakernel_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the shorthand's model folder installed, the driver logs it and
    stays off: no error, no collective."""
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
