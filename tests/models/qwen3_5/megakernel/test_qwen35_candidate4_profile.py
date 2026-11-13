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
"""`qwen35-candidate4` reproduces candidate4's per-bucket schedule.

The expected values below were read from candidate4's own kernel config
(`kernels/qwen35_megakernel/config.py` at 5a1c82686: `default_tiles` at the
real Qwen3.5-397B geometry, R=8, T=4, fused pool I/O, EP and TP, against
candidate30's `default_tiles` at 8743659cd): over the 4..64-row verify
buckets the two schedules differ only in the dense ring depth of fused
buckets and in `hx`; the bucket tables are candidate4's constants.
"""

from __future__ import annotations

import dataclasses

import pytest

from vllm_torchtpu.megakernel.core.profile import TileContext
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.profiles import PROFILES, candidate4
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.candidate30 import (
    candidate30_tiles,
)

SHEET_BUCKETS = tuple(range(4, 65, 4))
# candidate4 vs candidate30 `default_tiles`, fused T=4 buckets (EP and TP alike).
C4_FUSED_NBD = {32: 12, 40: 12, 44: 12, 48: 12, 52: 11, 56: 11, 60: 10, 64: 9}
C4_HX_BUCKETS = frozenset({48, 52, 56, 60, 64})
C4_MXU_DR = frozenset({4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64})
C4_ARPIPE_OFF = frozenset({20, 24, 32})
_LEVERS_ON = {
    "gqa": ("gdefer", "kvdefer", "qswap", "wgroup"),
    "gdn": ("lcv", "zdefer"),
    "moe": (
        "route_shrink",
        "cp_interleave",
        "route_early",
        "prim_now",
        "sh_pre",
        "sh2_pre_route",
        "flush_in_ar1",
        "flush_pre_drain",
        "ep_fast_refill",
    ),
}


def _tc(B: int, *, moe_tp: bool, fused: bool = True, T: int = 4, g=None):
    return TileContext(B=B, T=T, R=8, moe_tp=moe_tp, fused_pool_io=fused, g=g)


def _diff(a, b) -> dict[str, tuple]:
    """Core tile fields where `a` and `b` differ (module tunables excluded)."""
    return {
        f.name: (getattr(a, f.name), getattr(b, f.name))
        for f in dataclasses.fields(a)
        if f.name != "modules" and getattr(a, f.name) != getattr(b, f.name)
    }


_SCHEDULE_FLAGS = ("arpipe", "arpipe_h1", "ar0s", "dwait_hoist", "hier1_peers")


@pytest.mark.parametrize("moe_tp", [False, True], ids=["ep", "tp"])
@pytest.mark.parametrize("B", SHEET_BUCKETS)
def test_core_tiles_match_candidate4_on_fused_verify_buckets(B: int, moe_tp: bool):
    c4 = candidate4.candidate4_tiles(_tc(B, moe_tp=moe_tp))
    c30 = candidate30_tiles(_tc(B, moe_tp=moe_tp))
    want = {}
    if B in C4_FUSED_NBD:
        want["nbd"] = (C4_FUSED_NBD[B], c30.nbd)
    if B in C4_HX_BUCKETS:
        want["hx"] = (True, False)
    got = {k: v for k, v in _diff(c4, c30).items() if k not in _SCHEDULE_FLAGS}
    assert got == want
    assert c4.arpipe is (B not in C4_ARPIPE_OFF)
    assert (c4.arpipe_h1, c4.ar0s, c4.dwait_hoist) == (True, True, True)
    assert c4.hier1_peers == (3, 5, 7)


@pytest.mark.parametrize("moe_tp", [False, True], ids=["ep", "tp"])
@pytest.mark.parametrize("B", SHEET_BUCKETS)
def test_module_levers_match_candidate4(B: int, moe_tp: bool):
    c4 = candidate4.candidate4_tiles(_tc(B, moe_tp=moe_tp))
    c30 = candidate30_tiles(_tc(B, moe_tp=moe_tp))
    for mod, names in _LEVERS_ON.items():
        for name in names:
            assert getattr(c4.module(mod), name) is True, (mod, name)
    assert c4.module("gdn").mxu_dr is (B in C4_MXU_DR)
    assert c4.module("moe").sh_at == 0
    # Every candidate30 module tunable is kept.
    for mod in ("gqa", "gdn", "moe"):
        base = dataclasses.asdict(c30.module(mod))
        new = dataclasses.asdict(c4.module(mod))
        changed = {k for k in base if base[k] != new[k]}
        allowed = set(_LEVERS_ON.get(mod, ())) | {"mxu_dr", "sh_at"}
        assert changed <= allowed, (mod, changed)


@pytest.mark.parametrize("B", SHEET_BUCKETS)
def test_split_pool_io_buckets_keep_candidate30_ring_depths(B: int):
    c4 = candidate4.candidate4_tiles(_tc(B, moe_tp=False, fused=False))
    c30 = candidate30_tiles(_tc(B, moe_tp=False, fused=False))
    got = {k: v for k, v in _diff(c4, c30).items() if k not in _SCHEDULE_FLAGS}
    assert got == ({"hx": (True, False)} if B in C4_HX_BUCKETS else {})


def test_sh_at2_variant_only_on_small_buckets():
    rule = PROFILES["qwen35-candidate4-sh-at2"].tiles
    for B in SHEET_BUCKETS:
        sh = rule(_tc(B, moe_tp=True)).module("moe").sh_at
        assert sh == (2 if B <= candidate4.SH_AT_MAX_ROWS else 0), B


def test_attention_only_stack_uses_the_deep_dense_ring():
    g = dataclasses.replace(qwen35.TINY, att_every=1, L=1)
    for B in (4, 32, 64):
        t = candidate4.candidate4_tiles(_tc(B, moe_tp=False, g=g))
        assert t.nbd == 14
        assert t.arpipe is False  # one attention layer per cycle: no ARPIPE


@pytest.mark.parametrize("B", (4, 8, 16, 64))
def test_arpipe_only_on_fused_verify_buckets(B: int):
    rule = candidate4.candidate4_tiles
    assert rule(_tc(B, moe_tp=False)).arpipe is True
    assert rule(_tc(B, moe_tp=False, fused=False)).arpipe is False
    assert rule(_tc(B, moe_tp=False, T=1)).arpipe is False


def test_profiles_are_registered_opt_in_and_approx():
    p = PROFILES["qwen35-candidate4"]
    assert (p.numerics, p.fused_pool_io, p.verify_buckets) == (
        "approx",
        True,
        SHEET_BUCKETS,
    )
    assert PROFILES["qwen35-candidate4-sh-at2"].numerics == "approx"
    from vllm_torchtpu.models.qwen3_5.megakernel.profiles import DEFAULT_PROFILE

    assert DEFAULT_PROFILE.name == "qwen35-candidate30-fused"
    assert {"candidate4", "candidate4-sh-at2"} <= set(qwen35.Qwen35.profiles)


def test_bucket_tables_are_candidate4s():
    assert candidate4.FUSED_WIDE_NBD == C4_FUSED_NBD
    assert set(candidate4.MXU_DR_BUCKETS) == C4_MXU_DR
    assert set(candidate4.ARPIPE_OFF_B) == C4_ARPIPE_OFF
    assert candidate4.HX_MIN_ROWS == 48
