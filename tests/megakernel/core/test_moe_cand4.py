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
"""MoE schedule options of the Qwen3.5 candidate4 schedule (CPU interpret).

Each option is moved alone on a tiny EP geometry with redundant expert copies
(`e_slots > ER`), vectorized routing and the pipelined EP expert loop, where
every option's gate holds. An exact option keeps hidden states and caches
bit-identical; every option must reach its resolved module parameter, so a
gated-off option cannot pass vacuously.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import core as mk
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.baseline import TINY_TILES


@pytest.fixture(autouse=True)
def _use_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


R = 2


def _mesh() -> Mesh:
    devs = jax.devices("cpu")
    if len(devs) < R:
        pytest.skip(f"Need at least {R} CPU devices, have {len(devs)}")
    return Mesh(np.array(devs[:R]), ("tp",))


def _geom(**kw: Any) -> qwen35.Qwen35Geometry:
    # I=512 with ebank 128 KiB: T13 = 2, T2 = 1, TE = 3 = nbe (one expert per
    # ring, PIPE_EP); e_slots=68 at R=2: 4 redundant copies per rank.
    base = dict(
        L=4, H=256, I=512, IS=256, E=128, TOPK=2, NH=4, NKV=2, HD=256, ROT=64,
        KH=2, VH=4, KD=128, VD=128, att_every=4, moe_tp=False, e_slots=68,
    )  # fmt: skip
    return qwen35.Qwen35Geometry(**{**base, **kw})


# Tunables every case starts from: the vectorized routing and pipelined EP
# experts the candidate4 schedule runs, unrolled expert releases.
_BASE_MOE = {"pipe_ep": True, "route_vec": True, "ecopy_opt": 1}


# Core tile fields every case starts from: the fully unrolled dense matmul
# (as the candidate4 schedule runs it), which the register-summed shared
# expert of `sh_pre` reproduces.
_BASE_TILES = {"dense_acc_pair": True, "dmm_unroll": 8}


def _exec(
    T: int, B: int, moe: dict[str, Any], stack_e: bool = False, nbe: int = 3
) -> mk.ExecSpec:
    moe = dict(moe)
    core = {**_BASE_TILES, **moe.pop("tiles", {})}
    tiles = dataclasses.replace(
        TINY_TILES, bank=512 << 10, ebank=128 << 10, nbe=nbe, **core
    ).with_modules({"gqa": {"kv_blk": 256}, "moe": {**_BASE_MOE, **moe}})
    return mk.ExecSpec(
        R=R, B=B, T=T, ctx=256, stack_e=stack_e, interpret=True, tiles=tiles
    )


def _inputs(g: qwen35.Qwen35Geometry, ex: mk.ExecSpec, seed: int) -> tuple:
    """Logical inputs with the router biased toward the copied experts, so the
    redundant copies and Po2C take part in routing."""
    k_hf, k_in = jax.random.split(jax.random.key(seed))
    hf = mk.random_hf_weights(k_hf, g)
    er = g.ER(R)
    rt = hf["router"]
    for j, e in enumerate((0, 1, 2, er, er + 1, er + 2)):
        rt = rt.at[:, e, :].add(jnp.bfloat16(60.0 if j % 2 == 0 else -60.0))
    hf = dict(hf, router=rt)
    pos, hid, _, st = mk.random_logical_inputs(k_in, g, ex)
    w = mk.load_logical_weights(qwen35.Qwen35, ex, hf, g=g)
    return pos, hid, w, st


_OUT: dict[tuple, list] = {}


def _run(T: int, B: int, moe: dict[str, Any], seed: int = 7) -> list:
    """Bytes of hidden + output states of one step (cached per config)."""
    key = (T, B, repr(sorted(moe.items())), seed)
    if key not in _OUT:
        g, ex = _geom(), _exec(T, B, moe)
        pos, hid, w, st = _inputs(g, ex, seed)
        p_mk, h_mk, w_mk, s_mk = mk.pack_inputs(pos, hid, w, st, g, ex, mesh=_mesh())
        step, _ = qwen35.make_step(_mesh(), g, ex)
        out_h, out_st = step(p_mk, h_mk, w_mk, s_mk)
        _OUT[key] = [np.asarray(out_h).view(np.uint8)] + [
            np.asarray(out_st[k]).view(np.uint8) for k in sorted(out_st)
        ]
    return _OUT[key]


def _moe_params(T: int, B: int, moe: dict[str, Any]) -> Any:
    _, info = qwen35.build(_geom(), _exec(T, B, moe))
    return info["plan"].module_params["moe"]


def _assert_same(a: list, b: list) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_base_case_engages_the_gates():
    """The base case runs the vectorized routing with copies and PIPE_EP."""
    mp = _moe_params(4, 8, {})
    assert mp.ECOPY and mp.route_vec and mp.PIPE_EP and not mp.PIPE
    assert mp.NBE == mp.TE and not mp.stack_e


# (id, option, value, resolved MoEParams field, other options set in both arms)
_SHRINK = {"cp_interleave": True, "route_shrink": True}
_EXACT = [
    ("ep_fast_refill", "ep_fast_refill", True, "ep_fast", {}),
    ("route_early", "route_early", True, "route_early", {}),
    (
        "route_early-prim_now",
        "route_early",
        True,
        "route_early",
        {**_SHRINK, "prim_now": True},
    ),
    ("route_shrink", "route_shrink", True, "route_shrink", {"cp_interleave": True}),
    ("prim_now", "prim_now", True, "prim_now", _SHRINK),
    ("sh_pre", "sh_pre", True, "sh_pre", {}),
    ("sh_pre-hoist", "sh_pre", True, "sh_hoist", {"tiles": {"dwait_hoist": True}}),
    ("sh2_pre_route", "sh2_pre_route", True, "sh2_pre_route", {"sh_pre": True}),
    ("sh_at1", "sh_at", 1, "sh_at", {"sh_pre": True}),
    ("sh_at2", "sh_at", 2, "sh_at", {"sh_pre": True}),
    ("flush_pre_drain", "flush_pre_drain", True, "flush_pre_drain", {}),
    # the plain expert loop (no PIPE_EP): the refill inside the MoE all-reduce
    ("flush_in_ar1", "flush_in_ar1", True, "flush_in_ar1", {"pipe_ep": False}),
]


# Decode (T=1) rechecks of the options that touch every bucket's routing or
# expert refills; the rest run at the verify shape only (interpret-mode cost).
_DECODE_TOO = {"ep_fast_refill", "route_early", "sh_pre"}


@pytest.mark.parametrize(
    "opt,value,field,base,T,B",
    [
        pytest.param(*c[1:], T, 8, id=f"{c[0]}-{T}-8")
        for c in _EXACT
        for T in ((4, 1) if c[0] in _DECODE_TOO else (4,))
    ],
)
def test_exact_option_is_bit_identical(
    opt: str, value: Any, field: str, base: dict, T: int, B: int
):
    on = {**base, opt: value}
    before, after = _moe_params(T, B, base), _moe_params(T, B, on)
    assert getattr(after, field) != getattr(before, field)
    _assert_same(_run(T, B, on), _run(T, B, base))


def test_cp_interleave_changes_the_copy_scan_within_tolerance():
    """cp_interleave reorders the copy slots Po2C scans (approx): the step stays
    close to the rank-major order's, and the resolved flag reaches the plan."""
    assert _moe_params(4, 8, {"cp_interleave": True}).cp_interleave
    a = _run(4, 8, {})[0].view(np.uint16).view(jnp.bfloat16).astype(np.float32)
    b = (
        _run(4, 8, {"cp_interleave": True})[0]
        .view(np.uint16)
        .view(jnp.bfloat16)
        .astype(np.float32)
    )
    np.testing.assert_allclose(b, a, atol=3e-2, rtol=3e-2)


def test_route_shrink_requires_cp_interleave():
    with pytest.raises(mk.MegakernelConfigError, match="cp_interleave"):
        _moe_params(4, 8, {"route_shrink": True})


def test_sh_pre_and_flush_gates():
    """sh_pre needs a lazy MoE segment; sh_at / sh2_pre_route need sh_pre;
    flush_pre_drain needs PIPE_EP and a lazy segment; flush_in_ar1 needs a
    lazy segment off PIPE_EP."""
    assert not _moe_params(4, 8, {"sh_pre": True, "tiles": {"lazy": False}}).sh_pre
    assert _moe_params(4, 8, {"sh_at": 2}).sh_at == 0
    assert not _moe_params(4, 8, {"sh2_pre_route": True}).sh2_pre_route
    fpd = {"flush_pre_drain": True}
    assert not _moe_params(4, 8, {**fpd, "pipe_ep": False}).flush_pre_drain
    assert not _moe_params(4, 8, {**fpd, "tiles": {"lazy": False}}).flush_pre_drain
    with pytest.raises(mk.MegakernelConfigError, match="sh_at"):
        _moe_params(4, 8, {"sh_pre": True, "sh_at": 3})
    # flush_in_ar1 leaves PIPE_EP buckets to flush_pre_drain's placement
    assert not _moe_params(4, 8, {"flush_in_ar1": True}).flush_in_ar1


def test_prim_now_needs_route_shrink():
    assert not _moe_params(4, 8, {"prim_now": True}).prim_now
    assert _moe_params(4, 8, {**_SHRINK, "prim_now": True}).prim_now


def test_ep_fast_refill_gate():
    """Off unless PIPE_EP runs with one expert per ring on unstacked weights."""
    assert _moe_params(4, 8, {"ep_fast_refill": True}).ep_fast
    g = _geom()
    ex = _exec(4, 8, {"ep_fast_refill": True}, stack_e=True)
    _, info = qwen35.build(g, ex)
    assert not info["plan"].module_params["moe"].ep_fast
    assert not _moe_params(4, 8, {"ep_fast_refill": True, "pipe_ep": False}).ep_fast
    ex2 = _exec(4, 8, {"ep_fast_refill": True}, nbe=6)  # two experts per ring
    _, info2 = qwen35.build(g, ex2)
    mp2 = info2["plan"].module_params["moe"]
    assert mp2.PIPE_EP and not mp2.ep_fast
