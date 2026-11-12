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
"""GQA schedule options of the Qwen3.5 candidate4 schedule (CPU interpret).

Each option is moved alone on a tiny paged Qwen3.5 step, through the split
pool I/O and the 1-launch fused pool I/O. An exact option keeps hidden states
and every pool byte identical; every option must reach its resolved module
parameter, so a gated-off option cannot pass vacuously.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

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


def _geom() -> qwen35.Qwen35Geometry:
    # KH=4, VH=8 at R=2: kh=2, vh=4. NKV=1: both ranks share the KV head
    # (G=2), so the sequence-split attention and the Q exchange run, as at
    # the real geometry, and head-batched Q staging also engages at T=1.
    return qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=256, E=128, TOPK=2, NH=4, NKV=1, HD=256, ROT=64,
        KH=4, VH=8, KD=128, VD=128, att_every=4, moe_tp=False, e_slots=0,
    )  # fmt: skip


# Module tunables every case starts from (head-batched Q staging, which the
# QSW path needs; the fused GDN recurrence is head-major, so the split path
# runs it too).
_BASE = {"gqa": {"kv_blk": 256, "gqa_qv": True}, "gdn": {"gdn_head_major": True}}


def _exec(T: int, B: int, mods: dict[str, dict[str, Any]]) -> mk.ExecSpec:
    mods = dict(mods)
    core = {"dense_acc_pair": True, "dmm_unroll": 8, **mods.pop("tiles", {})}
    merged = {k: {**_BASE.get(k, {}), **mods.get(k, {})} for k in {*_BASE, *mods}}
    # bank 256 KiB: att_in and lin_in stream as 2 tiles each (unrolled, GA=2).
    tiles = dataclasses.replace(TINY_TILES, bank=256 << 10, **core).with_modules(merged)
    ex = mk.ExecSpec(
        R=R, B=B, T=T, ctx=256, stack_e=False, interpret=True, tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=512 if B <= 8 else 2048),
    )  # fmt: skip
    return mk.with_cache(ex, pool_gather=False)


_OUT: dict[tuple, list] = {}


def _run(T: int, B: int, mods: dict, fused: bool, seed: int = 3) -> list:
    """Bytes of hidden + pools of one paged step (cached per config)."""
    key = (T, B, repr(sorted(mods.items())), fused)
    if key in _OUT:
        return _OUT[key]
    mesh, g, ex = _mesh(), _geom(), _exec(T, B, mods)
    spec = qwen35.qwen35_spec(g)
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(seed), g, ex)
    if T > 1:  # cross a KV page boundary (pos 127 -> 128) on sequence 0
        pos = jnp.asarray(
            [
                (126 if s == 0 else 20 + 10 * s) + t
                for s in range(ex.NS)
                for t in range(T)
            ],
            dtype=jnp.int32,
        )
    st_l = dict(st_l, conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32))
    _, hid_mk, w_mk, _ = mk.pack_inputs(
        pos, hid_l, w_l, st_l, g, ex, mesh=mesh, build_default_pool=True
    )
    pt, ckpt, roff, _ = mk.default_pool_layout(g, ex)
    tp = NamedSharding(mesh, P("tp"))
    pools = tuple(
        jax.device_put(np.asarray(r).copy(), tp)
        for r in mk.states_to_pool(g, ex, st_l, pt, ckpt, roff)
    )
    step, _ = mk.PallasWriter(fused_pool_io=fused).make_step(
        mesh, spec, ex, donate=fused
    )
    out, raw = step(pos, hid_mk, w_mk, pools, pt, ckpt, roff)
    _OUT[key] = [np.asarray(x).view(np.uint8) for x in (out, *raw)]
    return _OUT[key]


def _params(T: int, B: int, mods: dict, fused: bool, module: str) -> Any:
    ex = _exec(T, B, mods)
    if fused:
        ex = mk.with_cache(ex, fused_pool_io=True)
    _, info = qwen35.build(_geom(), ex)
    return info["plan"].module_params[module]


def _assert_same(a: list, b: list) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def _hidden_f32(out: list) -> np.ndarray:
    return out[0].view(np.uint16).view(jnp.bfloat16).astype(np.float32)


# (id, module, option, value, resolved param, other options in both arms, fused)
_EXACT = [
    ("qswap-split", "gqa", "qswap", True, "QSW", {}, False),
    ("qswap-fused", "gqa", "qswap", True, "QSW", {}, True),
    ("gdefer-split", "gqa", "gdefer", True, "GDEF", {}, False),
    ("gdefer-fused", "gqa", "gdefer", True, "GDEF", {}, True),
    ("kvdefer", "gqa", "kvdefer", True, "KVDEF", {"gdefer": True}, True),
    ("wgroup", "gqa", "wgroup", True, "WGROUP", {"gdefer": True}, True),
    (
        "wgroup-hoist",
        "gqa",
        "wgroup",
        True,
        "WGROUP",
        {"gdefer": True, "tiles": {"dwait_hoist": True}},
        True,
    ),
]
# Decode (T=1) rechecks for one option per path; the rest run at the verify
# shape only (interpret-mode cost).
_DECODE_TOO = {"qswap-split", "gdefer-split"}


def _cases(rows: list) -> list:
    return [
        pytest.param(*c[1:], T, B, id=f"{c[0]}-{T}-{B}")
        for c in rows
        for T, B in (((4, 8), (1, 4)) if c[0] in _DECODE_TOO else ((4, 8),))
    ]


def _arms(module: str, opt: str, value: Any, base: dict) -> tuple[dict, dict]:
    base = dict(base)
    core = {"tiles": base.pop("tiles")} if "tiles" in base else {}
    return {module: dict(base), **core}, {module: {**base, opt: value}, **core}


@pytest.mark.parametrize("module,opt,value,field,base,fused,T,B", _cases(_EXACT))
def test_exact_option_is_bit_identical(
    module: str,
    opt: str,
    value: Any,
    field: str,
    base: dict,
    fused: bool,
    T: int,
    B: int,
):
    off, on = _arms(module, opt, value, base)
    assert getattr(_params(T, B, on, fused, module), field)
    assert not getattr(_params(T, B, off, fused, module), field)
    _assert_same(_run(T, B, on, fused), _run(T, B, off, fused))


def test_deferral_gates():
    """gdefer needs a ring slot per att_in tile and an unrolled or paired
    att_in; kvdefer / wgroup need gdefer."""
    assert _params(4, 8, {"gqa": {"gdefer": True}}, False, "gqa").GA == 2
    shallow = {"gqa": {"gdefer": True}, "tiles": {"dmm_unroll": 0, "nbd": 1}}
    with pytest.raises(mk.MegakernelConfigError):
        _params(4, 8, shallow, False, "gqa")  # ring too shallow to plan at all
    no_pair = {
        "gqa": {"gdefer": True},
        "tiles": {"dmm_unroll": 0, "dense_acc_pair": False},
    }
    assert not _params(4, 8, no_pair, False, "gqa").GDEF
    assert not _params(4, 8, {"gqa": {"kvdefer": True}}, False, "gqa").KVDEF
    assert not _params(4, 8, {"gqa": {"wgroup": True}}, False, "gqa").WGROUP


def test_qswap_needs_head_batched_q():
    off_qv = {"gqa": {"qswap": True, "gqa_qv": False}}
    assert not _params(4, 8, off_qv, False, "gqa").QSW
    rope32 = {"gqa": {"qswap": True, "rope_fp32": True}}
    assert not _params(4, 8, rope32, False, "gqa").QSW
