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
"""The GQA and GDN modules under the builder's layer schedule (ARPIPE, AR0S).

A tiny Qwen3.5 step (GDN + GQA mixer, MoE) runs on an 8-rank CPU interpret
mesh through the fused pool I/O at hier=2. AR0S only moves the output
projection's all-reduce packing and sends, so it is bit-identical to the same
schedule without it; ARPIPE defers the input norm's scale past the input
projection, so it is checked against the default schedule within tolerance,
on the plain dense paths and on the claim paths (GQA gdefer, GDN lcv).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import core as mk
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel.profiles.baseline import TINY_TILES

_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 1700
_R, _T, _B = 8, 4, 8


def _spawn_with_cpu_devices(tag: str) -> None:
    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={_R}"]),
    )
    out = subprocess.run(
        [sys.executable or "python3", __file__, tag],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-5000:]


def _geom() -> qwen35.Qwen35Geometry:
    return qwen35.Qwen35Geometry(
        L=4, H=256, I=256, IS=1024, E=128, TOPK=2, NH=8, NKV=2, HD=256, ROT=64,
        KH=8, VH=16, KD=128, VD=128, att_every=4, moe_tp=False, e_slots=0,
    )  # fmt: skip


def _exec(core: dict[str, Any], mods: dict[str, dict[str, Any]]) -> mk.ExecSpec:
    # bank 192 KiB: att_in and lin_in both stream as 2 tiles of 128 rows (the
    # ARPIPE hook needs equal tile shapes for the mixer branches' first kinds)
    base_mods = {
        "gqa": {"kv_blk": 256, "gqa_qv": True},
        "gdn": {"gdn_head_major": True},
    }
    merged = {
        k: {**base_mods.get(k, {}), **mods.get(k, {})} for k in {*base_mods, *mods}
    }
    tiles = dataclasses.replace(
        TINY_TILES,
        bank=192 << 10,
        nbd=10,
        hier=2,
        dense_acc_pair=True,
        dmm_unroll=8,
        **core,
    ).with_modules(merged)
    ex = mk.ExecSpec(
        R=_R, B=_B, T=_T, ctx=256, stack_e=False, interpret=True, tiles=tiles,
        cache=mk.CacheExec(paged_kv=True, num_pages=512),
    )  # fmt: skip
    return mk.with_cache(ex, pool_gather=False)


def _run(core: dict[str, Any], mods: dict | None = None) -> tuple[list, Any]:
    """Bytes of hidden + pools of one fused paged step, and the build info."""
    mods = mods or {}
    mesh = Mesh(np.array(jax.devices("cpu")[:_R]), ("tp",))
    g, ex = _geom(), _exec(core, mods)
    spec = qwen35.qwen35_spec(g)
    pos, hid_l, w_l, st_l = mk.random_logical_inputs(jax.random.key(11), g, ex)
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
    step, info = mk.PallasWriter(fused_pool_io=True).make_step(
        mesh, spec, ex, donate=True
    )
    out, raw = step(pos, hid_mk, w_mk, pools, pt, ckpt, roff)
    return [np.asarray(x).view(np.uint8) for x in (out, *raw)], info


def _same(a: list, b: list) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def _close(a: list, b: list) -> None:
    def h(o: list) -> np.ndarray:
        return o[0].view(np.uint16).view(jnp.bfloat16).astype(np.float32)

    np.testing.assert_allclose(h(a), h(b), atol=3e-2, rtol=3e-2)


def _sched(info: Any) -> Any:
    return info["schedule"]


def _check_schedule() -> None:
    base, binfo = _run({})
    assert not _sched(binfo).arpipe and not _sched(binfo).ar0s
    # AR0S: GQA's att_out and GDN's lin_out feed round 0; bit-identical
    ar0, info = _run({"ar0s": True})
    assert _sched(info).ar0s and not _sched(info).arpipe, _sched(info)
    _same(ar0, base)
    # ARPIPE: att_in / lin_in K halves around the builder's hook (approx)
    arp, info = _run({"arpipe": True})
    assert _sched(info).arpipe, _sched(info)
    _close(arp, base)
    both, info = _run({"arpipe": True, "ar0s": True})
    assert _sched(info).arpipe and _sched(info).ar0s, _sched(info)
    _same(both, arp)
    # claim paths under the hook: GQA gdefer (dbanks_wait + arp_mm) ...
    gd = {"gqa": {"gdefer": True, "kvdefer": True, "wgroup": True}}
    gd_arp, info = _run({"arpipe": True, "ar0s": True}, gd)
    assert _sched(info).arpipe and info["plan"].module_params["gqa"].GDEF
    _close(gd_arp, base)
    # ... and the GDN lcv path
    lcv = {"gdn": {"lcv": True}}
    lcv_arp, info = _run({"arpipe": True, "ar0s": True}, lcv)
    assert _sched(info).arpipe and info["plan"].module_params["gdn"].LCV
    _close(lcv_arp, base)


def test_gqa_gdn_under_arpipe_and_ar0s():
    if len(jax.devices("cpu")) < _R:
        _spawn_with_cpu_devices("schedule")
        return
    _check_schedule()


_CHECKS = {"schedule": _check_schedule}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        _CHECKS[sys.argv[1]]()
