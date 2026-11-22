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
"""Vectorized routed-expert dedup (`MoETunables.dedup_vec`) vs the scalar loop.

`dedup_rows_vec` must give exactly the scalar `dedup_local` (EP) hit list:
the same distinct local experts in the same first-appearance order (`b` major,
`k` minor) and the same count. Checked on the served geometry (E=512, TOPK=10,
R=8, ER=64) for every MK row bucket and rank, plus whole-layer interpret runs
(off vs on, bitwise hidden outputs).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.megakernel.megakernel_helpers import TINY_TILES, FakeGeometry
from vllm_torchtpu.megakernel import vmem
from vllm_torchtpu.megakernel.blocks import MoEBlock
from vllm_torchtpu.megakernel.blocks.moe.topk import dedup_rows_vec
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec

E, TOPK, R = 512, 10, 8
ER = E // R
BUCKETS = (1, 2, 4, 8, 16, 24, 32, 48, 64)


def _scalar_dedup(ids: np.ndarray, B: int, topk: int, me: int, er: int):
    """The scalar `dedup_local` loop (EP): first-appearance local hit list."""
    seen: set[int] = set()
    out: list[int] = []
    for b in range(B):
        for k in range(topk):
            e = int(ids[b, k])
            if e in seen:
                continue
            seen.add(e)
            if e // er == me:
                out.append(e)
    return out


def _ids_block(rows: np.ndarray, mr: int) -> np.ndarray:
    """`[MR, 128]` int32 like `route_topk`'s `ids`; padding rows hold junk ids."""
    b, k = rows.shape
    ids = np.zeros((mr, 128), np.int32)
    ids[:b, :k] = rows
    rng = np.random.default_rng(b * 131 + k)
    if mr > b:  # rows >= B must be ignored
        ids[b:, :k] = rng.integers(0, E, size=(mr - b, k))
    ids[:, k:] = rng.integers(0, E, size=(mr, 128 - k))  # lanes >= TOPK ignored
    return ids


def _routing(kind: str, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "random":
        return np.stack([rng.choice(E, TOPK, replace=False) for _ in range(B)])
    if kind == "alldup":  # every row picks the same experts (in shuffled order)
        base = rng.choice(E, TOPK, replace=False)
        return np.stack([rng.permutation(base) for _ in range(B)])
    if kind == "nodup":  # every pick distinct (as far as E allows)
        perm = rng.permutation(np.tile(np.arange(E), (B * TOPK) // E + 1))
        flat = list(perm[: B * TOPK])
        return np.asarray(flat, np.int64).reshape(B, TOPK)
    if kind == "local_hot":  # rank-local experts dominate, heavy duplicates
        me = seed % R
        pool = np.arange(me * ER, me * ER + 12)
        return np.stack([rng.choice(pool, TOPK, replace=False) for _ in range(B)])
    if kind == "full_rank":  # all ER local experts hit, mixed with remote ones
        me = seed % R
        loc = rng.permutation(np.arange(me * ER, (me + 1) * ER))
        rows = []
        for b in range(B):
            take = list(loc[(b * 5) % ER : (b * 5) % ER + 5])
            rem = [e for e in rng.permutation(E) if e // ER != me][: TOPK - len(take)]
            rows.append(rng.permutation(take + rem))
        return np.stack(rows)
    raise ValueError(kind)


def _vec(ids: np.ndarray, B: int, me: int, topk: int = TOPK, er: int = ER):
    out = np.asarray(
        jax.jit(lambda x, base: dedup_rows_vec(x, B, topk, base, er))(
            jnp.asarray(ids), jnp.int32(me * er)
        )
    )
    n = int(out[1, 0])
    assert (out[1, :] == n).all() and (out[2:] == 0).all()
    return list(out[0, :n]), out


@pytest.mark.parametrize("B", BUCKETS)
@pytest.mark.parametrize("kind", ["random", "alldup", "nodup", "local_hot", "full_rank"])
def test_dedup_rows_vec_matches_scalar(B: int, kind: str):
    for seed in range(3):
        rows = _routing(kind, B, seed + 17 * B)
        ids = _ids_block(rows, -(-B // 8) * 8)
        for me in range(R):
            want = _scalar_dedup(ids, B, TOPK, me, ER)
            got, out = _vec(ids, B, me)
            assert got == want, (kind, B, seed, me)
            assert (out[0, len(want) :] == 0).all()


def test_dedup_rows_vec_idle_rows_and_edges():
    """Idle rows routed like live ones; rank 0 / last rank; no local hit."""
    rng = np.random.default_rng(3)
    for B in (8, 64):
        rows = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(B)])
        rows[B // 2 :] = rows[0]  # idle padding rows: identical junk routing
        ids = _ids_block(rows, B)
        for me in (0, R - 1):
            assert _vec(ids, B, me)[0] == _scalar_dedup(ids, B, TOPK, me, ER)
    rows = np.stack([rng.choice(np.arange(ER, E), TOPK, replace=False) for _ in range(16)])
    ids = _ids_block(rows, 16)
    got, out = _vec(ids, 16, 0)
    assert got == [] and int(out[1, 0]) == 0


def test_dedup_rows_vec_mutations_detected():
    """Wrong variants (ascending-slot order, last appearance, padding rows) fail."""
    rng = np.random.default_rng(11)
    B, me = 32, 3
    rows = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(B)])
    ids = _ids_block(rows, B + 8)  # 8 junk rows past B
    want = _scalar_dedup(ids, B, TOPK, me, ER)
    got = _vec(ids, B, me)[0]
    assert got == want
    assert len(want) > 3
    assert sorted(want) != want  # ascending-slot order would differ
    # last-appearance order (iterate picks backwards) differs
    last = _scalar_dedup(ids[:B, :TOPK][::-1, ::-1].copy(), B, TOPK, me, ER)
    assert last != want
    # including the padding rows changes the list; the vec path ignores them
    assert _scalar_dedup(ids, B + 8, TOPK, me, ER) != want
    assert _vec(ids, B + 8, me)[0] == _scalar_dedup(ids, B + 8, TOPK, me, ER)
    # a corrupted id breaks equality (the comparison is not vacuous)
    bad = ids.copy()
    bad[0, 0] = (want[0] + 1) % E if want[0] % ER != ER - 1 else want[0] - 1
    assert _vec(bad, B, me)[0] != want


def _geom(moe_tp: bool = False, e_slots: int = 0):
    return FakeGeometry(
        L=1, att_every=1, H=256, NH=2, NKV=1, HD=128, E=128, TOPK=8, I=256 if moe_tp else 128,
        IS=256, moe_tp=moe_tp, e_slots=e_slots,
    )


def test_dedup_vec_resolution():
    """Resolves on EP without redundant copies only; scratch unchanged."""
    for moe_tp, e_slots, want in ((False, 0, True), (True, 0, False), (False, 128, False)):
        g = _geom(moe_tp, e_slots)
        moe = MoEBlock()
        ms = ModelSpec("moe_dv", config=g, layer_type=LayerType("lt", slots=(moe,)))
        for B in (4, 32):
            got = {}
            for on in (False, True):
                tiles = dataclasses.replace(
                    TINY_TILES, bank=512 << 10, nbd=8, nbe=6, ebank=32 << 10
                ).with_blocks({"moe": {"dedup_vec": on}})
                ex = ExecSpec(B=B, T=1, R=2, ctx=128, stack_e=True, tiles=tiles)
                cp, scratch = vmem.build_scratch(ms, ex)
                assert cp.block("moe").dedup_vec is (on and want), (moe_tp, e_slots, B)
                got[on] = {k: str(v) for k, v in scratch.items()}
            assert got[False] == got[True]


def test_dedup_vec_env_flag(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    monkeypatch.delenv("VLLM_TPU_MK_MOE_ROUTE_DEDUP_VEC", raising=False)
    assert "VLLM_TPU_MK_MOE_ROUTE_DEDUP_VEC" in envs.environment_variables
    assert wsc.wholestep_tiles(16).block("moe").dedup_vec is False
    monkeypatch.setenv("VLLM_TPU_MK_MOE_ROUTE_DEDUP_VEC", "1")
    assert envs.VLLM_TPU_MK_MOE_ROUTE_DEDUP_VEC is True
    assert wsc.wholestep_tiles(16).block("moe").dedup_vec is True
    assert wsc.wholestep_tiles(8).block("moe").dedup_vec is True
    assert wsc.wholestep_tiles(4).block("moe").dedup_vec is False  # b4: scalar is cheaper


def _compare(B: int, hot: bool) -> None:
    from tests.megakernel.test_moe_rowbucket import _run

    off, mp0, _ = _run(B, False, hot, dedup_vec=False)
    on, mp1, _ = _run(B, False, hot, dedup_vec=True)
    assert mp0.dedup_vec is False and mp1.dedup_vec is True
    assert np.isfinite(off.astype(np.float32)).all()
    np.testing.assert_array_equal(on.view(np.uint16), off.view(np.uint16))


@pytest.mark.slow
@pytest.mark.parametrize("B,hot", [(4, False), (16, False), (32, True), (64, False)])
def test_dedup_vec_layer_bitwise(B: int, hot: bool):
    """One EP MoE layer (R=2, ER=64, PIPE_EP) in interpret mode, off vs on: same bits."""
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=(
            f"{flags} --xla_allow_excess_precision=false --xla_cpu_max_isa=AVX"
        ).strip(),
    )
    code = (
        "from tests.megakernel.test_moe_dedup_vec import _compare; "
        f"_compare({B}, {hot}); print('OK')"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        timeout=1800,
    )
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-3000:] + r.stderr[-6000:]
