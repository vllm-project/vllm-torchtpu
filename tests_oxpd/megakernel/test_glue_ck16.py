# SPDX-License-Identifier: Apache-2.0
"""q38 GDN-CK16 stage 0 (on X-GLUE-1-D14, VLLM_TPU_MK_GDN_CKPT_INK=1): `write_ckpt` skips the f32 widen and the staging stores
for a row whose write id is null (a pad row); a live row is staged and DMA'd as before. CPU interpret.

The pool bytes are already pinned bitwise by tests/megakernel/test_glue_ckpt_ink.py (and the proof's cross-tree comparison
against 11b62cda8). This file pins what the bitwise pool tests cannot see: whether the staging buffers are written. One tiny
pallas kernel calls `write_ckpt` once, then `drain`, with the four slot-0 staging buffers and the pool as aliased outputs that
start as random bytes (copied in at kernel start: VMEM output blocks are not loaded from their aliased inputs):
  * null write id: the staging buffers and the pool come back byte-identical (no widen, no store, no DMA);
  * live write id m: every staging buffer changed, and pool block m's rows 0..399 equal the staged bytes (state chunks, then
    the conv window), every other pool byte unchanged.
The write id's neighbours in the id row are chosen with the opposite null-ness, so a guard that reads the wrong id fails too.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/megakernel/test_glue_ck16.py
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

try:
    import vllm.envs  # noqa: F401
except ImportError:
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _vllm.envs = _envs
    sys.modules.setdefault("vllm", _vllm)
    sys.modules.setdefault("vllm.envs", _envs)

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

from vllm_torchtpu.megakernel.models import qwen38_gdn_ckpt_ink as ink  # noqa: E402
from vllm_torchtpu.megakernel.models.qwen38_gdn_pool import CONV_ROWS, MANAGER_ROWS, SSM_BLOCKS, SSM_ROWS  # noqa: E402

VH, KD, VD, QKV, HR = 6, 128, 128, 1280, 8  # TINY at R = 2 per rank: 6 v-heads, 128 x 128 states, QKV 1280; convb rows 3 + T + 1
KB = 128  # pool rows per kernel block
NB = 4  # manager blocks in the toy pool (write ids 0..3)
F8 = jnp.float8_e4m3fn


def _rand_f8(rng, shape):
    return jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 256, shape, dtype=np.uint8)), F8)


def _u8(x) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


def _write_once(wb_row: list, r: int, t: int, seed: int = 3):
    """One `write_ckpt(o=0, r, d=0, s=0, t)` + `drain`; returns (pool in, pool out, staging in, staging out) as uint8."""
    rng = np.random.default_rng(seed)
    state = jnp.asarray(rng.standard_normal((VH, KD, VD)).astype(np.float32) * 0.05, jnp.bfloat16)
    convb = jnp.asarray(rng.standard_normal((1, HR, QKV)).astype(np.float32) * 0.05)
    wb = jnp.asarray(np.array([wb_row], np.int32))
    pool = _rand_f8(rng, (NB * MANAGER_ROWS // KB, KB, 1, 4, 256))
    stag = [_rand_f8(rng, (SSM_ROWS, 1, 4, 256)) for _ in range(SSM_BLOCKS)] + [_rand_f8(rng, (CONV_ROWS, 1, 4, 256))]
    gp = SimpleNamespace(vh=VH, KD=KD, VD=VD, QKV=QKV)

    def kern(state_ref, convb_ref, wb_ref, pool_in, s0, s1, s2, sc, pool_out, o0, o1, o2, oc, sem, pend, q0, q1, q2, qc):
        # VMEM output blocks start uninitialised (an aliased input is not loaded into them; i1 j-376d0044 showed it): give the staging
        # buffers their 'previous content' explicitly, as the gather leaves it in the real kernel.
        o0[...], o1[...], o2[...], oc[...] = s0[...], s1[...], s2[...], sc[...]
        scr = {"pg_raw00": o0, "pg_raw01": o1, "pg_raw02": o2, "pg_craw0": oc,
               "pg_raw10": q0, "pg_raw11": q1, "pg_raw12": q2, "pg_craw1": qc,
               "convb": convb_ref, "ck_sem": sem, "ck_pend": pend}
        ctx = SimpleNamespace(scr=scr, ex=SimpleNamespace(interpret=True), extra={ink.WRITE_BLOCKS: wb_ref, ink.POOL: pool_in},
                              O={ink.POOL_OUT: pool_out})
        for d in range(ink.DEPTH):
            pend[d] = jnp.int32(0)
        ink.write_ckpt(ctx, gp, 0, r, 0, state_ref, 0, t)
        ink.drain(ctx)

    out_shape = [jax.ShapeDtypeStruct(pool.shape, F8)] + [jax.ShapeDtypeStruct(x.shape, F8) for x in stag]
    scratch = [pltpu.SemaphoreType.DMA((ink.DEPTH, SSM_BLOCKS + 1)), pltpu.SMEM((ink.DEPTH,), jnp.int32)]
    scratch += [pltpu.VMEM((SSM_ROWS, 1, 4, 256), F8) for _ in range(SSM_BLOCKS)] + [pltpu.VMEM((CONV_ROWS, 1, 4, 256), F8)]
    vm, sm, hbm = (pl.BlockSpec(memory_space=x) for x in (pltpu.VMEM, pltpu.SMEM, pltpu.HBM))  # as the megakernel's specs
    outs = pl.pallas_call(kern, out_shape=out_shape, in_specs=[vm, vm, sm, hbm, vm, vm, vm, vm], out_specs=[hbm, vm, vm, vm, vm],
                          input_output_aliases={3: 0, 4: 1, 5: 2, 6: 3, 7: 4}, scratch_shapes=scratch,
                          interpret=pltpu.InterpretParams())(state, convb, wb, pool, *stag)
    return _u8(pool), _u8(outs[0]), [_u8(x) for x in stag], [_u8(x) for x in outs[1:]]


# name: (write-id row, r, t). The neighbours of r have the opposite null-ness (a guard on the wrong id fails).
CASES = {
    "null_r0": ([0, 2, 0, 3], 0, 0),
    "null_r2_t2": ([1, 3, 0, 2], 2, 2),
    "live_r0": ([2, 0, 3, 0], 0, 0),
    "live_r3_t3": ([1, 0, 0, 3], 3, 3),
}


@pytest.mark.parametrize("case", list(CASES))
def test_ck16_write_ckpt_staging(case):
    wb_row, r, t = CASES[case]
    m = wb_row[r]
    p_in, p_out, s_in, s_out = _write_once(wb_row, r, t)
    s_changed = [int((a != b).sum()) for a, b in zip(s_in, s_out)]
    print(f"Q38CK16 {case}: write id {m}; staging bytes changed per buffer {s_changed}; pool bytes changed {int((p_in != p_out).sum())}",
          flush=True)
    if m == 0:
        assert s_changed == [0] * (SSM_BLOCKS + 1), f"{case}: staging written for a null id ({s_changed} bytes changed)"
        assert np.array_equal(p_in, p_out), f"{case}: pool written for a null id"
        return
    assert all(c > 0 for c in s_changed), f"{case}: staging not written for a live id ({s_changed} bytes changed)"
    per = MANAGER_ROWS // KB
    blk = p_out.reshape(NB, per * KB, 1, 4, 256)
    staged = np.concatenate([x.reshape(-1, 1, 4, 256) for x in s_out], axis=0)  # rows 0..383 state chunks, 384..399 conv window
    assert staged.shape[0] == SSM_BLOCKS * SSM_ROWS + CONV_ROWS
    assert np.array_equal(blk[m, : staged.shape[0]], staged), f"{case}: pool block {m} != the staged bytes"
    rest = p_in.reshape(NB, per * KB, 1, 4, 256).copy()
    rest[m, : staged.shape[0]] = staged
    assert np.array_equal(rest.reshape(p_out.shape), p_out), f"{case}: pool bytes outside block {m} rows 0..399 changed"
