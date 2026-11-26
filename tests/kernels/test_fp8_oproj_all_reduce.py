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
"""Static geometry of the fused row-parallel FP8 matmul + all-reduce. The kernel itself
needs a multi-chip slice (remote DMA, barrier semaphore): it is checked on 2x2x2 against
the two production kernels, bitwise, by the t5-oproj-ar instruments. Its one-row build
is checked against the two-shot build on 16 CPU devices in TPU interpret mode (a
subprocess: the host device count must be set before JAX starts)."""

import contextlib
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as k
from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

N_RANKS = 16
ONE_ROW_ROWS = (4, 16)


def test_tile_matches_fp8_skinny_o_proj_config():
    split, tile, nbuf = fp8_skinny.TUNED[(1024, 6144)]
    assert split == "n" and k.TUNED[(1024, 6144)] == (tile, nbuf)


@pytest.mark.parametrize(
    "m,kk,n,cores,ok",
    [
        (4, 1024, 6144, 16, True),
        (16, 1024, 6144, 16, True),
        (16, 1024, 6144, 8, True),
        (32, 1024, 6144, 16, True),
        (64, 1024, 6144, 16, True),
        (65, 1024, 6144, 16, False),  # above 64 rows
        (4, 2048, 16384, 16, False),  # unmeasured shape
        (4, 1024, 6144, 1, False),
        (4, 1024, 6144, 32, False),  # 192-lane chunks
    ],
)
def test_supports(m, kk, n, cores, ok):
    assert k.supports(m, kk, n, cores) is ok


def test_owner_waves_cover_every_owner_once_after_its_last_column():
    waves = k.owner_waves(16, 6144, 2048)
    assert waves == [list(range(0, 5)), list(range(5, 10)), list(range(10, 16))]
    own = 6144 // 16
    for tile, owners in enumerate(waves):
        for j in owners:
            assert ((j + 1) * own - 1) // 2048 == tile


def test_one_row_build_needs_a_row_that_fills_a_row_tile():
    assert k.one_row_supported(6144)
    assert not k.one_row_supported(6144 + 128)


def _one_row_worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    pltpu.get_tpu_info = tpu7x
    tpu_info.get_tpu_info = tpu7x

    def interpret(fn, *args):
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out = jax.tree.map(
                lambda a: np.asarray(jax.block_until_ready(a)), fn(*args)
            )
        return out, bool(interpret_pallas_call.races.races_found)

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    mesh = Mesh(np.asarray(devices), ("x",))
    kk, n = 1024, 6144
    rank = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)

    def build(one_row):
        def body(x, w, ws, r, nv):
            return k.fp8_matmul_all_reduce(
                x,
                w,
                ws,
                r.reshape(1),
                num_cores=N_RANKS,
                axis="x",
                waves=True,
                n_valid=nv.reshape(1) if one_row else None,
            )

        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("x"),) * 4 + (P(),),
                out_specs=P("x"),
                check_vma=False,
            )
        )

    two, one = build(False), build(True)
    result = {}
    for m in ONE_ROW_ROWS:
        k1, k2, k3 = jax.random.split(jax.random.key(m), 3)
        x = (jax.random.normal(k1, (N_RANKS * m, kk)) * 2).astype(jnp.bfloat16)
        w = (jax.random.normal(k2, (N_RANKS * kk, n)) * 0.5).astype(jnp.float8_e4m3fn)
        ws = jax.random.uniform(k3, (N_RANKS * n,), jnp.float32, 0.5, 1.5)
        case = {}
        for nv in (1, 2, m):
            args = (x, w, ws, rank, jnp.asarray([nv], jnp.int32))
            a, races_a = interpret(two, *args)
            b, races_b = interpret(one, *args)
            a = a.reshape(N_RANKS, m, n).view(np.uint16)
            b = b.reshape(N_RANKS, m, n).view(np.uint16)
            if nv == 1:
                # Row 0 bitwise; padding rows +0 (bf16 bits 0).
                same = np.array_equal(a[:, 0], b[:, 0]) and not b[:, 1:].any()
            else:
                same = np.array_equal(a, b)
            case[nv] = bool(same and not races_a and not races_b)
        result[str(m)] = case
    # Verify m=32 pair-first hierarchical exchange (both one_row=False and True).
    m_d2d = 32
    k1, k2, k3 = jax.random.split(jax.random.key(m_d2d), 3)
    x32 = (jax.random.normal(k1, (N_RANKS * m_d2d, kk)) * 2).astype(jnp.bfloat16)
    w32 = (jax.random.normal(k2, (N_RANKS * kk, n)) * 0.5).astype(jnp.float8_e4m3fn)
    ws32 = jax.random.uniform(k3, (N_RANKS * n,), jnp.float32, 0.5, 1.5)
    q_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    x_r = x32.reshape(N_RANKS, m_d2d, kk)
    sc = jnp.max(jnp.abs(x_r), axis=2, keepdims=True).astype(jnp.float32) / q_max
    sc = jnp.where(sc == 0, 1.0, sc)
    xq = jnp.clip(x_r.astype(jnp.float32) / sc, -q_max, q_max).astype(jnp.float8_e4m3fn)
    part = (
        jax.lax.dot_general(
            xq.astype(jnp.float32),
            w32.reshape(N_RANKS, kk, n).astype(jnp.float32),
            (((2,), (1,)), ((0,), (0,))),
        )
        * sc
        * ws32.reshape(N_RANKS, 1, n)
    ).astype(jnp.bfloat16)
    ref32 = np.asarray(jnp.sum(part.astype(jnp.float32), axis=0))
    d2d_case = {}
    for nv in (1, m_d2d):
        args32 = (x32, w32, ws32, rank, jnp.asarray([nv], jnp.int32))
        out_two, races_two = interpret(two, *args32)
        out_one, races_one = interpret(one, *args32)
        out_two = out_two.reshape(N_RANKS, m_d2d, n)
        out_one = out_one.reshape(N_RANKS, m_d2d, n)
        same_ranks = np.array_equal(
            out_two.view(np.uint16),
            np.broadcast_to(out_two[:1].view(np.uint16), out_two.shape),
        ) and np.array_equal(
            out_one.view(np.uint16),
            np.broadcast_to(out_one[:1].view(np.uint16), out_one.shape),
        )
        rel_two = float(
            np.linalg.norm(out_two[0].astype(np.float32) - ref32)
            / np.linalg.norm(ref32)
        )
        if nv == 1:
            one_ok = bool(
                np.array_equal(
                    out_one[:, 0].view(np.uint16), out_two[:, 0].view(np.uint16)
                )
                and not out_one[:, 1:].view(np.uint16).any()
            )
        else:
            one_ok = bool(
                np.array_equal(out_one.view(np.uint16), out_two.view(np.uint16))
            )
        d2d_case[nv] = bool(
            same_ranks and one_ok and rel_two < 5e-3 and not races_two and not races_one
        )
    result["d2d_32"] = d2d_case

    # Verify fused add+RMSNorm epilogue on two-shot (m=8) and pair-first (m=32)
    # matches f32-intermediate weight multiply (XLA TPU excess precision) bitwise
    # and differs cleanly from bf16-truncated intermediate rounding.
    from vllm_torchtpu.kernels import (  # noqa: PLC0415 -- worker subprocess imports JAX lazily
        fused_add_rmsnorm,
    )

    eps = 1e-6

    def ref_add_rmsnorm_f32_mul(x_in, res_in, nw_in):
        s_f32 = x_in.astype(jnp.float32) + res_in.astype(jnp.float32)
        new_res = s_f32.astype(x_in.dtype)
        inv = jax.lax.rsqrt(
            jnp.mean(s_f32 * s_f32, axis=-1, keepdims=True) + jnp.float32(eps)
        )
        y_out = ((s_f32 * inv) * nw_in.astype(jnp.float32)).astype(x_in.dtype)
        return y_out, new_res

    def build_rn(one_row):
        def body(x, w, ws, r, nv, res, nw):
            return k.fp8_matmul_all_reduce(
                x,
                w,
                ws,
                r.reshape(1),
                num_cores=N_RANKS,
                axis="x",
                waves=True,
                n_valid=nv.reshape(1) if one_row else None,
                residual=res,
                norm_w=nw,
                eps=eps,
            )

        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("x"),) * 4 + (P(), P("x"), P("x")),
                out_specs=(P("x"), P("x")),
                check_vma=False,
            )
        )

    rn_two, rn_one = build_rn(False), build_rn(True)
    rn_case = {}
    for m_rn in (8, 32):
        k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(100 + m_rn), 5)
        xr = (jax.random.normal(k1, (N_RANKS * m_rn, kk)) * 2).astype(jnp.bfloat16)
        wr = (jax.random.normal(k2, (N_RANKS * kk, n)) * 0.5).astype(jnp.float8_e4m3fn)
        wsr = jax.random.uniform(k3, (N_RANKS * n,), jnp.float32, 0.5, 1.5)
        res_base = (jax.random.normal(k4, (m_rn, n)) * 0.5).astype(jnp.bfloat16)
        nw_base = jax.random.uniform(k5, (n,), jnp.float32, 0.5, 1.5).astype(
            jnp.bfloat16
        )
        res_sh = jnp.tile(res_base, (N_RANKS, 1))
        nw_sh = jnp.tile(nw_base, (N_RANKS,))
        m_ok = {}
        for nv in (1, m_rn):
            nv_arr = jnp.asarray([nv], jnp.int32)
            base_out, r_base = interpret(one, xr, wr, wsr, rank, nv_arr)
            ref_y, ref_res = ref_add_rmsnorm_f32_mul(
                jnp.asarray(base_out), res_sh, nw_base
            )
            trunc_y, _ = fused_add_rmsnorm.reference(
                jnp.asarray(base_out), res_sh, nw_base, eps
            )
            ref_y_u16 = np.asarray(ref_y).reshape(N_RANKS, m_rn, n).view(np.uint16)
            trunc_y_u16 = np.asarray(trunc_y).reshape(N_RANKS, m_rn, n).view(np.uint16)
            ref_res_u16 = np.asarray(ref_res).reshape(N_RANKS, m_rn, n).view(np.uint16)
            # Discriminating oracle check: f32-mul and bf16-truncated references
            # must diverge significantly on these inputs (>15% of elements).
            trunc_diff_frac = float(np.mean(ref_y_u16 != trunc_y_u16))
            (y_one, res_one), r_one = interpret(
                rn_one, xr, wr, wsr, rank, nv_arr, res_sh, nw_sh
            )
            y_one_u16 = np.asarray(y_one).reshape(N_RANKS, m_rn, n).view(np.uint16)
            res_one_u16 = np.asarray(res_one).reshape(N_RANKS, m_rn, n).view(np.uint16)
            ok = bool(
                trunc_diff_frac > 0.15
                and np.array_equal(y_one_u16, ref_y_u16)
                and np.array_equal(res_one_u16, ref_res_u16)
                and not r_base
                and not r_one
            )
            if nv == m_rn:
                (y_two, res_two), r_two = interpret(
                    rn_two, xr, wr, wsr, rank, nv_arr, res_sh, nw_sh
                )
                ok = ok and bool(
                    np.array_equal(
                        np.asarray(y_two).reshape(N_RANKS, m_rn, n).view(np.uint16),
                        ref_y_u16,
                    )
                    and np.array_equal(
                        np.asarray(res_two).reshape(N_RANKS, m_rn, n).view(np.uint16),
                        ref_res_u16,
                    )
                    and not r_two
                )
            m_ok[nv] = ok
        rn_case[str(m_rn)] = m_ok
    result["rmsnorm"] = rn_case
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def one_row_on_16_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}"
        " --xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("rows", ONE_ROW_ROWS)
def test_one_row_build_is_bitwise_the_two_shot(one_row_on_16_devices, rows):
    """n_valid 1: row 0 bitwise, padding +0; 2 and all rows: the whole output."""
    assert all(one_row_on_16_devices[str(rows)].values())


def test_should_reduce_m32_pair_first_without_races(one_row_on_16_devices):
    assert all(one_row_on_16_devices["d2d_32"].values())


@pytest.mark.parametrize("rows", ["8", "32"])
def test_should_match_add_rmsnorm_f32_mul_reference_bitwise(
    one_row_on_16_devices, rows
):
    assert all(one_row_on_16_devices["rmsnorm"][rows].values())


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _one_row_worker()
