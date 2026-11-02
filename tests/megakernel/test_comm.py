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
"""Standalone Pallas interpret tests for `vllm_torchtpu.megakernel.comm`."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import AllReduceRing, comm
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.ops import BF16, F32, I32

_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_CHILD_TIMEOUT_S = 540
_TINY_TILES = Tiles(bank=128 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32)


class _BadPeerOffsetRing(AllReduceRing):
    """Negative-control ring using linearly dependent peer offsets `{5, 2, 5}`."""

    def ar_copies(self, slot, rnd_i: int):
        off = 5 if rnd_i in (0, 2) else (1 << rnd_i)
        bit = lax.rem(self.me >> rnd_i, 2)
        return [
            pltpu.make_async_remote_copy(
                src_ref=self.scr["send"].at[rnd_i, slot],
                dst_ref=self.scr["recv"].at[rnd_i, slot, bit],
                send_sem=self.scr["ar_ssem"].at[rnd_i, slot, 0],
                recv_sem=self.scr["ar_rsem"].at[rnd_i, slot, 0],
                device_id=(self.me ^ jnp.int32(off),),
                device_id_type=pl.DeviceIdType.MESH,
            )
        ]


def _spawn_with_cpu_devices(R: int, tag: str) -> None:
    flags = [
        f
        for f in os.environ.get("XLA_FLAGS", "").split()
        if not f.startswith(_DEVICE_COUNT_FLAG)
    ]
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MALLOC_ARENA_MAX="2",
        XLA_FLAGS=" ".join([*flags, f"{_DEVICE_COUNT_FLAG}={R}"]),
    )
    out = subprocess.run(
        [sys.executable or "python3", __file__, tag],
        env=env,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-4000:]


def _stage_round_count(cp: comm.CommParams) -> int:
    """Count intermediate BF16 roundings inside one `AllReduceRing.allreduce`."""
    if cp.HIER == 3:
        return cp.N_HIER3 - 1
    if cp.HIER == 2:
        return 2
    if cp.HIER == 1 or cp.RSAG:
        return 1
    return 0


def _bf16_2layer_2slot_bound(
    x0_f64: np.ndarray, x1_f64: np.ndarray, n_stage: int
) -> float:
    """Derive worst-case `max_abs` error bound vs `float64` across 4 reductions.

    Each of 4 reductions applies `n_stage` BF16 roundings (`u = 2^-8`) at magnitude
    at most `step_bound`, and 4 `add_residual` calls round running sums bounded by
    `1, 2, 3, 4` times `step_bound` (`1 + 2 + 3 + 4 = 10`).
    """
    u_bf16 = 2.0**-8
    step_bound = float(
        max(
            np.max(np.sum(np.abs(x0_f64), axis=0)),
            np.max(np.sum(np.abs(x1_f64), axis=0)),
        )
    )
    return (4 * n_stage + 10) * u_bf16 * step_bound


def _execute_2layer_2slot_ring(
    R: int,
    *,
    hier: int,
    rsag: bool,
    war: int = 1,
    H: int = 1024,
    B: int = 4,
    seed: int = 42,
    heavy_tail: bool = False,
    ring_cls: type[AllReduceRing] = AllReduceRing,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, ExecSpec, comm.CommParams, dict[str, object]
]:
    """Run a 2-layer, 2-slot alternating schedule and drain all sends."""
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Expected >= {R} CPU devices, found {len(devs)}"

    tiles = dataclasses.replace(_TINY_TILES, hier=hier, rsag=rsag, war=war, mb8=2)
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    pc = H // 2 if war == 0 else H
    pr = ex.B * (H // pc)
    cp = comm.comm_params(ex, tiles, H=H, PR=pr, PC=pc)
    scratch = {
        "res": pltpu.VMEM((ex.MB, H), F32),
        **comm.ring_scratch(cp),
        **comm.stage_scratch(cp),
    }
    s_names = list(scratch.keys())

    def pallas_body(x0_ref, x1_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        ar = ring_cls(cp, scr, me)

        if R > 1:
            barrier = pltpu.get_barrier_semaphore()
            for off in range(1, R):
                pl.semaphore_signal(
                    barrier,
                    1,
                    device_id=(me ^ off,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(barrier, R - 1)

        scr["res"][...] = jnp.zeros((ex.MB, H), F32)
        # 2 layers x 2 alternating slots (`slot = step % 2`, `first = layer == 0`).
        for step_idx in range(4):
            layer = step_idx // 2
            slot = step_idx % 2
            first = jnp.bool_(layer == 0)
            src = x0_ref[...] if slot == 0 else x1_ref[...]
            tot = ar.allreduce(slot, src, first)
            ar.add_residual(tot)

        ar.drain_sends()
        out_ref[...] = scr["res"][...]

    call = pl.pallas_call(
        pallas_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, H), F32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[scratch[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams(),
    )

    mesh = Mesh(devs[:R], ("tp",))
    step = jax.jit(
        jax.shard_map(
            lambda x0, x1: call(x0[0], x1[0])[None],
            mesh=mesh,
            in_specs=(P("tp"), P("tp")),
            out_specs=P("tp"),
            check_vma=False,
        )
    )

    k0, k1, kt = jax.random.split(jax.random.PRNGKey(seed), 3)
    x0 = jax.random.normal(k0, (R, ex.MB, H), dtype=BF16).astype(F32)
    x1 = jax.random.normal(k1, (R, ex.MB, H), dtype=BF16).astype(F32)
    if heavy_tail:
        sp = jax.random.bernoulli(kt, 0.05, (R, ex.MB, H)).astype(F32) * 8.0
        x0 = (x0 + sp).astype(BF16).astype(F32)
        x1 = (x1 - sp).astype(BF16).astype(F32)
        if ex.B < ex.MB:
            x0 = x0.at[:, ex.B :].set(F32(77.0))
            x1 = x1.at[:, ex.B :].set(F32(-55.0))
    out = np.asarray(step(x0, x1), dtype=np.float64)
    return (
        np.asarray(x0[:, : ex.B], dtype=np.float64),
        np.asarray(x1[:, : ex.B], dtype=np.float64),
        out[:, : ex.B],
        ex,
        cp,
        scratch,
    )


def _check_r8_modes_in_process() -> None:
    modes = [
        (0, False, 1, 1024),
        (0, True, 1, 1024),
        (0, True, 0, 2048),
        (1, False, 1, 1024),
        (2, False, 1, 1024),
        (3, False, 1, 1024),
    ]
    for hier, rsag, war, H in modes:
        for seed in (0, 17, 42):
            x0_f64, x1_f64, out_f64, _, cp, _ = _execute_2layer_2slot_ring(
                8, hier=hier, rsag=rsag, war=war, H=H, seed=seed, heavy_tail=True
            )
            host_sum_f64 = 2.0 * (np.sum(x0_f64, axis=0) + np.sum(x1_f64, axis=0))
            bound = _bf16_2layer_2slot_bound(x0_f64, x1_f64, _stage_round_count(cp))
            for r in range(8):
                max_abs = float(np.max(np.abs(out_f64[r] - host_sum_f64)))
                assert max_abs <= bound, (
                    f"seed={seed} hier={hier} rsag={rsag} war={war} rank={r}: "
                    f"max_abs={max_abs:.4e} > bound={bound:.4e}"
                )
    for b_odd in (1, 3, 5):
        for hier, rsag, war, H in (
            (0, True, 1, 1024),
            (0, True, 0, 2048),
            (3, False, 1, 1024),
        ):
            x0_f64, x1_f64, out_f64, _, cp, _ = _execute_2layer_2slot_ring(
                8, hier=hier, rsag=rsag, war=war, H=H, B=b_odd, seed=17, heavy_tail=True
            )
            host_sum_f64 = 2.0 * (np.sum(x0_f64, axis=0) + np.sum(x1_f64, axis=0))
            bound = _bf16_2layer_2slot_bound(x0_f64, x1_f64, _stage_round_count(cp))
            for r in range(8):
                assert float(np.max(np.abs(out_f64[r] - host_sum_f64))) <= bound


def _check_negative_control_in_process() -> None:
    x0_f64, x1_f64, bad_out, _, cp, _ = _execute_2layer_2slot_ring(
        8, hier=3, rsag=False, ring_cls=_BadPeerOffsetRing
    )
    assert np.all(np.isfinite(bad_out))
    host_sum_f64 = 2.0 * (np.sum(x0_f64, axis=0) + np.sum(x1_f64, axis=0))
    bound = _bf16_2layer_2slot_bound(x0_f64, x1_f64, _stage_round_count(cp))
    max_abs = float(np.max(np.abs(bad_out[0] - host_sum_f64)))
    assert max_abs > 10.0 * bound, (
        f"Negative control did not exceed bound: max_abs={max_abs:.4e} <= {bound:.4e}"
    )


def test_allreduce_ring_modes_vs_float64_host_sum():
    """Verify flat, RS+AG, and hier 1/2/3 at R=8 against `numpy.float64` host sum."""
    if len(jax.devices("cpu")) < 8:
        _spawn_with_cpu_devices(8, "r8_modes")
        return
    _check_r8_modes_in_process()


def test_allreduce_ring_wrong_peer_offset_negative_control():
    """Verify a wrong peer offset in `AllReduceRing` fails the `float64` bound."""
    if len(jax.devices("cpu")) < 8:
        _spawn_with_cpu_devices(8, "neg_ctrl")
        return
    _check_negative_control_in_process()


def test_comm_module_import_preserves_host_xla_flags():
    """Importing `test_comm` must not mutate `os.environ['XLA_FLAGS']` in parent."""
    sentinel = "--xla_cpu_enable_fast_math=false"
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=sentinel)
    code = (
        "import importlib.util, os, sys; "
        "p = sys.argv[1]; "
        "spec = importlib.util.spec_from_file_location('tc_probe', p); "
        "assert spec is not None and spec.loader is not None; "
        "mod = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(mod); "
        "got = os.environ.get('XLA_FLAGS', ''); "
        "assert got == sys.argv[2], f'XLA_FLAGS mutated to {got!r}'"
    )
    out = subprocess.run(
        [sys.executable or "python3", "-c", code, __file__, sentinel],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_comm_params_invalid_raises_config_error() -> None:
    tiles = dataclasses.replace(_TINY_TILES, hier=0, rsag=True, war=1, mb8=2)
    ex = ExecSpec(B=4, T=1, R=8, ctx=128, tiles=tiles, interpret=True)
    with pytest.raises(MegakernelConfigError, match="multiple of 128"):
        comm.comm_params(ex, tiles, H=512, PR=4, PC=512)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "r8_modes":
            _check_r8_modes_in_process()
        elif sys.argv[1] == "neg_ctrl":
            _check_negative_control_in_process()
