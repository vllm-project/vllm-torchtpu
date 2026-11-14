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
"""Standalone Pallas interpret tests for `vllm_torchtpu.megakernel.core.collectives`."""

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
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as ipc
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.core import AllReduceRing, collectives
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.core.ops import BF16, F32, I32

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


def _stage_round_count(cp: collectives.CommParams) -> int:
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
    traced_slot: bool = False,
    skip: frozenset[str] = frozenset(),
    ring_cls: type[AllReduceRing] = AllReduceRing,
):
    """Run a 2-layer, 2-slot alternating schedule and drain all sends."""
    devs = jax.devices("cpu")
    assert len(devs) >= R, f"Expected >= {R} CPU devices, found {len(devs)}"

    tiles = dataclasses.replace(_TINY_TILES, hier=hier, rsag=rsag, war=war, mb8=2)
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    pc = H // 2 if war == 0 else None
    cp = collectives.comm_params(ex, tiles, H=H, PC=pc)
    scratch = {
        "res": pltpu.VMEM((ex.MB, H), F32),
        **collectives.ring_scratch(cp),
        **collectives.stage_scratch(cp),
    }
    s_names = list(scratch.keys())

    def pallas_body(x0_ref, x1_ref, out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        ar = ring_cls(cp, scr, me, skip=skip)

        if ar.comm:
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
        if traced_slot:

            def _loop_body(step_idx, _):
                layer = lax.div(step_idx, 2)
                slot = lax.rem(step_idx, 2)
                src = lax.select(slot == 0, x0_ref[...], x1_ref[...])
                tot = ar.allreduce(slot, src, layer == 0)
                ar.add_residual(tot)
                return None

            lax.fori_loop(0, 4, _loop_body, None)
        else:
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
    for hier, rsag, war, H in (
        (0, False, 1, 1024),
        (0, True, 1, 1024),
        (3, False, 1, 1024),
    ):
        x0_f64, x1_f64, out_f64, _, cp, _ = _execute_2layer_2slot_ring(
            4, hier=hier, rsag=rsag, war=war, H=H, seed=23, heavy_tail=True
        )
        host_sum_f64 = 2.0 * (np.sum(x0_f64, axis=0) + np.sum(x1_f64, axis=0))
        bound = _bf16_2layer_2slot_bound(x0_f64, x1_f64, _stage_round_count(cp))
        for r in range(4):
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


def _check_r2_traced_slot_and_group_copies_in_process() -> None:
    for hier, rsag in ((0, False), (0, True), (3, False)):
        _, _, out_py, _, _, _ = _execute_2layer_2slot_ring(
            2, hier=hier, rsag=rsag, war=1, H=512, B=4, seed=7, traced_slot=False
        )
        _, _, out_tr, _, _, _ = _execute_2layer_2slot_ring(
            2, hier=hier, rsag=rsag, war=1, H=512, B=4, seed=7, traced_slot=True
        )
        np.testing.assert_array_equal(out_tr, out_py)

    devs = jax.devices("cpu")[:2]
    tiles = dataclasses.replace(_TINY_TILES, hier=0, rsag=False, war=1, mb8=2)
    ex = ExecSpec(B=4, T=1, R=2, ctx=128, tiles=tiles, interpret=True)
    cp = collectives.comm_params(ex, tiles, H=256)
    scr_shapes = [
        pltpu.VMEM((8, 128), BF16),
        pltpu.VMEM((8, 128), BF16),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((1,)),
    ]

    def _group_body(in_ref, out_ref, send_ref, recv_ref, ssem, rsem):
        me = lax.axis_index("tp").astype(I32)
        ar = AllReduceRing(cp, {}, me)
        send_ref[...] = in_ref[...]
        descs = ar.group_copies(lambda _off: send_ref, recv_ref, ssem, rsem, G=2)
        ar.exchange(descs)
        out_ref[...] = recv_ref[...]

    call = pl.pallas_call(
        _group_body,
        out_shape=jax.ShapeDtypeStruct((8, 128), BF16),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=scr_shapes,
        compiler_params=pltpu.CompilerParams(collective_id=1),
        interpret=pltpu.InterpretParams(),
    )
    step = jax.jit(
        jax.shard_map(
            lambda x: call(x[0])[None],
            mesh=Mesh(devs, ("tp",)),
            in_specs=(P("tp"),),
            out_specs=P("tp"),
            check_vma=False,
        )
    )
    x = jax.random.normal(jax.random.PRNGKey(19), (2, 8, 128), dtype=BF16)
    got, exp = np.asarray(step(x), dtype=np.float32), np.asarray(x, dtype=np.float32)
    np.testing.assert_array_equal(got[0], exp[1])
    np.testing.assert_array_equal(got[1], exp[0])

    # Negative control: reusing slot 0 on every layer triggers a DMA race.
    racy_scr = {
        "res": pltpu.VMEM((ex.MB, 256), F32),
        **collectives.ring_scratch(cp),
        **collectives.stage_scratch(cp),
    }
    racy_names = list(racy_scr.keys())

    def _racy_body(x_ref, out_ref, *scr_refs):
        scr = dict(zip(racy_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        ar = AllReduceRing(cp, scr, me)
        barrier = pltpu.get_barrier_semaphore()
        pl.semaphore_signal(
            barrier, 1, device_id=(me ^ 1,), device_id_type=pl.DeviceIdType.MESH
        )
        pl.semaphore_wait(barrier, 1)
        scr["res"][...] = jnp.zeros((ex.MB, 256), F32)

        def _layer_fn(lyr, _):
            val = (x_ref[...] * (lyr + 1).astype(F32)).astype(BF16).astype(F32)
            tot = ar.allreduce(0, val, lyr == 0)
            ar.add_residual(tot)
            return 0

        lax.fori_loop(0, 2, _layer_fn, 0)
        ar.drain_sends()
        out_ref[...] = scr["res"][...]

    racy_call = pl.pallas_call(
        _racy_body,
        out_shape=jax.ShapeDtypeStruct((ex.MB, 256), F32),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[racy_scr[k] for k in racy_names],
        compiler_params=pltpu.CompilerParams(collective_id=0),
        interpret=pltpu.InterpretParams(detect_races=True),
    )
    racy_step = jax.jit(
        jax.shard_map(
            lambda x: racy_call(x[0])[None],
            mesh=Mesh(devs, ("tp",)),
            in_specs=(P("tp"),),
            out_specs=P("tp"),
            check_vma=False,
        )
    )
    x_racy = jax.random.normal(jax.random.PRNGKey(11), (2, ex.MB, 256), dtype=BF16)
    _ = jax.block_until_ready(racy_step(x_racy.astype(F32)))
    assert ipc.races is not None and ipc.races.races_found


def test_allreduce_ring_r1_and_skip_comm():
    """Verify `not self.comm` path (`R=1` and `skip={'comm'}`) rounds in-process."""
    for war in (1, 0):
        x0_f64, x1_f64, out_f64, _, cp, _ = _execute_2layer_2slot_ring(
            1, hier=0, rsag=False, war=war, H=512, B=3, seed=13
        )
        host_sum_f64 = 2.0 * (x0_f64[0] + x1_f64[0])
        bound = _bf16_2layer_2slot_bound(x0_f64, x1_f64, _stage_round_count(cp))
        assert float(np.max(np.abs(out_f64[0] - host_sum_f64))) <= bound

    x0_skip, x1_skip, out_skip, _, cp_skip, _ = _execute_2layer_2slot_ring(
        1,
        hier=0,
        rsag=False,
        war=1,
        H=512,
        B=4,
        seed=19,
        skip=frozenset({"comm"}),
    )
    host_sum_skip = 2.0 * (x0_skip[0] + x1_skip[0])
    bound_skip = _bf16_2layer_2slot_bound(x0_skip, x1_skip, _stage_round_count(cp_skip))
    assert float(np.max(np.abs(out_skip[0] - host_sum_skip))) <= bound_skip


def test_allreduce_ring_modes_vs_float64_host_sum():
    """Verify flat, RS+AG, and hier 1/2/3 at R=4 and R=8 against `numpy.float64`."""
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


def test_allreduce_traced_slot_parity_and_group_copies():
    """Verify traced `slot` parity, 2-rank `group_copies`, and slot-reuse race check."""
    if len(jax.devices("cpu")) < 2:
        _spawn_with_cpu_devices(2, "r2_checks")
        return
    _check_r2_traced_slot_and_group_copies_in_process()


def test_comm_module_import_preserves_host_xla_flags():
    """Importing `collectives` or `test_collectives` must not mutate `XLA_FLAGS`."""
    sentinel = "--xla_cpu_enable_fast_math=false"
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=sentinel)
    code = (
        "import importlib.util, os, sys; "
        "import vllm_torchtpu.megakernel.core.collectives; "
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


@pytest.mark.parametrize(
    "R,hier,rsag,war,H,PR,PC,match",
    [
        (8, 0, True, 1, 512, 4, 512, "multiple of 128"),
        (8, 0, True, 0, 512, None, None, "multiple of 128"),
        (3, 0, False, 1, 768, None, None, "unsupported all-reduce"),
        (3, 0, True, 1, 768, None, None, "unsupported all-reduce"),
        (6, 0, False, 1, 768, None, None, "unsupported all-reduce"),
        (6, 3, False, 1, 768, None, None, "unsupported all-reduce"),
        (4, 1, False, 1, 1024, None, None, "unsupported all-reduce"),
        (4, 2, False, 1, 1024, None, None, "unsupported all-reduce"),
        (8, -1, False, 1, 1024, None, None, "unsupported all-reduce"),
        (8, 99, False, 1, 1024, None, None, "unsupported all-reduce"),
        (8, 2, False, 1, 128, None, None, "hier=2 requires HC2=64 divisible by 128"),
        (8, 0, False, 1, 512, None, 300, "divisible by column chunk PC=300"),
        (8, 0, False, 0, 1024, 4, 512, "folded rows PR=4 must be >= B \\* HPC = 8"),
    ],
)
def test_comm_params_invalid_raises_config_error(
    R: int,
    hier: int,
    rsag: bool,
    war: int,
    H: int,
    PR: int | None,
    PC: int | None,
    match: str,
) -> None:
    tiles = dataclasses.replace(_TINY_TILES, hier=hier, rsag=rsag, war=war, mb8=2)
    ex = ExecSpec(B=4, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    with pytest.raises(MegakernelConfigError, match=match):
        collectives.comm_params(ex, tiles, H=H, PR=PR, PC=PC)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "r8_modes":
            _check_r8_modes_in_process()
        elif sys.argv[1] == "neg_ctrl":
            _check_negative_control_in_process()
        elif sys.argv[1] == "r2_checks":
            _check_r2_traced_slot_and_group_copies_in_process()
