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
"""Pallas CPU interpret tests for split-phase collectives and SPMD axis helper."""

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


def _barrier_all(me: jax.Array, R: int) -> None:
    if R > 1:
        barrier = pltpu.get_barrier_semaphore()
        for off in range(1, R):
            pl.semaphore_signal(
                barrier, 1, device_id=(me ^ off,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, R - 1)


def _run_split_vs_sync(
    R: int, *, hier: int, rsag: bool, hier_group: int = 8, num_layers: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Compare 1-AR-site-per-layer split-phase overlap vs sync `allreduce`."""
    devs = jax.devices("cpu")
    assert len(devs) >= R
    H, B = 1024, 4
    tiles = dataclasses.replace(
        _TINY_TILES, hier=hier, hier_group=hier_group, rsag=rsag, war=1, mb8=2
    )
    ex = ExecSpec(B=B, T=1, R=R, ctx=128, tiles=tiles, interpret=True)
    cp = comm.comm_params(ex, tiles, H=H, PR=B, PC=H, n_ar_slots=2)
    scratch = {
        "res": pltpu.VMEM((ex.MB, H), F32),
        **comm.ring_scratch(cp),
        **comm.stage_scratch(cp),
    }
    s_names = list(scratch.keys())

    def _make_step(async_mode: bool):
        def body(x_ref, out_ref, *scr_refs):
            scr = dict(zip(s_names, scr_refs))
            me = lax.axis_index("tp").astype(I32)
            ar = AllReduceRing(cp, scr, me)
            _barrier_all(me, R)
            scr["res"][...] = jnp.zeros((ex.MB, H), F32)
            if async_mode:

                def loop_body(step_i, _):
                    h = comm.start_allreduce(ar, x_ref[step_i], step_idx=step_i)
                    h = comm.advance_allreduce(ar, h)
                    tot = comm.wait_allreduce(ar, h)
                    ar.add_residual(tot)
                    return I32(0)

                lax.fori_loop(0, num_layers, loop_body, I32(0))
                comm.drain_async_sends(ar, num_layers)
            else:
                for step_i in range(num_layers):
                    slot, first = comm.slot_for_step(step_i)
                    tot = ar.allreduce(slot, x_ref[step_i], first)
                    ar.add_residual(tot)
                if num_layers >= 2:
                    ar.drain_sends()
                elif num_layers == 1 and ar.comm:
                    for rnd_i in ar.cp.AR_ROUNDS:
                        for d in ar.ar_copies(0, rnd_i):
                            d.wait_send()
            out_ref[...] = scr["res"][...]

        call = pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((ex.MB, H), F32),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[scratch[k] for k in s_names],
            compiler_params=pltpu.CompilerParams(collective_id=0),
            interpret=pltpu.InterpretParams(),
        )
        mesh = Mesh(devs[:R], ("tp",))
        return jax.jit(
            jax.shard_map(
                lambda x: call(x[0])[None],
                mesh=mesh,
                in_specs=(P("tp"),),
                out_specs=P("tp"),
                check_vma=False,
            )
        )

    xs = jax.random.normal(
        jax.random.PRNGKey(17), (R, num_layers, ex.MB, H), dtype=BF16
    ).astype(F32)
    y_sync = np.asarray(_make_step(False)(xs)[:, :B], dtype=np.float64)
    y_async = np.asarray(_make_step(True)(xs)[:, :B], dtype=np.float64)
    return y_sync, y_async


@pytest.mark.parametrize(
    ("R", "hier", "rsag"),
    [
        (1, 0, False),
        (2, 0, False),
        (2, 0, True),
        (8, 0, False),
        (8, 0, True),
        (8, 1, False),
        (8, 2, False),
        (8, 3, False),
        (8, 4, False),
    ],
)
def test_split_phase_allreduce_matches_sync_bit_exact(
    R: int, hier: int, rsag: bool
) -> None:
    y_sync, y_async = _run_split_vs_sync(R, hier=hier, rsag=rsag, num_layers=3)
    assert np.max(np.abs(y_sync - y_async)) == 0.0


def test_single_layer_dynamic_drain_does_not_hang() -> None:
    """`num_layers=1` uses only `slot=0`; `drain_async_sends(ar, 1)` skips `slot=1`."""
    y_sync, y_async = _run_split_vs_sync(2, hier=0, rsag=False, num_layers=1)
    assert np.max(np.abs(y_sync - y_async)) == 0.0


def test_start_allreduce_rejects_ambiguous_slot_args() -> None:
    ex = ExecSpec(B=4, T=1, R=2, ctx=128, tiles=_TINY_TILES, interpret=True)
    cp = comm.comm_params(ex, _TINY_TILES, H=1024, PR=4, PC=1024)
    ar = AllReduceRing(cp, {}, 0)
    with pytest.raises(MegakernelConfigError):
        comm.start_allreduce(ar, jnp.zeros((16, 1024), F32))
    with pytest.raises(MegakernelConfigError):
        comm.start_allreduce(
            ar, jnp.zeros((16, 1024), F32), step_idx=0, slot=0, first=True
        )


def test_standalone_all_gather_and_reduce_scatter_r8() -> None:
    """Verify MLA `[8, 640]` `all_gather` and `[8, 1024]` `reduce_scatter` on R=8."""
    R, G, stride = 8, 4, 2  # Two interleaved 4-rank DCP groups across 8 ranks
    rows, kv_cols, rs_cols = 8, 640, 1024
    devs = jax.devices("cpu")[:R]
    ag_scr = comm.all_gather_scratch(G, (rows, kv_cols), BF16, prefix="kv")
    rs_scr = comm.reduce_scatter_scratch(G, (rows, rs_cols), BF16, prefix="dcp")
    scr_spec = {**ag_scr, **rs_scr}
    s_names = list(scr_spec.keys())

    def body(kv_ref, part_ref, ag_out_ref, rs_out_ref, *scr_refs):
        scr = dict(zip(s_names, scr_refs))
        me = lax.axis_index("tp").astype(I32)
        _barrier_all(me, R)
        h_ag = comm.start_all_gather(
            kv_ref[...],
            scr["kv_send"],
            scr["kv_recv"],
            scr["kv_ssem"],
            scr["kv_rsem"],
            me,
            G,
            stride=stride,
            mode="stack",
        )
        h_rs = comm.start_reduce_scatter(
            part_ref[...],
            scr["dcp_send"],
            scr["dcp_recv"],
            scr["dcp_ssem"],
            scr["dcp_rsem"],
            me,
            G,
            stride=stride,
        )
        ag_out_ref[...] = comm.wait_all_gather(h_ag)
        rs_out_ref[...] = comm.wait_reduce_scatter(h_rs)

    call = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((G, rows, kv_cols), BF16),
            jax.ShapeDtypeStruct((rows, rs_cols // G), F32),
        ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=(
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ),
        scratch_shapes=[scr_spec[k] for k in s_names],
        compiler_params=pltpu.CompilerParams(collective_id=1),
        interpret=pltpu.InterpretParams(),
    )
    mesh = Mesh(devs, ("tp",))
    step = jax.jit(
        jax.shard_map(
            lambda kv, pt: tuple(y[None] for y in call(kv[0], pt[0])),
            mesh=mesh,
            in_specs=(P("tp"), P("tp")),
            out_specs=(P("tp"), P("tp")),
            check_vma=False,
        )
    )
    k1, k2 = jax.random.split(jax.random.PRNGKey(9))
    kv = jax.random.normal(k1, (R, rows, kv_cols), dtype=BF16)
    pt = jax.random.normal(k2, (R, rows, rs_cols), dtype=BF16).astype(F32)
    got_ag, got_rs = step(kv, pt)
    kv_np = np.asarray(kv, dtype=np.float32)
    pt_np = np.asarray(pt.astype(BF16), dtype=np.float32)
    cw = rs_cols // G
    for r in range(R):
        base = r % stride
        group_ranks = [base + g * stride for g in range(G)]
        gi = (r // stride) % G
        np.testing.assert_array_equal(
            np.asarray(got_ag[r], dtype=np.float32), kv_np[group_ranks]
        )
        expected_rs = np.sum(pt_np[group_ranks, :, gi * cw : (gi + 1) * cw], axis=0)
        assert np.max(np.abs(np.asarray(got_rs[r]) - expected_rs)) == 0.0


def test_spmd_safe_axis_index_eliminates_hlo_partition_id() -> None:
    """`spmd_safe_axis_index` avoids `partition-id` in HLO and restores cleanly."""
    devs = jax.devices("cpu")[:4]
    mesh = Mesh(np.asarray(devs), ("tp",))
    orig = jax.lax.axis_index

    with comm.spmd_safe_axis_index(interpret=True):
        fn = jax.jit(
            jax.shard_map(
                lambda: jax.lax.axis_index("tp")[None],
                mesh=mesh,
                in_specs=(),
                out_specs=P("tp"),
                check_vma=False,
            )
        )
        hlo = fn.lower().as_text()
        assert "partition-id" not in hlo
        np.testing.assert_array_equal(np.asarray(fn()), np.arange(4, dtype=np.int32))
    assert jax.lax.axis_index is orig


def _run_r32_child() -> None:
    y_sync, y_async = _run_split_vs_sync(
        32, hier=4, rsag=False, hier_group=8, num_layers=3
    )
    assert np.max(np.abs(y_sync - y_async)) == 0.0


def test_split_phase_allreduce_r32_hier4_subprocess() -> None:
    _spawn_with_cpu_devices(32, "_r32")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "_r32":
        _run_r32_child()
