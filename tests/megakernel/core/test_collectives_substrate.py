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
"""Unit tests for `PeerTopology`, `RemoteCopyBatch`, and SPMD axis helpers."""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from jax import export, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.core.collectives import (
    PeerTopology,
    RemoteCopyBatch,
    kernel_axis_index,
    make_remote_copy_batch,
    mesh_axis_index,
)

if TYPE_CHECKING:
    from jax._src.state.types import AbstractRef, TransformedRef
    from jax.typing import DTypeLike

    RefLike = jax.Array | AbstractRef | TransformedRef

_EXPECTED_NUM_GROUPS_32_BY_8: int = 4
_EXPECTED_NEXT_PEER_FROM_13: int = 14
_EXPECTED_PREV_PEER_FROM_13: int = 12
_MIN_CPU_DEVICES: int = 8
_WORLD_SIZE_32: int = 32


def _run_32rank_xor_allreduce_worker() -> None:
    """Execute 32-rank two-stage intra-group + inter-group XOR reduction on CPU."""
    devs = jax.devices("cpu")
    assert len(devs) == _WORLD_SIZE_32
    mesh32 = Mesh(devs[:_WORLD_SIZE_32], ("tp",))

    def step32(x: jax.Array) -> jax.Array:
        """Single-step shard_map body for 32-rank two-stage reduction."""

        def kern32(
            x_ref: RefLike,
            o_ref: RefLike,
            intra_refs: tuple[RefLike, RefLike, RefLike, RefLike],
            inter_refs: tuple[RefLike, RefLike, RefLike, RefLike],
        ) -> None:
            """Combine 7-peer intra-group and 3-peer inter-group copies."""
            s_intra, r_intra, ss_intra, rs_intra = intra_refs
            s_inter, r_inter, ss_inter, rs_inter = inter_refs
            me = kernel_axis_index("tp")
            topo = PeerTopology(rank=me, world_size=32, group_size=8)
            for idx in range(7):
                s_intra[idx] = x_ref[...]
            b_intra = make_remote_copy_batch(
                [s_intra.at[idx] for idx in range(7)],
                [r_intra.at[idx] for idx in range(7)],
                [topo.ring_next_peer(idx + 1) for idx in range(7)],
                [ss_intra.at[idx] for idx in range(7)],
                [rs_intra.at[idx] for idx in range(7)],
            )
            b_intra.start()
            b_intra.wait()
            group_sum = x_ref[...] + jnp.sum(r_intra[...], axis=0)

            for idx in range(3):
                s_inter[idx] = group_sum
            b_inter = make_remote_copy_batch(
                [s_inter.at[0], s_inter.at[1], s_inter.at[2]],
                [r_inter.at[0], r_inter.at[1], r_inter.at[2]],
                [
                    topo.inter_xor_peer(1),
                    topo.inter_xor_peer(2),
                    topo.inter_xor_peer(3),
                ],
                [ss_inter.at[0], ss_inter.at[1], ss_inter.at[2]],
                [rs_inter.at[0], rs_inter.at[1], rs_inter.at[2]],
            )
            b_inter.start()
            b_inter.wait()
            o_ref[...] = group_sum + jnp.sum(r_inter[...], axis=0)

        return pl.pallas_call(
            kern32,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                (
                    pltpu.VMEM((7, 8, 128), jnp.float32),
                    pltpu.VMEM((7, 8, 128), jnp.float32),
                    pltpu.SemaphoreType.DMA((7,)),
                    pltpu.SemaphoreType.DMA((7,)),
                ),
                (
                    pltpu.VMEM((3, 8, 128), jnp.float32),
                    pltpu.VMEM((3, 8, 128), jnp.float32),
                    pltpu.SemaphoreType.DMA((3,)),
                    pltpu.SemaphoreType.DMA((3,)),
                ),
            ],
            compiler_params=pltpu.CompilerParams(collective_id=1),
            interpret=pltpu.InterpretParams(),
        )(x[0])[None]

    sharded32 = jax.jit(
        jax.shard_map(
            step32,
            mesh=mesh32,
            in_specs=P("tp"),
            out_specs=P("tp"),
            check_vma=False,
        )
    )
    inp32 = (jnp.arange(32 * 8 * 128, dtype=jnp.float32) % 17).reshape(32, 8, 128)
    out32 = sharded32(inp32)
    expected32 = jnp.sum(inp32, axis=0)
    for r in range(_WORLD_SIZE_32):
        assert jnp.allclose(out32[r], expected32, atol=1e-5, rtol=1e-5)


def _assert_peer_topology_validation_errors(top0: PeerTopology) -> None:
    """Verify ValueError and TypeError guards on invalid topology inputs."""
    for bad_w, bad_g in ((0, 8), (8, 0), (12, 8), (24, 8)):
        with pytest.raises(ValueError, match=r"world_size|group_size|num_groups"):
            PeerTopology(rank=0, world_size=bad_w, group_size=bad_g)
    for bad_rank in (-1, 32):
        with pytest.raises(ValueError, match="out of bounds"):
            PeerTopology(rank=bad_rank, world_size=32, group_size=8)
    bool_true: object = True
    bool_false: object = False
    with pytest.raises(TypeError, match="cannot be bool"):
        PeerTopology(rank=bool_true, world_size=8, group_size=8)
    with pytest.raises(ValueError, match="out of bounds"):
        top0.intra_peer(-1)
    with pytest.raises(ValueError, match="out of bounds"):
        top0.intra_peer(8)
    with pytest.raises(TypeError, match="cannot be bool"):
        top0.intra_peer(bool_false)
    with pytest.raises(ValueError, match="out of bounds"):
        top0.inter_xor_peer(-1)
    with pytest.raises(ValueError, match="out of bounds"):
        top0.inter_xor_peer(4)
    with pytest.raises(TypeError, match="cannot be bool"):
        top0.inter_xor_peer(bool_true)


def test_peer_topology_intra_and_inter_xor_coordinates() -> None:
    """Verify intra-group and inter-group XOR coordinate arithmetic."""
    assert dataclasses.is_dataclass(PeerTopology)
    top0 = PeerTopology(rank=0, world_size=32, group_size=8)
    with pytest.raises(dataclasses.FrozenInstanceError):
        top0.world_size = 16

    for r in range(32):
        topo = PeerTopology(rank=r, world_size=32, group_size=8)
        assert topo.num_groups == _EXPECTED_NUM_GROUPS_32_BY_8
        assert topo.intra_rank == r % 8
        assert topo.group_idx == r // 8
        for s in range(8):
            assert topo.intra_peer(s) == (r // 8) * 8 + s
        assert topo.ring_next_peer() == (r // 8) * 8 + ((r + 1) % 8)
        assert topo.ring_next_peer(3) == (r // 8) * 8 + ((r + 3) % 8)
        assert topo.ring_prev_peer() == (r // 8) * 8 + ((r - 1) % 8)
        assert topo.ring_prev_peer(2) == (r // 8) * 8 + ((r - 2) % 8)
        visited_groups = {topo.group_idx}
        for mask in (1, 2, 3):
            peer = topo.inter_xor_peer(mask)
            assert peer == ((r // 8) ^ mask) * 8 + (r % 8)
            peer_topo = PeerTopology(rank=peer, world_size=32, group_size=8)
            assert peer_topo.intra_rank == topo.intra_rank
            assert peer_topo.inter_xor_peer(mask) == r
            visited_groups.add(peer_topo.group_idx)
        assert visited_groups == {0, 1, 2, 3}

    for world_size, group_size in ((8, 8), (16, 4)):
        for r in range(world_size):
            topo = PeerTopology(rank=r, world_size=world_size, group_size=group_size)
            assert topo.num_groups == world_size // group_size
            assert topo.intra_peer(topo.intra_rank) == r
            assert topo.inter_xor_peer(0) == r

    # Verify traced JAX scalar int32 rank compatibility inside jit.
    @jax.jit
    def traced_peers(r: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute next, previous, and XOR peers for traced scalar rank `r`."""
        t = PeerTopology(rank=r, world_size=32, group_size=8)
        return t.ring_next_peer(), t.ring_prev_peer(), t.inter_xor_peer(2)

    nxt, prv, xpeer = traced_peers(jnp.int32(13))
    assert int(nxt) == _EXPECTED_NEXT_PEER_FROM_13
    assert int(prv) == _EXPECTED_PREV_PEER_FROM_13
    assert int(xpeer) == ((1 ^ 2) * 8 + 5)

    _assert_peer_topology_validation_errors(top0)


def test_remote_copy_batch_split_phase_arbitrary_dtype() -> None:
    """Verify split-phase `RemoteCopyBatch` across dtypes and 32-rank XOR tree."""
    devs = jax.devices("cpu")
    assert len(devs) >= _MIN_CPU_DEVICES
    mesh8 = Mesh(devs[:_MIN_CPU_DEVICES], ("tp",))

    for dt in (jnp.bfloat16, jnp.float8_e4m3fn, jnp.int16, jnp.float32):

        def step_fn(x: jax.Array, _dt: DTypeLike = dt) -> jax.Array:
            """Execute one split-phase ring copy step on `mesh8` for dtype `_dt`."""

            def kern(
                x_ref: RefLike,
                out_ref: RefLike,
                buf_refs: tuple[RefLike, RefLike],
                sem_refs: tuple[RefLike, RefLike],
            ) -> None:
                """Issue remote DMA copy to next ring peer while copying locally."""
                send_ref, recv_ref = buf_refs
                ssem, rsem = sem_refs
                me = kernel_axis_index("tp")
                topo = PeerTopology(rank=me, world_size=8, group_size=8)
                send_ref[...] = x_ref[...]
                batch = make_remote_copy_batch(
                    [send_ref],
                    [recv_ref],
                    [topo.ring_next_peer()],
                    [ssem],
                    [rsem],
                    axis_name="tp",
                )
                assert type(batch) is RemoteCopyBatch
                batch.start()
                # Overlapped local compute while remote copy is in flight.
                out_ref[0] = x_ref[...]
                batch.wait_recv()
                batch.wait_send()
                out_ref[1] = recv_ref[...]

            stacked_y = pl.pallas_call(
                kern,
                out_shape=jax.ShapeDtypeStruct((2, 8, 128), _dt),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    (pltpu.VMEM((8, 128), _dt), pltpu.VMEM((8, 128), _dt)),
                    (pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.DMA(())),
                ],
                compiler_params=pltpu.CompilerParams(collective_id=0),
                interpret=pltpu.InterpretParams(),
            )(x[0])
            return stacked_y[None]

        sharded = jax.jit(
            jax.shard_map(
                step_fn,
                mesh=mesh8,
                in_specs=P("tp"),
                out_specs=P("tp"),
                check_vma=False,
            )
        )
        inp = (
            (jnp.arange(8 * 8 * 128, dtype=jnp.int32) % 64)
            .reshape(8, 8, 128)
            .astype(dt)
        )
        out = sharded(inp)
        assert jnp.array_equal(out[:, 0], inp)
        assert jnp.array_equal(out[:, 1], jnp.roll(inp, shift=1, axis=0))

    # 32-rank two-stage intra-group + inter-group XOR reduction in spawned process.
    prev_flags = os.environ.get("XLA_FLAGS")
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    try:
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(target=_run_32rank_xor_allreduce_worker)
        proc.start()
        proc.join(timeout=120)
        assert proc.exitcode == 0
    finally:
        if prev_flags is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = prev_flags

    with pytest.raises(ValueError, match="equal length"):
        make_remote_copy_batch([1, 2], [1], [0], [1], [1])
    empty = make_remote_copy_batch([], [], [], [], [])
    assert empty.copies == ()
    assert empty.ops == ()
    empty.start()
    empty.wait()


def test_mesh_axis_index_eliminates_hlo_partition_id() -> None:
    """Verify `mesh_axis_index` avoids outer HLO partition-id and runs on CPU."""
    devs = jax.devices("cpu")
    mesh8 = Mesh(devs[:_MIN_CPU_DEVICES], ("tp",))

    @jax.jit
    @jax.shard_map(mesh=mesh8, in_specs=(), out_specs=P("tp"), check_vma=False)
    def raw_axis_fn() -> jax.Array:
        """Return axis index via raw `lax.axis_index` outside Pallas."""
        return jnp.expand_dims(lax.axis_index("tp").astype(jnp.int32), 0)

    raw_hlo = raw_axis_fn.lower().compiler_ir("hlo").as_hlo_text()
    assert "partition-id" in raw_hlo

    @jax.jit
    @jax.shard_map(mesh=mesh8, in_specs=(), out_specs=P("tp"), check_vma=False)
    def pallas_axis_cpu() -> jax.Array:
        """Return axis index via `mesh_axis_index` in CPU interpret mode."""
        return jnp.expand_dims(mesh_axis_index("tp", interpret=True), 0)

    coords = pallas_axis_cpu()
    assert jnp.array_equal(coords, jnp.arange(8, dtype=jnp.int32))

    @jax.jit
    @jax.shard_map(mesh=mesh8, in_specs=(), out_specs=P("tp"), check_vma=False)
    def pallas_axis_tpu() -> jax.Array:
        """Return axis index via `mesh_axis_index` lowered for TPU."""
        return jnp.expand_dims(mesh_axis_index("tp", interpret=False), 0)

    exported = export.export(pallas_axis_tpu, platforms=["tpu"])()
    mlir_text = exported.mlir_module()
    assert "tpu_custom_call" in mlir_text
    assert "stablehlo.partition_id" not in mlir_text


def test_remote_copy_batch_tpu_lowering() -> None:
    """Verify make_remote_copy_batch lowers to TPU tpu_custom_call."""
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU for jax.export platforms=['tpu']")
    devs = jax.devices()
    mesh = Mesh(devs[:4], ("tp",))

    def step_fn(x: jax.Array) -> jax.Array:
        def kern(
            x_ref: RefLike,
            out_ref: RefLike,
            buf_refs: tuple[RefLike, RefLike],
            sem_refs: tuple[RefLike, RefLike],
        ) -> None:
            send_ref, recv_ref = buf_refs
            ssem, rsem = sem_refs
            me = kernel_axis_index("tp")
            topo = PeerTopology(rank=me, world_size=4, group_size=4)
            send_ref[...] = x_ref[...]
            batch = make_remote_copy_batch(
                [send_ref],
                [recv_ref],
                [topo.ring_next_peer()],
                [ssem],
                [rsem],
                axis_name="tp",
            )
            assert type(batch) is RemoteCopyBatch
            batch.start()
            batch.wait_recv()
            batch.wait_send()
            out_ref[...] = recv_ref[...]

        out = pl.pallas_call(
            kern,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                (pltpu.VMEM((8, 128), jnp.float32), pltpu.VMEM((8, 128), jnp.float32)),
                (pltpu.SemaphoreType.DMA(()), pltpu.SemaphoreType.DMA(())),
            ],
            compiler_params=pltpu.CompilerParams(collective_id=0),
            interpret=False,
        )(x[0])
        return out[None]

    @jax.jit
    @jax.shard_map(mesh=mesh, in_specs=P("tp"), out_specs=P("tp"), check_vma=False)
    def sharded(x: jax.Array) -> jax.Array:
        return step_fn(x)

    exported = export.export(sharded, platforms=["tpu"])(
        jax.ShapeDtypeStruct((4, 8, 128), jnp.float32)
    )
    mlir_text = exported.mlir_module()
    assert "tpu_custom_call" in mlir_text
