# SPDX-License-Identifier: Apache-2.0
"""DCP combine by head slices: an all-to-all instead of two all-gathers (DEC-COLL, plan v4).

Today (``qsa_op._dcp_finish`` + ``qsa_rpa.dcp_combine``) every rank of a DCP group all-gathers every member's
partial attention output ``[T, size*H, D]`` f32 and LSE ``[T, size*H]`` f32, then keeps only its own ``H`` heads
(``dynamic_slice(out_pair, rank * H, H, axis=2)``). Three quarters of the gathered bytes (at size 4) are thrown
away, and the LSE costs a second collective.

Here each rank packs, for every destination member ``j``, the head slice ``[j*H, (j+1)*H)`` of its partial output
and of its LSE into one ``[T, W]`` f32 block (``W = H*D`` output lanes plus one 128-lane block whose first ``H``
lanes hold the LSE), and one all-to-all delivers block ``j`` of every member to member ``j``. Member ``j`` then
holds exactly the values today's slice would have selected, in the same source order, and runs today's merge
arithmetic on them (``qsa_rpa.dcp_combine_heads``). Packing and unpacking are pure data movement, so the result is
bitwise today's whenever the merge arithmetic compiles to the same order (checked on device by DCB1).

Two transports:
- ``xla_head_a2a_local``: ``lax.all_to_all`` over the DCP axis.
- ``dcp_head_a2a_local``: one Pallas kernel of remote DMAs (PR 2821's ``dma_all_gather`` pattern): each member
  signals its peers once, then writes block ``j`` straight into member ``j``'s output with one DMA per peer, and
  waits until the other members' blocks have landed. It avoids XLA's sub-slice all-to-all, which a test note
  records as halting SparseCores before libtpu 0.0.51 (lane_shard_tp.md:105).

Both run inside ``shard_map``; the DCP axis is a mesh axis (other mesh axes, e.g. the KV-head group, keep the
caller's own index, so two DCP groups run side by side).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANE = 128
# Barrier semaphore id of this kernel. Others in the tree: fused_moe_decode 0, PR 2821's dma_all_gather 6,
# DEC-FB-2's H8 all-reduce 7.
COLLECTIVE_ID = 5


def packed_width(heads: int, head_dim: int) -> int:
    """Lanes of one destination block: ``heads * head_dim`` output lanes, then one 128-lane LSE block."""
    if (heads * head_dim) % LANE:
        raise ValueError(f"heads * head_dim = {heads * head_dim} is not a whole number of {LANE}-lane tiles")
    if heads > LANE:
        raise ValueError(f"{heads} heads do not fit one {LANE}-lane LSE block")
    return heads * head_dim + LANE


def pack_head_slices(out: jax.Array, lse: jax.Array, size: int) -> jax.Array:
    """``out`` [T, size*H, D] f32 and ``lse`` [T, size*H] f32 -> [size, T, W] f32; block ``j`` holds heads
    ``[j*H, (j+1)*H)``: their outputs in lanes ``[0, H*D)`` (head-major, as ``reshape(T, H*D)``) and their LSE in
    lanes ``[H*D, H*D + H)``. The remaining LSE-block lanes are 0. Values are copied, never computed on."""
    tokens, heads_all, dim = out.shape
    if heads_all % size:
        raise ValueError(f"{heads_all} heads do not split over {size} members")
    heads = heads_all // size
    o = out.astype(jnp.float32).reshape(tokens, size, heads * dim).transpose(1, 0, 2)
    s = lse.astype(jnp.float32).reshape(tokens, size, heads).transpose(1, 0, 2)
    s = jnp.pad(s, ((0, 0), (0, 0), (0, LANE - heads)))
    return jnp.concatenate([o, s], axis=-1)


def unpack_head_slices(recv: jax.Array, heads: int, head_dim: int) -> tuple[jax.Array, jax.Array]:
    """[size, T, W] -> (``out_sel`` [size, T, H, D], ``lse_sel`` [size, T, H]): source member ``i``'s slice for
    this member at index ``i``, which is the layout ``dcp_combine`` builds by slicing the gathered pair."""
    size, tokens, width = recv.shape
    if width != packed_width(heads, head_dim):
        raise ValueError(f"width {width} != packed_width({heads}, {head_dim})")
    out_sel = recv[:, :, : heads * head_dim].reshape(size, tokens, heads, head_dim)
    lse_sel = recv[:, :, heads * head_dim : heads * head_dim + heads]
    return out_sel, lse_sel


def xla_head_a2a_local(x: jax.Array, *, axis: str) -> jax.Array:
    """[size, T, W] -> [size, T, W]: block ``j`` goes to member ``j``; result index ``i`` is member ``i``'s block
    for this member. Call inside shard_map."""
    return lax.all_to_all(x, axis, split_axis=0, concat_axis=0, tiled=False)


def dcp_head_a2a_local(
    x: jax.Array,
    *,
    axis: str,
    size: int,
    collective_id: int = COLLECTIVE_ID,
    interpret: bool = False,
) -> jax.Array:
    """The same all-to-all as ``xla_head_a2a_local``, as one Pallas kernel of remote DMAs. ``size`` is the size of
    mesh axis ``axis``. ``interpret`` (True or ``pltpu.InterpretParams``) runs the Pallas TPU interpreter."""
    if x.shape[0] != size:
        raise ValueError(f"leading dim {x.shape[0]} != {size} members")
    if size == 1:
        return x

    def kernel(x_hbm, o_hbm, send_sem, recv_sem, local_sem):
        me = lax.axis_index(axis)
        peers = [lax.rem(me + i, size) for i in range(1, size)]
        # No block may land before its member runs this kernel: the output is only this call's buffer from then on.
        # A member leaves only after every peer's block has landed, i.e. after every peer passed this barrier, so a
        # round cannot meet the previous call's signals.
        barrier = pltpu.get_barrier_semaphore()
        for peer in peers:
            pl.semaphore_signal(barrier, 1, device_id={axis: peer}, device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, size - 1)
        local = pltpu.make_async_copy(x_hbm.at[me], o_hbm.at[me], local_sem.at[0])
        local.start()
        copies = []
        for peer in peers:
            copy = pltpu.make_async_remote_copy(
                x_hbm.at[peer],
                o_hbm.at[me],
                send_sem.at[0],
                recv_sem.at[0],
                device_id={axis: peer},
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait_send()
        # Every block has one size, so each wait counts one block's bytes.
        for copy in copies:
            copy.wait_recv()
        local.wait()

    if interpret:
        space, out_shape = pl.ANY, jax.ShapeDtypeStruct(x.shape, x.dtype)
        extra = {"interpret": interpret if isinstance(interpret, pltpu.InterpretParams) else pltpu.InterpretParams()}
    else:
        # The DMAs read blocks from HBM and peers write into the output; keep both out of VMEM.
        x = pltpu.with_memory_space_constraint(x, pltpu.HBM)
        space, out_shape, extra = pltpu.HBM, pltpu.HBM(x.shape, x.dtype), {}
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=[pl.BlockSpec(memory_space=space)],
        out_specs=pl.BlockSpec(memory_space=space),
        scratch_shapes=[pltpu.SemaphoreType.DMA((1,))] * 3,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"dcp_head_a2a-n_{size}",
        **extra,
    )(x)


# ---- engine wiring (qsa_op._dcp_finish under VLLM_TPU_QSA_DCP_HEAD_A2A=xla): flat [size*T, W] blocks for torch's
# all_to_all_single, which splits and concatenates the leading axis in rank order. Shapes are static under tracing.
def dcp_a2a_pack_step(out: jax.Array, lse: jax.Array, gate: jax.Array) -> jax.Array:
    """(out [T, size*H, D] f32, lse [T, size*H] f32, gate [T, H, D]) -> [size*T, W] f32 (rows d*T..(d+1)*T go to rank d)."""
    heads = gate.shape[1]
    size = out.shape[1] // heads
    x = pack_head_slices(out, lse, size)
    return x.reshape(size * x.shape[1], x.shape[2])


def dcp_a2a_merge_step(recv: jax.Array, gate: jax.Array) -> jax.Array:
    """(recv [size*T, W] f32 as all_to_all_single returns it, gate [T, H, D]) -> [T, H, D] bf16, bitwise
    ``qsa_rpa.dcp_combine`` on the all-gathered pair."""
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import dcp_combine_heads

    tokens, heads, dim = gate.shape
    size = recv.shape[0] // tokens
    out_sel, lse_sel = unpack_head_slices(recv.reshape(size, tokens, recv.shape[1]), heads, dim)
    return dcp_combine_heads(out_sel, lse_sel, gate)
