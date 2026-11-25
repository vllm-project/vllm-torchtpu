# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/dspark_mla_kernel.py
"""The fused TP32 kernel of the MLA DSpark draft (``kernels/dspark_mla.py``).

One call per decode step after the verify, over ``R`` requests with four or
eight verify rows each, on the decoder's 32-rank mesh:

1. **Acceptance.** The verify's vocabulary-sharded logits give each row's
   target (one 31-peer exchange); the accepted count is the leading drafts
   that match, the bonus token (the next anchor) the target after them
   (either may be forced by the caller), as in ``kernels/dspark.py``.
2. **Context.** The verify rows' aux features -> ``context_norm(
   context_proj(aux))`` (host-sharded fc and reductions, as there).
3. **Five layers** over the block (the anchor's embedding then mask
   embeddings, at the anchor's position on): RMSNorm, the fused q_a | kv_a
   projection (each rank computes one 128-lane tile and the ranks exchange
   them; the context rows' kv_a for every layer once, before the layers),
   q_a_layernorm,
   this rank's two heads' q_b, RoPE, the absorbed queries; the block and
   context rows' latents; DCP attention (``mla_dcp.DcpAttention``) over the
   draft pools' pages before the verify's first position, with the
   request's new keys (its accepted context rows and its block rows,
   non-causal); W_UV, o_proj (reduce-scattered), the MLP.
4. **Drafts.** The draft's final norm, the target's LM head (vocabulary
   sharded) and exact left-to-right Markov sampling.

Outputs: each request's ``[128]`` int32 vector (``dspark.OUT_*``
lanes: drafts, targets, count, bonus, margins) and the verify rows'
context latents ``[T, layers, 640]`` for the caller to commit to the draft
pools (``kernels/paged.py`` ``commit_latent_rows``).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Self, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from ..common import collectives32
from ..common.constants import (
    AXIS_TP,
    DSPARK_MLA_MAX_ROWS,
    FC_IN,
    MLA_LATENT_ROW_LANES,
    MLP_PADDED,
    ROW_PACKED,
    TP_SIZE,
    VREG_LANE_COUNT,
)
from ..common.constants import (
    DSPARK_HIDDEN_CHUNK as CHUNK,
)
from . import mla_dcp
from .decode_megakernel import PHASE_SCOPES, dot, phase_end, phase_start
from .dspark import (
    HOST_CHUNKS,
    MAX_REQUESTS,
    MIN_BLOCK_LEN,
    NEG,
    OUT_BONUS,
    OUT_COUNT,
    OUT_DRAFTS,
    OUT_MARGINS,
    OUT_TARGETS,
    SECOND_LOGIT_LANE,
    SEM_BCAST,
    SEM_GATHER,
    SEM_HOST,
    SEM_RS,
    SEM_TINY,
    SEMS_PER_SLOT,
    bf16,
    f32,
    pack,
    register_pytree_once,
    require_copy,
    select_rows,
    tile_rms_norm,
    unpack,
)
from .dspark_mla import (
    CONTEXT_TILES,
    HEADS_PER_RANK,
    QKV_TILES,
    ROWS,
    TILE,
    MlaDraftConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import numpy as np

    from ..common.protocols import AsyncCopyProtocol, PallasRefProtocol
    from .decode_megakernel import PhaseScopeProtocol

LM_CHUNK = 256
LM_CHUNKS = 20  # 20 * 256 = 5120 vocabulary rows per rank
Q_TILES = QKV_TILES - CONTEXT_TILES  # the q_a tiles; then kv_a's
_STREAMED_ROW_THRESHOLD = 32

VMEM_NAMES = (
    "ln1",
    "ln2",
    "q_norm",
    "kv_norm",
    "context_norm",
    "final_norm",
    "mask_embed",
    "rot",
)
HBM_NAMES = (
    "qkv_tile",
    "context_tile",
    "q_b",
    "w_uk",
    "w_uv",
    "o",
    "gate",
    "up",
    "down",
    "fc",
    "w2t",
    "w1",
    "rope",
    "lm_head",
    "embed",
)
WEIGHT_NAMES = (*VMEM_NAMES, *HBM_NAMES)
SMALL_NAMES = ("aux", "previous", *VMEM_NAMES)


class _Scalars:
    """Scalar ref viewed from ``offset``; ``rows`` > 1 maps ``i`` to ``i // rows``."""

    def __init__(
        self, ref: PallasRefProtocol, offset: int | jax.Array, rows: int = 1
    ) -> None:
        """Bind scalar-buffer view at offset with row stride."""
        self.ref, self.offset, self.rows = ref, offset, rows

    def __getitem__(self, index: int | jax.Array) -> jax.Array:
        """Read scalar entry for row or request index."""
        return self.ref[self.offset + (index // self.rows if self.rows > 1 else index)]


def _rope(
    x_bf16: jax.Array, rot_bf16: jax.Array, cos32: jax.Array, sin32: jax.Array
) -> jax.Array:
    """Interleaved-pair RoPE in FP32 (``x * cos2 + (x @ R) * sin2``), BF16 out."""
    rotated = dot(x_bf16, rot_bf16)  # exact: one +-1 term per lane
    return bf16(f32(x_bf16) * cos32 + rotated * sin32)


@dataclasses.dataclass(frozen=True, slots=True)
class _MlaStepSpec:
    """Static geometry and sharding parameters for one MLA fused draft step."""

    block: int
    num_requests: int
    rows: int
    pages: int
    owner_table: jax.Array | Sequence[int] | np.ndarray | None
    dp: int


@dataclasses.dataclass
class _MlaStepEnv:
    """Static and per-invocation state for one MLA DSpark fused kernel call."""

    c: MlaDraftConfig
    block: int
    num_requests: int
    rows: int
    pages: int
    owner_table: jax.Array | Sequence[int] | np.ndarray | None
    dp: int
    total_rows: int
    num_layers: int
    heads: int
    group_requests: int
    group_rows: int
    q_shard: int
    vocab: int
    hidden: int
    eps: float
    scale: float
    drafts_count: int
    streamed: bool
    streamed_o: bool
    num_keys: int
    # DMA semaphore indices (19 + 5 * R entries; index 10 unused).
    sem_fc: int
    sem_o: int
    sem_w2t: int
    sem_own: int
    sem_lm: int  # index 11 (index 10 unused)
    sem_embed: int  # base 12; per-request offset + req_idx
    sem_rope: int  # base 12 + R; per-request offset + 2 * req_idx (+1 for second half)
    sem_w1: int  # base 12 + 3 * R; offset + slot * R + req_idx
    sem_lat: int
    sem_down: int
    sem_tile: int  # base 14 + 5 * R; offset + slot
    sem_lm2: int
    sem_o2: int  # streamed o / down: the second chunk slot
    sem_down2: int  # streamed o / down: the second chunk slot
    scalars: PallasRefProtocol
    aux_hbm: PallasRefProtocol
    logits_hbm: PallasRefProtocol
    previous_hbm: PallasRefProtocol
    weights: dict[str, PallasRefProtocol]
    pools: tuple[PallasRefProtocol, ...]
    tokens_out: PallasRefProtocol
    latents_out: PallasRefProtocol
    small: dict[str, PallasRefProtocol]
    persistent: list[PallasRefProtocol]
    dcp_scoped: list[PallasRefProtocol]
    q_buf: PallasRefProtocol
    w_uk_buf: PallasRefProtocol
    w_uv_buf: PallasRefProtocol
    o_buf: PallasRefProtocol
    gate_buf: PallasRefProtocol
    up_buf: PallasRefProtocol
    down_buf: PallasRefProtocol
    w2t_buf: PallasRefProtocol
    w1_blk: PallasRefProtocol
    embed_blk: PallasRefProtocol
    bcast_buf: PallasRefProtocol
    rope_blk: PallasRefProtocol
    logits_vmem: PallasRefProtocol
    fc_part: PallasRefProtocol
    tokens_v: PallasRefProtocol
    stage: PallasRefProtocol
    local_recv: PallasRefProtocol
    host_packed: PallasRefProtocol
    gather_recv: PallasRefProtocol
    tiny_recv: PallasRefProtocol
    argmax_recv: PallasRefProtocol
    keys_vmem: PallasRefProtocol
    valid_vmem: PallasRefProtocol
    latents_vmem: PallasRefProtocol
    tile_buf: PallasRefProtocol
    tile_recv: PallasRefProtocol
    stage2_recv: PallasRefProtocol
    stage2_send: PallasRefProtocol
    stage2_sems: PallasRefProtocol
    dma_sems: PallasRefProtocol
    small_sems: PallasRefProtocol
    sends: PallasRefProtocol
    recvs: PallasRefProtocol
    rank: jax.Array
    lane: jax.Array
    host: jax.Array
    data_group: jax.Array
    # [R]: the verify's first position
    new_position: jax.Array
    forced_count: jax.Array
    forced_anchor: jax.Array
    # ---------------- Markov w1 rows, RoPE rows, anchor embeddings ----------------
    w1_copies: list[list[AsyncCopyProtocol | None]]
    pending: dict[str, AsyncCopyProtocol]

    @classmethod
    def from_refs(
        cls,
        c: MlaDraftConfig,
        spec: _MlaStepSpec,
        scalars: PallasRefProtocol,
        all_refs: tuple[PallasRefProtocol, ...],
    ) -> Self:
        """Construct the MLA DSpark step environment from Pallas operand refs."""
        block = spec.block
        num_requests = spec.num_requests
        rows = spec.rows
        pages = spec.pages
        owner_table = spec.owner_table
        dp = spec.dp
        total_rows = num_requests * rows
        num_layers = c.layers
        heads = HEADS_PER_RANK * dp  # the held TP32 shards' heads (attention DP)
        group_requests, group_rows = num_requests // dp, total_rows // dp
        # a shard's q_b columns: nopes, then ropes padded to a tile
        q_shard = HEADS_PER_RANK * (c.nope + TILE)
        # o / down stream one 896-column chunk at a time to bound scratch size.
        streamed = total_rows > _STREAMED_ROW_THRESHOLD
        streamed_o = streamed or dp > 1
        n_persist = len(mla_dcp.PERSISTENT_NAMES)
        n_scoped = len(mla_dcp.SCOPED_NAMES)
        aux_hbm, logits_hbm, previous_hbm = all_refs[:3]
        ref_list = list(all_refs[3:])
        weights = dict(zip(WEIGHT_NAMES, ref_list[: len(WEIGHT_NAMES)], strict=True))
        ref_list = ref_list[len(WEIGHT_NAMES) :]
        pools = tuple(ref_list[:num_layers])
        ref_list = ref_list[num_layers:]
        tokens_out, latents_out = ref_list[:2]
        scratch = ref_list[2:]
        small = dict(zip(SMALL_NAMES, scratch[: len(SMALL_NAMES)], strict=True))
        scratch = scratch[len(SMALL_NAMES) :]
        persistent = scratch[:n_persist]
        dcp_scoped = scratch[n_persist : n_persist + n_scoped]
        (
            q_buf,
            w_uk_buf,
            w_uv_buf,
            o_buf,
            gate_buf,
            up_buf,
            down_buf,
            w2t_buf,
            w1_blk,
            embed_blk,
            bcast_buf,
            rope_blk,
            logits_vmem,
            fc_part,
            tokens_v,
            stage,
            local_recv,
            host_packed,
            gather_recv,
            tiny_recv,
            argmax_recv,
            keys_vmem,
            valid_vmem,
            latents_vmem,
            tile_buf,
            tile_recv,
            stage2_recv,
            stage2_send,
            stage2_sems,
            dma_sems,
            small_sems,
            sends,
            recvs,
        ) = scratch[n_persist + n_scoped :]
        rank = jax.lax.axis_index(AXIS_TP)
        lane = rank % 8
        host = rank // 8
        data_group = rank // (TP_SIZE // dp)
        new_position = jnp.stack([scalars[j] for j in range(num_requests)])
        forced_count = jnp.stack(
            [scalars[num_requests + j] for j in range(num_requests)]
        )
        forced_anchor = jnp.stack(
            [scalars[2 * num_requests + j] for j in range(num_requests)]
        )
        w1_copies: list[list[AsyncCopyProtocol | None]] = [
            [None] * num_requests,
            [None] * num_requests,
        ]
        pending: dict[str, AsyncCopyProtocol] = {}
        return cls(
            c=c,
            block=block,
            num_requests=num_requests,
            rows=rows,
            pages=pages,
            owner_table=owner_table,
            dp=dp,
            total_rows=total_rows,
            num_layers=num_layers,
            heads=heads,
            group_requests=group_requests,
            group_rows=group_rows,
            q_shard=q_shard,
            vocab=c.vocab // TP_SIZE,
            hidden=c.hidden,
            eps=c.eps,
            scale=c.softmax_scale,
            drafts_count=rows - 1,  # verify rows after the anchor
            streamed=streamed,
            streamed_o=streamed_o,
            num_keys=2 * rows,
            # DMA semaphore indices (19 + 5 * R entries; index 10 unused).
            sem_fc=1,
            sem_o=5,
            sem_w2t=8,
            sem_own=9,
            sem_lm=11,  # index 11 (index 10 unused)
            sem_embed=12,  # per-request offset + req_idx
            sem_rope=12 + num_requests,  # + 2 * req_idx (+1 for second half)
            sem_w1=12 + 3 * num_requests,  # + slot * num_requests + req_idx
            sem_lat=12 + 5 * num_requests,
            sem_down=13 + 5 * num_requests,
            sem_tile=14 + 5 * num_requests,  # + slot
            sem_lm2=16 + 5 * num_requests,
            sem_o2=17 + 5 * num_requests,  # streamed o / down: the second chunk slot
            sem_down2=18 + 5 * num_requests,  # streamed o / down: the second chunk slot
            scalars=scalars,
            aux_hbm=aux_hbm,
            logits_hbm=logits_hbm,
            previous_hbm=previous_hbm,
            weights=weights,
            pools=pools,
            tokens_out=tokens_out,
            latents_out=latents_out,
            small=small,
            persistent=persistent,
            dcp_scoped=dcp_scoped,
            q_buf=q_buf,
            w_uk_buf=w_uk_buf,
            w_uv_buf=w_uv_buf,
            o_buf=o_buf,
            gate_buf=gate_buf,
            up_buf=up_buf,
            down_buf=down_buf,
            w2t_buf=w2t_buf,
            w1_blk=w1_blk,
            embed_blk=embed_blk,
            bcast_buf=bcast_buf,
            rope_blk=rope_blk,
            logits_vmem=logits_vmem,
            fc_part=fc_part,
            tokens_v=tokens_v,
            stage=stage,
            local_recv=local_recv,
            host_packed=host_packed,
            gather_recv=gather_recv,
            tiny_recv=tiny_recv,
            argmax_recv=argmax_recv,
            keys_vmem=keys_vmem,
            valid_vmem=valid_vmem,
            latents_vmem=latents_vmem,
            tile_buf=tile_buf,
            tile_recv=tile_recv,
            stage2_recv=stage2_recv,
            stage2_send=stage2_send,
            stage2_sems=stage2_sems,
            dma_sems=dma_sems,
            small_sems=small_sems,
            sends=sends,
            recvs=recvs,
            rank=rank,
            lane=lane,
            host=host,
            data_group=data_group,
            new_position=new_position,
            forced_count=forced_count,
            forced_anchor=forced_anchor,
            w1_copies=w1_copies,
            pending=pending,
        )

    def own_group(self, value: jax.Array) -> jax.Array:
        """Return this rank's DP group's share of ``value`` (group-major rows)."""
        if self.dp == 1:
            return value
        parts = jnp.split(value, self.dp, axis=0)
        selected = parts[0]
        for g in range(1, self.dp):
            selected = jnp.where(self.data_group == g, parts[g], selected)
        return selected

    def place_group(self, value: jax.Array) -> jax.Array:
        """Place the group's rows at their group-major slice, zeros elsewhere."""
        if self.dp == 1:
            return value
        zeros = jnp.zeros_like(value)
        return jnp.concatenate(
            [jnp.where(self.data_group == g, value, zeros) for g in range(self.dp)],
            axis=0,
        )

    # ---------------- DMA and collective helpers ----------------
    def copy(
        self, src: PallasRefProtocol, dst: PallasRefProtocol, sem: int
    ) -> AsyncCopyProtocol:
        """Create a local async DMA copy descriptor."""
        return tpu.make_async_copy(src, dst, self.dma_sems.at[sem])

    def remote(
        self,
        src: PallasRefProtocol,
        dst: PallasRefProtocol,
        sem_index: int | jax.Array,
        peer: int | jax.Array,
    ) -> AsyncCopyProtocol:
        """Create a remote ICI async DMA copy descriptor to peer."""
        return tpu.make_async_remote_copy(
            src,
            dst,
            self.sends.at[sem_index],
            self.recvs.at[sem_index],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def exchange(
        self,
        src_of: Callable[[int], PallasRefProtocol],
        dst_of: Callable[[int], PallasRefProtocol],
        offsets: Iterable[int],
        slot_and_base: tuple[int, int],
    ) -> None:
        """Run a synchronous XOR-peer remote DMA exchange across offsets."""
        slot, base = slot_and_base
        copies = [
            self.remote(
                src_of(o),
                dst_of(o),
                slot * SEMS_PER_SLOT + base + o,
                self.rank ^ o,
            )
            for o in offsets
        ]
        for item in copies:
            item.start()
        for item in copies:
            item.wait()

    def host_reduce_gather(
        self, chunk32: jax.Array, slot: int, tag: str = "fc"
    ) -> jax.Array:
        """Reduce ``[T, 896]`` FP32 partial across hosts and all-gather to 7168."""
        scope = phase_start(f"ar_{tag}_host")
        self.host_packed[slot, self.host] = pack(bf16(chunk32))
        self.exchange(
            lambda _offset: self.host_packed.at[slot, self.host],
            lambda _offset: self.host_packed.at[slot, self.host],
            (8, 16, 24),
            (slot, SEM_HOST),
        )
        reduced = cast(
            "jax.Array",
            sum(
                unpack(self.host_packed[slot, h], self.total_rows, CHUNK)
                for h in range(4)
            ),
        )
        phase_end(scope)
        scope = phase_start(f"ar_{tag}_gather")
        self.gather_recv[slot, self.lane] = pack(bf16(reduced))
        self.exchange(
            lambda _offset: self.gather_recv.at[slot, self.lane],
            lambda _offset: self.gather_recv.at[slot, self.lane],
            range(1, HOST_CHUNKS),
            (slot, SEM_GATHER),
        )
        result = jnp.concatenate(
            [
                unpack(self.gather_recv[slot, j], self.total_rows, CHUNK)
                for j in range(HOST_CHUNKS)
            ],
            axis=1,
        )
        phase_end(scope)
        return result

    def all_reduce(
        self,
        lhs_bf16: jax.Array,
        weight_ref: PallasRefProtocol,
        slot: int,
        refill: PallasRefProtocol | None = None,
    ) -> jax.Array:
        """Compute ``lhs @ W`` and reduce-scatter + all-gather across 32 ranks.

        ``lhs @ W`` (a chunk-permuted ``[K, 7168]`` weight: memory chunk k is
        logical chunk ``lane ^ k``) summed over the 32 ranks: the host-local
        reduce-scatter chunk by chunk, then the cross-host reduction and
        gather. ``refill``: the HBM weight, streamed a chunk at a time through
        ``weight_ref``'s two slots (``[2, K, 896]``, chunk 0 already in slot
        0), the next chunk in flight.
        """
        refill_sems = (
            (self.sem_o, self.sem_o2) if slot == 1 else (self.sem_down, self.sem_down2)
        )
        tag = "o" if slot == 1 else "down"
        scope = phase_start(f"ar_{tag}_local")
        copies: list[AsyncCopyProtocol] = []
        own: AsyncCopyProtocol | None = None
        loading: AsyncCopyProtocol | None = None
        for k in range(HOST_CHUNKS):
            if refill is not None:
                arrived = loading
                if k + 1 < HOST_CHUNKS:
                    loading = self.copy(
                        refill.at[:, pl.ds((k + 1) * CHUNK, CHUNK)],
                        weight_ref.at[(k + 1) % 2],
                        refill_sems[(k + 1) % 2],
                    )
                    loading.start()
                if arrived is not None:
                    arrived.wait()
                operand = weight_ref[k % 2]
            else:
                operand = weight_ref[:, pl.ds(k * CHUNK, CHUNK)]
            partial = dot(lhs_bf16, operand)  # [T, 896] FP32
            self.stage[slot, k] = pack(bf16(partial))
            if k == 0:
                own = self.copy(
                    self.stage.at[slot, 0],
                    self.local_recv.at[slot, self.lane],
                    self.sem_own,
                )
                own.start()
            else:
                item = self.remote(
                    self.stage.at[slot, k],
                    self.local_recv.at[slot, self.lane],
                    slot * SEMS_PER_SLOT + SEM_RS + k,
                    self.rank ^ k,
                )
                item.start()
                copies.append(item)
        phase_end(scope)
        scope = phase_start(f"ar_{tag}_rs")
        for item in copies:
            item.wait()
        require_copy(own).wait()
        chunk = cast(
            "jax.Array",
            sum(
                unpack(self.local_recv[slot, k], self.total_rows, CHUNK)
                for k in range(HOST_CHUNKS)
            ),
        )
        phase_end(scope)
        return self.host_reduce_gather(chunk, slot, tag)

    def gather_tiles(
        self, tile_bf16: jax.Array, slot: int, count: int
    ) -> list[jax.Array]:
        """Broadcast each rank's ``[T, 128]`` projection tile and unpack ``count``."""
        self.tile_recv[slot, self.rank] = pack(tile_bf16)
        # Only ranks below ``count`` hold a tile: they send, and each rank waits for
        # those peers only.
        copies = [
            self.remote(
                self.tile_recv.at[slot, self.rank],
                self.tile_recv.at[slot, self.rank],
                slot * SEMS_PER_SLOT + SEM_TINY + o,
                self.rank ^ o,
            )
            for o in range(1, 32)
        ]

        @pl.when(self.rank < count)
        def send_tile() -> None:
            """Start remote DMA sends from active tile-producing ranks."""
            for item in copies:
                item.start()

        for o, item in zip(range(1, 32), copies, strict=True):
            pl.when((self.rank ^ o) < count)(item.wait_recv)

        @pl.when(self.rank < count)
        def sent_tile() -> None:
            """Wait for remote DMA send completion on tile-producing ranks."""
            for item in copies:
                item.wait_send()

        return [
            unpack(self.tile_recv[slot, i], self.total_rows, TILE) for i in range(count)
        ]

    def tile_copy(self, source: PallasRefProtocol, slot: int) -> AsyncCopyProtocol:
        """Create async device-to-local DMA copy for a QKV projection tile."""
        return self.copy(source, self.tile_buf.at[slot], self.sem_tile + slot)

    def tiny_exchange(self, row_i32: jax.Array, slot: int) -> jax.Array:
        """Broadcast a compact int32 row across all 32 ranks."""
        self.tiny_recv[slot, self.rank] = row_i32
        self.exchange(
            lambda _offset: self.tiny_recv.at[slot, self.rank],
            lambda _offset: self.tiny_recv.at[slot, self.rank],
            range(1, 32),
            (slot, SEM_TINY),
        )
        return self.tiny_recv[slot]

    def start_w1(self, tokens: jax.Array, slot: int) -> None:
        """Start async device-to-local DMA reads for Markov w1 rows."""
        for j in range(self.num_requests):
            start = pl.multiple_of((tokens[j] // 16) * 16, 16)
            item = self.copy(
                self.weights["w1"].at[pl.ds(start, 16)],
                self.w1_blk.at[slot, j],
                self.sem_w1 + slot * self.num_requests + j,
            )
            item.start()
            self.w1_copies[slot][j] = item

    def w1_rows_for(self, tokens: jax.Array, slot: int) -> jax.Array:
        """Extract the exact Markov w1 row for each request from local buffer blocks."""
        return jnp.concatenate(
            [
                bf16(select_rows(f32(self.w1_blk[slot, j]), tokens[j] % 16, 1))
                for j in range(self.num_requests)
            ],
            axis=0,
        )

    def latent_rows(
        self, kv_bf16: jax.Array, layer: int, cos32: jax.Array, sin32: jax.Array
    ) -> jax.Array:
        """Build ``[T, 640]`` BF16 cache rows: kv_a_layernorm(c_kv) | RoPE(k_pe) | 0."""
        c_kv = tile_rms_norm(
            kv_bf16[:, : self.c.kv_lora],
            self.small["kv_norm"][layer][:, : self.c.kv_lora],
            self.eps,
        )
        k_pe = _rope(
            kv_bf16[:, self.c.kv_lora : self.c.latent],
            self.small["rot"][...],
            cos32,
            sin32,
        )
        return jnp.concatenate(
            (
                c_kv,
                k_pe,
                jnp.zeros(
                    (self.total_rows, MLA_LATENT_ROW_LANES - self.c.latent),
                    jnp.bfloat16,
                ),
            ),
            axis=1,
        )

    # Each layer buffer refills with the next layer's weights as soon as this layer is
    # done with it.
    def layer_copy(self, name: str, layer: int) -> AsyncCopyProtocol:
        """Start an async DMA prefetch for a per-layer weight buffer."""
        buffers = {
            "q_b": (self.weights["q_b"].at[layer], self.q_buf, 2),
            "w_uk": (self.weights["w_uk"].at[layer], self.w_uk_buf, 3),
            "w_uv": (self.weights["w_uv"].at[layer], self.w_uv_buf, 4),
            "o": (
                self.weights["o"].at[layer, :, pl.ds(0, self.o_buf.shape[-1])],
                self.o_buf.at[0] if self.streamed_o else self.o_buf,
                self.sem_o,
            ),
            "gate": (self.weights["gate"].at[layer], self.gate_buf, 6),
            "up": (self.weights["up"].at[layer], self.up_buf, 7),
            "down": (
                self.weights["down"].at[layer, :, pl.ds(0, self.down_buf.shape[-1])],
                self.down_buf.at[0] if self.streamed else self.down_buf,
                self.sem_down,
            ),
        }
        item = self.copy(*buffers[name])
        item.start()
        return item

    def refill(self, name: str, layer: int) -> None:
        """Prefetch the next layer's weight buffer if more layers remain."""
        if layer + 1 < self.num_layers:
            self.pending[name] = self.layer_copy(name, layer + 1)


def _mla_accept_and_issue_reads(
    env: _MlaStepEnv,
    small_copies: Sequence[AsyncCopyProtocol],
    logits_copy: AsyncCopyProtocol,
) -> tuple[jax.Array, jax.Array, jax.Array, list[AsyncCopyProtocol]]:
    """Run verify acceptance, populate tokens_v/valid_vmem, and start w1/rope/embed."""
    # ---------------- verify acceptance (as in dspark) ----------------
    scope = phase_start("d_accept")
    for item in small_copies:
        item.wait()
    logits_copy.wait()
    verify_logits = env.logits_vmem[...]  # [T, 5120] FP32
    vocab_iota = jax.lax.broadcasted_iota(jnp.int32, (env.total_rows, env.vocab), 1)
    local_best = jnp.max(verify_logits, axis=1, keepdims=True)
    local_id = jnp.min(
        jnp.where(verify_logits == local_best, vocab_iota, env.vocab),
        axis=1,
        keepdims=True,
    )
    local_second = jnp.max(
        jnp.where(vocab_iota == local_id, NEG, verify_logits),
        axis=1,
        keepdims=True,
    )
    out_lane = jax.lax.broadcasted_iota(jnp.int32, (env.total_rows, VREG_LANE_COUNT), 1)
    packet = jnp.where(
        out_lane == 0,
        jax.lax.bitcast_convert_type(local_best, jnp.int32),
        jnp.where(
            out_lane == 1,
            local_id + env.rank * env.vocab,
            jnp.where(
                out_lane == SECOND_LOGIT_LANE,
                jax.lax.bitcast_convert_type(local_second, jnp.int32),
                0,
            ),
        ),
    )
    # Rows along lanes: one [8, 128] tile a peer (lane t: row t; sublanes best, id,
    # second), not [T, 128].
    square = jnp.pad(packet, ((0, VREG_LANE_COUNT - env.total_rows), (0, 0)))
    env.argmax_recv[env.rank] = jnp.transpose(square)[:8]
    env.exchange(
        lambda _offset: env.argmax_recv.at[env.rank],
        lambda _offset: env.argmax_recv.at[env.rank],
        range(1, 32),
        (1, SEM_TINY),
    )
    gathered = env.argmax_recv[...]  # [TP_SIZE, 8, 128]
    rank_iota3 = jax.lax.broadcasted_iota(jnp.int32, (TP_SIZE, 1, VREG_LANE_COUNT), 0)
    bests = jax.lax.bitcast_convert_type(gathered[:, 0:1, :], jnp.float32)
    seconds = jax.lax.bitcast_convert_type(gathered[:, 2:3, :], jnp.float32)
    best = jnp.max(bests, axis=0)  # [1, 128]
    winner = jnp.min(jnp.where(bests == best[None], rank_iota3, TP_SIZE), axis=0)
    target_lanes = jnp.sum(
        jnp.where(rank_iota3 == winner[None], gathered[:, 1:2, :], 0), axis=0
    )
    runner_up = jnp.maximum(
        jnp.max(seconds, axis=0),
        jnp.max(jnp.where(rank_iota3 == winner[None], NEG, bests), axis=0),
    )
    lanes_back = jnp.concatenate(
        (
            target_lanes,
            jax.lax.bitcast_convert_type(best - runner_up, jnp.int32),
            jnp.zeros((VREG_LANE_COUNT - 2, VREG_LANE_COUNT), jnp.int32),
        ),
        axis=0,
    )
    # [T, 128]: lane 0 target, lane 1 margin
    rows_back = jnp.transpose(lanes_back)[: env.total_rows]
    targets = rows_back[:, 0:1].reshape(env.num_requests, env.rows, 1)
    margins = jax.lax.bitcast_convert_type(rows_back[:, 1:2], jnp.float32).reshape(
        env.num_requests, env.rows, 1
    )
    previous_vector = env.small["previous"][...]  # [R, 128]
    row_iota_col = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, 1), 1
    )
    # proposal k was verify row k + 1
    proposals = jnp.sum(
        jnp.where(
            jax.lax.broadcasted_iota(
                jnp.int32, (env.num_requests, env.rows, VREG_LANE_COUNT), 2
            )
            == row_iota_col + OUT_DRAFTS,
            jnp.broadcast_to(
                previous_vector[:, None, :],
                (env.num_requests, env.rows, VREG_LANE_COUNT),
            ),
            0,
        ),
        axis=2,
        keepdims=True,
    )
    match_rows = (row_iota_col < env.drafts_count) & (proposals == targets)
    first_miss = jnp.min(
        jnp.where(
            match_rows | (row_iota_col >= env.drafts_count), env.rows, row_iota_col
        ),
        axis=1,
    )[:, 0]
    count = jnp.where(
        env.forced_count >= 0,
        env.forced_count,
        jnp.minimum(first_miss, env.drafts_count),
    )
    anchor_token = jnp.sum(
        jnp.where(row_iota_col[:, :, 0] == count[:, None], targets[:, :, 0], 0),
        axis=1,
    )
    anchor_token = jnp.where(env.forced_anchor >= 0, env.forced_anchor, anchor_token)
    anchor_position = env.new_position + count + 1  # [R]
    result_lane = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, VREG_LANE_COUNT), 1
    )
    field_row = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, VREG_LANE_COUNT), 1
    )
    field_lane = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, VREG_LANE_COUNT), 2
    )
    targets_row = jnp.sum(
        jnp.where(field_lane == field_row + OUT_TARGETS, targets, 0), axis=1
    )
    margins_row = jnp.sum(
        jnp.where(
            field_lane == field_row + OUT_MARGINS,
            jax.lax.bitcast_convert_type(margins, jnp.int32),
            0,
        ),
        axis=1,
    )
    result = jnp.zeros((env.num_requests, VREG_LANE_COUNT), jnp.int32) - 1
    result = jnp.where(
        (result_lane >= OUT_TARGETS) & (result_lane < OUT_TARGETS + env.rows),
        targets_row,
        result,
    )
    result = jnp.where(
        (result_lane >= OUT_MARGINS) & (result_lane < OUT_MARGINS + env.rows),
        margins_row,
        result,
    )
    result = jnp.where(result_lane == OUT_COUNT, count[:, None], result)
    result = jnp.where(result_lane == OUT_BONUS, anchor_token[:, None], result)
    env.tokens_v[...] = result

    # Each request's new keys: context rows up to its accepted count, its block rows
    # (non-causal); the padding block row is no key. (Attention DP: the rank's group's
    # requests, whose rows it attends.)
    key_lane = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, 1, env.num_keys), 2
    )
    env.valid_vmem[...] = env.own_group(
        (
            ((key_lane < env.rows) & (key_lane <= count[:, None, None]))
            | ((key_lane >= env.rows) & (key_lane < env.rows + env.block))
        ).astype(jnp.int32)
    )

    return _mla_issue_initial_reads(env, scope, anchor_token, anchor_position)


def _mla_issue_initial_reads(
    env: _MlaStepEnv,
    scope: PhaseScopeProtocol,
    anchor_token: jax.Array,
    anchor_position: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, list[AsyncCopyProtocol]]:
    """Start w1, RoPE, and anchor embedding broadcast DMAs and close d_accept."""
    env.start_w1(anchor_token, 0)
    rope_copies: list[AsyncCopyProtocol] = []
    for j in range(env.num_requests):
        block_start = pl.multiple_of((anchor_position[j] // 8) * 8, 8)
        context_start = pl.multiple_of((env.new_position[j] // 8) * 8, 8)
        rope_copies += (
            env.copy(
                env.weights["rope"].at[pl.ds(block_start, 16)],
                env.rope_blk.at[0, j],
                env.sem_rope + 2 * j,
            ),
            env.copy(
                env.weights["rope"].at[pl.ds(context_start, 16)],
                env.rope_blk.at[1, j],
                env.sem_rope + 2 * j + 1,
            ),
        )
    for item in rope_copies:
        item.start()

    owner = anchor_token // env.vocab
    for j in range(env.num_requests):

        @pl.when(env.rank == owner[j])
        def send_anchor_row(j: int = j) -> None:
            """Broadcast request j's anchor token embedding from its owner rank."""
            local_id = anchor_token[j] - owner[j] * env.vocab
            item = env.copy(
                env.weights["embed"].at[
                    pl.ds(pl.multiple_of((local_id // 16) * 16, 16), 16)
                ],
                env.embed_blk,
                env.sem_embed + j,
            )
            item.start()
            item.wait()
            env.bcast_buf[j] = pack(
                bf16(select_rows(f32(env.embed_blk[...]), local_id % 16, 1))
            )
            copies = [
                env.remote(
                    env.bcast_buf.at[j],
                    env.bcast_buf.at[j],
                    SEM_BCAST + j,
                    env.rank ^ o,
                )
                for o in range(1, 32)
            ]
            for copy_item in copies:
                copy_item.start()
            for copy_item in copies:
                copy_item.wait_send()

    phase_end(scope)
    return anchor_token, anchor_position, owner, rope_copies


def _mla_host_fc_spans(env: _MlaStepEnv) -> list[list[tuple[int, int, int, int]]]:
    """Compute per-host auxiliary layer slice spans for FC context projection."""
    spans: list[list[tuple[int, int, int, int]]] = []
    for h in range(4):
        start, stop = h * FC_IN, (h + 1) * FC_IN
        pieces: list[tuple[int, int, int, int]] = []
        for layer_index in range(len(env.c.aux_layers)):
            lo, hi = layer_index * env.hidden, (layer_index + 1) * env.hidden
            a, b = max(start, lo), min(stop, hi)
            if a < b:
                pieces.append((layer_index, a - lo, b - lo, a - start))
        spans.append(pieces)
    return spans


def _mla_context_features(env: _MlaStepEnv) -> jax.Array:
    """Contract host-partitioned auxiliary features and apply context RMSNorm."""
    # ---------------- context features ----------------
    scope = phase_start("d_context")
    aux_ref = env.small["aux"]  # [aux layers, T, hidden]
    spans = _mla_host_fc_spans(env)

    # The fc weight streams in two halves to bound local scratch usage.
    fc_parts = 2
    part_rows = FC_IN // fc_parts

    def fc_phase(fc_buf: PallasRefProtocol) -> None:
        """Contract host-partitioned auxiliary features with streamed fc halves."""
        for p in range(fc_parts):
            item = env.copy(
                env.weights["fc"].at[pl.ds(p * part_rows, part_rows)],
                fc_buf,
                env.sem_fc,
            )
            item.start()
            item.wait()
            for h in range(4):

                @pl.when(env.host == h)
                def fc_branch(h: int = h, p: int = p) -> None:
                    """Accumulate host h's fc partial for half p."""
                    total: jax.Array | None = None if p == 0 else env.fc_part[...]
                    for layer_index, lo, hi, offset in spans[h]:
                        a, b = (
                            max(offset, p * part_rows),
                            min(offset + hi - lo, (p + 1) * part_rows),
                        )
                        if a >= b:
                            continue
                        part = dot(
                            aux_ref[layer_index, :, lo + a - offset : lo + b - offset],
                            fc_buf[pl.ds(a - p * part_rows, b - a), :],
                        )
                        total = part if total is None else total + part
                    if total is not None:
                        env.fc_part[...] = total

    pl.run_scoped(fc_phase, tpu.VMEM((part_rows, CHUNK), jnp.bfloat16))
    context = tile_rms_norm(
        bf16(env.host_reduce_gather(env.fc_part[...], 0)),
        env.small["context_norm"][...],
        env.eps,
    )
    phase_end(scope)
    return context


def _mla_prepare_inputs_and_cache(
    env: _MlaStepEnv,
    context: jax.Array,
    anchor_position: jax.Array,
    owner: jax.Array,
    handles: tuple[list[AsyncCopyProtocol], list[AsyncCopyProtocol]],
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    mla_dcp.DcpCache,
    mla_dcp.DcpWorkspace,
    int | jax.Array,
]:
    """Assemble RoPE tables, block input embeddings, context latents, and DCP cache."""
    rope_copies, tile_copies = handles
    # ---------------- RoPE rows and block input ----------------
    scope = phase_start("d_input")
    for item in rope_copies:
        item.wait()
    rope_block = jnp.concatenate(
        [
            select_rows(env.rope_blk[0, j], anchor_position[j] % 8, env.rows)
            for j in range(env.num_requests)
        ],
        axis=0,
    )
    rope_context = jnp.concatenate(
        [
            select_rows(env.rope_blk[1, j], env.new_position[j] % 8, env.rows)
            for j in range(env.num_requests)
        ],
        axis=0,
    )
    cos_block, sin_block = (
        rope_block[:, : env.c.rope],
        rope_block[:, env.c.rope :],
    )
    cos_context, sin_context = (
        rope_context[:, : env.c.rope],
        rope_context[:, env.c.rope :],
    )
    for j in range(env.num_requests):

        @pl.when(env.rank != owner[j])
        def receive_anchor_row(j: int = j) -> None:
            """Wait for request j's broadcast anchor row on non-owner ranks."""
            env.remote(
                env.bcast_buf.at[j], env.bcast_buf.at[j], SEM_BCAST + j, env.rank
            ).wait_recv()

    anchor_rows = jnp.concatenate(  # [R, 7168]
        [unpack(env.bcast_buf[j], 1, env.hidden) for j in range(env.num_requests)],
        axis=0,
    )
    mask_row = f32(env.small["mask_embed"][...])
    x_row = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, env.hidden), 1
    )
    x = bf16(
        jnp.where(x_row == 0, anchor_rows[:, None, :], mask_row[:, None, :])
    ).reshape(env.total_rows, env.hidden)

    # The context rows' kv_a for every layer: one tile per rank, one exchange.
    tile_copies[1].wait()
    context_tiles = env.gather_tiles(
        bf16(dot(context, env.tile_buf[1])), 0, env.num_layers * CONTEXT_TILES
    )
    if env.num_layers > 1:
        tile_copies[1] = env.tile_copy(env.weights["qkv_tile"].at[1], 1)
        tile_copies[1].start()
    for layer in range(env.num_layers):
        context_kv = bf16(
            jnp.concatenate(
                context_tiles[layer * CONTEXT_TILES : (layer + 1) * CONTEXT_TILES],
                axis=1,
            )
        )
        env.latents_vmem[layer] = env.latent_rows(
            context_kv, layer, cos_context, sin_context
        )

    # ---------------- DCP attention over the draft pools ----------------
    # Attention DP: the group's rows over its pages (DCP over the group).
    first_request = env.data_group * env.group_requests if env.dp > 1 else 0
    page_owner = None
    if env.owner_table is not None:
        page_owner = mla_dcp.owner_index(cast("Sequence[int]", env.owner_table))
        page_owner = page_owner % mla_dcp.GROUP if env.dp > 1 else page_owner
    cache = mla_dcp.DcpCache(
        pool_ref=env.pools,
        table_ref=_Scalars(env.scalars, 3 * env.num_requests),
        pages=env.pages,
        layer=jnp.int32(0),
        group=mla_dcp.GROUP,
        owner=page_owner,
        table_row=lambda row: first_request + row // env.rows,
    )
    cos_query, sin_query = env.own_group(cos_block), env.own_group(sin_block)
    workspace = mla_dcp.DcpWorkspace(*env.persistent, *env.dcp_scoped)
    env.pending = {
        name: env.layer_copy(name, 0)
        for name in ("q_b", "w_uk", "w_uv", "o", "gate", "up", "down")
    }
    phase_end(scope)
    return (
        x,
        cos_block,
        sin_block,
        cos_query,
        sin_query,
        cache,
        workspace,
        first_request,
    )


def _mla_layer_queries(
    env: _MlaStepEnv,
    layer: int,
    x: jax.Array,
    tile_copies: list[AsyncCopyProtocol],
    ropes: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> jax.Array:
    """Compute QKV latents, update keys_vmem, and return stacked MLA queries."""
    cos_block, sin_block, cos_query, sin_query = ropes
    scope = phase_start("d_qkv")
    h = tile_rms_norm(x, env.small["ln1"][layer], env.eps)
    slot = layer % 2
    tile_copies[slot].wait()
    tiles = env.gather_tiles(
        bf16(dot(h, env.tile_buf[slot])), (layer + 1) % 2, QKV_TILES
    )
    if layer + 2 < env.num_layers:
        tile_copies[slot] = env.tile_copy(env.weights["qkv_tile"].at[layer + 2], slot)
        tile_copies[slot].start()
    q_a = bf16(jnp.concatenate(tiles[:Q_TILES], axis=1))  # [T, 1536]
    block_kv = bf16(jnp.concatenate(tiles[Q_TILES:], axis=1))  # [T, 640]
    # [group rows, 1536]
    q_c = env.own_group(tile_rms_norm(q_a, env.small["q_norm"][layer], env.eps))
    block_latents = env.own_group(
        env.latent_rows(block_kv, layer, cos_block, sin_block)
    )
    context_latents = env.own_group(env.latents_vmem[layer])
    for j in range(env.group_requests):
        env.keys_vmem[j] = jnp.concatenate(
            (
                f32(context_latents[j * env.rows : (j + 1) * env.rows]),
                f32(block_latents[j * env.rows : (j + 1) * env.rows]),
            ),
            axis=0,
        )

    phase_end(scope)
    scope = phase_start("d_query")
    env.pending["q_b"].wait()
    # [group rows, dp Q_WIDTH]: each held shard's columns
    q = bf16(dot(q_c, env.q_buf[...]))
    env.refill("q_b", layer)
    env.pending["w_uk"].wait()
    queries = []
    for head in range(env.heads):
        shard, within = divmod(head, HEADS_PER_RANK)
        nope = shard * env.q_shard + within * env.c.nope
        # [group rows, 512]
        q_abs = bf16(dot(q[:, nope : nope + env.c.nope], env.w_uk_buf[head]))
        pe = (
            shard * env.q_shard + HEADS_PER_RANK * env.c.nope + within * VREG_LANE_COUNT
        )
        q_pe = _rope(
            q[:, pe : pe + env.c.rope], env.small["rot"][...], cos_query, sin_query
        )
        queries.append(
            jnp.concatenate(
                (
                    f32(q_abs) * env.scale,
                    f32(q_pe) * env.scale,
                    jnp.zeros(
                        (env.group_rows, MLA_LATENT_ROW_LANES - env.c.latent),
                        jnp.float32,
                    ),
                ),
                axis=1,
            )
        )
    queries_stacked = jnp.stack(queries, axis=1)  # [group rows, heads, 640]
    env.refill("w_uk", layer)
    phase_end(scope)
    return queries_stacked


def _mla_run_layer(
    env: _MlaStepEnv,
    layer: int,
    x: jax.Array,
    tile_copies: list[AsyncCopyProtocol],
    layer_ctx: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        mla_dcp.DcpCache,
        mla_dcp.DcpWorkspace,
        int | jax.Array,
    ],
) -> jax.Array:
    """Execute one MLA DSpark transformer layer (QKV, DCP attention, O, and MLP)."""
    (
        cos_block,
        sin_block,
        cos_query,
        sin_query,
        cache,
        workspace,
        first_request,
    ) = layer_ctx
    queries_stacked = _mla_layer_queries(
        env, layer, x, tile_copies, (cos_block, sin_block, cos_query, sin_query)
    )
    scope = phase_start("d_attend")
    attention = mla_dcp.DcpAttention(
        dataclasses.replace(cache, layer=jnp.int32(layer)),
        workspace,
        _Scalars(env.scalars, first_request, env.rows),
        env.group_rows,
        sequence_rows=env.rows,
        heads=env.heads,
        chunk_rows=env.rows,
    )
    # The block's padding row carries the partials' lse: the combine
    # sends 512 lanes, not 640.
    attention.lse_in_padding = env.block < env.rows
    attention.group_fold = True
    if env.block < env.rows and mla_dcp.GROUP == TP_SIZE:
        # Merge each host's partials before crossing hosts (chunked calls, one request a
        # chunk, even one request).
        attention.force_chunked = True
        attention.host_combine = (env.stage2_recv, env.stage2_send, env.stage2_sems)
    if PHASE_SCOPES:
        attention.trace = "a_"
    attention.prime()
    # [group rows, heads, 512]
    merged = attention.attend(
        queries_stacked,
        jnp.zeros((env.group_rows, MLA_LATENT_ROW_LANES), jnp.float32),
        layer % 2,
        keys=env.keys_vmem,
        key_valid=env.valid_vmem,
    )
    phase_end(scope)
    scope = phase_start("d_o")
    wuv_scope = phase_start("d_wuv")
    env.pending["w_uv"].wait()
    # [T, heads 128]: the other groups' rows are zero (their ranks add theirs).
    attended = env.place_group(
        jnp.concatenate(
            [
                bf16(dot(bf16(merged[:, head]), env.w_uv_buf[head]))
                for head in range(env.heads)
            ],
            axis=1,
        )
    )
    env.refill("w_uv", layer)
    env.pending["o"].wait()
    phase_end(wuv_scope)
    x = bf16(
        f32(x)
        + env.all_reduce(
            attended,
            env.o_buf,
            1,
            refill=env.weights["o"].at[layer] if env.streamed_o else None,
        )
    )
    env.refill("o", layer)

    phase_end(scope)
    scope = phase_start("d_mlp")
    h2 = tile_rms_norm(x, env.small["ln2"][layer], env.eps)
    env.pending["gate"].wait()
    env.pending["up"].wait()
    gate = f32(bf16(dot(h2, env.gate_buf[...])))
    up = bf16(dot(h2, env.up_buf[...]))
    env.refill("gate", layer)
    env.refill("up", layer)
    activation = bf16(f32(bf16(gate * jax.nn.sigmoid(gate))) * f32(up))
    phase_end(scope)
    scope = phase_start("d_down")
    env.pending["down"].wait()
    mlp = env.all_reduce(
        activation,
        env.down_buf,
        0,
        refill=env.weights["down"].at[layer] if env.streamed else None,
    )
    env.refill("down", layer)
    x = bf16(f32(x) + mlp)
    phase_end(scope)
    return x


def _mla_lm_and_markov_sampling(
    env: _MlaStepEnv,
    x: jax.Array,
    anchor_token: jax.Array,
    w2t_copy: AsyncCopyProtocol,
) -> None:
    """Run vocabulary-sharded LM head and exact left-to-right Markov sampling."""
    scope = phase_start("d_lm")
    latents_copy = env.copy(env.latents_vmem, env.latents_out, env.sem_lat)
    latents_copy.start()

    # ---------------- LM head (the target's, vocabulary sharded) ----------------
    hidden_rows = tile_rms_norm(x, env.small["final_norm"][...], env.eps)

    def lm_phase(chunk_a: PallasRefProtocol, chunk_b: PallasRefProtocol) -> None:
        """Stream vocabulary-sharded LM head chunks into local logits scratch."""
        buffers = (chunk_a, chunk_b)

        def chunk_copy(k: int) -> AsyncCopyProtocol:
            """Start async DMA copy for LM head chunk k."""
            item = env.copy(
                env.weights["lm_head"].at[:, pl.ds(k * LM_CHUNK, LM_CHUNK)],
                buffers[k % 2],
                (env.sem_lm, env.sem_lm2)[k % 2],
            )
            item.start()
            return item

        copies = [chunk_copy(0), chunk_copy(1)]
        for k in range(LM_CHUNKS):
            copies[k % 2].wait()
            logits = f32(bf16(dot(hidden_rows, buffers[k % 2][...])))
            if k + 2 < LM_CHUNKS:
                copies[k % 2] = chunk_copy(k + 2)
            env.logits_vmem[:, pl.ds(k * LM_CHUNK, LM_CHUNK)] = logits

    pl.run_scoped(
        lm_phase,
        tpu.VMEM((env.hidden, LM_CHUNK), jnp.bfloat16),
        tpu.VMEM((env.hidden, LM_CHUNK), jnp.bfloat16),
    )

    phase_end(scope)

    # ---------------- exact Markov sampling ----------------
    scope = phase_start("d_markov")
    w2t_copy.wait()
    vocab_lane = jax.lax.broadcasted_iota(jnp.int32, (1, env.vocab), 1)
    tokens = env.tokens_v[...]
    token_lane = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, VREG_LANE_COUNT), 1
    )
    rank_iota = jax.lax.broadcasted_iota(
        jnp.int32, (TP_SIZE, env.num_requests, VREG_LANE_COUNT), 0
    )
    previous = anchor_token
    for slot in range(env.block):
        for item in env.w1_copies[slot % 2]:
            require_copy(item).wait()
        bias = dot(env.w1_rows_for(previous, slot % 2), env.w2t_buf[...])  # [R, 5120]
        scores = (
            jnp.concatenate(
                [
                    env.logits_vmem[j * env.rows + slot : j * env.rows + slot + 1, :]
                    for j in range(env.num_requests)
                ],
                axis=0,
            )
            + bias
        )
        best_value = jnp.max(scores, axis=1, keepdims=True)
        best_id = jnp.min(
            jnp.where(scores == best_value, vocab_lane, env.vocab),
            axis=1,
            keepdims=True,
        )
        row = jnp.where(
            token_lane == 0,
            jax.lax.bitcast_convert_type(best_value, jnp.int32),
            jnp.where(token_lane == 1, best_id + env.rank * env.vocab, 0),
        )
        gathered = env.tiny_exchange(row, slot % 2)
        maxima = jax.lax.bitcast_convert_type(gathered[:, :, 0:1], jnp.float32)
        top = jnp.max(maxima, axis=0, keepdims=True)
        win = jnp.min(
            jnp.where(maxima == top, rank_iota[:, :, 0:1], TP_SIZE),
            axis=0,
            keepdims=True,
        )
        picked_token = jnp.sum(jnp.where(rank_iota == win, gathered, 0), axis=0)[:, 1:2]
        tokens = jnp.where(token_lane == OUT_DRAFTS + slot, picked_token, tokens)
        previous = picked_token[:, 0]
        if slot + 1 < env.block:
            env.start_w1(previous, (slot + 1) % 2)
    env.tokens_v[...] = tokens
    out = tpu.make_async_copy(env.tokens_v, env.tokens_out, env.small_sems.at[0])
    out.start()
    out.wait()
    latents_copy.wait()
    phase_end(scope)


def _make_body(
    c: MlaDraftConfig,
    spec: _MlaStepSpec,
) -> Callable[..., None]:
    """Build the Pallas kernel body closure for MLA DSpark fused drafting."""
    num_layers = c.layers

    def body(
        scalars: PallasRefProtocol,
        *all_refs: PallasRefProtocol,
    ) -> None:
        """Execute one fused MLA DSpark draft step across 32 ranks."""
        env = _MlaStepEnv.from_refs(c, spec, scalars, all_refs)

        # ---------------- prologue ----------------
        collectives32.barrier()
        sources = {
            "aux": env.aux_hbm,
            "previous": env.previous_hbm,
            **{n: env.weights[n] for n in VMEM_NAMES},
        }
        small_copies = [
            tpu.make_async_copy(sources[n], env.small[n], env.small_sems.at[i])
            for i, n in enumerate(SMALL_NAMES)
        ]
        for item in small_copies:
            item.start()
        logits_copy = env.copy(env.logits_hbm, env.logits_vmem, 0)
        logits_copy.start()
        w2t_copy = env.copy(env.weights["w2t"], env.w2t_buf, env.sem_w2t)
        w2t_copy.start()
        tile_copies = [
            env.tile_copy(env.weights["qkv_tile"].at[0], 0),
            env.tile_copy(env.weights["context_tile"], 1),
        ]
        for item in tile_copies:
            item.start()

        anchor_token, anchor_position, owner, rope_copies = _mla_accept_and_issue_reads(
            env, small_copies, logits_copy
        )
        context = _mla_context_features(env)  # [T, 7168]
        (
            x,
            cos_block,
            sin_block,
            cos_query,
            sin_query,
            cache,
            workspace,
            first_request,
        ) = _mla_prepare_inputs_and_cache(
            env, context, anchor_position, owner, (rope_copies, tile_copies)
        )
        layer_ctx = (
            cos_block,
            sin_block,
            cos_query,
            sin_query,
            cache,
            workspace,
            first_request,
        )
        for layer in range(num_layers):
            x = _mla_run_layer(env, layer, x, tile_copies, layer_ctx)
        _mla_lm_and_markov_sampling(env, x, anchor_token, w2t_copy)

    return body


@dataclasses.dataclass(frozen=True, slots=True)
class MlaDraftLocalConfig:
    """Per-call geometry and sharding configuration for ``fused_draft_local``."""

    pages: int
    requests: int = 1
    block: int | None = None
    rows: int = ROWS
    owner_table: jax.Array | Sequence[int] | np.ndarray | None = None
    attention_dp: int = 1

    def __post_init__(self) -> None:
        """Register this dataclass as a JAX PyTree node on first construction."""
        register_pytree_once(
            type(self),
            data_fields=("owner_table",),
            meta_fields=("pages", "requests", "block", "rows", "attention_dp"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class MlaDraftStepInputs:
    """Per-step local shard inputs for the compiled MLA DSpark draft callable."""

    pools: Sequence[jax.Array]
    aux: jax.Array
    logits: jax.Array
    previous: jax.Array
    position: jax.Array
    forced_count: jax.Array
    forced_anchor: jax.Array
    tables: jax.Array

    def __post_init__(self) -> None:
        """Register this dataclass as a JAX PyTree node on first construction."""
        register_pytree_once(type(self))


def fused_draft_local(
    c: MlaDraftConfig,
    cfg: MlaDraftLocalConfig,
) -> Callable[[dict[str, jax.Array], MlaDraftStepInputs], tuple[jax.Array, jax.Array]]:
    """Return the per-shard fused MLA DSpark draft callable for a TP32 ``shard_map``.

    The per-shard fused draft (call inside a TP32 ``shard_map``):
    ``(weights, step)`` -> ``(vectors [R, 128] int32, latents [layers, T, 640]
    bf16)``, where ``step`` is an :class:`MlaDraftStepInputs`.

    Under ``cfg.attention_dp`` (DCP over each group, ``mla_dcp.GROUP``) the
    first ``R / dp`` requests are group 0's, and so on: a rank attends its
    group's requests with its held TP32 shards' heads (``q_b``, ``w_uk``,
    ``w_uv`` and ``o`` hold ``dp`` shards, ``dspark_mla.held_shards``).

    ``weights``: ``WEIGHT_NAMES`` (this rank's arrays, including the target's
    ``lm_head`` ``[7168, 5120]`` and ``embed`` ``[5120, 7168]``); ``step.pools``:
    the draft layers' local pools ``[blocks, 128, packing, 640]``; ``step.aux``
    ``[aux layers, T, hidden]`` and ``step.logits`` ``[R, rows, vocab / 32]``
    the verify's (``T = rows * R``; four or eight rows per request);
    ``step.previous`` ``[R, 128]`` the last vectors; ``step.position`` ``[R]``
    each verify's first position; ``step.forced_*`` ``[R]`` (``>= 0``: given);
    ``step.tables`` ``[R, pages]``.
    """
    pages = cfg.pages
    num_requests = cfg.requests
    rows = cfg.rows
    owner_table = cfg.owner_table
    dp = cfg.attention_dp
    block = c.block if cfg.block is None else int(cfg.block)
    if rows not in (4, ROWS):
        msg = "the fused MLA draft takes 4 or 8 rows per request"
        raise ValueError(msg)
    if not MIN_BLOCK_LEN <= block <= rows:
        msg = f"block must be 2..{rows}"
        raise ValueError(msg)
    total_rows = num_requests * rows
    if not 1 <= num_requests <= MAX_REQUESTS or total_rows > DSPARK_MLA_MAX_ROWS:
        msg = "the fused MLA draft takes up to 8 requests at 8 rows, or 16 at 4 rows"
        raise ValueError(msg)
    if num_requests % dp or mla_dcp.GROUP * dp != TP_SIZE:
        msg = (
            f"attention DP{dp}: {num_requests} requests over DCP{mla_dcp.GROUP} groups"
        )
        raise ValueError(msg)
    heads, group_rows = HEADS_PER_RANK * dp, total_rows // dp
    packed = total_rows * CHUNK // 256
    streamed = total_rows > _STREAMED_ROW_THRESHOLD

    def call(
        weights: dict[str, jax.Array],
        step: MlaDraftStepInputs,
    ) -> tuple[jax.Array, jax.Array]:
        """Invoke the Pallas dspark_mla_fused_draft kernel on local shard operands."""
        pools_tuple = tuple(step.pools)
        logits_flat = step.logits.reshape(total_rows, c.vocab // TP_SIZE).astype(
            jnp.float32
        )
        cache_dtype = pools_tuple[0].dtype
        page_tokens = pools_tuple[0].shape[1] * pools_tuple[0].shape[2]
        # the two-stage combine (see the body)
        chunks = block < rows and mla_dcp.GROUP == TP_SIZE
        plan = mla_dcp.page_plan(
            group_rows, heads, rows, cache_dtype, force_chunks=chunks
        )
        scalars = jnp.concatenate(
            (
                step.position,
                step.forced_count,
                step.forced_anchor,
                step.tables.reshape(-1),
            )
        ).astype(jnp.int32)
        small_shapes: dict[str, jax.Array] = {
            "aux": step.aux,
            "previous": step.previous,
            **weights,
        }
        # FP8 caches: FP8 query packets (the operand the receiver would quantize to
        # anyway), half the gather's bytes.
        persistent = mla_dcp.persistent_scratch(
            group_rows,
            mla_dcp.GROUP,
            heads,
            cache_dtype=cache_dtype,
            sequence_rows=rows,
            fp8_wire=True,
            force_chunks=chunks,
            chunk_rows=rows,
        )
        ring = (
            persistent[0].shape[0] if mla_dcp.chunked(group_rows, force=chunks) else 1
        )
        hosts = TP_SIZE // mla_dcp.HOST_LANES
        scratch = (
            *(
                tpu.VMEM(small_shapes[n].shape, small_shapes[n].dtype)
                for n in SMALL_NAMES
            ),
            *persistent,
            *mla_dcp.scoped_scratch(
                group_rows,
                plan["span"],
                plan["buffers"],
                heads=heads,
                group=mla_dcp.GROUP,
                cache_dtype=cache_dtype,
                page_tokens=page_tokens,
                sequence_rows=rows,
                force_chunks=chunks,
                chunk_rows=rows,
            ),
            # q_buf
            tpu.VMEM((c.q_lora, dp * HEADS_PER_RANK * (c.nope + TILE)), jnp.bfloat16),
            tpu.VMEM((heads, c.nope, c.kv_lora), jnp.bfloat16),  # w_uk_buf
            tpu.VMEM((heads, c.kv_lora, c.v_head), jnp.bfloat16),  # w_uv_buf
            tpu.VMEM(
                (2, heads * c.v_head, CHUNK)
                if streamed or dp > 1
                else (heads * c.v_head, c.hidden),
                jnp.bfloat16,
            ),
            tpu.VMEM((c.hidden, MLP_PADDED), jnp.bfloat16),  # gate_buf
            tpu.VMEM((c.hidden, MLP_PADDED), jnp.bfloat16),  # up_buf
            tpu.VMEM(
                (2, MLP_PADDED, CHUNK) if streamed else (MLP_PADDED, c.hidden),
                jnp.bfloat16,
            ),
            tpu.VMEM((c.markov_rank, c.vocab // TP_SIZE), jnp.bfloat16),  # w2t_buf
            tpu.VMEM((2, num_requests, 16, c.markov_rank), jnp.bfloat16),
            tpu.VMEM((16, c.hidden), jnp.bfloat16),  # embed_blk
            tpu.VMEM((num_requests, ROW_PACKED, VREG_LANE_COUNT), jnp.float32),
            tpu.VMEM((2, num_requests, 16, 2 * c.rope), jnp.float32),
            tpu.VMEM((total_rows, c.vocab // TP_SIZE), jnp.float32),
            tpu.VMEM((total_rows, CHUNK), jnp.float32),
            tpu.VMEM((num_requests, VREG_LANE_COUNT), jnp.int32),
            tpu.VMEM((2, 8, packed, VREG_LANE_COUNT), jnp.float32),  # stage
            tpu.VMEM((2, 8, packed, VREG_LANE_COUNT), jnp.float32),  # local_recv
            tpu.VMEM((2, 4, packed, VREG_LANE_COUNT), jnp.float32),  # host_packed
            tpu.VMEM((2, 8, packed, VREG_LANE_COUNT), jnp.float32),  # gather_recv
            tpu.VMEM((2, TP_SIZE, num_requests, VREG_LANE_COUNT), jnp.int32),
            # argmax_recv: rows along lanes
            tpu.VMEM((TP_SIZE, 8, VREG_LANE_COUNT), jnp.int32),
            tpu.VMEM((num_requests // dp, 2 * rows, MLA_LATENT_ROW_LANES), jnp.float32),
            tpu.VMEM((num_requests // dp, 1, 2 * rows), jnp.int32),
            tpu.VMEM((c.layers, total_rows, MLA_LATENT_ROW_LANES), jnp.bfloat16),
            # tile_buf: q_a | kv_a projection tiles
            tpu.VMEM((2, c.hidden, TILE), jnp.bfloat16),
            tpu.VMEM(
                (2, TP_SIZE, total_rows * TILE // 256, VREG_LANE_COUNT), jnp.float32
            ),
            # The two-stage DCP combine (chunked calls; see mla_dcp.host_combine).
            # stage2_recv
            tpu.VMEM((ring, hosts, HEADS_PER_RANK * rows, mla_dcp.LANES), jnp.bfloat16),
            # stage2_send
            tpu.VMEM((hosts, HEADS_PER_RANK * rows, mla_dcp.LANES), jnp.bfloat16),
            tpu.SemaphoreType.DMA((2, ring)),  # stage2_sems
            # DMA semaphores (19 + 5 * R): fixed slots 0..11 (slot 10 unused),
            # plus embed, rope, w1, lat, down, tile, lm2, and second-chunk slots.
            tpu.SemaphoreType.DMA((19 + 5 * num_requests,)),
            tpu.SemaphoreType.DMA((len(SMALL_NAMES),)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
        )
        body = _make_body(
            c,
            _MlaStepSpec(
                block=block,
                num_requests=num_requests,
                rows=rows,
                pages=pages,
                owner_table=owner_table,
                dp=dp,
            ),
        )
        operands = (
            step.aux,
            logits_flat,
            step.previous,
            *[weights[n] for n in WEIGHT_NAMES],
            *pools_tuple,
        )
        in_specs = [pl.BlockSpec(memory_space=tpu.HBM) for _ in operands]
        return pl.pallas_call(
            body,
            name="dspark_mla_fused_draft",
            out_shape=(
                jax.ShapeDtypeStruct((num_requests, VREG_LANE_COUNT), jnp.int32),
                jax.ShapeDtypeStruct(
                    (c.layers, total_rows, MLA_LATENT_ROW_LANES), jnp.bfloat16
                ),
            ),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=in_specs,
                out_specs=[
                    pl.BlockSpec(memory_space=tpu.HBM),
                    pl.BlockSpec(memory_space=tpu.HBM),
                ],
                scratch_shapes=scratch,
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=26, vmem_limit_bytes=64 * 1024**2
            ),
        )(scalars, *operands)

    return call
