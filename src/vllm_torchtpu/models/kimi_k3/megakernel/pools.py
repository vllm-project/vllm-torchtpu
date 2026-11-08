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
# src/vllm_torchtpu/megakernel/kimi_k3/pools.py
"""Unified KV/KDA pools as the K3 programs see them.

Each ``KVCacheTensor`` is one raw pool of ``layout.cache_layout()`` blocks,
shared by one MLA layer and three KDA layers: MLA latent pages and, in a KDA
group's state block, the layer's state (rows 0-76 recurrent, 77-80 conv taps)
and the draft-ring rows (81-95), heads in vLLM TP-rank order.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
import re
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.logger import init_logger

from .layout import (
    CONV_HEAD_ROWS,
    CONV_TAPS,
    HEAD_DIM,
    KDA_HEADS,
    LAYOUT,
    cache_layout,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

logger = init_logger(__name__)

MESH_AXIS = "tp"
MLA_LAYERS = len(LAYOUT.attention_layers)
KDA_LAYERS = len(LAYOUT.state_layers)
SSM_ROWS = LAYOUT.region("recurrent").rows
CONV_ROWS = LAYOUT.region("conv").rows
DRAFT_ROW0 = LAYOUT.region("draft").first_row
DRAFT_ROWS = LAYOUT.region("draft").rows
CONV_CHANNEL_GROUPS = 3
BATCHED_CONV_BANK_NDIM = 7
DRAFT_LAYER0 = MLA_LAYERS + KDA_LAYERS
DEFAULT_VERIFY_BANK_SLOTS = 64
DP_BATCH_SIZES = (2, 4, 8, 16, 32, 64, 96, 128)
TOKEN_SCATTER_BATCHES = (8, 12, 16, 20, 24, 28, 32, 64, 96, 128)
ROUTED_RS_BATCHES = (6, *TOKEN_SCATTER_BATCHES)
MAX_SINGLE_OWNER_BATCH = 4
DEFAULT_ROW_OWNERS = 4
DP2_GROUPS = 2
DP4_MIN_BATCH = 32

TP_SIZE = 32
MAX_WHOLE_CACHE_TILE_BATCH = 8
LARGE_CACHE_TILE_MIN_BATCH = 32
LARGE_CACHE_TILE = 2
MID_CACHE_TILE = 4
MIN_ALIAS_EXPERT_SCALES_BATCH = 6

_MLA_INDEX = {layer: m for m, layer in enumerate(LAYOUT.attention_layers)}
_KDA_INDEX = {layer: k for k, layer in enumerate(LAYOUT.state_layers)}
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index(name: str) -> int:
    """Extract the integer model layer index from a module parameter name."""
    match = _LAYER_RE.search(name)
    if match is None:
        msg = f"cannot find a layer index in module name {name!r}"
        raise ValueError(msg)
    return int(match.group(1))


def mla_state_index(layer: int) -> int:
    """Return position of model layer ``layer`` on the kernel's MLA axis."""
    return _MLA_INDEX[layer]


def kda_state_index(layer: int) -> int:
    """Return position of model layer ``layer`` on the kernel's KDA axis."""
    return _KDA_INDEX[layer]


def group_slots(num_slots: int, groups: int) -> int:
    """Return a rank's KDA slots per held head shard under attention DP."""
    return -(-(num_slots - 1) // groups) + 1


class SlotMap:
    """Host-side request-id to private KDA slot allocator (slot 0 is null)."""

    def __init__(self, num_slots: int, groups: int = 1) -> None:
        """Initialize per-group free slot stacks across ``num_slots``."""
        share = group_slots(num_slots, groups) - 1
        self._free = [
            list(range(min((g + 1) * share, num_slots - 1), g * share, -1))
            for g in range(groups)
        ]
        self._share = share
        self._slots: dict[str, int] = {}

    def __contains__(self, req_id: str) -> bool:
        """Return True when ``req_id`` currently holds an allocated slot."""
        return req_id in self._slots

    def group(self, req_id: str) -> int | None:
        """Return the attention-DP group owning ``req_id``, or None."""
        slot = self._slots.get(req_id)
        return None if slot is None else (slot - 1) // self._share

    def allocate(self, req_id: str, group: int = 0) -> int:
        """Return the existing or newly allocated slot for ``req_id``."""
        if req_id in self._slots:
            return self._slots[req_id]
        if not self._free[group]:
            msg = (
                "K3 decode: no free KDA slot (more live requests than --max-num-seqs?)"
            )
            raise RuntimeError(msg)
        slot = self._free[group].pop()
        self._slots[req_id] = slot
        return slot

    def release(self, req_ids: Iterable[str]) -> None:
        """Return the slots of ``req_ids`` to their groups' free lists."""
        for req_id in req_ids:
            slot = self._slots.pop(req_id, None)
            if slot is not None:
                self._free[(slot - 1) // self._share].append(slot)


def _raw_tensor(
    kv_cache: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
    """Unwrap a single-element sequence pool or return the raw pool tensor."""
    if isinstance(kv_cache, torch.Tensor):
        return kv_cache
    if len(kv_cache) != 1:
        msg = "expected a single raw pool per layer"
        raise ValueError(msg)
    return kv_cache[0]


def is_mla_layer(layer: int) -> bool:
    """Return True if ``layer`` is a full-attention MLA layer in K3."""
    return layer in _MLA_INDEX


def draft_raw_pools(
    caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
) -> list[torch.Tensor]:
    """Return the MLA DSpark draft layers' raw pool tensors in layer order."""
    layers = sorted(
        (layer_index(name), cache)
        for name, cache in caches.items()
        if layer_index(name) >= DRAFT_LAYER0
    )
    return [_raw_tensor(cache) for _, cache in layers]


def bind_raw_pools(
    caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
    kv_cache_dtype: str | None = None,
) -> tuple[list[torch.Tensor], list[int], list[int], list[str], list[str]]:
    """Collect distinct unified-pool tensors and per-layer pool indices."""
    raw_pools: list[torch.Tensor] = []
    index_of_ptr: dict[int, int] = {}

    def index_of(tensor: torch.Tensor) -> int:
        """Return or assign the deduplicated pool index for ``tensor``."""
        key = tensor.data_ptr()
        if key not in index_of_ptr:
            index_of_ptr[key] = len(raw_pools)
            raw_pools.append(tensor)
        return index_of_ptr[key]

    layers = sorted(
        (layer_index(name), name, cache)
        for name, cache in caches.items()
        if layer_index(name) < DRAFT_LAYER0
    )
    mla = [(layer, name, cache) for layer, name, cache in layers if is_mla_layer(layer)]
    kda = [
        (layer, name, cache) for layer, name, cache in layers if not is_mla_layer(layer)
    ]
    valid_counts = (len(mla) == MLA_LAYERS and len(kda) == KDA_LAYERS) or (
        1 <= len(mla) <= MLA_LAYERS and len(kda) == len(mla) * CONV_CHANNEL_GROUPS
    )
    if not valid_counts:
        msg = (
            f"expected {MLA_LAYERS} MLA and {KDA_LAYERS} KDA layers, "
            f"found {len(mla)} and {len(kda)}"
        )
        raise RuntimeError(msg)
    mla_raw, mla_names = [0] * len(mla), [""] * len(mla)
    for layer, name, cache in mla:
        mla_raw[mla_state_index(layer)] = index_of(_raw_tensor(cache))
        mla_names[mla_state_index(layer)] = name
    kda_raw, kda_names = [0] * len(kda), [""] * len(kda)
    for layer, name, cache in kda:
        kda_raw[kda_state_index(layer)] = index_of(_raw_tensor(cache))
        kda_names[kda_state_index(layer)] = name
    if len({tuple(t.shape) for t in raw_pools}) != 1:
        shapes = [tuple(t.shape) for t in raw_pools]
        msg = f"the unified pools differ in shape: {shapes}"
        raise NotImplementedError(msg)
    layout = cache_layout(kv_cache_dtype or str(raw_pools[0].dtype))
    if len(mla) < MLA_LAYERS:
        layout = dataclasses.replace(
            layout,
            attention_layers=layout.attention_layers[: len(mla)],
            state_layers=layout.state_layers[: len(kda)],
        )
    layout.check(raw_pools)
    return raw_pools, mla_raw, kda_raw, mla_names, kda_names


def tp_rank_permutation(
    tp_rank: int,
    position: int,
) -> list[tuple[int, int]] | None:
    """Return ``(source, destination)`` mesh positions mapping TP rank to mesh."""
    gathered: list[tuple[int, int] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, (int(tp_rank), int(position)))
    position_of_tp = {
        int(pair[0]): int(pair[1]) for pair in gathered if pair is not None
    }
    if (
        sorted(position_of_tp) != list(range(TP_SIZE))
        or len(set(position_of_tp.values())) != TP_SIZE
    ):
        msg = f"K3 kernels: cannot map TP ranks onto the kernel mesh: {position_of_tp}"
        raise RuntimeError(msg)
    perm = [(position_of_tp[q], q) for q in range(TP_SIZE)]
    return None if all(src == dst for src, dst in perm) else perm


def kda_blocks_at_length(
    block_tables: Sequence[np.ndarray | Sequence[int]],
    kda_groups: Sequence[tuple[int, int]],
    seq_len: int,
) -> np.ndarray:
    """Return per-KDA-layer state block IDs for a sequence of ``seq_len``."""
    if seq_len <= 0:
        return np.zeros(len(kda_groups), np.int32)
    return np.array(
        [
            block_tables[g][
                min(max((seq_len - 1) // size, 0), len(block_tables[g]) - 1)
            ]
            for g, size in kda_groups
        ],
        np.int32,
    )


def kda_blocks_at_column(
    block_tables: Sequence[np.ndarray | Sequence[int]],
    kda_groups: Sequence[tuple[int, int]],
    column: int,
) -> np.ndarray:
    """Return per-KDA-layer block IDs at ``column`` of each group's table."""
    return np.array([block_tables[g][column] for g, _ in kda_groups], np.int32)


def write_draft_rings(
    raws: list[jax.Array],
    kda_raw: Sequence[int],
    packed: jax.Array,
    blocks: jax.Array,
) -> list[jax.Array]:
    """Scatter packed draft rings ``[n, 69, DRAFT_ROWS, lanes]`` into ``raws``."""
    n = packed.shape[0]
    for k in range(KDA_LAYERS):
        index = kda_raw[k]
        shape = jax.eval_shape(
            lambda raw=raws[index], ids=blocks[:, k]: pool_adapters.gather_region(
                raw,
                ids,
                tok0=DRAFT_ROW0,
                ntok=DRAFT_ROWS,
                out_dtype=jnp.bfloat16,
            )
        ).shape
        raws[index] = pool_adapters.scatter_region(
            raws[index],
            packed[:, k].reshape(n, *shape[1:]),
            blocks[:, k],
            tok0=DRAFT_ROW0,
            ntok=DRAFT_ROWS,
        )
    return raws


def read_draft_rings(
    raws: Sequence[jax.Array],
    kda_raw: Sequence[int],
    blocks: jax.Array,
    lanes: int,
) -> jax.Array:
    """Gather draft rows of each sequence's 69 KDA state blocks as bf16."""
    n = blocks.shape[0]
    return jnp.stack(
        [
            pool_adapters.gather_region(
                raws[kda_raw[k]],
                blocks[:, k],
                tok0=DRAFT_ROW0,
                ntok=DRAFT_ROWS,
                out_dtype=jnp.bfloat16,
            ).reshape(n, DRAFT_ROWS, lanes)
            for k in range(KDA_LAYERS)
        ],
        axis=1,
    )


def unpack_kda_state(
    raw_pool: torch.Tensor,
    block_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack one KDA layer's ``(conv, recurrent)`` state from ``raw_pool``."""
    conv_shape = (CONV_TAPS, CONV_CHANNEL_GROUPS, CONV_HEAD_ROWS, HEAD_DIM)
    rec_shape = (KDA_HEADS, HEAD_DIM, HEAD_DIM)
    dev = raw_pool.device
    if block_id == 0:
        return (
            torch.zeros(conv_shape, dtype=torch.bfloat16, device=dev),
            torch.zeros(rec_shape, dtype=torch.float32, device=dev),
        )
    raw_bytes = raw_pool[block_id].contiguous().view(torch.uint8)
    ssm_rows = (
        raw_bytes[:SSM_ROWS].contiguous().view(torch.float32).reshape(-1, HEAD_DIM)
    )
    recurrent = ssm_rows[: KDA_HEADS * HEAD_DIM].reshape(rec_shape).clone()
    conv_flat = (
        raw_bytes[SSM_ROWS : SSM_ROWS + CONV_ROWS]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1)
    )
    active_elems = CONV_TAPS * CONV_CHANNEL_GROUPS * KDA_HEADS * HEAD_DIM
    conv_active = conv_flat[:active_elems].reshape(
        CONV_TAPS,
        CONV_CHANNEL_GROUPS,
        KDA_HEADS,
        HEAD_DIM,
    )
    conv = torch.zeros(conv_shape, dtype=torch.bfloat16, device=dev)
    conv[:, :, :KDA_HEADS, :] = conv_active
    return conv, recurrent


def read_kda_states(
    raw_pools: Sequence[torch.Tensor],
    kda_raw: Sequence[int],
    kda_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack all KDA ``(conv, recurrent)`` states from unified pools."""
    num_kda = len(kda_raw) or KDA_LAYERS
    blocks_list = [
        int(x) for x in kda_blocks.detach().to(device="cpu").reshape(-1).tolist()
    ]
    if len(blocks_list) != num_kda:
        msg = f"expected {num_kda} KDA block ids, got {len(blocks_list)}"
        raise ValueError(msg)
    conv_rows: list[torch.Tensor] = []
    rec_rows: list[torch.Tensor] = []
    for k_idx in range(num_kda):
        c_row, r_row = unpack_kda_state(raw_pools[kda_raw[k_idx]], blocks_list[k_idx])
        conv_rows.append(c_row)
        rec_rows.append(r_row)
    return torch.stack(conv_rows, dim=0), torch.stack(rec_rows, dim=0)


def pack_kda_state(
    raw_pool: torch.Tensor,
    block_id: int,
    conv: torch.Tensor,
    recurrent: torch.Tensor,
) -> None:
    """Pack one KDA layer's ``(conv, recurrent)`` state into ``raw_pool``."""
    if block_id <= 0:
        return
    block_u8 = raw_pool[block_id].contiguous().view(torch.uint8)
    ssm_u8 = block_u8[:SSM_ROWS].contiguous().view(torch.float32).reshape(-1, HEAD_DIM)
    ssm_u8[: KDA_HEADS * HEAD_DIM] = recurrent.to(
        dtype=torch.float32, device=raw_pool.device
    ).reshape(KDA_HEADS * HEAD_DIM, HEAD_DIM)
    block_u8[:SSM_ROWS] = ssm_u8.view(torch.uint8).reshape(block_u8[:SSM_ROWS].shape)
    conv_u8 = (
        block_u8[SSM_ROWS : SSM_ROWS + CONV_ROWS]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1)
    )
    active_elems = CONV_TAPS * CONV_CHANNEL_GROUPS * KDA_HEADS * HEAD_DIM
    conv_active = (
        conv[:, :, :KDA_HEADS, :]
        .to(dtype=torch.bfloat16, device=raw_pool.device)
        .contiguous()
        .reshape(-1)
    )
    conv_u8[:active_elems] = conv_active[:active_elems]
    block_u8[SSM_ROWS : SSM_ROWS + CONV_ROWS] = conv_u8.view(torch.uint8).reshape(
        block_u8[SSM_ROWS : SSM_ROWS + CONV_ROWS].shape
    )
    raw_pool[block_id].copy_(
        block_u8.view(raw_pool.dtype).reshape(raw_pool[block_id].shape)
    )


def write_kda_states(
    raw_pools: Sequence[torch.Tensor],
    kda_raw: Sequence[int],
    kda_blocks: torch.Tensor,
    conv: torch.Tensor,
    recurrent: torch.Tensor,
) -> None:
    """Pack all KDA ``(conv, recurrent)`` states back into unified pools."""
    num_kda = len(kda_raw) or KDA_LAYERS
    blocks_list = [
        int(x) for x in kda_blocks.detach().to(device="cpu").reshape(-1).tolist()
    ]
    if len(blocks_list) != num_kda:
        msg = f"expected {num_kda} KDA block ids, got {len(blocks_list)}"
        raise ValueError(msg)
    for k_idx in range(num_kda):
        pack_kda_state(
            raw_pools[kda_raw[k_idx]],
            blocks_list[k_idx],
            conv[k_idx],
            recurrent[k_idx],
        )


class KdaImporter:
    """Copy a request's 69 KDA states from unified-pool blocks into its slot."""

    def __init__(
        self,
        mesh: Mesh,
        state_perm: Sequence[tuple[int, int]] | None = None,
        attention_dp: int = 1,
        num_slots: int = 0,
    ) -> None:
        """Initialize importer for ``mesh``, ``state_perm``, and ``attention_dp``."""
        self._state_perm = (
            None
            if state_perm is None
            else tuple((int(a), int(b)) for a, b in state_perm)
        )
        self.state_perm = self._state_perm
        self._mesh = mesh
        self._attention_dp = int(attention_dp)
        self._num_slots = int(num_slots)
        self.raw_pools: list[torch.Tensor] = []
        self.mla_raw: list[int] = []
        self.kda_raw: list[int] = []
        self.mla_layer_names: list[str] = []
        self.kda_layer_names: list[str] = []
        self._op: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
        self._export_op: Callable[..., tuple[torch.Tensor, ...]] | None = None
        self._ring_op: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None

    def bind(
        self,
        caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]],
        kv_cache_dtype: str = "auto",
    ) -> int:
        """Bind distinct raw pools and layer mappings; return ``num_blocks``."""
        (
            self.raw_pools,
            self.mla_raw,
            self.kda_raw,
            self.mla_layer_names,
            self.kda_layer_names,
        ) = bind_raw_pools(caches, kv_cache_dtype)
        return int(self.raw_pools[0].shape[0])

    @staticmethod
    def _gather_states(
        raws: Sequence[jax.Array],
        kda_raw: Sequence[int],
        kda_blocks: jax.Array,
        conv_dtype: jax.typing.DTypeLike,
        recurrent_dtype: jax.typing.DTypeLike,
    ) -> tuple[jax.Array, jax.Array]:
        """Gather all KDA conv and recurrent states from ``raws``."""
        conv_rows: list[jax.Array] = []
        recurrent_rows: list[jax.Array] = []
        for k in range(len(kda_raw)):
            raw, ids = raws[kda_raw[k]], kda_blocks[k : k + 1]
            ssm = pool_adapters.gather_region(
                raw,
                ids,
                tok0=0,
                ntok=SSM_ROWS,
                out_dtype=jnp.float32,
                out_lanes=HEAD_DIM,
            )
            ssm = jnp.where(ids[0] != 0, ssm, jnp.zeros_like(ssm))
            recurrent_rows.append(
                ssm[0, : KDA_HEADS * HEAD_DIM].reshape(KDA_HEADS, HEAD_DIM, HEAD_DIM)
            )
            taps = pool_adapters.gather_region(
                raw,
                ids,
                tok0=SSM_ROWS,
                ntok=CONV_ROWS,
                out_dtype=jnp.bfloat16,
            )
            taps = jnp.where(ids[0] != 0, taps, jnp.zeros_like(taps))
            taps = taps.reshape(-1)[
                : CONV_TAPS * CONV_CHANNEL_GROUPS * KDA_HEADS * HEAD_DIM
            ]
            conv_rows.append(
                jnp.pad(
                    taps.reshape(
                        CONV_TAPS,
                        CONV_CHANNEL_GROUPS,
                        KDA_HEADS,
                        HEAD_DIM,
                    ),
                    ((0, 0), (0, 0), (0, CONV_HEAD_ROWS - KDA_HEADS), (0, 0)),
                )
            )
        return (
            jnp.stack(conv_rows).astype(conv_dtype),
            jnp.stack(recurrent_rows).astype(recurrent_dtype),
        )

    def _build(self) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        """Register and return the TPU custom op for KDA state import."""
        if self._op is not None:
            return self._op
        bridge = importlib.import_module(".runtime_shim.bridge", __package__)
        register_op: Callable[..., Callable[..., tuple[torch.Tensor, torch.Tensor]]] = (
            bridge.register_op
        )
        num_raw, kda_raw, perm = (
            len(self.raw_pools),
            tuple(self.kda_raw),
            self._state_perm,
        )
        dp, num_slots = self._attention_dp, self._num_slots
        share = group_slots(num_slots, dp) - 1
        world = self._mesh.size
        group = world // dp

        def body(
            conv_in: jax.Array,
            recurrent_in: jax.Array,
            *args: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            """Import 69 KDA states into ``conv_in`` and ``recurrent_in``."""
            conv, recurrent = conv_in[0], recurrent_in[0]
            raws, (kda_blocks, slot) = args[:num_raw], args[num_raw:]
            conv_state, recurrent_state = KdaImporter._gather_states(
                raws,
                kda_raw,
                kda_blocks,
                conv.dtype,
                recurrent.dtype,
            )
            if perm is not None:
                conv_state = jax.lax.ppermute(conv_state, MESH_AXIS, perm)
                recurrent_state = jax.lax.ppermute(recurrent_state, MESH_AXIS, perm)
            rank = jax.lax.axis_index(MESH_AXIS)
            local = slot[0]
            if dp > 1:
                owner = (slot[0] - 1) // share
                local = jnp.where(
                    owner == rank // group,
                    slot[0] - owner * share,
                    0,
                )
            for shift in range(dp):
                pairs = [(q, (q - shift * group) % world) for q in range(world)]
                conv_k, recurrent_k = (
                    (conv_state, recurrent_state)
                    if shift == 0
                    else (
                        jax.lax.ppermute(conv_state, MESH_AXIS, pairs),
                        jax.lax.ppermute(recurrent_state, MESH_AXIS, pairs),
                    )
                )
                k = (rank + shift * group) % world // group
                index = local + k * (share + 1)
                conv = jax.lax.dynamic_update_index_in_dim(
                    conv,
                    conv_k,
                    index,
                    axis=0,
                )
                recurrent = jax.lax.dynamic_update_index_in_dim(
                    recurrent,
                    recurrent_k,
                    index,
                    axis=0,
                )
            return conv[None], recurrent[None]

        self._op = register_op(
            f"pallas::k3_decode_import_kda_r{num_raw}",
            self._mesh,
            body,
            bridge.OpRegistrationSpec(
                in_specs=(P(MESH_AXIS),) * (2 + num_raw) + (P(), P()),
                out_specs=(P(MESH_AXIS),) * 2,
                donate_argnums=(0, 1),
                export_key=repr(("import_kda", kda_raw, perm, dp, num_slots)),
            ),
        )
        return self._op

    def _import_state_cpu(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        kda_blocks: torch.Tensor,
        slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Import 69 KDA states from ``kda_blocks`` into ``slot`` on CPU."""
        conv_state, rec_state = read_kda_states(
            self.raw_pools,
            self.kda_raw,
            kda_blocks,
        )
        conv_cast = conv_state.to(dtype=conv.dtype, device=conv.device)
        rec_cast = rec_state.to(dtype=recurrent.dtype, device=recurrent.device)
        if slot <= 0:
            return conv.clone(), recurrent.clone()
        local_idx = slot
        if self._attention_dp > 1:
            share = group_slots(self._num_slots, self._attention_dp) - 1
            owner = (slot - 1) // share
            if owner != 0:
                return conv.clone(), recurrent.clone()
            local_idx = slot - owner * share
        conv_out = conv.clone()
        rec_out = recurrent.clone()
        if conv_out.ndim == BATCHED_CONV_BANK_NDIM:
            conv_out[0, local_idx] = conv_cast
            rec_out[0, local_idx] = rec_cast
            return conv_out, rec_out
        conv_out[local_idx] = conv_cast
        rec_out[local_idx] = rec_cast
        return conv_out, rec_out

    def import_state(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        kda_blocks: torch.Tensor,
        slot: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy the states in ``kda_blocks`` into ``slot`` of ``(conv, recurrent)``."""
        slot_val = int(slot.item()) if isinstance(slot, torch.Tensor) else int(slot)
        if conv.device.type == "cpu":
            return self._import_state_cpu(conv, recurrent, kda_blocks, slot_val)
        blocks_t = kda_blocks.to(dtype=torch.int32, device=conv.device).contiguous()
        slot_t = torch.tensor([slot_val], dtype=torch.int32, device=conv.device)
        if conv.device.type == "tpu":
            sync_mod = importlib.import_module("torch_tpu._internal.sync")
            sync_mod.synchronize([blocks_t, slot_t], wait=False)
        conv_out, rec_out = self._build()(
            conv,
            recurrent,
            *self.raw_pools,
            blocks_t,
            slot_t,
        )
        if conv.device.type == "tpu":
            sync_mod = importlib.import_module("torch_tpu._internal.sync")
            sync_mod.synchronize([conv_out, rec_out], wait=False)
        return conv_out, rec_out

    @staticmethod
    def _scatter_states(
        raws: Sequence[jax.Array],
        kda_raw: Sequence[int],
        kda_blocks: jax.Array,
        conv_state: jax.Array,
        recurrent_state: jax.Array,
    ) -> list[jax.Array]:
        """Scatter all KDA conv and recurrent states back into ``raws``."""
        out_raws = list(raws)
        for k in range(len(kda_raw)):
            pool_idx = kda_raw[k]
            raw, ids = out_raws[pool_idx], kda_blocks[k : k + 1]
            ssm_shape = jax.eval_shape(
                lambda r=raw, i=ids: pool_adapters.gather_region(
                    r,
                    i,
                    tok0=0,
                    ntok=SSM_ROWS,
                    out_dtype=jnp.float32,
                    out_lanes=HEAD_DIM,
                )
            ).shape
            ssm_prev = pool_adapters.gather_region(
                raw,
                ids,
                tok0=0,
                ntok=SSM_ROWS,
                out_dtype=jnp.float32,
                out_lanes=HEAD_DIM,
            )
            rec_flat = (
                recurrent_state[k]
                .astype(jnp.float32)
                .reshape(1, KDA_HEADS * HEAD_DIM, HEAD_DIM)
            )
            ssm_updated = ssm_prev.at[:, : KDA_HEADS * HEAD_DIM, :].set(rec_flat)
            raw = pool_adapters.scatter_region(
                raw,
                ssm_updated.reshape(ssm_shape),
                ids,
                tok0=0,
                ntok=SSM_ROWS,
            )
            taps_shape = jax.eval_shape(
                lambda r=raw, i=ids: pool_adapters.gather_region(
                    r,
                    i,
                    tok0=SSM_ROWS,
                    ntok=CONV_ROWS,
                    out_dtype=jnp.bfloat16,
                )
            ).shape
            taps_prev = pool_adapters.gather_region(
                raw,
                ids,
                tok0=SSM_ROWS,
                ntok=CONV_ROWS,
                out_dtype=jnp.bfloat16,
            ).reshape(-1)
            active_elems = CONV_TAPS * CONV_CHANNEL_GROUPS * KDA_HEADS * HEAD_DIM
            conv_active = (
                conv_state[k, :, :, :KDA_HEADS, :]
                .astype(jnp.bfloat16)
                .reshape(active_elems)
            )
            taps_updated = (
                taps_prev.at[:active_elems].set(conv_active).reshape(taps_shape)
            )
            out_raws[pool_idx] = pool_adapters.scatter_region(
                raw,
                taps_updated,
                ids,
                tok0=SSM_ROWS,
                ntok=CONV_ROWS,
            )
        return out_raws

    def _build_export(self) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Register and return the TPU custom op for KDA state export."""
        if self._export_op is not None:
            return self._export_op
        bridge = importlib.import_module(".runtime_shim.bridge", __package__)
        register_op: Callable[..., Callable[..., tuple[torch.Tensor, ...]]] = (
            bridge.register_op
        )
        num_raw, kda_raw, perm = (
            len(self.raw_pools),
            tuple(self.kda_raw),
            self._state_perm,
        )
        inv_perm = None if perm is None else tuple((dst, src) for src, dst in perm)
        dp, num_slots = self._attention_dp, self._num_slots
        share = group_slots(num_slots, dp) - 1
        world = self._mesh.size
        group = world // dp

        def body(
            conv_in: jax.Array,
            recurrent_in: jax.Array,
            *args: jax.Array,
        ) -> tuple[jax.Array, ...]:
            """Export ``slot`` from ``conv_in`` and ``recurrent_in`` into ``raws``."""
            conv, recurrent = conv_in[0], recurrent_in[0]
            raws, (kda_blocks, slot) = args[:num_raw], args[num_raw:]
            local = slot[0]
            if dp > 1:
                owner = (slot[0] - 1) // share
                rank = jax.lax.axis_index(MESH_AXIS)
                k = (owner - rank // group) % dp
                local = (slot[0] - owner * share) + k * (share + 1)
            conv_state = conv[local]
            recurrent_state = recurrent[local]
            if inv_perm is not None:
                conv_state = jax.lax.ppermute(conv_state, MESH_AXIS, inv_perm)
                recurrent_state = jax.lax.ppermute(recurrent_state, MESH_AXIS, inv_perm)
            updated = KdaImporter._scatter_states(
                raws,
                kda_raw,
                kda_blocks,
                conv_state,
                recurrent_state,
            )
            return tuple(r[None] for r in updated)

        self._export_op = register_op(
            f"pallas::k3_decode_export_kda_r{num_raw}",
            self._mesh,
            body,
            bridge.OpRegistrationSpec(
                in_specs=(P(MESH_AXIS),) * (2 + num_raw) + (P(), P()),
                out_specs=(P(MESH_AXIS),) * num_raw,
                donate_argnums=tuple(range(2, 2 + num_raw)),
                export_key=repr(("export_kda", kda_raw, perm, dp, num_slots)),
            ),
        )
        return self._export_op

    def _export_state_cpu(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        kda_blocks: torch.Tensor,
        slot: int,
    ) -> list[torch.Tensor]:
        """Export 69 KDA states from ``slot`` back into ``kda_blocks`` on CPU."""
        if slot <= 0:
            return list(self.raw_pools)
        local_idx = slot
        if self._attention_dp > 1:
            share = group_slots(self._num_slots, self._attention_dp) - 1
            owner = (slot - 1) // share
            if owner != 0:
                return list(self.raw_pools)
            local_idx = slot - owner * share
        if conv.ndim == BATCHED_CONV_BANK_NDIM:
            conv_slot = conv[0, local_idx]
            rec_slot = recurrent[0, local_idx]
        else:
            conv_slot = conv[local_idx]
            rec_slot = recurrent[local_idx]
        write_kda_states(
            self.raw_pools,
            self.kda_raw,
            kda_blocks,
            conv_slot,
            rec_slot,
        )
        return list(self.raw_pools)

    def export_state(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        kda_blocks: torch.Tensor,
        slot: int | torch.Tensor,
    ) -> list[torch.Tensor]:
        """Copy ``slot`` of ``(conv, recurrent)`` back into ``kda_blocks``."""
        slot_val = int(slot.item()) if isinstance(slot, torch.Tensor) else int(slot)
        if conv.device.type == "cpu":
            return self._export_state_cpu(conv, recurrent, kda_blocks, slot_val)
        blocks_t = kda_blocks.to(device=conv.device, dtype=torch.int32).contiguous()
        slot_tensor = torch.tensor([slot_val], dtype=torch.int32, device=conv.device)
        if conv.device.type == "tpu":
            sync_mod = importlib.import_module("torch_tpu._internal.sync")
            sync_mod.synchronize([blocks_t, slot_tensor], wait=False)
        out = self._build_export()(
            conv,
            recurrent,
            *self.raw_pools,
            blocks_t,
            slot_tensor,
        )
        out_list = list(out)
        if conv.device.type == "tpu":
            sync_mod = importlib.import_module("torch_tpu._internal.sync")
            sync_mod.synchronize(out_list, wait=False)
        return out_list

    def import_ring(
        self,
        draft_config: object,
        kda_blocks: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Import a transferred request's DSpark draft ring from KDA state blocks."""
        if self._ring_op is None:
            bridge = importlib.import_module(".runtime_shim.bridge", __package__)
            register_op: Callable[
                ..., Callable[..., tuple[torch.Tensor, torch.Tensor]]
            ] = bridge.register_op
            dspark = importlib.import_module(".kernels.dspark", __package__)
            num_raw, kda_raw, perm = (
                len(self.raw_pools),
                tuple(self.kda_raw),
                self._state_perm,
            )
            cfg = draft_config

            def body(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
                """Unpack draft ring rows and positions from KDA state blocks."""
                raws, (blocks_arg, len_arg) = args[:num_raw], args[num_raw:]
                packed = read_draft_rings(
                    raws,
                    kda_raw,
                    blocks_arg[None],
                    dspark.RING_ROW_LANES,
                )
                ring, meta = dspark.unpack_ring(cfg, packed[0], len_arg[0])
                if perm is not None:
                    ring = jax.lax.ppermute(ring, MESH_AXIS, perm)
                return ring[None], dspark.ring_positions(cfg, meta)[None]

            cfg_tuple = (
                dataclasses.astuple(cfg)
                if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type)
                else repr(cfg)
            )
            self._ring_op = register_op(
                f"pallas::k3_decode_import_ring_r{num_raw}",
                self._mesh,
                body,
                bridge.OpRegistrationSpec(
                    in_specs=(P(MESH_AXIS),) * num_raw + (P(), P()),
                    out_specs=(P(MESH_AXIS), P()),
                    export_key=repr(("import_ring", kda_raw, perm, cfg_tuple)),
                ),
            )
        dev = self.raw_pools[0].device
        blocks_t = kda_blocks.to(dtype=torch.int32, device=dev).contiguous()
        len_t = torch.tensor([length], dtype=torch.int32, device=dev)
        if dev.type == "tpu":
            sync_mod = importlib.import_module("torch_tpu._internal.sync")
            sync_mod.synchronize([blocks_t, len_t], wait=False)
        ring_out, pos_out = self._ring_op(
            *self.raw_pools,
            blocks_t,
            len_t,
        )
        if dev.type == "tpu":
            sync_mod = importlib.import_module("torch_tpu._internal.sync")
            sync_mod.synchronize([ring_out, pos_out], wait=False)
        return ring_out, pos_out


def dp_row_owners(batch_size: int) -> int:
    """Return local sequence-parallel row owners (1 up to B=4, else 4)."""
    return 1 if batch_size <= MAX_SINGLE_OWNER_BATCH else DEFAULT_ROW_OWNERS


def dp_group_rows(batch_size: int, attention_dp: int) -> np.ndarray:
    """Map group-major DP rows to interleaved batch rows owned by each rank pair."""
    owners = dp_row_owners(batch_size)
    per = batch_size // attention_dp // owners
    q = np.arange(batch_size // attention_dp)
    return np.concatenate(
        [
            q // per * (batch_size // owners) + grp * per + q % per
            for grp in range(attention_dp)
        ]
    ).astype(np.int32)


def dp_batch_sizes(requested: tuple[int, ...], attention_dp: int) -> tuple[int, ...]:
    """Resolve valid attention-DP decode batch sizes covering ``requested``."""
    sizes = tuple(
        b for b in DP_BATCH_SIZES if attention_dp == DP2_GROUPS or b >= DP4_MIN_BATCH
    )
    first = sum(b < min(requested) for b in sizes)
    last = sum(b < max(requested) for b in sizes)
    return sizes[first : last + 1]


def decode_step_options(batch_size: int, attention_dp: int = 1) -> dict[str, object]:
    """Return tuned ``decode_paged`` keyword options for ``batch_size`` and DP."""
    scatter = attention_dp > 1 or batch_size in TOKEN_SCATTER_BATCHES
    tile = (
        batch_size
        if batch_size <= MAX_WHOLE_CACHE_TILE_BATCH
        else LARGE_CACHE_TILE
        if batch_size >= LARGE_CACHE_TILE_MIN_BATCH
        else math.gcd(MID_CACHE_TILE, batch_size)
    )
    opts: dict[str, object] = {
        "gate_first_kda_projection": True,
        "residual_reduction_order": "relaxed",
        "mla_cache_batch_tile_size": min(tile, batch_size),
        "alias_expert_scales": batch_size >= MIN_ALIAS_EXPERT_SCALES_BATCH,
        "attention_token_scatter": scatter,
        "moe_token_scatter_output": scatter,
        "moe_routed_reduce_scatter": batch_size in ROUTED_RS_BATCHES,
        "sequence_parallel_pre_attention": scatter,
    }
    if attention_dp > 1:
        owners = dp_row_owners(batch_size)
        opts["attention_dp"] = attention_dp
        if owners != DEFAULT_ROW_OWNERS:
            opts["sequence_parallel_size"] = owners
    return opts


def select_verify_kda_rows(
    slots: jax.Array,
    accepted_counts: jax.Array,
    *,
    requests: int,
    rows_per: int | None = None,
    **dp_options: int | jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Select per-request verifier source rows and target KDA bank slots."""
    unknown_keys = set(dp_options) - {"attention_dp", "bank_slots", "group"}
    if unknown_keys:
        msg = f"Unexpected select_verify_kda_rows options: {sorted(unknown_keys)}"
        raise TypeError(msg)
    attention_dp_obj = dp_options.get("attention_dp", 1)
    attention_dp = int(attention_dp_obj) if isinstance(attention_dp_obj, int) else 1
    bank_slots_obj = dp_options.get("bank_slots", DEFAULT_VERIFY_BANK_SLOTS)
    bank_slots = (
        int(bank_slots_obj)
        if isinstance(bank_slots_obj, int)
        else DEFAULT_VERIFY_BANK_SLOTS
    )
    group = dp_options.get("group")
    width = int(slots.shape[0]) // requests if rows_per is None else rows_per
    targets = slots.reshape(requests, width)[:, 0]
    acc = accepted_counts.reshape(-1)[:requests]
    if attention_dp == 1:
        sources = jnp.arange(requests, dtype=jnp.int32) * width + jnp.where(
            targets == 0,
            width - 1,
            acc,
        )
        return sources, targets
    share = requests // attention_dp
    bank = bank_slots // attention_dp
    grp = (
        jax.lax.axis_index(MESH_AXIS) // (TP_SIZE // attention_dp)
        if group is None
        else jnp.asarray(group, dtype=jnp.int32)
    )
    own = jax.lax.dynamic_slice_in_dim(targets, grp * share, share)
    own_acc = jax.lax.dynamic_slice_in_dim(acc, grp * share, share)
    rows = jnp.arange(share, dtype=jnp.int32) * width + jnp.where(
        own == 0,
        width - 1,
        own_acc,
    )
    local = jnp.where(own == 0, 0, own - grp * (bank - 1))
    group_rows = share * width
    sources = jnp.concatenate([rows + k * group_rows for k in range(attention_dp)])
    targets_dp = jnp.concatenate([local + k * bank for k in range(attention_dp)])
    return sources, targets_dp
