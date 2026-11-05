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
"""vLLM's unified pools as the K3 programs see them (P/D over the Raiden KV
connector).

Each ``KVCacheTensor`` is one raw pool of ``layout.cache_layout()`` blocks,
shared by one MLA layer and three KDA layers: MLA latent pages and, in a KDA
group's state block, the layer's state (rows 0-76 recurrent, 77-80 conv taps)
and the draft-ring rows (81-95), heads in vLLM TP-rank order. The prefill
program writes them; the decoder reads the latent pages in place and imports a
request's KDA state (into its private slot) and draft ring (:class:`KdaImporter`),
writing the state back at the block boundaries of a recompute.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.kimi_k3.kernels.dsa_cache import index_block_shape
from vllm_torchtpu.megakernel.kimi_k3.layout import (
    CONV_HEAD_ROWS,
    CONV_TAPS,
    HEAD_DIM,
    KDA_HEADS,
    LAYOUT,
    STAGE_SPLIT,
    cache_layout,
)
from vllm_torchtpu.megakernel.kimi_k3.program import TP_SIZE

logger = init_logger(__name__)

MESH_AXIS = "tp"
MLA_LAYERS = len(LAYOUT.attention_layers)
KDA_LAYERS = len(LAYOUT.state_layers)
SSM_ROWS = LAYOUT.region("recurrent").rows
CONV_ROWS = LAYOUT.region("conv").rows
DRAFT_ROW0 = LAYOUT.region("draft").first_row
DRAFT_ROWS = LAYOUT.region("draft").rows
_MLA_INDEX = {layer: m for m, layer in enumerate(LAYOUT.attention_layers)}
_KDA_INDEX = {layer: k for k, layer in enumerate(LAYOUT.state_layers)}
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index(name: str) -> int:
    match = _LAYER_RE.search(name)
    if match is None:
        raise ValueError(f"cannot find a layer index in module name {name!r}")
    return int(match.group(1))


def mla_state_index(layer: int) -> int:
    """Position of model layer ``layer`` on the kernel's MLA axis."""
    return _MLA_INDEX[layer]


def kda_state_index(layer: int) -> int:
    """Position of model layer ``layer`` on the kernel's KDA axis."""
    return _KDA_INDEX[layer]


def group_slots(num_slots: int, groups: int) -> int:
    """A rank's KDA slots per held head shard: under attention DP each group
    allocates its own share ``n - 1`` of the ``num_slots - 1`` request slots
    (group ``g``: slots ``g (n - 1) + 1 .. (g + 1) (n - 1)``), which its ranks
    hold at bank entries ``1 .. n - 1``, after the null slot 0."""
    return -(-(num_slots - 1) // groups) + 1


class SlotMap:
    """Host-side request-id -> private KDA slot allocation; slot 0 is the null
    slot. Under attention DP a request's slot is in its group's share
    (:func:`group_slots`)."""

    def __init__(self, num_slots: int, groups: int = 1) -> None:
        share = group_slots(num_slots, groups) - 1
        self._free = [
            list(range(min((g + 1) * share, num_slots - 1), g * share, -1))
            for g in range(groups)
        ]
        self._share = share
        self._slots: dict[str, int] = {}

    def __contains__(self, req_id: str) -> bool:
        return req_id in self._slots

    def group(self, req_id: str) -> int | None:
        slot = self._slots.get(req_id)
        return None if slot is None else (slot - 1) // self._share

    def allocate(self, req_id: str, group: int = 0) -> int:
        if req_id in self._slots:
            return self._slots[req_id]
        if not self._free[group]:
            raise RuntimeError(
                "K3 decode: no free KDA slot (more live requests than --max-num-seqs?)"
            )
        slot = self._free[group].pop()
        self._slots[req_id] = slot
        return slot

    def release(self, req_ids) -> None:
        for req_id in req_ids:
            slot = self._slots.pop(req_id, None)
            if slot is not None:
                self._free[(slot - 1) // self._share].append(slot)


def _raw_tensor(kv_cache: Any) -> torch.Tensor:
    if isinstance(kv_cache, (list, tuple)):
        if len(kv_cache) != 1:
            raise ValueError("expected a single raw pool per layer")
        return kv_cache[0]
    return kv_cache


def is_mla_layer(layer: int) -> bool:
    """K3's full-attention (MLA) layers: every fourth, and the last."""
    return layer in _MLA_INDEX


# vLLM numbers a speculator's layers after the target's: the MLA DSpark
# draft's latent layers are 93.. (MLA pools of the target's MLA group).
DRAFT_LAYER0 = MLA_LAYERS + KDA_LAYERS


def draft_raw_pools(caches: Mapping[str, Any]) -> list[torch.Tensor]:
    """The MLA DSpark draft layers' pools (layers ``DRAFT_LAYER0`` on, in
    order), or none."""
    layers = sorted(
        (layer_index(name), cache)
        for name, cache in caches.items()
        if layer_index(name) >= DRAFT_LAYER0
    )
    return [_raw_tensor(cache) for _, cache in layers]


# Kimi-K3.1's index-key cache: one block-major tensor bound under every MLA
# layer's index-cache name (as a unified pool is under its four layers).
INDEX_CACHE = ".self_attn.indexer.k_cache"


def index_raw_cache(caches: Mapping[str, Any]) -> torch.Tensor | None:
    """The index-key cache ``[blocks, 24, tokens + 32, 128]`` (block ``b``'s
    page of kernel MLA layer ``m`` at ``[b, m]``), or None (Kimi-K3)."""
    found = {
        mla_state_index(layer_index(name)): _raw_tensor(cache)
        for name, cache in caches.items()
        if name.endswith(INDEX_CACHE)
    }
    if not found:
        return None
    # One object, not just one storage: the decoder adopts each step's cache
    # into it in place (set_), which other views of the storage would miss.
    tensors = {id(tensor): tensor for tensor in found.values()}
    if len(found) != MLA_LAYERS or len(tensors) != 1:
        raise RuntimeError(
            f"expected one index-key cache under all {MLA_LAYERS} MLA layers, "
            f"found {len(found)} layers over {len(tensors)} tensors"
        )
    (tensor,) = tensors.values()
    if tensor.dim() != 4 or tensor.shape[1] != MLA_LAYERS:
        raise RuntimeError(
            f"the index-key cache must be [blocks, {MLA_LAYERS}, rows, 128], "
            f"not {tuple(tensor.shape)}"
        )
    return tensor


def index_cache_names(names: Iterable[str]) -> list[str]:
    """The index-key cache names among ``names``, by layer."""
    return sorted((n for n in names if n.endswith(INDEX_CACHE)), key=layer_index)


def index_cache_rows(block_size: int, kv_cache_dtype: str) -> int:
    """Rows of one MLA layer's index page in a ``block_size``-token block:
    the decoder's page (its tokens then 32 scale rows) for each decode page
    the block holds -- one on the decoder, ``STAGE_SPLIT`` on a prefill stage."""
    page = cache_layout(kv_cache_dtype).tokens_per_block
    if block_size not in (page, STAGE_SPLIT * page):
        raise ValueError(
            f"index pages fill {page}- or {STAGE_SPLIT * page}-token blocks, "
            f"not {block_size}"
        )
    return block_size // page * index_block_shape(page)[0]


def allocate_index_cache(
    num_blocks: int,
    names: Sequence[str],
    block_size: int,
    kv_cache_dtype: str,
    device: Any,
) -> dict[str, torch.Tensor]:
    """One zeroed ``[num_blocks, len(names), rows, 128]`` uint8 index-key
    cache under every name (layer slot = the name's order by layer)."""
    ordered = index_cache_names(names)
    rows = index_cache_rows(block_size, kv_cache_dtype)
    tensor = torch.zeros(
        (num_blocks, len(ordered), rows, 128), dtype=torch.uint8, device=device
    )
    return {name: tensor for name in ordered}


def index_block_budget(
    available_bytes: int,
    block_bytes: int,
    layers: int,
    block_size: int,
    kv_cache_dtype: str,
) -> int:
    """Blocks that fit ``available_bytes`` when a block takes ``block_bytes``
    of unified pages and the index pages of ``layers`` MLA layers."""
    index_bytes = index_cache_rows(block_size, kv_cache_dtype) * 128
    return int(available_bytes) // (int(block_bytes) + layers * index_bytes)


def stage_index_cache(
    caches: Mapping[str, Any], mla_layers: Sequence[int]
) -> torch.Tensor | None:
    """A prefill stage's index-key cache ``[blocks, len(mla_layers), rows,
    128]`` (slot ``j``: the stage's ``j``-th MLA layer), or None (Kimi-K3)."""
    held = set(mla_layers)
    found = {
        layer_index(name): _raw_tensor(cache)
        for name, cache in caches.items()
        if name.endswith(INDEX_CACHE) and layer_index(name) in held
    }
    if not found:
        return None
    # One object, as index_raw_cache: the stage adopts its copies with set_.
    tensors = {id(tensor): tensor for tensor in found.values()}
    if len(found) != len(held) or len(tensors) != 1:
        raise RuntimeError(
            f"expected one index-key cache under the stage's {len(held)} MLA "
            f"layers, found {len(found)} layers over {len(tensors)} tensors"
        )
    (tensor,) = tensors.values()
    if tensor.dim() != 4 or tensor.shape[1] != len(held):
        raise RuntimeError(
            f"the stage's index-key cache must be [blocks, {len(held)}, rows, "
            f"128], not {tuple(tensor.shape)}"
        )
    return tensor


def bind_raw_pools(
    caches: Mapping[str, Any],
    kv_cache_dtype: str | None = None,
) -> tuple[list[torch.Tensor], list[int], list[int], list[str], list[str]]:
    """vLLM's unified-pool tensors (``caches``: layer name -> KV cache
    tensor) and which one each layer lives in.

    Returns ``(raw_pools, mla_raw, kda_raw, mla_layer_names,
    kda_layer_names)``: the distinct pools, the pool index of kernel MLA layer
    ``m`` and KDA layer ``k``, and the layers' names in kernel order.
    """
    raw_pools: list[torch.Tensor] = []
    index_of_ptr: dict[int, int] = {}

    def index_of(tensor: torch.Tensor) -> int:
        key = tensor.data_ptr()
        if key not in index_of_ptr:
            index_of_ptr[key] = len(raw_pools)
            raw_pools.append(tensor)
        return index_of_ptr[key]

    layers = sorted(
        (layer_index(name), name, cache)
        for name, cache in caches.items()
        # draft layers: draft_raw_pools; index keys: index_raw_cache
        if layer_index(name) < DRAFT_LAYER0 and not name.endswith(INDEX_CACHE)
    )
    mla = [(layer, name, cache) for layer, name, cache in layers if is_mla_layer(layer)]
    kda = [
        (layer, name, cache) for layer, name, cache in layers if not is_mla_layer(layer)
    ]
    if len(mla) != MLA_LAYERS or len(kda) != KDA_LAYERS:
        raise RuntimeError(
            f"expected {MLA_LAYERS} MLA and {KDA_LAYERS} KDA layers, "
            f"found {len(mla)} and {len(kda)}"
        )
    mla_raw, mla_names = [0] * MLA_LAYERS, [""] * MLA_LAYERS
    for layer, name, cache in mla:
        mla_raw[mla_state_index(layer)] = index_of(_raw_tensor(cache))
        mla_names[mla_state_index(layer)] = name
    kda_raw, kda_names = [0] * KDA_LAYERS, [""] * KDA_LAYERS
    for layer, name, cache in kda:
        kda_raw[kda_state_index(layer)] = index_of(_raw_tensor(cache))
        kda_names[kda_state_index(layer)] = name
    if len({tuple(t.shape) for t in raw_pools}) != 1:
        raise NotImplementedError(
            f"the unified pools differ in shape: {[tuple(t.shape) for t in raw_pools]}"
        )
    cache_layout(kv_cache_dtype or str(raw_pools[0].dtype)).check(raw_pools)
    return raw_pools, mla_raw, kda_raw, mla_names, kda_names


def tp_rank_permutation(tp_rank: int, position: int) -> list[tuple[int, int]] | None:
    """``(source, destination)`` mesh positions that bring the KDA head shard
    of TP rank ``q`` -- what the KV connector ships between devices of equal
    TP rank -- to mesh position ``q``, whose kernel rank owns those heads;
    None when the two orders coincide. A prefill writing the pool uses the
    inverse (each pair swapped). A collective over the default process group
    (one process per device)."""
    import torch

    gathered: list[Any] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, (int(tp_rank), int(position)))
    position_of_tp = {int(t): int(p) for t, p in gathered}
    if (
        sorted(position_of_tp) != list(range(TP_SIZE))
        or len(set(position_of_tp.values())) != TP_SIZE
    ):
        raise RuntimeError(
            f"K3 kernels: cannot map TP ranks onto the kernel mesh: {position_of_tp}"
        )
    perm = [(position_of_tp[q], q) for q in range(TP_SIZE)]
    return None if all(src == dst for src, dst in perm) else perm


def kda_blocks_at_length(
    block_tables: Sequence[np.ndarray],
    kda_groups: Sequence[tuple[int, int]],
    seq_len: int,
) -> np.ndarray:
    """Per KDA layer (``kda_groups``: its KV cache group and block size), the
    state block of a request ``seq_len`` tokens long: the block holding
    position ``seq_len - 1``, as vLLM's metadata builders pick the block a
    step writes; the null block for an empty request. int32 ``[69]``."""
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
    block_tables: Sequence[np.ndarray],
    kda_groups: Sequence[tuple[int, int]],
    column: int,
) -> np.ndarray:
    """Per KDA layer, the block in column ``column`` of its group's table. In
    vLLM's hybrid ``align`` mode column ``p`` holds the state after ``(p + 1)
    * block_size`` tokens -- where a prefix-cache hit of ``C`` tokens left it
    (column ``C // block_size - 1``). int32 ``[69]``."""
    return np.array([block_tables[g][column] for g, _ in kda_groups], np.int32)


def write_draft_rings(raws: list, kda_raw: Sequence[int], packed, blocks) -> list:
    """(inside an op) packed draft rings ``[n, 69, DRAFT_ROWS, lanes]``
    (``kernels.dspark.pack_ring``) into the draft rows of each sequence's 69
    KDA state blocks ``[n, 69]``."""
    from vllm_torchtpu.kernels import pool_adapters

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


def read_draft_rings(raws, kda_raw: Sequence[int], blocks, lanes: int):
    """(inside an op) the draft rows of each sequence's 69 KDA state blocks
    ``[n, 69]`` -> ``[n, 69, DRAFT_ROWS, lanes]`` bf16."""
    from vllm_torchtpu.kernels import pool_adapters

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


class KdaImporter:
    """Copies a request's 69 KDA states from their unified-pool blocks into
    its private slot (the decoder's first step of a transferred request), and
    back into blocks (a recompute's block boundaries)."""

    def __init__(
        self,
        mesh: Mesh,
        state_perm: list[tuple[int, int]] | None = None,
        attention_dp: int = 1,
        num_slots: int = 0,
    ) -> None:
        # (source, destination) mesh positions: a device's pool holds its
        # vLLM TP rank's head shard, the kernel rank at mesh position p owns
        # heads 3p..3p+2 (:func:`tp_rank_permutation`). None: the orders
        # coincide.
        self._state_perm = (
            None
            if state_perm is None
            else tuple((int(a), int(b)) for a, b in state_perm)
        )
        self._mesh = mesh
        # Attention DP: mesh position p holds the head shards p % G + k G
        # (G = 32 / attention_dp), shard k's state at bank entry s + k n for
        # its group's slot s of n (:func:`group_slots`).
        self._attention_dp = int(attention_dp)
        self._num_slots = int(num_slots)
        self.raw_pools: list[torch.Tensor] = []
        self.mla_raw: list[int] = []  # kernel MLA axis -> raw pool index
        self.kda_raw: list[int] = []  # kernel KDA axis -> raw pool index
        self.mla_layer_names: list[str] = []  # kernel MLA axis -> layer name
        self.kda_layer_names: list[str] = []  # kernel KDA axis -> layer name
        self._op: Any = None
        self._export_op: Any = None
        self._ring_op: Any = None

    def bind(self, caches: Mapping[str, Any], kv_cache_dtype: str = "auto") -> int:
        """Collect the distinct raw pools and each layer's pool; returns num_blocks."""
        (
            self.raw_pools,
            self.mla_raw,
            self.kda_raw,
            self.mla_layer_names,
            self.kda_layer_names,
        ) = bind_raw_pools(caches, kv_cache_dtype)
        return int(self.raw_pools[0].shape[0])

    def _build(self) -> Any:
        if self._op is not None:
            return self._op
        from vllm_torchtpu.kernels import pool_adapters
        from vllm_torchtpu.megakernel.bridge import register_op

        num_raw, kda_raw, perm = (
            len(self.raw_pools),
            tuple(self.kda_raw),
            self._state_perm,
        )
        dp, num_slots = self._attention_dp, self._num_slots
        share = group_slots(num_slots, dp) - 1
        world = self._mesh.size
        group = world // dp

        def body(conv, recurrent, *args):
            conv, recurrent = conv[0], recurrent[0]
            raws, (kda_blocks, slot) = args[:num_raw], args[num_raw:]
            conv_rows, recurrent_rows = [], []
            for k in range(KDA_LAYERS):
                raw, ids = raws[kda_raw[k]], kda_blocks[k : k + 1]
                ssm = pool_adapters.gather_region(
                    raw,
                    ids,
                    tok0=0,
                    ntok=SSM_ROWS,
                    out_dtype=jnp.float32,
                    out_lanes=HEAD_DIM,
                )
                # Block 0 is padding, not a stored KDA state. FP8 pools
                # are uninitialized, and warmup also writes MLA latents to
                # their null blocks; neither byte pattern is valid SSM data.
                ssm = jnp.where(ids[0] != 0, ssm, jnp.zeros_like(ssm))
                recurrent_rows.append(
                    ssm[0, : KDA_HEADS * HEAD_DIM].reshape(
                        KDA_HEADS, HEAD_DIM, HEAD_DIM
                    )
                )
                taps = pool_adapters.gather_region(
                    raw, ids, tok0=SSM_ROWS, ntok=CONV_ROWS, out_dtype=jnp.bfloat16
                )
                taps = jnp.where(ids[0] != 0, taps, jnp.zeros_like(taps))
                taps = taps.reshape(-1)[: CONV_TAPS * 3 * KDA_HEADS * HEAD_DIM]
                conv_rows.append(
                    jnp.pad(
                        taps.reshape(CONV_TAPS, 3, KDA_HEADS, HEAD_DIM),
                        ((0, 0), (0, 0), (0, CONV_HEAD_ROWS - KDA_HEADS), (0, 0)),
                    )
                )
            conv_state = jnp.stack(conv_rows).astype(conv.dtype)
            recurrent_state = jnp.stack(recurrent_rows).astype(recurrent.dtype)
            if perm is not None:
                conv_state = jax.lax.ppermute(conv_state, MESH_AXIS, perm)
                recurrent_state = jax.lax.ppermute(recurrent_state, MESH_AXIS, perm)
            rank = jax.lax.axis_index(MESH_AXIS)
            local = slot[0]
            if dp > 1:
                # Only the request's group holds its slot; the other groups'
                # ranks write their null slot.
                owner = (slot[0] - 1) // share
                local = jnp.where(owner == rank // group, slot[0] - owner * share, 0)
            for shift in range(dp):
                # Shard (p + shift G) mod 32 from its mesh position: held
                # shard k of this rank.
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
                conv = jax.lax.dynamic_update_index_in_dim(conv, conv_k, index, axis=0)
                recurrent = jax.lax.dynamic_update_index_in_dim(
                    recurrent, recurrent_k, index, axis=0
                )
            return conv[None], recurrent[None]

        self._op = register_op(
            f"pallas::k3_decode_import_kda_r{num_raw}",
            self._mesh,
            body,
            (P(MESH_AXIS),) * (2 + num_raw) + (P(), P()),
            (P(MESH_AXIS),) * 2,
            donate_argnums=(0, 1),
            export_key=repr(("import_kda", kda_raw, perm, dp, num_slots)),
        )
        return self._op

    def import_state(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        kda_blocks: torch.Tensor,
        slot: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The states in ``kda_blocks`` (``[69]`` int32) into ``slot``;
        returns the updated ``(conv, recurrent)`` slot arrays."""
        slot_tensor = torch.tensor([slot], dtype=torch.int32).to("tpu")
        return self._build()(
            conv,
            recurrent,
            *self.raw_pools,
            kda_blocks.to(torch.int32).contiguous(),
            slot_tensor,
        )

    def _build_export(self) -> Any:
        if self._export_op is not None:
            return self._export_op
        from vllm_torchtpu.kernels import pool_adapters
        from vllm_torchtpu.megakernel.bridge import register_op

        num_raw, kda_raw = len(self.raw_pools), tuple(self.kda_raw)
        perm = (
            None
            if self._state_perm is None
            else tuple((b, a) for a, b in self._state_perm)
        )
        dp, num_slots = self._attention_dp, self._num_slots
        share = group_slots(num_slots, dp) - 1
        world = self._mesh.size
        group = world // dp

        def body(conv, recurrent, *args):
            conv, recurrent = conv[0], recurrent[0]
            raws, (kda_blocks, slot) = list(args[:num_raw]), args[num_raw:]
            rank = jax.lax.axis_index(MESH_AXIS)
            owner = (slot[0] - 1) // share if dp > 1 else 0
            local = slot[0] - owner * share

            def held(shift):
                # Held shard (p + shift G) mod 32 of mesh position p.
                k = (rank + shift * group) % world // group
                index = local + k * (share + 1)
                return (
                    jax.lax.dynamic_index_in_dim(conv, index, keepdims=False),
                    jax.lax.dynamic_index_in_dim(recurrent, index, keepdims=False),
                )

            conv_state, recurrent_state = held(0)
            for shift in range(1, dp):
                # The import's placement reversed: each shard back to its mesh
                # position, from the rank of the request's group holding it.
                pairs = [(q, (q + shift * group) % world) for q in range(world)]
                conv_k, recurrent_k = (
                    jax.lax.ppermute(state, MESH_AXIS, pairs) for state in held(shift)
                )
                sender = (rank - shift * group) % world // group == owner
                conv_state = jnp.where(sender, conv_k, conv_state)
                recurrent_state = jnp.where(sender, recurrent_k, recurrent_state)
            if perm is not None:
                conv_state = jax.lax.ppermute(conv_state, MESH_AXIS, perm)
                recurrent_state = jax.lax.ppermute(recurrent_state, MESH_AXIS, perm)
            for k in range(KDA_LAYERS):
                index, ids = kda_raw[k], kda_blocks[k : k + 1]
                ssm_shape = jax.eval_shape(
                    lambda raw=raws[index], ids=ids: pool_adapters.gather_region(
                        raw,
                        ids,
                        tok0=0,
                        ntok=SSM_ROWS,
                        out_dtype=jnp.float32,
                        out_lanes=HEAD_DIM,
                    )
                ).shape
                ssm = recurrent_state[k].astype(jnp.float32).reshape(1, -1, HEAD_DIM)
                ssm = jnp.pad(ssm, ((0, 0), (0, ssm_shape[1] - ssm.shape[1]), (0, 0)))
                raws[index] = pool_adapters.scatter_region(
                    raws[index], ssm, ids, tok0=0, ntok=SSM_ROWS
                )
                conv_shape = jax.eval_shape(
                    lambda raw=raws[index], ids=ids: pool_adapters.gather_region(
                        raw, ids, tok0=SSM_ROWS, ntok=CONV_ROWS, out_dtype=jnp.bfloat16
                    )
                ).shape
                taps = conv_state[k, :, :, :KDA_HEADS].astype(jnp.bfloat16)
                taps = taps.reshape(1, -1)
                taps = jnp.pad(
                    taps, ((0, 0), (0, int(np.prod(conv_shape[1:])) - taps.shape[1]))
                )
                raws[index] = pool_adapters.scatter_region(
                    raws[index],
                    taps.reshape(conv_shape),
                    ids,
                    tok0=SSM_ROWS,
                    ntok=CONV_ROWS,
                )
            return tuple(raws)

        self._export_op = register_op(
            f"pallas::k3_decode_export_kda_r{num_raw}",
            self._mesh,
            body,
            (P(MESH_AXIS),) * (2 + num_raw) + (P(), P()),
            (P(MESH_AXIS),) * num_raw,
            donate_argnums=tuple(range(2, 2 + num_raw)),
            export_key=repr(("export_kda", kda_raw, perm, dp, num_slots)),
        )
        return self._export_op

    def export_state(
        self,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        kda_blocks: np.ndarray,
        slot: int,
    ) -> None:
        """``slot``'s states into ``kda_blocks`` (``[69]`` int32), in place."""
        pools = self._build_export()(
            conv,
            recurrent,
            *self.raw_pools,
            torch.from_numpy(np.asarray(kda_blocks, np.int32)).to("tpu"),
            torch.tensor([slot], dtype=torch.int32).to("tpu"),
        )
        for old, fresh in zip(self.raw_pools, pools, strict=True):
            if fresh is not old:
                old.set_(fresh)  # keep vLLM's and the connector's tensor identity

    def import_ring(
        self, draft_config: Any, kda_blocks: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A transferred request's DSpark draft ring from the draft rows of its
        KDA state blocks (``[69]``): ``(cache [1, layers, window, 128] this
        rank's KV head, positions [1, window])`` -- all ``-1`` (a cold ring)
        unless the prefill wrote a ring for ``length`` tokens there."""
        if self._ring_op is None:
            from vllm_torchtpu.megakernel.bridge import register_op
            from vllm_torchtpu.megakernel.kimi_k3.program import kernel_modules

            (dspark,) = kernel_modules("dspark")
            num_raw, kda_raw, perm = (
                len(self.raw_pools),
                tuple(self.kda_raw),
                self._state_perm,
            )
            c = draft_config

            def body(*args):
                raws, (kda_blocks, length) = args[:num_raw], args[num_raw:]
                packed = read_draft_rings(
                    raws, kda_raw, kda_blocks[None], dspark.RING_ROW_LANES
                )
                ring, meta = dspark.unpack_ring(c, packed[0], length[0])
                if perm is not None:  # pools hold TP-rank order (see the KDA import)
                    ring = jax.lax.ppermute(ring, MESH_AXIS, perm)
                return ring[None], dspark.ring_positions(c, meta)[None]

            self._ring_op = register_op(
                f"pallas::k3_decode_import_ring_r{num_raw}",
                self._mesh,
                body,
                (P(MESH_AXIS),) * num_raw + (P(), P()),
                (P(MESH_AXIS), P()),
                export_key=repr(("import_ring", kda_raw, perm, dataclasses.astuple(c))),
            )
        return self._ring_op(
            *self.raw_pools,
            kda_blocks.to(torch.int32).contiguous(),
            torch.tensor([length], dtype=torch.int32).to("tpu"),
        )
