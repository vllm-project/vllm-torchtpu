# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's unified pools as the K3 programs see them (P/D over the Raiden KV
connector).

Each ``KVCacheTensor`` is one raw pool of ``layout.cache_layout()`` blocks,
shared by one MLA layer and three KDA layers: MLA latent pages and, in a KDA
group's state block, the layer's state (rows 0-76 recurrent, 77-80 conv taps)
and the draft-ring rows (81-95), heads in vLLM TP-rank order. The prefill
program writes them; the decoder reads the latent pages in place and imports a
request's KDA state (into its private slot) and draft ring (:class:`KdaImporter`).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.kimi_k3.layout import (
    CONV_HEAD_ROWS,
    CONV_TAPS,
    HEAD_DIM,
    KDA_HEADS,
    LAYOUT,
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
        if layer_index(name) < DRAFT_LAYER0  # draft layers: draft_raw_pools
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
    its private slot (the decoder's first step of a transferred request)."""

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
