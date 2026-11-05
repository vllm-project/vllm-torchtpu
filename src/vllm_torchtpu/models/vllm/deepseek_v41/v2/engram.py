"""V2 binding of the shared DeepSeek V4.1 Engram."""

from typing import NamedTuple, override

import torch

from ..common import engram as common
from ..common.engram import (
    _INVALID_SCALE,
    _UNIT_SCALE,
    Engram,
    ParallelEngramEmbedding,
    _gather_engram_rows,
    _load_local_rows,
    _unpack_bytes,
    dequant_fp8_rows,
    register_row_slice,
    upstream,
)
from .attention import _cache_slots
from .cache import write_cache, write_cache_slots

__all__ = [
    "Engram",
    "NgramHashState",
    "ParallelEngramEmbedding",
    "_INVALID_SCALE",
    "_UNIT_SCALE",
    "_gather_engram_rows",
    "_load_local_rows",
    "_unpack_bytes",
    "dequant_fp8_rows",
    "register_row_slice",
    "upstream",
]


class _SlotMetadata(NamedTuple):
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor


def _slots_int32(kv_cache, block_tables, seq_lens, query_start_loc, positions):
    metadata = _SlotMetadata(block_tables, seq_lens, query_start_loc)
    return _cache_slots(kv_cache, metadata, positions, wide=False)


def _mapped_token_ids(token_map, input_ids, dead_mask, dead_id: int):
    return token_map[input_ids].masked_fill(dead_mask, dead_id)


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_slots_int32_compiled = _tpu_compile(_slots_int32)
_mapped_token_ids_compiled = _tpu_compile(_mapped_token_ids)


def engram_cache_slots(kv_cache, metadata, positions):
    """`_cache_slots` of the Engram slot cache in int32, as one program on TPU.

    Eagerly the int64 slot arithmetic is a program per op, each emulated on
    32-bit halves. The slot cache holds one int32 per SWA KV row, so its slots
    fit in int32.
    """
    fn = _slots_int32_compiled if positions.device.type == "tpu" else _slots_int32
    return fn(
        kv_cache,
        metadata.block_tables,
        metadata.seq_lens,
        metadata.query_start_loc,
        positions,
    )


class NgramHashState(common.NgramHashState):
    @override
    def write_cache(self, cache, slots, values, valid):
        write_cache(cache, slots, values, valid)

    @override
    def write_token_ids(self, cache, slots, input_ids, dead_mask):
        assert cache.numel() < 2**31
        mapped = _mapped_token_ids_compiled(
            self.token_map, input_ids, dead_mask, upstream.DEAD_ID
        )
        write_cache_slots(cache, slots, mapped)
