"""V2 binding of the shared DeepSeek V4.1 Engram."""

from typing import override

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
from .cache import write_cache

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


class NgramHashState(common.NgramHashState):
    @override
    def write_cache(self, cache, slots, values, valid):
        write_cache(cache, slots, values, valid)
