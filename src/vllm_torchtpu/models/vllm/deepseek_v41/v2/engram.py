"""V2 binding of the shared DeepSeek V4.1 Engram."""

from typing import override

from ..common import engram as common
from ..common.engram import Engram
from .cache import write_cache

__all__ = ["Engram", "NgramHashState"]


class NgramHashState(common.NgramHashState):
    @override
    def write_cache(self, cache, slots, values, valid):
        write_cache(cache, slots, values, valid)
