from __future__ import annotations

from enum import Enum

TAG_FA = "fa"
TAG_GDN_CONV = "gdn.conv"
TAG_GDN_SSM = "gdn.ssm"


class LayerType(str, Enum):
    FULL_ATTN = "full_attn"
    LINEAR_ATTN = "linear_attn"
    MAMBA_STATE = "mamba_state"


class TensorLayout(str, Enum):
    TOKEN_FIRST = "token_first"
    HEAD_FIRST = "head_first"
    BLOCKS_FIRST = "blocks_first"


def check_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def check_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def check_power_of_two(name: str, value: int) -> None:
    if not is_power_of_two(value):
        raise ValueError(f"{name} tp_size must be a power of two, got {value}")


def linear_rank(pcp_rank: int, tp_rank: int, tp_size: int) -> int:
    return pcp_rank * tp_size + tp_rank


def head_range(total_heads: int, tp_size: int, tp_rank: int) -> range:
    check_positive("total_heads", total_heads)
    check_positive("tp_size", tp_size)
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(
            f"tp_rank={tp_rank} is out of range for tp_size={tp_size}")

    if total_heads < tp_size:
        if tp_size % total_heads != 0:
            raise ValueError(
                f"tp_size={tp_size} is not divisible by total_heads={total_heads}"
            )
        tp_ranks_per_head = tp_size // total_heads
        head = tp_rank // tp_ranks_per_head
        return range(head, head + 1)

    if total_heads % tp_size != 0:
        raise ValueError(
            f"total_heads={total_heads} is not divisible by tp_size={tp_size}")
    heads_per_rank = total_heads // tp_size
    start = tp_rank * heads_per_rank
    return range(start, start + heads_per_rank)


def is_linear_state_layer(layer_type: LayerType) -> bool:
    return layer_type in (LayerType.LINEAR_ATTN, LayerType.MAMBA_STATE)
