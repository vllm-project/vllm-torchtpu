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
"""Rank replication factors when a sharded dimension has fewer groups than ranks.

With `n` shard groups and `R > n` ranks, `R // n` consecutive ranks hold the
same shard and their partial sums repeat that many times in an all-reduce.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp

# Minimum per-rank column width for shared-expert matmul tiles.
SHARED_EXPERT_TILE = 128


def shard_count(R: int, groups: int | None) -> int:
    """Number of distinct shards across `R` ranks for a dimension of `groups`."""
    if isinstance(R, bool) or not isinstance(R, int) or R <= 0:
        raise ValueError(f"R must be a positive int, got {R!r}")
    if groups is None:
        return R
    if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
        raise ValueError(f"groups must be a positive int or None, got {groups!r}")
    if groups > R and groups % R != 0:
        raise ValueError(f"shard groups {groups} > R={R} must be a multiple of R")
    num_shards = min(R, groups)
    if R % num_shards != 0:
        raise ValueError(
            f"Rank count R={R} must be divisible by shard count {num_shards}"
        )
    return num_shards


def split_sharded_width(
    R: int,
    splits: tuple[int, ...],
    split_groups: tuple[int | None, ...] | None,
) -> int:
    """Per-rank width of `splits` after sharding each part by `split_groups`."""
    groups = split_groups or (None,) * len(splits)
    total = 0
    for split_width, num_groups in zip(splits, groups, strict=True):
        num_shards = shard_count(R, num_groups)
        if split_width % num_shards != 0:
            raise ValueError(
                f"split width {split_width} must be divisible by "
                f"shard count {num_shards}"
            )
        total += split_width // num_shards
    return total


def unsharded_target_ranges(
    decl: Any,
    base_off: int,
    full_w: int,
    sharded_w: int,
    split_axis: int,
    R: int,
    tp_rank: int,
) -> list[tuple[int, int]]:
    """Return `(start, stop)` slice ranges on an unsharded axis for `tp_rank`."""
    transform = decl.transform
    dim_axis = split_axis % len(decl.shape)
    if (
        transform.channel_splits is not None
        and dim_axis == len(decl.shape) - 1
        and full_w != sharded_w
    ):
        groups = transform.split_groups or (None,) * len(transform.channel_splits)
        out: list[tuple[int, int]] = []
        split_offset = base_off
        for split_width, num_groups in zip(
            transform.channel_splits, groups, strict=True
        ):
            num_shards = shard_count(R, num_groups)
            shard_width = split_width // num_shards
            start = split_offset + (tp_rank * num_shards // R) * shard_width
            out.append((start, start + shard_width))
            split_offset += split_width
        return out
    if transform.shard_axis is not None and full_w != sharded_w:
        num_shards = shard_count(R, transform.shard_groups)
        start = base_off + (tp_rank * num_shards // R) * sharded_w
        return [(start, start + sharded_w)]
    return [(base_off, base_off + full_w)]


def slice_rank_ranges(
    ops: Any,
    buf: Any,
    axis: int,
    ranges: Sequence[tuple[int, int]],
    passthrough_w: int,
) -> Any:
    """Slice or concatenate `(start, stop)` ranges from `buf` along `axis`."""
    if len(ranges) == 1:
        start, stop = ranges[0]
        if start == 0 and stop == passthrough_w:
            return buf
        return ops.slice_axis(buf, axis, start, stop)
    return ops.concat(
        [ops.slice_axis(buf, axis, start, stop) for start, stop in ranges], axis
    )


def derive_kh_rep(R: int, KH: int) -> int:
    """Ranks sharing one key head when there are fewer key heads than ranks."""
    return max(1, R // KH) if KH > 0 else 1


def derive_sh_rep(R: int, IS: int, tile: int = SHARED_EXPERT_TILE) -> int:
    """Ranks sharing one shared-expert column slice at the `tile` lane floor."""
    return max(1, R // max(1, IS // tile))


def replicate_groups(x: Any, group_size: int) -> Any:
    """Copy rank `r - r % group_size` into ranks `r .. r + group_size - 1` along
    axis 0.
    """
    return x if group_size <= 1 else jnp.repeat(x[::group_size], group_size, axis=0)


__all__ = [
    "SHARED_EXPERT_TILE",
    "derive_kh_rep",
    "derive_sh_rep",
    "replicate_groups",
    "shard_count",
    "slice_rank_ranges",
    "split_sharded_width",
    "unsharded_target_ranges",
]
