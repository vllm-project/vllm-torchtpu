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
"""Pure-PyTorch sharding and staging helpers for installing converted RL weights.

This module is pure PyTorch (zero JAX imports) and owns manifest sharding
extraction, contiguous rank slicing, exact parameter key/shape/dtype
validation, and canonical tied-key consistency checks so ``TPUWorker`` stays
thin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.rl.weight_conversion import validate_manifest

__all__ = [
    "ShardingCoords",
    "extract_sharding_map",
    "slice_tensor_for_shard",
    "stage_converted_weights",
]


@dataclass(frozen=True)
class ShardingCoords:
    """Local ``(rank, world_size)`` coordinates for ``tp``, ``ep``, ``moe_tp``."""

    tp_rank: int = 0
    tp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1
    moe_tp_rank: int = 0
    moe_tp_size: int = 1

    def as_axis_map(self) -> dict[str, tuple[int, int]]:
        return {
            "tp": (self.tp_rank, self.tp_size),
            "ep": (self.ep_rank, self.ep_size),
            "moe_tp": (self.moe_tp_rank, self.moe_tp_size),
        }


def extract_sharding_map(
    manifest: Mapping[str, Any],
) -> dict[str, list[str | None] | None]:
    """Extract ``target_key -> sharding`` from a validated manifest dict.

    Propagates sharding specs from primary keys to tied alias keys declared in
    ``manifest["tied_keys"]`` when the alias has no explicit rule of its own.
    """
    sharding_by_key: dict[str, list[str | None] | None] = {}
    for rule in manifest["rules"]:
        rule_sharding = rule.get("sharding")
        for target_key in rule["target_keys"]:
            sharding_by_key[target_key] = rule_sharding

    for pair in manifest.get("tied_keys") or []:
        src_key, dst_key = pair[0], pair[1]
        if dst_key not in sharding_by_key and src_key in sharding_by_key:
            sharding_by_key[dst_key] = sharding_by_key[src_key]

    return sharding_by_key


def slice_tensor_for_shard(
    name: str,
    tensor: torch.Tensor,
    sharding: list[str | None] | None,
    coords: ShardingCoords | Mapping[str, tuple[int, int]],
) -> torch.Tensor:
    """Slice a global converted tensor contiguously down to a worker's shard."""
    axis_map = coords.as_axis_map() if isinstance(coords, ShardingCoords) else coords
    effective_sharding = [None] * tensor.ndim if sharding is None else list(sharding)
    if len(effective_sharding) != tensor.ndim:
        raise ValueError(
            f"Sharding rank mismatch for parameter {name!r}: "
            f"sharding={effective_sharding!r} vs tensor ndim={tensor.ndim}"
        )

    result = tensor
    for dim, axis_name in enumerate(effective_sharding):
        if axis_name is None:
            continue
        if axis_name not in axis_map:
            raise ValueError(
                f"Unsupported sharding axis {axis_name!r} for parameter {name!r}"
            )
        rank, world_size = axis_map[axis_name]
        if world_size < 1 or not (0 <= rank < world_size):
            raise ValueError(
                f"Invalid {axis_name} sharding coordinates for parameter "
                f"{name!r}: rank={rank}, world_size={world_size}"
            )
        if world_size == 1:
            continue

        dim_size = result.shape[dim]
        if dim_size % world_size != 0:
            raise ValueError(
                f"Parameter {name!r} dim {dim} size {dim_size} is not "
                f"divisible by {axis_name} world_size={world_size}"
            )
        shard_size = dim_size // world_size
        result = result.narrow(dim, rank * shard_size, shard_size)

    return result


def _validate_shard_matches_param(
    key: str,
    param: torch.nn.Parameter,
    shard: torch.Tensor,
    global_tensor: torch.Tensor,
    sharding: list[str | None] | None,
) -> None:
    if tuple(shard.shape) != tuple(param.shape):
        raise ValueError(
            f"Shape mismatch for parameter {key!r}: "
            f"expected {tuple(param.shape)}, got {tuple(shard.shape)} "
            f"(global {tuple(global_tensor.shape)}, sharding={sharding!r})"
        )
    if shard.dtype != param.dtype:
        raise ValueError(
            f"Dtype mismatch for parameter {key!r}: "
            f"expected {param.dtype}, got {shard.dtype}"
        )


def stage_converted_weights(
    model: torch.nn.Module,
    converted_weights: Mapping[str, torch.Tensor],
    manifest_json: str,
    coords: ShardingCoords | Mapping[str, tuple[int, int]],
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Validate and slice all converted tensors before any in-place mutation.

    Iterates in canonical ``model.named_parameters(remove_duplicate=True)``
    order so every unique parameter is staged deterministically from its
    primary key. Any additional tied-parameter alias present in
    ``converted_weights`` is sliced, validated for shape/dtype, and verified to
    equal the primary parameter's shard (raising ``ValueError`` on mismatch).
    """
    manifest = validate_manifest(manifest_json)
    sharding_by_key = extract_sharding_map(manifest)

    unique_params = dict(model.named_parameters(remove_duplicate=True))
    all_params = dict(model.named_parameters(remove_duplicate=False))

    provided_keys = set(converted_weights.keys())
    missing_keys = sorted(set(unique_params.keys()) - provided_keys)
    unexpected_keys = sorted(provided_keys - set(all_params.keys()))
    if missing_keys or unexpected_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing={missing_keys}")
        if unexpected_keys:
            details.append(f"unexpected={unexpected_keys}")
        raise ValueError(
            "Converted weights keys do not match live model parameters: "
            + ", ".join(details)
        )

    def _slice_and_validate(key: str, param: torch.nn.Parameter) -> torch.Tensor:
        if key not in sharding_by_key:
            raise ValueError(
                f"Parameter {key!r} is missing a rule/sharding entry in manifest"
            )
        tensor = converted_weights[key]
        shard = slice_tensor_for_shard(key, tensor, sharding_by_key[key], coords)
        _validate_shard_matches_param(key, param, shard, tensor, sharding_by_key[key])
        return shard

    staged: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    primary_by_param_id: dict[int, tuple[str, torch.Tensor]] = {}

    for key, param in unique_params.items():
        shard = _slice_and_validate(key, param)
        primary_by_param_id[id(param)] = (key, shard)
        staged.append((param, shard))

    for alias_key in sorted(provided_keys - set(unique_params.keys())):
        param = all_params[alias_key]
        alias_shard = _slice_and_validate(alias_key, param)
        primary_key, primary_shard = primary_by_param_id[id(param)]
        if not torch.equal(alias_shard, primary_shard):
            raise ValueError(
                f"Tied parameter alias {alias_key!r} does not match primary "
                f"parameter {primary_key!r}"
            )

    return staged
