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
"""FX graph code normalization and fingerprinting for TPU peer-artifact replay."""

import hashlib
from typing import Any

import torch
import torch.distributed as dist
import torch.fx as fx
from torch_tpu._internal.distributed import tpu_distributed

from vllm_torchtpu.distributed.utils import (
    _canonicalize_process_group_name,
    _resolve_group_metadata,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Parameter tensors up to this size are hashed by value; bigger Parameters
# (model weights) are hashed by dtype and shape, while non-Parameter constant
# tensors are always hashed in full.
_FINGERPRINT_MAX_NUMEL = 1 << 16

_GROUP_ARG_NAMES = frozenset({"group_name", "group"})
_GROUP_SIZE_ARG_NAMES = frozenset({"group_size", "world_size"})

_SYMMETRIC_COLLECTIVE_SCHEMAS = frozenset(
    {
        "_c10d_functional::all_reduce",
        "_c10d_functional::all_reduce_",
        "_c10d_functional::all_reduce_coalesced",
        "_c10d_functional::all_reduce_coalesced_",
        "_c10d_functional::all_gather_into_tensor",
        "_c10d_functional::all_gather_into_tensor_out",
        "_c10d_functional::all_gather_into_tensor_coalesced",
        "_c10d_functional::reduce_scatter_tensor",
        "_c10d_functional::reduce_scatter_tensor_out",
        "_c10d_functional::reduce_scatter_tensor_coalesced",
        "_c10d_functional::all_to_all_single",
        "_c10d_functional_autograd::all_gather_into_tensor",
        "_c10d_functional_autograd::reduce_scatter_tensor",
        "_c10d_functional_autograd::all_to_all_single",
        "tpu::_all_gather_dim",
        "tpu::_reduce_scatter_dim",
        "vllm::all_reduce",
        "vllm::all_gather",
        "vllm::reduce_scatter",
        "vllm::patched_fused_scaled_matmul_reduce_scatter",
    }
)


def _supports_global_subgroup_tiling() -> bool:
    """True if TorchTPU tiles subgroup collectives across all world devices.

    TorchTPU >= 0.2.0 (commit 2e0c3677a) exposes `global_device_attributes` on
    `tpu_distributed` and tiles every subgroup collective's `replica_groups`
    across all `world_size` partitions via `GatherAllSubgroups`. Older releases
    (e.g. TorchTPU 0.1.1 before `3b08b43fd`) bind pure subgroup graphs to the
    subgroup's local device IDs unless a world-sized op is present in the graph.
    """
    return hasattr(tpu_distributed, "global_device_attributes")


def _distributed_world_size() -> int | None:
    if dist.is_initialized():
        return int(dist.get_world_size())
    return None


def _distributed_rank() -> int | None:
    if dist.is_initialized():
        return int(dist.get_rank())
    return None


def _node_schema(target: Any) -> Any | None:
    if isinstance(target, torch._ops.OpOverload):
        return target._schema
    if (
        isinstance(target, torch._ops.OpOverloadPacket)
        and "default" in target.overloads()
    ):
        return target.default._schema
    return None


def _device_assignment_size(value: Any) -> int:
    if not isinstance(value, (list, tuple)):
        return 0
    total = 0
    for item in value:
        if isinstance(item, (list, tuple)):
            total += len(item)
        elif isinstance(item, int):
            total += 1
    return total


def _iter_node_group_names(node: fx.Node, schema: Any | None) -> list[str]:
    names: list[str] = []
    if schema is not None:
        for idx, arg_spec in enumerate(schema.arguments):
            if arg_spec.name not in _GROUP_ARG_NAMES:
                continue
            if idx < len(node.args):
                arg = node.args[idx]
                if isinstance(arg, str):
                    names.append(arg)
    for key in _GROUP_ARG_NAMES:
        val = node.kwargs.get(key)
        if isinstance(val, str):
            names.append(val)
    return names


def _graph_has_world_sized_op(module: fx.GraphModule, world_size: int) -> bool:
    """True if `module` contains an op spanning all `world_size` devices."""
    for submod in module.modules():
        if not isinstance(submod, fx.GraphModule):
            continue
        for node in submod.graph.nodes:
            if node.op != "call_function":
                continue
            schema = _node_schema(node.target)
            for group_name in _iter_node_group_names(node, schema):
                meta = _resolve_group_metadata(group_name)
                if meta.size is not None and meta.size >= world_size:
                    return True
            if schema is not None:
                for idx, arg_spec in enumerate(schema.arguments):
                    if idx >= len(node.args):
                        continue
                    arg_val = node.args[idx]
                    if (
                        arg_spec.name in _GROUP_SIZE_ARG_NAMES
                        and isinstance(arg_val, int)
                        and arg_val >= world_size
                    ):
                        return True
                    if (
                        arg_spec.name == "device_assignment"
                        and _device_assignment_size(arg_val) >= world_size
                    ):
                        return True
            for key in _GROUP_SIZE_ARG_NAMES:
                val = node.kwargs.get(key)
                if isinstance(val, int) and val >= world_size:
                    return True
            if (
                _device_assignment_size(node.kwargs.get("device_assignment"))
                >= world_size
            ):
                return True
    return False


def _uses_global_subgroup_tiling(module: fx.GraphModule) -> bool:
    world_size = _distributed_world_size()
    if world_size is None or world_size <= 1:
        return True
    if _supports_global_subgroup_tiling():
        return True
    return _graph_has_world_sized_op(module, world_size)


def _has_subgroup_follower_role(graph: fx.GraphModule) -> bool:
    """True when `graph` is subgroup-bound and this rank is not its subgroup leader."""
    if _uses_global_subgroup_tiling(graph):
        return False
    current_rank = _distributed_rank()
    if current_rank is None:
        return False
    for submod in graph.modules():
        if not isinstance(submod, fx.GraphModule):
            continue
        for node in submod.graph.nodes:
            if node.op != "call_function":
                continue
            schema = _node_schema(node.target)
            for group_name in _iter_node_group_names(node, schema):
                meta = _resolve_group_metadata(group_name)
                if (
                    meta.ranks
                    and len(meta.ranks) > 1
                    and current_rank in meta.ranks
                    and current_rank != meta.ranks[0]
                ):
                    return True
    return False


def _normalized_module_code(
    module: fx.GraphModule,
    canonical_by_raw: dict[str, str] | None = None,
    *,
    global_tiling: bool | None = None,
) -> str:
    """Return `module.code` with symmetric collective `group_name`s canonicalized.

    Only symmetric subgroup collectives in `_SYMMETRIC_COLLECTIVE_SCHEMAS` are
    canonicalized; rooted, point-to-point, or unlisted ops with a
    `group_name`/`group` string argument raise `ValueError` so
    `_graph_fingerprint` returns `None`.
    """
    if global_tiling is None:
        global_tiling = _uses_global_subgroup_tiling(module)
    saved: list[tuple[fx.Node, tuple[Any, ...], dict[str, Any]]] = []
    if canonical_by_raw is None:
        canonical_by_raw = {}

    def _canonical(raw: str) -> str:
        token = canonical_by_raw.get(raw)
        if token is None:
            token = _canonicalize_process_group_name(
                raw, len(canonical_by_raw), global_tiling=global_tiling
            )
            canonical_by_raw[raw] = token
        return token

    try:
        for node in module.graph.nodes:
            if node.op != "call_function":
                continue
            schema = _node_schema(node.target)
            if schema is None:
                continue
            new_args: list[Any] | None = None
            new_kwargs: dict[str, Any] | None = None

            for idx, arg_spec in enumerate(schema.arguments):
                if arg_spec.name not in _GROUP_ARG_NAMES:
                    continue
                has_pos_str = idx < len(node.args) and isinstance(node.args[idx], str)
                val = node.kwargs.get(arg_spec.name)
                has_kw_str = isinstance(val, str)
                if not has_pos_str and not has_kw_str:
                    continue
                if schema.name not in _SYMMETRIC_COLLECTIVE_SCHEMAS:
                    raise ValueError(
                        f"Cannot canonicalize non-symmetric collective {schema.name!r}"
                    )
                if has_pos_str:
                    if new_args is None:
                        new_args = list(node.args)
                    new_args[idx] = _canonical(node.args[idx])
                if has_kw_str:
                    if new_kwargs is None:
                        new_kwargs = dict(node.kwargs)
                    new_kwargs[arg_spec.name] = _canonical(val)

            if new_args is not None or new_kwargs is not None:
                saved.append((node, node.args, node.kwargs))
                if new_args is not None:
                    node.args = tuple(new_args)
                if new_kwargs is not None:
                    node.kwargs = new_kwargs

        if not saved:
            return module.code
        return module.graph.python_code(root_module="self").src
    finally:
        for node, orig_args, orig_kwargs in saved:
            node.args = orig_args
            node.kwargs = orig_kwargs


def _graph_fingerprint(graph: fx.GraphModule) -> str | None:
    """Hash of the code of `graph` and its submodules and of their constants.

    Two ranks with the same fingerprint traced the same program, so either can
    replay the other's artifact. A rank-dependent value baked into the graph,
    e.g. a Python int from `rank_in_group`, changes the code and so the hash.
    Rank-local `c10d` `group_name` identifiers on symmetric subgroup collectives
    are canonicalized via `_normalized_module_code`. None if the graph cannot
    be hashed (including when a group cannot be resolved or a non-symmetric
    collective is present); its artifact is then only replayed by the rank that
    saved it.
    """
    try:
        h = hashlib.sha256()
        global_tiling = _uses_global_subgroup_tiling(graph)
        canonical_by_raw: dict[str, str] = {}
        for name, module in graph.named_modules():
            if not isinstance(module, fx.GraphModule):
                continue
            norm_code = _normalized_module_code(
                module, canonical_by_raw, global_tiling=global_tiling
            )
            h.update(f"{name}\n{norm_code}".encode())
            for node in module.graph.nodes:
                if node.op != "get_attr":
                    continue
                value = module
                for atom in str(node.target).split("."):
                    value = getattr(value, atom)
                h.update(str(node.target).encode())
                if not isinstance(value, torch.Tensor):
                    h.update(type(value).__qualname__.encode())
                    continue
                h.update(f"{value.dtype}{tuple(value.shape)}".encode())
                if (
                    not isinstance(value, torch.nn.Parameter)
                    or value.numel() <= _FINGERPRINT_MAX_NUMEL
                ):
                    flat = value.detach().to("cpu").contiguous().reshape(-1)
                    h.update(flat.view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    except Exception as e:
        logger.warning("[TpuCompilerAdaptor] Could not fingerprint graph: %s", e)
        return None
