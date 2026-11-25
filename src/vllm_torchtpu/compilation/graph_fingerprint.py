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
import os
from typing import Any

import torch
import torch.fx as fx

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.utils import _canonicalize_process_group_name
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# get_attr tensors up to this size are hashed by value; bigger ones, which are
# weights rather than constants, by dtype and shape.
_FINGERPRINT_MAX_NUMEL = 1 << 16

_GROUP_ARG_NAMES = frozenset({"group_name", "group"})

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


def _node_schema(target: Any) -> Any | None:
    if isinstance(target, torch._ops.OpOverload):
        return target._schema
    if (
        isinstance(target, torch._ops.OpOverloadPacket)
        and "default" in target.overloads()
    ):
        return target.default._schema
    return None


def _normalized_module_code(
    module: fx.GraphModule,
    canonical_by_raw: dict[str, str] | None = None,
) -> str:
    """Return `module.code` with symmetric collective `group_name`s canonicalized.

    Only symmetric subgroup collectives in `_SYMMETRIC_COLLECTIVE_SCHEMAS` are
    canonicalized; rooted, point-to-point, or unlisted ops with a
    `group_name`/`group` string argument raise `ValueError` so
    `_graph_fingerprint` returns `None`.
    """
    saved: list[tuple[fx.Node, tuple[Any, ...], dict[str, Any]]] = []
    if canonical_by_raw is None:
        canonical_by_raw = {}

    def _canonical(raw: str) -> str:
        token = canonical_by_raw.get(raw)
        if token is None:
            token = _canonicalize_process_group_name(raw, len(canonical_by_raw))
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
    are canonicalized via `_normalized_module_code` because TorchTPU lowers
    symmetric subgroup collectives into global SPMD HLO covering all partitions.
    None if the graph cannot be hashed (including when a group cannot be
    resolved or a non-symmetric collective is present); its artifact is then
    only replayed by the rank that saved it.
    """
    try:
        h = hashlib.sha256()
        canonical_by_raw: dict[str, str] = {}
        # VLLM_TPU_FINGERPRINT_DUMP=<dir>: selected local ranks (default 0,1 via
        # VLLM_TPU_FINGERPRINT_DUMP_RANKS) write what went into the hash (each
        # module's code and a digest per constant), so two ranks' dumps of one
        # range diff to what makes them differ.
        dump_dir = os.path.expandvars(envs.VLLM_TPU_FINGERPRINT_DUMP)
        dump_ranks = envs.VLLM_TPU_FINGERPRINT_DUMP_RANKS.split(",")
        if dump_dir and os.environ.get("LOCAL_RANK", "0") not in dump_ranks:
            dump_dir = ""
        dump: list[str] = []
        for name, module in graph.named_modules():
            if not isinstance(module, fx.GraphModule):
                continue
            code = _normalized_module_code(module, canonical_by_raw)
            h.update(f"{name}\n{code}".encode())
            if dump_dir:
                dump.append(f"### module {name!r}\n{code}")
            for node in module.graph.nodes:
                if node.op != "get_attr":
                    continue
                value = module
                for atom in str(node.target).split("."):
                    value = getattr(value, atom)
                h.update(str(node.target).encode())
                if not isinstance(value, torch.Tensor):
                    h.update(type(value).__qualname__.encode())
                    if dump_dir:
                        dump.append(
                            f"### const {node.target} {type(value).__qualname__}"
                        )
                    continue
                h.update(f"{value.dtype}{tuple(value.shape)}".encode())
                digest = "large"
                if value.numel() <= _FINGERPRINT_MAX_NUMEL:
                    flat = value.detach().to("cpu").contiguous().reshape(-1)
                    raw = flat.view(torch.uint8).numpy().tobytes()
                    h.update(raw)
                    if dump_dir:
                        digest = hashlib.sha256(raw).hexdigest()[:16]
                        if flat.numel() <= 64:
                            digest += f" values={flat.tolist()}"
                if dump_dir:
                    shape_s = f"{value.dtype}{tuple(value.shape)}"
                    dump.append(f"### const {node.target} {shape_s} {digest}")
        fingerprint = h.hexdigest()
        if dump_dir:
            try:
                os.makedirs(dump_dir, exist_ok=True)
                path = os.path.join(
                    dump_dir,
                    f"rank{os.environ.get('LOCAL_RANK', '0')}-{fingerprint[:16]}.txt",
                )
                if not os.path.exists(path):
                    with open(path, "w") as f:
                        f.write("\n".join(dump))
            except Exception as e:  # noqa: BLE001 - diagnostics only
                logger.warning("[TpuCompilerAdaptor] fingerprint dump failed: %s", e)
        return fingerprint
    except Exception as e:
        logger.warning("[TpuCompilerAdaptor] Could not fingerprint graph: %s", e)
        return None
