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
"""Hybrid manifest generator for the Distributed RL Weight Conversion Protocol.

Introspects an instantiated post-``process_weights_after_loading()`` vLLM /
TorchTPU ``nn.Module`` to emit a validated ``weight_mapping`` manifest covering
unquantized (``bfloat16``, ``float16``, ``float32``) TP / EP / DP deployments.

Pure PyTorch only (zero JAX imports).

Fail-closed guardrails raise ``NotImplementedError`` on:
- Quantized weights, quantization scales/zero-points, or non-unquantized quant
  configs/methods (FP8, MXFP4, NVFP4, AWQ, GPTQ, INT8, etc.).
- Unsupported parallelism: ``PP > 1``, ``PCP > 1``, ``DCP > 1``, or EPLB
  redundant experts.
- LoRA adapters (config-level or LoRA-wrapped modules/parameters).
- Active vision (``visual.*``) or MTP (``mtp.*``) parameters / enabled MTP
  draft configs.
- Unsupported activations (``swigluoai``) or expert biases on ``RoutedExperts``.
- Unrecognized ``nn.Module`` subclasses or unsupported layouts.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

import torch

from vllm_torchtpu.rl.weight_conversion import SCHEMA_VERSION, validate_manifest

# Mirror vllm_torchtpu.layers.adapter.linear_common.WEIGHT_FLIPPED_ATTR without
# importing that module (which imports JAX).
WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"

_ALLOWED_UNQUANTIZED_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})

_ALLOWED_UNQUANTIZED_METHODS = frozenset(
    {
        "UnquantizedLinearMethod",
        "VllmUnquantizedLinearMethod",
        "UnquantizedEmbeddingMethod",
        "UnquantizedFusedMoEMethod",
        "VllmUnquantizedFusedMoEMethod",
    }
)

_ALLOWED_UNQUANTIZED_CONFIG_NAMES = frozenset({"", "unquantized", "tpu-unquantized"})

_QUANT_PARAM_TOKENS = frozenset(
    {
        "weight_scale",
        "weight_scale_inv",
        "input_scale",
        "qweight",
        "qzeros",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_input_scale",
        "w2_input_scale",
        "scales",
    }
)

_DISALLOWED_PREFIXES = ("visual.", "model.visual.", "mtp.", "model.mtp.")

_DEFAULT_PACKED_MODULES_MAPPING: dict[str, list[str]] = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


def _mro_names(obj: Any) -> set[str]:
    return {cls.__name__ for cls in type(obj).__mro__}


def _has_class_in_mro(obj: Any, names: str | tuple[str, ...]) -> bool:
    """Match module types by MRO class name rather than ``isinstance``.

    Importing concrete TorchTPU adapter classes (e.g. ``TpuRowParallelLinear``,
    ``GdnColumnParallelLinear``, ``VllmUnquantizedFusedMoEMethod``) transitively
    imports JAX / TPU runtime modules. Matching MRO names preserves the strict
    zero-JAX-import invariant in ``vllm_torchtpu.rl`` and allows CPU-only tests
    with lightweight mock layers.
    """
    if isinstance(names, str):
        names = (names,)
    mro = _mro_names(obj)
    return any(name in mro for name in names)


def _exact_div(numerator: int, denominator: int, context: str) -> int:
    if denominator <= 0 or numerator % denominator != 0:
        raise ValueError(
            f"{context}: expected {numerator} to be evenly divisible by {denominator}"
        )
    return numerator // denominator


def _check_no_gqa_head_padding(
    parent_mod: torch.nn.Module | None, param_name: str
) -> None:
    if parent_mod is None:
        return
    orig_heads = getattr(parent_mod, "orig_total_num_heads", None)
    total_heads_parent = getattr(parent_mod, "total_num_heads", None)
    if (
        isinstance(orig_heads, int)
        and isinstance(total_heads_parent, int)
        and orig_heads != total_heads_parent
    ):
        raise NotImplementedError(
            f"GQA query-head zero-padding ({orig_heads} -> {total_heads_parent}) "
            f"on {param_name!r} is not supported by the RL weight conversion "
            "manifest generator."
        )


def _tp_interleave_ops(
    n_segs: int,
    layer_tp: int,
    seg_local: int,
    tail: list[int],
) -> list[dict[str, dict[str, Any]]]:
    """Build reshape->transpose->reshape ops for homogeneous TP interleaving."""
    total_out = n_segs * layer_tp * seg_local
    tail_axes = list(range(3, 3 + len(tail)))
    return [
        {
            "reshape": {
                "src_shape": [total_out, *tail],
                "dst_shape": [n_segs, layer_tp, seg_local, *tail],
            }
        },
        {"transpose": {"axes": [1, 0, 2, *tail_axes]}},
        {
            "reshape": {
                "src_shape": [layer_tp, n_segs, seg_local, *tail],
                "dst_shape": [total_out, *tail],
            }
        },
    ]


def _emit_tp_sharded_segments(
    param_name: str,
    per_seg_inputs: list[str],
    seg_sizes: list[int],
    replication_factors: list[int],
    layer_tp: int,
    tail: list[int],
    *,
    flipped: bool,
    sharding: list[str | None],
) -> list[dict[str, Any]]:
    """Emit chained 12-op rules packing segments into contiguous TP-shard layout."""
    seg_keys = [f"__tmp__:{param_name}:seg{i}" for i in range(len(seg_sizes))]
    chained: list[dict[str, Any]] = []
    local_sizes: list[int] = []

    for idx, (inp_k, seg_k, sz, rep) in enumerate(
        zip(per_seg_inputs, seg_keys, seg_sizes, replication_factors)
    ):
        num_unique = _exact_div(
            layer_tp, rep, f"{param_name} segment[{idx}] replication"
        )
        local_sz = _exact_div(sz, num_unique, f"{param_name} segment[{idx}] size")
        local_sizes.append(local_sz)
        seg_ops: list[dict[str, dict[str, Any]]] = [
            {
                "reshape": {
                    "src_shape": [sz, *tail],
                    "dst_shape": [num_unique, local_sz, *tail],
                }
            }
        ]
        if rep > 1:
            seg_ops.append({"repeat": {"axis": 0, "times": rep}})
        chained.append(
            {
                "source_keys": [inp_k],
                "target_keys": [seg_k],
                "ops": seg_ops,
            }
        )

    local_out = sum(local_sizes)
    final_ops: list[dict[str, dict[str, Any]]] = [
        {"concat": {"axis": 1, "sizes": local_sizes}},
        {
            "reshape": {
                "src_shape": [layer_tp, local_out, *tail],
                "dst_shape": [layer_tp * local_out, *tail],
            }
        },
    ]
    if flipped:
        final_ops.append({"transpose": {"axes": [1, 0]}})
    chained.append(
        {
            "source_keys": seg_keys,
            "target_keys": [param_name],
            "ops": final_ops,
            "sharding": sharding,
        }
    )
    return chained


def _check_parallel_config(parallel_config: Any) -> None:
    if parallel_config is None:
        return
    pp = getattr(parallel_config, "pipeline_parallel_size", 1) or 1
    if pp > 1:
        raise NotImplementedError(
            f"Pipeline parallelism (PP={pp} > 1) is not supported by the RL "
            "weight conversion manifest generator."
        )
    pcp = getattr(parallel_config, "prefill_context_parallel_size", 1) or 1
    if pcp > 1:
        raise NotImplementedError(
            f"Prefill context parallelism (PCP={pcp} > 1) is not supported by "
            "the RL weight conversion manifest generator."
        )
    dcp = getattr(parallel_config, "decode_context_parallel_size", 1) or 1
    if dcp > 1:
        raise NotImplementedError(
            f"Decode context parallelism (DCP={dcp} > 1) is not supported by "
            "the RL weight conversion manifest generator."
        )
    if getattr(parallel_config, "enable_eplb", False):
        raise NotImplementedError(
            "EPLB (expert load balancing / redundant experts) is not supported "
            "by the RL weight conversion manifest generator."
        )
    eplb_cfg = getattr(parallel_config, "eplb_config", None)
    if (
        eplb_cfg is not None
        and (getattr(eplb_cfg, "num_redundant_experts", 0) or 0) > 0
    ):
        raise NotImplementedError(
            "Redundant experts (EPLB) are not supported by the RL weight "
            "conversion manifest generator."
        )


def _check_model_and_quant_config(
    model: torch.nn.Module,
    model_config: Any,
) -> None:
    if model_config is not None:
        quant = getattr(model_config, "quantization", None)
        if quant is not None:
            quant_str = str(quant).lower()
            if quant_str not in _ALLOWED_UNQUANTIZED_CONFIG_NAMES:
                raise NotImplementedError(
                    f"Quantization {quant!r} is not supported; only unquantized "
                    "(bfloat16/float16/float32) models are supported."
                )
        if getattr(model_config, "enable_lora", False) or (
            getattr(model_config, "lora_config", None) is not None
        ):
            raise NotImplementedError(
                "LoRA adapters are not supported by the RL weight conversion "
                "manifest generator."
            )
        if getattr(model_config, "enable_mtp", False):
            raise NotImplementedError(
                "MTP (multi-token prediction) is not supported by the RL weight "
                "conversion manifest generator."
            )

    quant_config = getattr(model, "quant_config", None)
    if quant_config is not None:
        qcfg_name = type(quant_config).__name__
        if qcfg_name != "VllmUnquantizedConfig":
            get_name = getattr(quant_config, "get_name", None)
            method_name = str(get_name()).lower() if callable(get_name) else qcfg_name
            if method_name not in _ALLOWED_UNQUANTIZED_CONFIG_NAMES:
                raise NotImplementedError(
                    f"Quantization config {qcfg_name!r} ({method_name!r}) is not "
                    "supported; only unquantized models are supported."
                )

    if getattr(model, "lora_config", None) is not None:
        raise NotImplementedError(
            "LoRA adapters are not supported by the RL weight conversion "
            "manifest generator."
        )


def _check_module_guardrails(mod_name: str, mod: torch.nn.Module) -> None:
    cls_name = type(mod).__name__
    if "lora" in cls_name.lower():
        raise NotImplementedError(
            f"LoRA module {mod_name!r} ({cls_name}) is not supported by the RL "
            "weight conversion manifest generator."
        )
    quant_method = getattr(mod, "quant_method", None)
    if quant_method is not None:
        qm_name = type(quant_method).__name__
        if qm_name not in _ALLOWED_UNQUANTIZED_METHODS:
            raise NotImplementedError(
                f"Quantized method {qm_name!r} on module {mod_name!r} is not "
                "supported; only unquantized methods are supported."
            )


def _check_param_guardrails(param_name: str, param: torch.nn.Parameter) -> None:
    for prefix in _DISALLOWED_PREFIXES:
        if param_name.startswith(prefix):
            raise NotImplementedError(
                f"Active vision/MTP parameter {param_name!r} is not supported "
                "by the RL weight conversion manifest generator."
            )
    leaf = param_name.rsplit(".", 1)[-1]
    if "lora" in param_name.lower():
        raise NotImplementedError(
            f"LoRA parameter {param_name!r} is not supported by the RL weight "
            "conversion manifest generator."
        )
    if leaf in _QUANT_PARAM_TOKENS:
        raise NotImplementedError(
            f"Quantization parameter {param_name!r} is not supported; only "
            "unquantized weights are supported."
        )
    if param.dtype not in _ALLOWED_UNQUANTIZED_DTYPES:
        raise NotImplementedError(
            f"Unsupported dtype {param.dtype} on parameter {param_name!r}; only "
            "unquantized bfloat16, float16, and float32 are supported."
        )


def _resolve_parallel_sizes(
    model: torch.nn.Module,
    parallel_config: Any,
    tp_size: int | None,
    dp_size: int | None,
    ep_size: int | None,
) -> tuple[int, int, int]:
    resolved_tp = tp_size
    if resolved_tp is None and parallel_config is not None:
        resolved_tp = getattr(parallel_config, "tensor_parallel_size", None)
    if resolved_tp is None:
        for mod in model.modules():
            mod_tp = getattr(mod, "tp_size", None)
            if isinstance(mod_tp, int) and mod_tp > 1:
                resolved_tp = mod_tp
                break
    if resolved_tp is None:
        resolved_tp = 1

    resolved_dp = dp_size
    if resolved_dp is None:
        env_dp = int(os.environ.get("TORCH_TPU_DP_SIZE", "0"))
        if env_dp > 0:
            resolved_dp = env_dp
        elif parallel_config is not None:
            resolved_dp = getattr(parallel_config, "data_parallel_size", None)
    if resolved_dp is None:
        resolved_dp = 1

    resolved_ep = ep_size
    if resolved_ep is None and parallel_config is not None:
        if getattr(parallel_config, "enable_expert_parallel", False):
            explicit_ep = getattr(parallel_config, "expert_parallel_size", None)
            resolved_ep = (
                explicit_ep
                if isinstance(explicit_ep, int) and explicit_ep > 0
                else resolved_tp * resolved_dp
            )
        else:
            resolved_ep = 1
    if resolved_ep is None:
        for mod in model.modules():
            if _has_class_in_mro(mod, "RoutedExperts"):
                moe_cfg = getattr(mod, "moe_config", None)
                if moe_cfg is not None:
                    moe_p = getattr(moe_cfg, "moe_parallel_config", None)
                    if moe_p is not None and getattr(moe_p, "use_ep", False):
                        resolved_ep = getattr(moe_cfg, "ep_size", 1) or 1
                        break
    if resolved_ep is None:
        resolved_ep = 1

    return int(resolved_tp), int(resolved_dp), int(resolved_ep)


def _collect_packed_mapping(model: torch.nn.Module) -> dict[str, list[str]]:
    mapping = dict(_DEFAULT_PACKED_MODULES_MAPPING)
    for owner in (model, getattr(model, "language_model", None)):
        if owner is None:
            continue
        packed = getattr(owner, "packed_modules_mapping", None)
        if isinstance(packed, Mapping):
            for k, v in packed.items():
                if isinstance(k, str) and isinstance(v, (list, tuple)):
                    mapping[k] = list(v)
    return mapping


def _to_hf_key(param_name: str, model: torch.nn.Module) -> str:
    """Map a vLLM parameter name to its canonical HuggingFace checkpoint key.

    Multimodal wrappers (e.g. ``Qwen3_5MoeForConditionalGeneration``) nest the
    text backbone under ``language_model.model.*`` and ``language_model.lm_head.*``,
    whereas HuggingFace checkpoints store them under ``model.language_model.*``
    and ``lm_head.*``.
    """
    if hasattr(model, "language_model"):
        if param_name.startswith("language_model.model."):
            return "model.language_model." + param_name[len("language_model.model.") :]
        if param_name.startswith("language_model.lm_head."):
            return "lm_head." + param_name[len("language_model.lm_head.") :]
    return param_name


def _replace_module_leaf(
    param_name: str, old_leaf: str, new_leaf: str, model: torch.nn.Module
) -> str:
    parts = param_name.split(".")
    if len(parts) < 2 or parts[-2] != old_leaf:
        raise ValueError(
            f"Expected module leaf {old_leaf!r} in parameter name {param_name!r}"
        )
    parts[-2] = new_leaf
    return _to_hf_key(".".join(parts), model)


def _emit_embedding_rule(
    param_name: str,
    param: torch.nn.Parameter,
    mod: torch.nn.Module,
    model: torch.nn.Module,
    tp_size: int,
) -> dict[str, Any]:
    leaf = param_name.rsplit(".", 1)[-1]
    if leaf not in ("weight", "bias"):
        raise NotImplementedError(
            f"Unsupported parameter {param_name!r} on {type(mod).__name__}"
        )
    hf_key = _to_hf_key(param_name, model)
    layer_tp = getattr(mod, "tp_size", tp_size)
    if not isinstance(layer_tp, int) or layer_tp < 1:
        layer_tp = tp_size
    tp_ax = "tp" if layer_tp > 1 else None

    global_vocab = param.shape[0] * layer_tp
    org_vocab = getattr(mod, "org_vocab_size", global_vocab)
    if not isinstance(org_vocab, int):
        org_vocab = global_vocab
    ops: list[dict[str, dict[str, Any]]] = []
    if org_vocab < global_vocab:
        ops.append({"pad": {"axis": 0, "sizes": [0, global_vocab - org_vocab]}})

    sharding: list[str | None] = [tp_ax, None] if leaf == "weight" else [tp_ax]
    return {
        "source_keys": [hf_key],
        "target_keys": [param_name],
        "ops": ops,
        "sharding": sharding,
    }


def _emit_qkv_rule(
    param_name: str,
    param: torch.nn.Parameter,
    mod: torch.nn.Module,
    parent_mod: torch.nn.Module | None,
    model: torch.nn.Module,
    packed_mapping: dict[str, list[str]],
    tp_size: int,
) -> list[dict[str, Any]]:
    _check_no_gqa_head_padding(parent_mod, param_name)

    parts = param_name.split(".")
    mod_leaf = parts[-2]
    param_leaf = parts[-1]
    if param_leaf not in ("weight", "bias"):
        raise NotImplementedError(
            f"Unsupported parameter {param_name!r} on QKVParallelLinear"
        )

    src_leaves = packed_mapping.get(mod_leaf, ["q_proj", "k_proj", "v_proj"])
    if len(src_leaves) != 3:
        raise NotImplementedError(
            f"Expected 3 source projections for QKVParallelLinear {param_name!r}, "
            f"got {src_leaves!r}"
        )
    source_keys = [
        _replace_module_leaf(param_name, mod_leaf, s, model) for s in src_leaves
    ]

    head_size = int(mod.head_size)
    total_num_heads = int(mod.total_num_heads)
    total_num_kv_heads = int(mod.total_num_kv_heads)
    layer_tp = int(getattr(mod, "tp_size", tp_size))
    replicas = int(
        getattr(mod, "num_kv_head_replicas", max(1, layer_tp // total_num_kv_heads))
    )

    q_total = total_num_heads * head_size
    kv_orig = total_num_kv_heads * head_size
    is_weight = param_leaf == "weight"
    flipped = is_weight and bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    tail = [int(param.shape[0] if flipped else param.shape[1])] if is_weight else []

    if layer_tp == 1:
        ops: list[dict[str, dict[str, Any]]] = [
            {"concat": {"axis": 0, "sizes": [q_total, kv_orig, kv_orig]}}
        ]
        if flipped:
            ops.append({"transpose": {"axes": [1, 0]}})
        sharding: list[str | None] = [None, None] if is_weight else [None]
        return [
            {
                "source_keys": source_keys,
                "target_keys": [param_name],
                "ops": ops,
                "sharding": sharding,
            }
        ]

    sharding = ([None, "tp"] if flipped else ["tp", None]) if is_weight else ["tp"]
    if q_total == kv_orig and replicas == 1:
        local_q = _exact_div(q_total, layer_tp, f"{param_name} q_total")
        ops = [
            {"concat": {"axis": 0, "sizes": [q_total, kv_orig, kv_orig]}},
            *_tp_interleave_ops(3, layer_tp, local_q, tail),
        ]
        if flipped:
            ops.append({"transpose": {"axes": [1, 0]}})
        return [
            {
                "source_keys": source_keys,
                "target_keys": [param_name],
                "ops": ops,
                "sharding": sharding,
            }
        ]

    return _emit_tp_sharded_segments(
        param_name,
        source_keys,
        [q_total, kv_orig, kv_orig],
        [1, replicas, replicas],
        layer_tp,
        tail,
        flipped=flipped,
        sharding=sharding,
    )


def _emit_gdn_column_parallel_rule(
    param_name: str,
    param: torch.nn.Parameter,
    mod: torch.nn.Module,
    model: torch.nn.Module,
    packed_mapping: dict[str, list[str]],
    tp_size: int,
) -> list[dict[str, Any]]:
    parts = param_name.split(".")
    mod_leaf = parts[-2]
    param_leaf = parts[-1]
    if param_leaf != "weight":
        raise NotImplementedError(
            f"Unsupported parameter {param_name!r} on GdnColumnParallelLinear"
        )

    checkpoint_sizes: list[int] = [int(x) for x in mod.checkpoint_sizes]
    replication_factors: list[int] = [int(x) for x in mod.replication_factors]
    layer_tp = int(getattr(mod, "tp_size", tp_size))

    # conv1d (3D weight [local_conv_dim, 1, kernel_size]) vs in_proj_qkvz (2D weight)
    if param.ndim == 3:
        hf_key = _to_hf_key(param_name, model)
        if layer_tp == 1:
            return [
                {
                    "source_keys": [hf_key],
                    "target_keys": [param_name],
                    "ops": [],
                    "sharding": [None, None, None],
                }
            ]
        raw_keys = [
            f"__tmp__:{param_name}:raw{i}" for i in range(len(checkpoint_sizes))
        ]
        split_rule: dict[str, Any] = {
            "source_keys": [hf_key],
            "target_keys": raw_keys,
            "ops": [{"split": {"axis": 0, "sizes": checkpoint_sizes}}],
        }
        return [
            split_rule,
            *_emit_tp_sharded_segments(
                param_name,
                raw_keys,
                checkpoint_sizes,
                replication_factors,
                layer_tp,
                [int(param.shape[1]), int(param.shape[2])],
                flipped=False,
                sharding=["tp", None, None],
            ),
        ]

    if param.ndim != 2:
        raise NotImplementedError(
            f"Unsupported ndim={param.ndim} on GdnColumnParallelLinear {param_name!r}"
        )

    flipped = bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    src_leaves = packed_mapping.get(mod_leaf)

    if layer_tp == 1:
        if (
            src_leaves is not None
            and len(src_leaves) == 2
            and len(checkpoint_sizes) == 4
        ):
            qkv_size = sum(checkpoint_sizes[:3])
            z_size = checkpoint_sizes[3]
            source_keys = [
                _replace_module_leaf(param_name, mod_leaf, s, model) for s in src_leaves
            ]
            ops: list[dict[str, dict[str, Any]]] = [
                {"concat": {"axis": 0, "sizes": [qkv_size, z_size]}}
            ]
        elif src_leaves is not None and len(src_leaves) == len(checkpoint_sizes):
            source_keys = [
                _replace_module_leaf(param_name, mod_leaf, s, model) for s in src_leaves
            ]
            ops = [{"concat": {"axis": 0, "sizes": checkpoint_sizes}}]
        else:
            source_keys = [_to_hf_key(param_name, model)]
            ops = []
        if flipped:
            ops.append({"transpose": {"axes": [1, 0]}})
        return [
            {
                "source_keys": source_keys,
                "target_keys": [param_name],
                "ops": ops,
                "sharding": [None, None],
            }
        ]

    prefix_rules: list[dict[str, Any]] = []
    if src_leaves is not None and len(src_leaves) == 2 and len(checkpoint_sizes) == 4:
        qkv_src = _replace_module_leaf(param_name, mod_leaf, src_leaves[0], model)
        z_src = _replace_module_leaf(param_name, mod_leaf, src_leaves[1], model)
        raw_qkv = [f"__tmp__:{param_name}:raw{i}" for i in range(3)]
        prefix_rules.append(
            {
                "source_keys": [qkv_src],
                "target_keys": raw_qkv,
                "ops": [{"split": {"axis": 0, "sizes": checkpoint_sizes[:3]}}],
            }
        )
        per_seg_inputs = [*raw_qkv, z_src]
    elif src_leaves is not None and len(src_leaves) == len(checkpoint_sizes):
        per_seg_inputs = [
            _replace_module_leaf(param_name, mod_leaf, s, model) for s in src_leaves
        ]
    else:
        raw_all = [f"__tmp__:{param_name}:raw{i}" for i in range(len(checkpoint_sizes))]
        prefix_rules.append(
            {
                "source_keys": [_to_hf_key(param_name, model)],
                "target_keys": raw_all,
                "ops": [{"split": {"axis": 0, "sizes": checkpoint_sizes}}],
            }
        )
        per_seg_inputs = raw_all

    hidden = int(param.shape[0] if flipped else param.shape[1])
    sharding: list[str | None] = [None, "tp"] if flipped else ["tp", None]
    return [
        *prefix_rules,
        *_emit_tp_sharded_segments(
            param_name,
            per_seg_inputs,
            checkpoint_sizes,
            replication_factors,
            layer_tp,
            [hidden],
            flipped=flipped,
            sharding=sharding,
        ),
    ]


def _emit_gdn_interleaved_rule(
    param_name: str,
    param: torch.nn.Parameter,
    mod: torch.nn.Module,
    model: torch.nn.Module,
    tp_size: int,
) -> dict[str, Any]:
    if param.ndim != 2 or not param_name.endswith(".weight"):
        raise NotImplementedError(
            f"Unsupported parameter {param_name!r} on "
            "GdnInterleavedColumnParallelLinear"
        )
    geometry = mod.geometry
    group_sizes: list[int] = [int(x) for x in mod.group_sizes]
    replicated_segments: list[bool] = [bool(x) for x in mod.replicated_segments]
    if len(set(group_sizes)) != 1 or any(replicated_segments):
        raise NotImplementedError(
            f"Heterogeneous/replicated GdnInterleavedColumnParallelLinear "
            f"{param_name!r} (group_sizes={group_sizes}, "
            f"replicated_segments={replicated_segments}) is not supported."
        )

    num_kq_heads = int(geometry.num_kq_heads)
    n_segs = len(group_sizes)
    seg_w = group_sizes[0]
    total_per_seg = num_kq_heads * seg_w
    layer_tp = int(getattr(mod, "tp_size", tp_size))
    flipped = bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    hidden = int(param.shape[0] if flipped else param.shape[1])

    ops: list[dict[str, dict[str, Any]]] = [
        {
            "reshape": {
                "src_shape": [num_kq_heads * n_segs * seg_w, hidden],
                "dst_shape": [num_kq_heads, n_segs, seg_w, hidden],
            }
        },
        {"transpose": {"axes": [1, 0, 2, 3]}},
        {
            "reshape": {
                "src_shape": [n_segs, num_kq_heads, seg_w, hidden],
                "dst_shape": [n_segs * total_per_seg, hidden],
            }
        },
    ]
    if layer_tp > 1:
        local_per_seg = _exact_div(
            total_per_seg, layer_tp, f"{param_name} total_per_seg"
        )
        ops.extend(_tp_interleave_ops(n_segs, layer_tp, local_per_seg, [hidden]))
    if flipped:
        ops.append({"transpose": {"axes": [1, 0]}})

    tp_ax = "tp" if layer_tp > 1 else None
    sharding: list[str | None] = [None, tp_ax] if flipped else [tp_ax, None]
    return {
        "source_keys": [_to_hf_key(param_name, model)],
        "target_keys": [param_name],
        "ops": ops,
        "sharding": sharding,
    }


def _emit_merged_column_parallel_rule(
    param_name: str,
    param: torch.nn.Parameter,
    mod: torch.nn.Module,
    model: torch.nn.Module,
    packed_mapping: dict[str, list[str]],
    tp_size: int,
) -> list[dict[str, Any]]:
    parts = param_name.split(".")
    mod_leaf = parts[-2]
    param_leaf = parts[-1]
    if param_leaf not in ("weight", "bias"):
        raise NotImplementedError(
            f"Unsupported parameter {param_name!r} on MergedColumnParallelLinear"
        )

    output_sizes = [int(x) for x in mod.output_sizes]
    src_leaves = packed_mapping.get(mod_leaf)
    if src_leaves is None or len(src_leaves) != len(output_sizes):
        raise NotImplementedError(
            f"Cannot resolve packed HF source names for {param_name!r} "
            f"(output_sizes={output_sizes}, mapped={src_leaves!r})"
        )
    source_keys = [
        _replace_module_leaf(param_name, mod_leaf, s, model) for s in src_leaves
    ]

    layer_tp = int(getattr(mod, "tp_size", tp_size))
    is_weight = param_leaf == "weight"
    flipped = is_weight and bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    tail = [int(param.shape[0] if flipped else param.shape[1])] if is_weight else []

    if layer_tp == 1:
        ops: list[dict[str, dict[str, Any]]] = [
            {"concat": {"axis": 0, "sizes": output_sizes}}
        ]
        if flipped:
            ops.append({"transpose": {"axes": [1, 0]}})
        sharding: list[str | None] = [None, None] if is_weight else [None]
        return [
            {
                "source_keys": source_keys,
                "target_keys": [param_name],
                "ops": ops,
                "sharding": sharding,
            }
        ]

    sharding = ([None, "tp"] if flipped else ["tp", None]) if is_weight else ["tp"]
    if len(set(output_sizes)) == 1:
        seg_local = _exact_div(output_sizes[0], layer_tp, f"{param_name} output_sizes")
        ops = [
            {"concat": {"axis": 0, "sizes": output_sizes}},
            *_tp_interleave_ops(len(output_sizes), layer_tp, seg_local, tail),
        ]
        if flipped:
            ops.append({"transpose": {"axes": [1, 0]}})
        return [
            {
                "source_keys": source_keys,
                "target_keys": [param_name],
                "ops": ops,
                "sharding": sharding,
            }
        ]

    return _emit_tp_sharded_segments(
        param_name,
        source_keys,
        output_sizes,
        [1] * len(output_sizes),
        layer_tp,
        tail,
        flipped=flipped,
        sharding=sharding,
    )


def _emit_column_parallel_rule(
    param_name: str,
    param: torch.nn.Parameter,
    mod: torch.nn.Module,
    model: torch.nn.Module,
    tp_size: int,
) -> dict[str, Any]:
    param_leaf = param_name.rsplit(".", 1)[-1]
    if param_leaf not in ("weight", "bias"):
        raise NotImplementedError(
            f"Unsupported parameter {param_name!r} on ColumnParallelLinear"
        )
    hf_key = _to_hf_key(param_name, model)
    layer_tp = int(getattr(mod, "tp_size", tp_size))
    is_weight = param_leaf == "weight"
    flipped = is_weight and bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    replicas = int(getattr(mod, "num_kv_head_replicas", 1))
    tail = [int(param.shape[0] if flipped else param.shape[1])] if is_weight else []

    ops: list[dict[str, dict[str, Any]]] = []
    if replicas > 1:
        head_size = getattr(mod, "head_size", None)
        total_kv = getattr(mod, "total_num_kv_heads", None)
        if not isinstance(head_size, int) or not isinstance(total_kv, int):
            raise NotImplementedError(
                f"Replicated ColumnParallelLinear {param_name!r} requires "
                "integer head_size and total_num_kv_heads attributes."
            )
        kv_orig = total_kv * head_size
        ops.extend(
            [
                {
                    "reshape": {
                        "src_shape": [kv_orig, *tail],
                        "dst_shape": [total_kv, head_size, *tail],
                    }
                },
                {"repeat": {"axis": 0, "times": replicas}},
                {
                    "reshape": {
                        "src_shape": [total_kv * replicas, head_size, *tail],
                        "dst_shape": [kv_orig * replicas, *tail],
                    }
                },
            ]
        )

    if flipped:
        ops.append({"transpose": {"axes": [1, 0]}})

    tp_ax = "tp" if layer_tp > 1 else None
    sharding: list[str | None] = (
        ([None, tp_ax] if flipped else [tp_ax, None]) if is_weight else [tp_ax]
    )
    return {
        "source_keys": [hf_key],
        "target_keys": [param_name],
        "ops": ops,
        "sharding": sharding,
    }


def _emit_row_parallel_rule(
    param_name: str,
    param: torch.nn.Parameter,
    mod: torch.nn.Module,
    parent_mod: torch.nn.Module | None,
    model: torch.nn.Module,
    tp_size: int,
) -> dict[str, Any]:
    _check_no_gqa_head_padding(parent_mod, param_name)

    param_leaf = param_name.rsplit(".", 1)[-1]
    if param_leaf not in ("weight", "bias"):
        raise NotImplementedError(
            f"Unsupported parameter {param_name!r} on RowParallelLinear"
        )
    hf_key = _to_hf_key(param_name, model)
    layer_tp = int(getattr(mod, "tp_size", tp_size))
    is_weight = param_leaf == "weight"
    flipped = is_weight and bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))

    ops: list[dict[str, dict[str, Any]]] = (
        [{"transpose": {"axes": [1, 0]}}] if flipped else []
    )
    tp_ax = "tp" if layer_tp > 1 else None
    if is_weight:
        sharding: list[str | None] = [tp_ax, None] if flipped else [None, tp_ax]
    else:
        sharding = [None]

    return {
        "source_keys": [hf_key],
        "target_keys": [param_name],
        "ops": ops,
        "sharding": sharding,
    }


def _emit_replicated_linear_rule(
    param_name: str,
    mod: torch.nn.Module,
    model: torch.nn.Module,
) -> dict[str, Any]:
    param_leaf = param_name.rsplit(".", 1)[-1]
    if param_leaf not in ("weight", "bias"):
        raise NotImplementedError(
            f"Unsupported parameter {param_name!r} on {type(mod).__name__}"
        )
    hf_key = _to_hf_key(param_name, model)
    is_weight = param_leaf == "weight"
    flipped = is_weight and bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    ops: list[dict[str, dict[str, Any]]] = (
        [{"transpose": {"axes": [1, 0]}}] if flipped else []
    )
    sharding: list[str | None] = [None, None] if is_weight else [None]
    return {
        "source_keys": [hf_key],
        "target_keys": [param_name],
        "ops": ops,
        "sharding": sharding,
    }


def _is_swigluoai_activation(value: Any) -> bool:
    if value is None:
        return False
    raw = getattr(value, "value", value)
    return str(raw).lower() == "swigluoai"


def _emit_routed_experts_rule(
    param_name: str,
    param: torch.nn.Parameter,
    mod: torch.nn.Module,
    model: torch.nn.Module,
    tp_size: int,
    dp_size: int,
    ep_size: int,
) -> tuple[dict[str, Any], int]:
    param_leaf = param_name.rsplit(".", 1)[-1]
    if param_leaf not in ("w13_weight", "w2_weight"):
        raise NotImplementedError(
            f"Unsupported RoutedExperts parameter {param_name!r}; only "
            "unquantized w13_weight and w2_weight without bias are supported."
        )

    moe_cfg = getattr(mod, "moe_config", None)
    if moe_cfg is not None and getattr(moe_cfg, "has_bias", False):
        raise NotImplementedError(
            f"RoutedExperts with expert bias ({param_name!r}) is not supported."
        )
    if _is_swigluoai_activation(
        getattr(moe_cfg, "activation", None)
    ) or _is_swigluoai_activation(getattr(mod, "activation", None)):
        raise NotImplementedError(
            f"RoutedExperts with swigluoai activation ({param_name!r}) is not "
            "supported by the RL weight conversion manifest generator."
        )

    if moe_cfg is not None:
        num_experts = int(moe_cfg.num_experts)
        moe_p = getattr(moe_cfg, "moe_parallel_config", None)
        use_ep = (
            bool(getattr(moe_p, "use_ep", False)) if moe_p is not None else ep_size > 1
        )
        ep_layer = int(getattr(moe_cfg, "ep_size", ep_size)) if use_ep else 1
        moe_tp = int(getattr(moe_cfg, "tp_size", 1 if use_ep else tp_size))
        unpadded_half = int(moe_cfg.intermediate_size_per_partition)
        hidden_size = int(moe_cfg.hidden_dim)
    else:
        ep_layer = ep_size
        moe_tp = (
            _exact_div(tp_size * dp_size, ep_layer, f"{param_name} moe_tp")
            if ep_layer > 1
            else tp_size
        )
        num_experts = param.shape[0] * ep_layer
        unpadded_half = int(mod.intermediate_size_per_partition)
        hidden_size = int(mod.hidden_size)

    local_e = param.shape[0]
    if local_e * ep_layer != num_experts:
        raise ValueError(
            f"RoutedExperts {param_name!r}: local_e={local_e} * ep={ep_layer} "
            f"!= num_experts={num_experts}"
        )

    # Determine whether TPU GMM transpose [E, 2*I, H] -> [E, H, 2*I] was applied.
    explicit_flipped = getattr(mod, WEIGHT_FLIPPED_ATTR, None)
    if explicit_flipped is not None:
        transposed = bool(explicit_flipped)
    elif param.shape[1] == param.shape[2] == hidden_size:
        quant_method = getattr(mod, "quant_method", None)
        qm_name = type(quant_method).__name__ if quant_method is not None else None
        if qm_name == "VllmUnquantizedFusedMoEMethod":
            transposed = True
        elif qm_name == "UnquantizedFusedMoEMethod":
            transposed = False
        else:
            raise NotImplementedError(
                f"Ambiguous square RoutedExperts layout {tuple(param.shape)} on "
                f"{param_name!r}: set {WEIGHT_FLIPPED_ATTR} or attach a "
                "recognized unquantized FusedMoEMethod."
            )
    elif param_leaf == "w13_weight":
        transposed = param.shape[1] == hidden_size
    else:
        transposed = param.shape[2] == hidden_size

    mod_prefix = param_name.rsplit(".", 1)[0]
    hf_prefix = _to_hf_key(mod_prefix, model)
    ckpt_gate = str(getattr(mod, "ckpt_gate_proj_name", "gate_proj"))
    ckpt_up = str(getattr(mod, "ckpt_up_proj_name", "up_proj"))
    ckpt_down = str(getattr(mod, "ckpt_down_proj_name", "down_proj"))

    ep_ax = "ep" if ep_layer > 1 else None
    tp_ax = "moe_tp" if moe_tp > 1 else None
    inter_total = unpadded_half * moe_tp

    if param_leaf == "w13_weight":
        w13_out = param.shape[2] if transposed else param.shape[1]
        aligned_half = _exact_div(w13_out, 2, f"{param_name} w13 output width")
        pad_amount = aligned_half - unpadded_half
        if pad_amount < 0:
            raise ValueError(
                f"{param_name!r}: aligned_half={aligned_half} < "
                f"unpadded_half={unpadded_half}"
            )
        source_keys = [
            f"{hf_prefix}.{e}.{ckpt_gate}.weight" for e in range(num_experts)
        ] + [f"{hf_prefix}.{e}.{ckpt_up}.weight" for e in range(num_experts)]

        ops: list[dict[str, dict[str, Any]]] = [
            {"stack": {"axis": 0}},
            {
                "reshape": {
                    "src_shape": [2 * num_experts, inter_total, hidden_size],
                    "dst_shape": [2, num_experts, moe_tp, unpadded_half, hidden_size],
                }
            },
            {"transpose": {"axes": [1, 2, 0, 3, 4]}},
        ]
        if pad_amount > 0:
            ops.append({"pad": {"axis": 3, "sizes": [0, pad_amount]}})
        ops.append(
            {
                "reshape": {
                    "src_shape": [num_experts, moe_tp, 2, aligned_half, hidden_size],
                    "dst_shape": [
                        num_experts,
                        moe_tp * 2 * aligned_half,
                        hidden_size,
                    ],
                }
            }
        )
        if transposed:
            ops.append({"transpose": {"axes": [0, 2, 1]}})
        sharding: list[str | None] = (
            [ep_ax, None, tp_ax] if transposed else [ep_ax, tp_ax, None]
        )
        return (
            {
                "source_keys": source_keys,
                "target_keys": [param_name],
                "ops": ops,
                "sharding": sharding,
            },
            moe_tp,
        )

    # w2_weight
    aligned_half = param.shape[1] if transposed else param.shape[2]
    pad_amount = aligned_half - unpadded_half
    if pad_amount < 0:
        raise ValueError(
            f"{param_name!r}: aligned_half={aligned_half} < "
            f"unpadded_half={unpadded_half}"
        )
    source_keys = [f"{hf_prefix}.{e}.{ckpt_down}.weight" for e in range(num_experts)]
    ops = [{"stack": {"axis": 0}}]
    if pad_amount > 0:
        ops.extend(
            [
                {
                    "reshape": {
                        "src_shape": [num_experts, hidden_size, inter_total],
                        "dst_shape": [num_experts, hidden_size, moe_tp, unpadded_half],
                    }
                },
                {"pad": {"axis": 3, "sizes": [0, pad_amount]}},
                {
                    "reshape": {
                        "src_shape": [num_experts, hidden_size, moe_tp, aligned_half],
                        "dst_shape": [
                            num_experts,
                            hidden_size,
                            moe_tp * aligned_half,
                        ],
                    }
                },
            ]
        )
    if transposed:
        ops.append({"transpose": {"axes": [0, 2, 1]}})
    sharding = [ep_ax, tp_ax, None] if transposed else [ep_ax, None, tp_ax]
    return (
        {
            "source_keys": source_keys,
            "target_keys": [param_name],
            "ops": ops,
            "sharding": sharding,
        },
        moe_tp,
    )


def generate_manifest_dict(
    model: torch.nn.Module,
    parallel_config: Any = None,
    model_config: Any = None,
    *,
    tp_size: int | None = None,
    dp_size: int | None = None,
    ep_size: int | None = None,
) -> dict[str, Any]:
    """Generate a validated ``weight_mapping`` manifest dict for ``model``.

    Args:
        model: Instantiated post-``process_weights_after_loading()`` vLLM /
            TorchTPU ``nn.Module``.
        parallel_config: Optional vLLM ``ParallelConfig``.
        model_config: Optional vLLM ``ModelConfig``.
        tp_size: Optional explicit tensor-parallel size override.
        dp_size: Optional explicit data-parallel size override.
        ep_size: Optional explicit expert-parallel size override.

    Returns:
        Validated manifest dictionary conforming to ``weight_mapping`` schema.

    Raises:
        NotImplementedError: On quantized models/weights, unsupported
            parallelism (PP/PCP/DCP > 1, EPLB), LoRA, active vision/MTP
            parameters, unsupported MoE activations/biases, or unrecognized
            module types.
    """
    _check_parallel_config(parallel_config)
    _check_model_and_quant_config(model, model_config)

    resolved_tp, resolved_dp, resolved_ep = _resolve_parallel_sizes(
        model, parallel_config, tp_size, dp_size, ep_size
    )
    packed_mapping = _collect_packed_mapping(model)

    modules_by_name: dict[str, torch.nn.Module] = dict(model.named_modules())
    for mod_name, mod in modules_by_name.items():
        _check_module_guardrails(mod_name, mod)

    rules: list[dict[str, Any]] = []
    tied_keys: list[list[str]] = []
    seen_param_ids: dict[int, str] = {}
    has_routed_experts = False
    resolved_moe_tp: int | None = None

    for param_name, param in model.named_parameters(remove_duplicate=False):
        _check_param_guardrails(param_name, param)

        param_id = id(param)
        if param_id in seen_param_ids:
            tied_keys.append([seen_param_ids[param_id], param_name])
            continue
        seen_param_ids[param_id] = param_name

        if "." in param_name:
            mod_name, param_leaf = param_name.rsplit(".", 1)
        else:
            mod_name, param_leaf = "", param_name
        mod = modules_by_name[mod_name]
        parent_name = mod_name.rsplit(".", 1)[0] if "." in mod_name else ""
        parent_mod = (
            modules_by_name.get(parent_name) if parent_name != mod_name else None
        )

        if _has_class_in_mro(mod, ("ParallelLMHead", "VocabParallelEmbedding")):
            rules.append(
                _emit_embedding_rule(param_name, param, mod, model, resolved_tp)
            )
        elif _has_class_in_mro(mod, "QKVParallelLinear"):
            rules.extend(
                _emit_qkv_rule(
                    param_name,
                    param,
                    mod,
                    parent_mod,
                    model,
                    packed_mapping,
                    resolved_tp,
                )
            )
        elif _has_class_in_mro(mod, "GdnColumnParallelLinear"):
            rules.extend(
                _emit_gdn_column_parallel_rule(
                    param_name,
                    param,
                    mod,
                    model,
                    packed_mapping,
                    resolved_tp,
                )
            )
        elif _has_class_in_mro(mod, "GdnInterleavedColumnParallelLinear"):
            rules.append(
                _emit_gdn_interleaved_rule(param_name, param, mod, model, resolved_tp)
            )
        elif _has_class_in_mro(mod, "MergedColumnParallelLinear"):
            rules.extend(
                _emit_merged_column_parallel_rule(
                    param_name,
                    param,
                    mod,
                    model,
                    packed_mapping,
                    resolved_tp,
                )
            )
        elif _has_class_in_mro(mod, "ColumnParallelLinear"):
            rules.append(
                _emit_column_parallel_rule(param_name, param, mod, model, resolved_tp)
            )
        elif _has_class_in_mro(mod, "RowParallelLinear"):
            rules.append(
                _emit_row_parallel_rule(
                    param_name, param, mod, parent_mod, model, resolved_tp
                )
            )
        elif _has_class_in_mro(mod, ("ReplicatedLinear", "GateLinear")):
            rules.append(_emit_replicated_linear_rule(param_name, mod, model))
        elif _has_class_in_mro(mod, "RoutedExperts"):
            has_routed_experts = True
            moe_rule, resolved_moe_tp = _emit_routed_experts_rule(
                param_name,
                param,
                mod,
                model,
                resolved_tp,
                resolved_dp,
                resolved_ep,
            )
            rules.append(moe_rule)
        elif _has_class_in_mro(
            mod, ("RMSNorm", "GemmaRMSNorm", "RMSNormGated", "Qwen3_5RMSNorm")
        ):
            if param_leaf != "weight":
                raise NotImplementedError(
                    f"Unsupported norm parameter {param_name!r} on {type(mod).__name__}"
                )
            rules.append(
                {
                    "source_keys": [_to_hf_key(param_name, model)],
                    "target_keys": [param_name],
                    "ops": [],
                    "sharding": [None] * param.ndim,
                }
            )
        elif _has_class_in_mro(
            mod, ("QwenGatedDeltaNetAttention", "VllmGatedDeltaNetAttention")
        ) and param_leaf in ("A_log", "dt_bias"):
            layer_tp = int(getattr(mod, "tp_size", resolved_tp))
            rules.append(
                {
                    "source_keys": [_to_hf_key(param_name, model)],
                    "target_keys": [param_name],
                    "ops": [],
                    "sharding": ["tp" if layer_tp > 1 else None],
                }
            )
        elif param_leaf in ("attn_layer_scale", "ffn_layer_scale"):
            rules.append(
                {
                    "source_keys": [_to_hf_key(param_name, model)],
                    "target_keys": [param_name],
                    "ops": [],
                    "sharding": [None] * param.ndim,
                }
            )
        else:
            raise NotImplementedError(
                f"Unrecognized module type {type(mod).__name__!r} for parameter "
                f"{param_name!r}."
            )

    stripes: dict[str, int] = {"tp_size": resolved_tp}
    if has_routed_experts:
        stripes["ep_size"] = resolved_ep
        stripes["moe_tp_size"] = (
            resolved_moe_tp if resolved_moe_tp is not None else resolved_tp
        )

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "expert_axis_grouped": has_routed_experts,
        "stripes": stripes,
        "tied_keys": tied_keys,
        "dropped_keys": [],
        "rules": rules,
    }
    validate_manifest(manifest)
    return manifest


def generate_manifest(
    model: torch.nn.Module,
    parallel_config: Any = None,
    model_config: Any = None,
    *,
    tp_size: int | None = None,
    dp_size: int | None = None,
    ep_size: int | None = None,
) -> str:
    """Generate a serialized JSON ``weight_mapping`` manifest for ``model``.

    Args:
        model: Instantiated post-``process_weights_after_loading()`` vLLM /
            TorchTPU ``nn.Module``.
        parallel_config: Optional vLLM ``ParallelConfig``.
        model_config: Optional vLLM ``ModelConfig``.
        tp_size: Optional explicit tensor-parallel size override.
        dp_size: Optional explicit data-parallel size override.
        ep_size: Optional explicit expert-parallel size override.

    Returns:
        JSON string conforming to the ``weight_mapping`` protocol schema.
    """
    manifest = generate_manifest_dict(
        model,
        parallel_config=parallel_config,
        model_config=model_config,
        tp_size=tp_size,
        dp_size=dp_size,
        ep_size=ep_size,
    )
    return json.dumps(manifest)


__all__ = [
    "WEIGHT_FLIPPED_ATTR",
    "generate_manifest",
    "generate_manifest_dict",
]
