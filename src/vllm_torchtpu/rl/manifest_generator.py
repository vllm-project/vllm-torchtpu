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
"""Model-agnostic ``weight_mapping`` manifest generator via layer reflection.

Walks ``model.named_parameters(remove_duplicate=False)`` and dispatches each
parameter by its exact owning vLLM/TorchTPU module class (patterned on PR #3363's
``param_layout.py``), resolving HF source keys via ``hf_to_vllm_mapper`` and
``packed_modules_mapping``. Fails closed with ``NotImplementedError`` on
unsupported layers, quantization, PP/PCP/DCP > 1, EPLB, LoRA, DP vocab parallel,
custom expert placement, active vision/MTP, or unclaimed parameters. Pure PyTorch
only (zero JAX imports).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import torch

from vllm_torchtpu.rl.weight_conversion import (
    SCHEMA_VERSION,
    OpSpec,
    validate_manifest,
)

WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"
DP_VOCAB_ATTR = "_tpu_dp_vocab"

_ALLOWED_DTYPES = frozenset({torch.bfloat16, torch.float16, torch.float32})
_ALLOWED_QUANT_METHODS = frozenset(
    {
        "UnquantizedLinearMethod",
        "VllmUnquantizedLinearMethod",
        "UnquantizedEmbeddingMethod",
        "UnquantizedFusedMoEMethod",
        "VllmUnquantizedFusedMoEMethod",
    }
)
_UNQUANTIZED_NAMES = frozenset({"", "none", "unquantized", "tpu-unquantized"})

# Exact-class whitelists (by class name so CPU tests work without torch_tpu).
_VOCAB_CLASSES = frozenset({"VocabParallelEmbedding", "ParallelLMHead"})
_ROW_LINEAR_CLASSES = frozenset({"RowParallelLinear", "TpuRowParallelLinear"})
_REPLICATED_LINEAR_CLASSES = frozenset({"ReplicatedLinear", "GateLinear"})
_COL_LINEAR_CLASSES = frozenset({"ColumnParallelLinear"})
_MERGED_COL_CLASSES = frozenset({"MergedColumnParallelLinear"})
_QKV_CLASSES = frozenset({"QKVParallelLinear"})
_GDN_INTERLEAVED_CLASSES = frozenset({"GdnInterleavedColumnParallelLinear"})
_GDN_SEG_CLASSES = frozenset({"GdnColumnParallelLinear", "GdnConv1d"})
_ROUTED_EXPERTS_CLASSES = frozenset({"RoutedExperts"})
_NORM_CLASSES = frozenset(
    {
        "RMSNorm",
        "GemmaRMSNorm",
        "Qwen3NextRMSNormGated",
        "RMSNormGated",
        "LayerNorm",
    }
)
_HEAD_SHARDED_1D_LEAVES = frozenset({"A_log", "dt_bias"})
_DEFAULT_PACKED: dict[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}


def _exact_div(total: int, divisor: int, context: str) -> int:
    if divisor < 1 or total % divisor != 0:
        raise ValueError(f"{context}: {total} is not evenly divisible by {divisor}")
    return total // divisor


def _check_leaf(name: str, leaf: str, allowed: tuple[str, ...]) -> None:
    if leaf not in allowed:
        raise NotImplementedError(
            f"Unsupported parameter {name!r} (expected one of {allowed})."
        )


def _rule(
    source_keys: list[str],
    target_keys: list[str],
    ops: list[OpSpec],
    sharding: list[str | None] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source_keys": source_keys,
        "target_keys": target_keys,
        "ops": ops,
    }
    if sharding is not None:
        out["sharding"] = sharding
    return out


# ---------------------------------------------------------------------------
# TP-interleaved fused column helpers (Option 1: __tmp__:* rule chaining)
# ---------------------------------------------------------------------------


def _tp_interleave_ops(
    num_segments: int, tp_size: int, local_seg_width: int, tail: list[int]
) -> list[OpSpec]:
    total = num_segments * tp_size * local_seg_width
    tail_axes = list(range(3, 3 + len(tail)))
    return [
        {
            "reshape": {
                "src_shape": [total, *tail],
                "dst_shape": [num_segments, tp_size, local_seg_width, *tail],
            }
        },
        {"transpose": {"axes": [1, 0, 2, *tail_axes]}},
        {
            "reshape": {
                "src_shape": [tp_size, num_segments, local_seg_width, *tail],
                "dst_shape": [total, *tail],
            }
        },
    ]


def _fused_col_rules(
    name: str,
    src_keys: list[str],
    seg_sizes: list[int],
    reps: list[int],
    tp_size: int,
    tail: list[int],
    *,
    flipped: bool,
) -> list[dict[str, Any]]:
    """Build fused column-parallel rules (contiguous TP-sharded via Option 1)."""
    is_matrix = len(tail) == 1
    if tp_size == 1:
        ops: list[OpSpec] = (
            [{"concat": {"axis": 0, "sizes": seg_sizes}}] if len(src_keys) > 1 else []
        )
        if flipped:
            ops.append({"transpose": {"axes": [1, 0]}})
        return [_rule(src_keys, [name], ops, [None] * (1 + len(tail)))]

    sharding: list[str | None] = (
        ([None, "tp"] if flipped else ["tp", None])
        if is_matrix
        else ["tp", *([None] * len(tail))]
    )
    if len(set(seg_sizes)) == 1 and all(r == 1 for r in reps) and len(src_keys) > 1:
        local_w = _exact_div(seg_sizes[0], tp_size, f"{name} segment width")
        ops = [
            {"concat": {"axis": 0, "sizes": seg_sizes}},
            *_tp_interleave_ops(len(seg_sizes), tp_size, local_w, tail),
        ]
        if flipped:
            ops.append({"transpose": {"axes": [1, 0]}})
        return [_rule(src_keys, [name], ops, sharding)]

    # Heterogeneous / replicated segments: chain per-segment __tmp__:* rules.
    rules: list[dict[str, Any]] = []
    tmp_keys: list[str] = []
    local_widths: list[int] = []
    for idx, (src_k, size, rep) in enumerate(
        zip(src_keys, seg_sizes, reps, strict=True)
    ):
        tmp_k = f"__tmp__:{name}:seg{idx}"
        seg_ops: list[OpSpec] = []
        if rep > 1:
            num_groups = _exact_div(tp_size, rep, f"{name} seg[{idx}] tp/rep")
            local_w = _exact_div(size, num_groups, f"{name} seg[{idx}] width")
            seg_ops.extend(
                [
                    {
                        "reshape": {
                            "src_shape": [size, *tail],
                            "dst_shape": [num_groups, local_w, *tail],
                        }
                    },
                    {"repeat": {"axis": 0, "times": rep}},
                ]
            )
        else:
            local_w = _exact_div(size, tp_size, f"{name} seg[{idx}] width")
            seg_ops.append(
                {
                    "reshape": {
                        "src_shape": [size, *tail],
                        "dst_shape": [tp_size, local_w, *tail],
                    }
                }
            )
        rules.append(_rule([src_k], [tmp_k], seg_ops))
        tmp_keys.append(tmp_k)
        local_widths.append(local_w)

    total_local = sum(local_widths)
    merge_ops: list[OpSpec] = [
        {"concat": {"axis": 1, "sizes": local_widths}},
        {
            "reshape": {
                "src_shape": [tp_size, total_local, *tail],
                "dst_shape": [tp_size * total_local, *tail],
            }
        },
    ]
    if flipped:
        merge_ops.append({"transpose": {"axes": [1, 0]}})
    rules.append(_rule(tmp_keys, [name], merge_ops, sharding))
    return rules


# ---------------------------------------------------------------------------
# Fail-closed support checks, parallelism, and HF key mapping
# ---------------------------------------------------------------------------


def _check_supported(
    model: torch.nn.Module, parallel_config: Any, model_config: Any
) -> None:
    if parallel_config is not None:
        for attr, label in (
            ("pipeline_parallel_size", "PP"),
            ("prefill_context_parallel_size", "PCP"),
            ("decode_context_parallel_size", "DCP"),
        ):
            if int(getattr(parallel_config, attr, 1) or 1) > 1:
                raise NotImplementedError(f"{label} > 1 is not supported.")
        if getattr(parallel_config, "enable_eplb", False):
            raise NotImplementedError("EPLB is not supported.")
        eplb_cfg = getattr(parallel_config, "eplb_config", None)
        if int(getattr(eplb_cfg, "num_redundant_experts", 0) or 0) > 0:
            raise NotImplementedError("EPLB redundant experts are not supported.")

    ep_env = os.environ.get("TPU_EXPERT_PLACEMENT", "").strip().lower()
    if ep_env and ep_env not in ("0", "false", "no", "linear"):
        raise NotImplementedError(f"TPU_EXPERT_PLACEMENT={ep_env!r} is not supported.")

    if model_config is not None:
        quant = getattr(model_config, "quantization", None)
        if quant is not None and str(quant).lower() not in _UNQUANTIZED_NAMES:
            raise NotImplementedError(f"Quantization ({quant!r}) is not supported.")
        hf_cfg = getattr(model_config, "hf_config", None)
        if hf_cfg is not None and getattr(hf_cfg, "quantization_config", None):
            raise NotImplementedError("Quantization (hf_config) is not supported.")
        if getattr(model_config, "enable_lora", False) or getattr(
            model_config, "lora_config", None
        ):
            raise NotImplementedError("LoRA is not supported.")
        if getattr(model_config, "enable_mtp", False):
            raise NotImplementedError("MTP is not supported.")

    if getattr(model, "lora_config", None):
        raise NotImplementedError("LoRA (model.lora_config) is not supported.")
    model_qc = getattr(model, "quant_config", None)
    if model_qc is not None:
        qc_fn = getattr(model_qc, "get_name", None)
        qc_name = str(qc_fn()).lower() if callable(qc_fn) else None
        if (
            type(model_qc).__name__ != "VllmUnquantizedConfig"
            and qc_name not in _UNQUANTIZED_NAMES
        ):
            raise NotImplementedError(
                f"Quantization (model.quant_config={type(model_qc).__name__!r}) "
                "is not supported."
            )

    for mod_name, mod in model.named_modules():
        if "lora" in type(mod).__name__.lower():
            raise NotImplementedError(f"LoRA module {mod_name!r} is not supported.")
        qm = getattr(mod, "quant_method", None)
        if qm is not None and type(qm).__name__ not in _ALLOWED_QUANT_METHODS:
            raise NotImplementedError(
                f"Quantized method {type(qm).__name__!r} on {mod_name!r} "
                "is not supported."
            )
        orig_h = getattr(mod, "orig_total_num_heads", None)
        tot_h = getattr(mod, "total_num_heads", None)
        if isinstance(orig_h, int) and isinstance(tot_h, int) and orig_h != tot_h:
            raise NotImplementedError(
                f"GQA query-head zero-padding ({orig_h} -> {tot_h}) on "
                f"{mod_name!r} is not supported."
            )

    for name, param in model.named_parameters(remove_duplicate=False):
        if name.startswith(("visual.", "mtp.", "model.visual.", "model.mtp.")):
            raise NotImplementedError(
                f"Active vision/MTP parameter {name!r} is not supported."
            )
        parts = name.split(".")
        if "lora" in name.lower() or "base_layer" in parts:
            raise NotImplementedError(f"LoRA parameter {name!r} is not supported.")
        if param.dtype not in _ALLOWED_DTYPES:
            raise NotImplementedError(
                f"Unsupported dtype {param.dtype} on {name!r}; "
                "only unquantized bf16/fp16/fp32 are supported."
            )
        leaf_lower = parts[-1].lower()
        if leaf_lower in (
            "scales",
            "weight_scale",
            "input_scale",
        ) or leaf_lower.endswith(("_scale", "_scale_inv", "_scales")):
            raise NotImplementedError(
                f"Quantization scale parameter {name!r} is not supported."
            )
        if leaf_lower in ("qzeros", "qweight", "g_idx"):
            raise NotImplementedError(
                f"Quantization parameter {name!r} is not supported."
            )


def _resolve_parallel(
    model: torch.nn.Module,
    parallel_config: Any,
    tp_size: int | None,
    dp_size: int | None,
    ep_size: int | None,
) -> tuple[int, int, int]:
    tp = (
        int(tp_size)
        if tp_size is not None
        else int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    )
    if dp_size is not None:
        dp = int(dp_size)
    else:
        env_dp = int(os.environ.get("TORCH_TPU_DP_SIZE", "0") or 0)
        dp = env_dp or int(getattr(parallel_config, "data_parallel_size", 1) or 1)

    if ep_size is not None:
        ep = int(ep_size)
    elif parallel_config is not None and getattr(
        parallel_config, "enable_expert_parallel", False
    ):
        cfg_ep = getattr(parallel_config, "expert_parallel_size", None)
        ep = int(cfg_ep) if isinstance(cfg_ep, int) and cfg_ep > 0 else tp * dp
    elif parallel_config is None:
        ep = 1
        for mod in model.modules():
            qm = getattr(mod, "quant_method", None)
            moe_cfg = getattr(mod, "moe_config", None) or getattr(qm, "moe", None)
            if moe_cfg is not None:
                moe_p = getattr(moe_cfg, "moe_parallel_config", None)
                if getattr(moe_p, "use_ep", False):
                    mod_ep = getattr(moe_cfg, "ep_size", None)
                    if isinstance(mod_ep, int) and mod_ep > ep:
                        ep = mod_ep
    else:
        ep = 1

    if tp_size is None and parallel_config is None:
        for mod in model.modules():
            mod_tp = getattr(mod, "tp_size", None)
            if isinstance(mod_tp, int) and mod_tp > tp:
                tp = mod_tp
    return tp, dp, ep


def _make_to_hf_key(model: torch.nn.Module) -> Callable[[str], str]:
    """Invert vLLM's ``hf_to_vllm_mapper`` prefixes (with VLM fallback)."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    prefixes = getattr(mapper, "orig_to_new_prefix", None)
    inv_prefixes: list[tuple[str, str]] = (
        [(new, orig) for orig, new in prefixes.items() if new and new != orig]
        if isinstance(prefixes, dict)
        else []
    )
    inv_prefixes.sort(key=lambda pair: len(pair[0]), reverse=True)

    def _to_hf(key: str) -> str:
        for new_pfx, orig_pfx in inv_prefixes:
            if key.startswith(new_pfx):
                return orig_pfx + key[len(new_pfx) :]
        if key.startswith("language_model.model."):
            return "model.language_model." + key[len("language_model.model.") :]
        if key.startswith("language_model."):
            return key[len("language_model.") :]
        return key

    return _to_hf


def _collect_packed_mapping(model: torch.nn.Module) -> dict[str, tuple[str, ...]]:
    lm = getattr(model, "language_model", None)
    merged: dict[str, Any] = {
        **(getattr(model, "packed_modules_mapping", None) or {}),
        **(getattr(lm, "packed_modules_mapping", None) or {}),
    }
    return {k: tuple(v) for k, v in merged.items() if v}


# ---------------------------------------------------------------------------
# Layer-class rule builders (dispatched by exact owning nn.Module class)
# ---------------------------------------------------------------------------


def _embedding_rule(
    name: str,
    leaf: str,
    param: torch.Tensor,
    mod: torch.nn.Module,
    tp_size: int,
    to_hf: Callable[[str], str],
) -> list[dict[str, Any]]:
    _check_leaf(name, leaf, ("weight", "bias"))
    if bool(getattr(mod, DP_VOCAB_ATTR, False)):
        raise NotImplementedError(
            f"{name!r}: TPU_DP_VOCAB_PARALLEL ({DP_VOCAB_ATTR}) is not supported."
        )
    layer_tp = int(getattr(mod, "tp_size", tp_size))
    tp_ax = "tp" if layer_tp > 1 else None
    org_vocab = getattr(mod, "org_vocab_size", None)
    padded_vocab = getattr(mod, "num_embeddings_padded", None)
    global_vocab = (
        int(padded_vocab)
        if isinstance(padded_vocab, int)
        else int(param.shape[0]) * layer_tp
    )
    pad_right = global_vocab - int(org_vocab) if isinstance(org_vocab, int) else 0
    if pad_right < 0:
        raise ValueError(
            f"{name!r}: global_vocab={global_vocab} < org_vocab_size={org_vocab}."
        )
    ops: list[OpSpec] = (
        [{"pad": {"axis": 0, "sizes": [0, pad_right]}}] if pad_right > 0 else []
    )
    sharding: list[str | None] = [tp_ax, None] if leaf == "weight" else [tp_ax]
    return [_rule([to_hf(name)], [name], ops, sharding)]


def _linear_rule(
    name: str,
    leaf: str,
    mod: torch.nn.Module,
    tp_size: int,
    to_hf: Callable[[str], str],
    *,
    shard_dim: int | None,
) -> list[dict[str, Any]]:
    _check_leaf(name, leaf, ("weight", "bias"))
    layer_tp = int(getattr(mod, "tp_size", tp_size))
    tp_ax = "tp" if layer_tp > 1 else None
    is_w = leaf == "weight"
    flipped = is_w and bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    ops: list[OpSpec] = [{"transpose": {"axes": [1, 0]}}] if flipped else []
    if not is_w:
        sharding: list[str | None] = [tp_ax if shard_dim == 0 else None]
    elif tp_ax is None or shard_dim is None:
        sharding = [None, None]
    else:
        dim = 1 - shard_dim if flipped else shard_dim
        sharding = [tp_ax if i == dim else None for i in range(2)]
    return [_rule([to_hf(name)], [name], ops, sharding)]


def _qkv_or_merged_col_rule(
    name: str,
    leaf: str,
    param: torch.Tensor,
    mod: torch.nn.Module,
    mod_prefix: str,
    tp_size: int,
    to_hf: Callable[[str], str],
    packed: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    _check_leaf(name, leaf, ("weight", "bias"))
    mod_any: Any = mod
    mod_tp = int(getattr(mod, "tp_size", tp_size))
    parent_pfx, mod_leaf = (
        mod_prefix.rsplit(".", 1) if "." in mod_prefix else ("", mod_prefix)
    )
    cls_name = type(mod).__name__
    if cls_name in _QKV_CLASSES:
        hs = int(mod_any.head_size)
        q_w = int(mod_any.total_num_heads) * hs
        kv_w = int(mod_any.total_num_kv_heads) * hs
        kv_rep = int(
            getattr(
                mod_any,
                "num_kv_head_replicas",
                max(1, mod_tp // int(mod_any.total_num_kv_heads)),
            )
        )
        src_names = packed.get(mod_leaf, _DEFAULT_PACKED["qkv_proj"])
        seg_sizes, reps = [q_w, kv_w, kv_w], [1, kv_rep, kv_rep]
    else:
        src_names = packed.get(mod_leaf) or _DEFAULT_PACKED.get(mod_leaf)
        if not src_names:
            raise NotImplementedError(
                f"{name!r}: MergedColumnParallelLinear {mod_leaf!r} missing from "
                "packed_modules_mapping."
            )
        seg_sizes = [int(x) for x in mod_any.output_sizes]
        reps = [1] * len(seg_sizes)

    flipped = leaf == "weight" and bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    tail = (
        [int(param.shape[0] if flipped else param.shape[1])] if leaf == "weight" else []
    )
    pfx = f"{parent_pfx}." if parent_pfx else ""
    src_keys = [to_hf(f"{pfx}{s}.{leaf}") for s in src_names]
    return _fused_col_rules(
        name, src_keys, seg_sizes, reps, mod_tp, tail, flipped=flipped
    )


def _gdn_rule(
    name: str,
    leaf: str,
    param: torch.Tensor,
    mod: torch.nn.Module,
    mod_prefix: str,
    tp_size: int,
    to_hf: Callable[[str], str],
    packed: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    mod_any: Any = mod
    layer_tp = int(getattr(mod, "tp_size", tp_size))
    parent_pfx, mod_leaf = (
        mod_prefix.rsplit(".", 1) if "." in mod_prefix else ("", mod_prefix)
    )
    packed_leaves = packed.get(mod_leaf)

    if type(mod).__name__ in _GDN_INTERLEAVED_CLASSES:
        _check_leaf(name, leaf, ("weight",))
        sizes = [int(x) for x in mod_any.group_sizes]
        if len(set(sizes)) != 1 or any(mod_any.replicated_segments):
            raise NotImplementedError(
                f"Heterogeneous/replicated GDN interleaved {mod_prefix!r} "
                "is not supported."
            )
        num_g, n_segs, seg_w = int(mod_any.geometry.num_kq_heads), len(sizes), sizes[0]
        total_seg = num_g * seg_w
        flipped = bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
        hidden = int(param.shape[0] if flipped else param.shape[1])
        ops: list[OpSpec] = [
            {
                "reshape": {
                    "src_shape": [num_g * n_segs * seg_w, hidden],
                    "dst_shape": [num_g, n_segs, seg_w, hidden],
                }
            },
            {"transpose": {"axes": [1, 0, 2, 3]}},
            {
                "reshape": {
                    "src_shape": [n_segs, num_g, seg_w, hidden],
                    "dst_shape": [n_segs * total_seg, hidden],
                }
            },
        ]
        if layer_tp > 1:
            local_seg = _exact_div(total_seg, layer_tp, f"{name} total_per_seg")
            ops.extend(_tp_interleave_ops(n_segs, layer_tp, local_seg, [hidden]))
        if flipped:
            ops.append({"transpose": {"axes": [1, 0]}})
        tp_ax = "tp" if layer_tp > 1 else None
        sharding: list[str | None] = [None, tp_ax] if flipped else [tp_ax, None]
        return [_rule([to_hf(name)], [name], ops, sharding)]

    # Segmented GDN projection (2D in_proj_qkvz or 3D conv1d)
    _check_leaf(name, leaf, ("weight", "bias"))
    sizes = [int(x) for x in mod_any.checkpoint_sizes]
    reps = [int(x) for x in mod_any.replication_factors]
    flipped = param.ndim == 2 and bool(getattr(mod, WEIGHT_FLIPPED_ATTR, False))
    if param.ndim == 1:
        tail: list[int] = []
    elif param.ndim == 2:
        tail = [int(param.shape[0] if flipped else param.shape[1])]
    elif param.ndim == 3:
        tail = [int(param.shape[1]), int(param.shape[2])]
    else:
        raise NotImplementedError(
            f"Unsupported GDN parameter rank {param.ndim} on {name!r}."
        )
    pfx = f"{parent_pfx}." if parent_pfx else ""
    if layer_tp == 1:
        if packed_leaves and len(packed_leaves) == 2 and len(sizes) == 4:
            src_keys = [to_hf(f"{pfx}{s}.{leaf}") for s in packed_leaves]
            seg_s = [sum(sizes[:3]), sizes[3]]
        elif packed_leaves and len(packed_leaves) == len(sizes):
            src_keys = [to_hf(f"{pfx}{s}.{leaf}") for s in packed_leaves]
            seg_s = sizes
        else:
            src_keys, seg_s = [to_hf(name)], [sum(sizes)]
        return _fused_col_rules(name, src_keys, seg_s, [1], 1, tail, flipped=flipped)

    rules: list[dict[str, Any]] = []
    if packed_leaves and len(packed_leaves) == 2 and len(sizes) == 4:
        qkv_src = to_hf(f"{pfx}{packed_leaves[0]}.{leaf}")
        z_src = to_hf(f"{pfx}{packed_leaves[1]}.{leaf}")
        raw = [f"__tmp__:{name}:raw{i}" for i in range(3)]
        rules.append(
            _rule([qkv_src], raw, [{"split": {"axis": 0, "sizes": sizes[:3]}}])
        )
        seg_inputs = [*raw, z_src]
    elif packed_leaves and len(packed_leaves) == len(sizes):
        seg_inputs = [to_hf(f"{pfx}{s}.{leaf}") for s in packed_leaves]
    else:
        raw = [f"__tmp__:{name}:raw{i}" for i in range(len(sizes))]
        rules.append(
            _rule([to_hf(name)], raw, [{"split": {"axis": 0, "sizes": sizes}}])
        )
        seg_inputs = raw
    rules.extend(
        _fused_col_rules(name, seg_inputs, sizes, reps, layer_tp, tail, flipped=flipped)
    )
    return rules


def _routed_experts_rule(
    name: str,
    leaf: str,
    param: torch.Tensor,
    experts: torch.nn.Module,
    mod_prefix: str,
    tp_size: int,
    ep_size: int,
    to_hf: Callable[[str], str],
    *,
    expert_axis_grouped: bool,
) -> tuple[list[dict[str, Any]], int, int]:
    qm = getattr(experts, "quant_method", None)
    moe_cfg = getattr(experts, "moe_config", None) or getattr(qm, "moe", None)
    if moe_cfg is None:
        raise NotImplementedError(
            f"RoutedExperts {mod_prefix!r} is missing moe_config."
        )
    if getattr(moe_cfg, "has_bias", False):
        raise NotImplementedError(
            f"RoutedExperts with bias ({mod_prefix!r}) is not supported."
        )
    placement = getattr(experts, "expert_placement_strategy", "linear")
    if placement != "linear":
        raise NotImplementedError(
            f"RoutedExperts {mod_prefix!r}: expert_placement_strategy={placement!r} "
            "is not supported."
        )
    num_e = int(moe_cfg.num_experts)
    num_logical = int(getattr(moe_cfg, "num_logical_experts", num_e))
    if num_e - num_logical > 0:
        raise NotImplementedError(
            f"RoutedExperts {mod_prefix!r}: redundant experts "
            f"({num_e - num_logical}) are not supported."
        )
    emm = getattr(experts, "expert_map_manager", None)
    if int(getattr(emm, "num_fused_shared_experts", 0) or 0) > 0:
        raise NotImplementedError(
            f"RoutedExperts {mod_prefix!r}: fused shared experts are not supported."
        )
    for act in (
        getattr(moe_cfg, "activation", None),
        getattr(experts, "activation", None),
        getattr(qm, "_tpu_activation_str", None),
    ):
        if act is not None and "swigluoai" in str(getattr(act, "value", act)).lower():
            raise NotImplementedError(
                f"RoutedExperts with swigluoai ({mod_prefix!r}) is not supported."
            )

    explicit_flip = getattr(experts, WEIGHT_FLIPPED_ATTR, None)
    if explicit_flip is not None:
        transposed_hint: bool | None = bool(explicit_flip)
    elif qm is not None and type(qm).__name__ == "VllmUnquantizedFusedMoEMethod":
        transposed_hint = True
    elif qm is not None and type(qm).__name__ == "UnquantizedFusedMoEMethod":
        transposed_hint = False
    else:
        transposed_hint = None

    moe_p = getattr(moe_cfg, "moe_parallel_config", None)
    use_ep = bool(getattr(moe_p, "use_ep", ep_size > 1))
    ep_layer = int(getattr(moe_cfg, "ep_size", ep_size)) if use_ep else 1
    moe_tp = int(getattr(moe_cfg, "tp_size", 1 if use_ep else tp_size))
    unpadded = int(moe_cfg.intermediate_size_per_partition)
    hidden = int(moe_cfg.hidden_dim)
    inter_total = unpadded * moe_tp
    ep_ax = "ep" if ep_layer > 1 else None
    tp_ax = "moe_tp" if moe_tp > 1 else None
    hf_pfx = to_hf(mod_prefix)

    if param.ndim != 3:
        raise ValueError(
            f"{name!r}: expected 3D expert weight, got {tuple(param.shape)}."
        )
    if param.shape[0] * ep_layer != num_e:
        raise ValueError(
            f"{name!r}: local_e={param.shape[0]} * ep={ep_layer} != {num_e}"
        )

    def _resolve_transposed(t_axis: int, nt_axis: int) -> bool:
        shape = tuple(param.shape)
        if transposed_hint is not None:
            exp_ax = t_axis if transposed_hint else nt_axis
            if shape[exp_ax] != hidden:
                raise ValueError(
                    f"{name!r}: shape {shape} != hidden={hidden} at axis {exp_ax}"
                )
            return transposed_hint
        mt, mnt = shape[t_axis] == hidden, shape[nt_axis] == hidden
        if mt and mnt:
            raise NotImplementedError(
                f"{name!r}: ambiguous square expert shape {shape} with "
                f"hidden={hidden} and no {WEIGHT_FLIPPED_ATTR!r} or quant_method."
            )
        if mt:
            return True
        if mnt:
            return False
        raise ValueError(f"{name!r}: shape {shape} does not match hidden={hidden}.")

    if leaf == "w13_weight":
        transposed = _resolve_transposed(t_axis=1, nt_axis=2)
        aligned = _exact_div(
            param.shape[2] if transposed else param.shape[1], 2, f"{name} w13 width"
        )
        pad_amt = aligned - unpadded
        if pad_amt < 0:
            raise ValueError(f"{name!r}: aligned {aligned} < unpadded {unpadded}.")
        if expert_axis_grouped:
            src_keys = [f"{hf_pfx}.gate_proj.weight", f"{hf_pfx}.up_proj.weight"]
            head_op: OpSpec = {"concat": {"axis": 0, "sizes": [num_e, num_e]}}
        else:
            src_keys = [f"{hf_pfx}.{e}.gate_proj.weight" for e in range(num_e)] + [
                f"{hf_pfx}.{e}.up_proj.weight" for e in range(num_e)
            ]
            head_op = {"stack": {"axis": 0}}
        ops: list[OpSpec] = [
            head_op,
            {
                "reshape": {
                    "src_shape": [2 * num_e, inter_total, hidden],
                    "dst_shape": [2, num_e, moe_tp, unpadded, hidden],
                }
            },
            {"transpose": {"axes": [1, 2, 0, 3, 4]}},
            *([{"pad": {"axis": 3, "sizes": [0, pad_amt]}}] if pad_amt > 0 else []),
            {
                "reshape": {
                    "src_shape": [num_e, moe_tp, 2, aligned, hidden],
                    "dst_shape": [num_e, moe_tp * 2 * aligned, hidden],
                }
            },
            *([{"transpose": {"axes": [0, 2, 1]}}] if transposed else []),
        ]
        sharding: list[str | None] = (
            [ep_ax, None, tp_ax] if transposed else [ep_ax, tp_ax, None]
        )
        return [_rule(src_keys, [name], ops, sharding)], ep_layer, moe_tp

    if leaf == "w2_weight":
        transposed = _resolve_transposed(t_axis=2, nt_axis=1)
        aligned = param.shape[1] if transposed else param.shape[2]
        pad_amt = aligned - unpadded
        if pad_amt < 0:
            raise ValueError(f"{name!r}: aligned {aligned} < unpadded {unpadded}.")
        if expert_axis_grouped:
            src_keys = [f"{hf_pfx}.down_proj.weight"]
            ops = []
        else:
            src_keys = [f"{hf_pfx}.{e}.down_proj.weight" for e in range(num_e)]
            ops = [{"stack": {"axis": 0}}]
        if pad_amt > 0:
            ops.extend(
                [
                    {
                        "reshape": {
                            "src_shape": [num_e, hidden, inter_total],
                            "dst_shape": [num_e, hidden, moe_tp, unpadded],
                        }
                    },
                    {"pad": {"axis": 3, "sizes": [0, pad_amt]}},
                    {
                        "reshape": {
                            "src_shape": [num_e, hidden, moe_tp, aligned],
                            "dst_shape": [num_e, hidden, moe_tp * aligned],
                        }
                    },
                ]
            )
        if transposed:
            ops.append({"transpose": {"axes": [0, 2, 1]}})
        sharding = [ep_ax, tp_ax, None] if transposed else [ep_ax, None, tp_ax]
        return [_rule(src_keys, [name], ops, sharding)], ep_layer, moe_tp

    raise NotImplementedError(f"Unsupported RoutedExperts param {name!r}.")


# ---------------------------------------------------------------------------
# Public entry points (model-agnostic parameter walk)
# ---------------------------------------------------------------------------


def generate_manifest_dict(
    model: torch.nn.Module,
    parallel_config: Any = None,
    model_config: Any = None,
    *,
    tp_size: int | None = None,
    dp_size: int | None = None,
    ep_size: int | None = None,
    expert_axis_grouped: bool = True,
) -> dict[str, Any]:
    """Generate a validated ``weight_mapping`` manifest dict for ``model``."""
    _check_supported(model, parallel_config, model_config)
    tp, _, ep = _resolve_parallel(model, parallel_config, tp_size, dp_size, ep_size)
    to_hf = _make_to_hf_key(model)
    packed = _collect_packed_mapping(model)
    modules_by_name = dict(model.named_modules())

    rules: list[dict[str, Any]] = []
    tied_keys: list[list[str]] = []
    seen_ids: dict[int, str] = {}
    has_moe, resolved_ep, resolved_moe_tp = False, ep, tp

    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in seen_ids:
            tied_keys.append([seen_ids[id(param)], name])
            continue
        seen_ids[id(param)] = name

        mod_prefix, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        mod = modules_by_name[mod_prefix]
        cls_name = type(mod).__name__

        if cls_name in _VOCAB_CLASSES:
            rules.extend(_embedding_rule(name, leaf, param, mod, tp, to_hf))
        elif cls_name in _QKV_CLASSES or cls_name in _MERGED_COL_CLASSES:
            rules.extend(
                _qkv_or_merged_col_rule(
                    name, leaf, param, mod, mod_prefix, tp, to_hf, packed
                )
            )
        elif cls_name in _GDN_INTERLEAVED_CLASSES or cls_name in _GDN_SEG_CLASSES:
            rules.extend(
                _gdn_rule(name, leaf, param, mod, mod_prefix, tp, to_hf, packed)
            )
        elif cls_name in _ROW_LINEAR_CLASSES:
            rules.extend(_linear_rule(name, leaf, mod, tp, to_hf, shard_dim=1))
        elif cls_name in _COL_LINEAR_CLASSES:
            rules.extend(_linear_rule(name, leaf, mod, tp, to_hf, shard_dim=0))
        elif cls_name in _REPLICATED_LINEAR_CLASSES:
            rules.extend(_linear_rule(name, leaf, mod, tp, to_hf, shard_dim=None))
        elif cls_name in _ROUTED_EXPERTS_CLASSES:
            has_moe = True
            exp_rules, resolved_ep, resolved_moe_tp = _routed_experts_rule(
                name,
                leaf,
                param,
                mod,
                mod_prefix,
                tp,
                ep,
                to_hf,
                expert_axis_grouped=expert_axis_grouped,
            )
            rules.extend(exp_rules)
        elif cls_name in _NORM_CLASSES:
            _check_leaf(name, leaf, ("weight", "bias"))
            rules.append(_rule([to_hf(name)], [name], [], [None] * param.ndim))
        elif leaf in _HEAD_SHARDED_1D_LEAVES and param.ndim == 1:
            mod_tp = int(getattr(mod, "tp_size", tp))
            rules.append(
                _rule([to_hf(name)], [name], [], ["tp" if mod_tp > 1 else None])
            )
        else:
            raise NotImplementedError(
                f"Unsupported layer {cls_name!r} for parameter {name!r}."
            )

    stripes: dict[str, int] = {"tp_size": tp}
    if has_moe:
        stripes["ep_size"] = resolved_ep
        stripes["moe_tp_size"] = resolved_moe_tp

    return validate_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "expert_axis_grouped": has_moe and expert_axis_grouped,
            "stripes": stripes,
            "tied_keys": tied_keys,
            "dropped_keys": [],
            "rules": rules,
        }
    )


def generate_manifest(
    model: torch.nn.Module,
    parallel_config: Any = None,
    model_config: Any = None,
    *,
    tp_size: int | None = None,
    dp_size: int | None = None,
    ep_size: int | None = None,
    expert_axis_grouped: bool = True,
) -> str:
    """Generate a serialized JSON ``weight_mapping`` manifest for ``model``."""
    return json.dumps(
        generate_manifest_dict(
            model,
            parallel_config=parallel_config,
            model_config=model_config,
            tp_size=tp_size,
            dp_size=dp_size,
            ep_size=ep_size,
            expert_axis_grouped=expert_axis_grouped,
        )
    )
