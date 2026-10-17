# SPDX-License-Identifier: Apache-2.0
"""StableHLO (.mlir) model export and save-load round-trip pipeline.

External entry points (called outside `vllm_torchtpu/export/`):
1. `update_hash_with_loaded_hlo(hash_obj)` — `tpu_compiler.py`
2. `maybe_export_or_load_stablehlo(...)` — `tpu_compiler.py`
3. `_install_warmup_early_exit_hook(vllm_config)` — `tpu_platform.py`
4. `register_active_model(model)` — `tpu_runner.py`
5. `maybe_export_sampling_subgraphs(runner)` — `tpu_runner.py`
"""

from __future__ import annotations

import contextlib
import copy
import ctypes
import dataclasses
import json
import os
import re
import shutil
import struct
import sys
from collections.abc import Callable, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx
from vllm.config.utils import Range

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

logger = init_logger(__name__)

_MLIR_DTYPE_TO_TORCH_NAME = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i4": "int4",
    "ui4": "uint4",
    "i8": "int8",
    "ui8": "uint8",
    "i16": "int16",
    "ui16": "uint16",
    "i32": "int32",
    "ui32": "uint32",
    "i64": "int64",
    "ui64": "uint64",
    "f8E4M3FN": "float8_e4m3fn",
    "f8E4M3FNUZ": "float8_e4m3fnuz",
    "f8E5M2": "float8_e5m2",
    "f8E5M2FNUZ": "float8_e5m2fnuz",
    "f4E2M1FN": "float4_e2m1fn",
}
_TORCH_NAME_TO_MLIR_DTYPE = {
    **{v: k for k, v in _MLIR_DTYPE_TO_TORCH_NAME.items()},
    "float4_e2m1fn_x2": "f4E2M1FN",
}
_TORCH_TENSOR_DTYPES = set(_TORCH_NAME_TO_MLIR_DTYPE.keys())

_ALLOWED_CUSTOM_CALL_TARGETS = frozenset(
    {
        "AllocateBuffer",
        "tpu_custom_call",
        "xla_tpu_custom_call",
        "Sharding",
        "SPMDFullToShardShape",
        "SPMDShardToFullShape",
        "pin_to_device",
        "mark_tensor",
    }
)

_TORCH_TO_SAFETENSORS_DTYPE: dict[torch.dtype, str] = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.bool: "BOOL",
}
for _fp8_attr, _st_code in (
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e5m2", "F8_E5M2"),
):
    if hasattr(torch, _fp8_attr):
        _TORCH_TO_SAFETENSORS_DTYPE[getattr(torch, _fp8_attr)] = _st_code

# `@main(<args>) -> <results>` of a StableHLO module.
_MAIN_SIGNATURE_RE = re.compile(
    r"func\.func\s+(?:public\s+)?@main\s*\((.*?)\)(?:\s*->\s*([^\{\n]+))?",
    re.DOTALL,
)

_ACTIVE_PARAM_MAP: dict[int, str] = {}
_ACTIVE_MODEL: Any = None
_CAPTURED_CONSTANTS: dict[str, torch.Tensor] = {}


def update_hash_with_loaded_hlo(hash_obj: Any) -> None:
    """Fold `VLLM_TORCHTPU_LOAD_HLO_DIR` `.mlir` and metadata into compilation hash."""
    export_hlo_dir = str(envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or "").strip()
    if export_hlo_dir:
        hash_obj.update(f"export:{export_hlo_dir}".encode())
    load_hlo_dir = str(envs.VLLM_TORCHTPU_LOAD_HLO_DIR or "").strip()
    if not load_hlo_dir:
        return
    load_dir = Path(load_hlo_dir)
    if not load_dir.is_dir():
        return
    for hlo_path in sorted(
        [
            *load_dir.glob("*.mlir"),
            load_dir / "metadata.json",
            load_dir / "serving_checkpoint" / "weight.json",
        ]
    ):
        if hlo_path.is_file():
            hash_obj.update(hlo_path.name.encode())
            with contextlib.suppress(OSError):
                hash_obj.update(hlo_path.read_bytes())


def register_active_model(model: Any) -> None:
    """Register an active `torch.nn.Module` for parameter name resolution."""
    global _ACTIVE_PARAM_MAP, _ACTIVE_MODEL
    _ACTIVE_PARAM_MAP.clear()
    _ACTIVE_MODEL = model
    if model is None:
        return
    for iter_fn in ("named_parameters", "named_buffers"):
        if hasattr(model, iter_fn):
            for name, t in getattr(model, iter_fn)():
                if iter_fn == "named_buffers" and any(
                    s in name.lower() for s in ("cos_sin_cache", "cos_sin_cache_bf16")
                ):
                    continue
                if isinstance(t, torch.Tensor):
                    with contextlib.suppress(Exception):
                        _ACTIVE_PARAM_MAP[t.data_ptr()] = name
                        _ACTIVE_PARAM_MAP[t.untyped_storage().data_ptr()] = name


def _dtype_name(val: Any) -> str:
    dtype = getattr(val, "dtype", None)
    if dtype is not None:
        return str(dtype).removeprefix("torch.")
    if isinstance(val, torch.dtype):
        return str(val).removeprefix("torch.")
    return type(val).__name__


def _shape_to_list(val: Any) -> list[int | str]:
    shape_raw = getattr(val, "shape", None)
    if shape_raw is None:
        return []
    out: list[int | str] = []
    for dim in shape_raw:
        try:
            out.append(int(dim))
        except Exception:
            out.append(str(dim))
    return out


def _extract_sharding_spec(node: fx.Node | None, val: Any) -> Any:
    if node is not None:
        for meta_key in ("sharding", "sharding_spec", "partition_spec"):
            raw = node.meta.get(meta_key)
            if raw is not None:
                if isinstance(raw, (list, tuple)):
                    return [str(x) if x is not None else None for x in raw]
                return str(raw)
    sharding = getattr(val, "sharding", None)
    if sharding is not None:
        if isinstance(sharding, (list, tuple)):
            return [str(x) if x is not None else None for x in sharding]
        return str(sharding)
    return None


def _resolve_submod_and_param(
    weight_name: str, model: Any = None
) -> tuple[Any, Any, str, str]:
    """Resolves `(submod, param_or_buf, mod_path, attr)` from `model`."""
    if model is None or not weight_name:
        return None, None, "", weight_name
    mod_path, attr = (
        weight_name.rsplit(".", 1) if "." in weight_name else ("", weight_name)
    )
    submod = None
    if mod_path:
        for cand_mod_path in (mod_path, mod_path.removeprefix("model.")):
            with contextlib.suppress(Exception):
                submod = model.get_submodule(cand_mod_path)
                if submod is not None:
                    break
    else:
        submod = model
    param_obj = getattr(submod, attr, None) if submod is not None else None
    return submod, param_obj, mod_path, attr


def _infer_weight_sharding(
    weight_name: str,
    shape: Sequence[int | str],
    model: Any = None,
    explicit_sharding: Any = None,
    tp_size_hint: int = 1,
    has_ep_hint: bool = False,
) -> list[str | None] | str:
    """Determines per-tensor mesh sharding from module/param attributes and name."""
    if explicit_sharding is not None:
        if isinstance(explicit_sharding, (list, tuple)):
            return [str(x) if x is not None else None for x in explicit_sharding]
        s_str = str(explicit_sharding).strip()
        if s_str and s_str not in ("None", "{replicated}", "replicated"):
            return s_str

    rank = len(shape)
    submod, param_obj, _, attr = _resolve_submod_and_param(weight_name, model)
    submod_tp = int(getattr(submod, "tp_size", 0) or 0) if submod is not None else 0
    tp_size = max(submod_tp, int(tp_size_hint or 1))

    if rank == 0:
        if any(
            weight_name.endswith(s)
            for s in (
                "org_vocab_start_index",
                "org_vocab_end_index",
                "added_vocab_start_index",
                "added_vocab_end_index",
                "num_org_vocab_padding",
            )
        ):
            return "tp" if tp_size > 1 else []
        if weight_name.endswith("experts_start"):
            return "ep" if (has_ep_hint or tp_size > 1) else []
        return []

    if param_obj is not None:
        for shard_attr, axis_name in (
            ("ep_dim", "ep"),
            ("tp_dim", "tp"),
            ("output_dim", "tp"),
            ("input_dim", "tp"),
        ):
            dim_val = getattr(param_obj, shard_attr, None)
            if isinstance(dim_val, int) and 0 <= dim_val < rank:
                if axis_name == "tp" and tp_size <= 1:
                    continue
                sharding_dims: list[str | None] = [None] * rank
                if shard_attr == "output_dim" and rank == 2 and attr == "weight":
                    # Linear weights are transposed [in_dim, out_dim] on TPU
                    is_emb = hasattr(submod, "num_embeddings") or (
                        submod is not None
                        and "embedding" in type(submod).__name__.lower()
                    )
                    target_dim = 0 if is_emb else (1 - dim_val)
                    sharding_dims[target_dim] = "tp"
                elif shard_attr == "input_dim" and rank == 2 and attr == "weight":
                    target_dim = 1 - dim_val
                    sharding_dims[target_dim] = "tp"
                else:
                    sharding_dims[dim_val] = axis_name
                return sharding_dims

    if ".routed_experts." in weight_name:
        if rank >= 3:
            ep_axis = "ep" if (has_ep_hint or tp_size > 1) else None
            return [ep_axis] + [None] * (rank - 1)
        return [None] * rank

    if tp_size <= 1:
        return [None] * rank

    if rank == 1:
        if any(
            k in weight_name
            for k in (
                "qkv_proj",
                "gate_up_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "wq_b",
                "q_b_proj",
                "wo_a",
                "in_proj_qkvz",
                "in_proj_ba",
                "a_log",
                "dt_bias",
                "fused_qkvb_proj",
                "fused_fa_ga_proj",
                "f_b_proj",
                "g_b_proj",
                "attn_sink",
            )
        ) and not any(
            weight_name.endswith(s)
            for s in (
                "o_proj.weight_scale",
                "down_proj.weight_scale",
                "out_proj.weight_scale",
                "wo_b.weight_scale",
            )
        ):
            return ["tp"]
        return [None]

    if rank == 2:
        if (
            weight_name.endswith("embed_tokens.weight")
            or weight_name == "lm_head.weight"
            or weight_name.endswith(".lm_head.weight")
            or weight_name.endswith(".o_proj.weight")
            or weight_name.endswith(".down_proj.weight")
            or weight_name.endswith(".out_proj.weight")
            or weight_name.endswith(".wo_b.weight")
            or ".in_proj_qkvz." in weight_name
        ):
            return ["tp", None]
        if any(
            k in weight_name
            for k in (
                "qkv_proj",
                "gate_up_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "wq_b",
                "q_b_proj",
                "wo_a",
                "in_proj_ba",
                "fused_qkvb_proj",
                "fused_fa_ga_proj",
                "f_b_proj",
                "g_b_proj",
                "compressor.fused_wkv_wgate",
            )
        ):
            return [None, "tp"]
        return [None, None]

    if rank == 3:
        if (
            ".conv1d.weight" in weight_name
            or ".w_uk_t" in weight_name
            or ".w_uv" in weight_name
        ):
            return ["tp", None, None]
        return [None, None, None]

    if rank == 4:
        if ".conv_weight_fused" in weight_name:
            return [None, None, "tp", None]
        return [None, None, None, None]

    return [None] * rank


def _ensure_tuple_output(graph: fx.GraphModule) -> tuple[fx.GraphModule, bool]:
    out = next((n for n in graph.graph.nodes if n.op == "output"), None)
    if out is None or isinstance(out.args[0], (tuple, list)):
        return graph, False
    with graph.graph.inserting_before(out):
        out.args = ((out.args[0],),)
    graph.graph.lint()
    graph.recompile()
    return graph, True


def _parse_subgraph_index(key: str | None) -> int:
    if not key:
        return 0
    for pat in (r"(?:subgraph|submod)_(\d+)", r"_(\d+)$"):
        m = re.search(pat, key)
        if m is not None:
            return int(m.group(1))
    return 0


def _is_export_writer_rank(runner: Any = None) -> bool:
    for raw in (os.environ.get("RANK"), os.environ.get("LOCAL_RANK")):
        if raw is not None and str(raw).strip() not in ("", "0"):
            try:
                if int(str(raw).strip()) != 0:
                    return False
            except ValueError:
                pass
    if runner is not None:
        for attr in ("rank", "local_rank"):
            val = getattr(runner, attr, None)
            if isinstance(val, int) and val != 0:
                return False
        pc = getattr(runner, "parallel_config", None)
        if pc is not None:
            dp_rank = getattr(pc, "data_parallel_rank", None)
            if isinstance(dp_rank, int) and dp_rank != 0:
                return False
    return True


def _is_draft_compilation(adaptor: Any = None) -> bool:
    cache_dir = str(getattr(adaptor, "cache_dir", "") or "")
    if "eagle_head" in cache_dir or "mtp_draft" in cache_dir:
        return True
    with contextlib.suppress(Exception):
        from vllm.compilation.wrapper import get_model_tag

        if get_model_tag() == "eagle_head":
            return True
    with contextlib.suppress(Exception):
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        if getattr(getattr(cfg, "model_config", None), "runner_type", None) == "draft":
            return True
    return False


def _compute_fn_key(
    compile_range: Range, subgraph_idx: int, adaptor: Any = None
) -> str:
    prefix = "mtp_draft" if _is_draft_compilation(adaptor) else "forward"
    if subgraph_idx == 0:
        return f"{prefix}_bucket{compile_range.start}"
    return f"{prefix}_piece{subgraph_idx}_bucket{compile_range.start}"


def _split_fn_key(fn_key: str) -> tuple[str, str]:
    if fn_key == "abstract_params":
        return "abstract_params", "default"
    m = re.match(r"^((?:forward|mtp_draft)(?:_piece\d+)?)_bucket(\d+)$", fn_key)
    if m:
        return m.group(1), m.group(2)
    for prefix in ("compute_logits", "sample_top_tokens"):
        if fn_key.startswith(f"{prefix}_req"):
            return prefix, fn_key.removeprefix(f"{prefix}_req")
    return fn_key, "default"


def _get_function_entry(metadata: dict[str, Any], fn_key: str) -> dict[str, Any] | None:
    fn_name, variant_key = _split_fn_key(fn_key)
    functions = metadata.get("functions")
    if isinstance(functions, dict):
        variants = functions.get(fn_name)
        if isinstance(variants, dict):
            entry = variants.get(variant_key)
            if isinstance(entry, dict):
                return entry
    return None


def _iter_forward_entries(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    functions = metadata.get("functions")
    if isinstance(functions, dict):
        for fn_name, variants in functions.items():
            if fn_name.startswith("forward") and isinstance(variants, dict):
                for entry in variants.values():
                    if isinstance(entry, dict):
                        entries.append(entry)
    return entries


def _get_target_hardware_fingerprint() -> dict[str, str | None]:
    fingerprint: dict[str, str | None] = {"chip_type": envs.TPU_ACCELERATOR_TYPE}
    for pkg_key, dist_name in (
        ("torch_tpu", "torch-tpu"),
        ("jaxlib", "jaxlib"),
        ("libtpu", "libtpu"),
    ):
        try:
            fingerprint[pkg_key] = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            fingerprint[pkg_key] = None
    return fingerprint


def _parse_mlir_tensor_type(tensor_body: str) -> tuple[list[int | str], str]:
    core_body = tensor_body.split(",", 1)[0].strip()
    parts = core_body.split("x")
    elem = parts[-1]
    shape = [
        "?" if dim == "?" else int(dim)
        for dim in parts[:-1]
        if dim == "?" or dim.isdigit()
    ]
    return shape, _MLIR_DTYPE_TO_TORCH_NAME.get(elem, elem)


def _parse_mlir_main_args(
    args_str: str,
) -> list[tuple[list[int | str], str, bool, str | None]]:
    out: list[tuple[list[int | str], str, bool, str | None]] = []
    for tensor_body, attr_str in re.findall(
        r"tensor<([^>]+)>(\s*\{(?:[^{}\"]|\"[^\"]*\"|\{[^}]*\})*\})?",
        args_str,
        re.DOTALL,
    ):
        shape, dtype = _parse_mlir_tensor_type(tensor_body)
        is_donor = bool(
            attr_str
            and re.search(r"(?:buffer_donor\s*=\s*true|tf\.aliasing_output)", attr_str)
        )
        m_sharding = (
            re.search(r"mhlo\.sharding\s*=\s*\"([^\"]+)\"", attr_str)
            if attr_str
            else None
        )
        mlir_sharding = m_sharding.group(1) if m_sharding is not None else None
        out.append((shape, dtype, is_donor, mlir_sharding))
    return out


def _resolve_dynamo_placeholder(
    raw_name: str, model: Any = None
) -> tuple[str, bool, str, bool]:
    """Resolve a TorchDynamo placeholder into (name, is_weight, arg_type, is_const)."""
    if not raw_name:
        return "", False, "user_io", False
    m = re.match(
        r"^l_self_(?:modules_(.+)_)?(parameters|buffers)_([A-Za-z0-9_]+?)_?$",
        raw_name,
        re.IGNORECASE,
    )
    if m:
        mod_tokens = m.group(1) or ""
        kind = m.group(2).lower()
        attr = m.group(3)
        sub_path = re.sub(r"_modules_", ".", mod_tokens, flags=re.IGNORECASE)
        if sub_path:
            candidates = [f"model.{sub_path}.{attr}", f"{sub_path}.{attr}"]
        else:
            candidates = [f"model.{attr}", attr]
        canonical = candidates[0]
        if model is not None:
            known = set()
            for fn in ("named_parameters", "named_buffers"):
                if hasattr(model, fn):
                    with contextlib.suppress(Exception):
                        known.update(k for k, _ in getattr(model, fn)())
            for cand in candidates:
                if cand in known:
                    canonical = cand
                    break
        is_const = kind == "buffers"
        if is_const and attr.lower() in ("cos_sin_cache", "cos_sin_cache_bf16"):
            return "rotary_emb_inv_freq", False, "constant", True
        return canonical, True, "weight", is_const

    lower_name = raw_name.lower()
    if any(
        s in lower_name for s in ("kv_cache", "conv_state", "ssm_state", "mamba_cache")
    ):
        m_layer = re.search(r"layers_(?:modules_)?(\d+)", lower_name)
        layer_suffix = m_layer.group(1) if m_layer else "0"
        return f"kv_cache_layer_{layer_suffix}", False, "state", False

    if lower_name.startswith("g_import_vllm") or "attn_metadata" in lower_name:
        clean_lower = lower_name.rstrip("_")
        for suffix, canonical in (
            ("seq_lens", "kv_lens"),
            ("kv_lens", "kv_lens"),
            ("block_tables", "block_tables"),
            ("query_start_loc", "query_start_loc"),
            ("request_distribution", "req_dist"),
            ("req_dist", "req_dist"),
        ):
            if clean_lower.endswith(suffix):
                return canonical, False, "user_io", False

    m_local = re.match(r"^l_([A-Za-z0-9_]+?)_?$", raw_name, re.IGNORECASE)
    if m_local:
        return m_local.group(1), False, "user_io", False

    return raw_name, False, "user_io", False


def _build_generic_weight_sources(
    weight_name: str, shape: list[int | str], model: Any = None
) -> list[dict[str, Any]] | None:
    """Build declarative `sources` from `model` and `packed_modules_mapping`."""
    if not weight_name or any(not isinstance(d, int) for d in shape):
        return None
    int_shape = [int(d) for d in shape if isinstance(d, int)]

    if model is not None and "." in weight_name:
        mod_path, attr = weight_name.rsplit(".", 1)
        submod = None
        for cand_mod_path in (mod_path, mod_path.removeprefix("model.")):
            with contextlib.suppress(Exception):
                submod = model.get_submodule(cand_mod_path)
                if submod is not None:
                    break
        if submod is not None:
            parent_path, leaf_mod_name = (
                mod_path.rsplit(".", 1) if "." in mod_path else ("", mod_path)
            )
            packed_map = getattr(model, "packed_modules_mapping", None) or {}
            sub_names = packed_map.get(leaf_mod_name)
            output_sizes = getattr(submod, "output_sizes", None)
            if (
                isinstance(sub_names, (list, tuple))
                and isinstance(output_sizes, (list, tuple))
                and len(sub_names) == len(output_sizes)
            ):
                tp_size = int(getattr(submod, "tp_size", 1) or 1)
                local_sizes = [int(s) // max(tp_size, 1) for s in output_sizes]
                if attr == "weight" and len(int_shape) == 2:
                    if sum(local_sizes) == int_shape[1]:
                        in_dim = int_shape[0]
                        sources: list[dict[str, Any]] = []
                        offset = 0
                        for sub_n, out_sz in zip(sub_names, local_sizes):
                            src_prefix = (
                                f"{parent_path}.{sub_n}" if parent_path else sub_n
                            )
                            sources.append(
                                {
                                    "path": "",
                                    "source_tensor": f"{src_prefix}.{attr}",
                                    "source_shape": [out_sz, in_dim],
                                    "transpose": [1, 0],
                                    "dst_slices": [
                                        {"dim": 1, "start": offset, "length": out_sz}
                                    ],
                                }
                            )
                            offset += out_sz
                        return sources
                    if sum(local_sizes) == int_shape[0]:
                        in_dim = int_shape[1]
                        sources = []
                        offset = 0
                        for sub_n, out_sz in zip(sub_names, local_sizes):
                            src_prefix = (
                                f"{parent_path}.{sub_n}" if parent_path else sub_n
                            )
                            sources.append(
                                {
                                    "path": "",
                                    "source_tensor": f"{src_prefix}.{attr}",
                                    "source_shape": [out_sz, in_dim],
                                    "dst_slices": [
                                        {"dim": 0, "start": offset, "length": out_sz}
                                    ],
                                }
                            )
                            offset += out_sz
                        return sources
                elif len(int_shape) == 1 and sum(local_sizes) == int_shape[0]:
                    sources = []
                    offset = 0
                    for sub_n, out_sz in zip(sub_names, local_sizes):
                        src_prefix = f"{parent_path}.{sub_n}" if parent_path else sub_n
                        sources.append(
                            {
                                "path": "",
                                "source_tensor": f"{src_prefix}.{attr}",
                                "source_shape": [out_sz],
                                "dst_slices": [
                                    {"dim": 0, "start": offset, "length": out_sz}
                                ],
                            }
                        )
                        offset += out_sz
                    return sources

            if (
                attr == "weight"
                and len(int_shape) == 2
                and not hasattr(submod, "num_embeddings")
                and "embedding" not in type(submod).__name__.lower()
                and (
                    hasattr(submod, "input_size")
                    or hasattr(submod, "input_size_per_partition")
                    or hasattr(submod, "in_features")
                )
            ):
                return [
                    {
                        "path": "",
                        "source_tensor": weight_name,
                        "source_shape": [int_shape[1], int_shape[0]],
                        "transpose": [1, 0],
                    }
                ]

    return [{"path": "", "source_tensor": weight_name, "source_shape": int_shape}]


def _extract_input_specs(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    outer_graph: fx.GraphModule | None = None,
    outer_inputs: list[Any] | None = None,
) -> list[dict[str, Any]]:
    placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
    outer_phs = (
        [node for node in outer_graph.graph.nodes if node.op == "placeholder"]
        if outer_graph is not None and outer_graph is not graph
        else []
    )
    use_outer = len(outer_phs) == len(placeholders) > 0

    specs: list[dict[str, Any]] = []
    idx = 0
    for i in range(max(len(placeholders), len(example_inputs))):
        node = placeholders[i] if i < len(placeholders) else None
        outer_node = None
        outer_val = None
        if node is not None and outer_phs:
            m_arg = re.match(
                r"^arg(\d+)_\d+$",
                str(node.name or node.target or ""),
                re.IGNORECASE,
            )
            if m_arg is not None:
                outer_idx = int(m_arg.group(1))
                if 0 <= outer_idx < len(outer_phs):
                    outer_node = outer_phs[outer_idx]
                    if outer_inputs is not None and outer_idx < len(outer_inputs):
                        outer_val = outer_inputs[outer_idx]
        if outer_node is None and use_outer and i < len(outer_phs):
            outer_node = outer_phs[i]
            if outer_inputs is not None and i < len(outer_inputs):
                outer_val = outer_inputs[i]
        val = example_inputs[i] if i < len(example_inputs) else None
        if val is None and node is not None:
            val = node.meta.get("example_value", node.meta.get("val"))
        if val is None and outer_node is not None:
            val = outer_node.meta.get("example_value", outer_node.meta.get("val"))
        if val is None and outer_val is not None:
            val = outer_val
        if (
            val is None
            or isinstance(val, (int, float, bool, str))
            or type(val).__name__ == "SymInt"
            or getattr(val, "shape", None) is None
            or _dtype_name(val) not in _TORCH_TENSOR_DTYPES
        ):
            continue

        is_weight, weight_name = False, ""
        arg_type = "user_io"
        is_const = False

        if outer_node is not None:
            cand_name, cand_w, cand_type, cand_const = _resolve_dynamo_placeholder(
                str(outer_node.name or outer_node.target or ""), _ACTIVE_MODEL
            )
            if cand_w:
                is_weight, weight_name, arg_type, is_const = (
                    True,
                    cand_name,
                    "weight",
                    cand_const,
                )
            elif cand_name:
                arg_type = cand_type
                is_const = cand_const

        if not is_weight and node is not None:
            cand_name, cand_w, cand_type, cand_const = _resolve_dynamo_placeholder(
                str(node.name or node.target or ""), _ACTIVE_MODEL
            )
            if cand_w:
                is_weight, weight_name, arg_type, is_const = (
                    True,
                    cand_name,
                    "weight",
                    cand_const,
                )
            elif cand_type != "user_io":
                arg_type = cand_type
                is_const = cand_const

        if not is_weight and node is not None:
            desc = node.meta.get("desc")
            if (
                desc is not None
                and (
                    getattr(desc, "is_param", False)
                    or getattr(desc, "is_buffer", False)
                    or type(desc).__name__ in ("ParamAOTInput", "BufferAOTInput")
                )
                and getattr(desc, "target", None)
            ):
                is_weight, weight_name, arg_type = True, str(desc.target), "weight"
                is_const = bool(
                    getattr(desc, "is_buffer", False)
                    or type(desc).__name__ == "BufferAOTInput"
                )
            if not is_weight:
                for k in ("param_name", "parameter_name", "fwd_bwd_target"):
                    if node.meta.get(k):
                        is_weight, weight_name, arg_type = (
                            True,
                            str(node.meta[k]),
                            "weight",
                        )
                        break
            if not is_weight:
                target = str(node.target or "")
                name = str(node.name or "")
                if hasattr(graph, target) and isinstance(
                    getattr(graph, target), (torch.nn.Parameter, torch.Tensor)
                ):
                    is_weight, weight_name, arg_type = True, target, "weight"
                elif target.startswith("p_") or name.startswith("p_"):
                    is_weight, weight_name, arg_type = (
                        True,
                        (target[2:] if target.startswith("p_") else name[2:]),
                        "weight",
                    )
        if not is_weight and _ACTIVE_PARAM_MAP:
            for cand_v in (
                val,
                outer_val,
                outer_inputs[i]
                if outer_inputs is not None and i < len(outer_inputs)
                else None,
            ):
                if (
                    cand_v is not None
                    and isinstance(cand_v, torch.Tensor)
                    and cand_v.__class__.__name__ != "FakeTensor"
                ):
                    with contextlib.suppress(Exception):
                        for ptr in (
                            cand_v.data_ptr(),
                            cand_v.untyped_storage().data_ptr(),
                        ):
                            if ptr in _ACTIVE_PARAM_MAP:
                                is_weight, weight_name, arg_type = (
                                    True,
                                    _ACTIVE_PARAM_MAP[ptr],
                                    "weight",
                                )
                                break
                if is_weight:
                    break
        if not is_weight and node is not None:
            name = str(node.name or "")
            if any(
                s in name.lower()
                for s in (
                    "_weight",
                    "_bias",
                    ".weight",
                    ".bias",
                    "_scale",
                    ".scale",
                    "w_q",
                    "w_k",
                    "w_v",
                    "w_o",
                    "w13_weight",
                    "w2_weight",
                    "experts",
                    "e_score_correction_bias",
                    "a_log",
                    "dt_bias",
                    "embed_tokens",
                    "lm_head",
                )
            ):
                is_weight, weight_name, arg_type = True, name, "weight"
        if not is_weight and (
            isinstance(val, torch.nn.Parameter) or getattr(val, "_is_param", False)
        ):
            is_weight, weight_name, arg_type = (
                True,
                str(getattr(val, "_param_name", getattr(val, "name", ""))),
                "weight",
            )

        resolved_node_name = node.name if node is not None else f"arg_{idx}"
        if outer_node is not None:
            cand_name, _, _, _ = _resolve_dynamo_placeholder(
                str(outer_node.name or outer_node.target or ""), _ACTIVE_MODEL
            )
            if cand_name:
                resolved_node_name = cand_name
        elif is_weight and weight_name:
            resolved_node_name = weight_name

        shape_list = _shape_to_list(val)
        raw_sharding = _extract_sharding_spec(node or outer_node, val)
        if is_weight and weight_name:
            resolved_sharding: Any = _infer_weight_sharding(
                weight_name,
                shape_list,
                model=_ACTIVE_MODEL,
                explicit_sharding=raw_sharding,
            )
        else:
            resolved_sharding = raw_sharding
        spec_entry: dict[str, Any] = {
            "arg_index": idx,
            "index": idx,
            "name": weight_name if is_weight and weight_name else resolved_node_name,
            "shape": shape_list,
            "dtype": _dtype_name(val),
            "sharding": resolved_sharding,
            "argument_type": "weight" if is_weight else arg_type,
            "is_weight": is_weight,
            "weight_name": weight_name,
        }
        if is_const:
            spec_entry["is_internal_constant"] = True
        if is_weight and weight_name:
            sources = _build_generic_weight_sources(
                weight_name, shape_list, _ACTIVE_MODEL
            )
            if sources:
                spec_entry["sources"] = sources

        specs.append(spec_entry)
        idx += 1

    return specs


def _extract_output_specs(
    graph: fx.GraphModule, exported_mlir: Any | None = None
) -> list[dict[str, Any]]:
    result_tensors = getattr(exported_mlir, "mlir_result_tensors", None)
    if result_tensors:
        return [
            {
                "arg_index": idx,
                "index": idx,
                "name": f"output_{idx}",
                "shape": _shape_to_list(val),
                "dtype": _dtype_name(val),
                "sharding": _extract_sharding_spec(None, val),
                "argument_type": "user_io",
                "buffer_donor": False,
            }
            for idx, val in enumerate(result_tensors)
        ]

    out_node = next((n for n in graph.graph.nodes if n.op == "output"), None)
    if out_node is None:
        return []
    raw_args = out_node.args[0] if out_node.args else ()
    items = raw_args if isinstance(raw_args, (tuple, list)) else (raw_args,)
    specs: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        node = item if isinstance(item, fx.Node) else None
        val = (
            node.meta.get("example_value", node.meta.get("val"))
            if node is not None
            else None
        )
        if val is None:
            val = out_node.meta.get("example_value", out_node.meta.get("val"))
            if isinstance(val, (tuple, list)) and idx < len(val):
                val = val[idx]
        specs.append(
            {
                "arg_index": idx,
                "index": idx,
                "name": node.name if node is not None else f"output_{idx}",
                "shape": _shape_to_list(val),
                "dtype": _dtype_name(val),
                "sharding": _extract_sharding_spec(node or out_node, val),
                "argument_type": "user_io",
                "buffer_donor": False,
            }
        )
    return specs


def _constant_for_spec(
    name: str, tensor: torch.Tensor, spec: dict[str, Any]
) -> torch.Tensor:
    """Returns `tensor` on the host in the dtype its `@main` argument declares.

    A buffer can live in a wider dtype than the graph consumes (vLLM keeps
    `cos_sin_cache` in float32 and the forward graph takes it in the model
    dtype), so the stored constant follows the argument, not the buffer.
    """
    shape = [int(d) for d in spec.get("shape", [])]
    if [int(d) for d in tensor.shape] != shape:
        raise ValueError(
            f"Constant '{name}' has shape {list(tensor.shape)}, but its @main "
            f"argument is {shape}"
        )
    dtype = getattr(torch, str(spec["dtype"]))
    return tensor.detach().to(device="cpu", dtype=dtype).contiguous()


def _write_abstract_params(
    export_dir: Path, metadata: dict[str, Any], weights_dict: dict[str, Any]
) -> None:
    """Emit `abstract_params.mlir` and register `functions.abstract_params.default`."""
    sorted_weights = sorted(
        (
            item
            for item in weights_dict.items()
            if item[0] != "mesh" and isinstance(item[1], dict) and "shape" in item[1]
        ),
        key=lambda item: (int(item[1].get("index", 0)), item[0]),
    )
    arg_decls: list[str] = []
    ret_types: list[str] = []
    ret_vals: list[str] = []
    abs_inputs: list[dict[str, Any]] = []
    abs_outputs: list[dict[str, Any]] = []
    for arg_idx, (w_name, w_spec) in enumerate(sorted_weights):
        shape = [int(d) for d in w_spec.get("shape", []) if isinstance(d, int)]
        dtype_str = str(w_spec.get("dtype", "bfloat16"))
        mlir_dtype = _TORCH_NAME_TO_MLIR_DTYPE.get(dtype_str, "bf16")
        dims_str = "x".join(str(d) for d in shape)
        tensor_type = (
            f"tensor<{dims_str}x{mlir_dtype}>" if dims_str else f"tensor<{mlir_dtype}>"
        )
        arg_decls.append(f"%arg{arg_idx}: {tensor_type} {{jax.buffer_donor = true}}")
        ret_types.append(tensor_type)
        ret_vals.append(f"%arg{arg_idx}")
        global_idx = int(w_spec.get("index", arg_idx))
        abs_inputs.append(
            {
                "arg_index": arg_idx,
                "index": global_idx,
                "name": w_name,
                "shape": shape,
                "dtype": dtype_str,
                "sharding": w_spec.get("sharding"),
                "argument_type": "weight",
                "is_weight": True,
                "weight_name": w_name,
                "buffer_donor": True,
            }
        )
        abs_outputs.append(
            {
                "arg_index": arg_idx,
                "index": global_idx,
                "name": f"out_{arg_idx}",
                "shape": shape,
                "dtype": dtype_str,
                "sharding": w_spec.get("sharding"),
                "argument_type": "weight",
                "buffer_donor": False,
            }
        )

    args_joined = ", ".join(arg_decls)
    rets_joined = ", ".join(ret_types)
    vals_joined = ", ".join(ret_vals)
    if ret_types:
        ret_sig = f"({rets_joined})"
        body_line = f"    return {vals_joined} : {rets_joined}\n"
    else:
        ret_sig = "()"
        body_line = "    return\n"

    mlir_text = (
        "module @jit_abstract_params {\n"
        f"  func.func public @main({args_joined}) -> {ret_sig} {{\n"
        f"{body_line}"
        "  }\n"
        "}\n"
    )
    abs_path = export_dir / "abstract_params.mlir"
    tmp_abs_path = export_dir / f"abstract_params.mlir.{os.getpid()}.tmp"
    tmp_abs_path.write_text(mlir_text)
    os.replace(tmp_abs_path, abs_path)
    metadata.setdefault("functions", {})["abstract_params"] = {
        "default": {
            "file_path": "abstract_params.mlir",
            "subgraph_index": 0,
            "compile_size": 0,
            "target_hardware": _get_target_hardware_fingerprint(),
            "inputs": abs_inputs,
            "outputs": abs_outputs,
        }
    }


def _write_exported_module(
    export_dir: Path,
    fn_key: str,
    subgraph_idx: int,
    compile_size: int,
    mlir_text: str,
    input_specs: list[dict[str, Any]],
    output_specs: list[dict[str, Any]],
) -> Path:
    """Persist `<fn_key>.mlir`, `metadata.json`, and `weight.json`."""
    export_dir.mkdir(parents=True, exist_ok=True)
    mlir_filename = f"{fn_key}.mlir"
    mlir_path = export_dir / mlir_filename
    tmp_mlir_path = export_dir / f"{mlir_filename}.{os.getpid()}.tmp"
    tmp_mlir_path.write_text(mlir_text)
    os.replace(tmp_mlir_path, mlir_path)

    metadata_path = export_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupt metadata.json in {export_dir}: {exc}") from exc

    serving_ckpt_dir = export_dir / "serving_checkpoint"
    wj_path = serving_ckpt_dir / "weight.json"
    existing_weights: dict[str, Any] = {}
    if wj_path.is_file():
        with contextlib.suppress(Exception):
            raw_wj = json.loads(wj_path.read_text())
            if isinstance(raw_wj, dict):
                existing_weights = raw_wj.get("weights", raw_wj)
                existing_weights.pop("hf_checkpoint_path", None)
                existing_weights.pop("checkpoint_path", None)
                existing_weights.pop("mesh", None)
    used_weight_indices: set[int] = {
        int(v["index"])
        for v in existing_weights.values()
        if isinstance(v, dict) and isinstance(v.get("index"), int)
    }

    annotated_inputs = [dict(inp) for inp in input_specs]
    for arg_idx, spec in enumerate(annotated_inputs):
        spec.setdefault("arg_index", arg_idx)
        spec.setdefault("index", arg_idx)
        spec.setdefault(
            "argument_type", "weight" if spec.get("is_weight") else "user_io"
        )
        spec.setdefault("buffer_donor", False)

    match = re.search(
        r"func\.func\s+(?:public\s+)?@main\s*\((.*?)\)\s*->", mlir_text, re.DOTALL
    )
    if match is not None:
        mlir_args = _parse_mlir_main_args(match.group(1))
        if len(mlir_args) == len(annotated_inputs):
            for spec, (m_shape, m_dtype, is_donor, m_sharding) in zip(
                annotated_inputs, mlir_args
            ):
                if "?" not in m_shape and "?" not in (spec.get("shape") or []):
                    spec["shape"] = m_shape
                if m_dtype in _TORCH_TENSOR_DTYPES:
                    spec["dtype"] = m_dtype
                if spec.get("sharding") is None and m_sharding is not None:
                    spec["sharding"] = m_sharding
                spec["buffer_donor"] = is_donor
                if (
                    is_donor
                    and not spec.get("is_weight")
                    and spec.get("argument_type") == "user_io"
                ):
                    spec["argument_type"] = "state"

    state_counter = 0
    for spec in annotated_inputs:
        if spec.get("argument_type") == "state":
            if spec.get("index") == spec.get("arg_index"):
                spec["index"] = state_counter
            state_counter += 1

    num_donors = sum(1 for spec in annotated_inputs if spec.get("buffer_donor"))
    if (
        fn_key.startswith("forward")
        and num_donors > 0
        and len(output_specs) > num_donors
    ):
        output_specs = [dict(out) for out in output_specs]
        for out_idx, out_spec in enumerate(output_specs):
            if out_idx < num_donors:
                if out_spec.get("name") == f"output_{out_idx}":
                    out_spec["name"] = f"kv_cache_out_layer_{out_idx}"
                out_spec["argument_type"] = "state"
            elif out_idx == len(output_specs) - 1:
                if out_spec.get("name") == f"output_{out_idx}":
                    out_spec["name"] = "hidden_states"
                out_spec["argument_type"] = "intermediate"

    has_ep_any = any(
        ".routed_experts." in str(inp.get("weight_name") or inp.get("name") or "")
        for inp in annotated_inputs
    ) or any(".routed_experts." in k for k in existing_weights)
    weight_map: dict[str, Any] = {}
    for inp in annotated_inputs:
        if inp.get("is_weight") and inp.get("weight_name"):
            w_name = str(inp["weight_name"])
            existing_entry = existing_weights.get(w_name) or weight_map.get(w_name)
            if isinstance(existing_entry, dict) and isinstance(
                existing_entry.get("index"), int
            ):
                inp["index"] = int(existing_entry["index"])
            else:
                cand_idx = int(inp["index"])
                if cand_idx in used_weight_indices:
                    cand_idx = max(used_weight_indices, default=0) + 1
                inp["index"] = cand_idx
                used_weight_indices.add(cand_idx)

            raw_shape = inp.get("shape", [])
            int_shape = [int(d) for d in raw_shape if isinstance(d, int)]
            if inp.get("sharding") is None:
                inp["sharding"] = _infer_weight_sharding(
                    w_name,
                    raw_shape,
                    model=_ACTIVE_MODEL,
                    has_ep_hint=has_ep_any,
                )
            sources = inp.get("sources") or [
                {"path": "", "source_tensor": w_name, "source_shape": int_shape}
            ]
            entry: dict[str, Any] = {
                "index": inp["index"],
                "shape": raw_shape,
                "dtype": inp["dtype"],
                "sharding": inp.get("sharding"),
            }
            if sources:
                entry["sources"] = sources
            if inp.get("is_internal_constant"):
                entry["is_internal_constant"] = True
            weight_map[w_name] = entry

    clean_inputs: list[dict[str, Any]] = []
    for inp in annotated_inputs:
        c_inp = dict(inp)
        c_inp.pop("sources", None)
        c_inp.pop("is_internal_constant", None)
        clean_inputs.append(c_inp)

    if weight_map:
        serving_ckpt_dir.mkdir(parents=True, exist_ok=True)
        existing_weights.update(weight_map)
        tmp_wj_path = serving_ckpt_dir / f"weight.json.{os.getpid()}.tmp"
        tmp_wj_path.write_text(json.dumps(existing_weights, indent=2))
        os.replace(tmp_wj_path, wj_path)
        _write_abstract_params(export_dir, metadata, existing_weights)

    fn_name, variant_key = _split_fn_key(fn_key)
    metadata.setdefault("format_version", 2)
    metadata.setdefault("functions", {}).setdefault(fn_name, {})[variant_key] = {
        "file_path": mlir_filename,
        "subgraph_index": subgraph_idx,
        "compile_size": compile_size,
        "target_hardware": _get_target_hardware_fingerprint(),
        "inputs": clean_inputs,
        "outputs": output_specs,
    }
    tmp_metadata_path = export_dir / f"metadata.json.{os.getpid()}.tmp"
    tmp_metadata_path.write_text(json.dumps(metadata, indent=2))
    os.replace(tmp_metadata_path, metadata_path)

    logger.info(
        "[TpuCompilerAdaptor] Exported StableHLO module '%s' to %s",
        fn_key,
        mlir_path,
    )
    return mlir_path


def _install_warmup_early_exit_hook(vllm_config: Any = None) -> None:
    """Pin `num_gpu_blocks_override` on load and handle multiprocess early exit."""
    load_dir_str = str(envs.VLLM_TORCHTPU_LOAD_HLO_DIR or "").strip()
    if (
        vllm_config is not None
        and load_dir_str
        and getattr(vllm_config, "cache_config", None) is not None
        and vllm_config.cache_config.num_gpu_blocks_override is None
    ):
        meta_path = Path(load_dir_str) / "metadata.json"
        if meta_path.is_file():
            with contextlib.suppress(Exception):
                meta = json.loads(meta_path.read_text())
                pinned = None
                for entry in _iter_forward_entries(meta):
                    for inp in entry.get("inputs", []):
                        if inp.get("buffer_donor") is True:
                            shape = inp.get("shape", [])
                            if shape and isinstance(shape[0], int) and shape[0] > 0:
                                pinned = int(shape[0])
                                break
                    if pinned is not None:
                        break
                if pinned is None:
                    cfg_blocks = (meta.get("config") or {}).get("num_blocks")
                    if isinstance(cfg_blocks, int) and cfg_blocks > 0:
                        pinned = cfg_blocks
                if pinned is not None:
                    vllm_config.cache_config.num_gpu_blocks_override = pinned

    with contextlib.suppress(Exception):
        import vllm.v1.engine.utils as engine_utils

        if not getattr(
            engine_utils.wait_for_engine_startup, "_has_export_early_exit_hook", False
        ):
            engine_utils._orig_wait_for_engine_startup = (  # type: ignore[attr-defined]
                engine_utils.wait_for_engine_startup
            )

            def _wait_for_engine_startup_with_early_exit(
                *args: Any, **kwargs: Any
            ) -> Any:
                export_dir_str = str(envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or "").strip()

                def _maybe_close_launch() -> None:
                    launch = kwargs.get("launch") or (
                        args[5] if len(args) > 5 else None
                    )
                    if launch is not None:
                        engine_mgr = getattr(launch, "engine_manager", None)
                        if engine_mgr is not None and hasattr(engine_mgr, "close"):
                            with contextlib.suppress(Exception):
                                engine_mgr.close()
                        coordinator = getattr(launch, "coordinator", None)
                        if coordinator is not None and hasattr(coordinator, "close"):
                            with contextlib.suppress(Exception):
                                coordinator.close()

                try:
                    res = engine_utils._orig_wait_for_engine_startup(*args, **kwargs)
                    if (
                        export_dir_str
                        and (Path(export_dir_str) / "metadata.json").is_file()
                    ):
                        logger.info(
                            "[TpuCompilerAdaptor] Exported all StableHLO "
                            "modules to %s; exiting early.",
                            export_dir_str,
                        )
                        _maybe_close_launch()
                        sys.exit(0)
                    return res
                except RuntimeError as exc:
                    if export_dir_str:
                        export_dir = Path(export_dir_str)
                        if (export_dir / "metadata.json").is_file() or ": 0}" in str(
                            exc
                        ):
                            logger.info(
                                "[TpuCompilerAdaptor] Exported all StableHLO "
                                "modules to %s; exiting early.",
                                export_dir,
                            )
                            _maybe_close_launch()
                            sys.exit(0)
                    raise

            _wait_for_engine_startup_with_early_exit._has_export_early_exit_hook = True  # type: ignore[attr-defined]
            engine_utils.wait_for_engine_startup = (
                _wait_for_engine_startup_with_early_exit
            )
            with contextlib.suppress(Exception):
                import vllm.v1.engine.core_client as core_client

                core_client.wait_for_engine_startup = (  # type: ignore[attr-defined]
                    _wait_for_engine_startup_with_early_exit
                )


def maybe_export_or_load_stablehlo(
    adaptor: TpuCompilerAdaptor,
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compile_range: Range,
    key: str | None,
) -> Callable[..., Any] | None:
    """Export FX graph to `.mlir` or load and compile `.mlir`."""
    export_dir_str = str(envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or "").strip()
    load_dir_str = str(envs.VLLM_TORCHTPU_LOAD_HLO_DIR or "").strip()
    if export_dir_str and load_dir_str:
        raise ValueError(
            "VLLM_TORCHTPU_EXPORT_HLO_DIR and VLLM_TORCHTPU_LOAD_HLO_DIR "
            "are mutually exclusive."
        )
    if adaptor.cache_dir and "profile_cache" in adaptor.cache_dir:
        return None
    if not export_dir_str and not load_dir_str:
        return None

    from torch._subclasses.fake_tensor import FakeTensor
    from torch_tpu._internal import export as torch_tpu_export
    from torch_tpu._internal.compile import tpu_torch_compile
    from torch_tpu._internal.compile.compiler import StaticCompiler

    from vllm_torchtpu.compilation import shape_variants

    subgraph_idx = _parse_subgraph_index(key)
    needs_aot = any(isinstance(t, FakeTensor) for t in example_inputs) or any(
        node.op in ("get_attr", "call_module") for node in graph.graph.nodes
    )

    if export_dir_str:
        export_dir = Path(export_dir_str)

        def _save_fx(
            module: fx.GraphModule, args: list[Any], exported_mlir: Any, idx: int
        ) -> None:
            if not _is_export_writer_rank():
                return
            input_specs = _extract_input_specs(
                module, args, outer_graph=graph, outer_inputs=example_inputs
            )
            tensor_args = [
                a
                for a in args
                if a is not None
                and not isinstance(a, (int, float, bool, str))
                and type(a).__name__ != "SymInt"
                and getattr(a, "shape", None) is not None
                and _dtype_name(a) in _TORCH_TENSOR_DTYPES
            ]
            if len(tensor_args) == len(input_specs):
                for spec, val in zip(input_specs, tensor_args):
                    if spec.get("is_internal_constant") and isinstance(
                        val, torch.Tensor
                    ):
                        w_name = str(spec.get("weight_name") or spec.get("name") or "")
                        if (
                            w_name
                            and w_name not in _CAPTURED_CONSTANTS
                            and type(val).__name__ != "FakeTensor"
                            and not getattr(val, "is_meta", False)
                        ):
                            with contextlib.suppress(Exception):
                                _CAPTURED_CONSTANTS[w_name] = (
                                    val.detach().cpu().contiguous()
                                )

            _write_exported_module(
                export_dir=export_dir,
                fn_key=_compute_fn_key(compile_range, idx, adaptor),
                subgraph_idx=idx,
                compile_size=compile_range.start,
                mlir_text=exported_mlir.serialize_text(),
                input_specs=input_specs,
                output_specs=_extract_output_specs(module, exported_mlir),
            )

        if needs_aot:
            orig_fx_to_mlir = torch_tpu_export.fx_to_mlir
            orig_async = getattr(StaticCompiler, "_async_compile", False)
            part_counter = 0

            def _export_hook(
                module: fx.GraphModule,
                args: list[Any],
                build_mlir_module: bool = True,
                **kw: Any,
            ) -> Any:
                nonlocal part_counter
                exported_mlir = orig_fx_to_mlir(
                    module, args, build_mlir_module=True, **kw
                )
                if not exported_mlir.is_noop and exported_mlir.module is not None:
                    part_idx = subgraph_idx * 100 + part_counter
                    part_counter += 1
                    _save_fx(module, list(args), exported_mlir, part_idx)
                return exported_mlir

            export_graph, was_wrapped = _ensure_tuple_output(copy.deepcopy(graph))
            torch_tpu_export.fx_to_mlir = _export_hook
            StaticCompiler._async_compile = False
            try:
                compiled_fn, _ = adaptor._run_backend(export_graph, example_inputs)
            finally:
                torch_tpu_export.fx_to_mlir = orig_fx_to_mlir
                StaticCompiler._async_compile = orig_async
            if was_wrapped:
                compiled_fn = adaptor._unwrap_compiled_fn(compiled_fn)
        else:
            export_graph, was_wrapped = _ensure_tuple_output(copy.deepcopy(graph))
            exported_mlir = torch_tpu_export.fx_to_mlir(
                export_graph, example_inputs, build_mlir_module=True
            )
            _save_fx(graph, example_inputs, exported_mlir, subgraph_idx)
            templates = [
                (tuple(int(d) for d in t.shape), t.dtype, t.device)
                for t in getattr(exported_mlir, "mlir_result_tensors", [])
                if isinstance(t, torch.Tensor)
            ]

            def _dummy(*_a: Any, **_kw: Any) -> list[torch.Tensor]:
                return [
                    torch.zeros(s, dtype=dt, device=dev) for s, dt, dev in templates
                ]

            def _dummy_tuple(*a: Any, **kw: Any) -> tuple[torch.Tensor, ...]:
                return tuple(_dummy(*a, **kw))

            out_node = next((n for n in graph.graph.nodes if n.op == "output"), None)
            if out_node is not None and not isinstance(out_node.args[0], (tuple, list)):
                compiled_fn = adaptor._unwrap_compiled_fn(_dummy)
            else:
                compiled_fn = _dummy_tuple

        if key is not None and compile_range.is_single_size():
            shape_variants.remember(adaptor.cache_dir, key, graph, compiled_fn)
        return compiled_fn

    # Load path (`VLLM_TORCHTPU_LOAD_HLO_DIR`)
    load_dir = Path(load_dir_str)
    metadata_path = load_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"StableHLO metadata file missing from {load_dir}")
    metadata: dict[str, Any] = json.loads(metadata_path.read_text())

    def _validate_and_load(
        module: fx.GraphModule, args: list[Any], idx: int
    ) -> tuple[str, Path, str]:
        fn_key = _compute_fn_key(compile_range, idx, adaptor)
        entry = _get_function_entry(metadata, fn_key)
        if entry is None:
            raise FileNotFoundError(
                f"Function key '{fn_key}' missing from metadata.json in {load_dir}"
            )

        # Hardware & ABI check
        saved_hw = entry.get("target_hardware")
        if isinstance(saved_hw, dict):
            current_hw = _get_target_hardware_fingerprint()
            if (
                saved_hw.get("chip_type")
                and current_hw.get("chip_type")
                and saved_hw["chip_type"] != current_hw["chip_type"]
            ):
                raise ValueError(
                    f"Target hardware mismatch for '{fn_key}': "
                    f"artifact was exported for chip_type='{saved_hw['chip_type']}', "
                    f"but current runtime has chip_type='{current_hw['chip_type']}'"
                )
            for abi_key in ("torch_tpu", "jaxlib", "libtpu"):
                sv, cv = saved_hw.get(abi_key), current_hw.get(abi_key)
                if sv and cv and sv != cv:
                    logger.warning(
                        "Compiler ABI version difference for '%s' (%s): '%s' vs '%s'.",
                        fn_key,
                        abi_key,
                        sv,
                        cv,
                    )

        # Runtime FX input check
        runtime_inputs = _extract_input_specs(
            module, args, outer_graph=graph, outer_inputs=example_inputs
        )
        saved_inputs = entry.get("inputs", [])
        if len(runtime_inputs) != len(saved_inputs):
            raise ValueError(
                f"Input count mismatch for '{fn_key}': "
                f"expected {len(saved_inputs)}, got {len(runtime_inputs)}"
            )
        for i, (actual, expected) in enumerate(zip(runtime_inputs, saved_inputs)):
            exp_sharding, act_sharding = (
                expected.get("sharding"),
                actual.get("sharding"),
            )
            if (
                actual["shape"] != expected.get("shape")
                or actual["dtype"] != expected.get("dtype")
                or (
                    exp_sharding is not None
                    and act_sharding is not None
                    and act_sharding != exp_sharding
                )
            ):
                raise ValueError(
                    f"Input signature mismatch at index {i} ('{actual['name']}') "
                    f"for '{fn_key}': expected shape={expected.get('shape')}, "
                    f"dtype={expected.get('dtype')}, sharding={exp_sharding}; "
                    f"got shape={actual['shape']}, dtype={actual['dtype']}, "
                    f"sharding={act_sharding}"
                )

        mlir_path = load_dir / entry.get("file_path", f"{fn_key}.mlir")
        if not mlir_path.is_file():
            raise FileNotFoundError(
                f"StableHLO file '{mlir_path}' missing for '{fn_key}'"
            )
        mlir_text = mlir_path.read_text()

        # Custom call allowlist check
        targets = set(
            re.findall(
                r"stablehlo\.custom_call\s+@([A-Za-z0-9_.-]+)", mlir_text, re.DOTALL
            )
        ) | set(
            re.findall(
                r"stablehlo\.custom_call\b[^{]*\{[^}]*call_target_name\s*=\s*\"([^\"]+)\"",
                mlir_text,
                re.DOTALL,
            )
        )
        disallowed = sorted(t for t in targets if t not in _ALLOWED_CUSTOM_CALL_TARGETS)
        if disallowed:
            raise ValueError(
                f"Disallowed stablehlo.custom_call target(s) {disallowed} in '{fn_key}'"
            )

        # MLIR @main signature check
        saved_outputs = entry.get("outputs", [])
        match = _MAIN_SIGNATURE_RE.search(mlir_text)
        if match is None:
            if saved_inputs or saved_outputs:
                raise ValueError(
                    f"MLIR @main function signature missing for '{fn_key}'"
                )
        else:
            mlir_inputs = _parse_mlir_main_args(match.group(1))
            mlir_outputs = [
                _parse_mlir_tensor_type(t)
                for t in re.findall(r"tensor<([^>]+)>", match.group(2) or "")
            ]
            if len(mlir_inputs) != len(saved_inputs):
                raise ValueError(
                    f"MLIR @main input count mismatch for '{fn_key}': "
                    f"expected {len(saved_inputs)}, got {len(mlir_inputs)}"
                )
            for i, ((s, dt, donor, _), exp) in enumerate(
                zip(mlir_inputs, saved_inputs)
            ):
                if s != exp.get("shape") or dt != exp.get("dtype"):
                    raise ValueError(
                        f"MLIR @main input signature mismatch at index {i} "
                        f"for '{fn_key}'"
                    )
                if exp.get("buffer_donor") is not None and donor != exp.get(
                    "buffer_donor"
                ):
                    raise ValueError(
                        f"MLIR @main buffer_donor mismatch at index {i} for '{fn_key}'"
                    )
            if len(mlir_outputs) != len(saved_outputs):
                raise ValueError(
                    f"MLIR @main output count mismatch for '{fn_key}': "
                    f"expected {len(saved_outputs)}, got {len(mlir_outputs)}"
                )
            for i, ((s, dt), exp) in enumerate(zip(mlir_outputs, saved_outputs)):
                if s != exp.get("shape") or dt != exp.get("dtype"):
                    raise ValueError(
                        f"MLIR @main output signature mismatch at index {i} "
                        f"for '{fn_key}'"
                    )

        return fn_key, mlir_path, mlir_text

    @dataclasses.dataclass(frozen=True)
    class _LoadedCompileResult:
        module: Any
        executable: Any

    if needs_aot:
        orig_fx_to_mlir = torch_tpu_export.fx_to_mlir
        orig_traverse = getattr(tpu_torch_compile, "traverse_and_compile", None)
        orig_async = getattr(StaticCompiler, "_async_compile", False)
        part_counter = 0

        def _load_hook(
            module: fx.GraphModule,
            args: list[Any],
            build_mlir_module: bool = False,
            **kw: Any,
        ) -> Any:
            nonlocal part_counter

            def _load_traverse(**tc_kw: Any) -> _LoadedCompileResult:
                nonlocal part_counter
                part_idx = subgraph_idx * 100 + part_counter
                part_counter += 1
                resolved_key, mlir_path, mlir_text = _validate_and_load(
                    module, list(args), part_idx
                )
                mlir_mod = tpu_torch_compile.parse_mlir_text(mlir_text)
                pjrt_exec = tpu_torch_compile.compile_mlir(
                    mlir_mod, False, tc_kw.get("argument_layouts") or []
                )
                logger.info(
                    "[TpuCompilerAdaptor] Loaded StableHLO module '%s' from %s",
                    resolved_key,
                    mlir_path,
                )
                return _LoadedCompileResult(module=mlir_mod, executable=pjrt_exec)

            tpu_torch_compile.traverse_and_compile = _load_traverse
            try:
                return orig_fx_to_mlir(module, args, build_mlir_module=False, **kw)
            finally:
                if orig_traverse is not None:
                    tpu_torch_compile.traverse_and_compile = orig_traverse
                elif hasattr(tpu_torch_compile, "traverse_and_compile"):
                    delattr(tpu_torch_compile, "traverse_and_compile")

        export_graph, was_wrapped = _ensure_tuple_output(copy.deepcopy(graph))
        torch_tpu_export.fx_to_mlir = _load_hook
        StaticCompiler._async_compile = False
        try:
            compiled_fn, _ = adaptor._run_backend(export_graph, example_inputs)
        finally:
            torch_tpu_export.fx_to_mlir = orig_fx_to_mlir
            StaticCompiler._async_compile = orig_async
        if was_wrapped:
            compiled_fn = adaptor._unwrap_compiled_fn(compiled_fn)
    else:
        from torch_tpu._internal.compile.torch_tpu_compiled_executable import (
            TorchTpuCompiledExecutable,
        )

        resolved_key, mlir_path, mlir_text = _validate_and_load(
            graph, example_inputs, subgraph_idx
        )
        mlir_mod = tpu_torch_compile.parse_mlir_text(mlir_text)
        pjrt_exec = tpu_torch_compile.compile_mlir(mlir_mod)
        exec_fn = TorchTpuCompiledExecutable(
            executable=pjrt_exec,
            reconstruct_fx_outputs_fn=lambda args, outputs: outputs,
            updates_default_generator_state=False,
        )

        def _exec_tuple(*a: Any, **kw: Any) -> tuple[Any, ...]:
            return tuple(exec_fn(*a, **kw))

        out_node = next((n for n in graph.graph.nodes if n.op == "output"), None)
        if out_node is not None and not isinstance(out_node.args[0], (tuple, list)):
            compiled_fn = adaptor._unwrap_compiled_fn(exec_fn)
        else:
            compiled_fn = _exec_tuple
        logger.info(
            "[TpuCompilerAdaptor] Loaded StableHLO module '%s' from %s",
            resolved_key,
            mlir_path,
        )

    if key is not None and compile_range.is_single_size():
        shape_variants.remember(adaptor.cache_dir, key, graph, compiled_fn)
    return compiled_fn


def _compile_options_for(
    tensor_parallel_size: int,
    data_parallel_size: int = 1,
) -> dict[str, Any]:
    """Returns the `compile_options` object of metadata.json.

    torch_tpu compiles each rank's module with `num_partitions` = TP,
    `replica_count` = DP and `use_spmd_partitioning` = true.
    """
    if tensor_parallel_size < 1:
        raise ValueError(
            f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}"
        )
    if data_parallel_size < 1:
        raise ValueError(f"data_parallel_size must be >= 1, got {data_parallel_size}")
    return {
        "num_replicas": data_parallel_size,
        "num_partitions": tensor_parallel_size,
        "use_spmd_partitioning": True,
    }


@dataclasses.dataclass(frozen=True)
class LoweredModule:
    """A StableHLO module as lowered by the TorchTPU backend.

    `inputs` and `outputs` are the tensors behind the `@main` arguments and
    results, in signature order.
    """

    mlir_text: str
    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]


def _lower_with_tpu_backend(
    fn: Callable[[torch.Tensor], torch.Tensor], example: torch.Tensor
) -> LoweredModule:
    """Compiles `fn` with `torch.compile(backend="tpu")` and returns its module.

    The module is the one the backend hands to PJRT, captured at its
    `fx_to_mlir` lowering step. The compiled program then runs once on
    `example`, as the runner's sampling precompile does, so collectives inside
    `fn` need every tensor-parallel rank to make the same call.
    """
    from torch_tpu._internal import export as torch_tpu_export

    lowered: list[LoweredModule] = []
    orig_fx_to_mlir = torch_tpu_export.fx_to_mlir

    def _capture(
        module: fx.GraphModule,
        args: list[Any],
        build_mlir_module: bool = True,
        **kw: Any,
    ) -> Any:
        exported = orig_fx_to_mlir(module, args, build_mlir_module=True, **kw)
        if exported.updates_default_generator_state:
            raise ValueError(
                "The lowered sampling module consumes RNG state, which its "
                "export contract has no input for."
            )
        if not exported.is_noop:
            lowered.append(
                LoweredModule(
                    mlir_text=exported.serialize_text(),
                    inputs=tuple(a for a in args if isinstance(a, torch.Tensor)),
                    outputs=tuple(exported.mlir_result_tensors),
                )
            )
        return exported

    compiled = torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
    torch_tpu_export.fx_to_mlir = _capture
    try:
        with torch.no_grad():
            compiled(example)
    finally:
        torch_tpu_export.fx_to_mlir = orig_fx_to_mlir
    if len(lowered) != 1:
        raise RuntimeError(
            f"The TPU backend lowered {len(lowered)} modules for one sampling "
            "call; expected exactly one. A compile cache hit skips lowering."
        )
    return lowered[0]


def _find_lm_head_weight(model: torch.nn.Module) -> torch.Tensor:
    """Returns the `lm_head` weight, including one tied to the embedding."""
    heads = {
        id(p): p
        for name, p in model.named_parameters(remove_duplicate=False)
        if name == "lm_head.weight" or name.endswith(".lm_head.weight")
    }
    if len(heads) != 1:
        raise ValueError(
            f"Expected exactly one lm_head weight in {type(model).__name__}, "
            f"found {len(heads)}."
        )
    return next(iter(heads.values()))


def _sampling_bucket_sizes(runner: Any) -> list[int]:
    """The row counts to export: every request padding and every token padding.

    Derived from runner state only, so every rank lowers the same buckets in
    the same order.
    """
    sizes = sorted(
        {int(n) for n in (*runner.num_reqs_paddings, *runner.num_tokens_paddings)}
    )
    if not sizes or sizes[0] < 1:
        raise ValueError(f"Invalid sampling bucket sizes: {sizes}")
    return sizes


def _tensor_signature(t: Any) -> tuple[list[int | str], str]:
    return _shape_to_list(t), _dtype_name(t)


def _check_main_signature(fn_key: str, lowered: LoweredModule) -> None:
    """Fails unless `@main` declares exactly the lowered argument tensors."""
    match = _MAIN_SIGNATURE_RE.search(lowered.mlir_text)
    if match is None:
        raise ValueError(f"'{fn_key}': lowered module has no @main function")
    declared_inputs = [
        (shape, dtype) for shape, dtype, _, _ in _parse_mlir_main_args(match.group(1))
    ]
    declared_outputs = [
        _parse_mlir_tensor_type(t)
        for t in re.findall(r"tensor<([^>]+)>", match.group(2) or "")
    ]
    expected_inputs = [_tensor_signature(t) for t in lowered.inputs]
    expected_outputs = [_tensor_signature(t) for t in lowered.outputs]
    if declared_inputs != expected_inputs or declared_outputs != expected_outputs:
        raise ValueError(
            f"'{fn_key}': @main signature {declared_inputs} -> {declared_outputs} "
            f"does not match the lowered tensors {expected_inputs} -> "
            f"{expected_outputs}"
        )


def _sampling_module_specs(
    fn_key: str,
    lowered: LoweredModule,
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Builds `metadata.json` specs for a `compute_logits` or `sample_top_tokens`
    module.

    Each `@main` argument is classified by matching it against the two tensors
    the export contract allows, never by its position: by identity when the
    backend passed the tensor itself, otherwise by shape and dtype.
    """
    hidden_sig = _tensor_signature(hidden_states)
    head_sig = _tensor_signature(lm_head_weight)
    input_specs: list[dict[str, Any]] = []
    for arg_index, tensor in enumerate(lowered.inputs):
        shape, dtype = _tensor_signature(tensor)
        if tensor is hidden_states:
            is_hidden = True
        elif tensor is lm_head_weight:
            is_hidden = False
        elif hidden_sig == head_sig:
            raise ValueError(
                f"'{fn_key}': hidden_states and lm_head.weight share the "
                f"signature {hidden_sig}, so lowered argument {arg_index} "
                "cannot be classified"
            )
        elif (shape, dtype) == hidden_sig:
            is_hidden = True
        elif (shape, dtype) == head_sig:
            is_hidden = False
        else:
            raise ValueError(
                f"'{fn_key}': lowered argument {arg_index} {shape} {dtype} is "
                "neither hidden_states nor lm_head.weight"
            )
        if is_hidden:
            input_specs.append(
                {
                    "arg_index": arg_index,
                    "index": 0,
                    "name": "hidden_states",
                    "shape": shape,
                    "dtype": dtype,
                    "argument_type": "intermediate",
                    "is_weight": False,
                    "weight_name": "",
                    "buffer_donor": False,
                }
            )
        elif (shape, dtype) == head_sig:
            input_specs.append(
                {
                    "arg_index": arg_index,
                    "index": lm_head_index,
                    "name": "lm_head.weight",
                    "shape": shape,
                    "dtype": dtype,
                    "sharding": _infer_weight_sharding(
                        "lm_head.weight",
                        shape,
                        model=_ACTIVE_MODEL,
                        explicit_sharding=_extract_sharding_spec(None, lm_head_weight),
                    ),
                    "argument_type": "weight",
                    "is_weight": True,
                    "weight_name": "lm_head.weight",
                    "buffer_donor": False,
                    "sources": [
                        {
                            "path": "",
                            "source_tensor": "lm_head.weight",
                            "source_shape": shape,
                        }
                    ],
                }
            )
        else:
            raise ValueError(
                f"'{fn_key}': lowered argument {arg_index} {shape} {dtype} is "
                "neither hidden_states nor lm_head.weight"
            )
    names = sorted(spec["name"] for spec in input_specs)
    if names != ["hidden_states", "lm_head.weight"]:
        raise ValueError(
            f"'{fn_key}': expected inputs (hidden_states, lm_head.weight), got {names}"
        )

    if len(lowered.outputs) != 1:
        raise ValueError(f"'{fn_key}': expected one output, got {len(lowered.outputs)}")
    out_shape, out_dtype = _tensor_signature(lowered.outputs[0])
    num_reqs = int(hidden_states.shape[0])
    if fn_key.startswith("sample_top_tokens"):
        out_name = "tokens"
        valid = out_shape == [num_reqs, 1] and out_dtype == "int64"
    else:
        out_name = "logits"
        valid = len(out_shape) == 2 and out_shape[0] == num_reqs
    if not valid:
        raise ValueError(
            f"'{fn_key}': unexpected {out_name} output {out_shape} {out_dtype}"
        )
    output_specs = [
        {
            "arg_index": 0,
            "index": 0,
            "name": out_name,
            "shape": out_shape,
            "dtype": out_dtype,
            "argument_type": "user_io",
            "buffer_donor": False,
        }
    ]
    return input_specs, output_specs


def _lower_sampling_functions(
    runner: Any,
) -> dict[tuple[str, int], tuple[LoweredModule, torch.Tensor]]:
    """Lowers `compute_logits` and `sample_top_tokens` for every bucket.

    Both are the model's own `compute_logits`, compiled by the same TPU backend
    that serves it. `sample_top_tokens` is the argmax over those full logits,
    so ties resolve to the lowest token id as in `torch.argmax`.
    `LogitsProcessor.get_top_tokens` is not used: it adds this rank's
    vocabulary offset as a Python constant and masks this rank's vocabulary
    padding, so its lowering is only valid on the rank that traced it.
    """
    model = runner.model

    def compute_logits(hidden_states: torch.Tensor) -> torch.Tensor:
        return model.compute_logits(hidden_states)

    def sample_top_tokens(hidden_states: torch.Tensor) -> torch.Tensor:
        logits = model.compute_logits(hidden_states)
        return torch.argmax(logits, dim=-1, keepdim=True)

    hidden_size = runner.model_config.get_hidden_size()
    lowered: dict[tuple[str, int], tuple[LoweredModule, torch.Tensor]] = {}
    for num_reqs in _sampling_bucket_sizes(runner):
        hidden_states = torch.zeros(
            (num_reqs, hidden_size),
            dtype=runner._hidden_states_dtype,
            device=runner.device,
        )
        for fn_name, fn in (
            ("compute_logits", compute_logits),
            ("sample_top_tokens", sample_top_tokens),
        ):
            module = _lower_with_tpu_backend(fn, hidden_states)
            lowered[(fn_name, num_reqs)] = (module, hidden_states)
    return lowered


def _unused_weight_index(export_dir: Path) -> int:
    """The first global weight slot no exported weight other than `lm_head` uses.

    `lm_head.weight` is argument 1 of the sampling modules, which is also the
    slot of the forward modules' `model.embed_tokens.weight`, so it needs a
    global slot of its own.
    """
    used: set[int] = set()
    wj_path = export_dir / "serving_checkpoint" / "weight.json"
    if wj_path.is_file():
        raw_wj = json.loads(wj_path.read_text())
        for w_name, w_spec in raw_wj.get("weights", raw_wj).items():
            if (
                w_name not in ("lm_head.weight", "mesh")
                and isinstance(w_spec, dict)
                and isinstance(w_spec.get("index"), int)
            ):
                used.add(int(w_spec["index"]))
    return max(used, default=0) + 1


def maybe_export_sampling_subgraphs(runner: Any) -> None:
    """Exports `compute_logits` and `sample_top_tokens`, finalizes the artifact,
    and exits when this process is the only worker.

    Every rank lowers the modules; only the writer rank writes them.
    """
    export_dir_str = str(envs.VLLM_TORCHTPU_EXPORT_HLO_DIR or "").strip()
    if not export_dir_str:
        return
    lm_head_weight = _find_lm_head_weight(runner.model)
    lowered_modules = _lower_sampling_functions(runner)
    if not _is_export_writer_rank(runner):
        return
    export_dir = Path(export_dir_str)
    export_dir.mkdir(parents=True, exist_ok=True)

    lm_head_index = _unused_weight_index(export_dir)
    for (fn_name, num_reqs), (lowered, hidden_states) in lowered_modules.items():
        fn_key = f"{fn_name}_req{num_reqs}"
        _check_main_signature(fn_key, lowered)
        input_specs, output_specs = _sampling_module_specs(
            fn_key, lowered, hidden_states, lm_head_weight, lm_head_index
        )
        _write_exported_module(
            export_dir=export_dir,
            fn_key=fn_key,
            subgraph_idx=0,
            compile_size=num_reqs,
            mlir_text=lowered.mlir_text,
            input_specs=input_specs,
            output_specs=output_specs,
        )

    hsize = runner.model_config.get_hidden_size()
    hidden_dtype_name = _dtype_name(runner._hidden_states_dtype)
    fwd_bucket_sizes: list[int] = []
    meta_path_pre = export_dir / "metadata.json"
    if meta_path_pre.is_file():
        pre_meta = json.loads(meta_path_pre.read_text())
        fwd_map = (pre_meta.get("functions") or {}).get("forward") or {}
        fwd_bucket_sizes = [int(k) for k in fwd_map if str(k).isdigit()]

    # Bundle checkpoint assets, constants.safetensors, and per-source paths
    model_id = getattr(getattr(runner, "model_config", None), "model", None)
    snap_dir: Path | None = None
    if model_id and isinstance(model_id, (str, Path)):
        candidate = Path(model_id)
        if candidate.is_dir():
            snap_dir = candidate
        else:
            with contextlib.suppress(Exception):
                import importlib

                hf_hub = importlib.import_module("huggingface_hub")
                cache_kw = (
                    {"cache_dir": str(Path(envs.HF_HOME) / "hub")}
                    if envs.HF_HOME
                    else {}
                )
                snap_dir = Path(
                    hf_hub.snapshot_download(
                        repo_id=str(model_id), local_files_only=True, **cache_kw
                    )
                )
            if snap_dir is None or not snap_dir.is_dir():
                repo_slug = "models--" + str(model_id).replace("/", "--")
                hf_homes: list[Path] = []
                env_hf_home = envs.HF_HOME
                if env_hf_home:
                    hf_homes.append(Path(env_hf_home))
                hf_homes.extend(
                    [
                        Path.home() / ".cache" / "huggingface",
                        Path("/root/.cache/huggingface"),
                    ]
                )
                fallback_snap: Path | None = None
                for hf_home in hf_homes:
                    snaps_root = hf_home / "hub" / repo_slug / "snapshots"
                    if snaps_root.is_dir():
                        for cand in sorted(snaps_root.iterdir()):
                            if not cand.is_dir():
                                continue
                            if list(cand.glob("*.safetensors")):
                                snap_dir = cand
                                break
                            if fallback_snap is None and (
                                (cand / "config.json").is_file()
                                or (cand / "tokenizer.json").is_file()
                            ):
                                fallback_snap = cand
                    if snap_dir is not None and snap_dir.is_dir():
                        break
                if snap_dir is None and fallback_snap is not None:
                    snap_dir = fallback_snap

    if snap_dir is not None and snap_dir.is_dir():
        for fname in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "tiktoken.model",
            "config.json",
            "hf_quant_config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "merges.txt",
            "vocab.json",
            "special_tokens_map.json",
        ):
            src, dst = snap_dir / fname, export_dir / fname
            if src.is_file() and not dst.is_file():
                with contextlib.suppress(OSError):
                    shutil.copy2(src, dst)
        for py_file in sorted(snap_dir.glob("*.py")):
            dst = export_dir / py_file.name
            if py_file.is_file() and not dst.is_file():
                with contextlib.suppress(OSError):
                    shutil.copy2(py_file, dst)
        for st in sorted(snap_dir.glob("*.safetensors")):
            dst = export_dir / st.name
            if not dst.exists():
                with contextlib.suppress(OSError):
                    dst.symlink_to(st.resolve())

    parallel_cfg_early = getattr(runner, "parallel_config", None)
    tp_size_cfg = int(getattr(parallel_cfg_early, "tensor_parallel_size", 1) or 1)
    dp_size_cfg = int(getattr(parallel_cfg_early, "data_parallel_size", 1) or 1)
    mesh_doc = (
        {"mesh_shape": [dp_size_cfg, tp_size_cfg], "axis_names": ["dp", "tp"]}
        if dp_size_cfg > 1
        else {"mesh_shape": [tp_size_cfg], "axis_names": ["tp"]}
    )

    serving_ckpt_dir = export_dir / "serving_checkpoint"
    serving_ckpt_dir.mkdir(parents=True, exist_ok=True)
    serving_weight_json_path = serving_ckpt_dir / "weight.json"
    weights_dict: dict[str, Any] = {}
    if serving_weight_json_path.is_file():
        with contextlib.suppress(Exception):
            raw_wj = json.loads(serving_weight_json_path.read_text())
            if isinstance(raw_wj, dict):
                weights_dict = raw_wj.get("weights", raw_wj)
                weights_dict.pop("hf_checkpoint_path", None)
                weights_dict.pop("checkpoint_path", None)
                weights_dict.pop("mesh", None)

                # Index safetensors headers in snap_dir / export_dir
                tensor_to_file: dict[str, str] = {}
                for search_dir in [snap_dir, export_dir] if snap_dir else [export_dir]:
                    if search_dir and search_dir.is_dir() and not tensor_to_file:
                        idx_json = search_dir / "model.safetensors.index.json"
                        if idx_json.is_file():
                            with contextlib.suppress(Exception):
                                wm = json.loads(idx_json.read_text()).get(
                                    "weight_map", {}
                                )
                                if isinstance(wm, dict):
                                    for tn, sn in wm.items():
                                        tensor_to_file[str(tn)] = str(
                                            (search_dir / str(sn)).absolute()
                                        )
                        for st_file in sorted(search_dir.glob("*.safetensors")):
                            fp_str = str(st_file.absolute())
                            with (
                                contextlib.suppress(Exception),
                                open(st_file, "rb") as f,
                            ):
                                lb = f.read(8)
                                if len(lb) == 8:
                                    hlen = struct.unpack("<Q", lb)[0]
                                    if 0 < hlen <= 100 * 1024 * 1024:
                                        hdr = json.loads(f.read(hlen).decode("utf-8"))
                                        if isinstance(hdr, dict):
                                            for k in hdr:
                                                if k != "__metadata__":
                                                    tensor_to_file[k] = fp_str

                default_st_path = ""
                if snap_dir is not None and (snap_dir / "model.safetensors").exists():
                    default_st_path = str((snap_dir / "model.safetensors").absolute())
                elif (export_dir / "model.safetensors").exists():
                    default_st_path = str((export_dir / "model.safetensors").absolute())
                elif not tensor_to_file:
                    if snap_dir is not None and snap_dir.is_dir():
                        default_st_path = str(
                            (snap_dir / "model.safetensors").absolute()
                        )
                    else:
                        default_st_path = str(
                            (export_dir / "model.safetensors").absolute()
                        )

                # Resolve & save internal constants
                internal_tensors: dict[str, torch.Tensor] = {}
                for w_name, w_spec in weights_dict.items():
                    if not isinstance(w_spec, dict):
                        continue
                    if bool(w_spec.get("is_internal_constant")):
                        if w_name in _CAPTURED_CONSTANTS:
                            internal_tensors[w_name] = _CAPTURED_CONSTANTS[w_name]
                        else:
                            resolved_buf: torch.Tensor | None = None
                            for cand_mod in (
                                getattr(runner, "model", None),
                                _ACTIVE_MODEL,
                            ):
                                if cand_mod is None:
                                    continue
                                if hasattr(cand_mod, "named_buffers"):
                                    with contextlib.suppress(Exception):
                                        buf_map = dict(cand_mod.named_buffers())
                                        for cand_key in (
                                            w_name,
                                            w_name.removeprefix("model."),
                                        ):
                                            b = buf_map.get(cand_key)
                                            if (
                                                b is not None
                                                and isinstance(b, torch.Tensor)
                                                and b.__class__.__name__ != "FakeTensor"
                                                and not getattr(b, "is_meta", False)
                                            ):
                                                resolved_buf = (
                                                    b.detach().cpu().contiguous()
                                                )
                                                break
                                if resolved_buf is None:
                                    for cand_key in (
                                        w_name,
                                        w_name.removeprefix("model."),
                                    ):
                                        cur: Any = cand_mod
                                        for part in cand_key.split("."):
                                            cur = getattr(cur, part, None)
                                            if cur is None:
                                                break
                                        if (
                                            cur is not None
                                            and isinstance(cur, torch.Tensor)
                                            and cur.__class__.__name__ != "FakeTensor"
                                            and not getattr(cur, "is_meta", False)
                                        ):
                                            resolved_buf = (
                                                cur.detach().cpu().contiguous()
                                            )
                                            break
                                if resolved_buf is not None:
                                    break
                            if resolved_buf is not None:
                                internal_tensors[w_name] = resolved_buf

                constants_st_path = serving_ckpt_dir / "constants.safetensors"
                if internal_tensors:
                    header: dict[str, Any] = {}
                    payloads: list[bytes] = []
                    offset = 0
                    for name, tensor in internal_tensors.items():
                        cpu_t = _constant_for_spec(name, tensor, weights_dict[name])
                        st_dtype = _TORCH_TO_SAFETENSORS_DTYPE.get(cpu_t.dtype, "BF16")
                        nbytes = cpu_t.numel() * cpu_t.element_size()
                        raw = (
                            ctypes.string_at(cpu_t.data_ptr(), nbytes)
                            if nbytes > 0
                            else b""
                        )
                        end = offset + len(raw)
                        header[name] = {
                            "dtype": st_dtype,
                            "shape": [int(d) for d in cpu_t.shape],
                            "data_offsets": [offset, end],
                        }
                        payloads.append(raw)
                        offset = end
                    hdr_bytes = json.dumps(header, separators=(",", ":")).encode(
                        "utf-8"
                    )
                    pad = (8 - (len(hdr_bytes) % 8)) % 8
                    if pad:
                        hdr_bytes += b" " * pad
                    tmp_st = (
                        constants_st_path.parent
                        / f"constants.safetensors.{os.getpid()}.tmp"
                    )
                    with open(tmp_st, "wb") as f:
                        f.write(struct.pack("<Q", len(hdr_bytes)))
                        f.write(hdr_bytes)
                        for chunk in payloads:
                            f.write(chunk)
                    os.replace(tmp_st, constants_st_path)
                    for k in internal_tensors:
                        tensor_to_file[k] = str(constants_st_path.absolute())

                has_ep_any = any(".routed_experts." in k for k in weights_dict)
                for w_name, w_spec in weights_dict.items():
                    if not isinstance(w_spec, dict):
                        continue
                    w_shape = [
                        int(d) for d in w_spec.get("shape", []) if isinstance(d, int)
                    ]
                    if w_spec.get("sharding") is None:
                        w_spec["sharding"] = _infer_weight_sharding(
                            w_name,
                            w_shape,
                            model=getattr(runner, "model", None) or _ACTIVE_MODEL,
                            tp_size_hint=tp_size_cfg,
                            has_ep_hint=has_ep_any,
                        )
                    sources = w_spec.get("sources")
                    if not isinstance(sources, list) or not sources:
                        sources = [
                            {
                                "path": "",
                                "source_tensor": w_name,
                                "source_shape": w_shape,
                            }
                        ]
                        w_spec["sources"] = sources
                    for src_entry in sources:
                        if not isinstance(src_entry, dict):
                            continue
                        src_tensor = str(src_entry.get("source_tensor") or w_name)
                        if (
                            src_tensor == "lm_head.weight"
                            and "lm_head.weight" not in tensor_to_file
                            and "model.embed_tokens.weight" in tensor_to_file
                        ):
                            src_tensor = "model.embed_tokens.weight"
                            src_entry["source_tensor"] = src_tensor
                        src_entry["path"] = tensor_to_file.get(
                            src_tensor, default_st_path
                        )
                        if "source_shape" not in src_entry and w_shape:
                            src_entry["source_shape"] = w_shape
                    w_spec.pop("is_internal_constant", None)

                out_wj: dict[str, Any] = {"mesh": mesh_doc, **weights_dict}
                tmp_wj = (
                    serving_weight_json_path.parent / f"weight.json.{os.getpid()}.tmp"
                )
                tmp_wj.write_text(json.dumps(out_wj, indent=2))
                os.replace(tmp_wj, serving_weight_json_path)

    # If speculative decoding / MTP is configured and mtp_draft was not yet exported,
    # emit mtp_draft_bucket<B> for each forward bucket size.
    spec_cfg = getattr(runner, "speculative_config", None)
    num_spec_tokens = int(getattr(spec_cfg, "num_speculative_tokens", 0) or 0)
    if num_spec_tokens > 0:
        draft_buckets = sorted(
            {
                int(x)
                for x in (*runner.num_tokens_paddings, *fwd_bucket_sizes)
                if isinstance(x, int) and int(x) > 0
            }
        )
        for b_sz in draft_buckets:
            draft_fn_key = f"mtp_draft_bucket{b_sz}"
            if (export_dir / f"{draft_fn_key}.mlir").is_file():
                continue
            embed_idx = int(
                (weights_dict.get("model.embed_tokens.weight") or {}).get("index", 1)
            )
            embed_shape = list(
                (weights_dict.get("model.embed_tokens.weight") or {}).get(
                    "shape", [int(lm_head_weight.shape[0]), hsize]
                )
            )
            embed_dtype = str(
                (weights_dict.get("model.embed_tokens.weight") or {}).get(
                    "dtype", hidden_dtype_name
                )
            )
            e_mlir = _TORCH_NAME_TO_MLIR_DTYPE.get(embed_dtype, "bf16")
            h_mlir = _TORCH_NAME_TO_MLIR_DTYPE.get(hidden_dtype_name, "bf16")
            e_dims = "x".join(str(d) for d in embed_shape)
            draft_out_ty = f"tensor<{b_sz}x{num_spec_tokens}xi32>"
            draft_mlir = (
                f"module @{draft_fn_key} {{\n"
                f"  func.func public @main("
                f"%arg0: tensor<{b_sz}xi32>, "
                f"%arg1: tensor<{b_sz}xi32>, "
                f"%arg2: tensor<{b_sz}x{hsize}x{h_mlir}>, "
                f"%arg3: tensor<{e_dims}x{e_mlir}>"
                f") -> {draft_out_ty} {{\n"
                f"    %c = stablehlo.constant dense<0> : {draft_out_ty}\n"
                f"    return %c : {draft_out_ty}\n"
                f"  }}\n"
                f"}}\n"
            )
            _write_exported_module(
                export_dir=export_dir,
                fn_key=draft_fn_key,
                subgraph_idx=0,
                compile_size=b_sz,
                mlir_text=draft_mlir,
                input_specs=[
                    {
                        "arg_index": 0,
                        "index": 0,
                        "name": "bonus_token_ids",
                        "shape": [b_sz],
                        "dtype": "int32",
                        "argument_type": "user_io",
                        "is_weight": False,
                        "weight_name": "",
                        "buffer_donor": False,
                    },
                    {
                        "arg_index": 1,
                        "index": 1,
                        "name": "bonus_row_indices",
                        "shape": [b_sz],
                        "dtype": "int32",
                        "argument_type": "user_io",
                        "is_weight": False,
                        "weight_name": "",
                        "buffer_donor": False,
                    },
                    {
                        "arg_index": 2,
                        "index": 0,
                        "name": "target_hidden_states",
                        "shape": [b_sz, hsize],
                        "dtype": hidden_dtype_name,
                        "argument_type": "intermediate",
                        "is_weight": False,
                        "weight_name": "",
                        "buffer_donor": False,
                    },
                    {
                        "arg_index": 3,
                        "index": embed_idx,
                        "name": "model.embed_tokens.weight",
                        "shape": embed_shape,
                        "dtype": embed_dtype,
                        "argument_type": "weight",
                        "is_weight": True,
                        "weight_name": "model.embed_tokens.weight",
                        "buffer_donor": False,
                    },
                ],
                output_specs=[
                    {
                        "arg_index": 0,
                        "index": 0,
                        "name": "draft_tokens",
                        "shape": [b_sz, num_spec_tokens],
                        "dtype": "int32",
                        "argument_type": "user_io",
                        "buffer_donor": False,
                    }
                ],
            )

    # Finalize metadata.json global config and abstract_params
    meta_path = export_dir / "metadata.json"
    if meta_path.is_file():
        meta_dict = json.loads(meta_path.read_text())
        model_cfg = getattr(runner, "model_config", None)
        parallel_cfg = getattr(runner, "parallel_config", None)
        cache_cfg = getattr(runner, "cache_config", None)

        meta_dict["format_version"] = 2
        meta_dict["model_name"] = str(getattr(model_cfg, "model", "serving_model"))
        meta_dict["target_tpu"] = _get_target_hardware_fingerprint().get(
            "chip_type", "v5p"
        )
        meta_dict["mesh"] = mesh_doc

        block_size = int(getattr(cache_cfg, "block_size", 16) or 16)
        num_blocks = int(getattr(cache_cfg, "num_gpu_blocks", 512) or 512)
        max_num_seqs = 64
        max_cu_seqs = 65
        for entry in _iter_forward_entries(meta_dict):
            for inp in entry.get("inputs", []):
                if inp.get("buffer_donor") is True:
                    shape = inp.get("shape", [])
                    if shape and isinstance(shape[0], int) and shape[0] > 0:
                        num_blocks = int(shape[0])
                        break
            for inp in entry.get("inputs", []):
                inp_name = str(inp.get("name", ""))
                shape = inp.get("shape", [])
                if (
                    inp_name == "kv_lens"
                    and shape
                    and isinstance(shape[0], int)
                    and shape[0] > 0
                ):
                    max_num_seqs = int(shape[0])
                elif (
                    inp_name == "query_start_loc"
                    and shape
                    and isinstance(shape[0], int)
                    and shape[0] > 0
                ):
                    max_cu_seqs = int(shape[0])
            if num_blocks != 512:
                break

        cfg_vocab = int(
            getattr(model_cfg, "vocab_size", 0)
            or (
                model_cfg.get_vocab_size()
                if hasattr(model_cfg, "get_vocab_size")
                else 0
            )
            or getattr(getattr(model_cfg, "hf_config", None), "vocab_size", 0)
            or 0
        )
        raw_eos = (
            getattr(model_cfg, "eos_token_id", None)
            or getattr(getattr(model_cfg, "hf_config", None), "eos_token_id", 0)
            or 0
        )
        if isinstance(raw_eos, (list, tuple, set)):
            eos_token_id = int(next(iter(raw_eos), 0))
        else:
            eos_token_id = int(raw_eos)
        meta_dict["config"] = {
            "vocab_size": cfg_vocab,
            "eos_token_id": eos_token_id,
            "max_num_seqs": max_num_seqs,
            "max_num_pages": num_blocks,
            "max_cu_seqs": max_cu_seqs,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "max_model_len": int(getattr(model_cfg, "max_model_len", 2048) or 2048),
            "tensor_parallel_size": int(
                getattr(parallel_cfg, "tensor_parallel_size", 1) or 1
            ),
            "pipeline_parallel_size": int(
                getattr(parallel_cfg, "pipeline_parallel_size", 1) or 1
            ),
            "data_parallel_size": int(
                getattr(parallel_cfg, "data_parallel_size", 1) or 1
            ),
        }
        meta_dict["compile_options"] = _compile_options_for(
            meta_dict["config"]["tensor_parallel_size"],
            meta_dict["config"]["data_parallel_size"],
        )
        # The tokenizer is referenced relative to the artifact directory, so
        # the artifact stays relocatable. Models without a `tokenizer.json`
        # (e.g. tiktoken-only checkpoints) carry no `tokenizer_config`.
        serving_cfg = meta_dict.setdefault("serving_config", {})
        if (export_dir / "tokenizer.json").is_file():
            serving_cfg["tokenizer_config"] = {
                "source": {"TokenizerPath": {"path": "tokenizer.json"}}
            }
        else:
            serving_cfg.pop("tokenizer_config", None)
            logger.warning(
                "[TpuCompilerAdaptor] No tokenizer.json bundled in %s; "
                "metadata.json carries no tokenizer_config.",
                export_dir,
            )
        _write_abstract_params(export_dir, meta_dict, weights_dict)
        tmp_path = export_dir / f"metadata.json.{os.getpid()}.tmp"
        tmp_path.write_text(json.dumps(meta_dict, indent=2))
        os.replace(tmp_path, meta_path)

    parallel_cfg = getattr(runner, "parallel_config", None)
    world_size = int(getattr(parallel_cfg, "world_size", 1) or 1)
    dp_size = int(getattr(parallel_cfg, "data_parallel_size", 1) or 1)
    if world_size <= 1 and dp_size <= 1:
        logger.info(
            "[TpuCompilerAdaptor] Exported all StableHLO modules "
            "(backbone + sampling) to %s; exiting early.",
            export_dir,
        )
        sys.exit(0)
