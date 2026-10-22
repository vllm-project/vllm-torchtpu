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
"""Safetensors checkpoint metadata inspection and weight mapping rules."""

from __future__ import annotations

import contextlib
import json
import re
import struct
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any


def read_safetensors_header(file_path: str | Path) -> dict[str, Any]:
    """Reads the JSON header of a `.safetensors` file without loading tensor bytes."""
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"Invalid safetensors file (too short): {file_path}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not a JSON object: {file_path}")
    return {k: v for k, v in header.items() if k != "__metadata__"}


def read_hf_safetensors_shapes(
    hf_dir: str | Path | None, tensor_to_file: Mapping[str, str]
) -> dict[str, list[int]]:
    """Full shape of every checkpoint tensor, read from the safetensors headers."""
    if not hf_dir or not tensor_to_file:
        return {}
    shapes: dict[str, list[int]] = {}
    for file_name in sorted(set(tensor_to_file.values())):
        path = Path(hf_dir) / file_name
        if not path.is_file():
            continue
        for tensor, info in read_safetensors_header(path).items():
            if isinstance(info, dict) and "shape" in info:
                shapes[tensor] = [int(d) for d in info["shape"]]
    return shapes


def read_hf_safetensors_metadata(
    hf_dir: str | Path | None,
) -> tuple[dict[str, str], str]:
    """Maps each Safetensors tensor in a HuggingFace checkpoint dir to its file.

    Reads `model.safetensors.index.json` if present, or scans `.safetensors` file
    headers (reading only the 8-byte length prefix + JSON header, loading zero
    weight bytes).

    Returns:
        Tuple of `(tensor_to_file, default_path)`.
    """
    if not hf_dir:
        return {}, "model.safetensors"
    root = Path(hf_dir)
    if not root.is_dir():
        return {}, "model.safetensors"

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        with contextlib.suppress(Exception):
            with open(index_path) as f:
                idx_doc = json.load(f)
            weight_map = idx_doc.get("weight_map")
            if isinstance(weight_map, dict) and weight_map:
                tensor_to_file = {str(k): str(v) for k, v in weight_map.items()}
                first_file = next(iter(tensor_to_file.values()), "model.safetensors")
                return tensor_to_file, first_file

    st_files = sorted(root.glob("*.safetensors"))
    tensor_to_file = {}
    default_path = st_files[0].name if st_files else "model.safetensors"
    for st_file in st_files:
        with contextlib.suppress(Exception):
            header = read_safetensors_header(st_file)
            for tensor_name in header:
                tensor_to_file[tensor_name] = st_file.name
    return tensor_to_file, default_path


def resolve_safetensor_name(
    w_name: str,
    valid_keys: Collection[str],
) -> str | None:
    """Resolves an internal PyTorch parameter name to its corresponding Safetensors key.

    Applies a prioritized rule chain to bridge PyTorch module hierarchical naming
    with upstream HuggingFace Safetensors checkpoint key naming conventions.
    """
    if not valid_keys:
        return None

    # Rule 0: Exact match
    if w_name in valid_keys:
        return w_name

    # Rule 1: Tied embeddings (lm_head.weight)
    if w_name == "lm_head.weight":
        for cand in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "transformer.wte.weight",
            "embed_tokens.weight",
        ):
            if cand in valid_keys:
                return cand

    # Rule 2: Special case casing (e.g. a_log -> A_log)
    if ".linear_attn.a_log" in w_name:
        casing_cand = w_name.replace(".linear_attn.a_log", ".linear_attn.A_log")
        res = resolve_safetensor_name(casing_cand, valid_keys)
        if res is not None:
            return res

    # Rule 3: MoE routing gate infix remapping
    if ".mlp.experts.gate." in w_name:
        moe_cand = w_name.replace(".mlp.experts.gate.", ".mlp.gate.")
        res = resolve_safetensor_name(moe_cand, valid_keys)
        if res is not None:
            return res

    if ".block_sparse_moe.experts.gate." in w_name:
        moe_cand = w_name.replace(
            ".block_sparse_moe.experts.gate.", ".block_sparse_moe.gate."
        )
        res = resolve_safetensor_name(moe_cand, valid_keys)
        if res is not None:
            return res

    # Rule 4: Shared expert naming remapping
    if ".mlp.experts.shared_experts.layer." in w_name:
        shared_cand = w_name.replace(
            ".mlp.experts.shared_experts.layer.", ".mlp.shared_expert."
        )
        if ".expert_gate." in shared_cand:
            shared_cand = shared_cand.replace(
                ".mlp.shared_expert.expert_gate.", ".mlp.shared_expert_gate."
            )
        res = resolve_safetensor_name(shared_cand, valid_keys)
        if res is not None:
            return res

    # Rule 5: Multimodal / VLM decoupled language model prefix
    if w_name.startswith("model."):
        vlm_cand = w_name.replace("model.", "model.language_model.", 1)
        if vlm_cand in valid_keys:
            return vlm_cand
        lm_cand = w_name.replace("model.", "language_model.", 1)
        if lm_cand in valid_keys:
            return lm_cand

    if w_name.startswith("language_model."):
        vlm_cand = f"model.{w_name}"
        if vlm_cand in valid_keys:
            return vlm_cand

    # Rule 6: General prefix normalization
    if not w_name.startswith("model."):
        model_prefixed = f"model.{w_name}"
        res = resolve_safetensor_name(model_prefixed, valid_keys)
        if res is not None:
            return res

    return None


def resolve_source_tensor_and_path(
    w_name: str,
    tensor_to_file: dict[str, str],
    default_path: str = "",
) -> tuple[str, str]:
    """Resolves the Safetensors source tensor identifier and shard file path."""
    resolved = resolve_safetensor_name(w_name, tensor_to_file)
    if resolved is not None:
        return resolved, tensor_to_file.get(resolved, default_path)
    return w_name, default_path


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


def infer_weight_sharding(
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
        # Handle explicit sharding spec objects if passed
        spec = getattr(explicit_sharding, "spec", None)
        if spec is not None:
            rank = len(shape)
            spec_tuple = tuple(spec)
            axes: list[str | None] = []
            for entry in spec_tuple + (None,) * (rank - len(spec_tuple)):
                if isinstance(entry, tuple):
                    entry = entry[0] if entry else None
                axes.append(str(entry) if entry is not None else None)
            return axes
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
                "kv_b_proj",
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
        if "experts." in weight_name and "gate_up_proj" in weight_name:
            return [None, None, "tp"]
        if "experts." in weight_name and "down_proj" in weight_name:
            return [None, "tp", None]
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


def build_generic_weight_sources(
    weight_name: str,
    shape: Sequence[int | str],
    model: Any = None,
) -> list[dict[str, Any]] | None:
    """Builds declarative `sources` from `model` and `packed_modules_mapping`."""
    if not weight_name or any(not isinstance(d, int) for d in shape):
        return None
    int_shape = [int(d) for d in shape if isinstance(d, int)]

    submod, _, mod_path, attr = _resolve_submod_and_param(weight_name, model)
    if submod is not None and "." in weight_name:
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
                        src_prefix = f"{parent_path}.{sub_n}" if parent_path else sub_n
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
                        src_prefix = f"{parent_path}.{sub_n}" if parent_path else sub_n
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


def _expand_equal_two_way_sources(
    first_resolved: str,
    second_resolved: str,
    shape: Sequence[int],
    tensor_to_file: dict[str, str],
    default_path: str,
) -> list[dict[str, Any]]:
    """Builds 2-way concat weight sources for `[in, 2*half]` or `[2*half, in]`."""
    if shape[0] >= shape[1]:
        in_dim = shape[0]
        half = shape[1] // 2
        return [
            {
                "path": tensor_to_file.get(first_resolved, default_path),
                "source_tensor": first_resolved,
                "source_shape": [half, in_dim],
                "transpose": [1, 0],
                "dst_slices": [{"dim": 1, "start": 0, "length": half}],
            },
            {
                "path": tensor_to_file.get(second_resolved, default_path),
                "source_tensor": second_resolved,
                "source_shape": [half, in_dim],
                "transpose": [1, 0],
                "dst_slices": [{"dim": 1, "start": half, "length": half}],
            },
        ]
    half = shape[0] // 2
    in_dim = shape[1]
    return [
        {
            "path": tensor_to_file.get(first_resolved, default_path),
            "source_tensor": first_resolved,
            "source_shape": [half, in_dim],
            "dst_slices": [{"dim": 0, "start": 0, "length": half}],
        },
        {
            "path": tensor_to_file.get(second_resolved, default_path),
            "source_tensor": second_resolved,
            "source_shape": [half, in_dim],
            "dst_slices": [{"dim": 0, "start": half, "length": half}],
        },
    ]


def expand_or_resolve_weight_sources(
    w_name: str,
    w_spec: dict[str, Any],
    tensor_to_file: dict[str, str],
    default_path: str = "",
) -> list[dict[str, Any]]:
    """Expands composite/packed target weights or resolves 1-to-1 weight sources."""
    shape = [int(d) for d in w_spec.get("shape", []) if isinstance(d, int)]

    # 0. Sources derived from the live module layout are authoritative. The
    # name-based rules below guess the packed orientation from the target shape
    # alone, which is ambiguous (e.g. gate_up_proj `[in, 2*inter]` with
    # `in < 2*inter` looks like `[2*half, in]`).
    existing_sources = w_spec.get("sources")
    if isinstance(existing_sources, list) and existing_sources:
        resolved_sources: list[dict[str, Any]] = []
        for src in existing_sources:
            if not isinstance(src, dict):
                continue
            entry = dict(src)
            raw = str(entry.get("source_tensor") or w_name)
            st_name, path = resolve_source_tensor_and_path(
                raw, tensor_to_file, default_path
            )
            entry["source_tensor"] = st_name
            entry["path"] = path
            if "source_shape" not in entry and shape:
                entry["source_shape"] = shape
            resolved_sources.append(entry)
        if resolved_sources:
            return resolved_sources

    # 1. GDN Attention: in_proj_qkvz.weight [20480, hidden_dim]
    if ".linear_attn.in_proj_qkvz.weight" in w_name and len(shape) == 2:
        in_dim = shape[1]
        qkv_cand_name = w_name.replace(".in_proj_qkvz.weight", ".in_proj_qkv.weight")
        z_cand_name = w_name.replace(".in_proj_qkvz.weight", ".in_proj_z.weight")
        qkv_resolved = resolve_safetensor_name(qkv_cand_name, tensor_to_file)
        z_resolved = resolve_safetensor_name(z_cand_name, tensor_to_file)
        if qkv_resolved is not None and z_resolved is not None:
            return [
                {
                    "path": tensor_to_file.get(qkv_resolved, default_path),
                    "source_tensor": qkv_resolved,
                    "source_shape": [12288, in_dim],
                    "dst_slices": [{"dim": 0, "start": 0, "length": 12288}],
                },
                {
                    "path": tensor_to_file.get(z_resolved, default_path),
                    "source_tensor": z_resolved,
                    "source_shape": [8192, in_dim],
                    "dst_slices": [{"dim": 0, "start": 12288, "length": 8192}],
                },
            ]

    # 2. GDN Attention: in_proj_ba.weight
    if ".linear_attn.in_proj_ba.weight" in w_name and len(shape) == 2:
        b_cand_name = w_name.replace(
            ".linear_attn.in_proj_ba.weight", ".linear_attn.in_proj_b.weight"
        )
        a_cand_name = w_name.replace(
            ".linear_attn.in_proj_ba.weight", ".linear_attn.in_proj_a.weight"
        )
        b_resolved = resolve_safetensor_name(b_cand_name, tensor_to_file)
        a_resolved = resolve_safetensor_name(a_cand_name, tensor_to_file)
        if b_resolved is not None and a_resolved is not None:
            return _expand_equal_two_way_sources(
                b_resolved, a_resolved, shape, tensor_to_file, default_path
            )

    # 3. Shared Experts or Standard MLP: gate_up_proj.weight
    if "gate_up_proj.weight" in w_name and len(shape) == 2:
        gate_cand = w_name.replace("gate_up_proj.weight", "gate_proj.weight")
        up_cand = w_name.replace("gate_up_proj.weight", "up_proj.weight")
        gate_resolved = resolve_safetensor_name(gate_cand, tensor_to_file)
        up_resolved = resolve_safetensor_name(up_cand, tensor_to_file)
        if gate_resolved is not None and up_resolved is not None:
            return _expand_equal_two_way_sources(
                gate_resolved, up_resolved, shape, tensor_to_file, default_path
            )

    # 4. MoE Routed Experts: routed_experts.w13_weight
    if ".mlp.experts.routed_experts.w13_weight" in w_name and len(shape) == 3:
        num_local_experts, hidden_dim, total_inter = shape
        intermediate_dim = total_inter // 2
        layer_match = re.search(r"layers\.(\d+)\.", w_name)
        if layer_match:
            layer_idx = layer_match.group(1)
            sources: list[dict[str, Any]] = []
            matched_all = True
            for e in range(num_local_experts):
                gate_key_cand = (
                    f"model.layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
                )
                up_key_cand = f"model.layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"
                gate_res = resolve_safetensor_name(gate_key_cand, tensor_to_file)
                up_res = resolve_safetensor_name(up_key_cand, tensor_to_file)
                if gate_res is None or up_res is None:
                    matched_all = False
                    break
                sources.append(
                    {
                        "path": tensor_to_file.get(gate_res, default_path),
                        "source_tensor": gate_res,
                        "source_shape": [1, intermediate_dim, hidden_dim],
                        "transpose": [0, 2, 1],
                        "dst_slices": [
                            {"dim": 0, "start": e, "length": 1},
                            {"dim": 2, "start": 0, "length": intermediate_dim},
                        ],
                    }
                )
                sources.append(
                    {
                        "path": tensor_to_file.get(up_res, default_path),
                        "source_tensor": up_res,
                        "source_shape": [1, intermediate_dim, hidden_dim],
                        "transpose": [0, 2, 1],
                        "dst_slices": [
                            {"dim": 0, "start": e, "length": 1},
                            {
                                "dim": 2,
                                "start": intermediate_dim,
                                "length": intermediate_dim,
                            },
                        ],
                    }
                )
            if matched_all and sources:
                return sources

    # 5. MoE Routed Experts: routed_experts.w2_weight
    if ".mlp.experts.routed_experts.w2_weight" in w_name and len(shape) == 3:
        num_local_experts, intermediate_dim, hidden_dim = shape
        layer_match = re.search(r"layers\.(\d+)\.", w_name)
        if layer_match:
            layer_idx = layer_match.group(1)
            sources = []
            matched_all = True
            for e in range(num_local_experts):
                down_key_cand = (
                    f"model.layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
                )
                down_res = resolve_safetensor_name(down_key_cand, tensor_to_file)
                if down_res is None:
                    matched_all = False
                    break
                sources.append(
                    {
                        "path": tensor_to_file.get(down_res, default_path),
                        "source_tensor": down_res,
                        "source_shape": [1, hidden_dim, intermediate_dim],
                        "transpose": [0, 2, 1],
                        "dst_slices": [{"dim": 0, "start": e, "length": 1}],
                    }
                )
            if matched_all and sources:
                return sources

    # 6. Single 1-to-1 default resolution
    st_name, path = resolve_source_tensor_and_path(w_name, tensor_to_file, default_path)
    return [
        {
            "path": path,
            "source_tensor": st_name,
            "source_shape": shape,
        }
    ]
