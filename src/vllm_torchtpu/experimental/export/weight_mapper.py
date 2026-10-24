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
"""Sharding rules for the weights of an exported model."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any


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

    # FP8 requantization replaces a layer's parameters, dropping their shard
    # attributes. A row-parallel layer's scales are computed over the rank's own
    # input slice, so they differ per rank and must not be declared replicated.
    if weight_name.endswith(
        (
            "o_proj.weight_scale",
            "down_proj.weight_scale",
            "out_proj.weight_scale",
            "wo_b.weight_scale",
        )
    ):
        return ["tp"] + [None] * (rank - 1)

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
                "A_log",
                "dt_bias",
                "fused_qkvb_proj",
                "fused_fa_ga_proj",
                "f_b_proj",
                "g_b_proj",
                "attn_sink",
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
