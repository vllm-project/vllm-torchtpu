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
from collections.abc import Iterable, Sequence
from typing import Any

from vllm_torchtpu import envs

# Attributes set by `vllm_torchtpu.layers.adapter.linear_common` on linear and
# embedding layers after weight loading (defined here so CPU export paths do
# not import `torch_tpu`).
KEEP_VLLM_LAYOUT_ATTR = "_tpu_keep_vllm_weight_layout"
WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"
# Attribute on a loaded target model holding its MTP draft module, if any.
MTP_DRAFT_ATTR = "_export_mtp_draft"


def _resolve_submod_and_param(
    weight_name: str, model: Any = None
) -> tuple[Any, Any, str, str]:
    """Resolves `(submod, param_or_buf, mod_path, attr)` from `model`."""
    wrapper = model
    if model is not None and not callable(getattr(model, "get_submodule", None)):
        model = getattr(model, "model", None)
    if model is None or not weight_name:
        return None, None, "", weight_name
    mod_path, attr = (
        weight_name.rsplit(".", 1) if "." in weight_name else ("", weight_name)
    )
    submod = None
    inner_model = getattr(model, "model", None)
    candidate_roots = [
        root
        for root in (model, inner_model)
        if root is not None and callable(getattr(root, "get_submodule", None))
    ]
    if mod_path and candidate_roots:
        for root in candidate_roots:
            for candidate_path in (mod_path, mod_path.removeprefix("model.")):
                with contextlib.suppress(Exception):
                    submod = root.get_submodule(candidate_path)
                    if submod is not None:
                        break
            if submod is not None:
                break
        if submod is None and (mod_path == "mtp" or mod_path.startswith("mtp.")):
            draft = (
                getattr(model, MTP_DRAFT_ATTR, None)
                or getattr(inner_model, MTP_DRAFT_ATTR, None)
                or getattr(wrapper, MTP_DRAFT_ATTR, None)
                or getattr(wrapper, "_mtp_draft", None)
            )
            draft_get_submodule = getattr(draft, "get_submodule", None)
            if callable(draft_get_submodule):
                suffix = mod_path.removeprefix("mtp").lstrip(".")
                draft_candidates = (
                    (f"model.{suffix}", suffix) if suffix else ("model", "")
                )
                for candidate_path in draft_candidates:
                    with contextlib.suppress(Exception):
                        submod = draft_get_submodule(candidate_path)
                        if submod is not None:
                            break
    elif not mod_path:
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
    keep_vllm_layout = bool(getattr(submod, KEEP_VLLM_LAYOUT_ATTR, False))
    weight_flipped = bool(getattr(submod, WEIGHT_FLIPPED_ATTR, False))
    submodule_class_names = (
        {cls.__name__ for cls in type(submod).__mro__} if submod is not None else set()
    )
    is_tp_replicated_submodule = submod is not None and (
        bool(getattr(submod, "disable_tp", False))
        or "ReplicatedLinear" in submodule_class_names
    )
    submod_tp = int(getattr(submod, "tp_size", 0) or 0) if submod is not None else 0
    tp_size = (
        1 if is_tp_replicated_submodule else max(submod_tp, int(tp_size_hint or 1))
    )

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
            if (
                shard_attr == "output_dim"
                and "RowParallelLinear" in submodule_class_names
                and not (attr == "weight_scale" and rank in (1, 2))
            ):
                continue
            if (
                shard_attr == "input_dim"
                and "ColumnParallelLinear" in submodule_class_names
            ):
                continue
            dim_val = getattr(param_obj, shard_attr, None)
            if isinstance(dim_val, int) and 0 <= dim_val < rank:
                if axis_name == "tp" and tp_size <= 1:
                    continue
                sharding_dims: list[str | None] = [None] * rank
                if shard_attr == "output_dim" and rank == 2 and attr == "weight":
                    is_embedding_or_lm_head = hasattr(submod, "num_embeddings") or (
                        submod is not None
                        and "embedding" in type(submod).__name__.lower()
                    )
                    if is_embedding_or_lm_head:
                        target_dim = 1 if weight_flipped else 0
                    else:
                        target_dim = dim_val if keep_vllm_layout else (1 - dim_val)
                    sharding_dims[target_dim] = "tp"
                elif shard_attr == "input_dim" and rank == 2 and attr == "weight":
                    target_dim = dim_val if keep_vllm_layout else (1 - dim_val)
                    sharding_dims[target_dim] = "tp"
                elif (
                    shard_attr == "output_dim" and rank == 4 and attr == "weight_scale"
                ):
                    sharding_dims[3] = "tp"
                elif shard_attr == "input_dim" and rank == 4 and attr == "weight_scale":
                    sharding_dims[1] = "tp"
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

    if "ColumnParallelLinear" in submodule_class_names:
        if attr == "weight" and rank == 2:
            return ["tp", None] if keep_vllm_layout else [None, "tp"]
        if attr == "bias" and rank == 1:
            return ["tp"]
        if attr == "weight_scale":
            if rank == 4:
                return [None, None, None, "tp"]
            return ["tp"] + [None] * (rank - 1)
        if attr == "weight_scale_inv" and rank == 2:
            return ["tp", None]

    if "RowParallelLinear" in submodule_class_names:
        if attr == "weight" and rank == 2:
            return [None, "tp"] if keep_vllm_layout else ["tp", None]
        if attr == "bias" and rank == 1:
            return [None]
        if attr == "weight_scale":
            if rank == 4:
                return [None, "tp", None, None]
            return ["tp"] + [None] * (rank - 1)
        if attr == "weight_scale_inv" and rank == 2:
            return [None, "tp"]

    # FP8 requantization replaces or reshapes a layer's scale parameters. A
    # row-parallel layer's scales are computed over the rank's own input slice,
    # so they differ per rank: rank-4 block scales `[1, n_in_blocks, 1, n_out]`
    # shard axis 1, while rank-1/2 per-channel scales shard axis 0.
    if weight_name.endswith(
        (
            "o_proj.weight_scale",
            "down_proj.weight_scale",
            "out_proj.weight_scale",
            "wo_b.weight_scale",
            "post_expert_proj.weight_scale",
        )
    ):
        if rank == 4:
            return [None, "tp", None, None]
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
                # The MTP predictor's column-parallel fc (Qwen3.5, Qwen3-Next).
                ".fc.",
                # Muse Spark's MoE projection to the expert width.
                ".pre_expert_proj.",
            )
        ):
            return ["tp"]
        return [None]

    if rank == 2:
        is_qkvz_weight = weight_name == "in_proj_qkvz.weight" or weight_name.endswith(
            ".in_proj_qkvz.weight"
        )
        if is_qkvz_weight:
            if submod is not None:
                return ["tp", None] if keep_vllm_layout else [None, "tp"]
            if envs.TPU_GDN_QKVZ_KERNEL_LAYOUT:
                raise NotImplementedError(
                    "infer_weight_sharding needs `model` to tell whether "
                    "in_proj_qkvz kept vLLM's [n_out, n_in] layout (e.g. under "
                    "prefill context parallelism) or flipped to (k, n) with "
                    "TPU_GDN_QKVZ_KERNEL_LAYOUT."
                )
            return ["tp", None]
        is_lm_head_weight = weight_name == "lm_head.weight" or weight_name.endswith(
            ".lm_head.weight"
        )
        if is_lm_head_weight and weight_flipped:
            return [None, "tp"]
        if (
            weight_name.endswith("embed_tokens.weight")
            or is_lm_head_weight
            or weight_name.endswith(".o_proj.weight")
            or weight_name.endswith(".down_proj.weight")
            or weight_name.endswith(".out_proj.weight")
            or weight_name.endswith(".wo_b.weight")
            # Muse Spark's MoE projection back from the expert width.
            or weight_name.endswith(".post_expert_proj.weight")
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
                ".fc.",
                ".pre_expert_proj.",
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
        if weight_name.endswith(".weight_scale") and (
            weight_name == "lm_head.weight_scale"
            or weight_name.endswith(".lm_head.weight_scale")
            or any(
                column_parallel_key in weight_name
                for column_parallel_key in (
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
                    "in_proj_qkvz",
                    "in_proj_ba",
                    "fused_qkvb_proj",
                    "fused_fa_ga_proj",
                    "f_b_proj",
                    "g_b_proj",
                    "compressor.fused_wkv_wgate",
                    ".fc.",
                    ".pre_expert_proj.",
                )
            )
        ):
            return [None, None, None, "tp"]
        return [None, None, None, None]

    return [None] * rank


# Embedding names a tied LM head shares, most specific first.
_EMBEDDING_NAMES = (
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "transformer.wte.weight",
)


def tied_lm_head_alias(names: Iterable[str]) -> tuple[str, str] | None:
    """`(lm_head name, embedding name)` when the LM head is tied to the embedding.

    None for an untied head. A tied head has no parameter of its own, so no
    `lm_head.weight` is among `names`; it then projects with the language model's
    embedding. A multimodal model nests the language model, so
    `language_model.model.embed_tokens.weight` gives `language_model.lm_head.weight`.
    """
    names = list(names)
    if any(n == "lm_head.weight" or n.endswith(".lm_head.weight") for n in names):
        return None
    found = []
    for n in names:
        for rank, emb in enumerate(_EMBEDDING_NAMES):
            if n == emb or n.endswith("." + emb):
                found.append(
                    (rank, len(n), f"{n[: len(n) - len(emb)]}lm_head.weight", n)
                )
                break
    # The language model's embedding: the most specific name (`model.embed_tokens`,
    # not an MTP head's `mtp.embed_tokens`), then the outermost.
    return min(found)[2:] if found else None
