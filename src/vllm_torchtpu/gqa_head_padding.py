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
"""Generic GQA query-head zero-padding utilities for Tensor Parallelism.

Enables GQA layers whose total query head count is not divisible by ``tp_size``
(e.g., 28 Q heads and 4 KV heads -> 7 Q heads per KV group at ``TP=8`` on
``tpu7x-2x2x1``) by zero-padding query heads per KV group in ``qkv_proj`` and
``o_proj`` while preserving mathematical equivalence after the ``o_proj``
``RowParallelLinear`` all-reduce.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to, get_num_kv_heads_by_tp

logger = init_logger(__name__)


@dataclass(frozen=True)
class GqaPaddingSpec:
    """Specification for per-KV-group query head zero-padding."""

    num_kv_heads: int
    orig_q_per_kv: int
    padded_q_per_kv: int
    head_dim: int

    @property
    def orig_total_heads(self) -> int:
        return self.num_kv_heads * self.orig_q_per_kv

    @property
    def padded_total_heads(self) -> int:
        return self.num_kv_heads * self.padded_q_per_kv


def compute_padded_gqa_heads(
    total_num_heads: int,
    total_num_kv_heads: int,
    tp_size: int,
) -> tuple[int, int, int]:
    """Computes per-KV-group padded Q head counts when heads % tp_size != 0.

    Unlike ``get_padded_num_heads`` in ``vllm_torchtpu.utils`` (which replicates
    heads when ``total_num_heads < tp_size`` and asserts divisibility when
    ``total_num_heads >= tp_size``), this helper pads query heads per KV group
    when ``total_num_heads > tp_size`` and KV heads are replicated across TP
    ranks.

    Returns:
        ``(padded_total_num_heads, orig_q_per_kv, padded_q_per_kv)``, or
        ``(total_num_heads, 0, 0)`` when no padding is needed or supported.
    """
    if tp_size <= 0 or total_num_heads <= 0 or total_num_heads % tp_size == 0:
        return total_num_heads, 0, 0
    if (
        total_num_kv_heads <= 0
        or total_num_heads % total_num_kv_heads != 0
        or tp_size % total_num_kv_heads != 0
    ):
        return total_num_heads, 0, 0
    orig_q_per_kv = total_num_heads // total_num_kv_heads
    replicated_kv_heads = get_num_kv_heads_by_tp(total_num_kv_heads, tp_size)
    padded_q_per_kv = align_to(orig_q_per_kv, replicated_kv_heads // total_num_kv_heads)
    return total_num_kv_heads * padded_q_per_kv, orig_q_per_kv, padded_q_per_kv


def update_gqa_head_padding_config(model_config: Any, parallel_config: Any) -> None:
    """Updates ``model_config.model_arch_config`` with the padded GQA head count.

    When ``total_num_attention_heads`` is not divisible by
    ``tensor_parallel_size`` and ``decode_context_parallel_size == 1``, sets
    ``model_arch_config.total_num_attention_heads`` to the padded count and
    records ``model_config.orig_total_num_attention_heads``. Idempotent.
    """
    if parallel_config.decode_context_parallel_size > 1:
        return
    arch_config = model_config.model_arch_config
    orig_heads = getattr(
        model_config,
        "orig_total_num_attention_heads",
        arch_config.total_num_attention_heads,
    )
    tp_size = parallel_config.tensor_parallel_size
    num_kv_heads = model_config.get_total_num_kv_heads()
    padded_heads, orig_q, padded_q = compute_padded_gqa_heads(
        orig_heads, num_kv_heads, tp_size
    )
    if padded_heads == orig_heads or padded_heads % tp_size != 0:
        return
    if arch_config.total_num_attention_heads == padded_heads:
        return
    logger.info(
        "TPU GQA head padding enabled: total_num_heads=%d -> %d "
        "(num_kv_heads=%d, q_per_kv=%d -> %d, tp_size=%d)",
        orig_heads,
        padded_heads,
        num_kv_heads,
        orig_q,
        padded_q,
        tp_size,
    )
    object.__setattr__(model_config, "orig_total_num_attention_heads", orig_heads)
    object.__setattr__(arch_config, "total_num_attention_heads", padded_heads)


def _spans_heads(tensor: torch.Tensor, dim: int, num_heads: int, head_dim: int) -> bool:
    """Whether ``tensor`` has exactly ``num_heads * head_dim`` entries along ``dim``."""
    return tensor.dim() > dim and tensor.shape[dim] == num_heads * head_dim


def _unsupported_shape_error(
    name: str, tensor: torch.Tensor, dim: int, expected_heads: str, head_dim: int
) -> ValueError:
    return ValueError(
        f"Unsupported {name} shape={tuple(tensor.shape)} along dim={dim}: "
        f"expected {expected_heads} heads x head_dim={head_dim}. Packed or "
        "quantized Q-head layouts are not supported by GQA head padding."
    )


def pad_gqa_tensor_along_dim(
    tensor: torch.Tensor, dim: int, spec: GqaPaddingSpec
) -> torch.Tensor:
    """Zero-pads a Q or O projection tensor per KV head group along ``dim``."""
    group_width = spec.orig_q_per_kv * spec.head_dim
    grouped = tensor.unflatten(dim, (spec.num_kv_heads, group_width))
    pad_shape = list(grouped.shape)
    pad_shape[dim + 1] = (spec.padded_q_per_kv - spec.orig_q_per_kv) * spec.head_dim
    padded = torch.cat([grouped, grouped.new_zeros(pad_shape)], dim=dim + 1)
    return padded.flatten(dim, dim + 1).contiguous()


def _pad_q_heads(
    tensor: torch.Tensor, dim: int, spec: GqaPaddingSpec, name: str
) -> torch.Tensor:
    """Pads unpadded Q-head tensors; passes already-padded ones through."""
    if _spans_heads(tensor, dim, spec.orig_total_heads, spec.head_dim):
        return pad_gqa_tensor_along_dim(tensor, dim, spec)
    # Idempotent re-entry: _load_fused_module_from_checkpoint slices an already
    # padded fused QKV tensor and delegates back to weight_loader_v2.
    if _spans_heads(tensor, dim, spec.padded_total_heads, spec.head_dim):
        return tensor
    raise _unsupported_shape_error(
        name,
        tensor,
        dim,
        f"{spec.orig_total_heads} (unpadded) or {spec.padded_total_heads} (padded)",
        spec.head_dim,
    )


def pad_qkv_weight(
    param: Any,
    loaded_weight: torch.Tensor,
    loaded_shard_id: str | None,
    spec: GqaPaddingSpec,
) -> torch.Tensor:
    """Pads Q weights in sharded ('q') or fused (``loaded_shard_id=None``) form."""
    output_dim = getattr(param, "output_dim", 0)
    if output_dim is None or loaded_weight.dim() <= output_dim:
        return loaded_weight
    if loaded_shard_id == "q":
        return _pad_q_heads(loaded_weight, output_dim, spec, "Q projection weight")
    if loaded_shard_id is not None:
        return loaded_weight

    total_heads = spec.orig_total_heads + 2 * spec.num_kv_heads
    if not _spans_heads(loaded_weight, output_dim, total_heads, spec.head_dim):
        raise _unsupported_shape_error(
            "fused QKV weight",
            loaded_weight,
            output_dim,
            f"{total_heads} (unpadded Q+K+V)",
            spec.head_dim,
        )
    kv_size = spec.num_kv_heads * spec.head_dim
    q_w, k_w, v_w = torch.split(
        loaded_weight,
        [spec.orig_total_heads * spec.head_dim, kv_size, kv_size],
        dim=output_dim,
    )
    q_w = pad_gqa_tensor_along_dim(q_w, output_dim, spec)
    return torch.cat([q_w, k_w, v_w], dim=output_dim).contiguous()


def pad_o_proj_weight(
    param: Any, loaded_weight: torch.Tensor, spec: GqaPaddingSpec
) -> torch.Tensor:
    """Pads ``o_proj`` weights along ``input_dim`` when defined on ``param``."""
    input_dim = getattr(param, "input_dim", None)
    if input_dim is None or loaded_weight.dim() <= input_dim:
        return loaded_weight
    return _pad_q_heads(loaded_weight, input_dim, spec, "o_proj weight")


def _wrap_layer_weight_loaders(
    layer: Any,
    make_wrapped_loader: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    """Wraps ``weight_loader`` and ``weight_loader_v2`` on ``layer`` and params.

    ``layer`` is a vLLM linear layer; its loaders are dynamic attributes.
    """
    orig_loader = layer.weight_loader
    orig_loader_v2 = getattr(layer, "weight_loader_v2", None)

    wrapped_loader = make_wrapped_loader(orig_loader)
    layer.weight_loader = wrapped_loader

    wrapped_loader_v2 = None
    if orig_loader_v2 is not None:
        wrapped_loader_v2 = make_wrapped_loader(orig_loader_v2)
        layer.weight_loader_v2 = wrapped_loader_v2

    for param in layer.parameters(recurse=False):
        if hasattr(param, "weight_loader"):
            if wrapped_loader_v2 is not None and param.weight_loader == orig_loader_v2:
                param.weight_loader = wrapped_loader_v2
            else:
                param.weight_loader = wrapped_loader


def install_qkv_padded_loader(layer: Any, spec: GqaPaddingSpec) -> None:
    """Installs Q-head zero-padding weight loaders on ``QKVParallelLinear``."""

    def _make_loader(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        def _padded_loader(
            param: Any,
            loaded_weight: torch.Tensor,
            loaded_shard_id: str | None = None,
        ) -> Any:
            padded = pad_qkv_weight(param, loaded_weight, loaded_shard_id, spec)
            return orig_fn(param, padded, loaded_shard_id)

        return _padded_loader

    _wrap_layer_weight_loaders(layer, _make_loader)


def install_o_proj_padded_loader(layer: Any, spec: GqaPaddingSpec) -> None:
    """Installs Q-head zero-padding weight loaders on ``RowParallelLinear``."""

    def _make_loader(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        def _padded_loader(param: Any, loaded_weight: torch.Tensor) -> Any:
            return orig_fn(param, pad_o_proj_weight(param, loaded_weight, spec))

        return _padded_loader

    _wrap_layer_weight_loaders(layer, _make_loader)
