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
"""Explicit native MTP identity and byte-exact TP/EP ownership.

MTP precision is independent of the target execution profile. The pinned NVIDIA
checkpoint stores every MTP tensor as BF16, including all routed experts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from . import MODEL_REVISION, Config


@dataclass(frozen=True)
class MTPConfig:
    target: Config
    layers: int = 1
    dedicated_embeddings: bool = False
    weight_dtype: str = "BF16"
    kv_dtype: str = "BF16"
    hidden_representation: str = "target_post_final_norm"
    position_alignment: str = "predecessor"

    def __post_init__(self):
        if self.layers != 1 or self.dedicated_embeddings:
            raise ValueError("Initial MTP support requires one layer and shared embedding/head")
        if self.weight_dtype != "BF16" or self.kv_dtype != "BF16":
            raise ValueError("Native MTP requires BF16 weights and its own BF16 KV cache")
        if self.hidden_representation != "target_post_final_norm":
            raise ValueError("MTP requires the target's post-final-normalization hidden state")
        if self.position_alignment != "predecessor":
            raise ValueError("MTP positions refer to the predecessor target hidden state")

    @classmethod
    def from_checkpoint(cls, checkpoint):
        raw = (
            dict(checkpoint)
            if isinstance(checkpoint, Mapping)
            else json.loads((Path(checkpoint) / "config.json").read_text())
        )
        if raw.get("mtp_num_hidden_layers") != 1:
            raise ValueError("Checkpoint must explicitly declare one MTP layer")
        if raw.get("mtp_use_dedicated_embeddings") is not False:
            raise ValueError("Checkpoint must explicitly share MTP embeddings")
        target = Config.from_checkpoint(raw)
        if target.quantization != "nvfp4":
            raise ValueError("Initial MTP checkpoint support is the pinned NVIDIA NVFP4 model")
        ignored = set(raw["quantization_config"].get("ignore", ()))
        if not {"mtp*", "mtp.layers.0*"}.issubset(ignored):
            raise ValueError("Native MTP must be explicitly excluded from target quantization")
        return cls(target)

    @property
    def checkpoint_revision(self):
        return MODEL_REVISION


@dataclass(frozen=True)
class OwnedTensor:
    """One native tensor slice; transpose only, with an optional expert slot."""

    source: str
    destination: str
    full_shape: tuple[int, ...]
    starts: tuple[int, ...]
    sizes: tuple[int, ...]
    transpose: bool = False
    expert_slot: int | None = None

    @property
    def selection(self):
        return tuple(
            slice(start, start + size) for start, size in zip(self.starts, self.sizes, strict=True)
        )

    @property
    def byte_count(self):
        return 2 * math.prod(self.sizes)


def native_tensor_shapes(config):
    """All and only checkpoint MTP tensors; no shared embedding/head payload."""
    c = config.target
    root = "mtp.layers.0."
    shapes = {
        "mtp.fc.weight": (c.dim, 2 * c.dim),
        "mtp.norm.weight": (c.dim,),
        "mtp.pre_fc_norm_hidden.weight": (c.dim,),
        "mtp.pre_fc_norm_embedding.weight": (c.dim,),
        root + "input_layernorm.weight": (c.dim,),
        root + "post_attention_layernorm.weight": (c.dim,),
        root + "self_attn.q_proj.weight": (2 * c.heads * c.head_dim, c.dim),
        root + "self_attn.k_proj.weight": (c.kv_heads * c.head_dim, c.dim),
        root + "self_attn.v_proj.weight": (c.kv_heads * c.head_dim, c.dim),
        root + "self_attn.o_proj.weight": (c.dim, c.heads * c.head_dim),
        root + "self_attn.q_norm.weight": (c.head_dim,),
        root + "self_attn.k_norm.weight": (c.head_dim,),
        root + "mlp.gate.weight": (c.experts, c.dim),
        root + "mlp.shared_expert_gate.weight": (1, c.dim),
    }
    for name in ("gate", "up", "down"):
        shapes[root + f"mlp.shared_expert.{name}_proj.weight"] = (
            (c.dim, c.intermediate) if name == "down" else (c.intermediate, c.dim)
        )
        for expert in range(c.experts):
            shapes[root + f"mlp.experts.{expert}.{name}_proj.weight"] = (
                (c.dim, c.moe_intermediate) if name == "down" else (c.moe_intermediate, c.dim)
            )
    return shapes


def owned_tensors(config, layout, rank):
    """Native BF16 ownership follows target TP and expert-TP, without requantizing."""
    if config.target != layout.config:
        raise ValueError("MTP identity and target execution layout differ")
    layout.validate_rank(rank)
    c = config.target
    if c.dim % layout.tp:
        raise ValueError("MTP fusion outputs must divide tensor parallelism")
    shapes = native_tensor_shapes(config)

    def item(source, destination, starts=None, sizes=None, *, transpose=False, expert=None):
        shape = shapes[source]
        return OwnedTensor(
            source,
            destination,
            shape,
            starts or (0,) * len(shape),
            sizes or shape,
            transpose,
            expert,
        )

    yield item(
        "mtp.fc.weight",
        "fc",
        (rank * (c.dim // layout.tp), 0),
        (c.dim // layout.tp, 2 * c.dim),
        transpose=True,
    )
    for source, dest in (
        ("norm", "norm"),
        ("pre_fc_norm_hidden", "hidden_norm"),
        ("pre_fc_norm_embedding", "embedding_norm"),
        ("layers.0.input_layernorm", "norm_in"),
        ("layers.0.post_attention_layernorm", "norm_post"),
        ("layers.0.self_attn.q_norm", "qn"),
        ("layers.0.self_attn.k_norm", "kn"),
    ):
        yield item(f"mtp.{source}.weight", dest)
    root = "mtp.layers.0."
    qwidth = layout.local_q_heads * c.head_dim
    qstart = rank * qwidth
    kvwidth = layout.local_kv_heads * c.head_dim
    kvstart = layout.key_head_start(rank, full=True) * c.head_dim
    yield item(
        root + "self_attn.q_proj.weight", "qw", (2 * qstart, 0), (2 * qwidth, c.dim), transpose=True
    )
    for name in ("k", "v"):
        yield item(
            root + f"self_attn.{name}_proj.weight",
            name + "w",
            (kvstart, 0),
            (kvwidth, c.dim),
            transpose=True,
        )
    yield item(root + "self_attn.o_proj.weight", "ow", (0, qstart), (c.dim, qwidth), transpose=True)
    router_width = c.experts // layout.router_tp
    yield item(
        root + "mlp.gate.weight",
        "router",
        ((rank % layout.router_tp) * router_width, 0),
        (router_width, c.dim),
        transpose=True,
    )
    yield item(root + "mlp.shared_expert_gate.weight", "shared_gate")
    shared_width = layout.shared_width
    for name, dest in (("gate", "sg"), ("up", "su"), ("down", "sd")):
        down = name == "down"
        yield item(
            root + f"mlp.shared_expert.{name}_proj.weight",
            dest,
            (0, rank * shared_width) if down else (rank * shared_width, 0),
            (c.dim, shared_width) if down else (shared_width, c.dim),
            transpose=True,
        )
    first, last = layout.expert_range(rank)
    begin, end = layout.expert_column_range(rank)
    for expert in range(first, last):
        for name, dest in (("gate", "eg"), ("up", "eu"), ("down", "ed")):
            down = name == "down"
            yield item(
                root + f"mlp.experts.{expert}.{name}_proj.weight",
                dest,
                (0, begin) if down else (begin, 0),
                (c.dim, end - begin) if down else (end - begin, c.dim),
                transpose=True,
                expert=expert - first,
            )


def weight_shapes(config, layout):
    shapes = {}
    for item in owned_tensors(config, layout, 0):
        shape = item.sizes[::-1] if item.transpose else item.sizes
        if item.expert_slot is not None:
            shape = (layout.local_experts, *shape)
        if item.destination in shapes and shapes[item.destination] != shape:
            raise ValueError("Inconsistent MTP destination shape")
        shapes[item.destination] = shape
    return shapes


def memory_plan(config, layout, context, batch_size):
    """Incremental logical/padded estimates; compiler and allocator admission remain required."""
    from .layout import _physical_nbytes_estimate

    if context < 1 or batch_size not in (1, 2, 4, 8, 16):
        raise ValueError("Positive context and a supported independent-request batch required")
    shapes = weight_shapes(config, layout)
    weights = {name: 2 * math.prod(shape) for name, shape in shapes.items()}
    physical_weights = {
        name: _physical_nbytes_estimate(SimpleNamespace(shape=shape, dtype="bfloat16"))
        for name, shape in shapes.items()
    }
    pages = layout.local_kv_pages(context)
    kv = 2 * 2 * batch_size * layout.local_kv_heads * pages * 128 * config.target.head_dim
    physical_kv = 2 * _physical_nbytes_estimate(
        SimpleNamespace(
            shape=(batch_size, layout.local_kv_heads, pages, 128, config.target.head_dim),
            dtype="bfloat16",
        )
    )
    recurrent = 0
    physical_recurrent = 0
    for name, spec in layout.state_specs(context, batch_size).items():
        if name in ("conv", "rec"):
            recurrent += math.prod(spec.shape) * (2 if name == "conv" or layout.fast else 4)
            physical_recurrent += _physical_nbytes_estimate(spec)
    hidden = 2 * batch_size * config.target.dim * 2
    physical_hidden = 2 * _physical_nbytes_estimate(
        SimpleNamespace(
            shape=(batch_size, config.target.dim),
            dtype="bfloat16",
        )
    )
    return {
        "weight_bytes_per_device": sum(weights.values()),
        "weight_families": weights,
        "draft_kv_bytes_per_device": kv,
        "one_target_boundary_bytes_per_device": recurrent,
        "two_hidden_carries_bytes_per_device": hidden,
        "incremental_logical_bytes_per_device": sum(weights.values()) + kv + recurrent + hidden,
        "shared_embedding_head_additional_bytes": 0,
        "physical_weight_bytes_per_device_estimate": sum(physical_weights.values()),
        "physical_weight_families_estimate": physical_weights,
        "physical_draft_kv_bytes_per_device_estimate": physical_kv,
        "physical_one_target_boundary_bytes_per_device_estimate": physical_recurrent,
        "physical_two_hidden_carries_bytes_per_device_estimate": physical_hidden,
        "incremental_physical_bytes_per_device_estimate": (
            sum(physical_weights.values()) + physical_kv + physical_recurrent + physical_hidden
        ),
        "physical_padding_assumption": (
            "Conservative compact 8x128 TPU tiles; excludes allocator/runtime overhead; "
            "compiler layouts may differ."
        ),
        "excludes": ["compiler workspace", "staging", "allocator overhead"],
        "hardware_admitted": False,
    }
