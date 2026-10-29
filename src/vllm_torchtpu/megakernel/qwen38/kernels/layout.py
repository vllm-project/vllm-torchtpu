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
"""Validated TP/EP ownership and byte-exact layouts for native FP8 and NVFP4 weights."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import Config

WEIGHT_NAMES = (
    "emb",
    "lmw",
    "norm_in",
    "norm_post",
    "fnorm",
    "qkvz",
    "outw",
    "convw",
    "alog",
    "dtb",
    "onorm",
    "qwkv",
    "ow",
    "qn",
    "kn",
    "gu",
    "dw",
    "shared_gate",
    "router",
    "egu",
    "egu_scale",
    "edw",
    "edw_scale",
)


NVFP4_WEIGHT_NAMES = WEIGHT_NAMES + ("eparams", "lparams", "fparams")


def align128(value):
    return (value + 127) // 128 * 128


def align8(value):
    return (value + 7) // 8 * 8


@dataclass(frozen=True)
class Layout:
    config: Config
    tp: int = 32
    expert_tp: int = 4
    router_tp: int = 4
    context_parallel: int = 1
    precision: str = "native"
    kv_page_size: int = 128
    expert_output_tile: int = 0
    dense_output_tile: int = 128
    # "exact" combines expert-TP partials before the W4A8 output quantization;
    # "residual" leaves each rank's FP32 partial to the layer's residual sum.
    expert_reduction: str = "exact"
    # "bfloat16" sends each rank's partial and each host sum of the C1/C2/C4
    # and verification residual reductions rounded to BF16; sums stay FP32.
    # None selects bfloat16 for the W4A8 profile and float32 for native.
    reduction_wire: str | None = None
    # Wide BF16 residual reductions (64 or more live rows) add each chip's two cores
    # first, which changes their FP32 association; narrower or FP32 packets never pair.
    reduction_pairs: bool = True
    # Each EP group also holds the first ``expert_overlap`` experts of its predecessor
    # group, so a busy group's routed experts can move to its successor.
    expert_overlap: int = 0

    def __post_init__(self):
        c = self.config
        if self.reduction_wire is None:
            object.__setattr__(self, "reduction_wire", "bfloat16" if self.fast else "float32")
        if self.dense_output_tile not in (128, 256, 512, 1024, 2048, 4096, 8192):
            raise ValueError("dense_output_tile must be 128, 256, 512, 1024, 2048, 4096 or 8192")
        if self.expert_output_tile not in (0, 256, 512):
            raise ValueError("expert_output_tile must be 0, 256 or 512")
        if self.expert_output_tile and self.precision == "native":
            raise ValueError("Expert output tiling currently requires the W4A8 profile")
        if self.precision not in ("native", "torchtpu_w4a8_b64"):
            raise ValueError(f"Unknown runtime precision {self.precision!r}")
        if self.precision != "native" and c.quantization != "nvfp4":
            raise ValueError("The W4A8 runtime profile requires a native NVFP4 checkpoint")
        if self.precision != "native" and self.expert_tp not in (1, 4, 8):
            raise ValueError("The W4A8 runtime profile supports expert_tp=1, 4 or 8")
        if self.expert_reduction not in ("exact", "residual"):
            raise ValueError("expert_reduction must be exact or residual")
        if self.expert_reduction == "residual" and not self.fast:
            raise ValueError("Residual expert reduction requires the W4A8 profile")
        if self.reduction_wire not in ("float32", "bfloat16"):
            raise ValueError("reduction_wire must be float32 or bfloat16")
        if self.expert_overlap and (
            self.expert_reduction != "residual" or not 0 < self.expert_overlap <= self.local_experts
        ):
            raise ValueError("Expert overlap needs residual reduction and at most one group")
        if self.context_parallel not in (1, 8) or self.tp % self.context_parallel:
            raise ValueError("context_parallel must be 1 or 8 and divide tp")
        if self.kv_page_size not in (128, 256):
            raise ValueError("KV pages must contain 128 or 256 tokens")
        if self.context_parallel > 1 and (
            c.kv_heads > self.tp // self.context_parallel
            or self.tp % (c.kv_heads * self.context_parallel)
        ):
            raise ValueError("Each CP group must share one replicated KV head")
        if self.tp < 1 or self.tp & (self.tp - 1):
            raise ValueError("tp must be a positive power of two")
        for name, width in (("expert_tp", self.expert_tp), ("router_tp", self.router_tp)):
            if width < 1 or self.tp % width:
                raise ValueError(f"{name} must divide tp")
        if c.experts % self.ep or c.experts % self.router_tp:
            raise ValueError("Expert count must divide EP and router groups")
        if c.dim % 128 or c.moe_intermediate % (128 * self.expert_tp):
            detail = (
                "128x128 FP8 blocks"
                if c.quantization == "fp8"
                else "128-wide MXU tiles and NVFP4 groups"
            )
            raise ValueError(f"Expert shards must preserve complete {detail}")
        if c.intermediate % self.tp or c.heads % self.tp or c.v_heads % self.tp:
            raise ValueError("Shared intermediate/query/value heads must divide tp")
        if self.shared_width % 128 and 2 * self.shared_width > 128:
            raise ValueError("Shared expert shards must fit gate and up in one tile or whole tiles")
        for heads in (c.k_heads, c.kv_heads):
            if heads % self.tp and self.tp % heads:
                raise ValueError("Key heads must split evenly or replicate evenly")
        if self.local_v_heads % self.local_k_heads:
            raise ValueError("Each value shard must contain complete key-head groups")
        if self.local_q_heads % self.local_kv_heads:
            raise ValueError("Each query shard must contain complete KV-head groups")

    @classmethod
    def for_config(cls, config, tp=32, expert_tp=4, router_tp=4, **kwargs):
        return cls(config, tp, expert_tp, router_tp, **kwargs)

    @property
    def fast(self):
        return self.precision == "torchtpu_w4a8_b64"

    @property
    def kv_dtype(self):
        return jnp.float8_e4m3fn if self.fast else jnp.bfloat16

    @property
    def recurrent_dtype(self):
        return jnp.bfloat16 if self.fast else jnp.float32

    def local_kv_pages(self, context):
        """Capacity per rank, including the final incomplete CP stripe."""
        stripe = self.context_parallel * self.kv_page_size
        return (context + stripe - 1) // stripe

    def kv_owner(self, position):
        return position // self.kv_page_size % self.context_parallel

    def kv_local_position(self, position):
        return (
            position // (self.context_parallel * self.kv_page_size),
            position % self.kv_page_size,
        )

    @property
    def ep(self):
        return self.tp // self.expert_tp

    @property
    def local_experts(self):
        return self.config.experts // self.ep

    @property
    def held_experts(self):
        return self.local_experts + self.expert_overlap

    @property
    def expert_width(self):
        return self.config.moe_intermediate // self.expert_tp

    @property
    def shared_width(self):
        return self.config.intermediate // self.tp

    @property
    def shared_width_padded(self):
        return align128(self.shared_width)

    @property
    def shared_gate_up_width(self):
        """Device gate and up shards share tiles: [gate | up | zero tail]."""
        return align128(2 * self.shared_width)

    @property
    def local_q_heads(self):
        return self.config.heads // self.tp

    @property
    def local_k_heads(self):
        return max(1, self.config.k_heads // self.tp)

    @property
    def local_v_heads(self):
        return self.config.v_heads // self.tp

    @property
    def local_kv_heads(self):
        return max(1, self.config.kv_heads // self.tp)

    @property
    def local_conv(self):
        c = self.config
        return 2 * self.local_k_heads * c.k_dim + self.local_v_heads * c.v_dim

    @property
    def vocab_shard(self):
        return (self.config.vocab + self.tp - 1) // self.tp

    @property
    def vocab_shard_padded(self):
        return align128(self.vocab_shard)

    def key_head_start(self, rank, *, full=False):
        self.validate_rank(rank)
        heads = self.config.kv_heads if full else self.config.k_heads
        return rank * (heads // self.tp) if heads >= self.tp else rank // (self.tp // heads)

    def expert_range(self, rank):
        self.validate_rank(rank)
        start = rank // self.expert_tp * self.local_experts
        return start, start + self.local_experts

    def expert_column_range(self, rank):
        self.validate_rank(rank)
        start = rank % self.expert_tp * self.expert_width
        return start, start + self.expert_width

    def validate_rank(self, rank):
        if not 0 <= rank < self.tp:
            raise ValueError(f"rank {rank} outside tp={self.tp}")

    @property
    def weight_names(self):
        return NVFP4_WEIGHT_NAMES if self.config.quantization == "nvfp4" else WEIGHT_NAMES

    @property
    def weight_specs(self):
        c = self.config
        nl, nf, h = len(c.linear_layers), len(c.full_layers), c.dim
        kh, vh, qh, kvh = (
            self.local_k_heads,
            self.local_v_heads,
            self.local_q_heads,
            self.local_kv_heads,
        )
        sw, ew, e = self.shared_width, self.expert_width, self.local_experts
        shapes = {
            "emb": (self.vocab_shard_padded, h),
            "lmw": (h, self.vocab_shard_padded),
            # Rank-two vector tables use aligned eight-row DMA gathers. Layer
            # tails and narrow head tables must contain the complete last tile.
            "norm_in": (align8(c.layers), h),
            "norm_post": (align8(c.layers), h),
            "fnorm": (h,),
            "qkvz": (nl, h, align128(2 * kh * c.k_dim + 2 * vh * c.v_dim + 2 * vh)),
            "outw": (nl, vh * c.v_dim, h),
            "convw": (nl, c.conv_size, self.local_conv),
            "alog": (align8(nl), align128(vh)),
            "dtb": (align8(nl), align128(vh)),
            "onorm": (align8(nl), c.v_dim),
            "qwkv": (nf, h, 2 * qh * c.head_dim + 2 * kvh * c.head_dim),
            "ow": (nf, qh * c.head_dim, h),
            "qn": (align8(nf), c.head_dim),
            "kn": (align8(nf), c.head_dim),
            "gu": (c.layers, h, self.shared_gate_up_width),
            "dw": (c.layers, sw, h),
            "shared_gate": (align8(c.layers), h),
            "router": (c.layers, h, c.experts // self.router_tp),
            "egu": (c.layers, e, h, 2 * ew),
            "egu_scale": (c.layers, e, h // 128, 2 * ew // 128),
            "edw": (c.layers, e, ew, h),
            "edw_scale": (c.layers, e, ew // 128, h // 128),
        }
        dtypes = {"egu": jnp.float8_e4m3fn, "edw": jnp.float8_e4m3fn, "alog": jnp.float32}
        if c.quantization == "nvfp4":
            # Keep the source K packing intact: transpose bytes, never unpack
            # the E2M1 nibbles or fold projection-specific global scales.
            shapes.update(
                egu=(c.layers, e, h // 2, 2 * ew),
                egu_scale=(c.layers, e, h // 16, 2 * ew),
                edw=(c.layers, e, ew // 2, h),
                edw_scale=(c.layers, e, ew // 16, h),
                # DMA-friendly scalar tiles: only row zero is populated.
                # Expert pairs: gate, up, down (input, weight_scale_2).
                # Linear pairs: qkv, z, b, a, out (input, weight_scale).
                # Full-attention pairs: q, k, v, o (input, weight_scale).
                eparams=(c.layers, e, 8, 128),
                lparams=(nl, 8, 128),
                fparams=(nf, 8, 128),
            )
            dtypes.update(egu=jnp.uint8, edw=jnp.uint8)
            dtypes.update(
                {
                    name: jnp.float8_e4m3fn
                    for name in ("egu_scale", "edw_scale", "qkvz", "outw", "qwkv", "ow")
                }
            )
            dtypes.update({name: jnp.float32 for name in ("eparams", "lparams", "fparams")})
        if self.fast:
            # Runtime conversion uses block-64 FP32 scales and packs eight
            # E2M1 values per U32 K word. The published checkpoint stays native.
            held = self.held_experts
            shapes.update(
                egu=(c.layers, held, h // 8, 2 * ew),
                edw=(c.layers, held, ew // 8, h),
                egu_scale=(c.layers, held, h // 64, 2 * ew),
                edw_scale=(c.layers, held, ew // 64, h),
            )
            dtypes.update(
                egu=jnp.uint32, edw=jnp.uint32, egu_scale=jnp.float32, edw_scale=jnp.float32
            )
        return {
            name: jax.ShapeDtypeStruct(shapes[name], dtypes.get(name, jnp.bfloat16))
            for name in self.weight_names
        }

    @property
    def record_specs(self):
        """Weights as ``load._rank_chunks`` yields them and pre-shard v1 stores them.

        Only the shared expert differs: its gate, up and down shards each keep
        whole 128-column tiles there, packed on load (``load.device_record``).
        """
        c, pad, specs = self.config, self.shared_width_padded, self.weight_specs
        return {
            **specs,
            "gu": jax.ShapeDtypeStruct((c.layers, c.dim, 2 * pad), specs["gu"].dtype),
            "dw": jax.ShapeDtypeStruct((c.layers, pad, c.dim), specs["dw"].dtype),
        }

    def state_specs(self, context, batch_size):
        if context < 1 or batch_size not in (1, 2, 4, 8, 16):
            raise ValueError("context must be positive and batch_size one of 1,2,4,8,16")
        c = self.config
        # CP1 preserves the public legacy shape. CP8 owns global pages
        # rank%CP, rank%CP+CP, ...; the final stripe is zero padded.
        cache_tail = (
            (context, c.head_dim)
            if self.context_parallel == 1
            else (self.local_kv_pages(context), self.kv_page_size, c.head_dim)
        )
        cache_shape = (len(c.full_layers), batch_size, self.local_kv_heads, *cache_tail)
        return {
            "conv": jax.ShapeDtypeStruct(
                (len(c.linear_layers), batch_size, c.conv_size - 1, self.local_conv), jnp.bfloat16
            ),
            "rec": jax.ShapeDtypeStruct(
                (len(c.linear_layers), batch_size, self.local_v_heads, c.v_dim, c.k_dim),
                self.recurrent_dtype,
            ),
            "kcache": jax.ShapeDtypeStruct(cache_shape, self.kv_dtype),
            "vcache": jax.ShapeDtypeStruct(cache_shape, self.kv_dtype),
        }


def _nbytes(spec):
    return math.prod(spec.shape) * np.dtype(spec.dtype).itemsize


def _physical_nbytes_estimate(spec):
    """Conservative compact-tile storage, not an observed compiler allocation.

    JAX 0.11.1 tpu_info.infer_tiling uses at most eight rows and 128 columns
    for its compact TPU layouts, with smaller dtype-dependent row factors.
    Include the leading singleton rank shard, and conservatively use the full
    eight-row tile. Compiler-assigned layouts and allocator overhead can differ.
    """
    shape = [1, *spec.shape]
    if not math.prod(shape):
        return 0
    row_tile = max(8, 4 // np.dtype(spec.dtype).itemsize)
    shape[-2] = (shape[-2] + row_tile - 1) // row_tile * row_tile
    shape[-1] = align128(shape[-1])
    return math.prod(shape) * np.dtype(spec.dtype).itemsize


def memory_plan(config, layout, context, batch_size):
    if config != layout.config:
        raise ValueError("Layout and model config differ")
    weights = {name: _nbytes(spec) for name, spec in layout.weight_specs.items()}
    states = {name: _nbytes(spec) for name, spec in layout.state_specs(context, batch_size).items()}
    weight_bytes, state_bytes = sum(weights.values()), sum(states.values())
    physical_weights = {
        name: _physical_nbytes_estimate(spec) for name, spec in layout.weight_specs.items()
    }
    physical_states = {
        name: _physical_nbytes_estimate(spec)
        for name, spec in layout.state_specs(context, batch_size).items()
    }
    physical_weight_bytes, physical_state_bytes = (
        sum(physical_weights.values()),
        sum(physical_states.values()),
    )
    return {
        "tp": layout.tp,
        "expert_tp": layout.expert_tp,
        "context_parallel": layout.context_parallel,
        "precision": layout.precision,
        "kv_page_size": layout.kv_page_size,
        "dense_output_tile": layout.dense_output_tile,
        "expert_output_tile": layout.expert_output_tile,
        "batch_size": batch_size,
        "context": context,
        "weight_bytes_per_device": weight_bytes,
        "state_bytes_per_device": state_bytes,
        "resident_bytes_per_device": weight_bytes + state_bytes,
        "weight_bytes_total": weight_bytes * layout.tp,
        "state_bytes_total": state_bytes * layout.tp,
        "resident_bytes_total": (weight_bytes + state_bytes) * layout.tp,
        "weight_families": weights,
        "state_families": states,
        "physical_weight_bytes_per_device_estimate": physical_weight_bytes,
        "physical_state_bytes_per_device_estimate": physical_state_bytes,
        "physical_resident_bytes_per_device_estimate": physical_weight_bytes + physical_state_bytes,
        "physical_resident_bytes_total_estimate": (physical_weight_bytes + physical_state_bytes)
        * layout.tp,
        "physical_weight_families_estimate": physical_weights,
        "physical_state_families_estimate": physical_states,
        "physical_padding_assumption": "Conservative compact 8x128 TPU tiles; excludes allocator/runtime overhead; compiler layouts may differ.",
    }


def make_mesh(logical_devices=None):
    """Order host-local TPU cores, validating the required four-host slice.

    Explicit devices permit small CPU fixtures without asserting TPU topology.
    """
    explicit = logical_devices is not None
    devices = list(logical_devices) if explicit else list(jax.devices())
    if not devices or len({d.id for d in devices}) != len(devices):
        raise ValueError("Mesh requires unique devices")
    if not explicit:
        hosts = {d.process_index for d in devices}
        if (
            len(devices) != 32
            or len(hosts) != 4
            or any(sum(d.process_index == h for d in devices) != 8 for h in hosts)
        ):
            raise ValueError("Expected 32 TPU cores across four hosts with eight cores each")
        if any(d.platform != "tpu" for d in devices):
            raise ValueError("The production mesh requires TPU devices")
        if not all(hasattr(d, "coords") and hasattr(d, "core_on_chip") for d in devices):
            raise ValueError("TPU topology requires coords and core_on_chip metadata")
        # Device IDs can be core-major. Keep hosts contiguous, then explicitly
        # pair the two cores of each physical chip for the decoder collectives.
        devices.sort(key=lambda d: (d.process_index, tuple(d.coords), d.core_on_chip))
        coords = {tuple(d.coords) for d in devices}
        extents = tuple(len({p[i] for p in coords}) for i in range(3))
        if len(coords) != 16 or sorted(extents) != [2, 2, 4]:
            raise ValueError(f"Expected a 2x2x4 chip slice, found {extents}")
        for rank in range(0, 32, 2):
            pair = devices[rank : rank + 2]
            if tuple(pair[0].coords) != tuple(pair[1].coords):
                raise ValueError("Adjacent logical ranks must be the two cores of one chip")
            if tuple(d.core_on_chip for d in pair) != (0, 1):
                raise ValueError("Each adjacent rank pair requires core_on_chip order (0, 1)")
    return jax.sharding.Mesh(np.asarray(devices), ("tp",))
