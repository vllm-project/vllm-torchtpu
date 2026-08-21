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
"""Statically tuned block sizes for flash attention."""

from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class TuningKey:
    """Static inputs which can change the best flash-attention tiling."""

    device_kind: str
    batch_size: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    head_dim: int
    q_dtype: str
    kv_dtype: str
    v_dtype: str
    causal: bool
    has_segment_ids: bool
    has_attention_bias: bool
    vmem_limit_bytes: int | None


@dataclass(frozen=True)
class TunableParams:
    block_q: int
    block_k_major: int
    block_k: int
    block_b: int


def _vision_tuning_key(*, num_heads: int, seq_len: int,
                       head_dim: int) -> TuningKey:
    return TuningKey(
        device_kind="TPU7x",
        batch_size=1,
        num_heads=num_heads,
        q_seq_len=seq_len,
        kv_seq_len=seq_len,
        head_dim=head_dim,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        v_dtype="bfloat16",
        causal=False,
        has_segment_ids=True,
        has_attention_bias=False,
        vmem_limit_bytes=64 * 1024 * 1024,
    )


# Exact TPU7x measurements for BF16 vision self-attention with segment IDs and
# a 64 MiB VMEM limit. Missing shapes use the generic kernel fallback.
tuned_params_mapping: dict[TuningKey, TunableParams] = {
    # Kimi-K3.
    _vision_tuning_key(num_heads=12, seq_len=256, head_dim=128):
    TunableParams(256, 256, 256, 1),
    _vision_tuning_key(num_heads=12, seq_len=512, head_dim=128):
    TunableParams(512, 512, 256, 1),
    _vision_tuning_key(num_heads=12, seq_len=1_024, head_dim=128):
    TunableParams(1_024, 1_024, 512, 1),
    _vision_tuning_key(num_heads=12, seq_len=2_048, head_dim=128):
    TunableParams(1_024, 2_048, 512, 1),
    _vision_tuning_key(num_heads=12, seq_len=4_096, head_dim=128):
    TunableParams(1_024, 4_096, 512, 1),
    _vision_tuning_key(num_heads=12, seq_len=8_192, head_dim=128):
    TunableParams(1_024, 8_192, 512, 1),
    _vision_tuning_key(num_heads=12, seq_len=16_384, head_dim=128):
    TunableParams(1_024, 8_192, 512, 1),
    _vision_tuning_key(num_heads=12, seq_len=32_768, head_dim=128):
    TunableParams(512, 16_384, 512, 1),
    _vision_tuning_key(num_heads=12, seq_len=65_536, head_dim=128):
    TunableParams(512, 16_384, 512, 1),
    _vision_tuning_key(num_heads=12, seq_len=67_328, head_dim=128):
    TunableParams(8_192, 256, 256, 1),
    # Qwen2.5-VL.
    _vision_tuning_key(num_heads=16, seq_len=256, head_dim=80):
    TunableParams(256, 256, 256, 1),
    _vision_tuning_key(num_heads=16, seq_len=512, head_dim=80):
    TunableParams(512, 512, 512, 1),
    _vision_tuning_key(num_heads=16, seq_len=1_024, head_dim=80):
    TunableParams(1_024, 1_024, 512, 1),
    _vision_tuning_key(num_heads=16, seq_len=2_048, head_dim=80):
    TunableParams(2_048, 2_048, 512, 1),
    _vision_tuning_key(num_heads=16, seq_len=4_096, head_dim=80):
    TunableParams(1_024, 4_096, 512, 1),
    _vision_tuning_key(num_heads=16, seq_len=8_192, head_dim=80):
    TunableParams(1_024, 8_192, 512, 1),
    _vision_tuning_key(num_heads=16, seq_len=16_384, head_dim=80):
    TunableParams(512, 16_384, 512, 1),
    _vision_tuning_key(num_heads=16, seq_len=32_768, head_dim=80):
    TunableParams(512, 16_384, 512, 1),
    _vision_tuning_key(num_heads=16, seq_len=65_536, head_dim=80):
    TunableParams(512, 16_384, 512, 1),
}


def current_device_kind() -> str:
    """Return the JAX device kind used to compile the kernel."""
    devices = jax.local_devices()
    return devices[0].device_kind if devices else "unknown"


def make_tuning_key(
    q,
    k,
    v,
    *,
    causal: bool,
    has_segment_ids: bool,
    has_attention_bias: bool,
    vmem_limit_bytes: int | None,
    device_kind: str | None = None,
) -> TuningKey:
    """Build a lookup key from static kernel inputs."""
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    kv_seq_len = k.shape[2]
    return TuningKey(
        device_kind=device_kind or current_device_kind(),
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        q_seq_len=int(q_seq_len),
        kv_seq_len=int(kv_seq_len),
        head_dim=int(head_dim),
        q_dtype=q.dtype.name,
        kv_dtype=k.dtype.name,
        v_dtype=v.dtype.name,
        causal=causal,
        has_segment_ids=has_segment_ids,
        has_attention_bias=has_attention_bias,
        vmem_limit_bytes=vmem_limit_bytes,
    )


def get_tuned_params(tuning_key: TuningKey) -> TunableParams | None:
    return tuned_params_mapping.get(tuning_key)
