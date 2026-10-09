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

"""Per-chip attention prologue presets for the models we serve on TPU."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import quant, rope

Norm = configs.Norm
Rope = configs.Rope
AttentionConfig = configs.AttentionConfig

# (x, residual, weights, cos, sin).
Inputs = tuple[
    jax.Array,
    jax.Array | None,
    configs.QkvWeights,
    jax.Array | None,
    jax.Array | None,
]


@dataclasses.dataclass(frozen=True)
class Model:
    name: str
    config: AttentionConfig
    hidden_size: int  # K.
    # positions (T,), or (3, T) with mrope -> (cos, sin), each (T, n_pairs).
    cos_sin: Callable[[jax.Array], tuple[jax.Array, jax.Array]] | None
    mrope: bool = False


# Qwen3-Coder-480B-A35B @TP8.
# Attention: third_party/py/vllm/model_executor/models/qwen3_moe.py Qwen3MoeAttention
# Config:
# third_party/py/torch_tpu/examples/benchmarks/huggingface_transformers/
# model_configs/BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic
QWEN3_CODER_TP8 = Model(
    "qwen3_coder_tp8",
    AttentionConfig(
        num_q_heads=12,
        num_kv_heads=1,
        head_dim=128,
        residual=True,
        q_norm=Norm.RMS,
        k_norm=Norm.RMS,
        rope=Rope.neox(128),
    ),
    hidden_size=6144,
    cos_sin=functools.partial(rope.cos_sin_default, rotary_dim=128, theta=1e7),
)


def _qwen3_5(tp: int) -> Model:
    return Model(
        f"qwen3_5_tp{tp}",
        AttentionConfig(
            num_q_heads=32 // tp,
            num_kv_heads=max(1, 2 // tp),
            head_dim=256,
            gate=True,
            input_norm=Norm.RMS_ONE_PLUS_W,
            residual=True,
            q_norm=Norm.RMS_ONE_PLUS_W,
            k_norm=Norm.RMS_ONE_PLUS_W,
            rope=Rope.neox(64),
        ),
        hidden_size=4096,
        cos_sin=functools.partial(
            rope.cos_sin_mrope_interleaved,
            rotary_dim=64,
            theta=1e7,
            section=(11, 11, 10),
        ),
        mrope=True,
    )


# Qwen3.5-397B-A17B full-attention layers.
# Attention: third_party/py/vllm/model_executor/models/qwen3_next.py Qwen3NextAttention
# Config:
# third_party/py/torch_tpu/examples/benchmarks/huggingface_transformers/
# model_configs/Qwen/Qwen3.5-397B-A17B
QWEN3_5_TP8 = _qwen3_5(8)
QWEN3_5_TP2 = _qwen3_5(2)

# gemma-4-31B-it @TP4.
# Attention: third_party/py/vllm/model_executor/models/gemma4.py Gemma4Attention
# Config:
# third_party/py/torch_tpu/examples/benchmarks/huggingface_transformers/
# model_configs/google/gemma-4-31b
GEMMA4_SLIDING_TP4 = Model(
    "gemma4_sliding_tp4",
    AttentionConfig(
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=256,
        q_norm=Norm.RMS,
        k_norm=Norm.RMS,
        v_norm=Norm.RMS_NOSCALE,
        act_quant_block=768,  # hidden 5376 = 7 * 768.
        rope=Rope.neox(256),
    ),
    hidden_size=5376,
    cos_sin=functools.partial(rope.cos_sin_default, rotary_dim=256, theta=1e4),
)
GEMMA4_FULL_TP4 = Model(
    "gemma4_full_tp4",
    AttentionConfig(
        num_q_heads=8,
        num_kv_heads=1,
        head_dim=512,
        v_from_k=True,
        q_norm=Norm.RMS,
        k_norm=Norm.RMS,
        v_norm=Norm.RMS_NOSCALE,
        act_quant_block=768,  # hidden 5376 = 7 * 768.
        rope=Rope(64, 256),
    ),
    hidden_size=5376,
    cos_sin=functools.partial(
        rope.cos_sin_proportional,
        head_dim=512,
        partial_rotary_factor=0.25,
        theta=1e6,
    ),
)

# gpt-oss-120b @TP4.
# Attention: third_party/py/vllm/model_executor/models/gpt_oss.py OAIAttention
# Config:
# third_party/py/torch_tpu/examples/benchmarks/huggingface_transformers/
# model_configs/openai/gpt-oss-120b
GPT_OSS_TP4 = Model(
    "gpt_oss_tp4",
    AttentionConfig(
        num_q_heads=16,
        num_kv_heads=2,
        head_dim=64,
        fp8=False,
        bias=True,
        residual=True,
        rope=Rope.neox(64),
        eps=1e-5,
    ),
    hidden_size=2880,
    cos_sin=functools.partial(
        rope.cos_sin_yarn,
        rotary_dim=64,
        theta=150000.0,
        factor=32.0,
        original_max_position=4096,
        truncate=False,
    ),
)

MODELS = (
    QWEN3_CODER_TP8,
    QWEN3_5_TP8,
    QWEN3_5_TP2,
    GEMMA4_SLIDING_TP4,
    GEMMA4_FULL_TP4,
    GPT_OSS_TP4,
)
BY_NAME = {m.name: m for m in MODELS}


def make_inputs(model: Model, t: int) -> Inputs:
    """Returns random (x, residual, weights, cos, sin) for t tokens."""
    return _make_inputs(model, t, exact=False)


def make_exact_inputs(model: Model, t: int) -> Inputs:
    """Like make_inputs, but with inputs on which floating point is exact.

    Small integers times powers of two are exact in bf16 and fp8, and their sums
    stay exact in any order while small. So the kernel must match the reference
    bit for bit, and any difference is a bug. This catches structural bugs
    (indexing, layouts, head order, packing, block scales), not rounding;
    test_accuracy covers rounding on random inputs.

    How the sums stay exact:
      - x + residual is ±1 per element, so the input norm's sum of squares is
        exactly the hidden size.
      - Each fp8 quant block has one marker column whose input-norm scale,
        56 * 2^-(block % 4), is the block's amax; its W row is zero. Every
        block's fp8 scale is then a different power of two (1/8 ... 1/64), and
        all quantized values are exact. The other scales are 1 or 2.
      - W is in {-1, 0, 1} with power-of-two W scales.

    Limits: all rows get the same scales, and the block scales repeat every 4
    blocks, so a bug that applies another row's scale, or the scale of a block
    4 away, can still pass.
    """
    return _make_inputs(model, t, exact=True)


def _make_inputs(model: Model, t: int, exact: bool) -> Inputs:
    s = model.config
    k, n, d = model.hidden_size, s.num_cols, s.head_dim
    ks = jax.random.split(jax.random.key(t), 10)
    x, residual, w, w_scale, input_norm_w = _projection_inputs(s, t, k, n, ks, exact)
    bias = jax.random.normal(ks[3], (1, n), jnp.bfloat16) if s.bias else None
    norm_w = []
    for i, kind in enumerate((s.q_norm, s.k_norm)):
        w_i = 0.1 * jax.random.normal(ks[4 + i], (1, d), jnp.bfloat16)
        norm_w.append(
            None
            if kind in (Norm.NONE, Norm.RMS_NOSCALE)
            else (w_i if kind == Norm.RMS_ONE_PLUS_W else 1 + w_i)
        )
    weights = configs.QkvWeights(w, w_scale, bias, *norm_w, input_norm_w=input_norm_w)
    cos = sin = None
    if s.rope is not None:
        assert model.cos_sin is not None
        pos = jax.random.randint(ks[6], (3, t) if model.mrope else (t,), 0, 32768)
        cos, sin = model.cos_sin(pos)
    return x, residual if s.residual else None, weights, cos, sin


def _projection_inputs(
    s: configs.AttentionConfig | configs.GdnConfig,
    t: int,
    k: int,
    n: int,
    ks: jax.Array,
    exact: bool,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array | None,
    jax.Array,
]:
    """Returns (x, residual, w, w_scale, input_norm_w); ks holds 10 PRNG keys."""
    w_dtype = quant.F8 if s.fp8 else jnp.bfloat16
    w_scale = None

    def pow2(
        key: jax.Array, shape: tuple[int, ...], lo: int, num: int = 2
    ) -> jax.Array:
        """Random powers of two in 2^lo ... 2^(lo + num - 1)."""
        return jnp.exp2(lo + jax.random.randint(key, shape, 0, num)).astype(jnp.float32)

    if exact:
        total = 2 * jax.random.randint(ks[9], (t, k), 0, 2) - 1  # x + residual.
        x = jax.random.randint(ks[0], (t, k), -1, 2) if s.residual else total
        residual = (total - x).astype(jnp.bfloat16)
        x = x.astype(jnp.bfloat16)
        markers = jnp.arange(0, k, s.act_quant_block if s.fp8 else k)
        marker_scale = 56 * jnp.exp2(-(jnp.arange(len(markers)) % 4))
        input_norm_w = pow2(ks[8], (1, k), 0).at[0, markers].set(marker_scale)
        w = jax.random.randint(ks[1], (k, n), -1, 2).at[markers].set(0)
        w = w.astype(w_dtype)
        if s.fp8:
            w_scale = pow2(ks[3], (1, n), -7)
    else:
        x = jax.random.normal(ks[0], (t, k), jnp.bfloat16)
        residual = jax.random.normal(ks[7], (t, k), jnp.bfloat16)
        input_norm_w = 1 + 0.1 * jax.random.normal(ks[8], (1, k), jnp.float32)
        w = jax.random.normal(ks[1], (k, n), jnp.float32)
        if s.fp8:
            w = w.astype(quant.F8)
            w_scale = jax.random.uniform(ks[2], (1, n), jnp.float32, 0.01, 0.02)
        else:
            w = (w * k**-0.5).astype(jnp.bfloat16)
    if s.input_norm == Norm.RMS_ONE_PLUS_W:
        input_norm_w = input_norm_w - 1  # Scales by 1 + w.
    input_norm_w = input_norm_w.astype(jnp.bfloat16)
    return x, residual, w, w_scale, input_norm_w


# GDN prologues.


@dataclasses.dataclass(frozen=True)
class GdnModel:
    name: str
    config: configs.GdnConfig
    hidden_size: int  # K.


# Qwen3.5-397B-A17B linear-attention layers @TP8.
# Layer: vllm-torchtpu Qwen3_5GDNAttention -> VllmGatedDeltaNetAttention.
# Config:
# third_party/py/torch_tpu/examples/benchmarks/huggingface_transformers/
# model_configs/Qwen/Qwen3.5-397B-A17B
QWEN3_5_GDN_TP8 = GdnModel(
    "qwen3_5_gdn_tp8",
    configs.GdnConfig(num_k_heads=16 // 8, num_v_heads=64 // 8),
    hidden_size=4096,
)

GDN_MODELS = (QWEN3_5_GDN_TP8,)
GDN_BY_NAME = {m.name: m for m in GDN_MODELS}


def make_gdn_inputs(
    model: GdnModel, t: int, exact: bool = False
) -> tuple[jax.Array, jax.Array | None, configs.QkvWeights]:
    """Returns random (x, residual, weights) for t tokens.

    With exact, inputs are as in make_exact_inputs, and side_w is in {-1, 0, 1}
    so the side projection is exact too.
    """
    s = model.config
    k, n = model.hidden_size, s.num_cols
    ks = jax.random.split(jax.random.key(t), 10)
    x, residual, w, w_scale, input_norm_w = _projection_inputs(s, t, k, n, ks, exact)
    side_key = jax.random.fold_in(ks[4], 1)
    if exact:
        markers = jnp.arange(0, k, s.act_quant_block)
        side_w = jax.random.randint(side_key, (k, s.side_cols), -1, 2)
        side_w = side_w.at[markers].set(0).astype(jnp.bfloat16)
    else:
        side_w = jax.random.normal(side_key, (k, s.side_cols), jnp.float32)
        side_w = (side_w * k**-0.5).astype(jnp.bfloat16)
    weights = configs.QkvWeights(w, w_scale, input_norm_w=input_norm_w, side_w=side_w)
    return x, residual if s.residual else None, weights
