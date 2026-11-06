# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/dspark_mla.py
"""MLA DSpark draft model for Kimi K3 (``Inferact/Kimi-K3-DSpark``).

A DSpark block drafter whose attention is DeepSeek-style MLA, like the
target's but with RoPE: 5 dense layers, 64 heads (q_lora 1536, kv_lora 512,
nope 128, rope 64, v 128, no output gate), YaRN RoPE (factor 32), a low-rank
Markov head, the target's embedding and LM head (the checkpoint has none of
its own LM head; vLLM ignores its embedding too).

Context: every token the target ran has, in each draft layer, the latent
``[kv_a_layernorm(kv[:512]) | rope(kv[512:576])]`` of ``kv =
kv_a_proj_with_mqa(context_norm(context_proj(aux)))``, where ``aux`` is the
concatenated residual streams entering target layers ``aux_layers``. The
latents live in five pools shaped like the target's MLA pools, read and
written through the target's block tables (decode context parallel: global
page ``g`` on rank ``g % 32``; under attention DP2 on member ``g % 16`` of the
request's group, whose ranks attend it with their two held shards' heads).

Block: the anchor (the last verified token) and ``block - 1`` mask tokens at
the positions after it; every block row attends every context token before
the anchor and every block row (non-causal), and predicts the token one
position after itself. Drafts are sampled left to right with the Markov bias
``markov_w2 @ markov_w1[previous token]``.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from safetensors import safe_open

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _value_error(message: str) -> ValueError:
    """Create a ValueError with the provided message."""
    return ValueError(message)


@dataclasses.dataclass(frozen=True)
class MlaDraftConfig:
    """Configuration for the 5-layer MLA DSpark speculative draft model."""

    hidden: int = 7168
    layers: int = 5
    heads: int = 64
    q_lora: int = 1536
    kv_lora: int = 512
    nope: int = 128
    rope: int = 64
    v_head: int = 128
    intermediate: int = 14336
    vocab: int = 163840
    # The target layers whose entering residual streams feed the context
    # (vLLM's aux ids: the checkpoint's ``target_layer_ids`` + 1).
    aux_layers: tuple[int, ...] = (3, 24, 48, 72, 90)
    mask_token: int = 163837
    eps: float = 1e-5
    markov_rank: int = 256
    block: int = 7
    rope_theta: float = 50000.0
    rope_factor: float = 32.0
    rope_original_max: int = 32768
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0

    @property
    def qk_head(self) -> int:
        """Return total query/key head dimension (nope + rope)."""
        return self.nope + self.rope

    @property
    def latent(self) -> int:
        """Return unpadded MLA latent dimension (kv_lora + rope)."""
        return self.kv_lora + self.rope

    @property
    def context_width(self) -> int:
        """Return concatenated target auxiliary hidden width."""
        return len(self.aux_layers) * self.hidden

    @property
    def softmax_scale(self) -> float:
        """Return ``qk_head ** -0.5`` times the YaRN mscale squared."""
        scale = self.qk_head**-0.5
        if self.rope_factor > 1:
            mscale = 0.1 * self.rope_mscale_all_dim * math.log(self.rope_factor) + 1.0
            scale *= mscale * mscale
        return scale

    @classmethod
    def from_checkpoint(cls, directory: Path | str) -> MlaDraftConfig:
        """Load and validate an MlaDraftConfig from checkpoint config.json."""
        raw = json.loads((Path(directory) / "config.json").read_text())
        if raw.get("model_type") != "k3_dspark":
            err = _value_error(
                f"{directory}: not a k3_dspark draft ({raw.get('model_type')})"
            )
            raise err
        if raw.get("mla_use_nope") or raw.get("mla_use_output_gate"):
            err = _value_error("the MLA DSpark draft has RoPE and no output gate")
            raise err
        rope = raw.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "yarn":
            err = _value_error(f"unsupported draft RoPE {rope}")
            raise err
        if raw.get("draft_vocab_size", raw["vocab_size"]) != raw["vocab_size"]:
            err = _value_error("the draft vocabulary must be the target's")
            raise err
        return cls(
            hidden=raw["hidden_size"],
            layers=raw["num_hidden_layers"],
            heads=raw["num_attention_heads"],
            q_lora=raw["q_lora_rank"],
            kv_lora=raw["kv_lora_rank"],
            nope=raw["qk_nope_head_dim"],
            rope=raw["qk_rope_head_dim"],
            v_head=raw["v_head_dim"],
            intermediate=raw["intermediate_size"],
            vocab=raw["vocab_size"],
            aux_layers=tuple(int(i) + 1 for i in raw["target_layer_ids"]),
            mask_token=raw["mask_token_id"],
            eps=raw["rms_norm_eps"],
            markov_rank=raw["markov_rank"],
            block=int(raw.get("block_size", raw.get("dspark_block_size", 7))),
            rope_theta=float(rope["rope_theta"]),
            rope_factor=float(rope["factor"]),
            rope_original_max=int(rope["original_max_position_embeddings"]),
            rope_beta_fast=float(rope.get("beta_fast", 32)),
            rope_beta_slow=float(rope.get("beta_slow", 1)),
            rope_mscale=float(rope.get("mscale", 1)),
            rope_mscale_all_dim=float(rope.get("mscale_all_dim", 0)),
        )


def is_mla_draft(directory: Path | str) -> bool:
    """Whether the draft checkpoint at ``directory`` is this module's."""
    raw = json.loads((Path(directory) / "config.json").read_text())
    return raw.get("model_type") == "k3_dspark"


# Checkpoint tensor names (PyTorch ``[out, in]``) the draft uses; the
# checkpoint's ``embed_tokens``/``confidence_head`` are not used.
def weight_shapes(c: MlaDraftConfig) -> dict[str, tuple[int, ...]]:
    """Return expected checkpoint tensor shapes keyed by safetensors name."""
    shapes: dict[str, tuple[int, ...]] = {
        "context_proj.weight": (c.hidden, c.context_width),
        "context_norm.weight": (c.hidden,),
        "final_norm.weight": (c.hidden,),
        "markov_head.markov_w1.weight": (c.vocab, c.markov_rank),
        "markov_head.markov_w2.weight": (c.vocab, c.markov_rank),
    }
    for layer in range(c.layers):
        prefix = f"layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (c.hidden,),
                prefix + "post_attention_layernorm.weight": (c.hidden,),
                prefix + "self_attn.q_a_proj.weight": (c.q_lora, c.hidden),
                prefix + "self_attn.q_a_layernorm.weight": (c.q_lora,),
                prefix + "self_attn.q_b_proj.weight": (c.heads * c.qk_head, c.q_lora),
                prefix + "self_attn.kv_a_proj_with_mqa.weight": (c.latent, c.hidden),
                prefix + "self_attn.kv_a_layernorm.weight": (c.kv_lora,),
                prefix + "self_attn.kv_b_proj.weight": (
                    c.heads * (c.nope + c.v_head),
                    c.kv_lora,
                ),
                prefix + "self_attn.o_proj.weight": (c.hidden, c.heads * c.v_head),
                prefix + "mlp.gate_proj.weight": (c.intermediate, c.hidden),
                prefix + "mlp.up_proj.weight": (c.intermediate, c.hidden),
                prefix + "mlp.down_proj.weight": (c.hidden, c.intermediate),
            }
        )
    return shapes


def load_checkpoint_weights(
    directory: Path | str,
    c: MlaDraftConfig | None = None,
    names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Checkpoint-named bf16 numpy arrays (``names``: a subset)."""
    directory = Path(directory)
    c = c or MlaDraftConfig.from_checkpoint(directory)
    shapes = weight_shapes(c)
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(directory / "model.safetensors"), framework="numpy") as handle:
        for name in names or shapes:
            value = handle.get_tensor(name)
            if value.dtype == np.uint16:
                value = value.view(ml_dtypes.bfloat16)
            if tuple(value.shape) != shapes[name]:
                err = _value_error(
                    f"{name}: expected {shapes[name]}, found {value.shape}"
                )
                raise err
            weights[name] = np.asarray(value).astype(ml_dtypes.bfloat16)
    return weights


# --------------------------------------------------------------------------
# YaRN RoPE (vLLM's DeepseekScalingRotaryEmbedding, GPT-J/interleaved pairs)


def rope_inv_freq(c: MlaDraftConfig) -> np.ndarray:
    """``[rope / 2]`` FP32 inverse frequencies."""
    dim, base = c.rope, c.rope_theta
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    extrapolation = 1.0 / pos_freqs
    interpolation = 1.0 / (c.rope_factor * pos_freqs)

    def correction_dim(rotations: float) -> float:
        """Compute YaRN wavelength correction dimension for a rotation count."""
        return (dim * math.log(c.rope_original_max / (rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    low = max(math.floor(correction_dim(c.rope_beta_fast)), 0)
    high = min(math.ceil(correction_dim(c.rope_beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return (interpolation * (1 - mask) + extrapolation * mask).astype(np.float32)


def rope_mscale(c: MlaDraftConfig) -> float:
    """Compute YaRN rotary magnitude scale factor."""

    def get(scale: float, mscale: float) -> float:
        """Return log-scaled YaRN multiplier for a single mscale parameter."""
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

    return float(
        get(c.rope_factor, c.rope_mscale) / get(c.rope_factor, c.rope_mscale_all_dim)
    )


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Apply interleaved-pair RoPE in FP32 and cast back to ``x``'s dtype."""
    x32 = x.astype(jnp.float32)
    shape = (cos.shape[0],) + (1,) * (x.ndim - 2) + (cos.shape[1],)
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    even, odd = x32[..., 0::2], x32[..., 1::2]
    rotated = jnp.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1)
    return rotated.reshape(x.shape).astype(x.dtype)


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """Apply RMSNorm with FP32 statistics, cast back, and scale by weight."""
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return normed.astype(x.dtype) * weight.astype(x.dtype)


# --------------------------------------------------------------------------
# Kernel layout (TP32: two heads per rank)

RANKS = 32
ROWS = 8  # block rows per request in the kernel (block <= 8; the rest are padding)
HEADS_PER_RANK = 2
LANES = 640  # a latent row: c_kv 512 | k_pe 64 | zero 64 (the pools' layout)
QKV_A = 1536 + LANES  # fused q_a | kv_a (padded) columns
MLP_PADDED = 512  # 448 MLP channels per rank padded to whole lane tiles
CHUNK = 7168 // 8  # 896-column chunk of the hidden vector per host lane
FC_IN = 35840 // 4  # aux features contracted per host (8960)
KERNEL_SHARD_FORMAT = 1
DRAFT_KIND = "k3_dspark_mla"
TILE = 128
QKV_TILES = QKV_A // TILE  # 17 per layer: q_a in 12, kv_a in 5
CONTEXT_TILES = LANES // TILE  # 5 per layer: the context rows' kv_a


def qkv_tiles(qkv_a: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Return rank's layer and context tiles of the fused projection ``qkv_a``."""
    layers, hidden, width = qkv_a.shape
    if width != QKV_A or layers * CONTEXT_TILES > RANKS:
        err = _value_error(f"qkv_a {qkv_a.shape}: not [layers <= 6, hidden, {QKV_A}]")
        raise err
    layer_tile = np.zeros((layers, hidden, TILE), qkv_a.dtype)
    if rank < QKV_TILES:
        layer_tile[...] = qkv_a[:, :, rank * TILE : (rank + 1) * TILE]
    context_tile = np.zeros((hidden, TILE), qkv_a.dtype)
    if rank < layers * CONTEXT_TILES:
        layer, k = divmod(rank, CONTEXT_TILES)
        start = QKV_A - LANES + k * TILE
        context_tile[...] = qkv_a[layer, :, start : start + TILE]
    return layer_tile, context_tile


def rope_table(c: MlaDraftConfig, max_position: int) -> np.ndarray:
    """Return ``[P, 2 * rope]`` FP32 interleaved cos/sin rotary table."""
    freqs = (
        np.arange(max_position, dtype=np.float32)[:, None] * rope_inv_freq(c)[None, :]
    )
    scale = rope_mscale(c)
    cos, sin = np.cos(freqs) * scale, np.sin(freqs) * scale
    return np.concatenate(
        (np.repeat(cos, 2, axis=1), np.repeat(sin, 2, axis=1)), axis=1
    ).astype(np.float32)


def rope_rotation(c: MlaDraftConfig) -> np.ndarray:
    """``[rope, rope]``: ``(x @ R)[2i] = -x[2i+1]``, ``(x @ R)[2i+1] = x[2i]``."""
    rotation = np.zeros((c.rope, c.rope), np.float32)
    for i in range(c.rope // 2):
        rotation[2 * i + 1, 2 * i] = -1.0
        rotation[2 * i, 2 * i + 1] = 1.0
    return rotation


def _replicated_mla_shards(
    c: MlaDraftConfig, w: dict[str, np.ndarray], ranks: int
) -> dict[str, np.ndarray]:
    """Build replicated MLA draft weight views across all ranks."""
    bf16 = ml_dtypes.bfloat16
    num_layers = c.layers

    def replicated(value: np.ndarray) -> np.ndarray:
        """Broadcast a layer tensor across the leading rank dimension."""
        value = np.ascontiguousarray(value)
        return np.broadcast_to(value[None], (ranks, *value.shape))

    def layer(name: str, layer_idx: int) -> np.ndarray:
        """Fetch weight array for a specific draft layer."""
        return w[f"layers.{layer_idx}.{name}"]

    qkv = np.zeros((num_layers, c.hidden, QKV_A), np.float32)
    kv_norm = np.zeros((num_layers, 1, LANES), np.float32)
    for layer_idx in range(num_layers):
        qkv[layer_idx, :, : c.q_lora] = layer("self_attn.q_a_proj.weight", layer_idx).T
        qkv[layer_idx, :, c.q_lora : c.q_lora + c.latent] = layer(
            "self_attn.kv_a_proj_with_mqa.weight", layer_idx
        ).T
        kv_norm[layer_idx, 0, : c.kv_lora] = layer(
            "self_attn.kv_a_layernorm.weight", layer_idx
        )
    return {
        "qkv_a": replicated(qkv.astype(bf16)),
        "q_norm": replicated(
            np.stack(
                [
                    layer("self_attn.q_a_layernorm.weight", layer_idx)[None]
                    for layer_idx in range(num_layers)
                ]
            ).astype(bf16)
        ),
        "kv_norm": replicated(kv_norm.astype(bf16)),
        "ln1": replicated(
            np.stack(
                [
                    layer("input_layernorm.weight", layer_idx)[None]
                    for layer_idx in range(num_layers)
                ]
            ).astype(bf16)
        ),
        "ln2": replicated(
            np.stack(
                [
                    layer("post_attention_layernorm.weight", layer_idx)[None]
                    for layer_idx in range(num_layers)
                ]
            ).astype(bf16)
        ),
        "context_norm": replicated(w["context_norm.weight"][None].astype(bf16)),
        "final_norm": replicated(w["final_norm.weight"][None].astype(bf16)),
        "w1": replicated(w["markov_head.markov_w1.weight"].astype(bf16)),
        "rot": replicated(rope_rotation(c).astype(bf16)),
    }


def kernel_weight_shards(
    c: MlaDraftConfig, weights: dict[str, Any], *, ranks: int = RANKS
) -> dict[str, np.ndarray]:
    """Build per-rank ``[ranks, ...]`` MLA draft weight shards."""
    bf16 = ml_dtypes.bfloat16
    if c.heads != HEADS_PER_RANK * ranks or c.intermediate % ranks or c.vocab % ranks:
        err = _value_error(
            "the MLA draft kernel needs 64 heads and dims dividing 32 ranks"
        )
        raise err
    w = {name: np.asarray(value) for name, value in weights.items()}
    mlp = c.intermediate // ranks
    vocab = c.vocab // ranks
    num_layers = c.layers
    rep = _replicated_mla_shards(c, w, ranks)
    shards: dict[str, np.ndarray] = {
        "qkv_a": rep["qkv_a"],
        "q_norm": rep["q_norm"],
        "kv_norm": rep["kv_norm"],
    }

    def per_rank(make: Callable[[int, int], np.ndarray]) -> np.ndarray:
        """Stack per-(layer, rank) shards into [ranks, layers, ...]."""
        return np.stack(
            [
                np.stack([make(layer_idx, rank) for layer_idx in range(num_layers)])
                for rank in range(ranks)
            ]
        )

    def permute_chunks(value: np.ndarray, rank: int) -> np.ndarray:
        """Permute hidden-dimension chunks into ring-reduce order for rank."""
        lane = rank % 8
        return np.concatenate(
            [value[:, (lane ^ k) * CHUNK : ((lane ^ k) + 1) * CHUNK] for k in range(8)],
            axis=1,
        )

    def layer(name: str, layer_idx: int) -> np.ndarray:
        """Fetch weight array for a specific draft layer."""
        return w[f"layers.{layer_idx}.{name}"]

    q_width = HEADS_PER_RANK * c.qk_head

    def q_b(layer_idx: int, rank_idx: int) -> np.ndarray:
        """Return this rank's two heads of q_b_proj padded to 128-lane tiles."""
        value = layer("self_attn.q_b_proj.weight", layer_idx)[
            rank_idx * q_width : (rank_idx + 1) * q_width
        ].T  # [1536, 2 x 192]
        heads = value.reshape(c.q_lora, HEADS_PER_RANK, c.qk_head)
        rope = np.pad(heads[:, :, c.nope :], ((0, 0), (0, 0), (0, 128 - c.rope)))
        return np.concatenate(
            (heads[:, :, : c.nope].reshape(c.q_lora, -1), rope.reshape(c.q_lora, -1)),
            axis=1,
        )

    shards["q_b"] = per_rank(q_b).astype(bf16)  # [1536, 512]

    def kv_b(layer_idx: int, rank_idx: int) -> np.ndarray:
        """Return this rank's two heads of kv_b_proj."""
        return layer("self_attn.kv_b_proj.weight", layer_idx).reshape(
            c.heads, c.nope + c.v_head, c.kv_lora
        )[rank_idx * HEADS_PER_RANK : (rank_idx + 1) * HEADS_PER_RANK]

    shards["w_uk"] = per_rank(
        lambda layer_idx, rank_idx: kv_b(layer_idx, rank_idx)[:, : c.nope, :]
    ).astype(bf16)  # [2, 128, 512]
    shards["w_uv"] = per_rank(
        lambda layer_idx, rank_idx: np.transpose(
            kv_b(layer_idx, rank_idx)[:, c.nope :, :], (0, 2, 1)
        )
    ).astype(bf16)  # [2, 512, 128]
    o_width = HEADS_PER_RANK * c.v_head
    shards["o"] = per_rank(
        lambda layer_idx, rank_idx: permute_chunks(
            layer("self_attn.o_proj.weight", layer_idx)[
                :, rank_idx * o_width : (rank_idx + 1) * o_width
            ].T,
            rank_idx,
        )
    ).astype(bf16)

    def pad_columns(value: np.ndarray) -> np.ndarray:
        """Zero-pad trailing columns of an MLP projection to MLP_PADDED."""
        return np.pad(value, ((0, 0), (0, MLP_PADDED - value.shape[1])))

    def pad_rows(value: np.ndarray) -> np.ndarray:
        """Zero-pad trailing rows of an MLP down projection to MLP_PADDED."""
        return np.pad(value, ((0, MLP_PADDED - value.shape[0]), (0, 0)))

    shards["gate"] = per_rank(
        lambda layer_idx, rank_idx: pad_columns(
            layer("mlp.gate_proj.weight", layer_idx)[
                rank_idx * mlp : (rank_idx + 1) * mlp
            ].T
        )
    ).astype(bf16)
    shards["up"] = per_rank(
        lambda layer_idx, rank_idx: pad_columns(
            layer("mlp.up_proj.weight", layer_idx)[
                rank_idx * mlp : (rank_idx + 1) * mlp
            ].T
        )
    ).astype(bf16)
    shards["down"] = per_rank(
        lambda layer_idx, rank_idx: permute_chunks(
            pad_rows(
                layer("mlp.down_proj.weight", layer_idx)[
                    :, rank_idx * mlp : (rank_idx + 1) * mlp
                ].T
            ),
            rank_idx,
        )
    ).astype(bf16)
    shards["ln1"] = rep["ln1"]
    shards["ln2"] = rep["ln2"]
    shards["context_norm"] = rep["context_norm"]
    shards["final_norm"] = rep["final_norm"]
    fc = w["context_proj.weight"]  # [7168 out, 35840 in]
    shards["fc"] = np.stack(
        [
            fc[
                (rank_idx % 8) * CHUNK : (rank_idx % 8 + 1) * CHUNK,
                (rank_idx // 8) * FC_IN : (rank_idx // 8) * FC_IN + FC_IN,
            ].T
            for rank_idx in range(ranks)
        ]
    ).astype(bf16)
    w2 = w["markov_head.markov_w2.weight"].reshape(ranks, vocab, c.markov_rank)
    shards["w2t"] = np.ascontiguousarray(np.transpose(w2, (0, 2, 1))).astype(
        bf16
    )  # [ranks, 256, 5120]
    shards["w1"] = rep["w1"]
    shards["rot"] = rep["rot"]
    return shards


def save_kernel_shards(
    shards: dict[str, np.ndarray], directory: Path | str, *, ranks: int = RANKS
) -> None:
    """One raw file per rank plus a shared index (``kind``: this module)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "format": KERNEL_SHARD_FORMAT,
        "kind": DRAFT_KIND,
        "ranks": ranks,
        "arrays": {},
    }
    offset = 0
    for name, value in shards.items():
        index["arrays"][name] = {
            "offset": offset,
            "shape": list(value.shape[1:]),
            "dtype": str(np.dtype(value.dtype)),
        }
        offset += int(np.prod(value.shape[1:])) * np.dtype(value.dtype).itemsize
    index["bytes_per_rank"] = offset
    for rank in range(ranks):
        with (directory / f"rank{rank:02d}.bin").open("wb") as handle:
            for value in shards.values():
                handle.write(np.ascontiguousarray(value[rank]).tobytes())
    (directory / "index.json").write_text(json.dumps(index, indent=1))


def load_rank_shards(
    directory: Path | str, rank: int, names: Iterable[str] | None = None
) -> dict[str, np.ndarray]:
    """Load ``save_kernel_shards`` arrays for ``rank`` from ``directory``."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if (
        index.get("kind") != DRAFT_KIND
        or index.get("format") != KERNEL_SHARD_FORMAT
        or int(index.get("ranks", 0)) != RANKS
    ):
        err = _value_error(f"{directory}: not a 32-rank MLA DSpark kernel shard set")
        raise err
    path = directory / f"rank{rank:02d}.bin"
    raw = (
        np.fromfile(path, dtype=np.uint8)
        if names is None
        else np.memmap(path, dtype=np.uint8, mode="r")
    )
    name_set = set(names) if names is not None else None
    out: dict[str, np.ndarray] = {}
    for name, spec in index["arrays"].items():
        if name_set is not None and name not in name_set:
            continue
        dtype = (
            np.dtype(ml_dtypes.bfloat16)
            if spec["dtype"] == "bfloat16"
            else np.dtype(spec["dtype"])
        )
        count = int(np.prod(spec["shape"]))
        value = (
            raw[spec["offset"] : spec["offset"] + count * dtype.itemsize]
            .view(dtype)
            .reshape(spec["shape"])
        )
        out[name] = value if names is None else np.array(value)
    return out


# The head-sharded arrays and the axis their TP32 shards stack on under
# attention DP (each ``[layers, ...]``).
HELD_AXES = {"q_b": 2, "w_uk": 1, "w_uv": 1, "o": 1}


def held_shards(
    directory: Path | str, rank: int, dp: int, own: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Assemble ``rank``'s Attention-DP head-sharded arrays from TP32 shards."""
    group = RANKS // dp
    if group % 8:
        err = _value_error(
            f"attention DP{dp}: groups of {group} ranks are not whole hosts"
        )
        raise err
    shards = [
        own
        if rank % group + k * group == rank
        else load_rank_shards(directory, rank % group + k * group, names=HELD_AXES)
        for k in range(dp)
    ]
    return {
        name: np.concatenate([shard[name] for shard in shards], axis=axis)
        for name, axis in HELD_AXES.items()
    }
