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

# Source provenance: inferact/k3-megakernel@30c9ba966b src/vllm_torchtpu/megakernel/kimi_k3/kernels/dspark_mla.py
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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class MlaDraftConfig:
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
        return self.nope + self.rope

    @property
    def latent(self) -> int:
        return self.kv_lora + self.rope

    @property
    def context_width(self) -> int:
        return len(self.aux_layers) * self.hidden

    @property
    def softmax_scale(self) -> float:
        """``qk_head ** -0.5`` times the YaRN mscale squared (vLLM's K3 MLA
        with ``deepseek_yarn``)."""
        scale = self.qk_head**-0.5
        if self.rope_factor > 1:
            mscale = 0.1 * self.rope_mscale_all_dim * math.log(self.rope_factor) + 1.0
            scale *= mscale * mscale
        return scale

    @classmethod
    def from_checkpoint(cls, directory: Path) -> MlaDraftConfig:
        raw = json.loads((Path(directory) / "config.json").read_text())
        if raw.get("model_type") != "k3_dspark":
            raise ValueError(f"{directory}: not a k3_dspark draft ({raw.get('model_type')})")
        if raw.get("mla_use_nope") or raw.get("mla_use_output_gate"):
            raise ValueError("the MLA DSpark draft has RoPE and no output gate")
        rope = raw.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "yarn":
            raise ValueError(f"unsupported draft RoPE {rope}")
        if raw.get("draft_vocab_size", raw["vocab_size"]) != raw["vocab_size"]:
            raise ValueError("the draft vocabulary must be the target's")
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


def is_mla_draft(directory: Path) -> bool:
    """Whether the draft checkpoint at ``directory`` is this module's."""
    raw = json.loads((Path(directory) / "config.json").read_text())
    return raw.get("model_type") == "k3_dspark"


# Checkpoint tensor names (PyTorch ``[out, in]``) the draft uses; the
# checkpoint's ``embed_tokens``/``confidence_head`` are not used.
def weight_shapes(c: MlaDraftConfig) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "context_proj.weight": (c.hidden, c.context_width),
        "context_norm.weight": (c.hidden,),
        "final_norm.weight": (c.hidden,),
        "markov_head.markov_w1.weight": (c.vocab, c.markov_rank),
        "markov_head.markov_w2.weight": (c.vocab, c.markov_rank),
    }
    for layer in range(c.layers):
        p = f"layers.{layer}."
        shapes.update({
            p + "input_layernorm.weight": (c.hidden,),
            p + "post_attention_layernorm.weight": (c.hidden,),
            p + "self_attn.q_a_proj.weight": (c.q_lora, c.hidden),
            p + "self_attn.q_a_layernorm.weight": (c.q_lora,),
            p + "self_attn.q_b_proj.weight": (c.heads * c.qk_head, c.q_lora),
            p + "self_attn.kv_a_proj_with_mqa.weight": (c.latent, c.hidden),
            p + "self_attn.kv_a_layernorm.weight": (c.kv_lora,),
            p + "self_attn.kv_b_proj.weight": (c.heads * (c.nope + c.v_head), c.kv_lora),
            p + "self_attn.o_proj.weight": (c.hidden, c.heads * c.v_head),
            p + "mlp.gate_proj.weight": (c.intermediate, c.hidden),
            p + "mlp.up_proj.weight": (c.intermediate, c.hidden),
            p + "mlp.down_proj.weight": (c.hidden, c.intermediate),
        })
    return shapes


def load_checkpoint_weights(directory: Path, c: MlaDraftConfig | None = None, names=None):
    """Checkpoint-named bf16 numpy arrays (``names``: a subset)."""
    import ml_dtypes
    from safetensors import safe_open

    directory = Path(directory)
    c = c or MlaDraftConfig.from_checkpoint(directory)
    shapes = weight_shapes(c)
    weights = {}
    with safe_open(str(directory / "model.safetensors"), framework="numpy") as handle:
        for name in names or shapes:
            value = handle.get_tensor(name)
            if value.dtype == np.uint16:
                value = value.view(ml_dtypes.bfloat16)
            if tuple(value.shape) != shapes[name]:
                raise ValueError(f"{name}: expected {shapes[name]}, found {value.shape}")
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

    def correction_dim(rotations):
        return (dim * math.log(c.rope_original_max / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(correction_dim(c.rope_beta_fast)), 0)
    high = min(math.ceil(correction_dim(c.rope_beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    mask = 1 - ramp
    return (interpolation * (1 - mask) + extrapolation * mask).astype(np.float32)


def rope_mscale(c: MlaDraftConfig) -> float:
    def get(scale, mscale):
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0
    return float(get(c.rope_factor, c.rope_mscale) / get(c.rope_factor, c.rope_mscale_all_dim))


def apply_rope(x, cos, sin):
    """Interleaved-pair RoPE in FP32: ``x`` ``[tokens, ..., rope]``, cos/sin
    ``[tokens, rope / 2]``; returns ``x``'s dtype."""
    x32 = x.astype(jnp.float32)
    shape = (cos.shape[0],) + (1,) * (x.ndim - 2) + (cos.shape[1],)
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    even, odd = x32[..., 0::2], x32[..., 1::2]
    rotated = jnp.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1)
    return rotated.reshape(x.shape).astype(x.dtype)


def rms_norm(x, weight, eps):
    """vLLM RMSNorm: FP32 statistics, cast back, then scale by the weight."""
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
    """Rank ``rank``'s share of the replicated fused projection ``qkv_a``
    ``[layers, 7168, QKV_A]``: ``(layer_tile [layers, 7168, 128], context_tile
    [7168, 128])``. Rank ``r < QKV_TILES`` computes every layer's block rows'
    columns ``128 r ..``; rank ``r < layers * CONTEXT_TILES`` the context rows'
    kv_a columns ``128 (r % 5) ..`` of layer ``r // 5``; the others hold zeros.
    The kernel gathers the tiles (one exchange per layer, one for the context)."""
    layers, hidden, width = qkv_a.shape
    if width != QKV_A or layers * CONTEXT_TILES > RANKS:
        raise ValueError(f"qkv_a {qkv_a.shape}: not [layers <= 6, hidden, {QKV_A}]")
    layer_tile = np.zeros((layers, hidden, TILE), qkv_a.dtype)
    if rank < QKV_TILES:
        layer_tile[...] = qkv_a[:, :, rank * TILE:(rank + 1) * TILE]
    context_tile = np.zeros((hidden, TILE), qkv_a.dtype)
    if rank < layers * CONTEXT_TILES:
        layer, k = divmod(rank, CONTEXT_TILES)
        start = QKV_A - LANES + k * TILE
        context_tile[...] = qkv_a[layer, :, start:start + TILE]
    return layer_tile, context_tile


def rope_table(c: MlaDraftConfig, max_position: int) -> np.ndarray:
    """``[P, 2 rope]`` FP32 rows ``cos2 | sin2``: each frequency's cos/sin
    duplicated over its interleaved pair (``x * cos2 + (x @ R) * sin2``)."""
    freqs = np.arange(max_position, dtype=np.float32)[:, None] * rope_inv_freq(c)[None, :]
    scale = rope_mscale(c)
    cos, sin = np.cos(freqs) * scale, np.sin(freqs) * scale
    return np.concatenate((np.repeat(cos, 2, axis=1), np.repeat(sin, 2, axis=1)), axis=1).astype(np.float32)


def rope_rotation(c: MlaDraftConfig) -> np.ndarray:
    """``[rope, rope]``: ``(x @ R)[2i] = -x[2i+1]``, ``(x @ R)[2i+1] = x[2i]``."""
    rotation = np.zeros((c.rope, c.rope), np.float32)
    for i in range(c.rope // 2):
        rotation[2 * i + 1, 2 * i] = -1.0
        rotation[2 * i, 2 * i + 1] = 1.0
    return rotation


def kernel_weight_shards(c: MlaDraftConfig, weights: dict[str, Any], *, ranks: int = RANKS) -> dict[str, np.ndarray]:
    """Per-rank ``[ranks, ...]`` arrays (replicated ones as broadcast views)
    in the kernel's layout from checkpoint-named arrays. Rank ``r`` holds heads
    ``2r, 2r + 1``."""
    import ml_dtypes

    bf16 = ml_dtypes.bfloat16
    if c.heads != HEADS_PER_RANK * ranks or c.intermediate % ranks or c.vocab % ranks:
        raise ValueError("the MLA draft kernel needs 64 heads and dims dividing 32 ranks")
    w = {name: np.asarray(value) for name, value in weights.items()}
    mlp = c.intermediate // ranks
    vocab = c.vocab // ranks
    L = c.layers
    shards: dict[str, np.ndarray] = {}

    def replicated(value):
        value = np.ascontiguousarray(value)
        return np.broadcast_to(value[None], (ranks, *value.shape))

    def per_rank(make):
        return np.stack([np.stack([make(layer, rank) for layer in range(L)]) for rank in range(ranks)])

    def permute_chunks(value, rank):
        lane = rank % 8
        return np.concatenate([value[:, (lane ^ k) * CHUNK:((lane ^ k) + 1) * CHUNK] for k in range(8)], axis=1)

    def layer(name, l):
        return w[f"layers.{l}.{name}"]

    qkv = np.zeros((L, c.hidden, QKV_A), np.float32)
    for l in range(L):
        qkv[l, :, :c.q_lora] = layer("self_attn.q_a_proj.weight", l).T
        qkv[l, :, c.q_lora:c.q_lora + c.latent] = layer("self_attn.kv_a_proj_with_mqa.weight", l).T
    shards["qkv_a"] = replicated(qkv.astype(bf16))
    shards["q_norm"] = replicated(np.stack([layer("self_attn.q_a_layernorm.weight", l)[None] for l in range(L)]).astype(bf16))
    kv_norm = np.zeros((L, 1, LANES), np.float32)
    for l in range(L):
        kv_norm[l, 0, :c.kv_lora] = layer("self_attn.kv_a_layernorm.weight", l)
    shards["kv_norm"] = replicated(kv_norm.astype(bf16))
    q_width = HEADS_PER_RANK * c.qk_head

    def q_b(l, r):
        """This rank's heads' q_b columns: nope 0 | nope 1 | rope 0, rope 1
        each padded to 128 lanes (every slice starts on a lane tile)."""
        value = layer("self_attn.q_b_proj.weight", l)[r * q_width:(r + 1) * q_width].T  # [1536, 2 x 192]
        heads = value.reshape(c.q_lora, HEADS_PER_RANK, c.qk_head)
        rope = np.pad(heads[:, :, c.nope:], ((0, 0), (0, 0), (0, 128 - c.rope)))
        return np.concatenate((heads[:, :, :c.nope].reshape(c.q_lora, -1), rope.reshape(c.q_lora, -1)), axis=1)

    shards["q_b"] = per_rank(q_b).astype(bf16)  # [1536, 512]

    def kv_b(l, r):
        return layer("self_attn.kv_b_proj.weight", l).reshape(c.heads, c.nope + c.v_head, c.kv_lora)[
            r * HEADS_PER_RANK:(r + 1) * HEADS_PER_RANK]

    shards["w_uk"] = per_rank(lambda l, r: kv_b(l, r)[:, :c.nope, :]).astype(bf16)  # [2, 128, 512]
    shards["w_uv"] = per_rank(lambda l, r: np.transpose(kv_b(l, r)[:, c.nope:, :], (0, 2, 1))).astype(bf16)  # [2, 512, 128]
    o_width = HEADS_PER_RANK * c.v_head
    shards["o"] = per_rank(
        lambda l, r: permute_chunks(layer("self_attn.o_proj.weight", l)[:, r * o_width:(r + 1) * o_width].T, r)
    ).astype(bf16)

    def pad_columns(value):
        return np.pad(value, ((0, 0), (0, MLP_PADDED - value.shape[1])))

    def pad_rows(value):
        return np.pad(value, ((0, MLP_PADDED - value.shape[0]), (0, 0)))

    shards["gate"] = per_rank(lambda l, r: pad_columns(layer("mlp.gate_proj.weight", l)[r * mlp:(r + 1) * mlp].T)).astype(bf16)
    shards["up"] = per_rank(lambda l, r: pad_columns(layer("mlp.up_proj.weight", l)[r * mlp:(r + 1) * mlp].T)).astype(bf16)
    shards["down"] = per_rank(
        lambda l, r: permute_chunks(pad_rows(layer("mlp.down_proj.weight", l)[:, r * mlp:(r + 1) * mlp].T), r)
    ).astype(bf16)
    shards["ln1"] = replicated(np.stack([layer("input_layernorm.weight", l)[None] for l in range(L)]).astype(bf16))
    shards["ln2"] = replicated(np.stack([layer("post_attention_layernorm.weight", l)[None] for l in range(L)]).astype(bf16))
    shards["context_norm"] = replicated(w["context_norm.weight"][None].astype(bf16))
    shards["final_norm"] = replicated(w["final_norm.weight"][None].astype(bf16))
    fc = w["context_proj.weight"]  # [7168 out, 35840 in]
    shards["fc"] = np.stack([
        fc[(r % 8) * CHUNK:(r % 8 + 1) * CHUNK, (r // 8) * FC_IN:(r // 8 + 1) * FC_IN].T for r in range(ranks)
    ]).astype(bf16)
    w2 = w["markov_head.markov_w2.weight"].reshape(ranks, vocab, c.markov_rank)
    shards["w2t"] = np.ascontiguousarray(np.transpose(w2, (0, 2, 1))).astype(bf16)  # [ranks, 256, 5120]
    shards["w1"] = replicated(w["markov_head.markov_w1.weight"].astype(bf16))
    shards["rot"] = replicated(rope_rotation(c).astype(bf16))
    return shards


def save_kernel_shards(shards: dict[str, np.ndarray], directory, *, ranks: int = RANKS) -> None:
    """One raw file per rank plus a shared index (``kind``: this module)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = {"format": KERNEL_SHARD_FORMAT, "kind": DRAFT_KIND, "ranks": ranks, "arrays": {}}
    offset = 0
    for name, value in shards.items():
        index["arrays"][name] = {"offset": offset, "shape": list(value.shape[1:]), "dtype": str(np.dtype(value.dtype))}
        offset += int(np.prod(value.shape[1:])) * np.dtype(value.dtype).itemsize
    index["bytes_per_rank"] = offset
    for rank in range(ranks):
        with open(directory / f"rank{rank:02d}.bin", "wb") as handle:
            for name, value in shards.items():
                handle.write(np.ascontiguousarray(value[rank]).tobytes())
    (directory / "index.json").write_text(json.dumps(index, indent=1))


def load_rank_shards(directory, rank: int, names=None) -> dict[str, np.ndarray]:
    """``save_kernel_shards``' arrays for ``rank`` (views of one read;
    ``names``: copies of only those arrays' bytes)."""
    import ml_dtypes

    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if (index.get("kind") != DRAFT_KIND or index.get("format") != KERNEL_SHARD_FORMAT
            or int(index.get("ranks", 0)) != RANKS):
        raise ValueError(f"{directory}: not a 32-rank MLA DSpark kernel shard set")
    path = directory / f"rank{rank:02d}.bin"
    raw = np.fromfile(path, dtype=np.uint8) if names is None else np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, spec in index["arrays"].items():
        if names is not None and name not in names:
            continue
        dtype = np.dtype(ml_dtypes.bfloat16) if spec["dtype"] == "bfloat16" else np.dtype(spec["dtype"])
        count = int(np.prod(spec["shape"]))
        value = raw[spec["offset"]: spec["offset"] + count * dtype.itemsize].view(dtype).reshape(spec["shape"])
        out[name] = value if names is None else np.array(value)
    return out


# The head-sharded arrays and the axis their TP32 shards stack on under
# attention DP (each ``[layers, ...]``).
HELD_AXES = {"q_b": 2, "w_uk": 1, "w_uv": 1, "o": 1}


def held_shards(directory, rank: int, dp: int, own: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Attention DP: ``rank``'s head-sharded arrays (``HELD_AXES``) holding
    its TP32 shards ``rank % (32 / dp) + k (32 / dp)`` in order ``k``, as the
    decoder's attention holds them (``rank_loader.attention_dp_families``);
    ``own`` is ``rank``'s shard. The shards share the rank's host lane, so
    their ``o`` chunk permutations agree."""
    group = RANKS // dp
    if group % 8:
        raise ValueError(f"attention DP{dp}: groups of {group} ranks are not whole hosts")
    shards = [own if rank % group + k * group == rank
              else load_rank_shards(directory, rank % group + k * group, names=HELD_AXES)
              for k in range(dp)]
    return {name: np.concatenate([shard[name] for shard in shards], axis=axis) for name, axis in HELD_AXES.items()}
