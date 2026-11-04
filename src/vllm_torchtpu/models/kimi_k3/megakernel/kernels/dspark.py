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

# Source provenance: inferact/k3-megakernel@30c9ba966b src/vllm_torchtpu/megakernel/kimi_k3/kernels/dspark.py
"""DSpark draft model for Kimi K3 (``RedHatAI/Kimi-K3-speculator.dspark``).

The draft is a DFlash backbone: a small Qwen3-style transformer whose keys
and values come from two sources, projected target hidden states (the
context, one row per token already run through the target) and the query
block itself (an anchor token followed by mask tokens). Every block slot
predicts the token one position after itself. DSpark adds a low-rank Markov
bias, ``markov_w2 @ markov_w1[previous token]``, applied while sampling the
block left to right.

This module holds:

* the checkpoint config and reader (``DraftConfig``,
  ``load_checkpoint_weights``) and the conversion to the fused kernel's
  per-rank layout (``kernel_weight_shards``, ``save_kernel_shards``);
* ``make_fused_draft``: one TP32 Pallas kernel per step that reads the verify
  logits, accepts drafts, extends each request's context ring and drafts the
  next block (output lanes ``OUT_*``);
* the ring from a prompt on the prefill side (``prefill_ring_update``,
  ``pack_ring`` / ``unpack_ring``);
* sampling helpers used outside the kernel (``nucleus_accept``,
  ``synthetic_count``).

Each request's draft context is a ring over ``window`` positions per layer:
slot ``position % window`` stores that position's packed key | value row and
a positions array records which position a slot holds (``-1`` when empty).
Rows written for positions that later turn out to be rejected drafts are
hidden by the ``position < anchor_position`` mask until the real token
overwrites them.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.models.kimi_k3.megakernel.common import collectives32

from .decode_megakernel import _attention_scores, _dot


@dataclasses.dataclass(frozen=True)
class DraftConfig:
    hidden: int = 7168
    layers: int = 5
    heads: int = 96
    kv_heads: int = 16
    head_dim: int = 64
    intermediate: int = 14336
    vocab: int = 163840
    aux_layers: tuple[int, ...] = (24, 48, 72, 88, 92)
    mask_token: int = 163837
    window: int = 2048
    rope_theta: float = 10000.0
    eps: float = 1e-5
    markov_rank: int = 256

    @property
    def q_width(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def context_width(self) -> int:
        return len(self.aux_layers) * self.hidden

    @classmethod
    def from_checkpoint(cls, directory: Path) -> DraftConfig:
        raw = json.loads((Path(directory) / "config.json").read_text())
        layer = raw["transformer_layer_config"]
        return cls(
            hidden=layer["hidden_size"],
            layers=layer["num_hidden_layers"],
            heads=layer["num_attention_heads"],
            kv_heads=layer["num_key_value_heads"],
            head_dim=layer["head_dim"],
            intermediate=layer["intermediate_size"],
            vocab=layer["vocab_size"],
            aux_layers=tuple(raw["aux_hidden_state_layer_ids"]),
            mask_token=raw["mask_token_id"],
            window=layer["sliding_window"],
            rope_theta=float(layer["rope_parameters"]["rope_theta"]),
            eps=layer["rms_norm_eps"],
            markov_rank=raw["markov_rank"],
        )


# Checkpoint tensor names (PyTorch ``[out, in]`` layouts) and their shapes.
def weight_shapes(c: DraftConfig) -> dict[str, tuple[int, ...]]:
    shapes = {
        "embed_tokens.weight": (c.vocab, c.hidden),
        "lm_head.weight": (c.vocab, c.hidden),
        "fc.weight": (c.hidden, c.context_width),
        "hidden_norm.weight": (c.hidden,),
        "norm.weight": (c.hidden,),
        "markov_head.markov_w1.weight": (c.vocab, c.markov_rank),
        "markov_head.markov_w2.weight": (c.vocab, c.markov_rank),
    }
    for layer in range(c.layers):
        prefix = f"layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (c.hidden,),
                prefix + "post_attention_layernorm.weight": (c.hidden,),
                prefix + "self_attn.q_proj.weight": (c.q_width, c.hidden),
                prefix + "self_attn.k_proj.weight": (c.kv_width, c.hidden),
                prefix + "self_attn.v_proj.weight": (c.kv_width, c.hidden),
                prefix + "self_attn.o_proj.weight": (c.hidden, c.q_width),
                prefix + "self_attn.q_norm.weight": (c.head_dim,),
                prefix + "self_attn.k_norm.weight": (c.head_dim,),
                prefix + "mlp.gate_proj.weight": (c.intermediate, c.hidden),
                prefix + "mlp.up_proj.weight": (c.intermediate, c.hidden),
                prefix + "mlp.down_proj.weight": (c.hidden, c.intermediate),
            }
        )
    return shapes


# --------------------------------------------------------------------------
# Shared arithmetic (mirrors transformers' Qwen3 bf16 modules)


def rms_norm(x, weight, eps):
    """Qwen3RMSNorm: FP32 statistics, cast back, then scale by the weight."""
    dtype = x.dtype
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return weight * normed.astype(dtype)


def synthetic_count(rates, seed, position):
    """Synthetic acceptance: the accepted draft count for one verify, where
    ``rates[i]`` (non-increasing, vLLM's ``synthetic_acceptance_rates``) is the
    probability that the first ``i + 1`` drafts are accepted, drawn
    deterministically from ``(seed, position)``."""
    uniform = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(seed), position))
    return jnp.sum(uniform < jnp.asarray(rates, jnp.float32)).astype(jnp.int32)


def nucleus_accept(values, ids, lse, drafts, forced_count, top_p, uniforms):
    """Speculative top-p sampling over gathered candidates; returns ``(count, bonus)``.

    ``values`` ``[rows, n]`` are candidate logits (temperature applied) with
    token ``ids``, ``lse`` ``[rows]`` the log-normalizer of each row over the
    full vocabulary, so ``exp(values - lse)`` are exact probabilities. Row k's
    nucleus p' keeps the largest candidates whose cumulative mass reaches
    ``top_p`` (at least one) and renormalizes. ``drafts[k]`` is the token the
    draft proposed for row k (-1: none; the last row never has one). The
    drafts are deterministic, so their distribution is one-hot and exact
    rejection sampling reduces to: accept draft k with probability p'_k(d_k)
    (``uniforms[k, 0] < p'``); ``count`` is the number of leading accepted
    drafts; the bonus token is drawn (``uniforms[count, 1]``) from p'_count
    with the rejected draft removed, or from the full p'_count when every
    draft was accepted. With ``forced_count >= 0`` the count is given and the
    bonus is drawn from the full p'_count (prefill). The output is
    distributed exactly as p' (up to the candidate truncation).
    """
    values, ids, lse, drafts, uniforms = (jnp.asarray(x) for x in (values, ids, lse, drafts, uniforms))
    rows, n = values.shape
    order = jnp.argsort(-values, axis=1)
    sorted_values = jnp.take_along_axis(values, order, axis=1)
    sorted_ids = jnp.take_along_axis(ids, order, axis=1)
    probs = jnp.exp(sorted_values - lse[:, None])  # [rows, n], descending
    cumulative = jnp.cumsum(probs, axis=1)
    keep = ((cumulative - probs) < top_p) | (jnp.arange(n)[None, :] == 0)
    kept = jnp.where(keep, probs, 0.0)
    nucleus = kept / jnp.sum(kept, axis=1, keepdims=True)  # p'
    draft_mask = sorted_ids == drafts[:, None]
    draft_prob = jnp.sum(jnp.where(draft_mask, nucleus, 0.0), axis=1)  # [rows]
    k = jnp.arange(rows, dtype=jnp.int32)
    drafted = drafts >= 0
    accept = drafted & (uniforms[:, 0] < draft_prob)  # a row without a draft is a rejection
    first_reject = jnp.min(jnp.where(~accept & (k < rows - 1), k, rows - 1))
    count = jnp.where(forced_count >= 0, forced_count, first_reject).astype(jnp.int32)
    row = nucleus[count]
    remove = (forced_count < 0) & drafted[count]  # the rejected draft leaves the residual
    residual = jnp.where(remove & draft_mask[count], 0.0, row)
    cumulative = jnp.cumsum(residual)
    total = cumulative[-1]
    pick = jnp.argmax(cumulative > uniforms[count, 1] * total)  # first index past the threshold
    bonus = jnp.where(total > 0, sorted_ids[count, pick], sorted_ids[count, 0])
    return count, bonus.astype(jnp.int32)


def load_checkpoint_weights(directory: Path, c: DraftConfig | None = None):
    """Read the safetensors file into checkpoint-named bf16 arrays."""
    from safetensors.numpy import load_file

    directory = Path(directory)
    c = c or DraftConfig.from_checkpoint(directory)
    import ml_dtypes

    raw = load_file(str(directory / "model.safetensors"))
    weights = {}
    for name, shape in weight_shapes(c).items():
        value = raw[name]
        if tuple(value.shape) != shape:
            raise ValueError(f"{name}: expected {shape}, found {value.shape}")
        # safetensors' numpy loader hands bf16 back as uint16 bit patterns.
        if value.dtype == np.uint16:
            value = value.view(ml_dtypes.bfloat16)
        weights[name] = np.asarray(value).astype(ml_dtypes.bfloat16)
    return weights
ROWS = 8  # maximum block width and padded public ABI (anchor + 7)
RANKS = 32
LM_CHUNK = 512  # LM head columns per chunk (the gate/up buffer width)
MLP_PADDED = 512  # 448 MLP channels per rank padded to whole lane tiles
LM_CHUNKS = 10  # 10 * 512 = 5120 vocabulary rows per rank
FC_IN = 35840 // 4  # aux features contracted per host (8960)
CHUNK = 7168 // 8  # 896-column chunk of the hidden vector per host lane
ROW_PACKED = 7168 // 256  # 28 FP32 rows hold one BF16 [1, 7168] row
NEG = -3.0e38

# Output vector layout (int32 [1, 128]): drafts, this step's verify targets,
# accepted count, bonus token (the next anchor), top-2 margins as FP32 bits.
OUT_DRAFTS, OUT_TARGETS, OUT_COUNT, OUT_BONUS, OUT_MARGINS = 0, 8, 16, 17, 24

# Semaphore index ranges (per slot) so a fast peer's next phase can never
# satisfy a wait of the previous phase.
SEM_RS, SEM_HOST, SEM_GATHER, SEM_TINY, SEM_BCAST, SEMS_PER_SLOT = 0, 8, 16, 32, 64, 96

VMEM_NAMES = (
    "ln1", "ln2", "qnorm", "kvnorm", "hidden_norm", "final_norm", "mask_embed",
    "rot_q", "rot_kv", "head_ones", "kv_ones", "head_select", "head_place",
)
HBM_NAMES = ("q", "kv", "o", "gate", "up", "down", "fc", "lm_head", "w2t", "w1", "embed", "rope_q", "rope_kv")
WEIGHT_NAMES = VMEM_NAMES + HBM_NAMES


# --------------------------------------------------------------------------
# Host-side weight preparation


def _rotation(head_dim: int) -> np.ndarray:
    """``x @ R`` = ``concat(-x[half:], x[:half])`` (neox rotate-half)."""
    half = head_dim // 2
    rotation = np.zeros((head_dim, head_dim), np.float32)
    for i in range(half):
        rotation[i + half, i] = -1.0
        rotation[i, i + half] = 1.0
    return rotation


def rope_tables(c: DraftConfig, max_position: int):
    """``[P, 384]`` (cos x3 | sin x3) for the 3 local q heads and ``[P, 256]``
    (cos | ones | sin | zeros) for the packed K/V row, FP32."""
    positions = np.arange(max_position, dtype=np.float32)
    inv_freq = 1.0 / c.rope_theta ** (np.arange(0, c.head_dim, 2, dtype=np.float32) / c.head_dim)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)
    heads = 3
    rope_q = np.concatenate([cos] * heads + [sin] * heads, axis=1).astype(np.float32)
    ones, zeros = np.ones_like(cos), np.zeros_like(sin)
    rope_kv = np.concatenate((cos, ones, sin, zeros), axis=1).astype(np.float32)
    return rope_q, rope_kv


def kernel_weight_shards(c: DraftConfig, weights: dict[str, Any], *, max_position: int = 8192,
                         ranks: int = RANKS) -> dict[str, np.ndarray]:
    """Per-rank ``[ranks, ...]`` arrays in the kernel's layout from checkpoint-named arrays."""
    import ml_dtypes

    bf16 = ml_dtypes.bfloat16
    w = {name: np.asarray(value) for name, value in weights.items()}
    q_width = c.heads // ranks * c.head_dim  # 192
    kv_share = ranks // c.kv_heads  # 2
    mlp = c.intermediate // ranks  # 448
    vocab = c.vocab // ranks  # 5120
    shards: dict[str, np.ndarray] = {}

    def stack_layers(make):
        return np.stack([np.stack([make(layer, rank) for rank in range(ranks)]) for layer in range(c.layers)], axis=1)

    shards["q"] = stack_layers(
        lambda l, r: w[f"layers.{l}.self_attn.q_proj.weight"][r * q_width:(r + 1) * q_width].T
    ).astype(bf16)

    def kv(l, r):
        head = r // kv_share
        k = w[f"layers.{l}.self_attn.k_proj.weight"][head * c.head_dim:(head + 1) * c.head_dim].T
        v = w[f"layers.{l}.self_attn.v_proj.weight"][head * c.head_dim:(head + 1) * c.head_dim].T
        return np.concatenate((k, v), axis=1)

    shards["kv"] = stack_layers(kv).astype(bf16)
    # o and down produce the partial that is reduce-scattered: memory block k of
    # their 896-column chunks holds logical chunk (lane ^ k), so chunk k can be
    # sent to peer rank ^ k as soon as the MXU produces it.
    def permute_chunks(value, rank):
        lane = rank % 8
        return np.concatenate([value[:, (lane ^ k) * CHUNK:((lane ^ k) + 1) * CHUNK] for k in range(8)], axis=1)

    shards["o"] = stack_layers(
        lambda l, r: permute_chunks(w[f"layers.{l}.self_attn.o_proj.weight"][:, r * q_width:(r + 1) * q_width].T, r)
    ).astype(bf16)
    # MLP channels padded from 448 to 512 per rank: lane windows in VMEM must be
    # whole 128-lane tiles. The padding columns/rows are zero.
    def pad_columns(value):
        return np.pad(value, ((0, 0), (0, MLP_PADDED - value.shape[1])))

    def pad_rows(value):
        return np.pad(value, ((0, MLP_PADDED - value.shape[0]), (0, 0)))

    shards["gate"] = stack_layers(
        lambda l, r: pad_columns(w[f"layers.{l}.mlp.gate_proj.weight"][r * mlp:(r + 1) * mlp].T)
    ).astype(bf16)
    shards["up"] = stack_layers(
        lambda l, r: pad_columns(w[f"layers.{l}.mlp.up_proj.weight"][r * mlp:(r + 1) * mlp].T)
    ).astype(bf16)
    shards["down"] = stack_layers(
        lambda l, r: permute_chunks(pad_rows(w[f"layers.{l}.mlp.down_proj.weight"][:, r * mlp:(r + 1) * mlp].T), r)
    ).astype(bf16)

    def replicated(value):
        return np.broadcast_to(np.asarray(value)[None], (ranks, *np.asarray(value).shape))

    shards["ln1"] = replicated(np.stack([w[f"layers.{l}.input_layernorm.weight"][None] for l in range(c.layers)])).astype(bf16)
    shards["ln2"] = replicated(np.stack([w[f"layers.{l}.post_attention_layernorm.weight"][None] for l in range(c.layers)])).astype(bf16)
    shards["qnorm"] = replicated(np.stack([np.tile(w[f"layers.{l}.self_attn.q_norm.weight"], 3)[None] for l in range(c.layers)])).astype(bf16)
    shards["kvnorm"] = replicated(np.stack([
        np.concatenate((w[f"layers.{l}.self_attn.k_norm.weight"], np.ones((c.head_dim,), np.float32)))[None]
        for l in range(c.layers)
    ])).astype(bf16)
    shards["hidden_norm"] = replicated(w["hidden_norm.weight"][None]).astype(bf16)
    shards["final_norm"] = replicated(w["norm.weight"][None]).astype(bf16)
    shards["mask_embed"] = replicated(w["embed_tokens.weight"][c.mask_token][None]).astype(bf16)

    # fc: rank (host h, lane j) contracts host h's quarter of the concatenated
    # aux features for output columns [896 j, 896 (j + 1)).
    fc = w["fc.weight"]  # [7168 out, 35840 in]
    shards["fc"] = np.stack([
        fc[(r % 8) * CHUNK:(r % 8 + 1) * CHUNK, (r // 8) * FC_IN:(r // 8 + 1) * FC_IN].T for r in range(ranks)
    ]).astype(bf16)

    assert vocab == LM_CHUNKS * LM_CHUNK
    lm = w["lm_head.weight"].reshape(ranks, LM_CHUNKS, LM_CHUNK, c.hidden)  # [ranks, 10, 512, 7168]
    shards["lm_head"] = np.ascontiguousarray(np.transpose(lm, (0, 1, 3, 2))).astype(bf16)  # [ranks, 10, 7168, 512]
    w2 = w["markov_head.markov_w2.weight"].reshape(ranks, vocab, c.markov_rank)
    shards["w2t"] = np.ascontiguousarray(np.transpose(w2, (0, 2, 1))).astype(bf16)  # [ranks, 256, 5120]
    shards["w1"] = replicated(w["markov_head.markov_w1.weight"]).astype(bf16)
    shards["embed"] = w["embed_tokens.weight"].reshape(ranks, vocab, c.hidden).astype(bf16)

    rotation = _rotation(c.head_dim)
    shards["rot_q"] = replicated(np.kron(np.eye(3, dtype=np.float32), rotation)).astype(bf16)
    rot_kv = np.zeros((2 * c.head_dim, 2 * c.head_dim), np.float32)
    rot_kv[:c.head_dim, :c.head_dim] = rotation
    shards["rot_kv"] = replicated(rot_kv).astype(bf16)
    shards["head_ones"] = replicated(np.kron(np.eye(3, dtype=np.float32), np.ones((c.head_dim, c.head_dim), np.float32))).astype(bf16)
    kv_ones = np.zeros((2 * c.head_dim, 2 * c.head_dim), np.float32)
    kv_ones[:c.head_dim, :c.head_dim] = 1.0
    shards["kv_ones"] = replicated(kv_ones).astype(bf16)
    select = np.zeros((3, 3 * c.head_dim, 2 * c.head_dim), np.float32)
    place = np.zeros((3, 2 * c.head_dim, 3 * c.head_dim), np.float32)
    for head in range(3):
        for i in range(c.head_dim):
            select[head, head * c.head_dim + i, i] = 1.0
            place[head, c.head_dim + i, head * c.head_dim + i] = 1.0
    shards["head_select"] = replicated(select).astype(bf16)
    shards["head_place"] = replicated(place).astype(bf16)
    rope_q, rope_kv = rope_tables(c, max_position + 32)
    shards["rope_q"] = replicated(rope_q)
    shards["rope_kv"] = replicated(rope_kv)
    return shards


KERNEL_SHARD_FORMAT = 1


def save_kernel_shards(shards: dict[str, np.ndarray], directory, *, ranks: int = RANKS) -> None:
    """Write ``kernel_weight_shards`` output as one raw file per rank plus a shared index."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(shards)
    index = {"format": KERNEL_SHARD_FORMAT, "ranks": ranks, "arrays": {}}
    offset = 0
    for name in names:
        value = shards[name]
        index["arrays"][name] = {"offset": offset, "shape": list(value.shape[1:]), "dtype": str(np.dtype(value.dtype))}
        offset += int(np.prod(value.shape[1:])) * np.dtype(value.dtype).itemsize
    index["bytes_per_rank"] = offset
    for rank in range(ranks):
        with open(directory / f"rank{rank:02d}.bin", "wb") as handle:
            for name in names:
                handle.write(np.ascontiguousarray(shards[name][rank]).tobytes())
    (directory / "index.json").write_text(json.dumps(index, indent=1))


# --------------------------------------------------------------------------
# In-kernel helpers


def pack(value):
    """``[rows, width]`` -> FP32 ``[rows * width / 256, 128]`` words holding BF16 pairs."""
    rows, width = value.shape
    lanes = value.astype(jnp.float32).reshape(rows * width // 128, 128).astype(jnp.bfloat16)
    return jax.lax.bitcast_convert_type(tpu.bitcast(lanes, jnp.uint32), jnp.float32)


def unpack(words, rows, width):
    """Inverse of :func:`pack`, returned as FP32 ``[rows, width]``."""
    values = tpu.bitcast(jax.lax.bitcast_convert_type(words, jnp.uint32), jnp.bfloat16)
    return values.astype(jnp.float32).reshape(rows, width)


def _bf16(x):
    return x.astype(jnp.bfloat16)


def _f32(x):
    return x.astype(jnp.float32)


def _rms_norm(x_bf16, weight_bf16, eps):
    """Qwen3RMSNorm as the reference computes it: FP32 statistics, BF16 cast, BF16 scale."""
    x32 = _f32(x_bf16)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return _bf16(_f32(_bf16(normed)) * _f32(weight_bf16))


def _grouped_sumsq(x32, ones_bf16):
    """Per-group sum of squares broadcast to the group's lanes: ``x^2 @ blockdiag(ones)``
    with the FP32 square split into BF16 high and low parts (relative error ~2^-16)."""
    squares = x32 * x32
    high = _bf16(squares)
    low = _bf16(squares - _f32(high))
    return _dot(high, ones_bf16) + _dot(low, ones_bf16)


def _rope(x_bf16, rotation_bf16, cos32, sin32):
    """Reference ``x * cos + rotate_half(x) * sin`` in BF16 arithmetic; cos/sin are BF16-valued."""
    rotated = _bf16(_dot(x_bf16, rotation_bf16))
    first = _bf16(_f32(x_bf16) * cos32)
    second = _bf16(_f32(rotated) * sin32)
    return _bf16(_f32(first) + _f32(second))


def _select_rows(block32, offset, rows):
    """Rows ``offset .. offset + rows`` of a ``[16, W]`` value as ``[rows, W]`` (masked sums)."""
    row_index = jax.lax.broadcasted_iota(jnp.int32, block32.shape, 0)
    return jnp.concatenate(
        [jnp.sum(jnp.where(row_index == offset + k, block32, 0.0), axis=0, keepdims=True) for k in range(rows)],
        axis=0,
    )


# --------------------------------------------------------------------------
# The kernel


SMALL_NAMES = ("aux", "positions", "previous") + VMEM_NAMES  # HBM operands copied into VMEM at entry (the logits land in logits_vmem)


def _fused_rows(requests: int, block: int) -> int:
    """Physical rows per request, bounded to 64 rows per fused draft.

    Eight rows per request up to 8 requests; four above (block <= 4), so the
    unused padded ABI rows do not consume projection and collective scratch.
    """
    if not 2 <= block <= ROWS:
        raise ValueError("block must be between 2 and 8")
    if not 1 <= requests <= 16 or (requests > 8 and block > 4):
        raise ValueError("fused draft supports up to 8 requests at block 8, or 16 at block 4")
    return ROWS if requests <= 8 else 4


def _make_body(c: DraftConfig, block: int, eps: float, requests: int = 1):
    """One call covers ``requests`` sequences: their block rows fold into the row
    axis of every projection, so the weights cross HBM to VMEM once per call,
    while each request keeps its own ring, positions, anchor and sampling chain.
    The rings stay in HBM (the per-request rings do not fit in VMEM): (layer,
    request) slices stream through two buffers, each updated, attended and
    written back. From eight requests the FC stages one request's aux features.
    The cross-host partial sums travel as BF16 pairs."""
    layers = c.layers
    window = c.window
    R = requests
    rows = _fused_rows(requests, block)
    T = R * rows
    stream_aux = R >= 8
    kv_width = 2 * c.head_dim  # 128
    q_width = 3 * c.head_dim  # 192
    vocab = c.vocab // RANKS  # 5120
    hidden = c.hidden
    scale = c.head_dim ** -0.5

    # DMA semaphore indices (the array has 10 + 6 * R + 4 entries).
    # 10 + j: embed blocks; 10 + R + j: rope_q; 10 + 2R/3R + j: rope_kv slots;
    # 10 + 4R/5R + j: w1 blocks; 10 + 6R .. +3: the ring slice stream (in/out
    # per buffer parity).
    SEM_FC, SEM_W2T, SEM_OWN = 1, 8, 9
    SEM_RING = 10 + 6 * R

    def body(scalars, aux_hbm, positions_hbm, logits_hbm, previous_hbm, *refs):
        weights = dict(zip(WEIGHT_NAMES, refs[: len(WEIGHT_NAMES)], strict=True))
        cache_in = refs[len(WEIGHT_NAMES)]
        tokens_out, positions_out, cache_out = refs[len(WEIGHT_NAMES) + 1: len(WEIGHT_NAMES) + 4]
        scratch = refs[len(WEIGHT_NAMES) + 4:]
        small = dict(zip(SMALL_NAMES, scratch[: len(SMALL_NAMES)], strict=True))
        (
            a_buf, q_buf, kv_buf, o_buf, gate_buf, up_buf, down_buf, w2t_buf, w1_blk, embed_blk, bcast_buf,
            rope_q_blk, rope_kv_blk, ring_vmem, logits_vmem, fc_part, tokens_v, stage, local_recv,
            host_packed, gather_recv, tiny_recv, argmax_recv, newkv_buf, dma_sems, small_sems, sends, recvs,
        ) = scratch[len(SMALL_NAMES):]

        rank = jax.lax.axis_index("tp")
        lane = rank % 8
        host = rank // 8
        # The scalars operand is SMEM-prefetched: only element loads.
        new_position = jnp.stack([scalars[j] for j in range(R)])  # [R]: the verify step's first row
        forced_count = jnp.stack([scalars[R + j] for j in range(R)])  # [R]: >= 0 fixes the accepted count (prefill); else from the matches
        forced_anchor = jnp.stack([scalars[2 * R + j] for j in range(R)])  # [R]: >= 0 fixes the bonus token (sampled by the caller); else the target at ``count``

        # ---------------- DMA helpers ----------------
        def copy(src, dst, sem):
            return tpu.make_async_copy(src, dst, dma_sems.at[sem])

        def remote(src, dst, sem_index, peer):
            return tpu.make_async_remote_copy(
                src, dst, sends.at[sem_index], recvs.at[sem_index],
                device_id=(peer,), device_id_type=pl.DeviceIdType.MESH,
            )

        def exchange_start(src_of, dst_of, offsets, base, slot):
            copies = [
                remote(src_of(offset), dst_of(offset), slot * SEMS_PER_SLOT + base + offset, rank ^ offset)
                for offset in offsets
            ]
            for item in copies:
                item.start()
            return copies

        def exchange_wait(copies):
            for item in copies:
                item.wait()

        def exchange(src_of, dst_of, offsets, base, slot):
            exchange_wait(exchange_start(src_of, dst_of, offsets, base, slot))

        # ---------------- collectives ----------------
        def host_reduce_gather(chunk32, slot):
            """FP32 [T, 896] chunk partial of this host lane -> full [T, 7168] sum over hosts,
            BF16-rounded (the reference casts the reduced value to BF16)."""
            host_packed[slot, host] = pack(_bf16(chunk32))
            exchange(lambda o: host_packed.at[slot, host], lambda o: host_packed.at[slot, host], (8, 16, 24), SEM_HOST, slot)
            reduced = sum(unpack(host_packed[slot, h], T, CHUNK) for h in range(4))
            gather_recv[slot, lane] = pack(_bf16(reduced))
            exchange(lambda o: gather_recv.at[slot, lane], lambda o: gather_recv.at[slot, lane], range(1, 8), SEM_GATHER, slot)
            return jnp.concatenate([unpack(gather_recv[slot, j], T, CHUNK) for j in range(8)], axis=1)

        def all_reduce_start(lhs_bf16, weight_ref, slot, refill=None, refill_sem=7):
            """Project ``lhs`` through a chunk-permuted ``[K, 7168]`` weight and issue the
            host-local reduce-scatter chunk by chunk: memory block k holds logical
            chunk ``lane ^ k`` and goes to peer ``rank ^ k`` right after its matmul."""
            copies = []
            own = None
            width = weight_ref.shape[1]
            stream = refill is not None and width == CHUNK
            # Streamed (one-chunk buffer): double-buffer with a chunk of the FC
            # scratch (unused until the LM head); a chunk is waited before use
            # and a buffer refilled only after the matmul that read it.
            spare = a_buf.at[pl.ds(0, weight_ref.shape[0]), pl.ds(0, CHUNK)] if stream else weight_ref
            pending_refill = None
            for k in range(8):
                current = weight_ref if not stream or k % 2 == 0 else spare
                if stream:
                    assert refill is not None
                    if pending_refill is not None:
                        pending_refill.wait()
                    if k < 7:
                        destination = spare if k % 2 == 0 else weight_ref
                        pending_refill = copy(refill.at[:, pl.ds((k + 1) * CHUNK, CHUNK)], destination, refill_sem)
                        pending_refill.start()
                elif k > 0 and k * CHUNK % width == 0 and refill is not None:
                    item = copy(refill.at[:, pl.ds(k * CHUNK, width)], weight_ref, refill_sem)
                    item.start()
                    item.wait()
                partial = _dot(lhs_bf16, current[:, pl.ds((k * CHUNK) % width, CHUNK)])  # [T, 896] FP32
                stage[slot, k] = pack(_bf16(partial))
                if k == 0:
                    own = copy(stage.at[slot, 0], local_recv.at[slot, lane], SEM_OWN)
                    own.start()
                else:
                    item = remote(stage.at[slot, k], local_recv.at[slot, lane], slot * SEMS_PER_SLOT + SEM_RS + k, rank ^ k)
                    item.start()
                    copies.append(item)
            return own, copies

        def all_reduce_finish(handles, slot):
            own, copies = handles
            exchange_wait(copies)
            own.wait()
            chunk = sum(unpack(local_recv[slot, k], T, CHUNK) for k in range(8))
            return host_reduce_gather(chunk, slot)

        def tiny_exchange(row_i32, slot):
            """One [R, 128] int32 row set from every rank -> [32, R, 128] int32 (axis 0 = rank)."""
            tiny_recv[slot, rank] = row_i32
            exchange(lambda o: tiny_recv.at[slot, rank], lambda o: tiny_recv.at[slot, rank], range(1, 32), SEM_TINY, slot)
            return tiny_recv[slot]

        # ---------------- the ring slice stream ----------------
        # The kernel visits (layer, request) ring slices in visit order
        # (v = layer * R + request): each visit copies its [window, 128] slice
        # into one of the two buffers (parity), applies the request's new
        # context rows, attends, and writes the slice back; the in-copy of a
        # visit follows the write-back two visits earlier (same buffer).
        ring: dict[str, Any] = {"out": [None, None], "in": [None, None], "done": 0}

        def ring_start_in(v):
            b = v % 2
            if ring["out"][b] is not None:
                ring["out"][b].wait()
                ring["out"][b] = None
            item = copy(cache_in.at[v % R, v // R], ring_vmem.at[b], SEM_RING + b)
            item.start()
            ring["in"][b] = item

        def ring_drain():
            """Every visit's slice written back; visits never loaded are copied through."""
            for b in (0, 1):
                if ring["in"][b] is not None:
                    ring["in"][b].wait()
                    ring["in"][b] = None
                if ring["out"][b] is not None:
                    ring["out"][b].wait()
                    ring["out"][b] = None
            for v in range(ring["done"], layers * R):
                b = v % 2
                item = copy(cache_in.at[v % R, v // R], ring_vmem.at[b], SEM_RING + b)
                item.start()
                item.wait()
                item = copy(ring_vmem.at[b], cache_out.at[v % R, v // R], SEM_RING + 2 + b)
                item.start()
                item.wait()

        # ---------------- prologue: barrier and DMA issue ----------------
        collectives32.barrier()
        small_sources = {
            "aux": aux_hbm, "positions": positions_hbm, "previous": previous_hbm,
            **{name: weights[name] for name in VMEM_NAMES},
        }
        small_copies = [
            tpu.make_async_copy(small_sources[name], small[name], small_sems.at[index])
            for index, name in enumerate(SMALL_NAMES)
            if name != "aux" or not stream_aux
        ]
        for item in small_copies:
            item.start()
        # The verify logits land in the LM head's logits buffer (dead by then).
        logits_copy = copy(logits_hbm, logits_vmem, 0)
        logits_copy.start()
        # This rank's [FC_IN, 896] fc columns, into 896 of a_buf's 1024 lanes.
        fc_copy = copy(weights["fc"], a_buf.at[:, pl.ds(0, CHUNK)], SEM_FC)
        fc_copy.start()
        ring_start_in(0)
        ring_start_in(1)

        # ---------------- verify argmax, acceptance, anchor ----------------
        for item in small_copies:
            item.wait()
        logits_copy.wait()
        verify_logits = logits_vmem[...]  # [R, 8, 5120] FP32 (this rank's vocabulary shard)
        vocab_iota = jax.lax.broadcasted_iota(jnp.int32, (R, rows, vocab), 2)
        local_best = jnp.max(verify_logits, axis=2, keepdims=True)  # [R, 8, 1]
        local_id = jnp.min(jnp.where(verify_logits == local_best, vocab_iota, vocab), axis=2, keepdims=True)
        local_second = jnp.max(jnp.where(vocab_iota == local_id, NEG, verify_logits), axis=2, keepdims=True)
        out_lane = jax.lax.broadcasted_iota(jnp.int32, (R, rows, 128), 2)
        packet = jnp.where(
            out_lane == 0, jax.lax.bitcast_convert_type(local_best, jnp.int32),
            jnp.where(out_lane == 1, local_id + rank * vocab,
                      jnp.where(out_lane == 2, jax.lax.bitcast_convert_type(local_second, jnp.int32), 0)),
        )  # [R, 8, 128] int32
        argmax_recv[rank] = packet.reshape(T, 128)
        exchange(lambda o: argmax_recv.at[rank], lambda o: argmax_recv.at[rank], range(1, 32), SEM_TINY, 1)
        gathered = argmax_recv[...].reshape(RANKS, R, rows, 128)  # [32, R, 8, 128] int32
        rank_iota4 = jax.lax.broadcasted_iota(jnp.int32, (RANKS, R, rows, 1), 0)
        bests = jax.lax.bitcast_convert_type(gathered[:, :, :, 0:1], jnp.float32)  # [32, R, 8, 1]
        seconds = jax.lax.bitcast_convert_type(gathered[:, :, :, 2:3], jnp.float32)
        best = jnp.max(bests, axis=0)  # [R, 8, 1]
        winner = jnp.min(jnp.where(bests == best[None], rank_iota4, RANKS), axis=0)  # [R, 8, 1], lowest rank on ties
        targets = jnp.sum(jnp.where(rank_iota4 == winner[None], gathered[:, :, :, 1:2], 0), axis=0)  # [R, 8, 1]
        runner_up = jnp.maximum(
            jnp.max(seconds, axis=0), jnp.max(jnp.where(rank_iota4 == winner[None], NEG, bests), axis=0)
        )
        margins = best - runner_up  # [R, 8, 1] FP32
        previous_vector = small["previous"][...]  # [R, 128] int32: last step's output vectors
        row_iota_col = jax.lax.broadcasted_iota(jnp.int32, (R, rows, 1), 1)
        proposals = jnp.sum(
            jnp.where(jax.lax.broadcasted_iota(jnp.int32, (R, rows, 128), 2) == row_iota_col + OUT_DRAFTS,
                      jnp.broadcast_to(previous_vector[:, None, :], (R, rows, 128)), 0),
            axis=2, keepdims=True,
        )  # [R, 8, 1]: proposal k was verify row k + 1
        drafts_count = block - 1
        # matches[j, k] = proposals[j, k] == targets[j, k] for k < drafts_count
        match_rows = (row_iota_col < drafts_count) & (proposals == targets)  # no select on i1 vectors
        first_miss = jnp.min(jnp.where(match_rows | (row_iota_col >= drafts_count), block, row_iota_col), axis=1)
        first_miss = first_miss[:, 0]  # [R]
        count = jnp.where(forced_count >= 0, forced_count, jnp.minimum(first_miss, drafts_count))  # [R]
        anchor_token = jnp.sum(jnp.where(row_iota_col[:, :, 0] == count[:, None], targets[:, :, 0], 0), axis=1)  # [R]
        anchor_token = jnp.where(forced_anchor >= 0, forced_anchor, anchor_token)
        anchor_position = new_position + count + 1  # [R]
        result_vector = jnp.zeros((R, 128), jnp.int32) - 1
        result_lane = jax.lax.broadcasted_iota(jnp.int32, (R, 128), 1)
        field_row = jax.lax.broadcasted_iota(jnp.int32, (R, rows, 128), 1)
        field_lane = jax.lax.broadcasted_iota(jnp.int32, (R, rows, 128), 2)
        targets_row = jnp.sum(jnp.where(field_lane == field_row + OUT_TARGETS, targets, 0), axis=1)  # [R, 128]
        margins_row = jnp.sum(
            jnp.where(field_lane == field_row + OUT_MARGINS, jax.lax.bitcast_convert_type(margins, jnp.int32), 0),
            axis=1,
        )
        result_vector = jnp.where((result_lane >= OUT_TARGETS) & (result_lane < OUT_TARGETS + rows), targets_row, result_vector)
        result_vector = jnp.where((result_lane >= OUT_MARGINS) & (result_lane < OUT_MARGINS + rows), margins_row, result_vector)
        result_vector = jnp.where(result_lane == OUT_COUNT, count[:, None], result_vector)
        result_vector = jnp.where(result_lane == OUT_BONUS, anchor_token[:, None], result_vector)
        if rows < ROWS:
            padding = ((result_lane >= OUT_TARGETS + rows) & (result_lane < OUT_TARGETS + ROWS)) | (
                (result_lane >= OUT_MARGINS + rows) & (result_lane < OUT_MARGINS + ROWS)
            )
            result_vector = jnp.where(padding, 0, result_vector)
        tokens_v[...] = result_vector
        w1_copies = [[None] * R, [None] * R]

        def start_w1(tokens, slot):
            """The aligned 16-row w1 block of each request's previous token."""
            for j in range(R):
                start = pl.multiple_of((tokens[j] // 16) * 16, 16)
                item = copy(weights["w1"].at[pl.ds(start, 16)], w1_blk.at[slot, j], 10 + (4 + slot) * R + j)
                item.start()
                w1_copies[slot][j] = item

        start_w1(anchor_token, 0)
        rope_copies = []
        for j in range(R):
            rope_q_start = pl.multiple_of((anchor_position[j] // 8) * 8, 8)
            rope_kv_start = pl.multiple_of((new_position[j] // 8) * 8, 8)
            rope_copies += (
                copy(weights["rope_q"].at[pl.ds(rope_q_start, 16)], rope_q_blk.at[j], 10 + R + j),
                copy(weights["rope_kv"].at[pl.ds(rope_q_start, 16)], rope_kv_blk.at[0, j], 10 + 2 * R + j),
                copy(weights["rope_kv"].at[pl.ds(rope_kv_start, 16)], rope_kv_blk.at[1, j], 10 + 3 * R + j),
            )
        for item in rope_copies:
            item.start()
        layer_buffers = {"q": q_buf, "kv": kv_buf, "o": o_buf, "gate": gate_buf, "up": up_buf, "down": down_buf}
        layer_sems = {"q": 2, "kv": 3, "o": 4, "gate": 5, "up": 6, "down": 7}

        def layer_copy(name, layer):
            if name in ("down", "o"):
                # R > 4 streams one projection chunk to stay within VMEM;
                # smaller batches retain the full weight to avoid extra DMAs.
                return copy(weights[name].at[layer, :, pl.ds(0, layer_buffers[name].shape[1])], layer_buffers[name], layer_sems[name])
            return copy(weights[name].at[layer], layer_buffers[name], layer_sems[name])

        # Anchor embeddings: each request's owner fetches its aligned 16-row
        # block, picks the row and sends it (packed) to the 31 peers on
        # semaphore SEM_BCAST + j.
        owner = anchor_token // vocab  # [R]

        for j in range(R):
            @pl.when(rank == owner[j])
            def send_anchor_row(j=j):
                local_id = anchor_token[j] - owner[j] * vocab
                embed_start = pl.multiple_of((local_id // 16) * 16, 16)
                item = copy(weights["embed"].at[pl.ds(embed_start, 16)], embed_blk, 10 + j)
                item.start()
                item.wait()
                row = _select_rows(_f32(embed_blk[...]), local_id % 16, 1)  # [1, 7168]
                bcast_buf[j] = pack(_bf16(row))
                copies = [remote(bcast_buf.at[j], bcast_buf.at[j], SEM_BCAST + j, rank ^ offset) for offset in range(1, 32)]
                for item in copies:
                    item.start()
                for item in copies:
                    item.wait_send()

        def receive_anchor_rows():
            for j in range(R):
                @pl.when(rank != owner[j])
                def receive_anchor_row(j=j):
                    remote(bcast_buf.at[j], bcast_buf.at[j], SEM_BCAST + j, rank).wait_recv()

        def w1_drain(slot):
            return [item for item in w1_copies[slot] if item is not None]

        # ---------------- fc: context features for the new rows ----------------
        fc_copy.wait()
        # Layer-0 weights and the Markov pieces stream in behind the fc phase.
        pending = {name: layer_copy(name, 0) for name in layer_buffers}
        for name in ("kv", "q", "o", "gate", "up", "down"):
            pending[name].start()
        w2t_copy = copy(weights["w2t"], w2t_buf, SEM_W2T)
        w2t_copy.start()

        aux_ref = small["aux"]  # [aux layers, T, hidden]
        spans = []
        for h in range(4):
            start, stop = h * FC_IN, (h + 1) * FC_IN
            pieces = []
            for layer_index in range(len(c.aux_layers)):
                lo, hi = layer_index * hidden, (layer_index + 1) * hidden
                a, b = max(start, lo), min(stop, hi)
                if a < b:
                    pieces.append((layer_index, a - lo, b - lo, a - start))
            spans.append(pieces)

        def fc_round(col_start, cols, fc_view):
            """One round of the fc product: this host's quarter of the aux
            features against the resident weight columns, into
            ``fc_part[:, col_start: col_start + cols]``."""
            for h in range(4):
                @pl.when(host == h)
                def fc_branch(h=h, col_start=col_start, cols=cols, fc_view=fc_view):
                    for j in range(R):
                        if stream_aux:
                            # Aux features are used only by this request's FC
                            # product: stage one request's rows at a time.
                            item = tpu.make_async_copy(
                                aux_hbm.at[:, pl.ds(j * rows, rows), :],
                                aux_ref, small_sems.at[0],
                            )
                            item.start()
                            item.wait()
                        row_start = 0 if stream_aux else j * rows
                        total = None
                        for layer_index, lo, hi, offset in spans[h]:
                            part = _dot(
                                aux_ref[layer_index, row_start:row_start + rows, lo:hi],
                                fc_view[pl.ds(offset, hi - lo), :],
                            )
                            total = part if total is None else total + part
                        fc_part.at[pl.ds(j * rows, rows), pl.ds(col_start, cols)][...] = total

        fc_round(0, CHUNK, a_buf.at[:, pl.ds(0, CHUNK)])
        context = _rms_norm(_bf16(host_reduce_gather(fc_part[...], 0)), small["hidden_norm"][...], eps)  # [T, 7168]

        # ---------------- positions, masks, RoPE ----------------
        positions = small["positions"][...]  # [R, window] int32
        slot_lane = jax.lax.broadcasted_iota(jnp.int32, (R, window), 1)
        new_slots = [[(new_position[j] + k) % window for k in range(block)] for j in range(R)]
        for k in range(block):
            positions_k = new_position + k  # [R]
            positions = jnp.where(slot_lane == (positions_k % window)[:, None], positions_k[:, None], positions)
        small["positions"][...] = positions
        positions_copy = tpu.make_async_copy(small["positions"], positions_out, small_sems.at[1])
        positions_copy.start()
        context_allowed = (
            (positions >= 0)
            & (positions < anchor_position[:, None])
            & (positions >= anchor_position[:, None] - window)
        )  # [R, window]
        row_iota = jax.lax.broadcasted_iota(jnp.int32, (rows, 16), 0)
        col_iota = jax.lax.broadcasted_iota(jnp.int32, (rows, 16), 1)
        block_allowed = (col_iota <= row_iota) & (col_iota < block)

        for item in rope_copies:
            item.wait()
        rope_q_rows = jnp.stack([_select_rows(rope_q_blk[j], anchor_position[j] % 8, rows) for j in range(R)])  # [R, 8, 384]
        cos_q = _f32(_bf16(rope_q_rows[:, :, :q_width])).reshape(T, q_width)
        sin_q = _f32(_bf16(rope_q_rows[:, :, q_width:])).reshape(T, q_width)
        rope_kv_rows = jnp.stack([_select_rows(rope_kv_blk[0, j], anchor_position[j] % 8, rows) for j in range(R)])  # [R, 8, 256]
        cos_kv = _f32(_bf16(rope_kv_rows[:, :, :kv_width])).reshape(T, kv_width)
        sin_kv = _f32(_bf16(rope_kv_rows[:, :, kv_width:])).reshape(T, kv_width)
        rope_new_rows = jnp.stack([_select_rows(rope_kv_blk[1, j], new_position[j] % 8, rows) for j in range(R)])
        cos_new = _f32(_bf16(rope_new_rows[:, :, :kv_width])).reshape(T, kv_width)
        sin_new = _f32(_bf16(rope_new_rows[:, :, kv_width:])).reshape(T, kv_width)

        # ---------------- block input rows ----------------
        receive_anchor_rows()

        anchor_rows = jnp.concatenate([unpack(bcast_buf[j], 1, hidden) for j in range(R)], axis=0)  # [R, 7168] FP32 (BF16-valued)
        mask_row = _f32(small["mask_embed"][...])  # [1, 7168]
        x_row = jax.lax.broadcasted_iota(jnp.int32, (R, rows, hidden), 1)
        x = _bf16(jnp.where(x_row == 0, anchor_rows[:, None, :], mask_row[:, None, :]))  # [R, 8, 7168]
        x = x.reshape(T, hidden)

        def kv_rows(features_bf16, layer, cos32, sin32):
            """Packed [T, 128] K (normed, RoPE) | V rows for all requests' feature rows."""
            kv = _bf16(_dot(features_bf16, kv_buf[...]))  # reference linear: BF16 output
            sumsq = _grouped_sumsq(_f32(kv), small["kv_ones"][...])  # K lanes only
            lane_index = jax.lax.broadcasted_iota(jnp.int32, kv.shape, 1)
            normed = _f32(kv) * jax.lax.rsqrt(sumsq * (1.0 / c.head_dim) + eps)
            k_scaled = _bf16(_f32(_bf16(normed)) * _f32(small["kvnorm"][layer]))
            kv = jnp.where(lane_index < c.head_dim, k_scaled, kv)
            return _rope(kv, small["rot_kv"][...], cos32, sin32)  # sin is 0 on V lanes

        def write_context_rows(j, b):
            """Request j's new rows into the resident ring slice (buffer ``b``):
            two aligned 128-row slabs cover the consecutive slots even across
            the wrap."""
            for slab_start in (new_slots[j][0] // 128 * 128, new_slots[j][block - 1] // 128 * 128):
                slab_start = pl.multiple_of(slab_start, 128)
                slab_ref = ring_vmem.at[b, pl.ds(slab_start, 128)]
                slab = slab_ref[...]
                slab_rows = jax.lax.broadcasted_iota(jnp.int32, slab.shape, 0) + slab_start
                for k in range(block):
                    slab = jnp.where(slab_rows == new_slots[j][k], newkv_buf[j * rows + k : j * rows + k + 1, :], slab)
                slab_ref[...] = slab

        # Slot-0 Markov bias needs only the anchors' w1 rows: computed during a
        # collective wait below and consumed by the sampler.
        slot0_bias = [None]

        def w1_rows_for(tokens, slot):
            return jnp.concatenate(
                [_bf16(_select_rows(_f32(w1_blk[slot, j]), tokens[j] % 16, 1)) for j in range(R)], axis=0
            )  # [R, 256]

        def compute_slot0_bias():
            for item in w1_drain(0):
                item.wait()
            w2t_copy.wait()
            slot0_bias[0] = _dot(w1_rows_for(anchor_token, 0), w2t_buf[...])  # [R, 5120] FP32

        # Layer 0's context rows; later layers' are computed inside the
        # previous layer's down-projection reduction. Each ring visit applies
        # its request's rows as the slice lands.
        pending["kv"].wait()
        pending["kv"] = None
        newkv_buf[...] = kv_rows(context, 0, cos_new, sin_new)

        # ---------------- layers ----------------
        for layer in range(layers):
            h = _rms_norm(x, small["ln1"][layer], eps)
            block_kv = kv_rows(h, layer, cos_kv, sin_kv)  # the block's own K/V (kv weights of this layer)
            if layer + 1 < layers:
                pending["kv"] = layer_copy("kv", layer + 1)
                pending["kv"].start()

            pending["q"].wait()
            q = _bf16(_dot(h, q_buf[...]))  # [T, 192]
            if layer + 1 < layers:
                pending["q"] = layer_copy("q", layer + 1)
                pending["q"].start()
            sumsq = _grouped_sumsq(_f32(q), small["head_ones"][...])
            normed = _f32(q) * jax.lax.rsqrt(sumsq * (1.0 / c.head_dim) + eps)
            q = _bf16(_f32(_bf16(normed)) * _f32(small["qnorm"][layer]))
            q = _rope(q, small["rot_q"][...], cos_q, sin_q)

            block_tile = jnp.pad(block_kv.reshape(R, rows, kv_width), ((0, 0), (0, 16 - rows), (0, 0)))  # [R, 16, 128]
            # One ring visit per request: its slice in, new rows, attention,
            # slice back out; the next visit on the buffer streams in behind.
            attended_rows = []
            for j in range(R):
                v = layer * R + j
                b = v % 2
                ring["in"][b].wait()
                ring["in"][b] = None
                write_context_rows(j, b)
                ring_tile = ring_vmem[b]  # [window, 128] bf16
                q_rows = q[j * rows:(j + 1) * rows]
                attended_j = jnp.zeros((rows, q_width), jnp.float32)
                for head in range(3):
                    q_head = _bf16(_dot(q_rows, small["head_select"][head]))  # [8, 128], zeros on V lanes
                    s_ctx = _attention_scores(q_head, ring_tile) * scale  # [8, window]
                    s_blk = _attention_scores(q_head, block_tile[j]) * scale  # [8, 16]
                    s_ctx = jnp.where(context_allowed[j], s_ctx, NEG)
                    s_blk = jnp.where(block_allowed, s_blk, NEG)
                    maximum = jnp.maximum(jnp.max(s_ctx, axis=1, keepdims=True), jnp.max(s_blk, axis=1, keepdims=True))
                    p_ctx = jnp.exp(s_ctx - maximum)
                    p_blk = jnp.exp(s_blk - maximum)
                    denominator = jnp.sum(p_ctx, axis=1, keepdims=True) + jnp.sum(p_blk, axis=1, keepdims=True)
                    inverse = 1.0 / denominator
                    out = _dot(_bf16(p_ctx * inverse), ring_tile) + _dot(_bf16(p_blk * inverse), block_tile[j])
                    attended_j = attended_j + _dot(_bf16(out), small["head_place"][head])  # V lanes -> head lanes
                attended_rows.append(attended_j)
                item = copy(ring_vmem.at[b], cache_out.at[j, layer], SEM_RING + 2 + b)
                item.start()
                ring["out"][b] = item
                ring["done"] += 1
                if v + 2 < layers * R:
                    ring_start_in(v + 2)
            attended = _bf16(jnp.concatenate(attended_rows, axis=0))

            pending["o"].wait()
            handles = all_reduce_start(attended, o_buf, 1, refill=weights["o"].at[layer], refill_sem=layer_sems["o"])  # [T, 7168] partial, chunked sends
            if layer + 1 < layers:
                pending["o"] = layer_copy("o", layer + 1)
                pending["o"].start()
            if layer == 0:
                compute_slot0_bias()  # overlaps the reduction's local phase
            x = _bf16(_f32(x) + all_reduce_finish(handles, 1))

            h2 = _rms_norm(x, small["ln2"][layer], eps)
            pending["gate"].wait()
            gate = _bf16(_dot(h2, gate_buf[...]))  # [T, 512], zero beyond 448
            pending["up"].wait()
            up = _bf16(_dot(h2, up_buf[...]))
            if layer + 1 < layers:
                pending["gate"] = layer_copy("gate", layer + 1)
                pending["gate"].start()
                pending["up"] = layer_copy("up", layer + 1)
                pending["up"].start()
            gate32 = _f32(gate)
            activation = _bf16(_f32(_bf16(gate32 * jax.nn.sigmoid(gate32))) * _f32(up))
            pending["down"].wait()
            handles = all_reduce_start(activation, down_buf, 0, refill=weights["down"].at[layer])
            if layer + 1 < layers:
                pending["down"] = layer_copy("down", layer + 1)
                pending["down"].start()
                # The next layer's context rows depend only on the fc features.
                pending["kv"].wait()
                pending["kv"] = None  # consumed; the next layer's block rows reuse the buffer
                newkv_buf[...] = kv_rows(context, layer + 1, cos_new, sin_new)
            x = _bf16(_f32(x) + all_reduce_finish(handles, 0))

        # ---------------- LM head ----------------
        hidden_rows = _rms_norm(x, small["final_norm"][...], eps)  # [T, 7168]
        chunk_buffers = (
            a_buf.at[pl.ds(0, hidden), pl.ds(0, LM_CHUNK)],
            gate_buf,
            up_buf,
        )
        chunk_sems = (0, 5, 6)

        def chunk_copy(chunk):
            return copy(weights["lm_head"].at[chunk], chunk_buffers[chunk % 3], chunk_sems[chunk % 3])

        chunk_copies = [chunk_copy(chunk) for chunk in range(3)]
        for item in chunk_copies:
            item.start()
        vocab_lane = jax.lax.broadcasted_iota(jnp.int32, (1, vocab), 1)
        for chunk in range(LM_CHUNKS):
            chunk_copies[chunk % 3].wait()
            logits = _f32(_bf16(_dot(hidden_rows, chunk_buffers[chunk % 3][...])))  # BF16-rounded like the reference
            if chunk + 3 < LM_CHUNKS:
                chunk_copies[chunk % 3] = chunk_copy(chunk + 3)
                chunk_copies[chunk % 3].start()
            logits_vmem[:, :, pl.ds(chunk * LM_CHUNK, LM_CHUNK)] = logits.reshape(R, rows, LM_CHUNK)

        # ---------------- exact Markov sampling ----------------
        tokens = tokens_v[...]  # [R, 128]
        token_lane = jax.lax.broadcasted_iota(jnp.int32, (R, 128), 1)
        rank_iota = jax.lax.broadcasted_iota(jnp.int32, (RANKS, R, 128), 0)
        previous = anchor_token  # [R]
        for slot in range(block):
            if slot == 0:
                bias = slot0_bias[0]
            else:
                for item in w1_drain(slot % 2):
                    item.wait()
                bias = _dot(w1_rows_for(previous, slot % 2), w2t_buf[...])  # [R, 5120] FP32
            scores = jnp.concatenate(
                [logits_vmem[j, slot: slot + 1, :] for j in range(R)], axis=0
            ) + bias  # [R, 5120]
            best_value = jnp.max(scores, axis=1, keepdims=True)  # [R, 1]
            best_id = jnp.min(jnp.where(scores == best_value, vocab_lane, vocab), axis=1, keepdims=True)
            local_max_bits = jax.lax.bitcast_convert_type(best_value, jnp.int32)  # [R, 1]
            global_id = best_id + rank * vocab
            row = jnp.where(token_lane == 0, local_max_bits, jnp.where(token_lane == 1, global_id, 0))  # [R, 128]
            gathered = tiny_exchange(row, slot % 2)  # [32, R, 128] int32
            maxima = jax.lax.bitcast_convert_type(gathered[:, :, 0:1], jnp.float32)  # [32, R, 1]
            best = jnp.max(maxima, axis=0, keepdims=True)  # [1, R, 1]
            winner = jnp.min(jnp.where(maxima == best, rank_iota[:, :, 0:1], RANKS), axis=0, keepdims=True)
            token_row = jnp.sum(jnp.where(rank_iota == winner, gathered, 0), axis=0)  # [R, 128]; lane 1 = id
            token = token_row[:, 1:2]  # [R, 1]
            tokens = jnp.where(token_lane == OUT_DRAFTS + slot, token, tokens)
            previous = token[:, 0]  # [R]
            if slot + 1 < block:
                start_w1(previous, (slot + 1) % 2)
        tokens_v[...] = tokens

        # ---------------- outputs ----------------
        tokens_copy = tpu.make_async_copy(tokens_v, tokens_out, small_sems.at[0])
        tokens_copy.start()
        ring_drain()
        tokens_copy.wait()
        positions_copy.wait()

    return body


def make_fused_draft(c: DraftConfig, mesh, block: int, *, eps: float | None = None, requests: int = 1):
    """``(weights, cache, positions, aux, logits, previous, positions0, forced_counts[, forced_anchors])`` ->
    ``(vectors [R, 128] int32, cache, positions)``.

    One call covers ``requests`` sequences: their rows share one weight stream
    while each keeps its own ring, positions, anchor and Markov chain. With one
    request ``cache`` has no request axis (``[1, layers, window, 128]`` per
    rank) and the other operands no leading one.

    The kernel consumes the verify step's outputs: ``logits [32, R, 8, vocab /
    32]`` (vocabulary sharded) give the per-row targets and top-2 margins
    through one 31-peer exchange; ``previous`` holds the last step's output
    vectors (its drafts were verify rows 1..), ``positions0`` the verify step's
    first-row positions. The count is the leading drafts equal to their targets
    unless ``forced_counts >= 0`` (prefill, or sampling outside the kernel); the
    bonus token (the next anchor) is the target at ``count`` unless
    ``forced_anchors >= 0`` (sampled outside the kernel). Output lanes: ``OUT_*``.
    """
    from jax.sharding import PartitionSpec as P

    rows = _fused_rows(requests, block)
    if mesh.size != RANKS:
        raise ValueError("the fused draft needs the 32-rank mesh")
    eps = c.eps if eps is None else eps
    window, hidden = c.window, c.hidden
    R, T = requests, requests * rows
    packed = T * CHUNK // 256  # FP32 rows holding one BF16 [T, 896] chunk
    body = _make_body(c, block, eps, requests)

    def local(weights, cache, positions, aux, logits, previous, position, forced_count, forced_anchor):
        weights = {name: weights[name][0] for name in WEIGHT_NAMES}
        if R == 1:
            # The single-request operands carry no request axis.
            cache = cache[0][None]
            logits = logits[0][None].astype(jnp.float32)
        else:
            cache = cache[0]
            positions = positions[0]
            logits = logits[0].astype(jnp.float32)
            previous = previous[0]
        if rows < ROWS:
            aux = aux.reshape(aux.shape[0], R, ROWS, aux.shape[-1])[:, :, :rows, :]
            aux = aux.reshape(aux.shape[0], T, aux.shape[-1])
            logits = logits[:, :rows, :]
        # [R, layers, window, 128], [R, window], [R, rows, vocab], [R, 128]
        scalars = jnp.stack((position, forced_count, forced_anchor)).reshape(3 * R).astype(jnp.int32)  # [3R]
        small_shapes = {"aux": aux, "positions": positions, "logits": logits, "previous": previous, **weights}
        if R >= 8:
            small_shapes["aux"] = jax.ShapeDtypeStruct(
                (aux.shape[0], rows, aux.shape[-1]), aux.dtype
            )
        in_specs = [pl.BlockSpec(memory_space=tpu.HBM) for _ in range(4 + len(WEIGHT_NAMES) + 1)]
        out_specs = [pl.BlockSpec(memory_space=tpu.HBM) for _ in range(3)]
        scratch = (
            *(tpu.VMEM(small_shapes[name].shape, small_shapes[name].dtype) for name in SMALL_NAMES),
            tpu.VMEM((FC_IN, 1024), jnp.bfloat16),  # a_buf: fc columns in 896 of 1024 lanes, then projection/LM scratch
            tpu.VMEM((hidden, 3 * c.head_dim), jnp.bfloat16),  # q_buf
            tpu.VMEM((hidden, 2 * c.head_dim), jnp.bfloat16),  # kv_buf
            tpu.VMEM((3 * c.head_dim, CHUNK if R > 4 else hidden), jnp.bfloat16),  # o_buf
            tpu.VMEM((hidden, LM_CHUNK), jnp.bfloat16),  # gate_buf: MLP lanes 0..447, or an LM chunk
            tpu.VMEM((hidden, LM_CHUNK), jnp.bfloat16),  # up_buf
            tpu.VMEM((MLP_PADDED, CHUNK if R > 4 else hidden), jnp.bfloat16),  # down_buf
            tpu.VMEM((c.markov_rank, c.vocab // RANKS), jnp.bfloat16),  # w2t_buf [256, 5120]
            tpu.VMEM((2, R, 16, c.markov_rank), jnp.bfloat16),  # w1_blk
            tpu.VMEM((16, hidden), jnp.bfloat16),  # embed_blk
            tpu.VMEM((R, ROW_PACKED, 128), jnp.float32),  # bcast_buf
            tpu.VMEM((R, 16, 6 * c.head_dim), jnp.float32),  # rope_q_blk
            tpu.VMEM((2, R, 16, 4 * c.head_dim), jnp.float32),  # rope_kv_blk
            tpu.VMEM((2, window, 2 * c.head_dim), jnp.bfloat16),  # ring_vmem: the ring slice stream, double-buffered
            tpu.VMEM((R, rows, c.vocab // RANKS), jnp.float32),  # logits_vmem: the verify logits, then the draft's
            tpu.VMEM((T, CHUNK), jnp.float32),  # fc_part
            tpu.VMEM((R, 128), jnp.int32),  # tokens_v
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # stage
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # local_recv
            tpu.VMEM((2, 4, packed, 128), jnp.float32),  # host_packed
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # gather_recv
            tpu.VMEM((2, RANKS, R, 128), jnp.int32),  # tiny_recv
            tpu.VMEM((RANKS, T, 128), jnp.int32),  # argmax_recv
            tpu.VMEM((T, 2 * c.head_dim), jnp.bfloat16),  # newkv_buf: the next layer's context rows
            tpu.SemaphoreType.DMA((10 + 6 * R + 4,)),
            tpu.SemaphoreType.DMA((len(SMALL_NAMES),)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
        )
        cache_index = 1 + 4 + len(WEIGHT_NAMES)  # scalars, aux, positions, logits, previous, weights
        tokens, positions_out, cache_out = pl.pallas_call(
            body,
            name="dspark_fused_draft",
            out_shape=(
                jax.ShapeDtypeStruct((R, 128), jnp.int32),
                jax.ShapeDtypeStruct((R, window), jnp.int32),
                jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            ),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1, in_specs=in_specs, out_specs=out_specs, scratch_shapes=scratch,
            ),
            input_output_aliases={cache_index: 2},
            compiler_params=tpu.CompilerParams(collective_id=25, vmem_limit_bytes=64 * 1024**2),
        )(scalars, aux, positions, logits, previous, *[weights[name] for name in WEIGHT_NAMES], cache)
        if R == 1:
            return tokens, cache_out[0][None], positions_out
        return tokens, cache_out[None], positions_out[None]

    program = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P(), P(), P("tp"), P(), P(), P(), P()),
            out_specs=(P(), P("tp"), P()),
            check_vma=False,
        ),
        donate_argnums=(1,),
    )

    def fused_draft(weights, cache, positions, aux, logits, previous, position, forced_count, forced_anchor=None):
        if forced_anchor is None:
            forced_anchor = jnp.full_like(position, -1)
        return program(weights, cache, positions, aux, logits, previous, position, forced_count, forced_anchor)

    return fused_draft


# --------------------------------------------------------------------------
# The ring from a prompt (prefill side)
#
# Each sequence-parallel prefill rank projects its own window rows for every KV
# head (replicated weights, :func:`ring_weights_from_shards`); one all-to-all
# hands rank r the rows of KV head ``r // (32 / kv_heads)``, stored at ring slot
# ``position % window`` in the fused kernel's packed K | V layout. A ring travels
# with the KDA state in ``RING_BLOCK_ROWS`` spare rows of each of a request's 69
# state blocks, behind a header (``RING_MAGIC``, length L, first position F); it
# holds position p at slot p % window for max(L - window, F) <= p < L.

RING_MAGIC = 0x44535231  # "DSR1"
RING_BLOCK_ROWS = 15  # spare state-block rows per block: 69 * 15 rows hold a ring and its header
RING_ROW_LANES = 1280  # bf16 per 2560-byte pool row


def _header_bytes(values):
    """int32 ``[..., k]`` -> bf16 ``[..., 4 k]``: each byte as an exact small
    number. (Bit-cast halves would include bf16 subnormals, which the TPU
    flushes to zero when it moves them.)"""
    shifts = jnp.asarray([0, 8, 16, 24], jnp.int32)
    parts = jnp.right_shift(values[..., None], shifts) & 0xFF
    return parts.astype(jnp.bfloat16).reshape(*values.shape[:-1], -1)


def _header_ints(parts):
    """Inverse of :func:`_header_bytes` on ``[..., k, 4]``."""
    shifts = jnp.asarray([0, 8, 16, 24], jnp.int32)
    return jnp.sum(jnp.left_shift(parts.astype(jnp.int32) & 0xFF, shifts), axis=-1)


def ring_rows(c: DraftConfig) -> int:
    """Pool rows a ring's K | V values take (``[layers, window, 2 head_dim]`` bf16)."""
    return -(-c.layers * c.window * 2 * c.head_dim // RING_ROW_LANES)


def pack_ring(ring, meta, blocks: int):
    """``ring`` ``[n, layers, window, 2 head_dim]`` bf16 and ``meta`` ``[n, 2]``
    int32 (length, first position) -> ``[n, blocks, RING_BLOCK_ROWS,
    RING_ROW_LANES]`` bf16: the ring's rows, then a header row, zero padded."""
    n, size = ring.shape[0], int(np.prod(ring.shape[1:]))
    flat = jnp.pad(ring.reshape(n, size), ((0, 0), (0, -size % RING_ROW_LANES)))
    rows = flat.reshape(n, -1, RING_ROW_LANES)
    header = jnp.concatenate((jnp.full((n, 1), RING_MAGIC, jnp.int32), meta.astype(jnp.int32)), axis=1)
    header = _header_bytes(header).reshape(n, 1, -1)
    header = jnp.pad(header, ((0, 0), (0, 0), (0, RING_ROW_LANES - header.shape[-1])))
    total = blocks * RING_BLOCK_ROWS
    if rows.shape[1] + 1 > total:
        raise ValueError(f"a ring needs {rows.shape[1] + 1} rows, {blocks} blocks hold {total}")
    packed = jnp.concatenate((rows, header), axis=1)
    packed = jnp.pad(packed, ((0, 0), (0, total - packed.shape[1]), (0, 0)))
    # The TPU compiler cannot fuse these pads with a reshape that splits the
    # padded dimensions (as the pool writes do): keep them apart.
    packed = jax.lax.optimization_barrier(packed)
    return packed.reshape(n, blocks, RING_BLOCK_ROWS, RING_ROW_LANES)


def unpack_ring(c: DraftConfig, packed, length, cap=None):
    """Inverse of :func:`pack_ring` for one ring read back from the pool:
    ``packed`` ``[blocks, RING_BLOCK_ROWS, RING_ROW_LANES]`` bf16 -> ``(ring
    [layers, window, 2 head_dim], meta [2])``. A ring whose header is not
    ``RING_MAGIC`` at ``length`` is empty: meta ``(length, length)``.
    ``cap``: raise the ring's first position to at least this (a prefix hit
    keeps what the ring held before it)."""
    rows = packed.reshape(-1, RING_ROW_LANES)
    count, shape = ring_rows(c), (c.layers, c.window, 2 * c.head_dim)
    ring = rows[:count].reshape(-1)[: int(np.prod(shape))].reshape(shape)
    header = _header_ints(rows[count, : 4 * 3].reshape(3, 4))
    valid = (header[0] == RING_MAGIC) & (header[1] == length)
    first = jnp.where(valid, header[2], length)
    if cap is not None:
        first = jnp.maximum(first, cap)
    return ring, jnp.stack((jnp.asarray(length, jnp.int32), first.astype(jnp.int32)))


def ring_positions(c: DraftConfig, meta):
    """The ring's slot positions ``[window]`` int32 (``-1`` empty) from its meta."""
    length, first = meta[0], meta[1]
    slot = jnp.arange(c.window, dtype=jnp.int32)
    position = length - 1 - jnp.mod(length - 1 - slot, c.window)
    return jnp.where((position >= 0) & (position >= first) & (position >= length - c.window), position, -1)


def ring_weights_from_shards(c: DraftConfig, fc, kv, kvnorm, hidden_norm, axis: str):
    """(inside a TP32 ``shard_map`` over ``axis``) this rank's kernel shards
    (:func:`kernel_weight_shards`: ``fc`` ``[FC_IN, CHUNK]``, ``kv`` ``[layers,
    hidden, 2 head_dim]``, ``kvnorm`` ``[layers, 1, 2 head_dim]``,
    ``hidden_norm`` ``[1, hidden]``) -> the replicated weights the prefill
    ring update takes: ``fc`` ``[context_width, hidden]``, ``kv`` ``[layers,
    hidden, kv_heads, 2 head_dim]``, ``k_norm`` ``[layers, head_dim]``,
    ``hidden_norm`` ``[hidden]``."""
    hosts, lanes = c.context_width // fc.shape[0], c.hidden // fc.shape[1]
    full_fc = jax.lax.all_gather(fc, axis)  # [32, in / hosts, out / lanes], rank = host * lanes + lane
    full_fc = full_fc.reshape(hosts, lanes, *fc.shape).transpose(0, 2, 1, 3).reshape(c.context_width, c.hidden)
    share = RANKS // c.kv_heads
    heads = jax.lax.all_gather(kv, axis)[::share]  # [kv_heads, layers, hidden, 2 head_dim]
    return {
        "fc": full_fc,
        "kv": heads.transpose(1, 2, 0, 3),
        "k_norm": kvnorm[:, 0, : c.head_dim],
        "hidden_norm": hidden_norm[0],
    }


def prefill_ring_update(c: DraftConfig, weights, aux, positions, sequences, rings, ends, stops, axis: str):
    """(inside a TP32 ``shard_map`` over ``axis``) one packed window's rows into
    its sequences' rings.

    ``aux`` ``[len(aux_layers), rows, hidden]`` bf16 is this rank's window rows
    of the residual streams entering the aux layers; ``positions`` /
    ``sequences`` ``[32 * rows]`` int32 give every window row's position and
    sequence (``-1``: padding), the ranks' rows in rank-major order; ``rings`` ``[slots, layers, window, 2
    head_dim]`` this rank's head of each sequence's ring; ``ends`` / ``stops``
    ``[slots]`` each sequence's end in the window and its checkpoint (``-1``:
    none). Returns ``(rings after the window, rings at the checkpoints)``."""
    rows, hd = aux.shape[1], c.head_dim
    rank = jax.lax.axis_index(axis)
    features = aux.transpose(1, 0, 2).reshape(rows, c.context_width)
    context = jnp.einsum("ri,io->ro", features, weights["fc"], preferred_element_type=jnp.float32)
    context = rms_norm(context.astype(jnp.bfloat16), weights["hidden_norm"], c.eps)
    # RoPE on the K lanes of each packed K | V row: cos | 1 and sin | 0 over
    # the 2 head_dim lanes, rotate-half as a +-1 matrix (exact), as the fused
    # kernel's ``rope_kv`` tables and ``rot_kv`` do -- no lane concatenations
    # (the TPU compiler cannot split padded dims they leave behind).
    local = jnp.maximum(jax.lax.dynamic_slice_in_dim(positions, rank * rows, rows), 0)
    inv_freq = 1.0 / c.rope_theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
    lane_freq = np.concatenate((inv_freq, inv_freq, np.zeros(hd, np.float32)))
    k_lane = np.arange(2 * hd) < hd
    angle = local.astype(jnp.float32)[:, None] * jnp.asarray(lane_freq)[None, :]
    cos = jnp.where(k_lane, jnp.cos(angle), 1.0).astype(jnp.bfloat16)[:, None, :]
    sin = jnp.where(k_lane, jnp.sin(angle), 0.0).astype(jnp.bfloat16)[:, None, :]
    rotation = np.zeros((2 * hd, 2 * hd), np.float32)
    rotation[:hd, :hd] = _rotation(hd)
    rotation = jnp.asarray(rotation, jnp.bfloat16)
    k_lanes = jnp.asarray(k_lane)
    packed = []
    for layer in range(c.layers):
        kv = jnp.einsum("rh,hkd->rkd", context, weights["kv"][layer], preferred_element_type=jnp.float32)
        kv = kv.astype(jnp.bfloat16)  # [rows, kv_heads, 2 head_dim]
        k32 = jnp.where(k_lanes, kv.astype(jnp.float32), 0.0)
        normed = kv.astype(jnp.float32) * jax.lax.rsqrt(jnp.sum(k32 * k32, axis=-1, keepdims=True) / hd + c.eps)
        k_weight = jnp.pad(weights["k_norm"][layer], (0, hd))
        k_scaled = (normed.astype(jnp.bfloat16) * k_weight).astype(jnp.bfloat16)
        kv = jnp.where(k_lanes, k_scaled, kv)
        rotated = jnp.einsum("rkd,de->rke", kv, rotation, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        packed.append(kv * cos + rotated * sin)
    packed = jnp.stack(packed)  # [layers, rows, kv_heads, 2 head_dim]
    share = RANKS // c.kv_heads
    # Destination rank d gets KV head d // share: repeat each head, then
    # destinations lead for the all-to-all.
    send = jnp.broadcast_to(packed[:, :, :, None], (*packed.shape[:3], share, 2 * hd))
    send = send.reshape(c.layers, rows, RANKS, 2 * hd).transpose(2, 0, 1, 3)
    mine = jax.lax.all_to_all(send, axis, 0, 0)  # [32 sources, layers, rows, 2hd]
    mine = mine.transpose(1, 0, 2, 3).reshape(c.layers, RANKS * rows, 2 * hd)
    slot = jnp.mod(positions, c.window)

    def scatter(ring, keep):
        index = jnp.where(keep, slot, c.window)  # out of range: dropped
        return ring.at[:, index].set(mine, mode="drop")

    after, at = [], []
    for s in range(rings.shape[0]):
        member = sequences == s
        after.append(scatter(rings[s], member & (positions >= ends[s] - c.window)))
        at.append(scatter(rings[s], member & (positions < stops[s]) & (positions >= stops[s] - c.window)))
    return jnp.stack(after), jnp.stack(at)
