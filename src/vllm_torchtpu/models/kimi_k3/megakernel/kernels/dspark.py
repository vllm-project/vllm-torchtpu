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
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/dspark.py
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
from typing import TYPE_CHECKING, Any, Protocol

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec
from safetensors.numpy import load_file

from ..common import collectives32
from ..common.constants import (
    AXIS_TP,
    SUBLANE_ROWS,
    TP_SIZE,
)
from ..common.constants import (
    CHUNK as CHUNK,
)
from ..common.constants import (
    FC_IN as FC_IN,
)
from ..common.constants import (
    KERNEL_SHARD_FORMAT as KERNEL_SHARD_FORMAT,
)
from ..common.constants import (
    MLP_PADDED as MLP_PADDED,
)
from ..common.constants import (
    ROW_PACKED as ROW_PACKED,
)
from .decode_megakernel import attention_scores, dot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from jax._src.pallas.mosaic import primitives as pl_primitives

    class _IndexUpdateRefLike(Protocol):
        """Structural protocol for JAX ``Array.at[...]`` index updater."""

        def set(self, values: object, /) -> jax.Array:
            """Return updated array with ``values`` written at the indexed slice."""
            ...


type _AsyncCopy = pl_primitives.AsyncCopyDescriptor
type _Buffer = jax.Array | _IndexUpdateRefLike


@dataclasses.dataclass(frozen=True)
class DraftConfig:
    """Configuration for the Qwen3-style DSpark speculative draft model."""

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
        """Return the total query projection width across all heads."""
        return self.heads * self.head_dim

    @property
    def kv_width(self) -> int:
        """Return the single key or value projection width across KV heads."""
        return self.kv_heads * self.head_dim

    @property
    def context_width(self) -> int:
        """Return the concatenated auxiliary target hidden-state width."""
        return len(self.aux_layers) * self.hidden

    @classmethod
    def from_checkpoint(cls, directory: Path | str) -> DraftConfig:
        """Load a DraftConfig from a HuggingFace checkpoint directory."""
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
    """Return expected checkpoint tensor names and shapes for a DraftConfig."""
    shapes: dict[str, tuple[int, ...]] = {
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


def rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """Qwen3RMSNorm: FP32 statistics, cast back, then scale by the weight."""
    dtype = x.dtype
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return weight * normed.astype(dtype)


def synthetic_count(
    rates: jax.Array | np.ndarray | Sequence[float],
    seed: int | jax.Array,
    position: int | jax.Array,
) -> jax.Array:
    """Return the deterministically sampled accepted draft count for one verify step.

    ``rates[i]`` (non-increasing, vLLM's ``synthetic_acceptance_rates``) is the
    probability that the first ``i + 1`` drafts are accepted, drawn
    deterministically from ``(seed, position)``.
    """
    uniform = jax.random.uniform(jax.random.fold_in(jax.random.PRNGKey(seed), position))
    return jnp.sum(uniform < jnp.asarray(rates, jnp.float32)).astype(jnp.int32)


def nucleus_accept(
    values: jax.Array | np.ndarray,
    ids: jax.Array | np.ndarray,
    lse: jax.Array | np.ndarray,
    drafts: jax.Array | np.ndarray,
    forced_count: jax.Array | np.ndarray | int,
    top_p: jax.Array | np.ndarray | float,
    uniforms: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """Run speculative top-p sampling and return ``(count, bonus)``.

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
    # forced_count >= 0 fixes the accepted count (prefill); else derived from matches
    values, ids, lse, drafts, uniforms = (
        jnp.asarray(x) for x in (values, ids, lse, drafts, uniforms)
    )
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
    # a row without a draft is a rejection
    accept = drafted & (uniforms[:, 0] < draft_prob)
    first_reject = jnp.min(jnp.where(~accept & (k < rows - 1), k, rows - 1))
    count = jnp.where(forced_count >= 0, forced_count, first_reject).astype(jnp.int32)
    row = nucleus[count]
    # the rejected draft leaves the residual
    remove = (forced_count < 0) & drafted[count]
    residual = jnp.where(remove & draft_mask[count], 0.0, row)
    cumulative = jnp.cumsum(residual)
    total = cumulative[-1]
    # first index past the threshold
    pick = jnp.argmax(cumulative > uniforms[count, 1] * total)
    bonus = jnp.where(total > 0, sorted_ids[count, pick], sorted_ids[count, 0])
    return count, bonus.astype(jnp.int32)


def load_checkpoint_weights(
    directory: Path | str, c: DraftConfig | None = None
) -> dict[str, np.ndarray]:
    """Read the safetensors file into checkpoint-named bf16 arrays."""
    directory = Path(directory)
    c = c or DraftConfig.from_checkpoint(directory)
    raw = load_file(str(directory / "model.safetensors"))
    weights: dict[str, np.ndarray] = {}
    for name, shape in weight_shapes(c).items():
        value = raw[name]
        if tuple(value.shape) != shape:
            raise ValueError(f"{name}: expected {shape}, found {value.shape}")
        # safetensors' numpy loader hands bf16 back as uint16 bit patterns.
        if value.dtype == np.uint16:
            value = value.view(ml_dtypes.bfloat16)
        weights[name] = np.asarray(value).astype(ml_dtypes.bfloat16)
    return weights


ROWS = SUBLANE_ROWS  # maximum block width and padded public ABI (anchor + 7)
RANKS = TP_SIZE
LM_CHUNK = 512  # LM head columns per chunk (the gate/up buffer width)
LM_CHUNKS = 10  # 10 * 512 = 5120 vocabulary rows per rank
NEG = -3.0e38
MIN_BLOCK_LEN = 2
MAX_REQUESTS = 16
_WIDE_REQUESTS = 8
_WIDE_MAX_BLOCK = 4
_LAST_HOST_CHUNK = 7
SECOND_LOGIT_LANE = 2

# Output vector layout (int32 [1, 128]): drafts, this step's verify targets, accepted
# count, bonus token (the next anchor), top-2 margins as FP32 bits.
OUT_DRAFTS, OUT_TARGETS, OUT_COUNT, OUT_BONUS, OUT_MARGINS = 0, 8, 16, 17, 24

SEM_RS, SEM_HOST, SEM_GATHER, SEM_TINY, SEM_BCAST, SEMS_PER_SLOT = (
    0,
    8,
    16,
    32,
    64,
    96,
)

VMEM_NAMES = (
    "ln1",
    "ln2",
    "qnorm",
    "kvnorm",
    "hidden_norm",
    "final_norm",
    "mask_embed",
    "rot_q",
    "rot_kv",
    "head_ones",
    "kv_ones",
    "head_select",
    "head_place",
)
HBM_NAMES = (
    "q",
    "kv",
    "o",
    "gate",
    "up",
    "down",
    "fc",
    "lm_head",
    "w2t",
    "w1",
    "embed",
    "rope_q",
    "rope_kv",
)
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


def rope_tables(c: DraftConfig, max_position: int) -> tuple[np.ndarray, np.ndarray]:
    """Build FP32 RoPE tables for local query heads and packed K/V rows.

    ``[P, 384]`` (cos x3 | sin x3) for the 3 local q heads and ``[P, 256]``
    (cos | ones | sin | zeros) for the packed K/V row, FP32.
    """
    # ---------------- positions, masks, RoPE ----------------
    positions = np.arange(max_position, dtype=np.float32)
    inv_freq = 1.0 / c.rope_theta ** (
        np.arange(0, c.head_dim, 2, dtype=np.float32) / c.head_dim
    )
    freqs = positions[:, None] * inv_freq[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)
    heads = 3
    rope_q = np.concatenate([cos] * heads + [sin] * heads, axis=1).astype(np.float32)
    ones, zeros = np.ones_like(cos), np.zeros_like(sin)
    rope_kv = np.concatenate((cos, ones, sin, zeros), axis=1).astype(np.float32)
    return rope_q, rope_kv


def _replicated_dspark_shards(
    c: DraftConfig,
    w: dict[str, np.ndarray],
    ranks: int,
    max_position: int,
) -> dict[str, np.ndarray]:
    """Build replicated per-rank weight views and RoPE tables for DSpark."""
    bf16_dtype = ml_dtypes.bfloat16

    def replicated(value: np.ndarray | jax.Array | Sequence[float]) -> np.ndarray:
        """Broadcast an array across all ranks along a leading rank axis."""
        return np.broadcast_to(
            np.asarray(value)[None], (ranks, *np.asarray(value).shape)
        )

    rotation = _rotation(c.head_dim)
    rot_kv = np.zeros((2 * c.head_dim, 2 * c.head_dim), np.float32)
    rot_kv[: c.head_dim, : c.head_dim] = rotation
    kv_ones = np.zeros((2 * c.head_dim, 2 * c.head_dim), np.float32)
    kv_ones[: c.head_dim, : c.head_dim] = 1.0
    select = np.zeros((3, 3 * c.head_dim, 2 * c.head_dim), np.float32)
    place = np.zeros((3, 2 * c.head_dim, 3 * c.head_dim), np.float32)
    for head in range(3):
        for i in range(c.head_dim):
            select[head, head * c.head_dim + i, i] = 1.0
            place[head, c.head_dim + i, head * c.head_dim + i] = 1.0
    rope_q, rope_kv = rope_tables(c, max_position + 32)
    return {
        "ln1": replicated(
            np.stack(
                [
                    w[f"layers.{layer_idx}.input_layernorm.weight"][None]
                    for layer_idx in range(c.layers)
                ]
            )
        ).astype(bf16_dtype),
        "ln2": replicated(
            np.stack(
                [
                    w[f"layers.{layer_idx}.post_attention_layernorm.weight"][None]
                    for layer_idx in range(c.layers)
                ]
            )
        ).astype(bf16_dtype),
        "qnorm": replicated(
            np.stack(
                [
                    np.tile(w[f"layers.{layer_idx}.self_attn.q_norm.weight"], 3)[None]
                    for layer_idx in range(c.layers)
                ]
            )
        ).astype(bf16_dtype),
        "kvnorm": replicated(
            np.stack(
                [
                    np.concatenate(
                        (
                            w[f"layers.{layer_idx}.self_attn.k_norm.weight"],
                            np.ones((c.head_dim,), np.float32),
                        )
                    )[None]
                    for layer_idx in range(c.layers)
                ]
            )
        ).astype(bf16_dtype),
        "hidden_norm": replicated(w["hidden_norm.weight"][None]).astype(bf16_dtype),
        "final_norm": replicated(w["norm.weight"][None]).astype(bf16_dtype),
        "mask_embed": replicated(w["embed_tokens.weight"][c.mask_token][None]).astype(
            bf16_dtype
        ),
        "w1": replicated(w["markov_head.markov_w1.weight"]).astype(bf16_dtype),
        "rot_q": replicated(np.kron(np.eye(3, dtype=np.float32), rotation)).astype(
            bf16_dtype
        ),
        "rot_kv": replicated(rot_kv).astype(bf16_dtype),
        "head_ones": replicated(
            np.kron(
                np.eye(3, dtype=np.float32),
                np.ones((c.head_dim, c.head_dim), np.float32),
            )
        ).astype(bf16_dtype),
        "kv_ones": replicated(kv_ones).astype(bf16_dtype),
        "head_select": replicated(select).astype(bf16_dtype),
        "head_place": replicated(place).astype(bf16_dtype),
        "rope_q": replicated(rope_q),
        "rope_kv": replicated(rope_kv),
    }


def kernel_weight_shards(
    c: DraftConfig,
    weights: Mapping[str, np.ndarray | jax.Array],
    *,
    max_position: int = 8192,
    ranks: int = RANKS,
) -> dict[str, np.ndarray]:
    """Build per-rank ``[ranks, ...]`` kernel shards from checkpoint arrays."""
    bf16_dtype = ml_dtypes.bfloat16
    w = {name: np.asarray(value) for name, value in weights.items()}
    q_width = c.heads // ranks * c.head_dim  # 192
    kv_share = ranks // c.kv_heads  # 2
    mlp = c.intermediate // ranks  # 448
    vocab = c.vocab // ranks  # 5120
    np.testing.assert_equal(vocab, LM_CHUNKS * LM_CHUNK)
    np.testing.assert_(ranks > 0)
    if vocab != LM_CHUNKS * LM_CHUNK:
        raise ValueError("vocab shard does not match LM chunk geometry")
    shards: dict[str, np.ndarray] = {}

    def stack_layers(make: Callable[[int, int], np.ndarray]) -> np.ndarray:
        """Stack per-layer, per-rank arrays into shape [ranks, layers, ...]."""
        return np.stack(
            [
                np.stack([make(layer, rank) for rank in range(ranks)])
                for layer in range(c.layers)
            ],
            axis=1,
        )

    shards["q"] = stack_layers(
        lambda layer_idx, rank_idx: (
            w[f"layers.{layer_idx}.self_attn.q_proj.weight"][
                rank_idx * q_width : (rank_idx + 1) * q_width
            ].T
        )
    ).astype(bf16_dtype)

    def kv(layer_idx: int, rank_idx: int) -> np.ndarray:
        """Build packed key/value projection shard for one layer and rank."""
        head = rank_idx // kv_share
        k = w[f"layers.{layer_idx}.self_attn.k_proj.weight"][
            head * c.head_dim : (head + 1) * c.head_dim
        ].T
        v = w[f"layers.{layer_idx}.self_attn.v_proj.weight"][
            head * c.head_dim : (head + 1) * c.head_dim
        ].T
        return np.concatenate((k, v), axis=1)

    shards["kv"] = stack_layers(kv).astype(bf16_dtype)

    # o and down produce the partial that is reduce-scattered: memory block k of their
    # 896-column chunks holds logical chunk (lane ^ k), so chunk k can be sent to peer
    # rank ^ k as soon as the MXU produces it.
    def permute_chunks(value: np.ndarray, rank_idx: int) -> np.ndarray:
        """Permute 896-column output chunks in XOR lane order for reduce-scatter."""
        lane = rank_idx % 8
        return np.concatenate(
            [value[:, (lane ^ k) * CHUNK : ((lane ^ k) + 1) * CHUNK] for k in range(8)],
            axis=1,
        )

    shards["o"] = stack_layers(
        lambda layer_idx, rank_idx: permute_chunks(
            w[f"layers.{layer_idx}.self_attn.o_proj.weight"][
                :, rank_idx * q_width : (rank_idx + 1) * q_width
            ].T,
            rank_idx,
        )
    ).astype(bf16_dtype)

    # MLP channels padded from 448 to 512 per rank: lane windows in VMEM must be whole
    # 128-lane tiles. The padding columns/rows are zero.
    def pad_columns(value: np.ndarray) -> np.ndarray:
        """Pad column dimension of MLP projection up to MLP_PADDED."""
        return np.pad(value, ((0, 0), (0, MLP_PADDED - value.shape[1])))

    def pad_rows(value: np.ndarray) -> np.ndarray:
        """Pad row dimension of MLP down projection up to MLP_PADDED."""
        return np.pad(value, ((0, MLP_PADDED - value.shape[0]), (0, 0)))

    shards["gate"] = stack_layers(
        lambda layer_idx, rank_idx: pad_columns(
            w[f"layers.{layer_idx}.mlp.gate_proj.weight"][
                rank_idx * mlp : (rank_idx + 1) * mlp
            ].T
        )
    ).astype(bf16_dtype)
    shards["up"] = stack_layers(
        lambda layer_idx, rank_idx: pad_columns(
            w[f"layers.{layer_idx}.mlp.up_proj.weight"][
                rank_idx * mlp : (rank_idx + 1) * mlp
            ].T
        )
    ).astype(bf16_dtype)
    shards["down"] = stack_layers(
        lambda layer_idx, rank_idx: permute_chunks(
            pad_rows(
                w[f"layers.{layer_idx}.mlp.down_proj.weight"][
                    :, rank_idx * mlp : (rank_idx + 1) * mlp
                ].T
            ),
            rank_idx,
        )
    ).astype(bf16_dtype)

    rep = _replicated_dspark_shards(c, w, ranks, max_position)
    for name in (
        "ln1",
        "ln2",
        "qnorm",
        "kvnorm",
        "hidden_norm",
        "final_norm",
        "mask_embed",
    ):
        shards[name] = rep[name]

    # fc: rank (host h, lane j) contracts host h's quarter of the concatenated aux
    # features for output columns [896 j, 896 (j + 1)).
    fc = w["fc.weight"]  # [7168 out, 35840 in]
    shards["fc"] = np.stack(
        [
            fc[
                (r % 8) * CHUNK : (r % 8 + 1) * CHUNK,
                (r // 8) * FC_IN : (r // 8 + 1) * FC_IN,
            ].T
            for r in range(ranks)
        ]
    ).astype(bf16_dtype)

    # Note Shape [ranks, 10, 512, 7168]
    lm = w["lm_head.weight"].reshape(ranks, LM_CHUNKS, LM_CHUNK, c.hidden)
    shards["lm_head"] = np.ascontiguousarray(np.transpose(lm, (0, 1, 3, 2))).astype(
        bf16_dtype
    )
    w2 = w["markov_head.markov_w2.weight"].reshape(ranks, vocab, c.markov_rank)
    shards["w2t"] = np.ascontiguousarray(np.transpose(w2, (0, 2, 1))).astype(bf16_dtype)
    shards["w1"] = rep["w1"]
    shards["embed"] = (
        w["embed_tokens.weight"].reshape(ranks, vocab, c.hidden).astype(bf16_dtype)
    )
    for name in (
        "rot_q",
        "rot_kv",
        "head_ones",
        "kv_ones",
        "head_select",
        "head_place",
        "rope_q",
        "rope_kv",
    ):
        shards[name] = rep[name]
    return shards


def save_kernel_shards(
    shards: dict[str, np.ndarray],
    directory: Path | str,
    *,
    ranks: int = RANKS,
) -> None:
    """Write ``kernel_weight_shards`` output as one raw file per rank and index.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(shards)
    index: dict[str, Any] = {
        "format": KERNEL_SHARD_FORMAT,
        "ranks": ranks,
        "arrays": {},
    }
    offset = 0
    for name in names:
        value = shards[name]
        index["arrays"][name] = {
            "offset": offset,
            "shape": list(value.shape[1:]),
            "dtype": str(np.dtype(value.dtype)),
        }
        offset += int(np.prod(value.shape[1:])) * np.dtype(value.dtype).itemsize
    index["bytes_per_rank"] = offset
    for rank in range(ranks):
        with (directory / f"rank{rank:02d}.bin").open("wb") as handle:
            for name in names:
                handle.write(np.ascontiguousarray(shards[name][rank]).tobytes())
    (directory / "index.json").write_text(json.dumps(index, indent=1))


# --------------------------------------------------------------------------
# In-kernel helpers


def pack(value: jax.Array) -> jax.Array:
    """Pack ``[rows, width]`` into FP32 ``[rows * width / 256, 128]`` BF16 pairs."""
    rows, width = value.shape
    lanes = (
        value.astype(jnp.float32).reshape(rows * width // 128, 128).astype(jnp.bfloat16)
    )
    return jax.lax.bitcast_convert_type(tpu.bitcast(lanes, jnp.uint32), jnp.float32)


def unpack(words: jax.Array, rows: int, width: int) -> jax.Array:
    """Inverse of :func:`pack`, returned as FP32 ``[rows, width]``."""
    values = tpu.bitcast(jax.lax.bitcast_convert_type(words, jnp.uint32), jnp.bfloat16)
    return values.astype(jnp.float32).reshape(rows, width)


def bf16(x: jax.Array) -> jax.Array:
    """Cast a JAX array to bfloat16."""
    return x.astype(jnp.bfloat16)


def f32(x: jax.Array) -> jax.Array:
    """Cast a JAX array to float32."""
    return x.astype(jnp.float32)


def tile_rms_norm(x_bf16: jax.Array, weight_bf16: jax.Array, eps: float) -> jax.Array:
    """Apply Qwen3RMSNorm with FP32 statistics, BF16 cast, and BF16 scaling."""
    x32 = f32(x_bf16)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return bf16(f32(bf16(normed)) * f32(weight_bf16))


def _grouped_sumsq(x32: jax.Array, ones_bf16: jax.Array) -> jax.Array:
    """Compute per-group sum of squares broadcast to the group's lanes.

    Uses ``x^2 @ blockdiag(ones)`` with the FP32 square split into BF16 high and
    low parts (relative error ~2^-16).
    """
    squares = x32 * x32
    high = bf16(squares)
    low = bf16(squares - f32(high))
    return dot(high, ones_bf16) + dot(low, ones_bf16)


def _rope(
    x_bf16: jax.Array,
    rotation_bf16: jax.Array,
    cos32: jax.Array,
    sin32: jax.Array,
) -> jax.Array:
    """Apply reference ``x * cos + rotate_half(x) * sin`` in BF16 arithmetic."""
    rotated = bf16(dot(x_bf16, rotation_bf16))
    first = bf16(f32(x_bf16) * cos32)
    second = bf16(f32(rotated) * sin32)
    return bf16(f32(first) + f32(second))


def select_rows(block32: jax.Array, offset: int | jax.Array, rows: int) -> jax.Array:
    """Select rows ``offset .. offset + rows`` of a ``[16, W]`` tile via masked sums."""
    row_index = jax.lax.broadcasted_iota(jnp.int32, block32.shape, 0)
    return jnp.concatenate(
        [
            jnp.sum(
                jnp.where(row_index == offset + k, block32, 0.0),
                axis=0,
                keepdims=True,
            )
            for k in range(rows)
        ],
        axis=0,
    )


# --------------------------------------------------------------------------
# The kernel


# HBM operands copied into VMEM at entry (the logits land in logits_vmem)
SMALL_NAMES = (
    "aux",
    "positions",
    "previous",
    *VMEM_NAMES,
)


def _fused_rows(requests: int, block: int) -> int:
    """Return physical rows per request, bounded to 64 rows per fused draft.

    Physical rows per request, bounded to 64 rows per fused draft.

    Eight rows per request up to 8 requests; four above (block <= 4), so the
    unused padded ABI rows do not consume projection and collective scratch.
    """
    if not MIN_BLOCK_LEN <= block <= ROWS:
        raise ValueError("block must be between 2 and 8")
    if not 1 <= requests <= MAX_REQUESTS or (
        requests > _WIDE_REQUESTS and block > _WIDE_MAX_BLOCK
    ):
        raise ValueError(
            "fused draft supports up to 8 requests at block 8, or 16 at block 4"
        )
    return ROWS if requests <= _WIDE_REQUESTS else _WIDE_MAX_BLOCK


@dataclasses.dataclass
class _DsparkStepEnv:
    """Static and per-invocation state for one Qwen3-style DSpark fused kernel call."""

    c: DraftConfig
    block: int
    eps: float
    layers: int
    window: int
    num_requests: int
    rows: int
    total_rows: int
    stream_aux: bool
    kv_width: int
    q_width: int
    vocab: int
    hidden: int
    scale: float
    sem_fc: int
    sem_w2t: int
    sem_own: int
    sem_ring: int
    scalars: Any = None
    aux_hbm: Any = None
    positions_hbm: Any = None
    logits_hbm: Any = None
    previous_hbm: Any = None
    weights: dict[str, Any] = dataclasses.field(default_factory=dict)
    cache_in: Any = None
    tokens_out: Any = None
    positions_out: Any = None
    cache_out: Any = None
    small: dict[str, Any] = dataclasses.field(default_factory=dict)
    a_buf: Any = None
    q_buf: Any = None
    kv_buf: Any = None
    o_buf: Any = None
    gate_buf: Any = None
    up_buf: Any = None
    down_buf: Any = None
    w2t_buf: Any = None
    w1_blk: Any = None
    embed_blk: Any = None
    bcast_buf: Any = None
    rope_q_blk: Any = None
    rope_kv_blk: Any = None
    ring_vmem: Any = None
    logits_vmem: Any = None
    fc_part: Any = None
    tokens_v: Any = None
    stage: Any = None
    local_recv: Any = None
    host_packed: Any = None
    gather_recv: Any = None
    tiny_recv: Any = None
    argmax_recv: Any = None
    newkv_buf: Any = None
    dma_sems: Any = None
    small_sems: Any = None
    sends: Any = None
    recvs: Any = None
    rank: Any = None
    lane: Any = None
    host: Any = None
    # The scalars operand is SMEM-prefetched: only element loads.
    new_position: Any = None  # [R]: the verify step's first row
    forced_count: Any = None
    # [R]: >= 0 fixes the bonus token (sampled by the caller); else the target at
    # ``count``
    forced_anchor: Any = None
    w1_copies: list[list[Any]] = dataclasses.field(default_factory=list)
    layer_buffers: dict[str, Any] = dataclasses.field(default_factory=dict)
    layer_sems: dict[str, int] = dataclasses.field(default_factory=dict)
    # Layer-0 weights and the Markov pieces stream in behind the fc phase.
    pending: dict[str, Any] = dataclasses.field(default_factory=dict)
    # ---------------- the ring slice stream ---------------- The kernel visits (layer,
    # request) ring slices in visit order (v = layer * R + request): each visit copies
    # its [window, 128] slice into one of the two buffers (parity), applies the
    # request's new context rows, attends, and writes the slice back; the in-copy of a
    # visit follows the write-back two visits earlier (same buffer).
    ring: dict[str, Any] = dataclasses.field(default_factory=dict)

    def bind_refs(
        self,
        scalars: jax.Array,
        aux_hbm: jax.Array,
        positions_hbm: jax.Array,
        logits_hbm: jax.Array,
        refs_and_prev: tuple[jax.Array, tuple[jax.Array, ...]],
    ) -> None:
        """Unpack Pallas operand and scratch refs into the step environment."""
        previous_hbm, refs = refs_and_prev
        self.scalars = scalars
        self.aux_hbm = aux_hbm
        self.positions_hbm = positions_hbm
        self.logits_hbm = logits_hbm
        self.previous_hbm = previous_hbm
        self.weights = dict(zip(WEIGHT_NAMES, refs[: len(WEIGHT_NAMES)], strict=True))
        self.cache_in = refs[len(WEIGHT_NAMES)]
        self.tokens_out, self.positions_out, self.cache_out = refs[
            len(WEIGHT_NAMES) + 1 : len(WEIGHT_NAMES) + 4
        ]
        scratch = refs[len(WEIGHT_NAMES) + 4 :]
        self.small = dict(zip(SMALL_NAMES, scratch[: len(SMALL_NAMES)], strict=True))
        (
            self.a_buf,
            self.q_buf,
            self.kv_buf,
            self.o_buf,
            self.gate_buf,
            self.up_buf,
            self.down_buf,
            self.w2t_buf,
            self.w1_blk,
            self.embed_blk,
            self.bcast_buf,
            self.rope_q_blk,
            self.rope_kv_blk,
            self.ring_vmem,
            self.logits_vmem,
            self.fc_part,
            self.tokens_v,
            self.stage,
            self.local_recv,
            self.host_packed,
            self.gather_recv,
            self.tiny_recv,
            self.argmax_recv,
            self.newkv_buf,
            self.dma_sems,
            self.small_sems,
            self.sends,
            self.recvs,
        ) = scratch[len(SMALL_NAMES) :]
        self.rank = jax.lax.axis_index("tp")
        self.lane = self.rank % 8
        self.host = self.rank // 8
        self.new_position = jnp.stack([scalars[j] for j in range(self.num_requests)])
        self.forced_count = jnp.stack(
            [scalars[self.num_requests + j] for j in range(self.num_requests)]
        )
        self.forced_anchor = jnp.stack(
            [scalars[2 * self.num_requests + j] for j in range(self.num_requests)]
        )
        self.w1_copies = [[None] * self.num_requests, [None] * self.num_requests]
        self.layer_buffers = {
            "q": self.q_buf,
            "kv": self.kv_buf,
            "o": self.o_buf,
            "gate": self.gate_buf,
            "up": self.up_buf,
            "down": self.down_buf,
        }
        self.layer_sems = {
            "q": 2,
            "kv": 3,
            "o": 4,
            "gate": 5,
            "up": 6,
            "down": 7,
        }
        self.ring = {"out": [None, None], "in": [None, None], "done": 0}

    # ---------------- DMA helpers ----------------
    def copy(self, src: _Buffer, dst: _Buffer, sem: int | jax.Array) -> _AsyncCopy:
        """Create a local async DMA copy descriptor using dma_sems[sem]."""
        return tpu.make_async_copy(src, dst, self.dma_sems.at[sem])

    def remote(
        self,
        src: _Buffer,
        dst: _Buffer,
        sem_index: int | jax.Array,
        peer: int | jax.Array,
    ) -> _AsyncCopy:
        """Create a remote ICI async DMA copy descriptor to mesh peer."""
        return tpu.make_async_remote_copy(
            src,
            dst,
            self.sends.at[sem_index],
            self.recvs.at[sem_index],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def exchange(
        self,
        src_of: Callable[[int], _Buffer],
        dst_of: Callable[[int], _Buffer],
        offsets: Iterable[int],
        slot_and_base: tuple[int, int],
    ) -> None:
        """Run a synchronous XOR-peer remote exchange across offsets."""
        slot, base = slot_and_base
        copies = [
            self.remote(
                src_of(offset),
                dst_of(offset),
                slot * SEMS_PER_SLOT + base + offset,
                self.rank ^ offset,
            )
            for offset in offsets
        ]
        for item in copies:
            item.start()
        for item in copies:
            item.wait()

    # ---------------- collectives ----------------
    def host_reduce_gather(self, chunk32: jax.Array, slot: int) -> jax.Array:
        """Reduce a host-lane FP32 chunk across hosts and gather across lanes."""
        self.host_packed[slot, self.host] = pack(bf16(chunk32))
        self.exchange(
            lambda _offset: self.host_packed.at[slot, self.host],
            lambda _offset: self.host_packed.at[slot, self.host],
            (8, 16, 24),
            (slot, SEM_HOST),
        )
        reduced: Any = sum(
            unpack(self.host_packed[slot, h], self.total_rows, CHUNK) for h in range(4)
        )
        self.gather_recv[slot, self.lane] = pack(bf16(reduced))
        self.exchange(
            lambda _offset: self.gather_recv.at[slot, self.lane],
            lambda _offset: self.gather_recv.at[slot, self.lane],
            range(1, 8),
            (slot, SEM_GATHER),
        )
        return jnp.concatenate(
            [
                unpack(self.gather_recv[slot, j], self.total_rows, CHUNK)
                for j in range(8)
            ],
            axis=1,
        )

    def all_reduce_start(
        self,
        lhs_bf16: jax.Array,
        weight_ref: jax.Array,
        slot: int,
        refill: jax.Array | None = None,
    ) -> tuple[_AsyncCopy, list[_AsyncCopy]]:
        """Project ``lhs`` through chunk-permuted weights and reduce-scatter."""
        refill_sem = self.layer_sems["o"] if slot == 1 else 7
        copies: list[Any] = []
        own: Any = None
        width = weight_ref.shape[1]
        stream = refill is not None and width == CHUNK
        # Streamed (one-chunk buffer): double-buffer with a chunk of the FC scratch
        # (unused until the LM head); a chunk is waited before use and a buffer refilled
        # only after the matmul that read it.
        spare = (
            self.a_buf.at[pl.ds(0, weight_ref.shape[0]), pl.ds(0, CHUNK)]
            if stream
            else weight_ref
        )
        pending_refill: Any = None
        for k in range(8):
            current = weight_ref if not stream or k % 2 == 0 else spare
            if stream and refill is not None:
                if pending_refill is not None:
                    pending_refill.wait()
                if k < _LAST_HOST_CHUNK:
                    destination = spare if k % 2 == 0 else weight_ref
                    pending_refill = self.copy(
                        refill.at[:, pl.ds((k + 1) * CHUNK, CHUNK)],
                        destination,
                        refill_sem,
                    )
                    pending_refill.start()
            elif k > 0 and k * CHUNK % width == 0 and refill is not None:
                item = self.copy(
                    refill.at[:, pl.ds(k * CHUNK, width)],
                    weight_ref,
                    refill_sem,
                )
                item.start()
                item.wait()
            # [T, 896] FP32
            partial = dot(lhs_bf16, current[:, pl.ds((k * CHUNK) % width, CHUNK)])
            self.stage[slot, k] = pack(bf16(partial))
            if k == 0:
                own = self.copy(
                    self.stage.at[slot, 0],
                    self.local_recv.at[slot, self.lane],
                    self.sem_own,
                )
                own.start()
            else:
                item = self.remote(
                    self.stage.at[slot, k],
                    self.local_recv.at[slot, self.lane],
                    slot * SEMS_PER_SLOT + SEM_RS + k,
                    self.rank ^ k,
                )
                item.start()
                copies.append(item)
        return own, copies

    def all_reduce_finish(
        self, handles: tuple[_AsyncCopy, list[_AsyncCopy]], slot: int
    ) -> jax.Array:
        """Complete a chunked reduce-scatter and cross-host all-gather."""
        own, copies = handles
        for item in copies:
            item.wait()
        own.wait()
        chunk: Any = sum(
            unpack(self.local_recv[slot, k], self.total_rows, CHUNK) for k in range(8)
        )
        return self.host_reduce_gather(chunk, slot)

    def tiny_exchange(self, row_i32: jax.Array, slot: int) -> jax.Array:
        """All-gather one [R, 128] int32 row set from every rank."""
        self.tiny_recv[slot, self.rank] = row_i32
        self.exchange(
            lambda _offset: self.tiny_recv.at[slot, self.rank],
            lambda _offset: self.tiny_recv.at[slot, self.rank],
            range(1, 32),
            (slot, SEM_TINY),
        )
        return self.tiny_recv[slot]

    def ring_start_in(self, visit: int) -> None:
        """Start prefetching the ring slice for the given visit index."""
        buf_idx = visit % 2
        if self.ring["out"][buf_idx] is not None:
            self.ring["out"][buf_idx].wait()
            self.ring["out"][buf_idx] = None
        item = self.copy(
            self.cache_in.at[visit % self.num_requests, visit // self.num_requests],
            self.ring_vmem.at[buf_idx],
            self.sem_ring + buf_idx,
        )
        item.start()
        self.ring["in"][buf_idx] = item

    def ring_drain(self) -> None:
        """Wait for all pending ring slice copies and flush unvisited slices."""
        for buf_idx in (0, 1):
            if self.ring["in"][buf_idx] is not None:
                self.ring["in"][buf_idx].wait()
                self.ring["in"][buf_idx] = None
            if self.ring["out"][buf_idx] is not None:
                self.ring["out"][buf_idx].wait()
                self.ring["out"][buf_idx] = None
        for visit in range(self.ring["done"], self.layers * self.num_requests):
            buf_idx = visit % 2
            item = self.copy(
                self.cache_in.at[visit % self.num_requests, visit // self.num_requests],
                self.ring_vmem.at[buf_idx],
                self.sem_ring + buf_idx,
            )
            item.start()
            item.wait()
            item = self.copy(
                self.ring_vmem.at[buf_idx],
                self.cache_out.at[
                    visit % self.num_requests, visit // self.num_requests
                ],
                self.sem_ring + 2 + buf_idx,
            )
            item.start()
            item.wait()

    def start_w1(self, tokens: jax.Array, slot: int) -> None:
        """Start prefetching the aligned 16-row w1 block of each request's token."""
        for j in range(self.num_requests):
            start = pl.multiple_of((tokens[j] // 16) * 16, 16)
            item = self.copy(
                self.weights["w1"].at[pl.ds(start, 16)],
                self.w1_blk.at[slot, j],
                10 + (4 + slot) * self.num_requests + j,
            )
            item.start()
            self.w1_copies[slot][j] = item

    def w1_drain(self, slot: int) -> list[_AsyncCopy]:
        """Return active w1 DMA copy handles for the given parity slot."""
        return [item for item in self.w1_copies[slot] if item is not None]

    def w1_rows_for(self, tokens: jax.Array, slot: int) -> jax.Array:
        """Select each request's Markov w1 row from its 16-row VMEM block."""
        return jnp.concatenate(
            [
                bf16(select_rows(f32(self.w1_blk[slot, j]), tokens[j] % 16, 1))
                for j in range(self.num_requests)
            ],
            axis=0,
        )

    def layer_copy(self, name: str, layer: int) -> _AsyncCopy:
        """Create an async HBM-to-VMEM copy for a layer weight buffer."""
        if name in ("down", "o"):
            return self.copy(
                self.weights[name].at[
                    layer, :, pl.ds(0, self.layer_buffers[name].shape[1])
                ],
                self.layer_buffers[name],
                self.layer_sems[name],
            )
        return self.copy(
            self.weights[name].at[layer],
            self.layer_buffers[name],
            self.layer_sems[name],
        )

    def prefetch_next(self, name: str, layer: int) -> None:
        """Start prefetching weight family ``name`` for ``layer + 1`` when in range."""
        if layer + 1 < self.layers:
            self.pending[name] = self.layer_copy(name, layer + 1)
            self.pending[name].start()

    def kv_rows(
        self,
        features_bf16: jax.Array,
        layer: int,
        cos32: jax.Array,
        sin32: jax.Array,
    ) -> jax.Array:
        """Project, RMS-normalize, and apply RoPE to packed [T, 128] K/V rows."""
        # reference linear: BF16 output
        kv = bf16(dot(features_bf16, self.kv_buf[...]))
        sumsq = _grouped_sumsq(f32(kv), self.small["kv_ones"][...])  # K lanes only
        lane_index = jax.lax.broadcasted_iota(jnp.int32, kv.shape, 1)
        normed = f32(kv) * jax.lax.rsqrt(sumsq * (1.0 / self.c.head_dim) + self.eps)
        k_scaled = bf16(f32(bf16(normed)) * f32(self.small["kvnorm"][layer]))
        kv = jnp.where(lane_index < self.c.head_dim, k_scaled, kv)
        return _rope(kv, self.small["rot_kv"][...], cos32, sin32)


def _dspark_accept_and_issue_reads(
    env: _DsparkStepEnv,
    small_copies: list[_AsyncCopy],
    logits_copy: _AsyncCopy,
) -> tuple[jax.Array, jax.Array, jax.Array, list[_AsyncCopy]]:
    """Run verify acceptance, populate tokens_v, and issue w1/rope/embed DMAs."""
    for item in small_copies:
        item.wait()
    logits_copy.wait()
    # [R, 8, 5120] FP32 (this rank's vocabulary shard)
    verify_logits = env.logits_vmem[...]
    vocab_iota = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, env.vocab), 2
    )
    local_best = jnp.max(verify_logits, axis=2, keepdims=True)  # [R, 8, 1]
    local_id = jnp.min(
        jnp.where(verify_logits == local_best, vocab_iota, env.vocab),
        axis=2,
        keepdims=True,
    )
    local_second = jnp.max(
        jnp.where(vocab_iota == local_id, NEG, verify_logits),
        axis=2,
        keepdims=True,
    )
    out_lane = jax.lax.broadcasted_iota(jnp.int32, (env.num_requests, env.rows, 128), 2)
    packet = jnp.where(
        out_lane == 0,
        jax.lax.bitcast_convert_type(local_best, jnp.int32),
        jnp.where(
            out_lane == 1,
            local_id + env.rank * env.vocab,
            jnp.where(
                out_lane == SECOND_LOGIT_LANE,
                jax.lax.bitcast_convert_type(local_second, jnp.int32),
                0,
            ),
        ),
    )
    env.argmax_recv[env.rank] = packet.reshape(env.total_rows, 128)
    env.exchange(
        lambda _offset: env.argmax_recv.at[env.rank],
        lambda _offset: env.argmax_recv.at[env.rank],
        range(1, 32),
        (1, SEM_TINY),
    )
    # [32, R, 8, 128] int32
    gathered = env.argmax_recv[...].reshape(RANKS, env.num_requests, env.rows, 128)
    rank_iota4 = jax.lax.broadcasted_iota(
        jnp.int32, (RANKS, env.num_requests, env.rows, 1), 0
    )
    # Note Shape [32, R, 8, 1]
    bests = jax.lax.bitcast_convert_type(gathered[:, :, :, 0:1], jnp.float32)
    seconds = jax.lax.bitcast_convert_type(gathered[:, :, :, 2:3], jnp.float32)
    best = jnp.max(bests, axis=0)  # [R, 8, 1]
    # [R, 8, 1], lowest rank on ties
    winner = jnp.min(jnp.where(bests == best[None], rank_iota4, RANKS), axis=0)
    targets = jnp.sum(  # [R, 8, 1]
        jnp.where(rank_iota4 == winner[None], gathered[:, :, :, 1:2], 0), axis=0
    )
    runner_up = jnp.maximum(
        jnp.max(seconds, axis=0),
        jnp.max(jnp.where(rank_iota4 == winner[None], NEG, bests), axis=0),
    )
    margins = best - runner_up  # [R, 8, 1] FP32
    # [R, 128] int32: last step's output vectors
    previous_vector = env.small["previous"][...]
    row_iota_col = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, 1), 1
    )
    proposals = jnp.sum(
        jnp.where(
            jax.lax.broadcasted_iota(jnp.int32, (env.num_requests, env.rows, 128), 2)
            == row_iota_col + OUT_DRAFTS,
            jnp.broadcast_to(
                previous_vector[:, None, :], (env.num_requests, env.rows, 128)
            ),
            0,
        ),
        axis=2,
        keepdims=True,
    )
    drafts_count = env.block - 1
    # matches[j, k] = proposals[j, k] == targets[j, k] for k < drafts_count
    # no select on i1 vectors
    match_rows = (row_iota_col < drafts_count) & (proposals == targets)
    first_miss = jnp.min(
        jnp.where(match_rows | (row_iota_col >= drafts_count), env.block, row_iota_col),
        axis=1,
    )[:, 0]
    count = jnp.where(
        env.forced_count >= 0,
        env.forced_count,
        jnp.minimum(first_miss, drafts_count),
    )
    anchor_token = jnp.sum(
        jnp.where(row_iota_col[:, :, 0] == count[:, None], targets[:, :, 0], 0),
        axis=1,
    )
    anchor_token = jnp.where(env.forced_anchor >= 0, env.forced_anchor, anchor_token)
    anchor_position = env.new_position + count + 1
    result_vector = jnp.zeros((env.num_requests, 128), jnp.int32) - 1
    result_lane = jax.lax.broadcasted_iota(jnp.int32, (env.num_requests, 128), 1)
    field_row = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, 128), 1
    )
    field_lane = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, 128), 2
    )
    targets_row = jnp.sum(
        jnp.where(field_lane == field_row + OUT_TARGETS, targets, 0), axis=1
    )
    margins_row = jnp.sum(
        jnp.where(
            field_lane == field_row + OUT_MARGINS,
            jax.lax.bitcast_convert_type(margins, jnp.int32),
            0,
        ),
        axis=1,
    )
    result_vector = jnp.where(
        (result_lane >= OUT_TARGETS) & (result_lane < OUT_TARGETS + env.rows),
        targets_row,
        result_vector,
    )
    result_vector = jnp.where(
        (result_lane >= OUT_MARGINS) & (result_lane < OUT_MARGINS + env.rows),
        margins_row,
        result_vector,
    )
    result_vector = jnp.where(result_lane == OUT_COUNT, count[:, None], result_vector)
    result_vector = jnp.where(
        result_lane == OUT_BONUS, anchor_token[:, None], result_vector
    )
    if env.rows < ROWS:
        padding = (
            (result_lane >= OUT_TARGETS + env.rows) & (result_lane < OUT_TARGETS + ROWS)
        ) | (
            (result_lane >= OUT_MARGINS + env.rows) & (result_lane < OUT_MARGINS + ROWS)
        )
        result_vector = jnp.where(padding, 0, result_vector)
    env.tokens_v[...] = result_vector

    return _dspark_issue_initial_reads(env, anchor_token, anchor_position)


def _dspark_issue_initial_reads(
    env: _DsparkStepEnv,
    anchor_token: jax.Array,
    anchor_position: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, list[_AsyncCopy]]:
    """Start initial w1, RoPE, and anchor embedding broadcast DMAs."""
    env.start_w1(anchor_token, 0)
    rope_copies = []
    for j in range(env.num_requests):
        rope_q_start = pl.multiple_of((anchor_position[j] // 8) * 8, 8)
        rope_kv_start = pl.multiple_of((env.new_position[j] // 8) * 8, 8)
        rope_copies += (
            env.copy(
                env.weights["rope_q"].at[pl.ds(rope_q_start, 16)],
                env.rope_q_blk.at[j],
                10 + env.num_requests + j,
            ),
            env.copy(
                env.weights["rope_kv"].at[pl.ds(rope_q_start, 16)],
                env.rope_kv_blk.at[0, j],
                10 + 2 * env.num_requests + j,
            ),
            env.copy(
                env.weights["rope_kv"].at[pl.ds(rope_kv_start, 16)],
                env.rope_kv_blk.at[1, j],
                10 + 3 * env.num_requests + j,
            ),
        )
    for item in rope_copies:
        item.start()

    # Anchor embeddings: each request's owner fetches its aligned 16-row block, picks
    # the row and sends it (packed) to the 31 peers on semaphore SEM_BCAST + j.
    owner = anchor_token // env.vocab
    for j in range(env.num_requests):

        @pl.when(env.rank == owner[j])
        def send_anchor_row(j: int = j) -> None:
            """Fetch request j's anchor embedding row and broadcast it to peers."""
            local_id = anchor_token[j] - owner[j] * env.vocab
            embed_start = pl.multiple_of((local_id // 16) * 16, 16)
            item = env.copy(
                env.weights["embed"].at[pl.ds(embed_start, 16)],
                env.embed_blk,
                10 + j,
            )
            item.start()
            item.wait()
            row = select_rows(f32(env.embed_blk[...]), local_id % 16, 1)
            env.bcast_buf[j] = pack(bf16(row))
            copies = [
                env.remote(
                    env.bcast_buf.at[j],
                    env.bcast_buf.at[j],
                    SEM_BCAST + j,
                    env.rank ^ offset,
                )
                for offset in range(1, 32)
            ]
            for copy_item in copies:
                copy_item.start()
            for copy_item in copies:
                copy_item.wait_send()

    return anchor_token, anchor_position, owner, rope_copies


def _dspark_fc_context(env: _DsparkStepEnv) -> jax.Array:
    """Contract host-sharded auxiliary features and apply hidden RMSNorm."""
    aux_ref = env.small["aux"]  # [aux layers, T, hidden]
    spans = []
    for h in range(4):
        start, stop = h * FC_IN, (h + 1) * FC_IN
        pieces = []
        for layer_index in range(len(env.c.aux_layers)):
            lo, hi = layer_index * env.hidden, (layer_index + 1) * env.hidden
            a, b = max(start, lo), min(stop, hi)
            if a < b:
                pieces.append((layer_index, a - lo, b - lo, a - start))
        spans.append(pieces)

    fc_view = env.a_buf.at[:, pl.ds(0, CHUNK)]
    for h in range(4):

        @pl.when(env.host == h)
        def fc_branch(
            h: int = h,
            col_start: int = 0,
            cols: int = CHUNK,
            fc_view: jax.Array = fc_view,
        ) -> None:
            """Accumulate host h's FC partial product across requests."""
            for j in range(env.num_requests):
                if env.stream_aux:
                    # Aux features are used only by this request's FC product: stage one
                    # request's rows at a time.
                    item = tpu.make_async_copy(
                        env.aux_hbm.at[:, pl.ds(j * env.rows, env.rows), :],
                        aux_ref,
                        env.small_sems.at[0],
                    )
                    item.start()
                    item.wait()
                row_start = 0 if env.stream_aux else j * env.rows
                total = None
                for layer_index, lo, hi, offset in spans[h]:
                    part = dot(
                        aux_ref[
                            layer_index,
                            row_start : row_start + env.rows,
                            lo:hi,
                        ],
                        fc_view[pl.ds(offset, hi - lo), :],
                    )
                    total = part if total is None else total + part
                env.fc_part.at[pl.ds(j * env.rows, env.rows), pl.ds(col_start, cols)][
                    ...
                ] = total

    return tile_rms_norm(
        bf16(env.host_reduce_gather(env.fc_part[...], 0)),
        env.small["hidden_norm"][...],
        env.eps,
    )


def _dspark_context_and_masks(
    env: _DsparkStepEnv,
    fc_copy: _AsyncCopy,
    anchor_position: jax.Array,
    owner: jax.Array,
    rope_copies: list[_AsyncCopy],
) -> tuple[
    jax.Array,
    jax.Array,
    list[list[jax.Array]],
    jax.Array,
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    _AsyncCopy,
    _AsyncCopy,
]:
    """Run FC context projection, ring position updates, RoPE unpacking, and input x."""
    # ---------------- fc: context features for the new rows ----------------
    fc_copy.wait()
    env.pending = {name: env.layer_copy(name, 0) for name in env.layer_buffers}
    for name in ("kv", "q", "o", "gate", "up", "down"):
        env.pending[name].start()
    w2t_copy = env.copy(env.weights["w2t"], env.w2t_buf, env.sem_w2t)
    w2t_copy.start()

    context = _dspark_fc_context(env)  # [T, 7168]

    positions = env.small["positions"][...]  # [R, window] int32
    slot_lane = jax.lax.broadcasted_iota(jnp.int32, (env.num_requests, env.window), 1)
    new_slots = [
        [(env.new_position[j] + k) % env.window for k in range(env.block)]
        for j in range(env.num_requests)
    ]
    for k in range(env.block):
        positions_k = env.new_position + k
        positions = jnp.where(
            slot_lane == (positions_k % env.window)[:, None],
            positions_k[:, None],
            positions,
        )
    env.small["positions"][...] = positions
    positions_copy = tpu.make_async_copy(
        env.small["positions"], env.positions_out, env.small_sems.at[1]
    )
    positions_copy.start()
    context_allowed = (
        (positions >= 0)
        & (positions < anchor_position[:, None])
        & (positions >= anchor_position[:, None] - env.window)
    )
    row_iota = jax.lax.broadcasted_iota(jnp.int32, (env.rows, 16), 0)
    col_iota = jax.lax.broadcasted_iota(jnp.int32, (env.rows, 16), 1)
    block_allowed = (col_iota <= row_iota) & (col_iota < env.block)

    for item in rope_copies:
        item.wait()
    rope_q_rows = jnp.stack(  # [R, 8, 384]
        [
            select_rows(env.rope_q_blk[j], anchor_position[j] % 8, env.rows)
            for j in range(env.num_requests)
        ]
    )
    cos_q = f32(bf16(rope_q_rows[:, :, : env.q_width])).reshape(
        env.total_rows, env.q_width
    )
    sin_q = f32(bf16(rope_q_rows[:, :, env.q_width :])).reshape(
        env.total_rows, env.q_width
    )
    rope_kv_rows = jnp.stack(  # [R, 8, 256]
        [
            select_rows(env.rope_kv_blk[0, j], anchor_position[j] % 8, env.rows)
            for j in range(env.num_requests)
        ]
    )
    cos_kv = f32(bf16(rope_kv_rows[:, :, : env.kv_width])).reshape(
        env.total_rows, env.kv_width
    )
    sin_kv = f32(bf16(rope_kv_rows[:, :, env.kv_width :])).reshape(
        env.total_rows, env.kv_width
    )
    rope_new_rows = jnp.stack(
        [
            select_rows(env.rope_kv_blk[1, j], env.new_position[j] % 8, env.rows)
            for j in range(env.num_requests)
        ]
    )
    cos_new = f32(bf16(rope_new_rows[:, :, : env.kv_width])).reshape(
        env.total_rows, env.kv_width
    )
    sin_new = f32(bf16(rope_new_rows[:, :, env.kv_width :])).reshape(
        env.total_rows, env.kv_width
    )

    for j in range(env.num_requests):

        @pl.when(env.rank != owner[j])
        def receive_anchor_row(j: int = j) -> None:
            """Wait on the remote receive semaphore for request j's anchor."""
            env.remote(
                env.bcast_buf.at[j], env.bcast_buf.at[j], SEM_BCAST + j, env.rank
            ).wait_recv()

    anchor_rows = jnp.concatenate(  # [R, 7168] FP32 (BF16-valued)
        [unpack(env.bcast_buf[j], 1, env.hidden) for j in range(env.num_requests)],
        axis=0,
    )
    mask_row = f32(env.small["mask_embed"][...])
    x_row = jax.lax.broadcasted_iota(
        jnp.int32, (env.num_requests, env.rows, env.hidden), 1
    )
    x = bf16(
        jnp.where(x_row == 0, anchor_rows[:, None, :], mask_row[:, None, :])
    ).reshape(env.total_rows, env.hidden)

    env.pending["kv"].wait()
    env.pending["kv"] = None
    env.newkv_buf[...] = env.kv_rows(context, 0, cos_new, sin_new)
    return (
        x,
        context,
        new_slots,
        context_allowed,
        block_allowed,
        (cos_q, sin_q, cos_kv, sin_kv, cos_new, sin_new),
        w2t_copy,
        positions_copy,
    )


def _dspark_write_context_rows(
    env: _DsparkStepEnv,
    j: int,
    buf_idx: int,
    new_slots: list[list[jax.Array]],
) -> None:
    """Write request j's new context K/V rows into ring buffer ``buf_idx``."""
    for raw_slab_start in (
        new_slots[j][0] // 128 * 128,
        new_slots[j][env.block - 1] // 128 * 128,
    ):
        slab_start = pl.multiple_of(raw_slab_start, 128)
        slab_ref = env.ring_vmem.at[buf_idx, pl.ds(slab_start, 128)]
        slab = slab_ref[...]
        slab_rows = jax.lax.broadcasted_iota(jnp.int32, slab.shape, 0) + slab_start
        for k in range(env.block):
            slab = jnp.where(
                slab_rows == new_slots[j][k],
                env.newkv_buf[j * env.rows + k : j * env.rows + k + 1, :],
                slab,
            )
        slab_ref[...] = slab


def _dspark_attend_request(
    env: _DsparkStepEnv,
    layer: int,
    j: int,
    q: jax.Array,  # [T, 192]
    attn_ctx: tuple[jax.Array, list[list[jax.Array]], jax.Array, jax.Array],
) -> jax.Array:
    """Update ring slice and compute 3-head sliding-window attention for request j."""
    block_tile, new_slots, context_allowed, block_allowed = attn_ctx  # [R, 16, 128]
    visit = layer * env.num_requests + j
    buf_idx = visit % 2
    env.ring["in"][buf_idx].wait()
    env.ring["in"][buf_idx] = None
    _dspark_write_context_rows(env, j, buf_idx, new_slots)
    ring_tile = env.ring_vmem[buf_idx]  # [window, 128] bf16
    q_rows = q[j * env.rows : (j + 1) * env.rows]
    # V lanes -> head lanes
    attended_j = jnp.zeros((env.rows, env.q_width), jnp.float32)
    for head in range(3):
        # [8, 128], zeros on V lanes
        q_head = bf16(dot(q_rows, env.small["head_select"][head]))
        s_ctx = attention_scores(q_head, ring_tile) * env.scale  # [8, window]
        s_blk = attention_scores(q_head, block_tile[j]) * env.scale  # [8, 16]
        s_ctx = jnp.where(context_allowed[j], s_ctx, NEG)
        s_blk = jnp.where(block_allowed, s_blk, NEG)
        maximum = jnp.maximum(
            jnp.max(s_ctx, axis=1, keepdims=True),
            jnp.max(s_blk, axis=1, keepdims=True),
        )
        p_ctx = jnp.exp(s_ctx - maximum)
        p_blk = jnp.exp(s_blk - maximum)
        denominator = jnp.sum(p_ctx, axis=1, keepdims=True) + jnp.sum(
            p_blk, axis=1, keepdims=True
        )
        inverse = 1.0 / denominator
        out = dot(bf16(p_ctx * inverse), ring_tile) + dot(
            bf16(p_blk * inverse), block_tile[j]
        )
        attended_j = attended_j + dot(bf16(out), env.small["head_place"][head])
    item = env.copy(
        env.ring_vmem.at[buf_idx],
        env.cache_out.at[j, layer],
        env.sem_ring + 2 + buf_idx,
    )
    item.start()
    env.ring["out"][buf_idx] = item
    env.ring["done"] += 1
    if visit + 2 < env.layers * env.num_requests:
        env.ring_start_in(visit + 2)
    return attended_j


def _dspark_run_layer(
    env: _DsparkStepEnv,
    layer: int,
    x: jax.Array,
    slot0_bias: jax.Array | None,
    layer_ctx: tuple[
        jax.Array,
        list[list[jax.Array]],
        jax.Array,
        jax.Array,
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        jax.Array,
        _AsyncCopy,
    ],
) -> tuple[jax.Array, jax.Array | None]:
    """Execute one Qwen3-style DSpark draft layer across all requests."""
    (
        context,
        new_slots,
        context_allowed,
        block_allowed,
        ropes,
        anchor_token,
        w2t_copy,
    ) = layer_ctx
    cos_q, sin_q, cos_kv, sin_kv, cos_new, sin_new = ropes
    h = tile_rms_norm(x, env.small["ln1"][layer], env.eps)
    # the block's own K/V (kv weights of this layer)
    block_kv = env.kv_rows(h, layer, cos_kv, sin_kv)
    env.prefetch_next("kv", layer)
    env.pending["q"].wait()
    q = bf16(dot(h, env.q_buf[...]))
    env.prefetch_next("q", layer)
    sumsq = _grouped_sumsq(f32(q), env.small["head_ones"][...])
    normed = f32(q) * jax.lax.rsqrt(sumsq * (1.0 / env.c.head_dim) + env.eps)
    q = bf16(f32(bf16(normed)) * f32(env.small["qnorm"][layer]))
    q = _rope(q, env.small["rot_q"][...], cos_q, sin_q)

    block_tile = jnp.pad(
        block_kv.reshape(env.num_requests, env.rows, env.kv_width),
        ((0, 0), (0, 16 - env.rows), (0, 0)),
    )
    attn_ctx = (block_tile, new_slots, context_allowed, block_allowed)
    # One ring visit per request: its slice in, new rows, attention, slice back out; the
    # next visit on the buffer streams in behind.
    attended_rows = [
        _dspark_attend_request(env, layer, j, q, attn_ctx)
        for j in range(env.num_requests)
    ]
    attended = bf16(jnp.concatenate(attended_rows, axis=0))

    env.pending["o"].wait()
    handles = env.all_reduce_start(  # [T, 7168] partial, chunked sends
        attended,
        env.o_buf,
        1,
        refill=env.weights["o"].at[layer],
    )
    env.prefetch_next("o", layer)
    if layer == 0:
        for item in env.w1_drain(0):
            item.wait()
        w2t_copy.wait()
        slot0_bias = dot(env.w1_rows_for(anchor_token, 0), env.w2t_buf[...])
    x = bf16(f32(x) + env.all_reduce_finish(handles, 1))

    h2 = tile_rms_norm(x, env.small["ln2"][layer], env.eps)
    env.pending["gate"].wait()
    gate = bf16(dot(h2, env.gate_buf[...]))  # [T, 512], zero beyond 448
    env.pending["up"].wait()
    up = bf16(dot(h2, env.up_buf[...]))
    env.prefetch_next("gate", layer)
    env.prefetch_next("up", layer)
    gate32 = f32(gate)
    activation = bf16(f32(bf16(gate32 * jax.nn.sigmoid(gate32))) * f32(up))
    env.pending["down"].wait()
    handles = env.all_reduce_start(
        activation, env.down_buf, 0, refill=env.weights["down"].at[layer]
    )
    if layer + 1 < env.layers:
        env.pending["down"] = env.layer_copy("down", layer + 1)
        env.pending["down"].start()
        env.pending["kv"].wait()
        env.pending["kv"] = None
        env.newkv_buf[...] = env.kv_rows(context, layer + 1, cos_new, sin_new)
    x = bf16(f32(x) + env.all_reduce_finish(handles, 0))
    return x, slot0_bias


def _dspark_lm_and_markov_sampling(
    env: _DsparkStepEnv,
    x: jax.Array,
    anchor_token: jax.Array,
    slot0_bias: jax.Array,
    positions_copy: _AsyncCopy,
) -> None:
    """Run vocabulary-sharded LM head, exact Markov sampling, and output DMAs."""
    hidden_rows = tile_rms_norm(x, env.small["final_norm"][...], env.eps)  # [T, 7168]
    chunk_buffers = (
        env.a_buf.at[pl.ds(0, env.hidden), pl.ds(0, LM_CHUNK)],
        env.gate_buf,
        env.up_buf,
    )
    chunk_sems = (0, 5, 6)

    def chunk_copy(chunk: int) -> _AsyncCopy:
        """Start prefetching LM head vocabulary chunk into its cyclic buffer."""
        return env.copy(
            env.weights["lm_head"].at[chunk],
            chunk_buffers[chunk % 3],
            chunk_sems[chunk % 3],
        )

    chunk_copies = [chunk_copy(chunk) for chunk in range(3)]
    for item in chunk_copies:
        item.start()
    vocab_lane = jax.lax.broadcasted_iota(jnp.int32, (1, env.vocab), 1)
    for chunk in range(LM_CHUNKS):
        chunk_copies[chunk % 3].wait()
        # BF16-rounded like the reference
        logits = f32(bf16(dot(hidden_rows, chunk_buffers[chunk % 3][...])))
        if chunk + 3 < LM_CHUNKS:
            chunk_copies[chunk % 3] = chunk_copy(chunk + 3)
            chunk_copies[chunk % 3].start()
        env.logits_vmem[:, :, pl.ds(chunk * LM_CHUNK, LM_CHUNK)] = logits.reshape(
            env.num_requests, env.rows, LM_CHUNK
        )

    tokens = env.tokens_v[...]
    token_lane = jax.lax.broadcasted_iota(jnp.int32, (env.num_requests, 128), 1)
    rank_iota = jax.lax.broadcasted_iota(jnp.int32, (RANKS, env.num_requests, 128), 0)
    previous = anchor_token
    for slot in range(env.block):
        if slot == 0:
            bias = slot0_bias  # [R, 5120] FP32
        else:
            for item in env.w1_drain(slot % 2):
                item.wait()
            bias = dot(env.w1_rows_for(previous, slot % 2), env.w2t_buf[...])
        scores = (
            jnp.concatenate(
                [
                    env.logits_vmem[j, slot : slot + 1, :]
                    for j in range(env.num_requests)
                ],
                axis=0,
            )
            + bias
        )
        best_value = jnp.max(scores, axis=1, keepdims=True)  # [R, 1]
        best_id = jnp.min(
            jnp.where(scores == best_value, vocab_lane, env.vocab),
            axis=1,
            keepdims=True,
        )
        local_max_bits = jax.lax.bitcast_convert_type(best_value, jnp.int32)  # [R, 1]
        global_id = best_id + env.rank * env.vocab
        row = jnp.where(
            token_lane == 0,
            local_max_bits,
            jnp.where(token_lane == 1, global_id, 0),
        )
        gathered = env.tiny_exchange(row, slot % 2)  # [32, R, 128] int32
        # Note Shape [32, R, 1]
        maxima = jax.lax.bitcast_convert_type(gathered[:, :, 0:1], jnp.float32)
        best = jnp.max(maxima, axis=0, keepdims=True)  # [1, R, 1]
        winner = jnp.min(
            jnp.where(maxima == best, rank_iota[:, :, 0:1], RANKS),
            axis=0,
            keepdims=True,
        )
        # [R, 128]; lane 1 = id
        token_row = jnp.sum(jnp.where(rank_iota == winner, gathered, 0), axis=0)
        picked_token = token_row[:, 1:2]
        tokens = jnp.where(token_lane == OUT_DRAFTS + slot, picked_token, tokens)
        previous = picked_token[:, 0]
        if slot + 1 < env.block:
            env.start_w1(previous, (slot + 1) % 2)
    env.tokens_v[...] = tokens

    tokens_copy = tpu.make_async_copy(
        env.tokens_v, env.tokens_out, env.small_sems.at[0]
    )
    tokens_copy.start()
    env.ring_drain()
    tokens_copy.wait()
    positions_copy.wait()


def _make_body(
    c: DraftConfig, block: int, eps: float, requests: int = 1
) -> Callable[..., None]:
    """Build the Pallas kernel body for the Qwen3-style DSpark fused draft.

    One call covers ``requests`` sequences: their block rows fold into the row
    axis of every projection, so the weights cross HBM to VMEM once per call,
    while each request keeps its own ring, positions, anchor and sampling chain.
    The rings stay in HBM (the per-request rings do not fit in VMEM): (layer,
    request) slices stream through two buffers, each updated, attended and
    written back. From eight requests the FC stages one request's aux features.
    The cross-host partial sums travel as BF16 pairs.
    """
    layers = c.layers
    window = c.window
    num_requests = requests
    rows = _fused_rows(requests, block)
    total_rows = num_requests * rows
    stream_aux = num_requests >= _WIDE_REQUESTS
    kv_width = 2 * c.head_dim
    q_width = 3 * c.head_dim
    vocab = c.vocab // RANKS
    hidden = c.hidden
    scale = c.head_dim**-0.5

    def body(
        scalars: jax.Array,
        aux_hbm: jax.Array,
        positions_hbm: jax.Array,
        logits_hbm: jax.Array,
        previous_hbm: jax.Array,
        *refs: jax.Array,
    ) -> None:
        """Execute one fused verify-acceptance and speculative draft block step."""
        env = _DsparkStepEnv(
            c=c,
            block=block,
            eps=eps,
            layers=layers,
            window=window,
            num_requests=num_requests,
            rows=rows,
            total_rows=total_rows,
            stream_aux=stream_aux,
            kv_width=kv_width,
            q_width=q_width,
            vocab=vocab,
            hidden=hidden,
            scale=scale,
            sem_fc=1,
            sem_w2t=8,
            sem_own=9,
            sem_ring=10 + 6 * num_requests,
        )  # [R, 8, 128] int32
        env.bind_refs(scalars, aux_hbm, positions_hbm, logits_hbm, (previous_hbm, refs))

        # ---------------- prologue: barrier and DMA issue ----------------
        collectives32.barrier()
        small_sources = {
            "aux": aux_hbm,
            "positions": positions_hbm,
            "previous": previous_hbm,
            **{name: env.weights[name] for name in VMEM_NAMES},
        }
        small_copies = [
            tpu.make_async_copy(
                small_sources[name], env.small[name], env.small_sems.at[index]
            )  # [R, 8, 1]: proposal k was verify row k + 1
            for index, name in enumerate(SMALL_NAMES)
            if name != "aux" or not stream_aux
        ]
        # ---------------- verify argmax, acceptance, anchor ----------------
        for item in small_copies:
            item.start()
        # The verify logits land in the LM head's logits buffer (dead by then).
        logits_copy = env.copy(logits_hbm, env.logits_vmem, 0)
        logits_copy.start()
        # This rank's [FC_IN, 896] fc columns, into 896 of a_buf's 1024 lanes.
        fc_copy = env.copy(
            env.weights["fc"], env.a_buf.at[:, pl.ds(0, CHUNK)], env.sem_fc
        )  # [R, window]
        fc_copy.start()
        env.ring_start_in(0)
        env.ring_start_in(1)

        anchor_token, anchor_position, owner, rope_copies = (
            _dspark_accept_and_issue_reads(env, small_copies, logits_copy)
        )  # [R, 256]
        (
            x,
            context,
            new_slots,
            context_allowed,
            block_allowed,
            ropes,
            w2t_copy,
            positions_copy,
        ) = _dspark_context_and_masks(env, fc_copy, anchor_position, owner, rope_copies)
        layer_ctx = (
            context,
            new_slots,
            context_allowed,
            block_allowed,
            ropes,
            anchor_token,
            w2t_copy,
        )
        # Slot-0 Markov bias needs only the anchors' w1 rows: computed during a
        # collective wait below and consumed by the sampler.
        slot0_bias: jax.Array | None = None
        for layer in range(layers):
            # Note Shape [R, 8, 7168]
            x, slot0_bias = _dspark_run_layer(env, layer, x, slot0_bias, layer_ctx)
        bias0 = (
            slot0_bias
            if slot0_bias is not None
            else jnp.zeros((num_requests, vocab), jnp.float32)
        )
        _dspark_lm_and_markov_sampling(env, x, anchor_token, bias0, positions_copy)

    return body


def make_fused_draft(
    c: DraftConfig,
    mesh: jax.sharding.Mesh,
    block: int,
    *,
    eps: float | None = None,
    requests: int = 1,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    """Build the JIT-compiled TP32 Pallas fused draft callable.

    ``(weights, cache, positions, aux, logits, previous, positions0,
    forced_counts[, forced_anchors])`` -> ``(vectors [R, 128] int32, cache,
    positions)``.

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
    ``forced_anchors >= 0`` (sampled outside the kernel). Output lanes:
    ``OUT_*``.
    """
    rows = _fused_rows(requests, block)
    if mesh.size != RANKS:
        raise ValueError("the fused draft needs the 32-rank mesh")
    eps = c.eps if eps is None else eps
    window, hidden = c.window, c.hidden
    num_requests, total_rows = requests, requests * rows
    packed = total_rows * CHUNK // 256  # FP32 rows holding one BF16 [T, 896] chunk
    body = _make_body(c, block, eps, requests)

    def local(
        weights: dict[str, jax.Array],
        cache: jax.Array,
        positions: jax.Array,
        aux: jax.Array,
        step: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Run the per-rank shard_map wrapper around the Pallas fused draft call."""
        logits, previous, position, forced_count, forced_anchor = step
        weights = {name: weights[name][0] for name in WEIGHT_NAMES}
        if num_requests == 1:
            # The single-request operands carry no request axis.
            cache = cache[0][None]
            logits = logits[0][None].astype(jnp.float32)
        else:
            cache = cache[0]
            positions = positions[0]
            logits = logits[0].astype(jnp.float32)
            previous = previous[0]
        if rows < ROWS:
            aux = aux.reshape(aux.shape[0], num_requests, ROWS, aux.shape[-1])[
                :, :, :rows, :
            ]
            aux = aux.reshape(aux.shape[0], total_rows, aux.shape[-1])
            logits = logits[:, :rows, :]
        # Note Shape [R, layers, window, 128], [R, window], [R, rows, vocab], [R, 128]
        scalars = (  # [3R]
            jnp.stack((position, forced_count, forced_anchor))
            .reshape(3 * num_requests)
            .astype(jnp.int32)
        )
        small_shapes: dict[str, Any] = {
            "aux": aux,
            "positions": positions,
            "logits": logits,
            "previous": previous,
            **weights,
        }
        if num_requests >= _WIDE_REQUESTS:
            small_shapes["aux"] = jax.ShapeDtypeStruct(
                (aux.shape[0], rows, aux.shape[-1]), aux.dtype
            )
        in_specs = [
            pl.BlockSpec(memory_space=tpu.HBM) for _ in range(4 + len(WEIGHT_NAMES) + 1)
        ]
        out_specs = [pl.BlockSpec(memory_space=tpu.HBM) for _ in range(3)]
        scratch = (
            *(
                tpu.VMEM(small_shapes[name].shape, small_shapes[name].dtype)
                for name in SMALL_NAMES
            ),
            # a_buf: fc columns in 896 of 1024 lanes, then projection/LM scratch
            tpu.VMEM((FC_IN, 1024), jnp.bfloat16),
            tpu.VMEM((hidden, 3 * c.head_dim), jnp.bfloat16),  # q_buf
            tpu.VMEM((hidden, 2 * c.head_dim), jnp.bfloat16),  # kv_buf
            tpu.VMEM(
                (
                    3 * c.head_dim,
                    CHUNK if num_requests > _WIDE_MAX_BLOCK else hidden,
                ),
                jnp.bfloat16,
            ),
            # gate_buf: MLP lanes 0..447, or an LM chunk
            tpu.VMEM((hidden, LM_CHUNK), jnp.bfloat16),
            tpu.VMEM((hidden, LM_CHUNK), jnp.bfloat16),  # up_buf
            tpu.VMEM(
                (MLP_PADDED, CHUNK if num_requests > _WIDE_MAX_BLOCK else hidden),
                jnp.bfloat16,
            ),
            # Note w2t_buf [256, 5120]
            tpu.VMEM((c.markov_rank, c.vocab // RANKS), jnp.bfloat16),
            tpu.VMEM((2, num_requests, 16, c.markov_rank), jnp.bfloat16),
            tpu.VMEM((16, hidden), jnp.bfloat16),  # embed_blk
            tpu.VMEM((num_requests, ROW_PACKED, 128), jnp.float32),
            tpu.VMEM((num_requests, 16, 6 * c.head_dim), jnp.float32),
            tpu.VMEM((2, num_requests, 16, 4 * c.head_dim), jnp.float32),
            # ring_vmem: the ring slice stream, double-buffered
            tpu.VMEM((2, window, 2 * c.head_dim), jnp.bfloat16),
            tpu.VMEM((num_requests, rows, c.vocab // RANKS), jnp.float32),
            tpu.VMEM((total_rows, CHUNK), jnp.float32),
            tpu.VMEM((num_requests, 128), jnp.int32),
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # stage
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # local_recv
            tpu.VMEM((2, 4, packed, 128), jnp.float32),  # host_packed
            tpu.VMEM((2, 8, packed, 128), jnp.float32),  # gather_recv
            tpu.VMEM((2, RANKS, num_requests, 128), jnp.int32),
            tpu.VMEM((RANKS, total_rows, 128), jnp.int32),
            tpu.VMEM((total_rows, 2 * c.head_dim), jnp.bfloat16),
            tpu.SemaphoreType.DMA((10 + 6 * num_requests + 4,)),
            tpu.SemaphoreType.DMA((len(SMALL_NAMES),)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
        )
        # scalars, aux, positions, logits, previous, weights
        cache_index = 1 + 4 + len(WEIGHT_NAMES)
        tokens, positions_out, cache_out = pl.pallas_call(
            body,
            name="dspark_fused_draft",
            out_shape=(
                jax.ShapeDtypeStruct((num_requests, 128), jnp.int32),
                jax.ShapeDtypeStruct((num_requests, window), jnp.int32),
                jax.ShapeDtypeStruct(cache.shape, cache.dtype),
            ),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=scratch,
            ),
            input_output_aliases={cache_index: 2},
            compiler_params=tpu.CompilerParams(
                collective_id=25, vmem_limit_bytes=64 * 1024**2
            ),
        )(
            scalars,
            aux,
            positions,
            logits,
            previous,
            *[weights[name] for name in WEIGHT_NAMES],
            cache,
        )
        if num_requests == 1:
            return tokens, cache_out[0][None], positions_out
        return tokens, cache_out[None], positions_out[None]

    program = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                PartitionSpec("tp"),
                PartitionSpec("tp"),
                PartitionSpec(),
                PartitionSpec(),
                (
                    PartitionSpec("tp"),
                    PartitionSpec(),
                    PartitionSpec(),
                    PartitionSpec(),
                    PartitionSpec(),
                ),
            ),
            out_specs=(PartitionSpec(), PartitionSpec("tp"), PartitionSpec()),
            check_vma=False,
        ),
        donate_argnums=(1,),
    )

    def fused_draft(
        weights: dict[str, jax.Array],
        cache: jax.Array,
        positions: jax.Array,
        aux: jax.Array,
        logits: jax.Array,
        previous: jax.Array,
        position: jax.Array,
        forced_count: jax.Array,
        forced_anchor: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Invoke the compiled TP32 DSpark fused draft program."""
        if forced_anchor is None:
            forced_anchor = jnp.full_like(position, -1)
        return program(
            weights,
            cache,
            positions,
            aux,
            (logits, previous, position, forced_count, forced_anchor),
        )

    return fused_draft


# --------------------------------------------------------------------------
# The ring from a prompt (prefill side)

# -------------------------------------------------------------------------- The ring
# from a prompt (prefill side) Each sequence-parallel prefill rank projects its own
# window rows for every KV head (replicated weights, :func:`ring_weights_from_shards`);
# one all-to-all hands rank r the rows of KV head ``r // (32 / kv_heads)``, stored at
# ring slot ``position % window`` in the fused kernel's packed K | V layout. A ring
# travels with the KDA state in ``RING_BLOCK_ROWS`` spare rows of each of a request's 69
# state blocks, behind a header (``RING_MAGIC``, length L, first position F); it holds
# position p at slot p % window for max(L - window, F) <= p < L.
RING_MAGIC = 0x44535231  # "DSR1"
# spare state-block rows per block: 69 * 15 rows hold a ring and its header
RING_BLOCK_ROWS = 15
RING_ROW_LANES = 1280  # bf16 per 2560-byte pool row


def _header_bytes(values: jax.Array) -> jax.Array:
    """Encode int32 ``[..., k]`` into bf16 ``[..., 4 k]`` as exact byte values.

    Bit-cast halves would include bf16 subnormals, which the TPU flushes to
    zero when it moves them.
    """
    shifts = jnp.asarray([0, 8, 16, 24], jnp.int32)
    parts = jnp.right_shift(values[..., None], shifts) & 0xFF
    return parts.astype(jnp.bfloat16).reshape(*values.shape[:-1], -1)


def _header_ints(parts: jax.Array) -> jax.Array:
    """Inverse of :func:`_header_bytes` on ``[..., k, 4]``."""
    shifts = jnp.asarray([0, 8, 16, 24], jnp.int32)
    return jnp.sum(jnp.left_shift(parts.astype(jnp.int32) & 0xFF, shifts), axis=-1)


def ring_rows(c: DraftConfig) -> int:
    """Pool rows a ring's K | V values take (``[layers, window, 2 head_dim]`` bf16)."""
    return -(-c.layers * c.window * 2 * c.head_dim // RING_ROW_LANES)


def pack_ring(ring: jax.Array, meta: jax.Array, blocks: int) -> jax.Array:
    """Pack a context ring and metadata header into spare KDA state pool rows.

    ``ring`` ``[n, layers, window, 2 head_dim]`` bf16 and ``meta`` ``[n, 2]``
    int32 (length, first position) -> ``[n, blocks, RING_BLOCK_ROWS,
    RING_ROW_LANES]`` bf16: the ring's rows, then a header row, zero padded.
    """
    n, size = ring.shape[0], int(np.prod(ring.shape[1:]))
    flat = jnp.pad(ring.reshape(n, size), ((0, 0), (0, -size % RING_ROW_LANES)))
    rows = flat.reshape(n, -1, RING_ROW_LANES)
    header = jnp.concatenate(
        (jnp.full((n, 1), RING_MAGIC, jnp.int32), meta.astype(jnp.int32)), axis=1
    )
    header = _header_bytes(header).reshape(n, 1, -1)
    header = jnp.pad(header, ((0, 0), (0, 0), (0, RING_ROW_LANES - header.shape[-1])))
    total = blocks * RING_BLOCK_ROWS
    if rows.shape[1] + 1 > total:
        raise ValueError(
            f"a ring needs {rows.shape[1] + 1} rows, {blocks} blocks hold {total}"
        )
    packed = jnp.concatenate((rows, header), axis=1)
    packed = jnp.pad(packed, ((0, 0), (0, total - packed.shape[1]), (0, 0)))
    # The TPU compiler cannot fuse these pads with a reshape that splits the padded
    # dimensions (as the pool writes do): keep them apart.
    packed = jax.lax.optimization_barrier(packed)
    return packed.reshape(n, blocks, RING_BLOCK_ROWS, RING_ROW_LANES)


def unpack_ring(
    c: DraftConfig,
    packed: jax.Array,
    length: int | jax.Array,
    cap: int | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Unpack one context ring and metadata header read back from the state pool.

    Inverse of :func:`pack_ring` for one ring read back from the pool:
    ``packed`` ``[blocks, RING_BLOCK_ROWS, RING_ROW_LANES]`` bf16 -> ``(ring
    [layers, window, 2 head_dim], meta [2])``. A ring whose header is not
    ``RING_MAGIC`` at ``length`` is empty: meta ``(length, length)``.
    ``cap``: raise the ring's first position to at least this (a prefix hit
    keeps what the ring held before it).
    """
    rows = packed.reshape(-1, RING_ROW_LANES)
    count, shape = ring_rows(c), (c.layers, c.window, 2 * c.head_dim)
    ring = rows[:count].reshape(-1)[: int(np.prod(shape))].reshape(shape)
    header = _header_ints(rows[count, : 4 * 3].reshape(3, 4))
    valid = (header[0] == RING_MAGIC) & (header[1] == length)
    first = jnp.where(valid, header[2], length)
    if cap is not None:
        first = jnp.maximum(first, cap)
    return ring, jnp.stack((jnp.asarray(length, jnp.int32), first.astype(jnp.int32)))


def ring_positions(c: DraftConfig, meta: jax.Array) -> jax.Array:
    """The ring's slot positions ``[window]`` int32 (``-1`` empty) from its meta."""
    length, first = meta[0], meta[1]
    slot = jnp.arange(c.window, dtype=jnp.int32)
    position = length - 1 - jnp.mod(length - 1 - slot, c.window)
    return jnp.where(
        (position >= 0) & (position >= first) & (position >= length - c.window),
        position,
        -1,
    )


def ring_weights_from_shards(
    c: DraftConfig,
    fc: jax.Array,
    kv: jax.Array,
    kvnorm: jax.Array,
    hidden_norm: jax.Array,
    axis: str = AXIS_TP,
) -> dict[str, jax.Array]:
    """All-gather and reshape TP32 weight shards into replicated ring-update weights.

    (inside a TP32 ``shard_map`` over ``axis``) this rank's kernel shards
    (:func:`kernel_weight_shards`: ``fc`` ``[FC_IN, CHUNK]``, ``kv`` ``[layers,
    hidden, 2 head_dim]``, ``kvnorm`` ``[layers, 1, 2 head_dim]``,
    ``hidden_norm`` ``[1, hidden]``) -> the replicated weights the prefill
    ring update takes: ``fc`` ``[context_width, hidden]``, ``kv`` ``[layers,
    hidden, kv_heads, 2 head_dim]``, ``k_norm`` ``[layers, head_dim]``,
    ``hidden_norm`` ``[hidden]``.
    """
    hosts, lanes = c.context_width // fc.shape[0], c.hidden // fc.shape[1]
    # [32, in / hosts, out / lanes], rank = host * lanes + lane
    full_fc = jax.lax.all_gather(fc, axis)
    full_fc = (
        full_fc.reshape(hosts, lanes, *fc.shape)
        .transpose(0, 2, 1, 3)
        .reshape(c.context_width, c.hidden)
    )
    share = RANKS // c.kv_heads
    # [kv_heads, layers, hidden, 2 head_dim]
    heads = jax.lax.all_gather(kv, axis)[::share]
    return {
        "fc": full_fc,
        "kv": heads.transpose(1, 2, 0, 3),
        "k_norm": kvnorm[:, 0, : c.head_dim],
        "hidden_norm": hidden_norm[0],
    }


def prefill_ring_update(
    c: DraftConfig,
    weights: dict[str, jax.Array],
    aux: jax.Array,
    positions: jax.Array,
    sequences: jax.Array,
    rings: jax.Array,
    ends: jax.Array,
    stops: jax.Array,
    axis: str = AXIS_TP,
) -> tuple[jax.Array, jax.Array]:
    """Update sequence context rings from one packed prefill window inside shard_map.

    ``aux`` ``[len(aux_layers), rows, hidden]`` bf16 is this rank's window rows
    of the residual streams entering the aux layers; ``positions`` /
    ``sequences`` ``[32 * rows]`` int32 give every window row's position and
    sequence (``-1``: padding), the ranks' rows in rank-major order; ``rings``
    ``[slots, layers, window, 2 head_dim]`` this rank's head of each sequence's
    ring; ``ends`` / ``stops`` ``[slots]`` each sequence's end in the window and
    its checkpoint (``-1``: none). Returns ``(rings after the window, rings at
    the checkpoints)``.
    """
    rows, hd = aux.shape[1], c.head_dim
    rank = jax.lax.axis_index(axis)
    features = aux.transpose(1, 0, 2).reshape(rows, c.context_width)
    context = jnp.einsum(
        "ri,io->ro", features, weights["fc"], preferred_element_type=jnp.float32
    )
    context = rms_norm(context.astype(jnp.bfloat16), weights["hidden_norm"], c.eps)
    # RoPE on the K lanes of each packed K | V row: cos | 1 and sin | 0 over the 2
    # head_dim lanes, rotate-half as a +-1 matrix (exact), as the fused kernel's
    # ``rope_kv`` tables and ``rot_kv`` do -- no lane concatenations (the TPU compiler
    # cannot split padded dims they leave behind).
    local = jnp.maximum(jax.lax.dynamic_slice_in_dim(positions, rank * rows, rows), 0)
    inv_freq = 1.0 / c.rope_theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
    lane_freq = np.concatenate((inv_freq, inv_freq, np.zeros(hd, np.float32)))
    k_lane = np.arange(2 * hd) < hd
    angle = local.astype(jnp.float32)[:, None] * jnp.asarray(lane_freq)[None, :]
    cos = jnp.where(k_lane, jnp.cos(angle), 1.0).astype(jnp.bfloat16)[:, None, :]
    sin = jnp.where(k_lane, jnp.sin(angle), 0.0).astype(jnp.bfloat16)[:, None, :]
    rotation_arr = np.zeros((2 * hd, 2 * hd), np.float32)
    rotation_arr[:hd, :hd] = _rotation(hd)
    rotation = jnp.asarray(rotation_arr, jnp.bfloat16)
    k_lanes = jnp.asarray(k_lane)
    packed_layers = []
    for layer in range(c.layers):
        kv = jnp.einsum(
            "rh,hkd->rkd",
            context,
            weights["kv"][layer],
            preferred_element_type=jnp.float32,
        )
        kv = kv.astype(jnp.bfloat16)  # [rows, kv_heads, 2 head_dim]
        k32 = jnp.where(k_lanes, kv.astype(jnp.float32), 0.0)
        normed = kv.astype(jnp.float32) * jax.lax.rsqrt(
            jnp.sum(k32 * k32, axis=-1, keepdims=True) / hd + c.eps
        )
        k_weight = jnp.pad(weights["k_norm"][layer], (0, hd))
        k_scaled = (normed.astype(jnp.bfloat16) * k_weight).astype(jnp.bfloat16)
        kv = jnp.where(k_lanes, k_scaled, kv)
        rotated = jnp.einsum(
            "rkd,de->rke", kv, rotation, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        packed_layers.append(kv * cos + rotated * sin)
    packed = jnp.stack(packed_layers)  # [layers, rows, kv_heads, 2 head_dim]
    share = RANKS // c.kv_heads
    # Destination rank d gets KV head d // share: repeat each head, then destinations
    # lead for the all-to-all.
    send = jnp.broadcast_to(packed[:, :, :, None], (*packed.shape[:3], share, 2 * hd))
    send = send.reshape(c.layers, rows, RANKS, 2 * hd).transpose(2, 0, 1, 3)
    mine = jax.lax.all_to_all(send, axis, 0, 0)  # [32 sources, layers, rows, 2hd]
    mine = mine.transpose(1, 0, 2, 3).reshape(c.layers, RANKS * rows, 2 * hd)
    slot = jnp.mod(positions, c.window)

    def scatter(ring: jax.Array, keep: jax.Array) -> jax.Array:
        """Scatter selected window rows into the sequence ring slots."""
        index = jnp.where(keep, slot, c.window)  # out of range: dropped
        return ring.at[:, index].set(mine, mode="drop")

    after, at = [], []
    for s in range(rings.shape[0]):
        member = sequences == s
        after.append(scatter(rings[s], member & (positions >= ends[s] - c.window)))
        at.append(
            scatter(
                rings[s],
                member & (positions < stops[s]) & (positions >= stops[s] - c.window),
            )
        )
    return jnp.stack(after), jnp.stack(at)
