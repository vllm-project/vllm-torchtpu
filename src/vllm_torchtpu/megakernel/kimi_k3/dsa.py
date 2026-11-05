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
"""Kimi-K3.1's DeepSeek sparse attention (DSA) on the K3 megakernels.

K3.1 is K3 with an indexer on every MLA layer: each query scores the cached
tokens with ``sum_h w_h ReLU(q_h . k_t)`` over 32 heads of 128 lanes and
attends only the ``index_topk`` best. This module holds the model
configuration and the synthetic indexer weights the tools load in place of a
pre-sharded checkpoint; it needs no JAX (the rank loader imports it).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vllm_torchtpu.megakernel.kimi_k3.layout import MLA

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 2048  # contexts up to the top-k select every token
# The only dsa_config values the kernels implement. A layer reuses an earlier
# layer's selection when max(layer_id - index_skip_topk_offset + 1, 0) %
# index_topk_freq != 0, or, given an index_topk_pattern, where the pattern says
# "S" (vLLM's DeepSeek-V2 attention): at frequency 1 without a pattern every
# layer selects its own tokens.
_SUPPORTED = {
    "index_num_heads": INDEX_HEADS,
    "index_head_dim": INDEX_DIM,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "indexer_use_fp8": False,
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
    "index_topk_pattern": None,
    "index_topk": INDEX_TOPK,
}


@dataclasses.dataclass(frozen=True)
class DsaConfig:
    index_topk: int = INDEX_TOPK
    norm_eps: float = 1e-5  # the key LayerNorm's (the model's rms_norm_eps)

    @classmethod
    def from_text_config(cls, text: Any) -> DsaConfig | None:
        """The text config's ``dsa_config`` (None: Kimi-K3, no indexer)."""

        def get(key: str, default: Any = None) -> Any:
            if isinstance(text, dict):
                return text.get(key, default)
            return getattr(text, key, default)

        config = get("dsa_config")
        if config is None:
            return None
        config = dict(config) if isinstance(config, dict) else dict(vars(config))
        for key, value in _SUPPORTED.items():
            if config.get(key, value) != value:
                raise NotImplementedError(
                    f"K3.1 DSA supports {key}={value!r}, not {config.get(key)!r}"
                )
        layers = config.get("dsa_layers")
        full = (get("linear_attn_config") or {}).get("full_attn_layers")
        if layers is not None and full is not None and list(layers) != list(full):
            raise NotImplementedError("K3.1 DSA must cover exactly the MLA layers")
        return cls(
            index_topk=int(config.get("index_topk", INDEX_TOPK)),
            norm_eps=float(get("rms_norm_eps", 1e-5)),
        )


def load_dsa_config(path: str | Path) -> DsaConfig | None:
    """The DSA configuration of a model directory (or its ``config.json``)."""
    path = Path(path)
    config = json.loads((path / "config.json" if path.is_dir() else path).read_text())
    return DsaConfig.from_text_config(config.get("text_config", config))


# Synthetic indexers for kernel work before the K3.1 checkpoint is pre-sharded:
# "random" draws wk and the key LayerNorm from a seed; "rope-probe" makes the
# key projection [rope, -rope] of the MLA kv_a projection's rope columns (the
# latent row stores them unnormalized), with a unit-weight, zero-bias LayerNorm,
# so a key is a known function of its committed latent row. Both modes draw the
# head-sharded wq_b heads and weights_proj rows from the seed.
SYNTHETIC_INDEXERS = ("random", "rope-probe")
# m_knorm lanes ahead of a head's weights_proj row: the KV LayerNorm (512),
# then the index key LayerNorm's weight and bias (128 each). Under attention
# DP m_knorm holds every held shard's row, shard ``k`` (TP32 rank ``rank %
# (32 / dp) + k * (32 / dp)``) from lane ``INDEX_ROW_LANE + k * hidden``.
INDEX_ROW_LANE = 768
_ROPE = slice(512, 576)  # kv_a's rope columns in m_ka


def synthetic_key_projection(seed: int, layer: int, hidden: int, dtype) -> np.ndarray:
    """``[hidden, 128]``: the random synthetic indexer's ``wk`` of one MLA
    layer, transposed (``x -> k``)."""
    draw = np.random.default_rng((seed, layer)).standard_normal(
        (hidden, INDEX_DIM), np.float32
    )
    return (draw / np.float32(np.sqrt(hidden))).astype(dtype)


def synthetic_key_norm(seed: int, layers: int, dtype) -> tuple[np.ndarray, np.ndarray]:
    """``(weight, bias)`` ``[layers, 1, 128]``: the random synthetic indexer's
    key LayerNorm."""
    rng = np.random.default_rng((seed, layers))
    shape = (layers, 1, INDEX_DIM)
    weight = (1 + 0.125 * rng.standard_normal(shape, np.float32)).astype(dtype)
    bias = (0.125 * rng.standard_normal(shape, np.float32)).astype(dtype)
    return weight, bias


def synthetic_query_heads(
    seed: int, layer: int, q_lora: int, dtype, heads=None
) -> np.ndarray:
    """``[q_lora, 128 len(heads)]``: synthetic ``wq_b`` index heads of one MLA
    layer, transposed (``q_c -> q_h``), in the order of ``heads`` (default all
    32), each N(0, 1 / q_lora) from its own generator: a rank loads its own
    head, and a check rebuilds them all."""
    heads = range(INDEX_HEADS) if heads is None else heads
    return np.concatenate(
        [
            (
                np.random.default_rng((seed, layer, head, 1)).standard_normal(
                    (q_lora, INDEX_DIM), np.float32
                )
                / np.float32(np.sqrt(q_lora))
            ).astype(dtype)
            for head in heads
        ],
        axis=1,
    )


def synthetic_weight_rows(
    seed: int, layer: int, hidden: int, dtype, heads=None
) -> np.ndarray:
    """``[len(heads), hidden]``: synthetic ``weights_proj`` rows of one MLA
    layer (``x -> w_h``), each N(0, 1 / hidden) from its own generator."""
    heads = range(INDEX_HEADS) if heads is None else heads
    return np.stack(
        [
            (
                np.random.default_rng((seed, layer, head, 2)).standard_normal(
                    hidden, np.float32
                )
                / np.float32(np.sqrt(hidden))
            ).astype(dtype)
            for head in heads
        ]
    )


def with_synthetic_indexer(families: dict, mode: str, seed: int, head: int = 0) -> dict:
    """``families`` with K3.1's synthetic indexer appended:

    - ``m_ka`` columns 640:768 hold the key projection (``wk`` transposed);
    - ``m_knorm`` lanes 512:640 and 640:768 hold the key LayerNorm's weight
      and bias (all three the same on every rank);
    - ``m_qb`` columns 768:896 hold the index-query head ``head`` (this
      rank's ``wq_b`` rows, transposed);
    - ``m_knorm`` lanes 768:768 + hidden hold that head's ``weights_proj``
      row."""
    if mode not in SYNTHETIC_INDEXERS:
        raise ValueError(
            f"synthetic indexer {mode!r} is not one of {SYNTHETIC_INDEXERS}"
        )
    m_ka, m_knorm, m_qb = families["m_ka"], families["m_knorm"], families["m_qb"]
    layers, hidden, _ = m_ka.shape
    norm_shape = (layers, 1, INDEX_DIM)
    if mode == "rope-probe":
        rope = m_ka[..., _ROPE]
        keys = np.concatenate((rope, -rope), axis=-1)
        weight = np.ones(norm_shape, m_knorm.dtype)
        bias = np.zeros(norm_shape, m_knorm.dtype)
    else:
        keys = np.stack(
            [
                synthetic_key_projection(seed, layer, hidden, m_ka.dtype)
                for layer in range(layers)
            ]
        )
        weight, bias = synthetic_key_norm(seed, layers, m_knorm.dtype)
    query = np.stack(
        [
            synthetic_query_heads(seed, layer, m_qb.shape[1], m_qb.dtype, heads=(head,))
            for layer in range(layers)
        ]
    )
    rows = np.stack(
        [
            synthetic_weight_rows(seed, layer, hidden, m_knorm.dtype, heads=(head,))
            for layer in range(layers)
        ]
    )
    return _with_indexer(families, keys, weight, bias, query, rows)


# The pre-shard's indexer families (``indexer.bin``), this rank's index head:
# ``index_wk`` [24, hidden, 128] (``wk`` transposed), ``index_knorm_weight``
# and ``index_knorm_bias`` [24, 1, 128], ``index_wq`` [24, q_lora, 128] (the
# rank's ``wq_b`` head rows, transposed) and ``index_wv`` [24, 1, hidden] (its
# ``weights_proj`` row).
PRESHARD_INDEXER = (
    "index_wk",
    "index_knorm_weight",
    "index_knorm_bias",
    "index_wq",
    "index_wv",
)

INDEXER_FILE = "indexer.bin"  # beside a rank's dense.bin (``preshard indexer``)
INDEXER_RANKS = INDEX_HEADS  # TP32: one index head per rank


def rank_indexer_values(
    wq_b, wk, weights_proj, k_norm_weight, k_norm_bias, rank: int
) -> dict:
    """One MLA layer's checkpoint indexer tensors as ``rank``'s families
    (without the layer axis): ``index_wk`` ``[hidden, 128]``, ``index_knorm_*``
    ``[1, 128]``, and head ``rank``'s ``index_wq`` ``[q_lora, 128]`` and
    ``index_wv`` ``[1, hidden]``."""
    return {
        "index_wk": np.ascontiguousarray(wk.T),
        "index_knorm_weight": k_norm_weight[None],
        "index_knorm_bias": k_norm_bias[None],
        "index_wq": np.ascontiguousarray(
            wq_b[rank * INDEX_DIM : (rank + 1) * INDEX_DIM].T
        ),
        "index_wv": weights_proj[rank : rank + 1],
    }


def presharded_has_indexer(presharded: str | Path, layers: int = 93) -> bool:
    """Whether the pre-shard at ``presharded`` carries Kimi-K3.1's indexer
    (``PRESHARD_INDEXER`` in rank 0's index; False without a pre-shard
    there). It reads no kernel config, so the engine config may ask before
    the kernel config is installed."""
    path = Path(presharded) / "rank00" / "index.json"
    if not path.is_file():
        return False
    index = json.loads(path.read_text())
    if index.get("rank") != 0 or index.get("layers") != layers:
        raise ValueError(f"unexpected pre-shard index in {path.parent}")
    return all(name in index["families"] for name in PRESHARD_INDEXER)


def with_presharded_indexer(families: dict) -> dict:
    """``families`` with the pre-shard's indexer families moved to where
    :func:`with_synthetic_indexer` puts its synthetic ones."""
    out = {k: v for k, v in families.items() if k not in PRESHARD_INDEXER}
    return _with_indexer(
        out,
        families["index_wk"],
        families["index_knorm_weight"],
        families["index_knorm_bias"],
        families["index_wq"],
        families["index_wv"],
    )


def _with_indexer(families: dict, keys, weight, bias, query, rows) -> dict:
    m_ka, m_knorm, m_qb = families["m_ka"], families["m_knorm"], families["m_qb"]
    out = dict(families)
    out["m_ka"] = np.ascontiguousarray(np.concatenate((m_ka, keys), axis=-1))
    out["m_qb"] = np.ascontiguousarray(np.concatenate((m_qb, query), axis=-1))
    out["m_knorm"] = np.ascontiguousarray(
        np.concatenate((m_knorm, weight, bias, rows), axis=-1)
    )
    return out


# The synthetic indexer families the prefill stages load and pass to their ops.
STAGE_DSA_FAMILIES = ("dsa_wk", "dsa_knorm", "dsa_wq", "dsa_wv")


def stage_synthetic_indexer(
    mode: str, seed: int, ordinals: Sequence[int], q_lora: int, hidden: int, dtype
) -> dict[str, np.ndarray]:
    """The synthetic indexer of the MLA layers ``ordinals`` (their global MLA
    indices) as the decoder draws it, stacked over the layers: ``dsa_wq``
    ``[L, q_lora, 32 * 128]`` (all heads, head-major columns), ``dsa_wv``
    ``[L, hidden, 32]``, ``dsa_wk`` ``[L, hidden, 128]`` and ``dsa_knorm``
    ``[L, 2, 128]`` (the key LayerNorm's weight, then its bias). Under
    rope-probe the stage keys the latent rows' rope lanes itself: ``dsa_wk``
    is zero and the LayerNorm unit-weight, zero-bias."""
    if mode not in SYNTHETIC_INDEXERS:
        raise ValueError(
            f"synthetic indexer {mode!r} is not one of {SYNTHETIC_INDEXERS}"
        )
    layers = len(ordinals)
    if mode == "rope-probe":
        keys = np.zeros((layers, hidden, INDEX_DIM), dtype)
        norm = np.stack(
            (np.ones((layers, INDEX_DIM), dtype), np.zeros((layers, INDEX_DIM), dtype)),
            axis=1,
        )
    else:
        keys = np.stack(
            [synthetic_key_projection(seed, m, hidden, dtype) for m in ordinals]
        )
        weight, bias = synthetic_key_norm(seed, len(MLA), dtype)
        index = list(ordinals)
        norm = np.concatenate((weight[index], bias[index]), axis=1)
    return {
        "dsa_wq": np.stack(
            [synthetic_query_heads(seed, m, q_lora, dtype) for m in ordinals]
        ),
        "dsa_wv": np.stack(
            [synthetic_weight_rows(seed, m, hidden, dtype).T for m in ordinals]
        ),
        "dsa_wk": np.ascontiguousarray(keys),
        "dsa_knorm": np.ascontiguousarray(norm),
    }



def stage_presharded_indexer(heads: Sequence[dict]) -> dict[str, np.ndarray]:
    """:func:`stage_synthetic_indexer`'s families from the pre-shard's
    (``PRESHARD_INDEXER``, as ``RankDir.read_indexer`` returns them) of
    every TP32 rank in order (rank ``r`` holds index head ``r``), for the
    same MLA layers: ``dsa_wq`` the heads' ``index_wq`` side by side,
    ``dsa_wv`` their ``index_wv`` rows as columns, ``dsa_wk`` and
    ``dsa_knorm`` (weight, then bias) the replicated families of rank 0."""
    if len(heads) != INDEX_HEADS:
        raise ValueError(f"the stage indexer takes {INDEX_HEADS} heads, not {len(heads)}")
    first = heads[0]
    return {
        "dsa_wq": np.ascontiguousarray(
            np.concatenate([h["index_wq"] for h in heads], axis=-1)
        ),
        "dsa_wv": np.ascontiguousarray(
            np.concatenate([h["index_wv"] for h in heads], axis=1).transpose(0, 2, 1)
        ),
        "dsa_wk": np.ascontiguousarray(first["index_wk"]),
        "dsa_knorm": np.ascontiguousarray(
            np.concatenate(
                (first["index_knorm_weight"], first["index_knorm_bias"]), axis=1
            )
        ),
    }

# Kimi-K3.1 decode takes the MLA norm rows as one operand: with the index-key
# cache, its plain op would otherwise exceed Torch's 64 operands.
QUERY_NORM_LANES = 1536  # m_qnorm: the q-down LoRA's
PACKED_NORMS = "m_norms"


def pack_mla_norms(families: dict) -> dict:
    """``families`` with ``m_qnorm`` and ``m_knorm`` as one family
    ``m_norms`` (``[layers, 1, 1536 + m_knorm lanes]``)."""
    out = dict(families)
    out[PACKED_NORMS] = np.ascontiguousarray(
        np.concatenate((out.pop("m_qnorm"), out.pop("m_knorm")), axis=-1)
    )
    return out


def unpack_mla_norms(weights: dict) -> dict:
    """:func:`pack_mla_norms` undone (NumPy or JAX arrays); weights without
    packed norms (Kimi-K3) are returned as they are."""
    if PACKED_NORMS not in weights:
        return weights
    out = dict(weights)
    norms = out.pop(PACKED_NORMS)
    out["m_qnorm"] = norms[..., :QUERY_NORM_LANES]
    out["m_knorm"] = norms[..., QUERY_NORM_LANES:]
    return out
