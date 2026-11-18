# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_PREFILL_SEQ_PARALLEL's model forward, no TPU needed: 16 TP ranks as
threads run a small DeepseekV2 stack (two dense layers, then MoE layers)
through upstream's DeepseekV2Model.forward and DeepseekV2DecoderLayer.forward,
and through prefill_token_split's split forward, and get the same hidden
states.

The rank threads exchange tensors through a barrier; the attention and MoE
stand-ins keep the contracts of the TPU MLA wrapper (`token_split`: project
the core's rows, all-gather the latents, reduce-scatter o_proj) and of
moe_tp_decode's split op (all-gather the rows, reduce-scatter the sum into the
MoE kernel's chunk blocks). The attention mixes tokens (a causal running
mean), so a row in the wrong place shows. Positive controls: a split that
takes another core's rows, and one whose o_proj reduce-scatter uses contiguous
blocks instead of the MoE's chunk blocks, both fail."""

import threading
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models import deepseek_v2

from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel_v2 as v2,
)
from vllm_torchtpu.layers.adapter import prefill_token_split
from vllm_torchtpu.layers.adapter.linear_common import PrequantSlot

RANKS = 16
H, LATENT, KEYS, HEAD, INTER = 64, 48, 16, 8, 8
DENSE, MOE = 2, 3


def own(x, rank):
    chunks = v2.token_chunks(x.shape[0])
    return x.reshape(chunks, RANKS, -1, *x.shape[1:])[:, rank].reshape(-1, *x.shape[1:])


def contiguous(x, rank):
    return x.reshape(RANKS, -1, *x.shape[1:])[rank]


def put_back(parts):
    chunks = v2.token_chunks(parts[0].shape[0] * RANKS)
    blocks = [p.reshape(chunks, -1, *p.shape[1:]) for p in parts]
    return torch.stack(blocks, 1).reshape(-1, *parts[0].shape[1:])


class Group:
    def __init__(self):
        self.barrier = threading.Barrier(RANKS, timeout=120)
        self.slots = [None] * RANKS

    def exchange(self, rank, x):
        self.slots[rank] = x
        self.barrier.wait()
        parts = list(self.slots)
        self.barrier.wait()
        return parts

    def all_reduce(self, rank, x):
        parts = self.exchange(rank, x)
        total = parts[0].clone()
        for part in parts[1:]:
            total += part
        return total


class Norm:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
        return y if residual is None else (y, residual)


class RowParallel:
    def __init__(self, weight):
        self.weight = weight
        self.quant_method = self

    def apply(self, layer, x, bias):
        assert bias is None
        return x @ layer.weight


class Attention:
    """The TPU MLA wrapper's contract with a token-mixing core."""

    def __init__(self, w, rank, group):
        self.w, self.rank, self.group = w, rank, group
        self.o_proj = RowParallel(w["o"][rank])

    def __call__(
        self, positions, hidden_states, llama_4_scaling=None, token_split=None
    ):
        assert llama_4_scaling is None
        latent = hidden_states @ self.w["latent"]
        keys = hidden_states @ self.w["keys"]
        if token_split is not None:
            latent, keys = token_split.gather_columns(positions.shape[0], latent, keys)
        steps = torch.arange(1, latent.shape[0] + 1, dtype=latent.dtype)[:, None]
        mixed = torch.cumsum(latent, 0) / steps + positions[:, None] * 1e-3
        out = torch.tanh(mixed @ self.w["v"][self.rank] + keys @ self.w["k"][self.rank])
        if token_split is not None:
            return token_split.project_reduce_scatter(out, self.o_proj)
        return self.group.all_reduce(self.rank, out @ self.o_proj.weight)


class SelfAttention:
    def __init__(self, mla_attn):
        self.mla_attn = mla_attn

    def __call__(self, positions, hidden_states, llama_4_scaling):
        return self.mla_attn(positions, hidden_states, llama_4_scaling)


class Experts:
    """moe_tp_decode's contract: a full TP partial summed over the ranks, or,
    with the split slot filled, the ranks' rows gathered and the sum
    reduce-scattered into the MoE kernel's chunk blocks."""

    def __init__(self, w1, w2, rank, group, slot):
        self.w1, self.w2, self.rank, self.group, self.slot = w1, w2, rank, group, slot

    def __call__(self, x):
        split = self.slot is not None and self.slot.value is not None
        if split:
            self.slot.value = None
            x = put_back(self.group.exchange(self.rank, x))
        partial = torch.nn.functional.silu(x @ self.w1) @ self.w2
        total = self.group.all_reduce(self.rank, partial)
        return own(total, self.rank) if split else total


class Layer:
    def __init__(self, index, w, rank, group, slot):
        self.layer_idx = index
        self.use_sequence_parallel_moe = False
        self.use_mha = False
        self.routed_scaling_factor = 1.0
        self.input_layernorm = Norm(w["norm_in"][index])
        self.post_attention_layernorm = Norm(w["norm_post"][index])
        self.self_attn = SelfAttention(Attention(w["attn"][index], rank, group))
        moe_slot = slot if index >= DENSE else None
        self.mlp = Experts(
            w["w1"][index][rank], w["w2"][index][rank], rank, group, moe_slot
        )

    def __call__(self, positions, hidden_states, residual, llama_4_scaling):
        return deepseek_v2.DeepseekV2DecoderLayer.forward(
            self, positions, hidden_states, residual, llama_4_scaling
        )


class Model:
    def __init__(self, w, rank, group, slot):
        self.embedding = w["embed"]
        self.layers = [Layer(i, w, rank, group, slot) for i in range(DENSE + MOE)]
        self.start_layer, self.end_layer = 0, len(self.layers)
        self.norm = Norm(w["norm_final"])
        self.aux_hidden_state_layers = ()
        self.config = SimpleNamespace()

    def embed_input_ids(self, input_ids):
        return self.embedding[input_ids]


class Split(prefill_token_split.TokenSplit):
    """TokenSplit with its three collectives on the rank threads."""

    def __init__(self, rank, group, slot, take=own, scatter=own):
        self.ranks = RANKS
        self.first_layer = DENSE
        self.edge = 128
        self.moe_slot = slot
        self.rotated = False
        self.fused_projection = False
        self._rank = rank
        self._own = take
        self._gather = lambda x: put_back(group.exchange(rank, x))
        self._scatter = lambda x: scatter(group.all_reduce(rank, x), rank)

    def rank_of(self, layer):
        return self._rank


def _weights():
    g = torch.Generator().manual_seed(0)

    def rand(*shape, scale=0.3):
        return torch.randn(*shape, generator=g, dtype=torch.float64) * scale

    layers = range(DENSE + MOE)
    return {
        "embed": rand(512, H, scale=1.0),
        "norm_in": [1 + rand(H, scale=0.1) for _ in layers],
        "norm_post": [1 + rand(H, scale=0.1) for _ in layers],
        "norm_final": 1 + rand(H, scale=0.1),
        "attn": [
            {
                "latent": rand(H, LATENT),
                "keys": rand(H, KEYS),
                "v": rand(RANKS, LATENT, HEAD),
                "k": rand(RANKS, KEYS, HEAD),
                "o": rand(RANKS, HEAD, H),
            }
            for _ in layers
        ],
        "w1": [rand(RANKS, H, INTER) for _ in layers],
        "w2": [rand(RANKS, INTER, H) for _ in layers],
    }


@pytest.fixture
def upstream(monkeypatch):
    first_and_last = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(deepseek_v2, "get_pp_group", lambda: first_and_last)
    monkeypatch.setattr(
        prefill_token_split,
        "_UPSTREAM_MODEL_FORWARD",
        deepseek_v2.DeepseekV2Model.forward,
    )


def _run(tokens, **split_kwargs):
    """Per rank: (today's hidden states, the split forward's)."""
    w = _weights()
    group = Group()
    input_ids = torch.randint(
        0, 512, (tokens,), generator=torch.Generator().manual_seed(1)
    )
    positions = torch.arange(tokens, dtype=torch.float64)
    results, errors = [None] * RANKS, []

    def rank_main(rank):
        try:
            slot = PrequantSlot()
            model = Model(w, rank, group, slot)
            today = deepseek_v2.DeepseekV2Model.forward(
                model, input_ids, positions, None
            )
            setattr(
                model,
                prefill_token_split.TOKEN_SPLIT_ATTR,
                Split(rank, group, slot, **split_kwargs),
            )
            split = prefill_token_split._model_forward(
                model, input_ids, positions, None
            )
            results[rank] = (today, split)
        except BaseException as e:  # noqa: BLE001 - re-raised below
            errors.append(e)
            group.barrier.abort()

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(RANKS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


# 384 tokens with 128-token combine chunks: three chunk blocks per core.
@pytest.mark.parametrize("tokens,chunk_tokens", [(256, 2048), (384, 128)])
def test_split_forward_matches_today(upstream, monkeypatch, tokens, chunk_tokens):
    monkeypatch.setattr(v2, "_COMBINE_CHUNK_TOKENS", chunk_tokens)
    results = _run(tokens)
    today0 = results[0][0]
    assert today0.shape == (tokens, H)
    for today, split in results:
        torch.testing.assert_close(today, today0, rtol=0, atol=0)
        assert split.shape == (tokens, H)
        torch.testing.assert_close(split, today, rtol=1e-12, atol=1e-12)


def test_small_buckets_keep_todays_forward(upstream):
    for today, split in _run(64):
        torch.testing.assert_close(split, today, rtol=0, atol=0)


def _other_core(x, rank):
    return own(x, (rank + 1) % RANKS)


@pytest.mark.parametrize(
    "broken",
    [dict(take=_other_core), dict(scatter=contiguous)],
    ids=["take", "scatter"],
)
def test_a_broken_split_fails(upstream, monkeypatch, broken):
    monkeypatch.setattr(v2, "_COMBINE_CHUNK_TOKENS", 128)
    results = _run(384, **broken)
    worst = max((split - today).abs().max().item() for today, split in results)
    assert worst > 1e-3
