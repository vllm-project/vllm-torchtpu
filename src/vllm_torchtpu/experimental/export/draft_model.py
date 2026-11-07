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
"""A separate draft model (DSpark, DFlash) as the artifact's drafter.

A DSpark draft is a small decoder of its own, not a layer of the target. It
reads the target's residual stream at a few layers (`aux_hidden_states`, which
`forward` returns for every row), projects each row of it to the draft's
context keys and values, and drafts a whole block in one non-causal pass: the
anchor token (the one sampled after the last accepted token) followed by mask
tokens. A DSpark draft's Markov head then picks the block's tokens left to
right; a DFlash draft takes each mask slot's argmax.

The steps below are copies of the TPU runner's DFlash / DSpark proposer
(`vllm_torchtpu.spec_decode.dflash.DFlashProposer` and
`vllm_torchtpu.spec_decode.dspark.DSparkProposer`), taking the draft model as
an argument instead of the proposer. They are kept inside the experimental
exporter; `tests/experimental/export/test_draft_model_parity.py` checks that
each agrees with the runner's.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_torchtpu.experimental.export.trace import trace

# The attribute of a loaded target model holding its separate draft model.
DRAFT_MODEL_ATTR = "_export_draft_model"
# The draft model's own weights are exported under this prefix, for its `model.`.
DRAFT_PARAM_PREFIX = "draft."
# The `forward` output a draft model reads: the target's residual stream at the
# draft's target layers, concatenated per row.
AUX_HIDDEN_STATES = "aux_hidden_states"
# `speculative_config.method` values an artifact declares for a separate draft
# model: the runtime drives DSpark and DFlash drafts alike, as `dspark`.
DRAFT_MODEL_METHODS = ("dspark",)
# vLLM's `speculative_config.method` values that load a separate draft model.
VLLM_DRAFT_MODEL_METHODS = ("dspark", "dflash")


def is_dspark(draft_model: Any) -> bool:
    """Whether the draft picks its tokens with a DSpark Markov head."""
    return callable(getattr(draft_model, "markov_embed", None))


def draft_block_size(draft_model: Any, k: int) -> int:
    """Query tokens per request of a draft drafting `k` tokens: a DSpark
    draft's `query_block_size`, a DFlash draft's anchor and `k` mask slots
    (`DFlashProposer._query_block_size`)."""
    if is_dspark(draft_model):
        return query_block_size(k, sample_from_anchor(draft_model))
    return k + 1


def sample_from_anchor(draft_model: Any) -> bool:
    """Whether the draft's query block starts at its anchor
    (`DSparkProposer.sample_from_anchor`): dense DSpark checkpoints omit the
    key, so it defaults to True."""
    hf_config = draft_model.config
    hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}
    return bool(hf_dict.get("sample_from_anchor", True))


def query_block_size(k: int, from_anchor: bool) -> int:
    """Query tokens per request (`DSparkProposer._query_block_size`): `k` when
    the anchor is the first prediction, else the anchor plus `k` mask slots."""
    return k if from_anchor else k + 1


@trace("export.load_draft_model")
def load_draft_model(vllm_config: Any, target: Any, mesh: Any) -> nn.Module | None:
    """Loads the draft model `vllm_config.speculative_config` declares, or None.

    As `DFlashProposer.load_model` and `DSparkProposer.load_model` load it:
    the target is tagged to return its residual stream at the draft's target
    layers, the draft is built from the speculative config's draft model
    config, shares the target's embedding and LM head unless the checkpoint
    has its own, and each draft attention layer gets the checkpoint's
    causality.
    """
    from vllm.compilation.backends import set_model_tag
    from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.model_executor.model_loader import get_model_loader

    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )
    from vllm_torchtpu.spec_decode.utils import (
        _force_draft_tp1,
        maybe_share_embeddings,
        maybe_share_lm_head,
        normalize_draft_config,
    )

    spec = getattr(vllm_config, "speculative_config", None)
    method = getattr(spec, "method", None)
    if spec is None or method not in VLLM_DRAFT_MODEL_METHODS:
        return None
    dflash_config = normalize_draft_config(spec.draft_model_config.hf_config)
    target_layer_ids = list(dflash_config.get("target_layer_ids", []))
    if not target_layer_ids:
        raise ValueError(
            "the draft model's config names no target layer ids (one of "
            "eagle_aux_hidden_state_layer_ids, dspark_target_layer_ids, "
            "target_layer_ids, or dflash_config.target_layer_ids)"
        )
    target_tp = vllm_config.parallel_config.tensor_parallel_size
    draft_tp = spec.draft_tensor_parallel_size or target_tp
    if draft_tp != target_tp:
        raise ValueError(
            f"draft_tensor_parallel_size {draft_tp} must equal "
            f"tensor_parallel_size {target_tp}: the drafter takes the target's "
            "per-rank inputs"
        )

    layers = tuple(idx + 1 for idx in target_layer_ids)
    if hasattr(target, "set_aux_hidden_state_layers"):
        target.set_aux_hidden_state_layers(layers)
    elif hasattr(getattr(target, "model", None), "_set_aux_hidden_state_layers"):
        target.model._set_aux_hidden_state_layers(layers)
    else:
        raise ValueError(
            f"{type(target).__name__} cannot return the residual stream of its "
            "layers (set_aux_hidden_state_layers), which a draft model reads"
        )

    target_layer_names = set(
        get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    )
    draft_vllm_config = copy.copy(vllm_config)
    draft_model_config = copy.copy(spec.draft_model_config)
    draft_model_config.runner_type = "draft"
    replicated = draft_tp == 1 and target_tp > 1
    with (
        set_model_tag("dflash_head"),
        set_vllm_model_wrapper_context(mesh=mesh),
        set_current_vllm_config(draft_vllm_config),
        _force_draft_tp1() if replicated else contextlib.nullcontext(),
    ):
        loader: Any = get_model_loader(vllm_config.load_config).load_model
        draft: Any = loader(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    all_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    for name in set(all_layers) - target_layer_names:
        all_layers[name].is_draft_layer = True  # type: ignore[attr-defined]

    dspark = method == "dspark"
    if dspark:
        for attr in ("compute_draft_logits", "markov_embed", "markov_bias"):
            if not hasattr(draft, attr):
                raise ValueError(
                    f"DSpark draft model {type(draft).__name__} lacks {attr}(); "
                    "expected a Qwen3DSpark/Gemma4DSpark-family model"
                )
    # A DFlash draft always predicts the target vocabulary with the target's
    # embedding and LM head. A reduced-vocabulary DSpark checkpoint carries its
    # own LM head (and maybe embedding): share the target's only where the
    # checkpoint has none.
    maybe_share_embeddings(draft, target, True, force_share=not dspark)
    maybe_share_lm_head(draft, target, replicated, True, force_share=not dspark)

    # The caller builds every attention layer's kernel after loading, the draft's
    # with the causality set here.
    causal = (
        draft.get_draft_attn_causal()
        if hasattr(draft, "get_draft_attn_causal")
        else draft.model.get_draft_attn_causal()
    )
    for i, layer in enumerate(draft.model.layers):
        attn = getattr(getattr(layer, "self_attn", None), "attn", None)
        if attn is not None and isinstance(attn.impl, PallasAttentionBackendImpl):
            attn.impl.layer_idx = i
            attn.impl.use_causal_mask = causal[i]
            bind_draft_attention_kernel(attn.impl, draft_tp)
    return draft


def bind_draft_attention_kernel(impl: Any, draft_tp: int) -> None:
    """Points a draft attention layer at the kernel the TPU runner runs it with.

    A copy of the draft branch of `TPUModelRunner._initialize_attention_kernels`:
    a tp=1 draft runs the local (non-shard_map) ragged paged attention kernel
    with a 1-token decode tile, whatever the attention backend. That kernel
    takes non-causal layers (DSpark and DFlash in-block attention), which the
    batched kernel of `--attention-backend CUSTOM` rejects. Must run before
    `initialize_kernel`.
    """
    if draft_tp != 1:
        return
    from vllm_torchtpu.layers.adapter.attention import _pallas_rpa_kernel_local

    impl.decode_query_size = 1
    # Instance attributes shadow the class's, as the runner sets them.
    impl._kernel_entry = _pallas_rpa_kernel_local
    impl._kernel_op_prefix = "pallas::rpa_kernel_local"


def bind_fused_kv_weight(draft_model: Any) -> None:
    """Sets the draft's `_fused_kv_weight` / `_fused_kv_bias`, which
    `precompute_context_kv` projects with, from its layers' bound `qkv_proj`
    weights, as vLLM's `_build_fused_kv_buffers` stacks them after loading.

    vLLM keeps them as plain tensors built from the loaded weights; built from
    the weights the export binds, they are graph inputs rather than constants.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    self_model: Any = draft_model.model
    attns = [layer.self_attn for layer in self_model.layers]
    if not all(
        isinstance(a.qkv_proj.quant_method, UnquantizedLinearMethod) for a in attns
    ):
        raise NotImplementedError(
            "a draft model with quantized qkv_proj weights is not supported by export"
        )
    self_model._fused_kv_weight = torch.cat(
        [a.qkv_proj.weight[a.q_size :] for a in attns], dim=0
    )
    self_model._fused_kv_bias = (
        torch.cat([a.qkv_proj.bias[a.q_size :] for a in attns], dim=0)
        if attns[0].qkv_proj.bias is not None
        else None
    )


def precompute_context_kv(
    draft_model: Any,
    hidden_states: Any,
    positions: torch.Tensor,
    draft_md_tuple: Sequence[Any],
) -> None:
    """Writes each row's context keys and values to the draft's paged KV.

    A copy of `DFlashProposer._tpu_precompute_and_update_kv_cache`: the
    concatenated target hidden states are combined to the draft's width,
    normed, projected to every layer's K and V at once, K-normed and RoPE'd,
    and each layer's attention writes them through its metadata.
    """
    self_model: Any = draft_model.model
    target_hidden: Any
    if isinstance(hidden_states, (list, tuple)):
        target_hidden = (
            hidden_states[0]
            if len(hidden_states) == 1
            else torch.cat(hidden_states, dim=-1)
        )
    else:
        target_hidden = hidden_states

    tpu_precompute = getattr(draft_model, "tpu_precompute_and_store_context_kv", None)
    if callable(tpu_precompute):
        tpu_precompute(target_hidden, positions, draft_md_tuple)
        return
    if getattr(type(draft_model), "owns_context_kv", False):
        draft_model.precompute_and_store_context_kv(  # type: ignore[operator]
            target_hidden, positions, draft_md_tuple
        )
        return

    if hasattr(draft_model, "combine_hidden_states"):
        target_hidden = draft_model.combine_hidden_states(target_hidden)  # type: ignore[operator]
    elif (fc_layer := getattr(self_model, "fc", None)) is not None:
        target_hidden = fc_layer(target_hidden)
        if isinstance(target_hidden, tuple):
            target_hidden = target_hidden[0]

    if hasattr(self_model, "hidden_norm"):
        target_hidden = self_model.hidden_norm(target_hidden)

    all_kv_flat = F.linear(
        target_hidden, self_model._fused_kv_weight, self_model._fused_kv_bias
    )

    num_ctx = all_kv_flat.shape[0]
    L = len(self_model.layers)
    hd = self_model.layers[0].self_attn.head_dim
    nkv = self_model.layers[0].self_attn.num_kv_heads

    all_kv = (
        all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    )
    all_k = all_kv[0]
    all_v = all_kv[1]

    if hasattr(self_model.layers[0].self_attn, "k_norm"):
        kw = [layer.self_attn.k_norm.weight for layer in self_model.layers]
        kw_stack = torch.stack(kw, dim=0).view(L, 1, 1, hd)
        eps = self_model.layers[0].self_attn.k_norm.variance_epsilon
        variance = all_k.pow(2).mean(-1, keepdim=True)
        all_k_normed = all_k * torch.rsqrt(variance + eps) * kw_stack
    else:
        all_k_normed = all_k

    all_k_flat2 = all_k_normed.view(L * num_ctx, nkv, hd)
    positions_repeated = positions.repeat(L)
    nq = self_model.layers[0].self_attn.num_heads
    dummy_q = torch.zeros(
        (L * num_ctx, nq, hd), device=target_hidden.device, dtype=target_hidden.dtype
    )
    dummy_q_roped_flat, roped_k_flat = self_model.layers[0].self_attn.rotary_emb(
        positions_repeated, dummy_q, all_k_flat2
    )
    roped_k_all = roped_k_flat.view(L, num_ctx, nkv, hd)
    roped_q_all = dummy_q_roped_flat.view(L, num_ctx, nq, hd)

    for i, layer in enumerate(self_model.layers):
        attn_obj = layer.self_attn.attn
        attn_obj.impl.forward(
            layer=attn_obj,
            query=roped_q_all[i],
            key=roped_k_all[i],
            value=all_v[i],
            kv_cache=attn_obj.kv_cache,
            attn_metadata=draft_md_tuple[i],
        )


def layout_draft_block(
    base_tokens: torch.Tensor,
    base_pos: torch.Tensor,
    block_size: int,
    padded_len: int,
    mask_token_id: int,
    max_model_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`[base, MASK x (block_size - 1)]` per request, at positions from
    `base_pos` (`[rows, 1]`); returns `(input_ids, positions, seq_lens)`.

    A copy of `DFlashProposer._layout_draft_block`.
    """
    device = base_tokens.device
    padded_num_reqs = base_tokens.shape[0]

    input_ids = torch.full(
        (padded_len,), mask_token_id, dtype=torch.int32, device=device
    )
    scatter_indices = torch.arange(
        0, padded_num_reqs * block_size, block_size, dtype=torch.int32, device=device
    )
    input_ids.scatter_(0, scatter_indices, base_tokens)

    offsets = torch.arange(0, block_size, dtype=torch.int32, device=device).unsqueeze(0)
    positions_unpadded = (base_pos + offsets).flatten()
    positions = F.pad(
        positions_unpadded, (0, padded_len - positions_unpadded.shape[0]), value=0
    )
    positions = torch.clamp(positions, max=max_model_len - 1)
    seq_lens = torch.clamp(base_pos.squeeze(1) + block_size, max=max_model_len)
    return input_ids, positions, seq_lens


def dspark_forward_and_sample(
    draft_model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    from_anchor: bool,
) -> torch.Tensor:
    """The draft block's tokens `[rows, k]`: one backbone pass over every
    block, then the Markov head's greedy picks left to right.

    A copy of `DSparkProposer._dflash_forward_and_sample` without the
    confidence head, which only adaptive verification reads.
    """
    hidden = draft_model.model(input_ids=input_ids, positions=positions)

    padded_num_reqs = hidden.shape[0] // block_size
    valid_hidden = hidden[: padded_num_reqs * block_size]

    logits = draft_model.compute_draft_logits(valid_hidden)
    logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])

    ids_3d = input_ids[: padded_num_reqs * block_size].view(padded_num_reqs, block_size)
    prev = ids_3d[:, 0]
    draft_tokens = []
    sample_start = 0 if from_anchor else 1
    for i in range(sample_start, block_size):
        markov_embed = draft_model.markov_embed(prev)
        bias = draft_model.markov_bias(markov_embed)
        step = draft_model.map_draft_to_target((logits_3d[:, i] + bias).argmax(dim=-1))
        draft_tokens.append(step)
        prev = step
    return torch.stack(draft_tokens, dim=1)


def dflash_forward_and_sample(
    draft_model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """The draft block's tokens `[rows, block_size - 1]`: one backbone pass over
    every block, then the argmax of each mask slot (slot 0 is the anchor).

    A copy of `DFlashProposer._dflash_forward_and_sample`.
    """
    hidden = draft_model.model(input_ids=input_ids, positions=positions)

    padded_num_reqs = hidden.shape[0] // block_size
    valid_hidden = hidden[: padded_num_reqs * block_size]

    hidden_reshaped = valid_hidden.view(padded_num_reqs, block_size, hidden.shape[-1])
    draft_hidden = hidden_reshaped[:, 1:, :].reshape(-1, hidden.shape[-1])

    logits = draft_model.compute_logits(draft_hidden)
    logits_3d = logits.view(padded_num_reqs, block_size - 1, logits.shape[-1])
    return logits_3d.argmax(dim=-1)


def draft_forward_and_sample(
    draft_model: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """The draft block's tokens `[rows, k]`, picked as the draft's proposer
    picks them: DSpark's Markov head or DFlash's argmax."""
    if is_dspark(draft_model):
        return dspark_forward_and_sample(
            draft_model,
            input_ids,
            positions,
            block_size,
            sample_from_anchor(draft_model),
        )
    return dflash_forward_and_sample(draft_model, input_ids, positions, block_size)
