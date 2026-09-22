# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp resident FP8 PLE; private history-major cache and explicit metadata."""
import re
from collections.abc import Iterable

import torch
from torch import nn
from vllm.models.qwen4_exp.common.ple import compute_ple_shard_overlap

from vllm_torchtpu.kernels.qwen4_exp.ple import (build_lookup_op,
                                                 build_short_conv_op,
                                                 build_values_op,
                                                 compute_ngram_ids)


class ResidentPLEEmbedding(nn.Module):
    """Checkpoint shard intersections staged once, then one compact-table H2D.

    The generic geometry makes loader boundaries testable without allocating
    the deployment's 6.4GB/rank. No decoded copy of the table is retained.
    ``load_weights`` accepts names relative to ``ngram_embedding.`` and
    accounts for every supplied global shard, including non-owned shards.
    """

    def __init__(self,
                 *,
                 tp_rank: int,
                 tp_size: int = 8,
                 total_rows: int = 320001536,
                 checkpoint_shards: int = 128,
                 device=None):
        super().__init__()
        if (tp_size <= 0 or not 0 <= tp_rank < tp_size or total_rows <= 0
                or total_rows % (tp_size * 4) or checkpoint_shards <= 0
                or checkpoint_shards > total_rows):
            raise ValueError(
                "PLE requires positive shards and TP ranges of complete four-row groups"
            )
        self.local_rows = total_rows // tp_size
        if total_rows >= 2**31 or self.local_rows // 4 * 5 >= 2**31:
            raise ValueError(
                "PLE global row and compact segment indices must fit int32")
        self.vocab_start = tp_rank * self.local_rows
        self.total_rows = total_rows
        self.checkpoint_shards = checkpoint_shards
        self.shard_rows = (total_rows + checkpoint_shards -
                           1) // checkpoint_shards
        self.register_buffer("table",
                             torch.empty(0, dtype=torch.uint8, device=device))
        self.register_buffer(
            "weight_scale", torch.empty(1, dtype=torch.bfloat16,
                                        device=device))
        self._staging = None
        self._scale = None
        self._shards = set()
        self._finalized = False
        self._lookup = None

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        if self._finalized:
            raise ValueError("PLE weights are already finalized")
        loaded = set()
        for name, tensor in weights:
            if name == "weight_scale":
                if tensor.shape != (1, ) or tensor.dtype != torch.bfloat16:
                    raise ValueError(
                        "PLE weight_scale must be a BF16 [1] global scale")
                scale = tensor.detach().cpu()
                if not bool(torch.isfinite(scale).all()) or float(
                        scale[0]) <= 0:
                    raise ValueError(
                        "PLE weight_scale must be finite and positive")
                if self._scale is not None and not torch.equal(
                        self._scale, scale):
                    raise ValueError(
                        "PLE checkpoint has conflicting global weight_scale")
                self._scale = scale.clone()
                loaded.add(name)
                continue
            match = re.fullmatch(r"shard_(\d+)\.weight", name)
            if match is None:
                raise ValueError(
                    f"Unknown PLE embedding checkpoint tensor: {name}")
            index = int(match[1])
            if not 0 <= index < self.checkpoint_shards:
                raise ValueError(f"Invalid PLE checkpoint shard {index}")
            if index in self._shards:
                raise ValueError(f"Duplicate PLE checkpoint shard {index}")
            start = index * self.shard_rows
            rows = max(0, min(self.shard_rows, self.total_rows - start))
            if tensor.shape != (rows,
                                160) or tensor.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    f"PLE shard {index} requires FP8 e4m3fn [{rows},160]")
            overlap = compute_ple_shard_overlap(checkpoint_start=start,
                                                checkpoint_rows=rows,
                                                tp_start=self.vocab_start,
                                                tp_end=self.vocab_start +
                                                self.local_rows)
            if overlap is not None:
                if self._staging is None:
                    self._staging = torch.empty((self.local_rows, 160),
                                                dtype=torch.uint8,
                                                device="cpu")
                source = tensor.detach().narrow(0, overlap.source_start,
                                                overlap.row_count).view(
                                                    torch.uint8)
                self._staging.narrow(0, overlap.destination_start,
                                     overlap.row_count).copy_(source)
            self._shards.add(index)
            loaded.add(name)
        return loaded

    def finalize_weights(self):
        if self._finalized:
            return
        missing = set(range(self.checkpoint_shards)) - self._shards
        if missing:
            raise ValueError(
                f"Missing PLE checkpoint shards: {sorted(missing)}")
        if self._scale is None:
            raise ValueError("Missing PLE global weight_scale")
        assert self._staging is not None, "Complete checkpoint must cover this TP range"
        self.table = self._staging.reshape(-1,
                                           128).to(device=self.table.device)
        self.weight_scale.copy_(self._scale)
        self._staging = None
        self._scale = None
        self._lookup = build_lookup_op(self.vocab_start, self.local_rows)
        self._finalized = True

    def forward(self, ids):
        assert self._finalized, "PLE finalize_weights must run before inference"
        return self._lookup(self.table, ids, self.weight_scale)


class _HashEmbedding(nn.Module):

    def __init__(self, tp_rank, tp_size, device):
        super().__init__()
        for name, size in (("layer_multipliers", 3), ("ngram_heads_offsets",
                                                      16),
                           ("ngram_heads_vocab_sizes", 16)):
            self.register_buffer(
                name, torch.empty(size, dtype=torch.int64, device=device))
        self.ngram_embedding = ResidentPLEEmbedding(tp_rank=tp_rank,
                                                    tp_size=tp_size,
                                                    device=device)

    def forward(self, tokens, starts, context, eos_token_id):
        ids = compute_ngram_ids(tokens,
                                starts,
                                context,
                                self.layer_multipliers,
                                self.ngram_heads_vocab_sizes,
                                self.ngram_heads_offsets,
                                eos_token_id=eos_token_id)
        return self.ngram_embedding(ids)


class _NormWeight(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(10240,
                                               dtype=torch.bfloat16,
                                               device=device),
                                   requires_grad=False)


class Qwen4ExpPLE(nn.Module):
    """S1 TP8/DP1 PLE delta; caller adds the hidden residual exactly once.

    Loader names are relative to ``model.language_model.layers.1.ple.``.
    Construct/load on CPU then move and finalize, or construct on device and
    stream checkpoint tensors on CPU. Finalize is required before compile.
    The model cache planner binds ``kv_cache=(history,)`` or ``bind_history``;
    metadata is passed once per step, never retrieved from runner state.
    """

    def __init__(self,
                 *,
                 tp_rank: int,
                 tp_size: int = 8,
                 device=None,
                 prefix: str = "",
                 eos_token_id: int = 248044):
        super().__init__()
        if tp_size != 8 or not 0 <= tp_rank < tp_size:
            raise ValueError("Qwen4Exp S1 PLE requires TP8")
        self.prefix, self.eos_token_id = prefix, eos_token_id
        self.ple_embedding = _HashEmbedding(tp_rank, tp_size, device)
        self.key_proj = nn.Linear(2560,
                                  10240,
                                  bias=False,
                                  dtype=torch.bfloat16,
                                  device=device)
        self.value_proj = nn.Linear(2560,
                                    2560,
                                    bias=False,
                                    dtype=torch.bfloat16,
                                    device=device)
        for name in ("norm_key", "norm_query", "norm_conv"):
            setattr(self, name, _NormWeight(device))
        self.conv1d = nn.Conv1d(10240,
                                10240,
                                4,
                                groups=10240,
                                bias=False,
                                dtype=torch.bfloat16,
                                device=device)
        self.kv_cache = ()
        self._loaded = set()
        self._finalized = False
        self._values = None
        self._conv = None

    def get_state_shape(self):
        return (9, 10240),

    def get_state_dtype(self):
        return torch.bfloat16,

    def bind_history(self, history):
        if history.ndim != 3 or history.shape[1:] != (
                9, 10240) or history.dtype != torch.bfloat16:
            raise ValueError("PLE history must be BF16 [slots,9,10240]")
        self.kv_cache = (history, )

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        if self._finalized:
            raise ValueError("PLE weights are already finalized")
        tensors = dict(self.named_parameters())
        tensors.update(self.named_buffers())
        loaded = set()
        embedding_prefix = "ple_embedding.ngram_embedding."
        for name, tensor in weights:
            if name.startswith(embedding_prefix):
                self.ple_embedding.ngram_embedding.load_weights([
                    (name[len(embedding_prefix):], tensor)
                ])
            else:
                if name not in tensors or tensor.shape != tensors[
                        name].shape or tensor.dtype != tensors[name].dtype:
                    raise ValueError(f"Invalid PLE checkpoint tensor: {name}")
                with torch.no_grad():
                    tensors[name].copy_(tensor)
            self._loaded.add(name)
            loaded.add(name)
        return loaded

    def finalize_weights(self):
        if self._finalized:
            return
        required = set(dict(self.named_parameters())) | {
            "ple_embedding.layer_multipliers",
            "ple_embedding.ngram_heads_offsets",
            "ple_embedding.ngram_heads_vocab_sizes"
        }
        missing = required - self._loaded
        if missing:
            raise ValueError(
                f"Missing PLE checkpoint tensors: {sorted(missing)}")
        # Validate the checkpoint's vocabulary layout; never regenerate it from
        # a physical layer number (PLE dense id and model layer id differ).
        embedding = self.ple_embedding
        sizes = embedding.ngram_heads_vocab_sizes.cpu()
        offsets = embedding.ngram_heads_offsets.cpu()
        expected = torch.cat((torch.zeros(1, dtype=torch.int64,
                                          device="cpu"), sizes.cumsum(0)[:-1]))
        if bool(
            (sizes <= 0).any()) or not torch.equal(offsets, expected) or int(
                sizes.sum()) > embedding.ngram_embedding.total_rows:
            raise ValueError("Invalid PLE checkpoint vocabulary sizes/offsets")
        embedding.ngram_embedding.finalize_weights()
        self._values, self._conv = build_values_op(), build_short_conv_op()
        self._finalized = True

    def forward(self, hidden_states, input_ids, query_start_loc, ngram_context,
                state_indices, has_initial_states):
        from vllm.distributed import tensor_model_parallel_all_reduce

        assert self._finalized, "PLE finalize_weights must run before inference"
        assert len(
            self.kv_cache) == 1, "PLE history must be bound before inference"
        embeddings = self.ple_embedding(input_ids, query_start_loc,
                                        ngram_context, self.eos_token_id)
        embeddings = tensor_model_parallel_all_reduce(embeddings)
        gated, normalized = self._values(embeddings, hidden_states,
                                         self.key_proj.weight,
                                         self.value_proj.weight,
                                         self.norm_key.weight,
                                         self.norm_query.weight,
                                         self.norm_conv.weight)
        conv = self._conv(normalized, self.conv1d.weight[:,
                                                         0], self.kv_cache[0],
                          query_start_loc, state_indices, has_initial_states)
        t = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        # Padding cannot leak a nonzero PLE delta despite safe row-zero lookup.
        if state_indices.shape[0] == 0:
            return torch.zeros_like(hidden_states)
        req = torch.sum(t[:, None] >= query_start_loc[None, 1:], dim=1)
        req = req.clamp(max=state_indices.shape[0] - 1)
        valid = (t < query_start_loc[-1]) & (state_indices[req] >= 0)
        return torch.where(valid[:, None], gated + conv, 0)
