# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp resident BF16/FP8 PLE and private history-major cache."""
import re
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from vllm.models.qwen4_exp.common.ple import compute_ple_shard_overlap

from vllm_torchtpu.kernels.qwen4_exp import ple as ple_kernels

from .parallel import PLEParallel
from .ple_host import SharedPLETable


class ResidentPLEEmbedding(nn.Module):
    """Checkpoint shard intersections staged once, then one compact-table H2D.

    The generic geometry makes loader boundaries testable without allocating
    the deployment's 6.4GB/rank. No decoded copy of the table is retained.
    ``load_weights`` accepts names relative to ``ngram_embedding.`` and
    accounts for every supplied global shard, including non-owned shards.
    ``tp_rank/tp_size`` describe the embedding partition (ETP), which may
    differ from ordinary model TP when embedding_across_dp is enabled.
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
        self._weight_dtype = None
        self._scale = None
        self._shards = set()
        self._finalized = False
        self._lookup = None
        self._raw_lookup = None
        self._decode = None

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        if self._finalized:
            raise ValueError("PLE weights are already finalized")
        loaded = set()
        for name, tensor in weights:
            if name == "weight_scale":
                if self._weight_dtype == torch.bfloat16:
                    raise ValueError(
                        "BF16 PLE must not have an FP8 weight_scale")
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
            if tensor.shape != (rows, 160) or tensor.dtype not in (
                    torch.float8_e4m3fn, torch.bfloat16):
                raise ValueError(
                    f"PLE shard {index} requires BF16 or FP8 e4m3fn [{rows},160]"
                )
            if self._weight_dtype not in (None, tensor.dtype):
                raise ValueError("PLE checkpoint shards must have one dtype")
            if tensor.dtype == torch.bfloat16 and self._scale is not None:
                raise ValueError("BF16 PLE must not have an FP8 weight_scale")
            self._weight_dtype = tensor.dtype
            carrier = (torch.int16
                       if tensor.dtype == torch.bfloat16 else torch.uint8)
            overlap = compute_ple_shard_overlap(checkpoint_start=start,
                                                checkpoint_rows=rows,
                                                tp_start=self.vocab_start,
                                                tp_end=self.vocab_start +
                                                self.local_rows)
            if overlap is not None:
                source = tensor.detach().narrow(
                    0, overlap.source_start, overlap.row_count).view(carrier)
                self._copy_rows(source, overlap.destination_start, carrier)
            self._shards.add(index)
            loaded.add(name)
        return loaded

    def _copy_rows(self, source, start, carrier):
        if self._staging is None:
            self._staging = torch.empty((self.local_rows, 160),
                                        dtype=carrier,
                                        device="cpu")
        self._staging.narrow(0, start, source.shape[0]).copy_(source)

    def _validate_checkpoint(self):
        missing = set(range(self.checkpoint_shards)) - self._shards
        if missing:
            raise ValueError(
                f"Missing PLE checkpoint shards: {sorted(missing)}")
        if self._weight_dtype == torch.float8_e4m3fn and self._scale is None:
            raise ValueError("Missing PLE global weight_scale")

    def finalize_weights(self):
        if self._finalized:
            return
        self._validate_checkpoint()
        assert self._staging is not None, "Complete checkpoint must cover this TP range"
        self.table = self._staging.reshape(-1,
                                           128).to(device=self.table.device)
        if self._scale is None:
            self.weight_scale.fill_(1)
        else:
            self.weight_scale.copy_(self._scale)
        self._staging = None
        self._scale = None
        self._lookup = ple_kernels.build_lookup_op(self.vocab_start,
                                                   self.local_rows)
        self._raw_lookup = ple_kernels.build_raw_lookup_op(
            self.vocab_start, self.local_rows)
        self._decode = ple_kernels.build_decode_op()
        self._finalized = True

    def forward(self, ids):
        assert self._finalized, "PLE finalize_weights must run before inference"
        return self._lookup(self.table, ids, self.weight_scale)

    def lookup_raw(self, ids):
        assert self._finalized, "PLE finalize_weights must run before inference"
        return self._raw_lookup(self.table, ids)

    def decode(self, raw_embeddings):
        return self._decode(raw_embeddings, self.weight_scale)


class HostPLEEmbedding(ResidentPLEEmbedding):
    """Reuse checkpoint validation while retaining row160 storage on the CPU.

    Only the small scale buffer belongs to the device computation. Host lookup
    returns integer bits; the adapter owns their explicit device transfer.
    """

    def __init__(self, *, shared=None, **kwargs):
        super().__init__(**kwargs)
        self._shared_config = shared
        self._shared_table = None

    def _copy_rows(self, source, start, carrier):
        if self._shared_config is None:
            return super()._copy_rows(source, start, carrier)
        if self._shared_table is None:
            options = self._shared_config
            self._shared_table = SharedPLETable(
                directory=options["directory"],
                key=options["key"],
                identity=dict(options["identity"],
                              dtype=str(self._weight_dtype),
                              rows=self.local_rows,
                              start=self.vocab_start),
                nbytes=self.local_rows * 160 * source.element_size(),
                participants=options["participants"])
        if self._shared_table.is_writer:
            # The shard intersection is contiguous row160, with no table-sized
            # staging copy and no floating-point conversion.
            value = source.cpu().numpy().view(np.uint8).reshape(-1)
            self._shared_table.write(start * 160 * source.element_size(),
                                     value)

    def finalize_weights(self):
        if self._finalized:
            return
        self._validate_checkpoint()
        if self._shared_config is None:
            assert self._staging is not None, "Checkpoint must cover the host range"
        else:
            assert self._shared_table is not None, "Checkpoint must cover the shared range"
            scale_bits = (None if self._scale is None else int(
                self._scale.view(torch.int16)[0]))
            self._shared_table.finish(metadata=dict(scale_bits=scale_bits))
        if self._scale is None:
            self.weight_scale.fill_(1)
        else:
            self.weight_scale.copy_(self._scale)
        self._scale = None
        self._decode = ple_kernels.build_decode_op()
        self._finalized = True

    def lookup_host(self, ids):
        assert self._finalized, "Host PLE must be finalized before lookup"
        indices = ids.numpy().reshape(-1)
        owned = ((indices >= self.vocab_start)
                 & (indices < self.vocab_start + self.local_rows))
        dtype = np.int16 if self._weight_dtype == torch.bfloat16 else np.uint8
        output = np.zeros((indices.size, 160), dtype=dtype)
        rows = indices[owned] - self.vocab_start
        if self._shared_table is None:
            output[owned] = self._staging.numpy()[rows]
        else:
            with self._shared_table.buffer() as buffer:
                output[owned] = np.frombuffer(buffer, dtype=dtype).reshape(
                    self.local_rows, 160)[rows]
        raw = torch.from_numpy(output)
        if self._weight_dtype == torch.float8_e4m3fn:
            raw = raw.view(torch.int8)
        return raw.reshape(ids.shape[0], 2560)

    def close(self):
        self._staging = None
        if self._shared_table is not None:
            self._shared_table.close()
            self._shared_table = None
        self._finalized = False


class _HashEmbedding(nn.Module):

    def __init__(self,
                 tp_rank,
                 tp_size,
                 device,
                 *,
                 cpu_offload=False,
                 shared=None):
        super().__init__()
        for name, size in (("layer_multipliers", 3), ("ngram_heads_offsets",
                                                      16),
                           ("ngram_heads_vocab_sizes", 16)):
            self.register_buffer(
                name, torch.empty(size, dtype=torch.int64, device=device))
        if cpu_offload:
            self.ngram_embedding = HostPLEEmbedding(tp_rank=tp_rank,
                                                    tp_size=tp_size,
                                                    device=device,
                                                    shared=shared)
        else:
            self.ngram_embedding = ResidentPLEEmbedding(tp_rank=tp_rank,
                                                        tp_size=tp_size,
                                                        device=device)

    def forward(self, tokens, starts, context, eos_token_id, valid_tokens):
        ids = ple_kernels.compute_ngram_ids(tokens,
                                            starts,
                                            context,
                                            self.layer_multipliers,
                                            self.ngram_heads_vocab_sizes,
                                            self.ngram_heads_offsets,
                                            eos_token_id=eos_token_id)
        # Keep the signed INT64 hash through remainder, then use the bounded
        # global IDs as an INT32 collective carrier. Padding has no owner.
        return torch.where(valid_tokens[:, None], ids, -1).to(torch.int32)


class _NormWeight(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(10240,
                                               dtype=torch.bfloat16,
                                               device=device),
                                   requires_grad=False)


class Qwen4ExpPLE(nn.Module):
    """Resident PLE delta; caller adds the hidden residual exactly once.

    Loader names are relative to ``model.language_model.layers.1.ple.``.
    Construct/load on CPU then move and finalize, or construct on device and
    stream checkpoint tensors on CPU. Finalize is required before compile.
    The model cache planner binds ``kv_cache=(history,)`` or ``bind_history``;
    metadata is passed once per step, never retrieved from runner state.
    """

    def __init__(self,
                 *,
                 parallel: PLEParallel,
                 device=None,
                 prefix: str = "",
                 eos_token_id: int = 248044,
                 cpu_offload: bool = False,
                 shared=None):
        super().__init__()
        self.parallel = parallel
        self.cpu_offload = cpu_offload
        self.prefix, self.eos_token_id = prefix, eos_token_id
        self.ple_embedding = _HashEmbedding(parallel.rank,
                                            parallel.world_size,
                                            device,
                                            cpu_offload=cpu_offload,
                                            shared=shared)
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
        self._host_ids = None

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
        self._values, self._conv = ple_kernels.build_values_op(
        ), ple_kernels.build_short_conv_op()
        self._finalized = True

        if self.cpu_offload:
            self._host_ids = torch.compile(self._prepare_host_ids,
                                           backend="tpu",
                                           fullgraph=True,
                                           dynamic=False)

    def _prepare_host_ids(self, input_ids, starts, context, valid_tokens):
        ids = self.ple_embedding(input_ids, starts, context, self.eos_token_id,
                                 valid_tokens)
        return self.parallel.gather_ids(ids)

    def prepare_host(self, input_ids, starts, context, valid_tokens):
        """Finish ID D2H, gather immutable CPU rows, transfer only this batch."""
        ids = self._host_ids(input_ids, starts, context, valid_tokens).cpu()
        raw = self.ple_embedding.ngram_embedding.lookup_host(ids)
        # vLLM specializes every free shape symbol to the local token bucket.
        # Keep DP static and expose only T, never an independent D*T symbol.
        return raw.reshape(-1, input_ids.shape[0], 2560).to(input_ids.device)

    def forward(self,
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
                state_indices,
                has_initial_states,
                valid_tokens,
                ple_raw_embeddings=None):
        assert self._finalized, "PLE finalize_weights must run before inference"
        assert len(
            self.kv_cache) == 1, "PLE history must be bound before inference"
        embedding = self.ple_embedding.ngram_embedding
        if self.cpu_offload:
            assert ple_raw_embeddings is not None, "Host PLE requires prepared raw rows"
            # Collectives mutate dtype views. Own their buffer inside the
            # compiled graph: mutating an external INT16 argument through an
            # INT32 view corrupts the mixed TP/DP graph's input update.
            raw = ple_raw_embeddings.reshape(-1, 2560).clone()
        else:
            ids = self.ple_embedding(input_ids, query_start_loc, ngram_context,
                                     self.eos_token_id, valid_tokens)
            ids = self.parallel.gather_ids(ids)
            raw = embedding.lookup_raw(ids)
        raw = self.parallel.combine(raw, input_ids.shape[0])
        embeddings = embedding.decode(raw)
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
