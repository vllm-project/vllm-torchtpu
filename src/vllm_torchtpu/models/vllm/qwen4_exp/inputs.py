# SPDX-License-Identifier: Apache-2.0
"""One reordered token snapshot and shared GDN/QSA/PLE ragged geometry."""
import math
import time

import torch
from vllm.forward_context import set_forward_context

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.utils import synchronize_tensors

from .cache import Qwen4ExpCache


def ngram_context_from_store(token_store,
                             row_indices,
                             context_lens,
                             eos_token_id=248044):
    columns = context_lens[:, None] + torch.tensor(
        [-2, -1], dtype=torch.int64, device=token_store.device)
    gathered = token_store[row_indices[:, None], columns.clamp(min=0)]
    return torch.where(columns >= 0, gathered.to(torch.int64), eos_token_id)


def dummy_query_lengths(num_tokens, num_reqs, max_active):
    active = min(num_tokens, num_reqs, max_active)
    lengths = [1] * active + [0] * (num_reqs - active)
    lengths[active - 1] += num_tokens - active
    return lengths


class Qwen4ExpAdapter:

    def __init__(self, runner):
        self.runner = runner
        self.cache = None
        self.token_store = None

    def bind_model(self, model):
        self.cache = Qwen4ExpCache(self.runner, model)

    def get_kv_cache_spec(self):
        return self.cache.get_specs()

    def initialize_kv_cache(self, config):
        self.cache.initialize(config)

    def begin_step(self):
        # Exactly one immutable H2D snapshot after runner request reordering.
        # Row numbers in this store are unrelated to state slots/page IDs.
        batch = self.runner.input_batch
        self.token_store = batch.token_ids_cpu_tensor.to(self.runner.device,
                                                         non_blocking=True)
        return self.token_store

    def prepare_inputs(self, snapshot):
        runner = self.runner
        ctx = runner._attn_metadata_builder_ctx
        starts = ctx.query_start_loc
        rows = starts.shape[0] - 1
        context = (ctx.seq_lens - (starts[1:] - starts[:-1])).to(torch.int64)
        valid_rows = torch.arange(rows,
                                  device=runner.device) < snapshot.num_requests
        slots = torch.where(valid_rows, ctx.mamba_state_indices,
                            -1).to(torch.int32)
        row_ids = torch.arange(rows, dtype=torch.int64,
                               device=runner.device) + snapshot.row_start
        row_ids = row_ids.clamp(max=snapshot.token_store.shape[0] - 1)
        history = ngram_context_from_store(snapshot.token_store, row_ids,
                                           context)
        tables = []
        for role in ("self_attn", "self_attn.compressed_k"):
            name = f"model.layers.3.{role}"
            # Metadata uses the genuine native group table, never state slots.
            table = snapshot.attention_metadata[name].block_tables.reshape(
                rows, -1)
            tables.append(table)
        return dict(query_start_loc=starts,
                    ngram_context=history,
                    state_slots=slots,
                    context_lens=context,
                    valid_tokens=torch.arange(snapshot.num_tokens,
                                              device=runner.device)
                    < starts[-1],
                    main_block_table=tables[0],
                    compressed_block_table=tables[1])

    def dummy_run(self,
                  num_tokens,
                  num_reqs,
                  profile_prefix=None,
                  *,
                  dp_lockstep=False):
        runner = self.runner
        if dp_lockstep:
            return self._idle_run(num_tokens, num_reqs)
        assert runner.input_batch.num_reqs == 0, "S1 dummy calls require no live requests"
        self.cache.allocate()
        lengths = dummy_query_lengths(num_tokens, num_reqs,
                                      runner.scheduler_config.max_num_seqs)
        starts_cpu = torch.tensor([0] + lengths,
                                  dtype=torch.int32,
                                  device="cpu").cumsum(0, dtype=torch.int32)
        context_cpu = torch.zeros(num_reqs, dtype=torch.int64, device="cpu")
        if profile_prefix is not None:
            context_cpu[0] = profile_prefix
        if max(context_cpu.tolist()[i] + q
               for i, q in enumerate(lengths)) > runner.max_model_len:
            raise ValueError("Dummy context exceeds admitted Qwen4Exp context")
        active = sum(q > 0 for q in lengths)
        slots_cpu = torch.tensor(list(range(1, active + 1)) + [-1] *
                                 (num_reqs - active),
                                 dtype=torch.int32,
                                 device="cpu")
        positions_cpu = torch.cat([
            torch.arange(int(context_cpu[i]),
                         int(context_cpu[i]) + q,
                         dtype=torch.int32,
                         device="cpu") for i, q in enumerate(lengths)
        ])
        pages_per_req = math.ceil(runner.max_model_len / 128)
        tables_cpu = torch.zeros((num_reqs, pages_per_req),
                                 dtype=torch.int32,
                                 device="cpu")
        next_page = 1
        for i, q in enumerate(lengths):
            count = math.ceil((int(context_cpu[i]) + q) / 128)
            tables_cpu[i, :count] = torch.arange(next_page,
                                                 next_page + count,
                                                 dtype=torch.int32,
                                                 device="cpu")
            next_page += count
        assert next_page <= self.cache.pages, "Dummy pages exceed allocated scratch capacity"
        starts, context, slots, positions, table = [
            t.to(runner.device) for t in (starts_cpu, context_cpu, slots_cpu,
                                          positions_cpu, tables_cpu)
        ]
        seq_lens = (context + starts[1:] - starts[:-1]).to(torch.int32)
        # All profile/capture tokens do real work. Fresh requests use prefill
        # recurrence; a one-token query remains a valid mixed-path query.
        distribution = torch.tensor([0, 0, active],
                                    dtype=torch.int32,
                                    device=runner.device)
        metadata = AttentionMetadata(input_positions=positions,
                                     block_tables=table.reshape(-1),
                                     seq_lens=seq_lens,
                                     query_start_loc=starts,
                                     request_distribution=distribution,
                                     mamba_state_indices=slots.clamp(min=0))
        self.token_store = torch.zeros_like(
            runner.input_batch.token_ids_cpu_tensor,
            device="cpu").to(runner.device)
        history = ngram_context_from_store(
            self.token_store,
            torch.arange(num_reqs, device=runner.device, dtype=torch.int64),
            context)
        kwargs = dict(query_start_loc=starts,
                      ngram_context=history,
                      state_slots=slots,
                      context_lens=context,
                      valid_tokens=torch.ones(num_tokens,
                                              dtype=torch.bool,
                                              device=runner.device),
                      main_block_table=table,
                      compressed_block_table=table)
        ids = (
            torch.arange(num_tokens, dtype=torch.int32, device=runner.device) *
            37 + 17) % 248320
        return self._dummy_forward(ids, positions, kwargs, metadata,
                                   num_tokens, profile_prefix)

    def _idle_run(self, num_tokens, num_reqs):
        """Participate in remote work without touching local live owners."""
        device = self.runner.device
        starts = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        context = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        slots = torch.full((num_reqs, ), -1, dtype=torch.int32, device=device)
        positions = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        table = torch.zeros(
            (num_reqs, math.ceil(self.runner.max_model_len / 128)),
            dtype=torch.int32,
            device=device)
        metadata = AttentionMetadata(
            input_positions=positions,
            block_tables=table.reshape(-1),
            seq_lens=torch.zeros(num_reqs, dtype=torch.int32, device=device),
            query_start_loc=starts,
            request_distribution=torch.zeros(3,
                                             dtype=torch.int32,
                                             device=device),
            mamba_state_indices=torch.zeros_like(slots))
        kwargs = dict(query_start_loc=starts,
                      ngram_context=torch.full((num_reqs, 2),
                                               248044,
                                               dtype=torch.int64,
                                               device=device),
                      state_slots=slots,
                      context_lens=context,
                      valid_tokens=torch.zeros(num_tokens,
                                               dtype=torch.bool,
                                               device=device),
                      main_block_table=table,
                      compressed_block_table=table)
        return self._dummy_forward(torch.zeros_like(positions),
                                   positions,
                                   kwargs,
                                   metadata,
                                   num_tokens,
                                   None,
                                   dp_lockstep=True)

    def _dummy_forward(self,
                       ids,
                       positions,
                       kwargs,
                       metadata,
                       num_tokens,
                       profile_prefix,
                       *,
                       dp_lockstep=False):
        runner = self.runner
        per_layer = {name: metadata for name in self.cache.specs}
        with set_forward_context(per_layer, runner.vllm_config, num_tokens=num_tokens,
                                 num_tokens_across_dp=runner._dp_num_tokens_across_dp(num_tokens)), \
                set_vllm_model_wrapper_context(mesh=runner.mesh, vllm_config=runner.vllm_config):
            started = time.perf_counter()
            out, _ = runner.forward_model(input_ids=ids,
                                          positions=positions,
                                          model_kwargs=kwargs)
            if dp_lockstep and runner.parallel_config.tensor_parallel_size > 1:
                # Match the generic runner's idle-DP protocol. The LM head
                # has a TP collective even when this DP has no live tokens;
                # omitting it leaves busy DP engines waiting on their logits.
                indices = torch.zeros(kwargs["state_slots"].shape[0],
                                      dtype=torch.int32,
                                      device=out.device)
                runner.compute_selected_logits(out, indices)
            else:
                synchronize_tensors(out)
        runner._hidden_states_dtype = out.dtype
        return (time.perf_counter() -
                started) * 1000 if profile_prefix is not None else None
