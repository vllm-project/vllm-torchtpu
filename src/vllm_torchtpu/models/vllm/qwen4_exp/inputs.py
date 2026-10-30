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
"""One reordered token snapshot and shared GDN/QSA/PLE ragged geometry."""

import math
import time
from typing import TYPE_CHECKING

import torch
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.kv_cache_materializer import LayerKVCache
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import synchronize_tensors

from .config import require_unified_pool

if TYPE_CHECKING:
    from .pooled_cache import Qwen4ExpPooledCache


def ngram_context_from_store(
    token_store, row_indices, context_lens, eos_token_id, ngram_size
):
    columns = context_lens[:, None] + torch.arange(
        1 - ngram_size, 0, dtype=torch.int64, device=token_store.device
    )
    gathered = token_store[row_indices[:, None], columns.clamp(min=0)]
    return torch.where(columns >= 0, gathered.to(torch.int64), eos_token_id)


def dummy_query_lengths(num_tokens, num_reqs, max_active, capacities):
    active = min(num_tokens, num_reqs, max_active)
    lengths = [int(i < active and capacities[i] > 0) for i in range(num_reqs)]
    remaining = num_tokens - sum(lengths)
    for i in reversed(range(active)):
        count = min(remaining, max(0, capacities[i] - lengths[i]))
        lengths[i] += count
        remaining -= count
    return lengths


class Qwen4ExpAdapter:
    def __init__(self, runner):
        self.runner = runner
        require_unified_pool()
        self.config = runner.model_config.hf_text_config
        self.block_size = runner.cache_config.block_size
        self.cache: Qwen4ExpPooledCache | None = None
        self.ngram_context = None

    def bind_model(self, model):
        from .pooled_cache import Qwen4ExpPooledCache

        self.cache = Qwen4ExpPooledCache(self.runner, model)

    def get_kv_cache_spec(self):
        assert self.cache is not None
        return self.cache.get_specs()

    def initialize_kv_cache(self, config) -> dict[str, LayerKVCache] | None:
        assert self.cache is not None
        return self.cache.initialize(config)

    def begin_step(self):
        # Reordering is complete. The pending async sample is the current input,
        # substituted on device; predecessors below num_computed are committed.
        batch = self.runner.input_batch
        count = batch.num_reqs if self.config.ple_layer_ids else 0
        context = torch.as_tensor(
            batch.num_computed_tokens_cpu[:count], dtype=torch.int64
        )
        history = ngram_context_from_store(
            batch.token_ids_cpu_tensor,
            torch.arange(count, dtype=torch.int64),
            context,
            self.config.eos_token_id,
            self.config.ngram_size,
        )
        # Gather owns its CPU storage, so later commits/reuse cannot mutate this
        # step's snapshot while its nonblocking transfer or later chunks run.
        self.ngram_context = history.to(self.runner.device, non_blocking=True)
        return self.ngram_context

    def prepare_inputs(self, snapshot):
        runner = self.runner
        ctx = runner._attn_metadata_builder_ctx
        starts = ctx.query_start_loc
        rows = starts.shape[0] - 1
        context = (ctx.seq_lens - (starts[1:] - starts[:-1])).to(torch.int64)
        valid_rows = torch.arange(rows, device=runner.device) < snapshot.num_requests
        ngram_context = snapshot.step_context
        if ngram_context.shape[0]:
            row_ids = (
                torch.arange(rows, dtype=torch.int64, device=runner.device)
                + snapshot.row_start
            ).clamp(max=ngram_context.shape[0] - 1)
            history = ngram_context[row_ids]
            history = torch.where(
                valid_rows[:, None], history, self.config.eos_token_id
            )
        else:
            history = torch.full(
                (rows, self.config.ngram_size - 1),
                self.config.eos_token_id,
                dtype=torch.int64,
                device=runner.device,
            )
        kwargs = dict(
            query_start_loc=starts,
            ngram_context=history,
            context_lens=context,
            valid_tokens=torch.arange(snapshot.num_tokens, device=runner.device)
            < starts[-1],
        )
        return kwargs

    def dummy_run(
        self,
        num_tokens,
        num_reqs,
        profile_prefix=None,
        *,
        num_blocks=None,
        use_max_model_len=True,
        dp_lockstep=False,
        through_handoff=False,
    ):
        runner = self.runner
        context_limit = (
            runner.max_model_len if use_max_model_len else runner.most_model_len
        )
        columns = (
            math.ceil(context_limit / self.block_size)
            if num_blocks is None
            else num_blocks
        )
        if dp_lockstep:
            return self._idle_run(
                num_tokens,
                num_reqs,
                columns,
                use_max_model_len=use_max_model_len,
                through_handoff=through_handoff,
            )
        assert runner.input_batch.num_reqs == 0, (
            "S1 dummy calls require no live requests"
        )
        assert self.cache is not None
        self.cache.allocate()
        context_cpu = torch.zeros(num_reqs, dtype=torch.int64, device="cpu")
        if profile_prefix is not None:
            if not 0 <= profile_prefix < context_limit:
                raise ValueError("Profiling prefix must leave room for query tokens")
            context_cpu[0] = profile_prefix
        lengths = dummy_query_lengths(
            num_tokens,
            num_reqs,
            runner.scheduler_config.max_num_seqs,
            (context_limit - context_cpu).tolist(),
        )
        starts_cpu = torch.tensor(
            [0] + lengths, dtype=torch.int32, device="cpu"
        ).cumsum(0, dtype=torch.int32)
        active = sum(q > 0 for q in lengths)
        slots_cpu = torch.tensor(
            list(range(1, active + 1)) + [-1] * (num_reqs - active),
            dtype=torch.int32,
            device="cpu",
        )
        positions_cpu = torch.cat(
            [
                torch.arange(
                    int(context_cpu[i]),
                    int(context_cpu[i]) + q,
                    dtype=torch.int32,
                    device="cpu",
                )
                for i, q in enumerate(lengths)
            ]
        )
        positions_cpu = torch.nn.functional.pad(
            positions_cpu, (0, num_tokens - sum(lengths))
        )
        tables_cpu = torch.zeros((num_reqs, columns), dtype=torch.int32, device="cpu")
        next_page = 1
        for i, q in enumerate(lengths):
            count = math.ceil((int(context_cpu[i]) + q) / self.block_size)
            tables_cpu[i, :count] = torch.arange(
                next_page, next_page + count, dtype=torch.int32, device="cpu"
            )
            next_page += count
        assert next_page <= self.cache.pages, (
            "Dummy pages exceed allocated scratch capacity"
        )
        starts, context, slots, positions, table = [
            t.to(runner.device)
            for t in (starts_cpu, context_cpu, slots_cpu, positions_cpu, tables_cpu)
        ]
        seq_lens = (context + starts[1:] - starts[:-1]).to(torch.int32)
        # All profile/capture tokens do real work. Fresh requests use prefill
        # recurrence; a one-token query remains a valid mixed-path query.
        distribution = torch.tensor(
            [0, 0, active], dtype=torch.int32, device=runner.device
        )
        metadata = AttentionMetadata(
            input_positions=positions,
            block_tables=table.reshape(-1),
            seq_lens=seq_lens,
            query_start_loc=starts,
            request_distribution=distribution,
            mamba_state_indices=slots.clamp(min=0),
        )
        columns = context[:, None] + torch.arange(
            1 - self.config.ngram_size, 0, dtype=torch.int64, device=runner.device
        )
        history = torch.where(columns >= 0, 0, self.config.eos_token_id)
        self.ngram_context = history
        kwargs = dict(
            query_start_loc=starts,
            ngram_context=history,
            context_lens=context,
            valid_tokens=torch.arange(num_tokens, device=runner.device) < starts[-1],
        )
        ids = (
            torch.arange(num_tokens, dtype=torch.int32, device=runner.device) * 37 + 17
        ) % self.config.vocab_size
        return self._dummy_forward(
            ids,
            positions,
            kwargs,
            metadata,
            num_tokens,
            profile_prefix,
            active_rows=active,
            use_max_model_len=use_max_model_len,
            through_handoff=through_handoff,
        )

    def _idle_run(
        self, num_tokens, num_reqs, columns, *, use_max_model_len, through_handoff
    ):
        """Participate in remote work without touching local live owners."""
        device = self.runner.device
        starts = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        context = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        slots = torch.full((num_reqs,), -1, dtype=torch.int32, device=device)
        positions = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        table = torch.zeros(
            (num_reqs, columns),
            dtype=torch.int32,
            device=device,
        )
        metadata = AttentionMetadata(
            input_positions=positions,
            block_tables=table.reshape(-1),
            seq_lens=torch.zeros(num_reqs, dtype=torch.int32, device=device),
            query_start_loc=starts,
            request_distribution=torch.zeros(3, dtype=torch.int32, device=device),
            mamba_state_indices=torch.zeros_like(slots),
        )
        kwargs = dict(
            query_start_loc=starts,
            ngram_context=torch.full(
                (num_reqs, self.config.ngram_size - 1),
                self.config.eos_token_id,
                dtype=torch.int64,
                device=device,
            ),
            context_lens=context,
            valid_tokens=torch.zeros(num_tokens, dtype=torch.bool, device=device),
        )
        return self._dummy_forward(
            torch.zeros_like(positions),
            positions,
            kwargs,
            metadata,
            num_tokens,
            None,
            dp_lockstep=True,
            use_max_model_len=use_max_model_len,
            through_handoff=through_handoff,
        )

    def _dummy_forward(
        self,
        ids,
        positions,
        kwargs,
        metadata,
        num_tokens,
        profile_prefix,
        *,
        dp_lockstep=False,
        active_rows=0,
        use_max_model_len=True,
        through_handoff=False,
    ):
        runner = self.runner
        assert self.cache is not None
        per_layer = self.cache.dummy_metadata(
            metadata, active_rows, use_max_model_len=use_max_model_len
        )
        intermediate_tensors = None
        if through_handoff:
            assert runner._pp_wave is not None and profile_prefix is None
            intermediate_tensors = runner._pp_wave.start_forward(num_tokens)
            runner._pp_take_topk_indices(intermediate_tensors, num_tokens)
        elif not runner._pp_is_first:
            intermediate_tensors = runner._pp_intermediate_tensors(
                num_tokens,
                zeros=profile_prefix is None,
                random=profile_prefix is not None,
            )
            runner._pp_take_topk_indices(intermediate_tensors, num_tokens)
        inputs_embeds = None
        if runner.supports_mm_inputs and runner._pp_is_first:
            ids, inputs_embeds = runner._get_model_inputs(ids, None)
        runner._record_input_staging_fence()
        with (
            set_forward_context(
                per_layer,
                runner.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=runner._dp_num_tokens_across_dp(num_tokens),
            ),
            set_vllm_model_wrapper_context(
                mesh=runner.mesh, vllm_config=runner.vllm_config
            ),
        ):
            started = time.perf_counter()
            out, _ = runner.forward_model(
                input_ids=ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
                intermediate_tensors=intermediate_tensors,
                model_kwargs=kwargs,
            )
            if isinstance(out, IntermediateTensors):
                if through_handoff:
                    assert runner._pp_wave is not None
                    runner._pp_wave.end_forward(
                        runner._pp_outgoing_tensors(out.tensors, num_tokens), num_tokens
                    )
                    synchronize_tensors(None, wait=False)
                    return None
                synchronize_tensors(list(out.tensors.values()))
                return (
                    (time.perf_counter() - started) * 1000
                    if profile_prefix is not None
                    else None
                )
            if through_handoff:
                assert runner._pp_wave is not None
                runner._pp_wave.end_forward(None, num_tokens)
            if dp_lockstep and runner.parallel_config.tensor_parallel_size > 1:
                # Match the generic runner's idle-DP protocol. The LM head
                # has a TP collective even when this DP has no live tokens;
                # omitting it leaves busy DP engines waiting on their logits.
                indices = torch.zeros(
                    metadata.query_start_loc.shape[0] - 1,
                    dtype=torch.int32,
                    device=out.device,
                )
                runner.compute_selected_logits(out, indices)
            else:
                synchronize_tensors(out)
        runner._hidden_states_dtype = out.dtype
        return (
            (time.perf_counter() - started) * 1000
            if profile_prefix is not None
            else None
        )
