# SPDX-License-Identifier: Apache-2.0
"""One reordered token snapshot and shared GDN/QSA/PLE ragged geometry."""

import math
import time
from typing import TYPE_CHECKING, cast

import torch
from vllm.forward_context import set_forward_context

from vllm_torchtpu import envs
from vllm_torchtpu.kv_cache_materializer import LayerKVCache
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.utils import synchronize_tensors

from .cache import Qwen4ExpCache
from .config import mtp_layer_prefixes, num_speculative_tokens, use_unified_pool

if TYPE_CHECKING:
    from .pooled_cache import Qwen4ExpPooledCache

logger = init_logger(__name__)


def ngram_context_from_store(
    token_store, row_indices, context_lens, eos_token_id, ngram_size
):
    columns = context_lens[:, None] + torch.arange(
        1 - ngram_size, 0, dtype=torch.int64, device=token_store.device
    )
    gathered = token_store[row_indices[:, None], columns.clamp(min=0)]
    return torch.where(columns >= 0, gathered.to(torch.int64), eos_token_id)


def ngram_window_from_host(
    token_ids, num_computed, row_start, num_requests, rows, width, eos_token_id
):
    """Host side of the windowed n-gram context: per live row the last ``width`` host-store tokens
    before its host context ``num_computed[row_start + i]`` (async spec decoding advances the host
    context optimistically; the device context is at most ``width - ngram_size + 1`` tokens shorter),
    as one ``[rows, width + 1]`` int64 array whose last column is the window's first position."""
    import numpy as np

    out = np.zeros((rows, width + 1), dtype=np.int64)
    if num_requests:
        ctx = np.asarray(num_computed[row_start : row_start + num_requests], dtype=np.int64)
        lo = ctx - width
        cols = lo[:, None] + np.arange(width)
        vals = token_ids[np.arange(row_start, row_start + num_requests)[:, None], np.clip(cols, 0, None)]
        out[:num_requests, :width] = np.where(cols >= 0, vals, eos_token_id)
        out[:num_requests, width] = lo
    return out


def ngram_context_from_window(window, context_lens, eos_token_id, ngram_size):
    """``ngram_context_from_store`` over ``ngram_window_from_host``'s device copy."""
    width = window.shape[1] - 1
    columns = context_lens[:, None] + torch.arange(
        1 - ngram_size, 0, dtype=torch.int64, device=window.device
    )
    index = (columns - window[:, width:]).clamp(0, width - 1)
    gathered = torch.gather(window[:, :width], 1, index)
    return torch.where(columns >= 0, gathered, eos_token_id)


def merge_recent_ngram_context(
    history, recent_tokens, recent_context, recent_lengths, previous_rows, context
):
    """Replace host-store context tokens the previous step saw on device only.

    With async scheduling and speculative decoding, the drafts a step accepts
    reach the host token store only after the following step has prepared its
    inputs, so the store holds placeholders at positions ``(c, c + accepted]``
    of the previous chunk ``c``. ``recent_tokens[r]`` is that previous chunk's
    n-gram context followed by its first chunk tokens (as verified on
    device), ``recent_context[r]`` its context length and ``recent_lengths[r]``
    how many chunk tokens it holds. Context positions inside that span read
    it; all others keep the host value.

    ``previous_rows[i]`` is current row ``i``'s row in the previous step's
    record, or -1 when the request did not run in the previous step.
    """
    width = history.shape[1]
    rows = previous_rows.clamp(min=0)
    tokens = recent_tokens[rows]
    base = recent_context[rows]
    columns = torch.arange(width, device=history.device, dtype=torch.int64)
    positions = context[:, None] - width + columns[None]
    index = positions - base[:, None] + width
    usable = (
        (previous_rows >= 0)[:, None]
        & (positions >= 0)
        & (index >= 0)
        & (index < width + recent_lengths[rows][:, None])
    )
    gathered = torch.gather(tokens, 1, index.clamp(0, tokens.shape[1] - 1))
    return torch.where(usable, gathered, history)


def dummy_query_lengths(num_tokens, num_reqs, max_active):
    active = min(num_tokens, num_reqs, max_active)
    lengths = [1] * active + [0] * (num_reqs - active)
    lengths[active - 1] += num_tokens - active
    return lengths


class Qwen4ExpAdapter:
    def __init__(self, runner):
        self.runner = runner
        self.pooled = use_unified_pool()
        self.config = runner.model_config.hf_text_config
        self.block_size = runner.cache_config.block_size
        self.attention_prefix = (
            f"model.layers.{self.config.layer_types.index('full_attention')}"
        )
        self.cache: Qwen4ExpCache | Qwen4ExpPooledCache | None = None
        self.token_store = None
        # Speculative decoding: each step's verified chunk tokens and n-gram
        # context, kept on device for the next step's n-gram history (see
        # merge_recent_ngram_context). Rows are batch rows of that step.
        spec = num_speculative_tokens(getattr(runner, "vllm_config", None))
        self.spec_window = spec + 1 if spec else 0
        # Two records: the previous step's (read) and this step's (written),
        # so a later chunk of this step never reads this step's rows.
        self._recent: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._previous_rows: dict[str, int] = {}
        self._current_rows: dict[str, int] = {}

    def bind_model(self, model):
        if self.pooled:
            from .pooled_cache import Qwen4ExpPooledCache

            self.cache = Qwen4ExpPooledCache(self.runner, model)
        else:
            self.cache = Qwen4ExpCache(self.runner, model)

    def bind_draft(self, drafter):
        """Attach the MTP proposer once the draft model has loaded.

        The draft's QSA layer reads its compressed-key and raw-ring group
        metadata by name, so the proposer must carry those entries (in step
        0's target metadata and through its loop-step rebuilds); it only
        discovers the main attention name on its own.
        """
        names = getattr(drafter, "_draft_attn_layer_names", None)
        if names is not None:
            for prefix in mtp_layer_prefixes(self.runner.vllm_config):
                attention = prefix + ".self_attn"
                if attention in names:
                    names.update(
                        {self._compressed_owner(attention), attention + ".raw_ring"}
                    )
        if self.pooled and self.cache is not None:
            cast("Qwen4ExpPooledCache", self.cache).bind_draft()

    def _compressed_owner(self, attention: str) -> str:
        """Cache group name addressing a QSA layer's compressed keys.

        Packed pools (``QWEN4_EXP_PACK_COMPRESSED_K``) store the MTP draft's
        keys in a group shared with target layers.
        """
        layout = getattr(self.cache, "layout", None) if self.pooled else None
        if layout is None:
            return attention + ".compressed_k"
        return layout.compressed_owner(attention)[0]

    def _main_owner(self, attention: str) -> str:
        """Cache group name addressing a QSA layer's main KV.

        Paired pools (``QWEN4_EXP_PACK_MAIN_KV``) store a layer's main KV in
        the group of the first layer of its pair.
        """
        layout = getattr(self.cache, "layout", None) if self.pooled else None
        if layout is None:
            return attention
        return layout.main_owner(attention)[0]

    def get_kv_cache_spec(self):
        assert self.cache is not None
        return self.cache.get_specs()

    def initialize_kv_cache(self, config) -> dict[str, LayerKVCache] | None:
        assert self.cache is not None
        return self.cache.initialize(config)

    def kv_cache_memory_budget(self) -> int | None:
        """Bytes to report for the KV cache after profiling, or None.

        The unified pool under HBM sizing reports its own budget; every
        other configuration keeps the worker's default accounting.
        """
        cache = self.cache
        if not self.pooled or cache is None:
            return None
        pooled = cast("Qwen4ExpPooledCache", cache)
        return pooled.memory_budget() if pooled.hbm_sizing else None

    def begin_step(self):
        # Exactly one immutable H2D snapshot after runner request reordering.
        # Row numbers in this store are unrelated to state slots/page IDs.
        batch = self.runner.input_batch
        mode = envs.VLLM_TPU_QWEN4_NGRAM_WINDOW
        if mode == "1" or (mode == "check" and getattr(self, "_ngram_check_left", 200) <= 0):
            # Windowed n-gram context: prepare_inputs copies only the rows' windows (`check`:
            # once its comparisons are done, it runs exactly like `1`).
            self.token_store = batch.token_ids_cpu_tensor
        else:
            self.token_store = batch.token_ids_cpu_tensor.to(
                self.runner.device, non_blocking=True
            )
        if self.spec_window:
            self._previous_rows, self._current_rows = self._current_rows, {}
            self._recent.reverse()
        return self.token_store

    def _recent_buffers(self, device):
        """(previous step's record, this step's record)."""
        if not self._recent:
            # One spare row absorbs the writes of padded batch rows.
            rows = self.runner.max_num_reqs + 1
            width = self.config.ngram_size - 1 + self.spec_window
            self._recent = [
                (
                    torch.zeros((rows, width), dtype=torch.int64, device=device),
                    torch.zeros(rows, dtype=torch.int64, device=device),
                    torch.zeros(rows, dtype=torch.int64, device=device),
                )
                for _ in range(2)
            ]
        return self._recent[0], self._recent[1]

    def _speculative_ngram_context(self, snapshot, history, context, starts):
        """Correct and record n-gram context across speculative steps."""
        runner = self.runner
        rows = starts.shape[0] - 1
        req_ids = runner.input_batch.req_ids
        previous_np = torch.full((rows,), -1, dtype=torch.int64)
        for i in range(snapshot.num_requests):
            req_id = req_ids[snapshot.row_start + i]
            previous_np[i] = self._previous_rows.get(req_id, -1)
            self._current_rows[req_id] = snapshot.row_start + i
        previous, current = self._recent_buffers(runner.device)
        history = merge_recent_ngram_context(
            history,
            *previous,
            previous_np.to(runner.device, non_blocking=True),
            context,
        )
        tokens, recent_context, lengths = current
        # Record this chunk for the next step: corrected context, then its
        # first spec_window tokens (all a verify window can accept).
        ids = snapshot.input_ids.to(torch.int64)
        offsets = torch.arange(self.spec_window, device=ids.device)
        chunk = ids[
            (starts[:-1, None].to(torch.int64) + offsets).clamp(0, ids.shape[0] - 1)
        ]
        live = torch.arange(rows, device=ids.device) < snapshot.num_requests
        target = torch.where(
            live,
            torch.arange(rows, device=ids.device) + snapshot.row_start,
            tokens.shape[0] - 1,
        )
        lengths_now = (starts[1:] - starts[:-1]).to(torch.int64)
        tokens.index_copy_(0, target, torch.cat([history, chunk], dim=1))
        recent_context.index_copy_(0, target, context)
        lengths.index_copy_(0, target, lengths_now.clamp(max=self.spec_window))
        return history

    def prepare_inputs(self, snapshot):
        runner = self.runner
        ctx = runner._attn_metadata_builder_ctx
        starts = ctx.query_start_loc
        rows = starts.shape[0] - 1
        context = (ctx.seq_lens - (starts[1:] - starts[:-1])).to(torch.int64)
        valid_rows = torch.arange(rows, device=runner.device) < snapshot.num_requests
        state_indices = (
            torch.arange(rows, dtype=torch.int32, device=runner.device)
            if self.pooled
            else ctx.mamba_state_indices
        )
        slots = torch.where(valid_rows, state_indices, -1).to(torch.int32)
        row_ids = (
            torch.arange(rows, dtype=torch.int64, device=runner.device)
            + snapshot.row_start
        )
        row_ids = row_ids.clamp(max=snapshot.token_store.shape[0] - 1)
        mode = envs.VLLM_TPU_QWEN4_NGRAM_WINDOW
        if mode in ("1", "check"):
            # Device context >= host context - spec tokens: a window of spec + ngram - 1 tokens.
            width = max(self.spec_window - 1, 0) + self.config.ngram_size - 1
            window = ngram_window_from_host(
                runner.input_batch.token_ids_cpu_tensor.numpy(),
                runner.input_batch.num_computed_tokens_cpu,
                snapshot.row_start,
                snapshot.num_requests,
                rows,
                width,
                self.config.eos_token_id,
            )
            history = ngram_context_from_window(
                torch.from_numpy(window).to(runner.device, non_blocking=True),
                context,
                self.config.eos_token_id,
                self.config.ngram_size,
            )
            if mode == "check" and snapshot.token_store.device.type != "cpu":
                self._check_ngram_window(snapshot, row_ids, context, history)
        else:
            history = ngram_context_from_store(
                snapshot.token_store,
                row_ids,
                context,
                self.config.eos_token_id,
                self.config.ngram_size,
            )
        if self.spec_window:
            history = self._speculative_ngram_context(
                snapshot, history, context, starts
            )
        tables = []
        attention = f"{self.attention_prefix}.self_attn"
        for name in (self._main_owner(attention), self._compressed_owner(attention)):
            # Metadata uses the genuine native group table, never state slots.
            table = snapshot.attention_metadata[name].block_tables.reshape(rows, -1)
            tables.append(table)
        kwargs = dict(
            query_start_loc=starts,
            ngram_context=history,
            state_slots=slots,
            context_lens=context,
            valid_tokens=torch.arange(snapshot.num_tokens, device=runner.device)
            < starts[-1],
            main_block_table=tables[0],
            compressed_block_table=tables[1],
        )
        return kwargs

    def _check_ngram_window(self, snapshot, row_ids, context, history):
        """``VLLM_TPU_QWEN4_NGRAM_WINDOW=check``: compare the windowed context with the full-store
        one on the live rows for the first 200 steps (one sync each), log the mismatches."""
        left = getattr(self, "_ngram_check_left", 200)
        if left <= 0 or snapshot.num_requests == 0:
            return
        self._ngram_check_left = left - 1
        ref = ngram_context_from_store(
            snapshot.token_store, row_ids, context, self.config.eos_token_id, self.config.ngram_size
        )
        n = snapshot.num_requests
        bad = int((ref[:n] != history[:n]).any(dim=1).sum().item())
        self._ngram_check_bad = getattr(self, "_ngram_check_bad", 0) + bad
        if bad or left in (200, 1):
            logger.info(
                "qwen4 ngram window check: %d of %d rows differ this step, %d total, %d checks left",
                bad, n, self._ngram_check_bad, left - 1,
            )

    def dummy_run(
        self, num_tokens, num_reqs, profile_prefix=None, *, dp_lockstep=False
    ):
        runner = self.runner
        if dp_lockstep:
            return self._idle_run(num_tokens, num_reqs)
        assert runner.input_batch.num_reqs == 0, (
            "S1 dummy calls require no live requests"
        )
        assert self.cache is not None
        lengths = dummy_query_lengths(
            num_tokens, num_reqs, runner.scheduler_config.max_num_seqs
        )
        if self.pooled:
            # Before the KV cache is planned this sizes the profiling
            # scratch pool to exactly what the dummy forward addresses.
            seq_lens = [
                q + (profile_prefix or 0) * (i == 0) for i, q in enumerate(lengths)
            ]
            pooled = cast("Qwen4ExpPooledCache", self.cache)
            pooled.allocate(
                max(
                    pooled.dummy_pages(seq_lens, sum(q > 0 for q in lengths)),
                    1 + sum(math.ceil(n / self.block_size) for n in seq_lens),
                )
            )
        else:
            self.cache.allocate()
        starts_cpu = torch.tensor(
            [0] + lengths, dtype=torch.int32, device="cpu"
        ).cumsum(0, dtype=torch.int32)
        context_cpu = torch.zeros(num_reqs, dtype=torch.int64, device="cpu")
        if profile_prefix is not None:
            context_cpu[0] = profile_prefix
        if (
            max(context_cpu.tolist()[i] + q for i, q in enumerate(lengths))
            > runner.max_model_len
        ):
            raise ValueError("Dummy context exceeds admitted Qwen4Exp context")
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
        pages_per_req = math.ceil(runner.max_model_len / self.block_size)
        tables_cpu = torch.zeros(
            (num_reqs, pages_per_req), dtype=torch.int32, device="cpu"
        )
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
        self.token_store = torch.zeros_like(
            runner.input_batch.token_ids_cpu_tensor, device="cpu"
        ).to(runner.device)
        history = ngram_context_from_store(
            self.token_store,
            torch.arange(num_reqs, device=runner.device, dtype=torch.int64),
            context,
            self.config.eos_token_id,
            self.config.ngram_size,
        )
        kwargs = dict(
            query_start_loc=starts,
            ngram_context=history,
            state_slots=slots,
            context_lens=context,
            valid_tokens=torch.ones(num_tokens, dtype=torch.bool, device=runner.device),
            main_block_table=table,
            compressed_block_table=table,
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
        )

    def _idle_run(self, num_tokens, num_reqs):
        """Participate in remote work without touching local live owners."""
        device = self.runner.device
        starts = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        context = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        slots = torch.full((num_reqs,), -1, dtype=torch.int32, device=device)
        positions = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        table = torch.zeros(
            (num_reqs, math.ceil(self.runner.max_model_len / self.block_size)),
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
            state_slots=slots,
            context_lens=context,
            valid_tokens=torch.zeros(num_tokens, dtype=torch.bool, device=device),
            main_block_table=table,
            compressed_block_table=table,
        )
        return self._dummy_forward(
            torch.zeros_like(positions),
            positions,
            kwargs,
            metadata,
            num_tokens,
            None,
            dp_lockstep=True,
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
    ):
        runner = self.runner
        assert self.cache is not None
        per_layer = (
            cast("Qwen4ExpPooledCache", self.cache).dummy_metadata(
                metadata, active_rows
            )
            if self.pooled
            else {name: metadata for name in self.cache.specs}
        )
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
                input_ids=ids, positions=positions, model_kwargs=kwargs
            )
            if dp_lockstep and runner.parallel_config.tensor_parallel_size > 1:
                # Match the generic runner's idle-DP protocol. The LM head
                # has a TP collective even when this DP has no live tokens;
                # omitting it leaves busy DP engines waiting on their logits.
                indices = torch.zeros(
                    kwargs["state_slots"].shape[0], dtype=torch.int32, device=out.device
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
