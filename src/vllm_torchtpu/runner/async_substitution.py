# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _substitute_placeholder_token(
        input_ids: torch.Tensor, token_in_tpu_cur_input_indices: torch.Tensor,
        token_in_tpu_pre_next_tokens_indices: torch.Tensor,
        next_tokens: torch.Tensor):
    """Substitute placeholder tokens from TPU for async scheduler.

    Padding scheme (set up by `_apply_async_token_substitution`):
      - `token_in_tpu_cur_input_indices`: real-slot indices followed by all
        other indices in ascending order; the tail entries re-target slots
        that should keep their current values.
      - `token_in_tpu_pre_next_tokens_indices`: real `next_tokens` indices
        followed by `-1` sentinels at the padded positions.
    The `-1` sentinel is the mask source; the program is shape-only so it
    does not recompile when the active count changes.

    Args:
        input_ids: device tensor [N] where N is the bucketed input size.
        token_in_tpu_cur_input_indices: int tensor [N], destination slots.
        token_in_tpu_pre_next_tokens_indices: int tensor [N], source slots
            in `next_tokens`; -1 marks a padding slot to leave unchanged.
        next_tokens: int tensor of tokens from the previous async step.
    Return:
        input_ids with real placeholders replaced; padding slots untouched.
    """
    assert input_ids.shape[0] == token_in_tpu_cur_input_indices.shape[
        0] == token_in_tpu_pre_next_tokens_indices.shape[0]
    mask = token_in_tpu_pre_next_tokens_indices > -1
    # clamp_min(0) gives a safe in-range gather index for the -1 sentinel
    # slots; their gathered values are discarded by `mask` in `where`.
    safe_idx = torch.clamp_min(token_in_tpu_pre_next_tokens_indices, 0)
    new_token_values = next_tokens[safe_idx].to(input_ids.dtype)
    original_values = input_ids[token_in_tpu_cur_input_indices]
    update_values = torch.where(mask, new_token_values, original_values)
    input_ids.scatter_(0, token_in_tpu_cur_input_indices, update_values)
    return input_ids


class AsyncSubstitutionMixin:
    """Placeholder token substitution for async scheduling."""

    def _prepare_async_token_substitution_indices(
        self, start_index: int, num_reqs: int,
        num_scheduled_tokens_per_req: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._pre_async_results is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        token_in_tpu_cur_input_indices_list = []
        token_in_tpu_pre_next_tokens_indices_list = []
        acc_cur_len = 0
        layout_plan = self._last_sequence_layout_plan

        # 1+K for spec — `[bonus, draft_1..K]`), so a request's source span
        # starts at `position * stride`.
        stride = 1
        if (self.speculative_config is not None
                and self._pre_async_results.spec_decode_num_rejected_tokens
                is not None):
            stride = 1 + self.speculative_config.num_speculative_tokens

        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[start_index + i]
            n_sched = int(num_scheduled_tokens_per_req[i])
            acc_cur_len += n_sched
            assert req_id is not None
            if req_id not in self._pre_async_results.req_id_to_index_copy:
                continue

            # Map this request's last `n_sched` input slots to its source span
            # `[position*stride .. +n_sched-1]`. Non-spec (stride 1, n_sched 1)
            # reduces to the original single-slot mapping.
            src_start = (self._pre_async_results.req_id_to_index_copy[req_id] *
                         stride)
            for j in range(n_sched):
                request_major_index = acc_cur_len - n_sched + j
                if layout_plan is None:
                    cur_input_index = request_major_index
                else:
                    cur_input_index = (
                        layout_plan.local_index_for_request_major_token(
                            request_major_index))
                    if cur_input_index is None:
                        continue
                token_in_tpu_cur_input_indices_list.append(cur_input_index)
                token_in_tpu_pre_next_tokens_indices_list.append(src_start + j)

        if len(token_in_tpu_cur_input_indices_list) > 0:
            return (np.array(token_in_tpu_cur_input_indices_list,
                             dtype=np.int32),
                    np.array(token_in_tpu_pre_next_tokens_indices_list,
                             dtype=np.int32))
        else:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    def _apply_async_token_substitution(
            self, input_ids: torch.Tensor,
            token_in_tpu_cur_input_indices: np.ndarray,
            token_in_tpu_pre_next_tokens_indices: np.ndarray) -> torch.Tensor:
        """Apply async token substitution if needed."""
        if len(token_in_tpu_cur_input_indices) == 0:
            return input_ids

        idx_pad_len = len(input_ids) - len(token_in_tpu_cur_input_indices)

        # Pad according to the instructions written inside
        # _substitute_placeholder_token: missing = the complement of
        # cur_input_indices over [0, len(input_ids)). O(n) boolean mask instead
        # of np.setdiff1d's O(n log n) sort; flatnonzero returns the same
        # ascending order setdiff1d did over the (sorted, unique) full range.
        _missing_mask = np.ones(len(input_ids), dtype=bool)
        _missing_mask[token_in_tpu_cur_input_indices] = False
        missing_values = np.flatnonzero(_missing_mask).astype(np.int32)
        padded_token_in_tpu_cur_input_indices = np.concatenate(
            (token_in_tpu_cur_input_indices, missing_values))

        padded_token_in_tpu_pre_next_tokens_indices = np.pad(
            token_in_tpu_pre_next_tokens_indices, (0, idx_pad_len),
            mode='constant',
            constant_values=-1)

        cur_input_indices = torch.from_numpy(
            padded_token_in_tpu_cur_input_indices).to(self.device,
                                                      non_blocking=True)
        pre_next_tokens_indices = torch.from_numpy(
            padded_token_in_tpu_pre_next_tokens_indices).to(self.device,
                                                            non_blocking=True)

        return _substitute_placeholder_token(
            input_ids, cur_input_indices, pre_next_tokens_indices,
            self._pre_async_results.next_tokens_tpu)

    def _flush_disjoint_async_results(self) -> None:
        if self._pre_async_results is None:
            return
        current_req_ids = set(
            self.input_batch.req_ids[:self.input_batch.num_reqs])
        previous_req_ids = self._pre_async_results.req_id_to_index_copy
        if any(req_id in current_req_ids for req_id in previous_req_ids):
            return
        self._modify_prev_results()
        self._pre_async_results = None

    def _install_spec_token_room_guard(self) -> None:
        """Wrap InputBatch.update_req_spec_token_ids to clamp its *write*.

        Upstream stages the scheduled drafts at num_tokens_no_spec[req_index].
        Under async scheduling that position is optimistic -- every in-flight
        step charges 1+K assuming full acceptance, and the over-count is only
        rolled back by _modify_prev_results, which runs later (sample_tokens)
        than this call (execute_model -> _update_states). Near max_model_len
        the inflated position leaves fewer slots than K, and the unclamped
        write runs off the end of token_ids_cpu, killing the engine.

        Only the write may be clamped; the scheduled draft *count* must stay
        exactly what the scheduler charged. num_scheduled_tokens is already
        fixed at 1+K, and get_spec_decode_metadata anchors each request's
        logits window at the END of its query
        (cu_num_scheduled_tokens - num_sampled_tokens). Trimming the count
        slides that window forward by the trimmed amount, so the rejection
        sampler verifies the LAST `room` drafts while the tokens actually
        staged are the first ones -- silent greedy divergence instead of a
        crash. Dropping only the tail of the write is safe because these
        slots are placeholders under async: the substitution rewrites every
        scheduled slot from the device [bonus, draft_1..K] source, and
        draft_token_ids is re-extracted from the post-substitution input_ids.
        """
        input_batch = self.input_batch
        if getattr(input_batch, "_tpu_spec_room_guard", False):
            return
        orig = input_batch.update_req_spec_token_ids

        def guarded(request, scheduled_spec_tokens):
            req_id = request.req_id
            ids = scheduled_spec_tokens.get(req_id)
            idx = input_batch.req_id_to_index.get(req_id)
            if not ids or idx is None:
                return orig(request, scheduled_spec_tokens)

            room = max(
                0, input_batch.token_ids_cpu.shape[1] -
                int(input_batch.num_tokens_no_spec[idx]))
            if len(ids) <= room:
                return orig(request, scheduled_spec_tokens)

            # Fires routinely for a request's final steps near max_model_len:
            # async placeholder accounting overshoots the scheduler-side bound
            # there. Stage only the drafts that fit, then restore the full
            # scheduled list so every downstream count is untouched.
            logger.debug(
                "Clamping the draft-token write for %s to %d of %d slots at "
                "the context limit.", req_id, room, len(ids))
            scheduled_spec_tokens[req_id] = ids[:room]
            try:
                orig(request, scheduled_spec_tokens)
            finally:
                scheduled_spec_tokens[req_id] = ids

            # orig() recorded the clamped length; restore the scheduled one.
            request.prev_num_draft_len = len(ids)
            cur_spec_token_ids = getattr(input_batch, "spec_token_ids", None)
            if cur_spec_token_ids is not None:
                cur_spec_token_ids[idx].clear()
                cur_spec_token_ids[idx].extend(ids)
            return None

        input_batch.update_req_spec_token_ids = guarded
        input_batch._tpu_spec_room_guard = True

    def _modify_prev_results(self):
        if self._pre_async_results is None:
            return

        pre_req_ids = self._pre_async_results.req_ids
        pre_request_seq_lens = self._pre_async_results.request_seq_lens
        pre_discard_sampled_tokens_req_indices = self._pre_async_results.discard_sampled_tokens_req_indices
        # Per-request draft count from the previous step (keyed by the
        # request_seq_lens index); sizes the optimistic-placeholder rollback
        # below. None/absent -> 0 drafts (non-spec or a non-spec mixed chunk).
        pre_num_draft_per_req = self._pre_async_results.num_draft_per_req

        pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()
        assert pre_next_tokens_cpu is not None

        pre_next_tokens_cpu = pre_next_tokens_cpu[:len(pre_req_ids)]
        max_gen_len = pre_next_tokens_cpu.shape[-1]

        if max_gen_len == 1:
            valid_sampled_token_ids = pre_next_tokens_cpu.tolist()
        else:
            valid_mask = pre_next_tokens_cpu != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in pre_next_tokens_cpu[valid_mask].split(gen_lens)
            ]

        for i in pre_discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        for pre_req_idx, req_state, seq_len, req_id in pre_request_seq_lens:
            sampled_ids = valid_sampled_token_ids[pre_req_idx]
            if not sampled_ids:
                continue

            # The previous step optimistically appended (1 + num_draft)
            # placeholder tokens to output_token_ids, assuming every draft was
            # accepted. Now that the true acceptance is known, drop the entire
            # optimistic guess and append the actual sampled tokens.
            n_placeholder = 1 + (pre_num_draft_per_req.get(pre_req_idx, 0)
                                 if pre_num_draft_per_req else 0)
            # At most (1 bonus + num_draft) tokens can commit this step, so the
            # optimistic placeholder count must cover the real sampled count.
            # num_sampled_tokens <= pre_num_placeholder_tokens guard.
            assert len(sampled_ids) <= n_placeholder, (
                f"req {req_id}: sampled {len(sampled_ids)} tokens > "
                f"{n_placeholder} optimistic placeholders -- output_token_ids "
                "rollback would drop the wrong elements")
            del req_state.output_token_ids[-n_placeholder:]
            req_state.output_token_ids.extend(sampled_ids)

            if req_id not in self.input_batch.req_id_to_index:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]

            # num_tokens_no_spec was advanced to (seq_len + n_placeholder)
            # optimistically; roll back the over-count (= num_rejected) so it
            # reflects the real committed length (start_idx + len(sampled_ids)).
            end_idx = self.input_batch.num_tokens_no_spec[req_idx]
            start_idx = end_idx - n_placeholder
            self.input_batch.num_tokens_no_spec[req_idx] = (start_idx +
                                                            len(sampled_ids))

            target_slice = slice(seq_len - len(sampled_ids) + 1, seq_len + 1)
            # The committed tokens must fit within token_ids_cpu
            # ([num_reqs, max_model_len]); writing past the end is a sizing bug.
            assert seq_len + 1 <= self.input_batch.token_ids_cpu.shape[1], (
                f"req {req_id}: write end {seq_len + 1} exceeds max_model_len "
                f"{self.input_batch.token_ids_cpu.shape[1]}")
            self.input_batch.token_ids_cpu[req_idx, target_slice] = sampled_ids

    def _update_placeholder(self,
                            discard_sampled_tokens_req_indices,
                            request_seq_lens,
                            next_token_indices: dict[int, int],
                            num_draft_per_req: dict[int, int] | None = None):
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_set = set(discard_sampled_tokens_req_indices)
        for req_idx, req_state, seq_len, req_id in request_seq_lens:
            if req_idx in discard_set:
                continue

            # Async spec: optimistically advance by 1 (bonus) + num_draft;
            # the over-count is corrected on-device next step by subtract_num_rejected_tokens.
            n_new = 1 + (num_draft_per_req.get(req_idx, 0)
                         if num_draft_per_req else 0)
            end_idx = seq_len + n_new
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_state.output_token_ids.extend([0] * n_new)

            next_token_index = next_token_indices[req_idx]
            placeholder_req_id_to_index[req_state.req_id] = next_token_index

        return placeholder_req_id_to_index

    def _sync_replicated_drafts_across_tp(
            self, drafts: torch.Tensor | None) -> torch.Tensor | None:
        """Pin a replicated drafter's device proposal to rank 0's on every rank.

        Broadcast emulated as mask + all_reduce (tpu_dist has no broadcast);
        tiny (num_reqs x K int32), fully on device. No-ops for a sharded
        drafter, TP=1, the sync host-list form, and empty steps.
        """
        if drafts is None or not isinstance(drafts, torch.Tensor):
            return drafts
        if (self.speculative_config is None
                or self.speculative_config.draft_tensor_parallel_size != 1):
            return drafts
        from vllm.distributed import (get_tensor_model_parallel_rank,
                                      get_tensor_model_parallel_world_size)
        if get_tensor_model_parallel_world_size() <= 1:
            return drafts
        if self._tp_rank0_mask is None:
            self._tp_rank0_mask = torch.tensor(
                1 if get_tensor_model_parallel_rank() == 0 else 0,
                dtype=drafts.dtype,
                device=drafts.device)
        return self._tpu_pin_drafts_to_rank0(drafts, self._tp_rank0_mask)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_pin_drafts_to_rank0(self, drafts: torch.Tensor,
                                 rank0_mask: torch.Tensor) -> torch.Tensor:
        # Must be compiled (eager collectives wedge the TPU cores), and the
        # rank must enter as runtime data (the 0/1 mask), not a trace-time
        # constant: XLA pairs collectives by channel id, so every rank must
        # compile the identical program.
        from vllm.distributed import tensor_model_parallel_all_reduce
        return tensor_model_parallel_all_reduce(drafts * rank0_mask)

    def _assemble_async_spec_substitution(self, drafts, next_tokens_per_chunk,
                                          state):
        """Async-spec substitution assembler.

        From the already-proposed on-device ``drafts`` (the unified eagle3
        propose call in ``sample_tokens``), builds the per-request
        ``[bonus, draft_1..K]`` substitution source and the stride-``1+K``
        ``next_token_indices``, and computes the per-request rejected count.
        Returns ``(next_tokens_tpu_chunks, next_token_indices,
        spec_decode_num_rejected_tokens, num_draft_per_req)``.
        """
        from vllm_torchtpu.runner.tpu_runner_async_output import (
            assemble_spec_next_tokens, compute_num_rejected)
        assert drafts is not None, (
            "async spec substitution requires the eagle3 drafts proposed "
            "earlier in sample_tokens")
        K = self.speculative_config.num_speculative_tokens
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        num_rejected_chunks: list[torch.Tensor] = []
        num_draft_per_req: dict[int, int] = {}
        req_offset = 0
        # Uniform `1+K` source span per request: req position `p` maps to source
        # `[p*(1+K) .. p*(1+K)+K]`, and `req_id_to_index_copy` stores `p` (the
        # index builder multiplies by the stride). Mixed-batch non-spec chunks
        # pad their single token to the same `1+K` width.
        for nt_chunk, num_reqs, md in zip(next_tokens_per_chunk,
                                          state.num_reqs_list,
                                          state.spec_decode_metadata_list):
            if md is not None:
                drafts_chunk = drafts[req_offset:req_offset + num_reqs]
                src = assemble_spec_next_tokens(nt_chunk, drafts_chunk,
                                                num_reqs)  # [num_reqs*(1+K)]
                num_rejected_chunks.append(
                    compute_num_rejected(nt_chunk,
                                         md.draft_lengths[:num_reqs]))
                for r in range(num_reqs):
                    num_draft_per_req[req_offset + r] = int(
                        md.draft_lengths_cpu[r])
            else:
                # Mixed-batch non-spec chunk: pad the single sampled token to the
                # uniform 1+K span (token at slot 0, rest INVALID).
                token = nt_chunk[:num_reqs, 0]
                src = torch.full((num_reqs, 1 + K),
                                 INVALID_TOKEN_ID,
                                 dtype=token.dtype,
                                 device=token.device)
                src[:, 0] = token
                src = src.reshape(-1)  # [num_reqs*(1+K)]
                num_rejected_chunks.append(
                    torch.zeros(num_reqs,
                                dtype=torch.int32,
                                device=token.device))
            next_tokens_tpu_chunks.append(src)
            for r in range(num_reqs):
                next_token_indices[req_offset + r] = req_offset + r
            req_offset += num_reqs
        spec_num_rejected = (torch.cat(num_rejected_chunks)
                             if num_rejected_chunks else None)
        return (next_tokens_tpu_chunks, next_token_indices, spec_num_rejected,
                num_draft_per_req)

    def _assemble_async_prefill_bootstrap(self, drafts,
                                          combined_selected_tokens,
                                          combined_real_lens):
        """Async eagle3 bootstrap assembler for the prefill / pure-non-spec step.

        From the already-proposed prompt-context ``drafts`` (the unified eagle3
        propose call in ``sample_tokens``, device_seed path), parks a
        stride-(1+K) `[bonus, draft_1..K]` source so the NEXT step substitutes
        REAL prompt-context drafts. Returns ``(next_tokens_tpu_chunks,
        next_token_indices, spec_num_rejected=zeros, num_draft_per_req)``.
        """
        assert drafts is not None, (
            "async prefill bootstrap requires the eagle3 drafts proposed "
            "earlier in sample_tokens")
        K = self.speculative_config.num_speculative_tokens
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        num_rejected_chunks: list[torch.Tensor] = []
        num_draft_per_req: dict[int, int] = {}
        req_offset = 0
        for sel, n in zip(combined_selected_tokens, combined_real_lens):
            bonus = sel.view(-1)[:n].unsqueeze(1)  # [n, 1]
            drafts_chunk = drafts[req_offset:req_offset + n]  # [n, K]
            src = torch.cat([bonus.to(drafts_chunk.dtype), drafts_chunk],
                            dim=1)  # [n, 1+K] = [bonus, draft_1..K]
            next_tokens_tpu_chunks.append(src.reshape(-1))
            num_rejected_chunks.append(
                torch.zeros(n, dtype=torch.int32, device=src.device))
            for r in range(n):
                next_token_indices[req_offset + r] = req_offset + r
                num_draft_per_req[req_offset + r] = K
            req_offset += n
        spec_num_rejected = (torch.cat(num_rejected_chunks)
                             if num_rejected_chunks else None)
        return (next_tokens_tpu_chunks, next_token_indices, spec_num_rejected,
                num_draft_per_req)

    def _precompile_substitute_placeholder_token(self) -> None:
        if not self.scheduler_config.async_scheduling:
            return
        # next_tokens source lengths the runtime produces: stride-1 num_reqs
        # buckets (non-spec / pure-async), plus the bounded stride-(1+K) span
        # async eagle3 parks (max_num_reqs*(1+K); see sample_tokens) so the spec
        # substitute doesn't recompile per real num_reqs.
        next_lens = list(self.num_reqs_paddings)
        if (self._is_async_drafter):
            next_lens.append(
                self.max_num_reqs *
                (1 + self.speculative_config.num_speculative_tokens))
        with self._precompile_timed("substitute_placeholder_token"):
            for num_tokens in self.num_tokens_paddings:
                input_ids = torch.zeros(num_tokens,
                                        dtype=torch.int32,
                                        device=self.device)
                cur_input_indices = torch.zeros(num_tokens,
                                                dtype=torch.int32,
                                                device=self.device)
                # -1 sentinel marks every slot as padding so the precompiled
                # program is byte-identical to the runtime JIT call.
                pre_next_tokens_indices = torch.full((num_tokens, ),
                                                     -1,
                                                     dtype=torch.int32,
                                                     device=self.device)
                for nlen in next_lens:
                    next_tokens = torch.zeros(nlen,
                                              dtype=torch.int64,
                                              device=self.device)
                    out = _substitute_placeholder_token(
                        input_ids, cur_input_indices, pre_next_tokens_indices,
                        next_tokens)
                    synchronize_tensors(out)
                    logger.info("  -- num_tokens: %d, next_len: %d",
                                num_tokens, nlen)
