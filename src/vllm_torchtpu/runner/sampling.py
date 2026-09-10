# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from vllm.v1.outputs import LogprobsLists, LogprobsTensors, ModelRunnerOutput

from vllm_torchtpu.layers.vllm.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.runner.kv_connector_glue import _build_kv_connector_output
from vllm_torchtpu.runner.speculative_decoding_manager import \
    SpecDecodeMetadata
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID, AsyncPreResults, AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

SAMPLING_EPS = 1e-5


class SamplingMixin:
    """Logits to sampled tokens and logprobs."""

    def _get_sampling_generator(self) -> torch.Generator:
        """Return the dedicated non-greedy sampling generator, seeded once.

        Seeded lazily on first use from ``model_config.seed`` -- the same value
        on every worker -- then advanced naturally by the draws (never reseeded
        per step). A replica's TP ranks run in lockstep (same ``SchedulerOutput``
        -> same bucketed draw shapes each step), so with the same seed they stay
        at the same offset in the stream and sample identical tokens -- the
        property spec decode needs to keep the TP ranks from diverging. (DP
        replicas share the same base sequence but draw at their own pace on
        independent request streams, so they decorrelate, which is fine.)
        """
        if self._sampling_generator is None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(self.model_config.seed or 0))
            self._sampling_generator = gen
        return self._sampling_generator

    def _greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """argmax logits, only used when `all_greedy` is True.
        """
        dummy = torch.empty((1, 1), dtype=logits.dtype, device=logits.device)
        return self.sample_from_logits_func(logits,
                                            dummy,
                                            dummy,
                                            torch.empty((1, 1),
                                                        dtype=torch.int32,
                                                        device=logits.device),
                                            torch.empty((1, 1),
                                                        dtype=torch.float32,
                                                        device=logits.device),
                                            all_greedy=True)

    def _sample_spec_verify_chunk(
        self,
        logits: torch.Tensor,
        md: SpecDecodeMetadata,
        grammar_output: "GrammarOutput | None",
        scheduler_output: "SchedulerOutput",
        cur_start_idx: int,
        cur_end_idx: int,
        all_greedy: bool,
        sampling_generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Bonus-token sampling + draft rejection for one spec-verify chunk.

        Returns the rejection sampler's [num_reqs, 1 + num_spec_tokens]
        selected tokens for the chunk.
        """
        if all_greedy and grammar_output is None:
            bonus_token_ids, target_logits = (
                self.spec_bonus_and_target_logits(logits,
                                                  md.bonus_logits_indices,
                                                  md.target_logits_indices))
        else:
            bonus_logits, target_logits = (
                self.spec_gather_bonus_and_target_logits(
                    logits, md.bonus_logits_indices, md.target_logits_indices))
            if grammar_output is not None:
                target_logits, bonus_logits = (
                    self.structured_decoding_manager.mask_spec_logits(
                        target_logits, bonus_logits, grammar_output,
                        scheduler_output.scheduled_spec_decode_tokens,
                        md.draft_lengths_cpu, cur_start_idx, cur_end_idx))
            if all_greedy:
                bonus_token_ids = self._greedy_sample(bonus_logits).view(-1)
            else:
                req_temperatures, req_top_k, req_top_p = (
                    self._build_padded_sampling_params(cur_start_idx,
                                                       cur_end_idx,
                                                       bonus_logits))
                bonus_u = torch.rand_like(bonus_logits,
                                          generator=sampling_generator)
                bonus_token_ids = self.sample_from_logits_func(
                    bonus_logits,
                    req_temperatures,
                    bonus_u,
                    req_top_k,
                    req_top_p,
                    all_greedy=False).view(-1)
        accept_u = None
        if self.rejection_sampler.synthetic_mode or not all_greedy:
            accept_u = torch.rand(md.draft_token_ids.shape,
                                  dtype=torch.float32,
                                  device=target_logits.device,
                                  generator=sampling_generator)
        if all_greedy:
            return self.rejection_sampler(
                draft_token_ids=md.draft_token_ids,
                num_draft_tokens=md.draft_lengths,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                segment_ids=md.segment_ids,
                group_indices=md.group_indices,
                max_draft_tokens=self.speculative_config.
                num_speculative_tokens,
                accept_u=accept_u,
            )
        assert accept_u is not None
        recover_u = torch.rand_like(target_logits,
                                    dtype=torch.float32,
                                    generator=sampling_generator)
        return self.rejection_sampler(
            draft_token_ids=md.draft_token_ids,
            num_draft_tokens=md.draft_lengths,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            segment_ids=md.segment_ids,
            group_indices=md.group_indices,
            max_draft_tokens=self.speculative_config.num_speculative_tokens,
            temperatures=req_temperatures[md.segment_ids],
            top_k=req_top_k[md.segment_ids],
            top_p=req_top_p[md.segment_ids],
            accept_u=accept_u,
            recover_u=recover_u,
            do_sampling=True,
        )

    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            return None  # type: ignore[return-value]

        state = self.execute_model_state
        scheduler_output = state.scheduler_output
        self.execute_model_state = None

        # Used to keep rand draws consistent across TP ranks.
        sampling_generator = self._get_sampling_generator()

        # Hand the per-chunk draft inputs to the drafter.
        if self._is_async_drafter:
            self.drafter.draft_chunks = state.draft_chunks

        # Prepare inputs, the requests might be split into multiple
        # executions, combine the result of each execution.

        max_num_logprobs = self.input_batch.max_num_logprobs
        if max_num_logprobs == -1:
            raise NotImplementedError(
                "TPU runner does not support full logprobs (`logprobs=-1`) "
                "with the merged vLLM v1 sampler path yet.")
        needs_logprobs = max_num_logprobs is not None

        # Per-chunk bucketed (padded) tensors plus their real (unpadded)
        # request counts; kept at bucketed shapes so the async D2H copy hits
        # a precompiled program. Host-side trim happens in AsyncTPUCopyState.
        combined_selected_tokens: list[torch.Tensor] = []
        combined_selected_tokens_real_lens: list[int] = []
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        next_tokens_tpu_offset = 0
        combined_logprobs: list[Any] = []

        # True if the step has drafts to verify (a request carried drafts in
        # from the previous step). Not the same as self.speculative_config, the
        # static "spec configured" flag: on prefill/first-decode steps spec is
        # configured but no drafts exist yet, so is_spec_step is False there.
        is_spec_step = any(md is not None
                           for md in state.spec_decode_metadata_list)
        # Per-chunk mamba slot ids (hybrid + spec decode only; None entries
        # otherwise) for the read-offset scatter after sampling.
        mamba_idx_list = (state.mamba_state_indices_list
                          or [None] * len(state.logits_list))
        if self.is_pooling_model:
            req_ids = cast(
                list[str],
                self.input_batch.req_ids[:self.input_batch.num_reqs])
        elif is_spec_step:
            if needs_logprobs:
                raise NotImplementedError(
                    "Logprobs are not supported with speculative decoding on "
                    "TPU yet.")
            # Per-chunk device rejection outputs, kept for the async-spec
            # producer below.
            next_tokens_per_chunk: list[torch.Tensor] = []
            cur_start_idx = 0
            all_greedy = self.input_batch.all_greedy
            for logits, num_reqs, md, mamba_indices in zip(
                    state.logits_list, state.num_reqs_list,
                    state.spec_decode_metadata_list, mamba_idx_list):
                cur_end_idx = cur_start_idx + num_reqs
                if md is not None:
                    next_tokens = self._sample_spec_verify_chunk(
                        logits, md, grammar_output, scheduler_output,
                        cur_start_idx, cur_end_idx, all_greedy,
                        sampling_generator)
                    combined_selected_tokens.append(next_tokens)
                    combined_selected_tokens_real_lens.append(num_reqs)
                    next_tokens_per_chunk.append(next_tokens)
                    self._update_mamba_slot_read_offsets(
                        mamba_indices, next_tokens, num_reqs)
                else:
                    if grammar_output is not None:
                        _, logits = (
                            self.structured_decoding_manager.mask_spec_logits(
                                None, logits, grammar_output,
                                scheduler_output.scheduled_spec_decode_tokens,
                                None, cur_start_idx, cur_end_idx))
                    if all_greedy:
                        selected = self._greedy_sample(logits)
                    else:
                        temperatures_tpu, top_k_tpu, top_p_tpu = (
                            self._build_padded_sampling_params(
                                cur_start_idx, cur_end_idx, logits))
                        u = torch.rand_like(logits,
                                            generator=sampling_generator)
                        selected = self.sample_from_logits_func(
                            logits,
                            temperatures_tpu,
                            u,
                            top_k_tpu,
                            top_p_tpu,
                            all_greedy=False)
                    padded = torch.full(
                        (selected.shape[0],
                         self.speculative_config.num_speculative_tokens + 1),
                        INVALID_TOKEN_ID,
                        dtype=selected.dtype,
                        device=selected.device)
                    padded[:, 0] = selected.view(-1)
                    combined_selected_tokens.append(padded)
                    combined_selected_tokens_real_lens.append(num_reqs)
                    next_tokens_per_chunk.append(padded)
                    # Non-verify chunk in a spec step: reset the read offsets
                    # (the latest state checkpoint is at the group base).
                    self._update_mamba_slot_read_offsets(
                        mamba_indices, None, num_reqs)
                self._update_num_xla_graphs("spec_step")
                cur_start_idx = cur_end_idx
        else:
            cur_start_idx = 0
            req_ids = cast(
                list[str],
                self.input_batch.req_ids[:self.input_batch.num_reqs])
            all_greedy = self.input_batch.all_greedy
            for logits, num_reqs, mamba_indices in zip(state.logits_list,
                                                       state.num_reqs_list,
                                                       mamba_idx_list):
                cur_end_idx = cur_start_idx + num_reqs
                # Non-spec step (prefill / plain decode): reset the mamba
                # read offsets so the next step resumes from the group base.
                self._update_mamba_slot_read_offsets(mamba_indices, None,
                                                     num_reqs)
                if grammar_output is not None:
                    logits = self.structured_decoding_manager.mask_logits(
                        logits, grammar_output, cur_start_idx, cur_end_idx)
                if all_greedy:
                    selected_token_ids = self._greedy_sample(logits)
                else:
                    temperatures_tpu, top_k_tpu, top_p_tpu = (
                        self._build_padded_sampling_params(
                            cur_start_idx, cur_end_idx, logits))
                    u = torch.rand_like(logits, generator=sampling_generator)
                    selected_token_ids = self.sample_from_logits_func(
                        logits,
                        temperatures_tpu,
                        u,
                        top_k_tpu,
                        top_p_tpu,
                        all_greedy=all_greedy)
                # NOTE (NickLucche) Use the original logits (before any penalties or
                # temperature scaling) for the top-k logprobs. We can't enforce it
                # due to recompilations outside torch.compiled code, so just make
                # sure `sample_from_logits` does not modify the logits in-place.
                logprobs = (self.gather_logprobs(logits, selected_token_ids)
                            if needs_logprobs else None)

                # Keep the bucketed (padded) tensor; trim happens on the host
                # inside AsyncTPUCopyState to avoid per-`num_reqs` recompiles.
                combined_selected_tokens.append(selected_token_ids)
                combined_selected_tokens_real_lens.append(num_reqs)
                if self.scheduler_config.async_scheduling:
                    next_tokens_tpu_chunks.append(selected_token_ids.view(-1))
                    for req_idx in range(cur_start_idx, cur_end_idx):
                        next_token_indices[req_idx] = (next_tokens_tpu_offset +
                                                       req_idx - cur_start_idx)
                    next_tokens_tpu_offset += selected_token_ids.shape[0]
                if needs_logprobs:
                    sliced_logprobs = LogprobsTensors(
                        logprobs.logprob_token_ids[:num_reqs],
                        logprobs.logprobs[:num_reqs],
                        logprobs.selected_token_ranks[:num_reqs],
                        logprobs.cu_num_generated_tokens)
                    combined_logprobs.append(sliced_logprobs)

                self._update_num_xla_graphs("decoding_step")
                cur_start_idx = cur_end_idx

        # NOTE: current kv load and save get h2d/d2h copies involved.
        # Those copies are blocking. Once they become async., kv_save
        # should be called right after each single forward pass,
        # instead of the forwards of the entire input batch.
        logprobs = []
        if needs_logprobs and len(combined_logprobs):
            # TODO: concatenate the LogprobsTensors (torch) first, then
            # call .tolists() once to match the GPU path and avoid the
            # per-chunk numpy round-trip.
            combined_logprobs_lists = []
            for i in range(len(combined_logprobs)):
                logprobs = combined_logprobs[i].tolists()
                combined_logprobs_lists.append(logprobs)

            logprobs_lists = LogprobsLists(
                logprob_token_ids=np.concatenate(
                    [lp.logprob_token_ids for lp in combined_logprobs_lists]),
                logprobs=np.concatenate(
                    [lp.logprobs for lp in combined_logprobs_lists]),
                sampled_token_ranks=np.concatenate([
                    lp.sampled_token_ranks for lp in combined_logprobs_lists
                ]),
            )
            logprobs = logprobs_lists

        discard_sampled_tokens_req_indices = []
        num_reqs = self.input_batch.num_reqs
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        request_seq_lens = []
        for i, req_id in enumerate(req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            # Ignore the sampled token from the partial request.
            # Rewind the generator state as if the token was not sampled.
            if seq_len < req_state.num_tokens:
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)
            else:
                request_seq_lens.append((i, req_state, seq_len, req_id))

        next_tokens_tpu = None
        copy_state = None
        if combined_selected_tokens:
            copy_state = AsyncTPUCopyState.from_device_chunks(
                combined_selected_tokens, combined_selected_tokens_real_lens)

        if self.scheduler_config.async_scheduling:
            self._modify_prev_results()

        # Poll the KV connector only now: the wait above proves the previous
        # forward finished, so a store deferred behind it can be handed to
        # raiden in this step rather than the next.
        self.maybe_wait_for_kv_save()
        (finished_sending, finished_recving, kv_worker_meta, invalid_block_ids,
         invalid_block_group_index,
         kv_connector_stats) = self.get_finished_kv_transfers(scheduler_output)

        kv_connector_output = (None if (
            finished_sending is None and finished_recving is None
            and kv_worker_meta is None and not invalid_block_ids
            and kv_connector_stats is None) else _build_kv_connector_output(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                kv_connector_worker_meta=kv_worker_meta,
                invalid_block_ids=invalid_block_ids,
                invalid_block_group_index=invalid_block_group_index,
                kv_connector_stats=kv_connector_stats,
            ))

        # Unified eagle3 draft propose -- ONE call serving both sync and async
        # (mirrors the tpu-inference reference, where a single
        # propose_draft_token_ids call runs every step and an async bool only
        # picks the return form). Seeds on-device from this step's rejection
        # output (spec step) or the just-sampled tokens (non-spec / prefill
        # step), so neither mode waits on a host copy of the sampled tokens.
        # Async gets the raw [num_reqs, K] device tensor back (consumed by the
        # substitution assemblers below); sync caches the host list in the
        # manager for take_draft_token_ids(). ngram has no device path and
        # keeps proposing from the host-materialized output in the sync block
        # below.
        is_async = self.scheduler_config.async_scheduling
        eagle3_drafts = None
        if (self._is_async_drafter and combined_selected_tokens):
            if is_spec_step:
                eagle3_drafts = (
                    self.spec_decode_manager.propose_draft_token_ids(
                        sampled_token_ids=None,
                        discard_sampled_tokens_req_indices=
                        discard_sampled_tokens_req_indices,
                        num_rejected_tokens_np=None,
                        scheduler_output=scheduler_output,
                        return_device=is_async,
                        next_tokens_per_chunk=next_tokens_per_chunk))
            else:
                # Seed = the just-sampled tokens, kept on-device (no D2H):
                # propose takes the sync seed path but reads the seed from
                # `device_seed`.
                device_seed = torch.cat([
                    sel.view(-1)[:n]
                    for sel, n in zip(combined_selected_tokens,
                                      combined_selected_tokens_real_lens)
                ])
                eagle3_drafts = (
                    self.spec_decode_manager.propose_draft_token_ids(
                        sampled_token_ids=[],
                        discard_sampled_tokens_req_indices=
                        discard_sampled_tokens_req_indices,
                        num_rejected_tokens_np=None,
                        scheduler_output=scheduler_output,
                        return_device=is_async,
                        device_seed=device_seed))

        if self.scheduler_config.async_scheduling:
            # A replicated (draft_tp=1) drafter proposes independently per TP
            # rank and the proposals are not bit-identical; async verifies
            # each rank's OWN drafts (no driver round-trip like sync), so the
            # ranks drift apart and the output corrupts. Pin every rank to
            # rank 0's proposal; sharded drafters skip this.
            eagle3_drafts = self._sync_replicated_drafts_across_tp(
                eagle3_drafts)

            if (eagle3_drafts is not None
                    and scheduler_output.has_structured_output_requests):
                # The deferred grammar-bitmask will fetch these drafts via
                # take_draft_token_ids() to replace the async scheduler's -1
                # splaceholders. Stage after the TP pin above so the host
                # copy matches what gets substituted on device.
                self.spec_decode_manager.stage_draft_token_ids_for_host(
                    eagle3_drafts)

            # Build the async substitution source from the drafts proposed
            # above: the [bonus, draft_1..K] source + 1+K next_token_indices,
            # and the per-request rejected count to park.
            spec_num_rejected = None
            num_draft_per_req = None
            if is_spec_step:
                (next_tokens_tpu_chunks, next_token_indices, spec_num_rejected,
                 num_draft_per_req) = (self._assemble_async_spec_substitution(
                     eagle3_drafts, next_tokens_per_chunk, state))
            elif (self._is_async_drafter and next_tokens_tpu_chunks):
                # Prefill / pure-non-spec eagle3 bootstrap: the non-spec sampling
                # branch parked a stride-1 source (sampled token only). Rebuild
                # it as a stride-(1+K) [bonus, draft_1..K] source carrying real
                # prompt-context drafts so the NEXT step's verify gets real
                # drafts instead of placeholders.
                (next_tokens_tpu_chunks, next_token_indices, spec_num_rejected,
                 num_draft_per_req) = (self._assemble_async_prefill_bootstrap(
                     eagle3_drafts, combined_selected_tokens,
                     combined_selected_tokens_real_lens))
            req_id_to_index_copy = {}
            if not self.is_pooling_model:
                req_id_to_index_copy = self._update_placeholder(
                    discard_sampled_tokens_req_indices, request_seq_lens,
                    next_token_indices, num_draft_per_req)
            if next_tokens_tpu_chunks:
                if len(next_tokens_tpu_chunks) == 1:
                    next_tokens_tpu = next_tokens_tpu_chunks[0]
                else:
                    next_tokens_tpu = torch.cat(next_tokens_tpu_chunks, dim=0)
                # Bound the async-spec source length so the torch.compiled
                # _substitute_placeholder_token (next step) sees a fixed shape
                # instead of recompiling per real num_reqs. Append-pad to the
                # max stride-(1+K) span: the substitution only reads
                # [position*stride .. +n_sched-1] for real reqs (positions <
                # num_reqs), so the appended tail is never indexed -- real
                # positions are unchanged. Spec only (stride 1+K); pure-non-spec
                # async keeps its stride-1 source + existing precompile.
                if spec_num_rejected is not None:
                    bound = self.max_num_reqs * (
                        1 + self.speculative_config.num_speculative_tokens)
                    if next_tokens_tpu.shape[0] < bound:
                        next_tokens_tpu = torch.nn.functional.pad(
                            next_tokens_tpu,
                            (0, bound - next_tokens_tpu.shape[0]))
                self._pre_async_results = AsyncPreResults(
                    req_ids=req_ids,
                    next_tokens_tpu=next_tokens_tpu,
                    request_seq_lens=request_seq_lens,
                    discard_sampled_tokens_req_indices=
                    discard_sampled_tokens_req_indices,
                    req_id_to_index_copy=req_id_to_index_copy,
                    copy_state=copy_state,
                    spec_decode_num_rejected_tokens=spec_num_rejected,
                    num_draft_per_req=num_draft_per_req,
                )
            else:
                self._pre_async_results = None

        pooler_output = []
        if self.is_pooling_model:
            for out in state.pooler_output_list:
                # Move to CPU as expected by vLLM
                if isinstance(out, torch.Tensor):
                    pooler_output.extend(list(out.cpu().unbind(dim=0)))
                else:
                    pooler_output.extend(out)

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            # Snapshot of the req_id_to_index for the VLLM scheduler.
            req_id_to_index=dict(self.input_batch.req_id_to_index),
            sampled_token_ids=
            [],  # Filled in AsyncTPUModelRunnerOutput get_output
            logprobs=logprobs,
            prompt_logprobs_dict={req_id: None
                                  for req_id in req_ids},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
        )

        async_output = AsyncTPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            copy_state=copy_state,
            discard_sampled_tokens_req_indices=
            discard_sampled_tokens_req_indices)

        if not self.scheduler_config.async_scheduling:
            final_output = async_output.get_output()
            if not self.is_pooling_model:
                for i, req_state, seq_len, req_id in request_seq_lens:
                    if i in discard_sampled_tokens_req_indices:
                        continue
                    valid_tokens = final_output.sampled_token_ids[i]
                    if not valid_tokens:
                        continue
                    req_idx = self.input_batch.req_id_to_index[req_id]

                    # Update the persistent batch.
                    start_tok_idx = self.input_batch.num_tokens_no_spec[
                        req_idx]
                    end_tok_idx = start_tok_idx + len(valid_tokens)
                    self.input_batch.token_ids_cpu[
                        req_idx, start_tok_idx:end_tok_idx] = valid_tokens
                    self.input_batch.num_tokens_no_spec[req_idx] = end_tok_idx

                    req_state.output_token_ids.extend(valid_tokens)

            if (self.speculative_config
                    and self.speculative_config.method == "ngram"):
                # ngram drafts on the host from the committed token ids, so it
                # must run after materialization; eagle3/dflash were already proposed
                # from device state by the unified call above.
                self.spec_decode_manager.propose_draft_token_ids(
                    sampled_token_ids=final_output.sampled_token_ids,
                    discard_sampled_tokens_req_indices=
                    discard_sampled_tokens_req_indices,
                    num_rejected_tokens_np=None,
                    scheduler_output=scheduler_output)

            return final_output

        return async_output

    def _build_padded_sampling_params(
        self,
        cur_start_idx: int,
        cur_end_idx: int,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stage [padded_num_reqs] on CPU with neutral padding, then use one
        # fixed-shape H2D copy per tensor to keep decode shape-stable.
        padded_num_reqs = logits.shape[0]
        num_active_reqs = cur_end_idx - cur_start_idx
        temps_cpu = torch.ones(padded_num_reqs, dtype=logits.dtype)
        top_k_cpu = torch.zeros(padded_num_reqs, dtype=torch.int32)
        top_p_cpu = torch.ones(padded_num_reqs, dtype=torch.float32)

        temps_cpu[:num_active_reqs].copy_(
            self.input_batch.temperature_cpu_tensor[cur_start_idx:cur_end_idx])
        active_top_k = self.input_batch.top_k_cpu_tensor[
            cur_start_idx:cur_end_idx]
        top_k_cpu[:num_active_reqs].copy_(
            torch.where(active_top_k >= self.vocab_size,
                        torch.zeros_like(active_top_k), active_top_k))
        top_p_cpu[:num_active_reqs].copy_(
            self.input_batch.top_p_cpu_tensor[cur_start_idx:cur_end_idx])

        return (
            temps_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
            top_k_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
            top_p_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
        )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def compute_selected_logits(
            self, hidden_states: torch.Tensor,
            indices_do_sample: torch.Tensor) -> torch.Tensor:
        if self.is_pooling_model:
            return torch.empty((0, ), device=hidden_states.device)
        selected = torch.index_select(hidden_states, 0, indices_do_sample)
        return self.model.compute_logits(selected)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def compute_logits_from_hidden_states(
            self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def spec_bonus_and_target_logits(
        self, logits: torch.Tensor, bonus_logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Spec-decode verify prelude: gather the bonus-token logits + argmax, and
        # gather the target_logits rows for the rejection sampler, in ONE compiled
        # region keyed on [num_tokens, num_reqs] shapes. Run raw (the
        # logits[indices] gathers + argmax at the verify site), these are eager
        # ops torch-tpu's DEFER_AND_FUSE fuses with the live per-step dispatch
        # into per-context programs (the cold target-side recompiles). index_select
        # is value-identical to logits[indices]. Same pattern as the draft wraps.
        bonus_token_ids = torch.argmax(torch.index_select(
            logits, 0, bonus_logits_indices),
                                       dim=-1)
        target_logits = torch.index_select(logits, 0, target_logits_indices)
        return bonus_token_ids, target_logits

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def spec_gather_bonus_and_target_logits(
        self, logits: torch.Tensor, bonus_logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Non-greedy verify prelude: gather the bonus-token logits and the
        # target_logits rows in ONE compiled region keyed on [num_reqs,
        # num_tokens] shapes. Unlike the greedy spec_bonus_and_target_logits,
        # the bonus row is kept as FULL logits (not argmax'd) because the
        # non-greedy path samples the bonus token from them. Run raw, the
        # logits[indices] gathers are eager ops torch-tpu's DEFER_AND_FUSE fuses
        # with the live per-step dispatch into per-context programs (cold
        # recompiles).index_select is value-identical to logits[indices].
        bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
        target_logits = torch.index_select(logits, 0, target_logits_indices)
        return bonus_logits, target_logits

    def _apply_temperature(self, logits: torch.Tensor,
                           temperatures: torch.Tensor) -> torch.Tensor:
        safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
        return logits / safe_temperatures

    # TODO: Under SPMD mode, sample_from_logits has correctness issue.
    #       Re-enable the torch.compile once the issue is fixed in torchxla.
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def sample_from_logits(self,
                           logits: torch.Tensor,
                           temperatures: torch.Tensor,
                           u: torch.Tensor,
                           top_k: torch.Tensor,
                           top_p: torch.Tensor,
                           all_greedy: bool = False) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced
        separately from `forward` for lighter compilation overhead.
        """
        if all_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        is_greedy = temperatures <= SAMPLING_EPS
        scaled_logits = self._apply_temperature(logits, temperatures)
        masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p)
        u_clamped = torch.clamp(u,
                                min=torch.finfo(u.dtype).tiny,
                                max=1.0 - torch.finfo(u.dtype).eps)
        gumbel_noise = -torch.log(-torch.log(u_clamped))
        noisy_logits = masked_logits + gumbel_noise
        final_logits = torch.where(is_greedy, logits, noisy_logits)
        return torch.argmax(final_logits, dim=-1, keepdim=True)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def gather_logprobs(self, logits: torch.Tensor,
                        sampled_tokens: torch.Tensor) -> LogprobsTensors:
        """
        Gather the top_logprobs with corresponding tokens. Use a fixed number
        of logprobs as an alternative to having multiple pre-compiled graphs.
        Select the number of logprobs actually demanded by each request on CPU.
        """
        token_ids = sampled_tokens.to(torch.int64)
        token_logits = logits.gather(-1, token_ids)
        token_ranks = (logits >= token_logits).sum(dim=-1, dtype=torch.int32)
        log_normalizers = torch.logsumexp(logits, dim=-1, keepdim=True)
        token_logprobs = (token_logits - log_normalizers).to(torch.float32)

        max_logprobs = self.model_config.max_logprobs
        if max_logprobs > 0:
            topk_logits, topk_indices = torch.topk(logits,
                                                   max_logprobs,
                                                   dim=-1)
            topk_logprobs = (topk_logits - log_normalizers).to(torch.float32)
            logprob_token_ids = torch.cat((token_ids, topk_indices), dim=1)
            logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        else:
            logprob_token_ids = token_ids
            logprobs = token_logprobs

        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids.to(torch.int32),
            logprobs=logprobs,
            selected_token_ranks=token_ranks,
        )
