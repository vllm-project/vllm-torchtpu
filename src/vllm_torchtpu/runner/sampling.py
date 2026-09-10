# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

import torch
from vllm.v1.outputs import LogprobsTensors

from vllm_torchtpu.layers.vllm.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.runner.speculative_decoding_manager import \
    SpecDecodeMetadata

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
