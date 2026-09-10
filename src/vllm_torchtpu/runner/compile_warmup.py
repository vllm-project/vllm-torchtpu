# SPDX-License-Identifier: Apache-2.0
import contextlib
import time

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.input_preparation import \
    _get_sequence_layout_planner_for_runner
from vllm_torchtpu.runner.padding import _get_padded_token_len
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


class CompileWarmupMixin:
    """Ahead-of-time XLA graph precompilation."""

    @contextlib.contextmanager
    def _precompile_timed(self, name: str):
        """Common boilerplate for each AOT bucket precompile pass."""
        logger.info("Compiling %s with different input shapes.", name)
        start = time.perf_counter()
        yield
        logger.info("Compilation finished in %.2f [secs].",
                    time.perf_counter() - start)
        self._update_num_xla_graphs(name)

    def _dummy_logits(self, num_reqs: int) -> torch.Tensor:
        return torch.zeros((num_reqs, self.vocab_size),
                           device=self.device,
                           dtype=self._hidden_states_dtype)

    def _precompile_compute_selected_logits(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_selected_logits"):
            for num_tokens in self.num_tokens_paddings:
                dummy_hidden = torch.zeros((num_tokens, hsize),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                for num_reqs in self.num_reqs_paddings:
                    indices = torch.zeros(num_reqs,
                                          dtype=torch.int32,
                                          device=self.device)
                    out = self.compute_selected_logits(dummy_hidden, indices)
                    synchronize_tensors(out)
                    logger.info("  -- num_tokens: %d, num_seqs: %d",
                                num_tokens, num_reqs)
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_compute_logits_from_hidden_states(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_logits_from_hidden_states"):
            for num_reqs in self.num_reqs_paddings:
                dummy_hidden = torch.zeros((num_reqs, hsize),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                out = self.compute_logits_from_hidden_states(dummy_hidden)
                synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_structured_decoding(self) -> None:
        with self._precompile_timed("structured_decoding"):
            arange = self.structured_decoding_manager.structured_decode_arange
            for num_reqs in self.num_reqs_paddings:
                out = self.structured_decoding_manager.structured_decode(
                    self.structured_decoding_manager.
                    require_structured_out_cpu[:num_reqs].to(self.device),
                    self.structured_decoding_manager.
                    grammar_bitmask_cpu[:num_reqs].to(self.device),
                    self._dummy_logits(num_reqs),
                    arange,
                )
                synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)
            # Prefill-only PCP MTP K1 skips _precompile_rejection_sampler(), so
            # we skip it here as well.
            if self.speculative_config is None or self._pcp_mtp_k1_enabled:
                return
            # For spec decoding, target logits have padded_logits_length rows.
            max_target_rows = self.structured_decoding_manager.target_grammar_bitmask_cpu.shape[
                0]
            for num_tokens in self.num_tokens_paddings:
                if num_tokens > max_target_rows:
                    break
                out = self.structured_decoding_manager.structured_decode(
                    self.structured_decoding_manager.
                    require_structured_out_target_cpu[:num_tokens].to(
                        self.device),
                    self.structured_decoding_manager.
                    target_grammar_bitmask_cpu[:num_tokens].to(self.device),
                    self._dummy_logits(num_tokens),
                    arange,
                )
                synchronize_tensors(out)
                logger.info("  -- target_logits_length: %d", num_tokens)

    def _precompile_sample_from_logits(self) -> None:
        with self._precompile_timed("sample_from_logits"):
            for num_reqs in self.num_reqs_paddings:
                dummy_logits = self._dummy_logits(num_reqs)
                dummy_temperatures = torch.ones(
                    (num_reqs, 1),
                    dtype=self._hidden_states_dtype,
                    device=self.device)
                dummy_u = torch.rand_like(dummy_logits)
                dummy_top_k = torch.zeros((num_reqs, 1),
                                          dtype=torch.int32,
                                          device=self.device)
                dummy_top_p = torch.ones((num_reqs, 1),
                                         dtype=torch.float32,
                                         device=self.device)
                for all_greedy in [False, True]:
                    out = self.sample_from_logits_func(dummy_logits,
                                                       dummy_temperatures,
                                                       dummy_u,
                                                       dummy_top_k,
                                                       dummy_top_p,
                                                       all_greedy=all_greedy)
                    synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_gather_logprobs(self) -> None:
        with self._precompile_timed("gather_logprobs"):
            for num_reqs in self.num_reqs_paddings:
                out = self.gather_logprobs(
                    self._dummy_logits(num_reqs),
                    torch.zeros((num_reqs, 1),
                                dtype=torch.int64).to(self.device),
                )
                synchronize_tensors(out.logprobs)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_rejection_sampler(self) -> None:
        """Warm the spec-decode verify path so it doesn't recompile at runtime.

        The rejection sampler is ``@torch.compile(backend="tpu",
        dynamic=False)`` -> a fresh graph per input shape, and the
        ``logits[...]`` gathers around it are eager. None are covered by
        ``_precompile_sampling_subgraphs`` (non-spec sampling only) or
        ``drafter.precompile()`` (draft forward only), so they recompile every
        time the verify length (``padded_logits_length``) lands in a new
        num-tokens bucket as requests finish and the batch shrinks. Replay the
        verify inner body (``_sample_spec_verify_chunk``) at every
        ``(padded_logits_length, padded_num_reqs)`` bucket the runtime can
        produce so the real call is a cache hit. Shared sync + async.

        Both verify sub-branches are distinct ``dynamic=False`` graphs, so warm
        each: greedy (``do_sampling=False`` ->
        ``_greedy_rejection_sample_with_segment``, prelude
        ``spec_bonus_and_target_logits``) and non-greedy (``do_sampling=True``
        -> ``_random_rejection_sample_with_segment``, prelude
        ``spec_gather_bonus_and_target_logits``). Which one runs is decided per
        step by ``input_batch.all_greedy`` and ``grammar_output``, so both are
        reachable and must be pre-compiled here; the greedy + structured-output
        combination reuses graphs warmed by these two passes.
        """
        if not (self._is_async_drafter):
            return
        k = self.speculative_config.num_speculative_tokens
        # The verify chunk samples at most num_reqs*(K+1) positions; bound the
        # padded_logits_length sweep to the bucket that covers the full batch.
        # Use min here because num_reqs*(K+1) can exceed max_num_batched_tokens
        # (which is the largest bucket), and logits rows never exceed the largest
        # bucket.
        max_logits_len = _get_padded_token_len(
            self.num_tokens_paddings,
            min(self.max_num_reqs * (k + 1), self.num_tokens_paddings[-1]))
        with self._precompile_timed("rejection_sampler"):
            for num_tokens in self.num_tokens_paddings:
                if num_tokens > max_logits_len:
                    break
                # padded_logits_length-shaped inputs; dtypes mirror
                # get_spec_decode_metadata so the graph is byte-identical.
                dummy_logits = torch.zeros((num_tokens, self.vocab_size),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                draft_token_ids = torch.zeros(num_tokens,
                                              dtype=torch.int32,
                                              device=self.device)
                target_logits_indices = torch.zeros(num_tokens,
                                                    dtype=torch.int32,
                                                    device=self.device)
                segment_ids = torch.zeros(num_tokens,
                                          dtype=torch.int64,
                                          device=self.device)
                group_indices = torch.zeros(num_tokens,
                                            dtype=torch.int32,
                                            device=self.device)
                for num_reqs in self.num_reqs_paddings:
                    # padded_num_reqs-shaped inputs.
                    draft_lengths = torch.zeros(num_reqs,
                                                dtype=torch.int32,
                                                device=self.device)
                    bonus_logits_indices = torch.zeros(num_reqs,
                                                       dtype=torch.int32,
                                                       device=self.device)
                    warm_accept_u = torch.zeros(num_tokens,
                                                dtype=torch.float32,
                                                device=self.device)
                    # --- greedy verify path (do_sampling=False) ---
                    bonus_token_ids, target_logits_warm = (
                        self.spec_bonus_and_target_logits(
                            dummy_logits, bonus_logits_indices,
                            target_logits_indices))
                    out = self.rejection_sampler(
                        draft_token_ids=draft_token_ids,
                        num_draft_tokens=draft_lengths,
                        target_logits=target_logits_warm,
                        bonus_token_ids=bonus_token_ids,
                        segment_ids=segment_ids,
                        group_indices=group_indices,
                        max_draft_tokens=k,
                        accept_u=(warm_accept_u
                                  if self.rejection_sampler.synthetic_mode else
                                  None),
                    )
                    synchronize_tensors(out)
                    # --- non-greedy verify path (do_sampling=True) ---
                    _, target_logits_ng = (
                        self.spec_gather_bonus_and_target_logits(
                            dummy_logits, bonus_logits_indices,
                            target_logits_indices))
                    warm_temps = torch.zeros((num_tokens, 1),
                                             dtype=self._hidden_states_dtype,
                                             device=self.device)
                    warm_top_k = torch.zeros((num_tokens, 1),
                                             dtype=torch.int32,
                                             device=self.device)
                    warm_top_p = torch.ones((num_tokens, 1),
                                            dtype=torch.float32,
                                            device=self.device)
                    warm_recover_u = torch.zeros((num_tokens, self.vocab_size),
                                                 dtype=torch.float32,
                                                 device=self.device)
                    out = self.rejection_sampler(
                        draft_token_ids=draft_token_ids,
                        num_draft_tokens=draft_lengths,
                        target_logits=target_logits_ng,
                        bonus_token_ids=bonus_token_ids,
                        segment_ids=segment_ids,
                        group_indices=group_indices,
                        max_draft_tokens=k,
                        temperatures=warm_temps,
                        top_k=warm_top_k,
                        top_p=warm_top_p,
                        accept_u=warm_accept_u,
                        recover_u=warm_recover_u,
                        do_sampling=True,
                    )
                    synchronize_tensors(out)
                    logger.info("  -- padded_logits_length: %d, num_seqs: %d",
                                num_tokens, num_reqs)
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_sampling_subgraphs(self) -> None:
        """Compile sampling-path subgraphs so their bottom-HBM reservations
        are visible to vLLM's available-memory probe in profile_run."""
        self._precompile_compute_selected_logits()
        if (_get_sequence_layout_planner_for_runner(
                self).uses_selected_logits_hidden_states):
            self._precompile_compute_logits_from_hidden_states()
        self._precompile_structured_decoding()
        self._precompile_sample_from_logits()
        self._precompile_gather_logprobs()
