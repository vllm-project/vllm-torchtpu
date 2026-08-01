# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import copy
import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_tpu._internal import sync
from vllm.compilation.backends import set_model_tag
from vllm.config import (VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader

from vllm_torchtpu.layers.common.attention_metadata import \
    AttentionMetadataBuilderContext
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.spec_decode.utils import (DraftChunkInputs,
                                             _force_draft_tp1,
                                             maybe_share_embeddings)

# Sentinel for rejected / padding slots in the rejection-sampler output (and the
# async substitution tensors). Matches RejectionSampler.PLACEHOLDER_TOKEN_ID and
# the async runner's INVALID_TOKEN_ID.
INVALID_TOKEN_ID = -1


def _maybe_pad_dim0(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Right-pad a 1-D or 2-D tensor along dim 0 to `target_len` with zeros.

    If `target_len <= t.shape[0]` the tensor is returned unchanged (no-op).
    All current callers only pad upward; a caller passing a smaller target_len
    gets back the original tensor, not a truncation — callers must not rely on
    truncation behaviour.
    """
    n = t.shape[0]
    if target_len <= n:
        return t
    if t.ndim == 1:
        return torch.nn.functional.pad(t, (0, target_len - n))
    if t.ndim == 2:
        return torch.nn.functional.pad(t, (0, 0, 0, target_len - n))
    raise ValueError(
        f"_maybe_pad_dim0: unsupported ndim {t.ndim}, expected 1 or 2")


def _maybe_pad_dim1(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad the 2nd dimension (dim=1) of a 2D tensor up to target_len."""
    n = t.shape[1]
    if target_len <= n:
        return t
    return torch.nn.functional.pad(t, (0, target_len - n))


if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class Eagle3Proposer:
    """Eagle3/MTP draft proposer for TPU.

    This class is responsible for loading the draft model and generating draft
    tokens based on the target model's outputs.

    Current limitations:
    - Greedy decoding only (temperature=0). The draft emits argmax tokens, not
      a probability distribution, so the rejection sampler cannot perform
      proper non-greedy rejection sampling. Non-greedy requests are rejected
      at add_request time in tpu_platform.py before reaching this proposer.
    - draft_tensor_parallel_size selects the draft's parallelism: 1 runs the
      eagle3/mtp head fully replicated on every TPU worker; target tp runs it
      sharded across the TP group.
    """

    def __init__(
        self,
        runner: "TPUModelRunner",
        vllm_config: VllmConfig,
    ):
        self.runner = runner
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        draft_tp = self.speculative_config.draft_tensor_parallel_size
        target_tp = self.vllm_config.parallel_config.tensor_parallel_size
        # Default to (draft_tp == target tp) when unset. vLLM's
        # _verify_and_get_draft_tp already resolves an unset value to target tp
        # for eagle3/mtp, but set it explicitly here so self-documented.
        if draft_tp is None:
            draft_tp = target_tp
            self.speculative_config.draft_tensor_parallel_size = draft_tp
        # Only two draft parallelisms are supported: fully REPLICATED (tp=1) or
        # SHARDED across the whole TP group (draft_tp == target_tp).
        if draft_tp not in (1, target_tp):
            raise ValueError(
                f"{self.speculative_config.method} draft_tensor_parallel_size={draft_tp} is unsupported "
                f"on TPU: it must be 1 (replicated draft) or {target_tp} "
                f"(== target tensor_parallel_size, sharded draft).")
        self._draft_replicated = (draft_tp == 1)
        logger.info(
            "%s draft parallelism: %s (draft_tp=%s).",
            self.speculative_config.method,
            "REPLICATED (tp=1)" if self._draft_replicated else "SHARDED",
            draft_tp)
        self.draft_model = None
        # Attention-layer names belonging to the draft model. Populated in
        # load_model by diffing the global attention-layer registry before
        # vs after the draft model loads.
        self._draft_attn_layer_names: set[str] | None = None
        # Populated per-step by the runner with the per-chunk draft inputs
        # (token ids, positions, attn ctx, aux hidden states) captured
        # during the target verify forward.
        self.draft_chunks: list[DraftChunkInputs] | None = None
        # Per-propose-call cache of loop-step draft attention metadata, keyed
        # by id(chunk). Between loop iterations of one step everything in the
        # metadata is constant except seq_lens advancing by +1, so iterations
        # >= 2 swap the seq_lens field on the cached objects instead of paying
        # the full upstream builder walk per iteration.
        self._draft_md_cache: dict[int, tuple[dict, torch.Tensor]] = {}

    def load_model(self, target_model) -> None:
        """Load the draft model and share embeddings/lm_head with target.

        Also registers the eagle3 aux hidden state layers on the target so
        its forward returns (hidden_states, aux_hidden_states) at runtime.
        """
        # Snapshot the target's attention layer names before the draft model
        # adds its own; diff after load gives us the draft layer names.
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        AttentionLayerBase).keys())
        self._load_draft_model()
        all_attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                      AttentionLayerBase)
        self._draft_attn_layer_names = (set(all_attn_layers.keys()) -
                                        target_attn_layer_names)

        maybe_share_embeddings(self.draft_model, target_model,
                               self._draft_replicated)
        self._maybe_share_lm_head(target_model)
        if self.speculative_config.method == "eagle3":
            # Lazy + guarded: only eagle3 needs this vLLM internal, so a wrong vLLM
            # checkout shouldn't break unrelated TPU runs at import time.
            try:
                from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import \
                    set_eagle3_aux_hidden_state_layers
            except ImportError as e:
                raise ImportError(
                    "Eagle3 speculative decoding requires a vLLM build with "
                    "vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils; the "
                    "installed vLLM lacks it — check the vLLM version/checkout."
                ) from e
            set_eagle3_aux_hidden_state_layers(target_model,
                                               self.speculative_config)

    def _load_draft_model(self) -> None:
        logger.info(f"Loading {self.speculative_config.method} draft model...")
        model_loader = get_model_loader(self.vllm_config.load_config)
        # Tag the draft compile with "eagle_head" so its torch.compile cache
        # lives in a separate prefix from the target's "backbone" prefix.
        draft_tp1_ctx = (_force_draft_tp1() if self._draft_replicated else
                         contextlib.nullcontext())
        draft_vllm_config = copy.copy(self.vllm_config)
        draft_model_config = copy.copy(
            self.speculative_config.draft_model_config)
        draft_model_config.runner_type = "draft"
        draft_compilation_config = copy.copy(
            self.vllm_config.compilation_config)
        draft_compilation_config.inductor_compile_config = copy.copy(
            self.vllm_config.compilation_config.inductor_compile_config)
        draft_compilation_config.inductor_compile_config[
            "_vllm_model_tag"] = "eagle_head"
        draft_vllm_config.compilation_config = draft_compilation_config
        with set_model_tag("eagle_head"), set_vllm_model_wrapper_context(
                mesh=self.runner.mesh), set_current_vllm_config(
                    draft_vllm_config), draft_tp1_ctx:
            self.draft_model = model_loader.load_model(
                vllm_config=draft_vllm_config,
                model_config=draft_model_config,
            )

    def _maybe_share_lm_head(self, target_model) -> None:
        """Override of LLMBaseProposer._maybe_share_lm_head.

        Upstream conditionally shares the target's lm_head weights; we don't
        share weights, we align the logits all_gather with the draft's
        parallelism:

        - REPLICATED (tp=1) draft: the head holds the full vocab on every
          worker, so each already has the complete logits.
          otherwise force (it reads the runtime TP group, size = target tp).
        - SHARDED (draft_tp == target tp) draft: the head is column-parallel
          over the vocab like the target, so each rank holds only vocab/tp
          logits and the native _gather_logits all_gather IS required — leave it
          untouched (early return below).
        """
        del target_model  # unused; kept to match upstream override signature
        if not self._draft_replicated:
            return
        lp = getattr(self.draft_model, "logits_processor", None)
        if lp is not None:
            # Fail loudly if upstream renames the attr: a silent no-op here
            # would leave the cross-rank all_gather enabled, producing wrong
            # draft logits under a TP target.
            assert hasattr(lp, "_gather_logits"), (
                "draft logits_processor has no _gather_logits to override; "
                "vLLM may have renamed it.")
            lp._gather_logits = lambda logits: logits

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_propose_token(self, hidden: torch.Tensor) -> torch.Tensor:
        # Draft lm-head + greedy argmax, wrapped in one torch.compile region
        # keyed on input shape only (mirrors the target's compute_selected_logits).
        # Run raw, compute_logits + argmax + the int32 cast are eager ops the
        # torch-tpu DEFER_AND_FUSE path fuses with per-step-varying neighbors in
        # the K-step propose loop -> a fresh fused program per context. Enclosing
        # them makes a fixed, bucketed program per [n, hidden] shape.
        d2t = getattr(self.draft_model, "draft_id_to_target_id", None)
        if d2t is None:
            # No draft->target mapping table. Upstream compute_logits then asserts
            # the head is already target-vocab-width, so plain argmax yields
            # target ids.
            return self.draft_model.compute_logits(hidden).argmax(dim=-1).to(
                torch.int32)
        draft_id = self.draft_model.logits_processor(self.draft_model.lm_head,
                                                     hidden).argmax(dim=-1)
        return (draft_id + d2t[draft_id]).to(torch.int32)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_gather_carries(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        last_hidden: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # First-pass per-request carries: gather the last-token hidden, position,
        # and pre-lm-head hidden in one compiled region. Run raw (advanced
        # indexing), these three gathers are eager ops the DEFER_AND_FUSE path
        # fuses with the step's seq_lens/positions arithmetic into a per-context
        # program. index_select is value-identical to hidden[indices].
        pos_carry = (torch.index_select(positions, 1, indices)
                     if positions.ndim == 2 else torch.index_select(
                         positions, 0, indices))
        return (torch.index_select(hidden, 0, indices), pos_carry,
                torch.index_select(last_hidden, 0, indices))

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_combine_hidden_states(self, aux0: torch.Tensor,
                                     aux1: torch.Tensor,
                                     aux2: torch.Tensor) -> torch.Tensor:
        # Eagle3 always exposes exactly 3 aux hidden states. Fold the torch.cat
        # and the combine Linear into ONE compiled region keyed on
        # [num_tokens, aux_w] so torch-tpu's eager DEFER_AND_FUSE can't (a) split
        # the cat into a standalone program, (b) emit a distinct cat+mm grouping
        # per live dispatch context, or (c) fuse the combine mm forward into the
        # downstream draft_input_ids seed scatter. Same pattern as
        # _draft_propose_token / _draft_gather_carries.
        return self.draft_model.combine_hidden_states(
            torch.cat((aux0, aux1, aux2), dim=-1))

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _draft_seed_input_ids(self, input_ids: torch.Tensor,
                              last_token_indices: torch.Tensor,
                              next_token_ids: torch.Tensor) -> torch.Tensor:
        # Functional (no in-place) eagle3 first-pass input prep so fullgraph
        # holds: (1) left-shift by one (mirrors set_inputs_first_pass), (2)
        # scatter the per-request seed token at last_token_indices. Run raw, the
        # clone+shift and the index assignment are eager ops the DEFER_AND_FUSE
        # path fuses with the combine mm / metadata per context. Wrapping in one
        # compiled region keyed on [len(input_ids), len(indices)] removes them
        # from the per-context re-fusion. torch.cat((ids[1:], ids[-1:])) is
        # value-identical to clone()+[:-1]=ids[1:] (last slot keeps its own
        # value), and index_put is the functional twin of ids[idx] = val.
        shifted = input_ids
        if input_ids.shape[0] > 1:
            shifted = torch.cat((input_ids[1:], input_ids[-1:]), dim=0)
        return shifted.index_put((last_token_indices, ),
                                 next_token_ids.to(input_ids.dtype))

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output,
        return_device: bool = False,
        next_tokens_per_chunk: list[torch.Tensor] | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> list[list[int]] | torch.Tensor:
        """Generate K draft tokens per request via the eagle3 draft model.

        Args:
            sampled_token_ids: post-rejection sampled tokens per request.
            discard_sampled_tokens_req_indices: request indices whose sampled
                tokens should be discarded (partial-prefill case).
            num_rejected_tokens_np: per-request count of draft tokens that
                were rejected this step.
            scheduler_output: vLLM SchedulerOutput; used for partial-prefill
                next-token lookup via scheduler_output.num_scheduled_tokens.
            return_device: when True, return the raw ``[num_reqs, K]`` int32
                device tensor instead of host nested lists. The async path uses
                this to pack drafts into the next-step substitution source
                without a D2H on the critical path; the sync path keeps the
                list return.
            next_tokens_per_chunk: spec verify step (sync and async) —
                per-chunk on-device rejection-sampler output, passed to each
                chunk's ``_prepare_draft_inputs`` so the draft seed is read
                from device instead of the host ``sampled_token_ids``.
            device_seed: prefill / non-spec step (sync and async) —
                ``[total_reqs]`` device tensor of the just-sampled tokens.

        Returns (sync) a list of length num_reqs, each inner list holding K
        draft tokens; or (return_device) the ``[num_reqs, K]`` device tensor.
        """
        runner = self.runner
        self._draft_md_cache.clear()
        num_reqs = runner.input_batch.num_reqs
        if num_reqs == 0:
            return []

        K = self.speculative_config.num_speculative_tokens

        chunks = self.draft_chunks
        if not chunks:
            raise RuntimeError(
                f"{self.__class__.__name__}.propose() called but no draft chunks were "
                "captured (aux_hidden_states was None for every target chunk in eagle3, or "
                "no hidden_states were captured). "
                "Ensure target verify step correctly recorded state.")
        # First pass, per chunk — mirrors the target's chunked verify forward.
        # _prepare_draft_inputs returns last_token_indices padded to the chunk's
        # loop bucket p, so the per-request gathers/carries below are
        # bucket-shaped (constant) and don't recompile as num_reqs shrinks. Each
        # chunk's drafts stay [p]-shaped through the loop; we slice to the real
        # num_reqs only once at the end.
        hidden_carry_per_chunk = []
        positions_carry_per_chunk = []
        rejected_per_chunk = []
        uses_aux_hidden_state = self._draft_uses_aux_hidden_state()
        draft_tokens_per_chunk = []
        is_async = next_tokens_per_chunk is not None
        for ci, chunk in enumerate(chunks):
            if uses_aux_hidden_state:
                with set_model_tag("eagle_head"):
                    target_hidden_states = self._draft_combine_hidden_states(
                        *chunk.aux_hidden_states)
            else:
                assert chunk.hidden_states is not None, (
                    "DraftChunkInputs.hidden_states is required when the "
                    "draft does not use aux hidden states, but no plain "
                    "hidden state was captured for this chunk.")
                if self.speculative_config.method == 'eagle3':
                    with set_model_tag("eagle_head"):
                        target_hidden_states = self.draft_model.combine_hidden_states(
                            chunk.hidden_states)
                else:
                    # MTP: does not use combine_hidden_states
                    target_hidden_states = chunk.hidden_states

            (input_ids, positions, last_token_indices,
             num_rejected) = self._prepare_draft_inputs(
                 chunk,
                 sampled_token_ids,
                 discard_sampled_tokens_req_indices,
                 num_rejected_tokens_np,
                 scheduler_output,
                 next_tokens_device=(None if not is_async else
                                     next_tokens_per_chunk[ci]),
                 device_seed=device_seed,
             )
            rejected_per_chunk.append(num_rejected)
            last_hidden, hidden = self._forward_draft(
                chunk=chunk,
                input_ids=input_ids,
                positions=positions,
                target_hidden_states=target_hidden_states,
                step_idx=0,
                seq_lens_delta=0,
                num_rejected_np=(None if is_async else num_rejected),
                num_tokens_padded=input_ids.shape[0],
            )
            # [p, h] / [p] carries + first-pass draft token, computed in compiled
            # regions (the three gathers, then lm-head+argmax) so torch-tpu's
            # eager DEFER_AND_FUSE path can't fuse them into per-context programs.
            hidden_carry, positions_carry, last_hidden_carry = (
                self._draft_gather_carries(hidden, positions, last_hidden,
                                           last_token_indices))
            hidden_carry_per_chunk.append(hidden_carry)
            positions_carry_per_chunk.append(positions_carry)
            with set_model_tag("eagle_head"):
                draft_tokens_per_chunk.append(
                    [self._draft_propose_token(last_hidden_carry)])

        if K > 1:
            # Loop steps: uniform decode shape, one query per request. The
            # carries are already padded to the chunk's bucket p, so the loop
            # draft forward hits a precompiled trace with no per-num_reqs
            # recompile.
            padded_nr_per_chunk = [
                self._loop_bucket(c.num_reqs) for c in chunks
            ]
            loop_hidden = [
                _maybe_pad_dim0(h, p)
                for h, p in zip(hidden_carry_per_chunk, padded_nr_per_chunk)
            ]
            loop_positions = [
                _maybe_pad_dim1(pos, p) if
                (runner.uses_mrope and pos.ndim == 2) else _maybe_pad_dim0(
                    pos, p) for pos, p in zip(positions_carry_per_chunk,
                                              padded_nr_per_chunk)
            ]
            # Per-chunk rejected count padded to kernel_num_reqs so the loop's
            # seq_lens subtraction is full-length (constant) — no per-num_reqs
            # recompile. Sync pads on the host (free, reused across the K-1 loop
            # steps); async pads the device tensor.
            rejected_dev_per_chunk = []
            for c, nr in zip(chunks, rejected_per_chunk):
                knr = (runner.num_reqs_max_model_len
                       if c.attn_ctx.use_max_model_len else
                       runner.num_reqs_most_model_len)
                if is_async:
                    rejected_dev_per_chunk.append(
                        _maybe_pad_dim0(nr, knr) if nr is not None else None)
                elif nr is not None and np.any(nr):
                    padded_nr = np.zeros(knr, dtype=np.int32)
                    padded_nr[:c.num_reqs] = nr.astype(np.int32, copy=False)
                    rejected_dev_per_chunk.append(
                        torch.from_numpy(padded_nr).to(runner.device))
                else:
                    rejected_dev_per_chunk.append(None)
            # Loop-step query_start_loc and request_distribution depend only on
            # num_reqs (not the step), so build them once per chunk instead of
            # rebuilding (with an H2D) on every one of the K-1 steps.
            loop_qsl_per_chunk = []
            loop_reqdist_per_chunk = []
            for c in chunks:
                knr = (runner.num_reqs_max_model_len
                       if c.attn_ctx.use_max_model_len else
                       runner.num_reqs_most_model_len)
                qsl_np = np.minimum(np.arange(knr + 1, dtype=np.int32),
                                    c.num_reqs)
                loop_qsl_per_chunk.append(
                    torch.from_numpy(qsl_np).to(runner.device))
                loop_reqdist_per_chunk.append(
                    torch.tensor([c.num_reqs] * 3,
                                 dtype=torch.int32,
                                 device=runner.device))
            for step in range(1, K):
                for ci, chunk in enumerate(chunks):
                    loop_positions[ci] = loop_positions[ci] + 1
                    # Prev step's [p] tokens feed directly — already bucketed.
                    loop_input_ids = draft_tokens_per_chunk[ci][-1]
                    last_hidden, hidden = self._forward_draft(
                        chunk=chunk,
                        input_ids=loop_input_ids,
                        positions=loop_positions[ci],
                        target_hidden_states=loop_hidden[ci],
                        step_idx=step,
                        seq_lens_delta=step,
                        num_rejected_np=(None if is_async else
                                         rejected_per_chunk[ci]),
                        num_tokens_padded=padded_nr_per_chunk[ci],
                        num_rejected_dev=rejected_dev_per_chunk[ci],
                        loop_query_start_loc=loop_qsl_per_chunk[ci],
                        loop_request_distribution=loop_reqdist_per_chunk[ci],
                    )
                    with set_model_tag("eagle_head"):
                        draft_tokens_per_chunk[ci].append(
                            self._draft_propose_token(last_hidden))
                    loop_hidden[ci] = hidden

        # EP-DP lockstep: pad this rank's draft-forward count up to the
        # step's coordinated chunk bound so ranks with fewer chunks still
        # run the same number of draft forwards as their peers.
        if runner._dp_lockstep_enabled():
            extra_chunks = runner._dp_step_num_chunks - len(chunks)
            if extra_chunks > 0:
                self.run_dp_dummy_draft(extra_chunks)

        # Assemble [total_num_reqs, K]. Each chunk's tokens are padded to p; slice
        # to the real num_reqs at the end. Sync slices on the host (the [p, K] D2H
        # is constant-shape, so no recompile); async keeps the device slice
        # (matches prior behaviour for the substitution source).
        per_chunk_stacked = [
            torch.stack(toks, dim=1) for toks in draft_tokens_per_chunk
        ]
        if return_device:
            return torch.cat(
                [s[:c.num_reqs] for s, c in zip(per_chunk_stacked, chunks)],
                dim=0)
        result = []
        for s, c in zip(per_chunk_stacked, chunks):
            result.extend(s.cpu().tolist()[:c.num_reqs])
        return result

    def _prepare_draft_inputs(
        self,
        chunk: DraftChunkInputs,
        sampled_token_ids: list[list[int]],
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output,
        next_tokens_device: torch.Tensor | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray
               | torch.Tensor]:
        """Build draft first-pass inputs for one chunk.

        All token/request indices are chunk-local; the batch-level inputs
        (sampled_token_ids, num_rejected_tokens_np, discard indices) are
        sliced/offset by chunk.start_index.

        Args:
            next_tokens_device: async path only — the chunk's on-device
                rejection-sampler output ``[>=num_reqs, K+1]`` (accepted prefix +
                bonus, padded with INVALID_TOKEN_ID). When given, each request's
                seed token is gathered from it on-device instead of from the
                host ``sampled_token_ids`` list (which the async path lacks).
            device_seed: prefill / non-spec async bootstrap only —
                ``[total_reqs]`` device tensor of the just-sampled tokens
                (sliced ``[start:start+num_reqs]`` per chunk).

        Returns:
            input_ids: [padded_chunk_tokens] shifted + bonus-patched
            positions: [padded_chunk_tokens] chunk positions
            last_token_indices: [p] padded to the chunk's loop bucket; the
                first num_reqs entries hold the end of each request's accepted
                prefix, the padded tail holds filler indices (sliced off in
                propose()).
            num_rejected_np: [num_reqs] clamped per-request rejection count
        """
        runner = self.runner
        num_reqs = chunk.num_reqs
        start = chunk.start_index

        query_start_loc_np = chunk.query_start_loc_np[:num_reqs + 1]
        orig_num_tokens_per_req = (query_start_loc_np[1:] -
                                   query_start_loc_np[:-1])

        # Per-request rejection count (chunk's slice of the batch array).
        if num_rejected_tokens_np is None:
            num_rejected_np = np.zeros(num_reqs, dtype=np.int32)
        else:
            num_rejected_np = num_rejected_tokens_np[start:start +
                                                     num_reqs].astype(
                                                         np.int32, copy=False)
            # Clamp to orig-1, not orig: a request always retains at least the
            # bonus token, so at most orig-1 drafts can be rejected. Clamping to
            # orig would let last_token_indices fall to qsl[i]-1 (=-1 for the
            # first request, wrapping to the last padded row).
            num_rejected_np = np.minimum(num_rejected_np,
                                         orig_num_tokens_per_req - 1)

        # Eagle3 input shift (mirrors vLLM eagle's set_inputs_first_pass), then
        # patch each request's last accepted slot with next_token below.
        # E.g. [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        # The shift bleeds the next request's first token into the previous
        # request's last slot — those slots are exactly last_token_indices,
        # overwritten immediately below.
        # The eagle3 input shift + seed scatter are folded into the compiled
        # _draft_seed_input_ids helper applied at the scatter sites below (a raw
        # clone/shift here would be a per-context eager program).

        # Last-token index per request = end of the accepted prefix (where
        # the bonus token actually lives), not the original last verify slot.
        last_token_indices_np = (
            query_start_loc_np[1:num_reqs + 1].astype(np.int64) - 1 -
            num_rejected_np.astype(np.int64))
        # The async (next_tokens_device) path recomputes last_token_indices
        # fully on-device below, only the sync / device_seed paths consume this H2D.
        if next_tokens_device is None:
            last_token_indices = torch.from_numpy(last_token_indices_np).to(
                runner.device)

        discard_set = set(discard_sampled_tokens_req_indices)
        req_ids = runner.input_batch.req_ids[start:start + num_reqs]
        if next_tokens_device is not None:
            # Async path: derive the seed token, the seed *position*
            # (last_token_indices) and the rejected count from the on-device
            # rejection-sampler output instead of the host num_rejected /
            # sampled_token_ids — avoiding a D2H on the critical path.
            nt = next_tokens_device[:num_reqs]
            num_valid = (nt != INVALID_TOKEN_ID).sum(dim=1).clamp(min=1)
            last_valid_col = num_valid - 1
            next_token_ids = nt.gather(
                1, last_valid_col.unsqueeze(1)).squeeze(1).to(torch.int32)
            # Anchor the seed position at qsl_end-1 (like the sync path) using
            # the REAL per-request draft count.
            # num_draft is read on-device from the chunk's draft_lengths snapshot
            # draft_lengths is None for a chunk with no spec metadata.
            if chunk.draft_lengths is not None:
                num_draft_dev = chunk.draft_lengths[:num_reqs].to(torch.int64)
            else:
                num_draft_np = np.array([
                    len(
                        scheduler_output.scheduled_spec_decode_tokens.get(
                            rid, ())) for rid in req_ids
                ],
                                        dtype=np.int64)
                num_draft_dev = torch.from_numpy(num_draft_np).to(
                    runner.device)
            accepted_drafts = torch.minimum((num_valid - 1).clamp(min=0),
                                            num_draft_dev)
            num_rejected_out = (num_draft_dev - accepted_drafts).to(
                torch.int32)
            # qsl_end already lives on-device in the chunk's attn context (the
            # same tensor _build_draft_attn_metadata reads at step_idx=0); slice
            # it instead of a fresh H2D of query_start_loc_np.
            qsl_end = chunk.attn_ctx.query_start_loc[1:num_reqs + 1].to(
                torch.int64)
            last_token_indices = qsl_end - 1 - num_rejected_out.to(torch.int64)
            # Partial-prefill (discard) requests have no sampled token; seed
            # them from the host request state (same value as the sync path).
            ov_i = [i for i in range(num_reqs) if start + i in discard_set]
            if ov_i:
                ov_v = []
                for i in ov_i:
                    req_state = runner.requests[req_ids[i]]
                    seq_len = (
                        req_state.num_computed_tokens +
                        scheduler_output.num_scheduled_tokens[req_ids[i]])
                    ov_v.append(req_state.get_token_id(seq_len))
                next_token_ids[torch.tensor(
                    ov_i, dtype=torch.long,
                    device=runner.device)] = (torch.tensor(
                        ov_v, dtype=next_token_ids.dtype,
                        device=runner.device))
        elif device_seed is not None:
            # On-device prefill bootstrap: seed from the device sampled tokens
            # (no D2H). num_rejected stays 0 (num_rejected_tokens_np is None ->
            # num_rejected_np is zeros) and last_token_indices = qsl_end-1 (set
            # above) is the last prompt position -- the correct prefill seed.
            num_rejected_out = num_rejected_np
            next_token_ids = device_seed[start:start + num_reqs].to(
                torch.int32)
            # Partial-prefill (discard) reqs: override seed from host req_state.
            ov_i = [i for i in range(num_reqs) if (start + i) in discard_set]
            if ov_i:
                next_token_ids = next_token_ids.clone()
                ov_v = [
                    runner.requests[req_ids[i]].get_token_id(
                        runner.requests[req_ids[i]].num_computed_tokens +
                        scheduler_output.num_scheduled_tokens[req_ids[i]])
                    for i in ov_i
                ]
                next_token_ids[torch.tensor(
                    ov_i, dtype=torch.long,
                    device=runner.device)] = torch.tensor(
                        ov_v, dtype=next_token_ids.dtype, device=runner.device)
        else:
            num_rejected_out = num_rejected_np
            next_token_ids_np = np.zeros(num_reqs, dtype=np.int32)
            for i in range(num_reqs):
                batch_i = start + i  # batch-level request index
                if batch_i in discard_set:
                    req_id = req_ids[i]
                    req_state = runner.requests[req_id]
                    seq_len = (req_state.num_computed_tokens +
                               scheduler_output.num_scheduled_tokens[req_id])
                    next_token_ids_np[i] = req_state.get_token_id(seq_len)
                else:
                    ids = sampled_token_ids[batch_i] if batch_i < len(
                        sampled_token_ids) else []
                    next_token_ids_np[i] = ids[-1] if ids else 0
            next_token_ids = torch.from_numpy(next_token_ids_np).to(
                runner.device)
        # Bucket the seed scatter + the returned gather index to the chunk's loop
        # bucket p so neither recompiles as num_reqs shrinks (the dominant draft
        # propose recompile). Pad by repeating the last real entry: the padded
        # rows re-write request (num_reqs-1)'s seed to its own slot (idempotent),
        # so draft_input_ids stays correct, and the padded gather tail re-reads a
        # real row that propose() slices off.
        p = self._loop_bucket(num_reqs)
        if next_tokens_device is None and device_seed is None:
            # Pure sync: build the padded index + values on the host (free H2D,
            # no recompile), then scatter at the constant [p] shape.
            lti_p_np = np.full(p, last_token_indices_np[-1], dtype=np.int64)
            lti_p_np[:num_reqs] = last_token_indices_np
            ntids_p_np = np.full(p,
                                 next_token_ids_np[-1],
                                 dtype=next_token_ids_np.dtype)
            ntids_p_np[:num_reqs] = next_token_ids_np
            last_token_indices = torch.from_numpy(lti_p_np).to(runner.device)
            next_token_ids = torch.from_numpy(ntids_p_np).to(runner.device)
            draft_input_ids = self._draft_seed_input_ids(
                chunk.input_ids, last_token_indices, next_token_ids)
            gather_indices = last_token_indices
        else:
            # Async / bootstrap: the seed scatter stays real [num_reqs] (its
            # recompile is async-only); pad just the returned gather index.
            draft_input_ids = self._draft_seed_input_ids(
                chunk.input_ids, last_token_indices, next_token_ids)
            if next_tokens_device is not None:
                gather_indices = _maybe_pad_dim0(last_token_indices, p)
            else:
                padded_lti_np = np.zeros(p, dtype=np.int64)
                padded_lti_np[:num_reqs] = last_token_indices_np
                gather_indices = torch.from_numpy(padded_lti_np).to(
                    runner.device)

        return (draft_input_ids, chunk.position_ids, gather_indices,
                num_rejected_out)

    def _build_draft_attn_metadata(
        self,
        chunk: DraftChunkInputs,
        step_idx: int,
        seq_lens_delta: int,
        num_rejected_np: np.ndarray | None,
        num_tokens_padded: int | None = None,
        num_rejected_dev: torch.Tensor | None = None,
        loop_query_start_loc: torch.Tensor | None = None,
        loop_request_distribution: torch.Tensor | None = None,
    ) -> dict:
        """Build draft attention metadata for one chunk.

        Uses the chunk's captured ctx (chunk-local seq_lens /
        query_start_loc / request_distribution and the chunk's start_index
        for block-table slicing. num_rejected_np is already the chunk's slice.
        """
        runner = self.runner
        num_reqs = chunk.num_reqs
        chunk_ctx = chunk.attn_ctx

        # Loop-step metadata reuse: between loop iterations of one step the
        # metadata is identical except seq_lens (+1 per iteration) — qsl and
        # request_distribution are hoisted, block tables can't change
        # mid-propose, shapes are fixed.
        cacheable = (step_idx >= 1 and not runner._has_mamba_state
                     and not runner._unified_kv_layout
                     and chunk_ctx.position_ids_override is None)
        if cacheable and step_idx >= 2:
            cached = self._draft_md_cache.get(id(chunk))
            if cached is not None:
                md_dict, prev_seq_lens = cached
                # base + (s-1) - rejected, advanced by 1 == base + s - rejected:
                # value-identical to the full rebuild below.
                new_seq_lens = prev_seq_lens + 1
                replaced: dict[int, object] = {}
                new_md = {}
                for lname, md in md_dict.items():
                    if id(md) not in replaced:
                        replaced[id(md)] = dataclasses.replace(
                            md, seq_lens=new_seq_lens)
                    new_md[lname] = replaced[id(md)]
                self._draft_md_cache[id(chunk)] = (new_md, new_seq_lens)
                return new_md

        use_max_model_len = chunk_ctx.use_max_model_len
        kernel_num_reqs = (runner.num_reqs_max_model_len if use_max_model_len
                           else runner.num_reqs_most_model_len)

        if step_idx == 0:
            seq_lens = chunk_ctx.seq_lens
            query_start_loc = chunk_ctx.query_start_loc
            request_distribution = chunk_ctx.request_distribution
            num_tokens = int(chunk.query_start_loc_np[num_reqs])
            # Placeholder: _build_attention_metadata requires max_query_len but
            # the TPU backend never reads it. Any value is correct here.
            max_query_len = 1
        else:
            num_tokens = num_reqs
            # query_start_loc / request_distribution depend only on num_reqs;
            # the loop hoists them across steps. Fall back to building them here
            # when not supplied (keeps this helper usable standalone).
            if loop_query_start_loc is not None:
                query_start_loc = loop_query_start_loc
            else:
                qsl_np = np.arange(kernel_num_reqs + 1, dtype=np.int32)
                qsl_np = np.minimum(qsl_np, num_reqs)
                query_start_loc = torch.from_numpy(qsl_np).to(runner.device)
            # Full-length ops on the padded [kernel_num_reqs] seq_lens: the tail
            # [num_reqs:] gets the same delta / a zero subtraction but is ignored
            # by the kernel (query_start_loc caps queries at num_reqs), so this
            # stays correct while avoiding a per-num_reqs slice-assign that
            # recompiles as the batch shrinks. (`+` returns a fresh tensor, so
            # chunk_ctx.seq_lens is not mutated — no .clone() needed.)
            seq_lens = chunk_ctx.seq_lens + seq_lens_delta
            # ALWAYS subtract a (possibly all-zero) num_rejected so this metadata
            # program is ONE shape — the variant WITH a num_rejected operand —
            # regardless of whether this step actually had rejections. The spec
            # warmup's synthetic request is all-accept (dummy logits ->
            # draft==target -> zero rejections), so it only ever exercised the
            # add-only branch; the sub-with-num_rejected variant then recompiled
            # cold on the first real serving rejection (the residual tt_jit_sub at
            # the first reject step). Subtracting zeros is value-identical to the
            # add-only path, and building num_rejected via from_numpy().to(device)
            # keeps it a runtime PARAMETER (matching the real-rejection program),
            # not a folded constant. num_rejected_dev (async) is already
            # pre-padded; the tail [num_reqs:] is zeros and ignored by the kernel
            # (query_start_loc caps queries at num_reqs).
            if num_rejected_dev is None:
                padded_rej = np.zeros(kernel_num_reqs, dtype=np.int32)
                if num_rejected_np is not None:
                    padded_rej[:num_reqs] = num_rejected_np
                num_rejected_dev = torch.from_numpy(padded_rej).to(
                    runner.device)
            seq_lens = seq_lens - num_rejected_dev
            # Pure decode distribution; matches precompile loop-step shape.
            if loop_request_distribution is not None:
                request_distribution = loop_request_distribution
            else:
                request_distribution = torch.tensor(
                    [num_reqs, num_reqs, num_reqs],
                    dtype=torch.int32,
                    device=runner.device,
                )
            max_query_len = 1

        effective_num_tokens = (num_tokens_padded
                                if num_tokens_padded else num_tokens)

        mamba_state_indices = (runner._build_mamba_state_indices(
            chunk.start_index, num_reqs, kernel_num_reqs)
                               if runner._has_mamba_state else None)
        saved_ctx = runner._attn_metadata_builder_ctx
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=chunk.start_index,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
            # None for real chunks; the EP-DP dummy-run pairing sets fixed
            # positions so the block-table build uses the zeroed-table branch
            # instead of slicing the real (idle-rank) block table.
            position_ids_override=chunk.attn_ctx.position_ids_override,
        )
        try:
            slot_mappings = runner.empty_slot_mappings
            per_layer_attn_metadata, _ = runner._build_attention_metadata(
                num_tokens=effective_num_tokens,
                num_reqs=kernel_num_reqs,
                max_query_len=max_query_len,
                num_tokens_padded=effective_num_tokens,
                num_reqs_padded=kernel_num_reqs,
                slot_mappings=slot_mappings,
            )
        finally:
            runner._attn_metadata_builder_ctx = saved_ctx

        assert self._draft_attn_layer_names is not None, (
            "draft attn layer names were not captured during load_model")
        result = {
            name: md
            for name, md in per_layer_attn_metadata.items()
            if name in self._draft_attn_layer_names
        }
        if cacheable:
            self._draft_md_cache[id(chunk)] = (result, seq_lens)
        return result

    @staticmethod
    def _unwrap_model_out(out) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise a draft-model forward result to (last_hidden, carry_hidden).

        The draft model may return a bare tensor or a (last_hidden, aux_hidden)
        2-tuple (optionally wrapped in a 1-element list/tuple). A bare tensor is
        used for both outputs.
        """
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
        if isinstance(out, (list, tuple)):
            if len(out) != 2:
                raise RuntimeError(
                    f"Draft model returned a {len(out)}-element tuple; expected "
                    "a tensor or a 2-element (last_hidden, aux_hidden) tuple.")
            last_hidden, hidden = out
        else:
            last_hidden = hidden = out
        return last_hidden, hidden

    def _forward_draft(
        self,
        chunk: DraftChunkInputs,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        step_idx: int,
        seq_lens_delta: int,
        num_rejected_np: np.ndarray | None,
        num_tokens_padded: int | None = None,
        num_rejected_dev: torch.Tensor | None = None,
        loop_query_start_loc: torch.Tensor | None = None,
        loop_request_distribution: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_metadata = self._build_draft_attn_metadata(
            chunk=chunk,
            step_idx=step_idx,
            seq_lens_delta=seq_lens_delta,
            num_rejected_np=num_rejected_np,
            num_tokens_padded=num_tokens_padded,
            num_rejected_dev=num_rejected_dev,
            loop_query_start_loc=loop_query_start_loc,
            loop_request_distribution=loop_request_distribution,
        )

        num_tokens = input_ids.shape[0]

        kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "hidden_states": target_hidden_states,
        }
        if self.speculative_config.method == "mtp":
            kwargs["spec_step_idx"] = step_idx

        draft_vllm_cfg = getattr(self.draft_model, "vllm_config",
                                 self.vllm_config)
        with set_model_tag("eagle_head"), set_forward_context(
                attn_metadata,
                draft_vllm_cfg,
                num_tokens=num_tokens,
                num_tokens_across_dp=self.runner._dp_num_tokens_across_dp(
                    num_tokens),
        ), set_vllm_model_wrapper_context(mesh=self.runner.mesh):
            out = self.draft_model(**kwargs)

        return self._unwrap_model_out(out)

    def precompile(self) -> None:
        """Precompile the draft-side @torch.compile wrapper subgraphs.

        Warms the wrappers across their bucket shapes so they don't recompile on
        the first real propose(): combine_hidden_states (token-count buckets),
        seed_input_ids ((token, index) buckets), and the lm-head/argmax
        compute_logits (num_reqs buckets). The draft forward and the per-step
        fused propose/verify programs are warmed by the real-path
        _warmup_spec_decode generation, not here (a zeros-precompile of those
        emits programs that never match the runtime eager fusion).
        """
        if self.runner.enforce_eager:
            return
        self._precompile_combine_hidden_states()
        self._precompile_draft_seed()
        self._precompile_compute_logits()

    def _draft_hidden_size(self) -> int:
        return self.draft_model.config.hidden_size

    def _draft_uses_aux_hidden_state(self) -> bool:
        """Whether this draft checkpoint feeds combine_hidden_states the
        concatenation of several target aux hidden-state layers.
        """
        if self.speculative_config.method == "mtp":
            return False
        return bool(
            getattr(self.draft_model.model, "use_aux_hidden_state", True))

    def _draft_combine_input_size(self) -> int:
        """Width of the per-token tensor that combine_hidden_states consumes.

        When the draft uses aux hidden states, this is the draft's own
        `fc_input_size`. Otherwise combine_hidden_states is an identity over the plain hidden_size-wide
        target hidden state, so the input width is just hidden_size.
        """
        if self._draft_uses_aux_hidden_state():
            return self.draft_model.model.fc_input_size
        return self._draft_hidden_size()

    def _dp_lockstep_sharded(self) -> bool:
        """True when the draft is TP-SHARDED under EP-DP lockstep.

        In that one topology the draft's internal collectives are part of the
        cross-rank collective program, so every rank must execute the same collective trace.
        The replicated (draft_tp=1) draft has no collectives and needs none of this.
        """
        return (not self._draft_replicated
                and self.runner._dp_lockstep_enabled())

    def _loop_bucket(self, num_reqs: int) -> int:
        """Loop-phase token bucket p (carries, lm-head gathers, loop forwards).
        """
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len
        runner = self.runner
        if self._dp_lockstep_sharded():
            num_reqs = runner.max_num_reqs
        return _get_padded_token_len(runner.num_tokens_paddings, num_reqs)

    def _precompile_combine_hidden_states(self) -> None:
        if self.speculative_config.method == "mtp":
            return  # MTP does not use combine_hidden_states

        runner = self.runner
        if not self._draft_uses_aux_hidden_state():
            # combine_hidden_states is an identity for non-aux drafts.
            return
        # Warm the compiled _draft_combine_hidden_states wrapper with 3 SEPARATE
        # aux dummies (not one pre-cat [N, 3*aux] dummy) so the compiled
        # cat+mm grouping's shape-key matches the real first-pass dispatch.
        aux_w = self._draft_combine_input_size() // 3
        with set_model_tag("eagle_head"), runner._precompile_timed(
                "drafter combine_hidden_states"):
            for num_tokens in runner.num_tokens_paddings:
                aux = [
                    torch.zeros((num_tokens, aux_w),
                                dtype=runner._hidden_states_dtype,
                                device=runner.device) for _ in range(3)
                ]
                out = self._draft_combine_hidden_states(*aux)
                sync.synchronize(out, wait=True)
                logger.info("  -- drafter combine num_tokens: %d", num_tokens)

    def _precompile_draft_seed(self) -> None:
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len
        runner = self.runner
        # input_ids length = first-pass token bucket; index length = p =
        # pad(num_reqs), bounded by the loop bucket pad(max_num_reqs). Mirror the
        # bucket selection in _precompile_compute_logits so every sync first-pass
        # (T, p) shape-key is a startup cache hit.
        max_loop_bucket = _get_padded_token_len(runner.num_tokens_paddings,
                                                runner.max_num_reqs)
        idx_sizes = sorted(
            set(runner.num_reqs_paddings)
            | {t
               for t in runner.num_tokens_paddings if t <= max_loop_bucket})
        with runner._precompile_timed("drafter seed_input_ids"):
            for T in runner.num_tokens_paddings:
                for p in idx_sizes:
                    if p > T:
                        continue
                    ids = torch.zeros(T,
                                      dtype=torch.int32,
                                      device=runner.device)
                    lti = torch.zeros(p,
                                      dtype=torch.int64,
                                      device=runner.device)
                    nti = torch.zeros(p,
                                      dtype=torch.int32,
                                      device=runner.device)
                    out = self._draft_seed_input_ids(ids, lti, nti)
                    sync.synchronize(out, wait=True)

    def _precompile_compute_logits(self) -> None:
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len
        runner = self.runner
        hidden_size = self._draft_hidden_size()
        # compute_logits is called on (a) num_reqs buckets in the first pass and
        # (b) the loop's padded carry = _get_padded_token_len(num_tokens_paddings,
        # c.num_reqs), with c.num_reqs <= max_num_reqs. So the loop can only reach
        # token buckets up to the one just past max_num_reqs — precompile those
        # plus the num_reqs buckets. (The full num_tokens_paddings also runs up to
        # max_num_batched_tokens, compiling large [n, vocab] matmuls the loop never
        # hits.) Filtering by <= max_loop_bucket keeps every intermediate bucket
        # the loop can land on, unlike sampling only at num_reqs_paddings points.
        max_loop_bucket = _get_padded_token_len(runner.num_tokens_paddings,
                                                runner.max_num_reqs)
        all_sizes = set(runner.num_reqs_paddings) | {
            t
            for t in runner.num_tokens_paddings if t <= max_loop_bucket
        }
        with set_model_tag("eagle_head"), runner._precompile_timed(
                "drafter compute_logits"):
            for n in all_sizes:
                dummy_hidden = torch.zeros(
                    (n, hidden_size),
                    dtype=runner._hidden_states_dtype,
                    device=runner.device,
                )
                out = self._draft_propose_token(dummy_hidden)
                sync.synchronize(out, wait=True)
                logger.info("  -- drafter compute_logits n: %d", n)

    def run_dp_dummy_draft(self, num_chunks: int) -> None:
        """Replay a busy rank's DRAFT collective trace on an idle/padding rank,
        for `num_chunks` padding chunks (EP-DP lockstep).

        SHARDED draft (draft_tp == target tp): the draft's collectives are matched
        across ranks by graph-structure channel IDs, so each padding chunk must
        replay the REAL propose trace, channel IDs attach to collectives only.

        REPLICATED draft (draft_tp=1): entirely local — it emits no
        collectives, so there is nothing for an idle rank to pair with and the
        forwards would be pure wasted device time.
        """
        if not self._dp_lockstep_sharded():
            return
        self._draft_md_cache.clear()
        runner = self.runner
        K = self.speculative_config.num_speculative_tokens
        if num_chunks * K == 0:
            return
        bucket = runner._dp_target_bucket
        num_reqs = runner.num_reqs_max_model_len
        input_ids = torch.zeros(bucket,
                                dtype=torch.int32,
                                device=runner.device)
        positions = torch.zeros(bucket,
                                dtype=torch.int32,
                                device=runner.device)
        target_hidden_states = torch.zeros((bucket, self._draft_hidden_size()),
                                           dtype=runner._hidden_states_dtype,
                                           device=runner.device)

        actual_num_reqs = min(bucket, num_reqs)
        qsl_np = np.arange(num_reqs + 1, dtype=np.int32)
        attn_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=0,
            use_max_model_len=True,
            seq_lens=torch.ones(num_reqs,
                                dtype=torch.int32,
                                device=runner.device),
            query_start_loc=torch.from_numpy(qsl_np).to(runner.device),
            request_distribution=torch.tensor(
                [actual_num_reqs, actual_num_reqs, actual_num_reqs],
                dtype=torch.int32,
                device=runner.device),
            # Fixed positions => the attn-metadata build takes the dummy-safe
            # zeroed-block-table branch (attention_metadata.py) instead of
            # slicing the real block table, which on an idle rank is shorter
            # than num_reqs and would fail to broadcast.
            position_ids_override=positions,
        )
        chunk = DraftChunkInputs(
            input_ids=input_ids,
            position_ids=positions,
            query_start_loc_np=qsl_np,
            attn_ctx=attn_ctx,
            start_index=0,
            num_reqs=num_reqs,
            aux_hidden_states=[],
        )
        # Replay the real propose trace per padding chunk.
        p = self._loop_bucket(num_reqs)
        draft_hidden = self._draft_hidden_size()
        dtype = runner._hidden_states_dtype
        # First-pass lm-head input mirrors the gathered carry [p, hidden].
        first_pass_carry = torch.zeros((p, draft_hidden),
                                       dtype=dtype,
                                       device=runner.device)
        loop_input_ids = torch.zeros(p,
                                     dtype=torch.int32,
                                     device=runner.device)
        loop_positions = torch.zeros(p,
                                     dtype=torch.int32,
                                     device=runner.device)
        loop_hidden = torch.zeros((p, draft_hidden),
                                  dtype=dtype,
                                  device=runner.device)
        # Loop-step metadata mirrors propose()'s per-chunk loop tensors.
        loop_qsl = torch.from_numpy(
            np.minimum(np.arange(num_reqs + 1, dtype=np.int32),
                       actual_num_reqs)).to(runner.device)
        loop_reqdist = torch.tensor([actual_num_reqs] * 3,
                                    dtype=torch.int32,
                                    device=runner.device)
        uses_aux = self._draft_uses_aux_hidden_state()
        if uses_aux:
            aux_width = self._draft_combine_input_size() // 3
            combine_aux = [
                torch.zeros((bucket, aux_width),
                            dtype=dtype,
                            device=runner.device) for _ in range(3)
            ]
        else:
            combine_plain = torch.zeros(
                (bucket, self._draft_combine_input_size()),
                dtype=dtype,
                device=runner.device)
        for _ in range(num_chunks):
            # Mirror the real first pass: combine -> forward @bucket -> lm head.
            with set_model_tag("eagle_head"):
                if uses_aux:
                    self._draft_combine_hidden_states(*combine_aux)
                else:
                    self.draft_model.combine_hidden_states(combine_plain)
            last_hidden, _ = self._forward_draft(
                chunk=chunk,
                input_ids=input_ids,
                positions=positions,
                target_hidden_states=target_hidden_states,
                step_idx=0,
                seq_lens_delta=0,
                num_rejected_np=None,
                num_tokens_padded=bucket,
            )
            with set_model_tag("eagle_head"):
                tok = self._draft_propose_token(first_pass_carry)
            # Mirror the K-1 loop steps: forward @p -> lm head @p.
            for step in range(1, K):
                last_hidden, _ = self._forward_draft(
                    chunk=chunk,
                    input_ids=loop_input_ids,
                    positions=loop_positions,
                    target_hidden_states=loop_hidden,
                    step_idx=step,
                    seq_lens_delta=step,
                    num_rejected_np=None,
                    num_tokens_padded=p,
                    loop_query_start_loc=loop_qsl,
                    loop_request_distribution=loop_reqdist,
                )
                with set_model_tag("eagle_head"):
                    tok = self._draft_propose_token(last_hidden)
            # Force the chunk to execute so its collectives fire in lockstep
            # with the peer ranks' real propose (nothing consumes the result).
            sync.synchronize(tok, wait=True)
