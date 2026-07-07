# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from dataclasses import dataclass
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


@dataclass
class DraftChunkInputs:
    # Token ids the target consumed (post async-token-substitution).
    # Device tensor, [padded_chunk_tokens].
    input_ids: torch.Tensor
    # Device tensor, [padded_chunk_tokens].
    position_ids: torch.Tensor
    # Host copy of the chunk-local cumsum of scheduled tokens. [num_reqs + 1]
    query_start_loc_np: np.ndarray
    # The chunk's attention-metadata builder context (chunk-local seq_lens /
    # query_start_loc / request_distribution device tensors, plus the
    # chunk's start_index for block-table slicing).
    attn_ctx: "AttentionMetadataBuilderContext"
    # First request of the chunk in batch order.
    start_index: int
    # Real (unpadded) request count in the chunk.
    num_reqs: int
    # Per-layer aux hidden states from the target forward; each is
    # [padded_chunk_tokens, aux_hidden]. Only consumed when the draft
    # checkpoint wants them.
    aux_hidden_states: list[torch.Tensor]
    # Plain target hidden state for the chunk,[padded_chunk_tokens, hidden_size].
    # Used directly -- bypassing the aux-hidden-state concatenation.
    hidden_states: torch.Tensor | None = None


@contextlib.contextmanager
def _force_draft_tp1():
    """Collapse the TP group to world_size=1 / rank=0 for the duration of draft
    construction + weight load, so the eagle3 draft loads fully replicated
    (tp=1) on every worker instead of TP-sharded.

    NOTE: mutates the singleton GroupCoordinator returned by get_tp_group() in
    place. This is safe only because model loading is single-threaded — no
    other code reads tp.world_size or tp.rank_in_group concurrently. Do not
    widen this context manager to cover parallel operations.

    Currently acceptable given the single-threaded load. The cleaner long-term
    fix is to construct/pass a dedicated tp=1 GroupCoordinator to the draft
    load path instead of mutating the shared singleton; deferred to future work.
    """
    from vllm.distributed.parallel_state import get_tp_group
    tp = get_tp_group()
    saved_ws, saved_rank = tp.world_size, tp.rank_in_group
    tp.world_size = 1
    tp.rank_in_group = 0
    try:
        yield
    finally:
        tp.world_size = saved_ws
        tp.rank_in_group = saved_rank


if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class Eagle3Proposer:
    """Eagle3 draft proposer for TPU.

    This class is responsible for loading the draft model and generating draft
    tokens based on the target model's outputs.

    Current limitations:
    - Greedy decoding only (temperature=0). The draft emits argmax tokens, not
      a probability distribution, so the rejection sampler cannot perform
      proper non-greedy rejection sampling. Non-greedy requests are rejected
      at add_request time in tpu_platform.py before reaching this proposer.
    - draft_tensor_parallel_size selects the draft's parallelism: 1 runs the
      eagle3 head fully replicated on every TPU worker; target tp runs it
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
        # for eagle3, but set it explicitly here so self-documented.
        if draft_tp is None:
            draft_tp = target_tp
            self.speculative_config.draft_tensor_parallel_size = draft_tp
        # Only two draft parallelisms are supported: fully REPLICATED (tp=1) or
        # SHARDED across the whole TP group (draft_tp == target_tp).
        if draft_tp not in (1, target_tp):
            raise ValueError(
                f"eagle3 draft_tensor_parallel_size={draft_tp} is unsupported "
                f"on TPU: it must be 1 (replicated draft) or {target_tp} "
                f"(== target tensor_parallel_size, sharded draft).")
        self._draft_replicated = (draft_tp == 1)
        logger.info(
            "eagle3 draft parallelism: %s (draft_tp=%s).",
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

        self._maybe_share_embeddings(target_model)
        self._maybe_share_lm_head(target_model)
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
        logger.info("Loading Eagle3 draft model...")
        model_loader = get_model_loader(self.vllm_config.load_config)
        # Tag the draft compile with "eagle_head" so its torch.compile cache
        # lives in a separate prefix from the target's "backbone" prefix.
        draft_tp1_ctx = (_force_draft_tp1() if self._draft_replicated else
                         contextlib.nullcontext())
        with set_model_tag("eagle_head"), set_vllm_model_wrapper_context(
                mesh=self.runner.mesh), set_current_vllm_config(
                    self.vllm_config), draft_tp1_ctx:
            self.draft_model = model_loader.load_model(
                vllm_config=self.vllm_config,
                model_config=self.speculative_config.draft_model_config,
            )

    def _maybe_share_embeddings(self, target_model) -> None:
        """Give the draft the target's input embedding (eagle3 checkpoints
        typically ship no embed_tokens of their own). The form depends on the
        draft's parallelism:

        - REPLICATED (tp=1) draft: needs the FULL vocab on every worker, so
          host-gather the target's sharded embed shards into a replicated
          nn.Embedding (_populate_draft_embed_from_target).
        - SHARDED (draft_tp == target tp) draft: its embed is sharded over the
          vocab like the target, so share the target's VocabParallelEmbedding
          module directly — same layout, no gather / no replicated copy.

        Only fires when the draft opts to share (has_own_embed_tokens False or
        absent); a draft with its own embed keeps it.
        """
        target_lm = target_model.get_language_model() if hasattr(
            target_model, "get_language_model") else target_model
        target_lm_model = getattr(target_lm, "model", None)
        target_embed = getattr(target_lm_model, "embed_tokens",
                               None) if target_lm_model is not None else None

        if hasattr(self.draft_model, "has_own_embed_tokens"):
            share_embed = not self.draft_model.has_own_embed_tokens
        else:
            logger.info(
                "EAGLE draft model does not declare "
                "`has_own_embed_tokens`; defaulting to share embed_tokens "
                "with the target.")
            share_embed = True

        if share_embed:
            if target_embed is None:
                raise RuntimeError(
                    "Eagle3 embedding sharing requires target_lm.model."
                    "embed_tokens, but the target model does not expose that "
                    "attribute. Set has_own_embed_tokens=True on the draft "
                    "model to skip sharing.")
            if self._draft_replicated:
                logger.info(
                    "Populating draft's own embed_tokens with a host-gathered, "
                    "per-worker replicated copy of the target embedding.")
                self._populate_draft_embed_from_target(target_embed)
            else:
                logger.info(
                    "Sharing the target's sharded embed_tokens with the "
                    "sharded draft.")
                self.draft_model.model.embed_tokens = target_embed

    def _populate_draft_embed_from_target(self, target_embed) -> None:
        """Replicated (tp=1) draft only: fill the draft's own full-vocab
        embed_tokens with the target's embedding, assembled on the host.

        Memory note: allocates a full [org_vocab, dim] fp32 tensor on every
        worker's CPU for the all_reduce (e.g. ~2 GB for Llama-3.1-8B:
        128256 x 4096 x 4B), plus the down-cast device copy. One-time load-time
        cost, freed once embed_tokens is replaced, but can spike host RAM on
        memory-constrained hosts.
        """
        import torch.distributed as dist
        from vllm.distributed.parallel_state import get_tp_group

        draft_embed = self.draft_model.model.embed_tokens
        org_vocab = target_embed.org_vocab_size
        dim = target_embed.embedding_dim
        assert draft_embed.org_vocab_size == org_vocab, (
            f"draft embed org_vocab {draft_embed.org_vocab_size} != target "
            f"{org_vocab}; cannot populate")

        si = target_embed.shard_indices
        num_added = si.added_vocab_end_index - si.added_vocab_start_index
        if num_added > 0:
            raise NotImplementedError(
                "Eagle3 on TPU does not support LoRA with vocabulary "
                "expansion (num_added_vocab_tokens > 0).")

        num_org = si.org_vocab_end_index - si.org_vocab_start_index
        org_vocab = target_embed.org_vocab_size
        full = torch.zeros(org_vocab, dim, dtype=torch.float32)
        my_rows = target_embed.weight.data[:num_org].to(torch.float32).cpu()
        full[si.org_vocab_start_index:si.org_vocab_end_index] = my_rows
        # Reconstruct the full vocab by summing the disjoint TP shards. Use the
        # TP group (the axis the embedding is sharded over), NOT the world group:
        # world would re-add each shard once per DP replica, scaling the draft
        # embedding by the DP factor (silent acceptance collapse, not a crash).
        # Runs outside _force_draft_tp1, so get_tp_group() is the real TP group.
        # NOTE: only DP=1 is supported/validated for eagle3 today; DP>1 is
        # deferred to future work — reducing over the TP group keeps this
        # reconstruction correct for when DP>1 support lands.
        dist.all_reduce(full, group=get_tp_group().cpu_group)

        draft_dtype = draft_embed.weight.dtype
        draft_device = draft_embed.weight.device
        full_dev = full.to(draft_dtype).to(draft_device)
        new_embed = torch.nn.Embedding(org_vocab, dim, _weight=full_dev)
        new_embed.weight.requires_grad_(False)
        del self.draft_model.model.embed_tokens
        self.draft_model.model.embed_tokens = new_embed
        if draft_device.type == "tpu":
            sync.synchronize(new_embed.weight, wait=True)
        logger.info(
            "Draft embed_tokens replaced with full replicated nn.Embedding: "
            "%d x %d per worker.", org_vocab, dim)

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

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output,
    ) -> list[list[int]]:
        """Generate K draft tokens per request via the eagle3 draft model.

        Args:
            sampled_token_ids: post-rejection sampled tokens per request.
            discard_sampled_tokens_req_indices: request indices whose sampled
                tokens should be discarded (partial-prefill case).
            num_rejected_tokens_np: per-request count of draft tokens that
                were rejected this step.
            scheduler_output: vLLM SchedulerOutput; used for partial-prefill
                next-token lookup via scheduler_output.num_scheduled_tokens.

        Returns a list of length num_reqs; each inner list holds K draft
        tokens.
        """
        runner = self.runner
        num_reqs = runner.input_batch.num_reqs
        if num_reqs == 0:
            return []

        K = self.speculative_config.num_speculative_tokens

        # Import here to avoid circular import at module load time.
        from vllm_torchtpu.runner.tpu_runner import (
            _get_padded_num_reqs_with_upper_limit, _get_padded_token_len)

        chunks = self.draft_chunks
        if not chunks:
            raise RuntimeError(
                "Eagle3Proposer.propose() called but no draft chunks were "
                "captured (aux_hidden_states was None for every target chunk). "
                "Ensure set_eagle3_aux_hidden_state_layers() ran during model "
                "load.")
        # First pass, per chunk — mirrors the target's chunked verify forward.
        sample_hidden_per_chunk = []
        hidden_carry_per_chunk = []
        positions_carry_per_chunk = []
        rejected_per_chunk = []
        uses_aux_hidden_state = self._draft_uses_aux_hidden_state()
        for chunk in chunks:
            if uses_aux_hidden_state:
                combine_input = torch.cat(chunk.aux_hidden_states, dim=-1)
            else:
                assert chunk.hidden_states is not None, (
                    "DraftChunkInputs.hidden_states is required when the "
                    "eagle3 draft does not use aux hidden states "
                    "(eagle_config.use_aux_hidden_state=False), but no plain "
                    "hidden state was captured for this chunk.")
                combine_input = chunk.hidden_states
            target_hidden_states = self.draft_model.combine_hidden_states(
                combine_input)
            (input_ids, positions, last_token_indices,
             num_rejected_np) = self._prepare_draft_inputs(
                 chunk,
                 sampled_token_ids,
                 discard_sampled_tokens_req_indices,
                 num_rejected_tokens_np,
                 scheduler_output,
             )
            rejected_per_chunk.append(num_rejected_np)
            last_hidden, hidden = self._forward_draft(
                chunk=chunk,
                input_ids=input_ids,
                positions=positions,
                target_hidden_states=target_hidden_states,
                step_idx=0,
                seq_lens_delta=0,
                num_rejected_np=num_rejected_np,
                num_tokens_padded=input_ids.shape[0],
            )
            sample_hidden_per_chunk.append(last_hidden[last_token_indices])
            hidden_carry_per_chunk.append(hidden[last_token_indices])
            positions_carry_per_chunk.append(positions[last_token_indices])

        sample_hidden = torch.cat(sample_hidden_per_chunk, dim=0)
        draft_logits = self.draft_model.compute_logits(
            _maybe_pad_dim0(
                sample_hidden,
                _get_padded_num_reqs_with_upper_limit(num_reqs,
                                                      runner.max_num_reqs)))
        draft_tokens_step = draft_logits.argmax(dim=-1).to(torch.int32)
        draft_tokens_list = [draft_tokens_step[:num_reqs]]

        if K > 1:
            # Loop steps: uniform decode shape, one query per request. Pad each chunk's carries to the
            # next num_tokens bucket so the loop draft forward hits a precompiled trace.
            padded_nr_per_chunk = [
                _get_padded_token_len(runner.num_tokens_paddings, c.num_reqs)
                for c in chunks
            ]
            loop_hidden = [
                _maybe_pad_dim0(h, p)
                for h, p in zip(hidden_carry_per_chunk, padded_nr_per_chunk)
            ]
            loop_positions = [
                _maybe_pad_dim0(pos, p) for pos, p in zip(
                    positions_carry_per_chunk, padded_nr_per_chunk)
            ]
            # Pre-convert per-chunk rejection counts to device tensors once.
            # Reused across all K-1 loop steps instead of re-doing H2D each step.
            rejected_dev_per_chunk = [
                torch.from_numpy(nr.astype(np.int32, copy=False)).to(
                    runner.device) if nr is not None and np.any(nr) else None
                for nr in rejected_per_chunk
            ]
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
                prev_tokens = draft_tokens_list[-1]
                step_tokens = []
                req_offset = 0
                for ci, chunk in enumerate(chunks):
                    loop_positions[ci] = loop_positions[ci] + 1
                    chunk_prev = prev_tokens[req_offset:req_offset +
                                             chunk.num_reqs]
                    req_offset += chunk.num_reqs
                    loop_input_ids = _maybe_pad_dim0(chunk_prev,
                                                     padded_nr_per_chunk[ci])
                    last_hidden, hidden = self._forward_draft(
                        chunk=chunk,
                        input_ids=loop_input_ids,
                        positions=loop_positions[ci],
                        target_hidden_states=loop_hidden[ci],
                        step_idx=step,
                        seq_lens_delta=step,
                        num_rejected_np=rejected_per_chunk[ci],
                        num_tokens_padded=padded_nr_per_chunk[ci],
                        num_rejected_dev=rejected_dev_per_chunk[ci],
                        loop_query_start_loc=loop_qsl_per_chunk[ci],
                        loop_request_distribution=loop_reqdist_per_chunk[ci],
                    )
                    draft_logits = self.draft_model.compute_logits(last_hidden)
                    step_tokens.append(
                        draft_logits.argmax(dim=-1).to(
                            torch.int32)[:chunk.num_reqs])
                    loop_hidden[ci] = hidden
                draft_tokens_list.append(torch.cat(step_tokens, dim=0))

        draft_tokens = torch.stack(draft_tokens_list, dim=1)
        return draft_tokens.cpu().tolist()

    def _prepare_draft_inputs(
        self,
        chunk: DraftChunkInputs,
        sampled_token_ids: list[list[int]],
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Build draft first-pass inputs for one chunk.

        All token/request indices are chunk-local; the batch-level inputs
        (sampled_token_ids, num_rejected_tokens_np, discard indices) are
        sliced/offset by chunk.start_index.

        Returns:
            input_ids: [padded_chunk_tokens] shifted + bonus-patched
            positions: [padded_chunk_tokens] chunk positions
            last_token_indices: [num_reqs] end of accepted prefix per req
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
        draft_input_ids = chunk.input_ids.clone()
        if draft_input_ids.shape[0] > 1:
            draft_input_ids[:-1] = chunk.input_ids[1:]

        # Last-token index per request = end of the accepted prefix (where
        # the bonus token actually lives), not the original last verify slot.
        last_token_indices_np = (
            query_start_loc_np[1:num_reqs + 1].astype(np.int64) - 1 -
            num_rejected_np.astype(np.int64))
        last_token_indices = torch.from_numpy(last_token_indices_np).to(
            runner.device)

        discard_set = set(discard_sampled_tokens_req_indices)
        req_ids = runner.input_batch.req_ids[start:start + num_reqs]
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
        next_token_ids = torch.from_numpy(next_token_ids_np).to(runner.device)
        draft_input_ids[last_token_indices] = next_token_ids.to(
            draft_input_ids.dtype)

        return (draft_input_ids, chunk.position_ids, last_token_indices,
                num_rejected_np)

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
            seq_lens = chunk_ctx.seq_lens.clone()
            seq_lens[:num_reqs] = seq_lens[:num_reqs] + seq_lens_delta
            if num_rejected_np is not None and np.any(num_rejected_np):
                if num_rejected_dev is None:
                    num_rejected_dev = torch.from_numpy(
                        num_rejected_np.astype(np.int32,
                                               copy=False)).to(runner.device)
                seq_lens[:num_reqs] = seq_lens[:num_reqs] - num_rejected_dev
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

        saved_ctx = runner._attn_metadata_builder_ctx
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=chunk.start_index,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
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
        return {
            name: md
            for name, md in per_layer_attn_metadata.items()
            if name in self._draft_attn_layer_names
        }

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
        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
        ), set_vllm_model_wrapper_context(mesh=self.runner.mesh):
            out = self.draft_model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=target_hidden_states,
            )

        return self._unwrap_model_out(out)

    def precompile(self) -> None:
        """Precompile every draft-side bucket shape so VLLM_XLA_CHECK_RECOMPILATION
        doesn't fire on the first real propose() call.

        Mirrors runner.capture_model's pattern. Covers:
            * first-pass forward (token-count buckets × max/most_model_len)
            * combine_hidden_states (token-count buckets)
            * compute_logits (num_reqs buckets)

        Loop-step forward is not precompiled separately: _dummy_draft_forward
        builds identical dummy attention metadata for both first-pass and
        loop-step, so the XLA graphs are the same.
        """
        if self.runner.enforce_eager:
            return
        self._precompile_combine_hidden_states()
        self._precompile_first_pass()
        self._precompile_compute_logits()

    def _draft_hidden_size(self) -> int:
        return self.draft_model.config.hidden_size

    def _draft_uses_aux_hidden_state(self) -> bool:
        """Whether this draft checkpoint feeds combine_hidden_states the
        concatenation of several target aux hidden-state layers.
        """
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

    def _precompile_combine_hidden_states(self) -> None:
        runner = self.runner
        in_size = self._draft_combine_input_size()
        with runner._precompile_timed("drafter combine_hidden_states"):
            for num_tokens in runner.num_tokens_paddings:
                dummy = torch.zeros(
                    (num_tokens, in_size),
                    dtype=runner._hidden_states_dtype,
                    device=runner.device,
                )
                out = self.draft_model.combine_hidden_states(dummy)
                sync.synchronize(out, wait=True)
                logger.info("  -- drafter combine num_tokens: %d", num_tokens)

    def _precompile_first_pass(self) -> None:
        runner = self.runner
        with runner._precompile_timed("drafter first pass"):
            for num_tokens in runner.num_tokens_paddings:
                self._dummy_draft_forward(
                    num_tokens=num_tokens,
                    num_reqs=runner.num_reqs_max_model_len,
                    use_max_model_len=True,
                )
                if runner.most_model_len is not None:
                    self._dummy_draft_forward(
                        num_tokens=num_tokens,
                        num_reqs=runner.num_reqs_most_model_len,
                        use_max_model_len=False,
                    )
                logger.info("  -- drafter first pass num_tokens: %d",
                            num_tokens)

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
        with runner._precompile_timed("drafter compute_logits"):
            for n in all_sizes:
                dummy_hidden = torch.zeros(
                    (n, hidden_size),
                    dtype=runner._hidden_states_dtype,
                    device=runner.device,
                )
                out = self.draft_model.compute_logits(dummy_hidden)
                sync.synchronize(out, wait=True)
                logger.info("  -- drafter compute_logits n: %d", n)

    def _dummy_draft_forward(
        self,
        num_tokens: int,
        num_reqs: int,
        use_max_model_len: bool,
    ) -> None:
        runner = self.runner

        input_ids = torch.zeros((num_tokens),
                                dtype=torch.int32).to(runner.device)
        positions = torch.zeros(num_tokens,
                                dtype=torch.int32).to(runner.device)
        target_hidden_states = torch.zeros(
            (num_tokens, self._draft_hidden_size()),
            dtype=runner._hidden_states_dtype).to(runner.device)

        actual_num_reqs = min(num_tokens, num_reqs)
        query_lens = [1] * num_reqs
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32).to(runner.device)
        seq_lens = torch.ones((num_reqs, ),
                              dtype=torch.int32).to(runner.device)
        request_distribution = torch.tensor(
            [actual_num_reqs, actual_num_reqs, actual_num_reqs],
            dtype=torch.int32).to(runner.device)

        saved_ctx = getattr(runner, "_attn_metadata_builder_ctx", None)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=0,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            position_ids_override=positions,
        )
        try:
            slot_mappings = runner.empty_slot_mappings
            per_layer_attn_metadata, _ = runner._build_attention_metadata(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_query_len=1,
                num_tokens_padded=num_tokens,
                num_reqs_padded=num_reqs,
                slot_mappings=slot_mappings,
            )
            with (
                    runner.maybe_select_dummy_loras(
                        runner.lora_config,
                        np.array([num_tokens], dtype=np.int32)),
                    set_forward_context(per_layer_attn_metadata,
                                        self.vllm_config, 0),
                    set_vllm_model_wrapper_context(mesh=runner.mesh),
            ):
                out = self.draft_model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=target_hidden_states,
                )
            last_hidden, _ = self._unwrap_model_out(out)
            sync.synchronize(last_hidden, wait=True)
        finally:
            runner._attn_metadata_builder_ctx = saved_ctx
