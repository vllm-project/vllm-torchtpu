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
"""DSpark speculative decoding for TPU.

DSpark is semi-autoregressive parallel drafting: one DFlash-style parallel
block forward (anchor + noise tokens, non-causal within the block, context
KV precomputed from the target's aux hidden states), followed by a
lightweight sequential Markov head that biases each position's draft logits
on the previously sampled token. Upstream implements DSparkSpeculator as a
small DFlashSpeculator subclass; this proposer mirrors that on top of the
TPU DFlashProposer.

Dense-checkpoint layout (``sample_from_anchor`` absent or true, as in
upstream DSparkSpeculator): the query block is K tokens — the anchor is the
FIRST prediction position and every slot predicts the NEXT token — versus
DFlash's 1 + K (anchor is bonus only). Speculators-format checkpoints set
``sample_from_anchor=False`` and keep the DFlash layout.

PoC scope: greedy draft sampling only (argmax + d2t remap); probabilistic
Gumbel-coupled sampling is not implemented.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
from vllm.config import VllmConfig

import vllm_torchtpu.envs as envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.utils import DraftChunkInputs

logger = init_logger(__name__)


class DraftConfidences(NamedTuple):
    """One proposal's per-chunk ``[rows, K]`` acceptance confidences at the
    chunk's bucketed rows, and each chunk's ``(start_index, num_reqs)`` span
    of the input batch."""

    chunks: list[torch.Tensor]
    spans: list[tuple[int, int]]


class DSparkProposer(DFlashProposer):
    def __init__(self, runner, vllm_config: VllmConfig):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        hf_config = speculative_config.draft_model_config.hf_config
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}

        super().__init__(runner, vllm_config)

        # Same key/default as upstream DSparkSpeculator: dense checkpoints
        # omit it (-> True); the speculators converter writes it explicitly.
        self.sample_from_anchor = bool(hf_dict.get("sample_from_anchor", True))
        K = self.speculative_config.num_speculative_tokens
        # vLLM 0.29 drafting options this proposer does not implement yet. They
        # are ignored, not rejected: drafting stays greedy over a fixed block.
        spec = self.speculative_config
        if (
            spec.draft_sample_method != "greedy"
            or spec.enable_adaptive_verification
            or spec.dspark_draft_topk is not None
        ):
            logger.warning(
                "DSpark on TPU does not support draft_sample_method=%r, "
                "enable_adaptive_verification=%s, dspark_draft_topk=%s; "
                "ignoring them and using greedy drafting of a fixed block.",
                spec.draft_sample_method,
                spec.enable_adaptive_verification,
                spec.dspark_draft_topk,
            )
        # Whether the compiled draft step scores its drafts; it traces against
        # this, and nothing turns it on until a caller takes the confidences.
        self.use_confidence_head = False
        self._draft_confidences: DraftConfidences | None = None
        # Whether the greedy Markov loop argmaxes each rank's vocab shard
        # instead of the all-gathered full-vocab logits; set by load_model
        # when the draft model opts in. The compiled draft step traces on it.
        self._vocab_parallel_markov = False
        # Whether the vocab-parallel Markov chain runs as one fused kernel
        # (TPU_DSPARK_MARKOV_KERNEL); set by load_model, traced like the above.
        self._fused_markov = False
        # The query block follows K, not the block length the checkpoint was
        # trained at; upstream vLLM likewise only enforces K == block for
        # Qwen3-Omni DSpark. The checkpoint's block is a query-block length in
        # both layouts (K for dense, 1 + K for speculators-format), so compare
        # it to block_size rather than K. In-block attention is non-causal, so
        # a mismatch shifts every slot's hidden state, not just the slots past
        # the trained length: outputs stay exact (the target verifies), but
        # acceptance can drop at every position.
        trained_block = hf_dict.get("dspark_block_size") or hf_dict.get("block_size")
        if trained_block is not None and int(trained_block) != self.block_size:
            logger.warning(
                "DSpark: num_speculative_tokens=%d gives a %d-token query "
                "block, but the checkpoint was trained with block size %s; "
                "every draft slot is out of the training distribution, so "
                "check the acceptance rate.",
                K,
                self.block_size,
                trained_block,
            )
        logger.info("DSpark: sample_from_anchor=%s, K=%d.", self.sample_from_anchor, K)

    # A reduced-vocab DSpark checkpoint carries its own lm_head
    # ([draft_vocab, hidden] plus a d2t remap) and possibly its own
    # embed_tokens; force-sharing the target's would clobber them. Honor the
    # has_own_embed_tokens / has_own_lm_head flags the loader sets (upstream
    # load_dspark_model does the same via _should_share).
    _force_share_target_embeddings = False

    def _query_block_size(self, K: int) -> int:
        # Dense layout: K query tokens (anchor + K-1 noise), every slot
        # predicts the next token. Speculators-format checkpoints keep the
        # DFlash 1 + K layout.
        return K if self.sample_from_anchor else K + 1

    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        for attr in (
            "compute_draft_logits",
            "markov_embed",
            "markov_bias",
            "map_draft_to_target",
            "has_own_lm_head",
            "has_own_embed_tokens",
        ):
            if not hasattr(self.draft_model, attr):
                raise RuntimeError(
                    f"DSpark draft model lacks {attr}(); expected a "
                    "Qwen3DSpark/Gemma4DSpark-family model."
                )
        # Opt-in interface for drafts whose LM head is vocab-parallel: the
        # draft's enable_vocab_parallel_markov_argmax() prepares its shard of
        # the Markov head and returns whether the shard path applies; then the
        # draft step calls compute_draft_logits_shard() (no all-gather) and
        # markov_argmax_shard() (bias on the shard, argmax, exchange of one
        # (max, id) pair per rank and row). Greedy ids are identical to the
        # argmax of the gathered logits. Drafts without the method keep the
        # gathered path.
        enable = getattr(self.draft_model, "enable_vocab_parallel_markov_argmax", None)
        self._vocab_parallel_markov = (
            callable(enable) and not self._draft_replicated and bool(enable())
        )
        if self._vocab_parallel_markov:
            logger.info(
                "DSpark: greedy Markov drafting argmaxes the vocab-parallel "
                "logits per rank; full-vocab logits are not gathered."
            )
        # Opt-in on top of the shard path: the draft's enable_fused_markov(k)
        # arms one Pallas call for the whole K-step chain (collective over the
        # TP group, self-tested) and returns whether it armed; the draft step
        # then calls markov_draft_fused() instead of the K-step loop.
        enable_fused = getattr(self.draft_model, "enable_fused_markov", None)
        if (
            envs.TPU_DSPARK_MARKOV_KERNEL
            and self._vocab_parallel_markov
            and callable(enable_fused)
        ):
            k_steps = self.block_size - (0 if self.sample_from_anchor else 1)
            self._fused_markov = bool(enable_fused(k_steps))
            if self._fused_markov and os.environ.get("TPU_DSPARK_MARKOV_KERNEL_CHECK"):
                # Debug: run every draft step both ways and count differing
                # drafts (two compiled variants; host syncs every step).
                self._markov_check = [0, 0, 0]
                self._dflash_forward_and_sample = self._checked_draft_step

    def _observe_draft_aux(
        self, aux: torch.Tensor, chunk_index: int, chunk: DraftChunkInputs
    ) -> None:
        if not self.use_confidence_head:
            return
        if chunk_index == 0:
            self._draft_confidences = DraftConfidences([], [])
        confidences = self._draft_confidences
        assert confidences is not None
        confidences.chunks.append(aux)
        confidences.spans.append((chunk.start_index, chunk.num_reqs))

    def take_draft_confidences(self) -> DraftConfidences | None:
        """The last proposal's confidences; None once taken."""
        confidences, self._draft_confidences = self._draft_confidences, None
        return confidences

    def _checked_draft_step(self, input_ids, positions, block_size):
        """TPU_DSPARK_MARKOV_KERNEL_CHECK: the fused step, then the K-step loop
        on the same inputs; logs the running count of differing drafts."""
        step = DSparkProposer._dflash_forward_and_sample
        fused, aux = step(self, input_ids, positions, block_size)
        self._fused_markov = False
        try:
            looped, _ = step(self, input_ids, positions, block_size)
        finally:
            self._fused_markov = True
        a, b = fused.cpu(), looped.cpu()
        stats = self._markov_check
        stats[0] += 1
        stats[1] += a.numel()
        stats[2] += int((a != b).sum())
        if stats[0] <= 3 or stats[0] % 50 == 0:
            logger.info(
                "TPU_DSPARK_MARKOV_KERNEL_CHECK: steps=%d drafts=%d differing=%d "
                "(last rows=%d, first row fused=%s loop=%s)",
                stats[0],
                stats[1],
                stats[2],
                a.shape[0],
                a[0].tolist(),
                b[0].tolist(),
            )
        return fused, aux

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone forward + sequential greedy Markov sampling, in-graph.

        Dense layout: every query slot predicts the token at its position + 1,
        so all block_size slots are sampled (DFlash samples only the mask
        slots). Bonus-anchor layout: the anchor is a pure bonus token, so only
        the K mask slots (1..block_size-1) are sampled. Either way the Markov
        chain seeds from the anchor input id and runs a static K-iteration
        loop: embedding gather + rank-r bias + argmax.

        With ``use_confidence_head`` the second output is each sampled slot's
        acceptance confidence ``[padded_num_reqs, K]`` instead of the hidden
        states. A slot's confidence reads its hidden state and the previous
        token, never the token it drafts.
        """
        assert self.draft_model is not None
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]

        # Draft-vocab logits for every slot; the Markov bias is added in
        # draft space and ids are remapped to target vocab after argmax.
        # The vocab-parallel path keeps each rank's vocab shard of them.
        vocab_parallel = self._vocab_parallel_markov
        if vocab_parallel:
            logits = self.draft_model.compute_draft_logits_shard(valid_hidden)
        else:
            logits = self.draft_model.compute_draft_logits(valid_hidden)
        logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])

        ids_3d = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )
        prev = ids_3d[:, 0]
        sample_start = 0 if self.sample_from_anchor else 1
        if (
            vocab_parallel
            and self._fused_markov
            and padded_num_reqs <= self.draft_model.fused_markov_max_rows
            and not self.use_confidence_head
            and logits.dtype == torch.bfloat16
        ):
            # The whole chain in one kernel: same ids as the loop below.
            draft_tokens_chunk = self.draft_model.markov_draft_fused(
                logits_3d[:, sample_start:], prev
            )
            return draft_tokens_chunk, hidden
        draft_tokens = []
        markov_embeds = []
        for i in range(sample_start, block_size):
            markov_embed = self.draft_model.markov_embed(prev)
            if vocab_parallel:
                draft_ids = self.draft_model.markov_argmax_shard(
                    logits_3d[:, i], markov_embed
                )
            else:
                bias = self.draft_model.markov_bias(markov_embed)
                draft_ids = (logits_3d[:, i] + bias).argmax(dim=-1)
            step = self.draft_model.map_draft_to_target(draft_ids)
            draft_tokens.append(step)
            markov_embeds.append(markov_embed)
            prev = step
        draft_tokens_chunk = torch.stack(draft_tokens, dim=1)

        if not self.use_confidence_head:
            return draft_tokens_chunk, hidden
        sampled_hidden = valid_hidden.view(padded_num_reqs, block_size, -1)[
            :, sample_start:
        ]
        confidence = self.draft_model.compute_confidence(
            sampled_hidden.flatten(0, 1),
            torch.stack(markov_embeds, dim=1).flatten(0, 1),
        )
        return draft_tokens_chunk, confidence.view(padded_num_reqs, -1)
