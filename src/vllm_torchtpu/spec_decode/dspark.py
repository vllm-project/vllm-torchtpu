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

import torch
from vllm.config import VllmConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    def __init__(self, runner, vllm_config: VllmConfig):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        hf_config = speculative_config.draft_model_config.hf_config
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}

        super().__init__(runner, vllm_config)

        if not self.dflash_config.get("target_layer_ids"):
            raise ValueError(
                "DSpark requires target layer ids in the draft config (one of "
                "eagle_aux_hidden_state_layer_ids, dspark_target_layer_ids, "
                "target_layer_ids, or dflash_config.target_layer_ids)."
            )

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
        # Drafting is always greedy here (see the warning above), which the
        # vocabulary-parallel Markov chain requires.
        self._vocab_parallel_markov = envs.TPU_DSPARK_VOCAB_PARALLEL_MARKOV

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
        if self._vocab_parallel_markov:
            if not hasattr(self.draft_model, "prepare_vocab_parallel_markov"):
                raise RuntimeError(
                    "TPU_DSPARK_VOCAB_PARALLEL_MARKOV: "
                    f"{type(self.draft_model).__name__} has no vocabulary-parallel "
                    "Markov head."
                )
            self.draft_model.prepare_vocab_parallel_markov()
        if envs.TPU_DSPARK_SHARDED_CONTEXT_KV:
            prepare = getattr(
                self.draft_model.model, "prepare_sharded_context_kv", None
            )
            if prepare is None:
                raise RuntimeError(
                    "TPU_DSPARK_SHARDED_CONTEXT_KV: "
                    f"{type(self.draft_model).__name__} has no sharded context-KV "
                    "projection."
                )
            prepare()

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
        """
        assert self.draft_model is not None
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]
        sample_start = 0 if self.sample_from_anchor else 1
        num_slots = block_size - sample_start

        # Reorder the narrow hidden states before the vocabulary projection.
        # Each Markov step then reads contiguous request rows without
        # relayouting the much wider [requests, slots, vocab] logits.
        slot_hidden = (
            valid_hidden.view(padded_num_reqs, block_size, hidden.shape[-1])[
                :, sample_start:
            ]
            .transpose(0, 1)
            .reshape(num_slots * padded_num_reqs, hidden.shape[-1])
        )
        ids_3d = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )
        prev = ids_3d[:, 0]
        draft_tokens = []
        if getattr(self, "_vocab_parallel_markov", False):
            # Greedy on this rank's vocabulary shard; one (max, argmax) pair
            # per rank and row is exchanged per draft position.
            logits = self.draft_model.compute_draft_logits_local(slot_hidden)
            logits_3d = logits.view(num_slots, padded_num_reqs, logits.shape[-1])
            for i in range(num_slots):
                bias = self.draft_model.markov_bias_local(
                    self.draft_model.markov_embed(prev)
                )
                step = self.draft_model.vocab_parallel_argmax(logits_3d[i] + bias)
                draft_tokens.append(step)
                prev = step
            return torch.stack(draft_tokens, dim=1), hidden

        logits = self.draft_model.compute_draft_logits(slot_hidden)
        logits_3d = logits.view(num_slots, padded_num_reqs, logits.shape[-1])
        for i in range(num_slots):
            bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
            step = self.draft_model.map_draft_to_target(
                (logits_3d[i] + bias).argmax(dim=-1)
            )
            draft_tokens.append(step)
            prev = step
        draft_tokens_chunk = torch.stack(draft_tokens, dim=1)

        return draft_tokens_chunk, hidden
