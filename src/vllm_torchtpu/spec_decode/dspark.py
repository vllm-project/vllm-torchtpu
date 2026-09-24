# SPDX-License-Identifier: Apache-2.0
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

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    def __init__(self, runner, vllm_config: VllmConfig):
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
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
        block = hf_dict.get("dspark_block_size") or hf_dict.get("block_size")
        K = self.speculative_config.num_speculative_tokens
        if block is not None and int(block) != K:
            raise ValueError(
                f"num_speculative_tokens={K} does not match the DSpark "
                f"checkpoint's block size {block}; the block/Markov-head "
                "machinery is trained for exactly that block length."
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
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]

        # Draft-vocab logits for every slot; the Markov bias is added in
        # draft space and ids are remapped to target vocab after argmax.
        logits = self.draft_model.compute_draft_logits(valid_hidden)
        logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])

        ids_3d = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )
        prev = ids_3d[:, 0]
        draft_tokens = []
        sample_start = 0 if self.sample_from_anchor else 1
        for i in range(sample_start, block_size):
            bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
            step = self.draft_model.map_draft_to_target(
                (logits_3d[:, i] + bias).argmax(dim=-1)
            )
            draft_tokens.append(step)
            prev = step
        draft_tokens_chunk = torch.stack(draft_tokens, dim=1)

        return draft_tokens_chunk, hidden
