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

Greedy draft sampling (argmax + d2t remap) by default. With
TPU_DSPARK_SAMPLED_DRAFT=1 each draft token is SAMPLED from the draft
distribution q = softmax((base + markov(prev)) / T) at the request's
temperature (Gumbel-max: argmax(logits + T * G)), and the pre-temperature
draft logits are kept per request so the rejection sampler can run the
lossless test u * q(x) <= p(x) with residual max(0, p - q). At T > 0 this
accepts sum(min(p, q)) per position instead of p(argmax q).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from vllm.config import VllmConfig

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
        self._decode_samples = None
        # Sampled drafting (see module docstring). The instance attribute
        # shadows the greedy compiled method that DFlashProposer.propose and
        # the precompile path call.
        self.sampled_draft = os.environ.get("TPU_DSPARK_SAMPLED_DRAFT", "0") == "1"
        self._cur_chunk = None
        self._q_entries: list = []
        self._q_all: torch.Tensor | None = None
        self._q_tok_all: torch.Tensor | None = None
        self._q_rows: dict[str, int] = {}
        # Draft temperature scale tau: drafts come from softmax(q / (tau * T)).
        # Any q keeps the output distribution exact; tau > 1 flattens an
        # over-confident drafter towards the target's distribution.
        self.draft_temp_scale = float(os.environ.get("TPU_DSPARK_DRAFT_TEMP_SCALE", "1.0"))
        self._q_log_every = int(os.environ.get("TPU_DSPARK_SAMPLED_DRAFT_LOG", "0"))
        self._q_calls = 0
        if self.sampled_draft:
            self._dflash_forward_and_sample = self._sampled_forward_and_sample

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
        # vLLM 0.29 drafting options this proposer does not implement yet.
        spec = self.speculative_config
        if (
            spec.draft_sample_method != "greedy"
            or spec.enable_adaptive_verification
            or spec.dspark_draft_topk is not None
        ):
            raise ValueError(
                "DSpark on TPU does not support draft_sample_method="
                f"{spec.draft_sample_method!r}, enable_adaptive_verification="
                f"{spec.enable_adaptive_verification}, dspark_draft_topk="
                f"{spec.dspark_draft_topk}; only greedy drafting of a fixed "
                "block is implemented."
            )
        if self.sampled_draft and not self.sample_from_anchor:
            raise ValueError("TPU_DSPARK_SAMPLED_DRAFT=1 needs sample_from_anchor.")
        logger.info(
            "DSpark: sample_from_anchor=%s, K=%d, sampled_draft=%s.",
            self.sample_from_anchor,
            K,
            self.sampled_draft,
        )

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

    def _target_aux_layer_ids(self) -> list[int]:
        hf_config = self.speculative_config.draft_model_config.hf_config
        hf_dict = hf_config.to_dict()
        if hf_dict.get("model_type") == "deepseek_v41":
            aux_ids = hf_dict.get("eagle_aux_hidden_state_layer_ids")
            if aux_ids:
                return list(aux_ids)
            # V4.1 DSpark conditions on layer inputs, which the delayed mHC
            # target captures at the same index, rather than layer outputs.
            return list(self._target_layer_ids)
        return super()._target_aux_layer_ids()

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

        # Optional model capability, initialized after target weight sharing.
        # Returning None keeps the standard DSpark sampling contract.
        prepare = getattr(self.draft_model, "prepare_tpu_draft_sampler", None)
        if prepare is not None:
            self._decode_samples = prepare(
                tp_size=self.runner.mesh.size,
                block_size=self.block_size,
                sample_from_anchor=self.sample_from_anchor,
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
        assert self.draft_model is not None
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]

        if self._decode_samples is not None:
            tokens = self._decode_samples(valid_hidden, input_ids, block_size)
            if tokens is not None:
                return tokens, hidden

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

    # ------------------------------------------------------------------ #
    # Sampled drafting (TPU_DSPARK_SAMPLED_DRAFT=1)
    # ------------------------------------------------------------------ #
    def _prepare_dflash_inputs(self, chunk, *args, **kwargs):
        self._cur_chunk = chunk
        return super()._prepare_dflash_inputs(chunk, *args, **kwargs)

    def propose(self, *args, **kwargs):
        if not self.sampled_draft:
            return super().propose(*args, **kwargs)
        self._q_entries = []
        result = super().propose(*args, **kwargs)
        self._finish_q_cache()
        return result

    def _sampled_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk, self._cur_chunk = self._cur_chunk, None
        padded_num_reqs = input_ids.shape[0] // block_size
        temps_cpu = torch.zeros(padded_num_reqs, dtype=torch.float32)
        if chunk is not None and chunk.num_reqs > 0:
            start = chunk.start_index
            temps_cpu[: chunk.num_reqs].copy_(
                self.runner.input_batch.temperature_cpu_tensor[
                    start : start + chunk.num_reqs
                ].to(torch.float32)
            )
        temps = temps_cpu.to(input_ids.device)
        gumbel_u = torch.rand(
            (padded_num_reqs, block_size, self._draft_vocab_size()),
            dtype=torch.float32,
            device=input_ids.device,
        )
        inv_tau = torch.full(
            (1,), 1.0 / self.draft_temp_scale, dtype=torch.float32
        ).to(input_ids.device)
        tokens, hidden, q_logits = self._sampled_forward_compiled(
            input_ids, positions, block_size, temps, gumbel_u, inv_tau
        )
        if chunk is not None:
            self._q_entries.append((chunk.start_index, chunk.num_reqs, q_logits, tokens))
        return tokens, hidden

    def _draft_vocab_size(self) -> int:
        if not hasattr(self, "_vocab_cache"):
            w = self.draft_model.model.markov_head.markov_w2.weight
            self._vocab_cache = int(w.shape[0])
        return self._vocab_cache

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _sampled_forward_compiled(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
        temps: torch.Tensor,
        gumbel_u: torch.Tensor,
        inv_tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backbone forward + sequential Markov sampling from q, in-graph.

        x_k = argmax(q_k + T * G_k), q_k = base_k + markov(x_{k-1}), which is
        a draw from softmax(q_k / T) (T = 0 rows: argmax, the greedy draft).
        Returns the drafts, the hidden states and the pre-temperature draft
        logits q [padded_num_reqs, block_size, vocab] (bf16).
        """
        assert self.draft_model is not None
        hidden = self.draft_model.model(input_ids=input_ids, positions=positions)
        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]
        logits = self.draft_model.compute_draft_logits(valid_hidden).to(torch.float32)
        vocab = logits.shape[-1]
        logits_3d = logits.view(padded_num_reqs, block_size, vocab)
        u = gumbel_u[:, :, :vocab].clamp(min=1e-20, max=1.0 - 1e-7)
        noise = -torch.log(-torch.log(u)) * temps.view(padded_num_reqs, 1, 1)
        ids_3d = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )
        prev = ids_3d[:, 0]
        draft_tokens = []
        q_list = []
        for i in range(block_size):
            bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
            # Cached q is already divided by tau, so the rejection sampler's
            # softmax(q / T) is exactly the distribution sampled here.
            q_i = (logits_3d[:, i] + bias.to(torch.float32)) * inv_tau
            step = self.draft_model.map_draft_to_target(
                (q_i + noise[:, i]).argmax(dim=-1)
            )
            draft_tokens.append(step)
            q_list.append(q_i.to(torch.bfloat16))
            prev = step
        return (
            torch.stack(draft_tokens, dim=1),
            hidden,
            torch.stack(q_list, dim=1),
        )

    def _finish_q_cache(self) -> None:
        """Index this step's draft logits by request id for the next verify."""
        rows: dict[str, int] = {}
        qs, toks = [], []
        offset = 0
        req_ids = self.runner.input_batch.req_ids
        for start, num_reqs, q, tok in self._q_entries:
            padded, k, vocab = q.shape
            for r in range(num_reqs):
                rid = req_ids[start + r]
                if rid is not None:
                    rows[rid] = offset + r * k
            qs.append(q.reshape(padded * k, vocab))
            toks.append(tok.reshape(padded * k).to(torch.int32))
            offset += padded * k
        self._q_entries = []
        self._q_rows = rows
        if not qs:
            self._q_all = self._q_tok_all = None
            return
        self._q_all = qs[0] if len(qs) == 1 else torch.cat(qs, dim=0)
        self._q_tok_all = toks[0] if len(toks) == 1 else torch.cat(toks, dim=0)

    def sampled_draft_rows(
        self,
        req_ids: list,
        draft_lengths_cpu: np.ndarray,
        draft_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Draft logits for the verify rows of ``req_ids`` (in batch order,
        ``draft_lengths_cpu`` drafts each), aligned with ``draft_token_ids``.

        A row is valid only if its request was drafted by the last propose
        and the cached draft token equals the token being verified; invalid
        rows fall back to the greedy-draft test in the rejection sampler.
        """
        if not self.sampled_draft or self._q_all is None:
            return None
        padded_len = draft_token_ids.shape[0]
        idx = np.zeros(padded_len, dtype=np.int64)
        has = np.zeros(padded_len, dtype=np.bool_)
        pos = 0
        for rid, n in zip(req_ids, draft_lengths_cpu):
            n = int(n)
            base = self._q_rows.get(rid)
            if base is not None and n > 0:
                idx[pos : pos + n] = base + np.arange(n)
                has[pos : pos + n] = True
            pos += n
        device = draft_token_ids.device
        idx_t = torch.from_numpy(idx).to(device)
        has_t = torch.from_numpy(has).to(device)
        rows = self._q_all.index_select(0, idx_t)
        cached = self._q_tok_all.index_select(0, idx_t)
        valid = has_t & (cached == draft_token_ids.to(torch.int32))
        if self._q_log_every > 0:
            self._q_calls += 1
            if self._q_calls % self._q_log_every == 0:
                n_has = int(has.sum())
                n_valid = int(valid.sum().item())
                logger.info(
                    "DSpark sampled-draft rows: drafted=%d cached=%d valid=%d (%.1f%%)",
                    int(sum(int(n) for n in draft_lengths_cpu[: len(req_ids)])),
                    n_has,
                    n_valid,
                    100.0 * n_valid / max(n_has, 1),
                )
        return rows, valid
