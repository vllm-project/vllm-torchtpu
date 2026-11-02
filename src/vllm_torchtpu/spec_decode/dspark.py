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

Draft sampling: greedy by default. With
``draft_sample_method="probabilistic"``, each Markov step samples at the
request temperature, and the draft distribution q goes to the rejection
sampler: accept with min(1, p / q) (synthetic: by its schedule), recover
from max(0, p - q). Reduced-vocab (d2t) drafters stay greedy.

On TPU, sampling runs in its own program after the draft forward
(``_markov_sample_probs``); the forward stays the greedy graph plus its
logits as an output. Other splits hang under DP + EP or return NaN logits.
"""

from __future__ import annotations

import collections

import numpy as np
import torch
from vllm.config import VllmConfig

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.utils import DraftChunkInputs, sample_draft_tokens
from vllm_torchtpu.utils import synchronize_device

logger = init_logger(__name__)


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
        spec = self.speculative_config
        # As upstream DSparkSpeculator, under any rejection method: synthetic
        # rejection takes acceptance from its schedule, and q still shapes
        # the recovery distribution.
        self.probabilistic = spec.draft_sample_method == "probabilistic"
        # Upstream drafting options this proposer does not implement yet. They
        # are ignored, not rejected: drafting stays over a fixed block.
        if spec.enable_adaptive_verification or spec.dspark_draft_topk is not None:
            logger.warning(
                "DSpark on TPU does not support enable_adaptive_verification=%s, "
                "dspark_draft_topk=%s; ignoring them and drafting a fixed block.",
                spec.enable_adaptive_verification,
                spec.dspark_draft_topk,
            )
        # Per-step draft distributions (probabilistic drafting): one
        # [rows, K, vocab] float32 tensor per draft chunk from the latest
        # propose(), plus each drafted request's flat row into their
        # concatenation. The next step's verify gathers q for the drafts it
        # checks; a request without an entry (e.g. drafted before a greedy
        # step) verifies as one-hot, which stays exact.
        self._draft_probs: list[torch.Tensor] | None = None
        self._draft_probs_rows: dict[str, int] = {}
        # (temperatures [rows, 1], uniform [K, rows, vocab]) for the next
        # draft forward, built by `_prepare_draft_sampling`.
        self._pending_sampling_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        # Host tensors whose non-blocking uploads may still be in flight: a
        # `.to(device, non_blocking=True)` keeps reading the host memory until
        # the transfer completes, so a temporary that dies at function return
        # uploads garbage once the device queue is deep (async scheduling).
        # Held for a few steps, like the runner's `_h2d_reuse`.
        self._host_keepalive: collections.deque = collections.deque(maxlen=8)
        # (draft logits [rows * block, vocab], input_ids) from
        # `_run_draft_forward`, consumed by `_finish_draft_forward`
        # (probabilistic path).
        self._pending_logits: tuple[torch.Tensor, torch.Tensor] | None = None
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
        vocab_size = int(self.vllm_config.model_config.get_vocab_size())
        if self.probabilistic:
            hf_config = self.speculative_config.draft_model_config.hf_config
            hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}
            draft_vocab = hf_dict.get("draft_vocab_size")
            if draft_vocab is not None and int(draft_vocab) != vocab_size:
                logger.warning(
                    "DSpark: probabilistic drafting needs a full-vocab drafter "
                    "(draft_vocab_size=%s != vocab_size=%d); drafting greedily.",
                    draft_vocab,
                    vocab_size,
                )
                self.probabilistic = False
            else:
                logger.info("DSpark: probabilistic drafting enabled.")
                # Create the sampling generator now, with no upload in flight:
                # torch_tpu's Generator constructor uploads its state and can
                # hang behind an in-flight non-blocking upload, such as the
                # block table a draft forward stages before its first draw.
                synchronize_device()
                self.runner._get_sampling_generator()

    # --- Probabilistic drafting -------------------------------------------

    @torch.no_grad()
    def propose(self, *args, **kwargs):
        # The previous step's q was consumed by the verify that ran before
        # this propose; drop it before drafting the new one.
        self._draft_probs = None
        self._draft_probs_rows = {}
        return super().propose(*args, **kwargs)

    def _needs_sampling_inputs(
        self, chunk: DraftChunkInputs | None, warmup: bool
    ) -> bool:
        """Whether the next draft forward runs the probabilistic graph."""
        if not self.probabilistic:
            return False
        if warmup:
            return True
        return chunk is not None and not self.runner.input_batch.all_greedy

    def _prepare_draft_sampling(
        self,
        chunk: DraftChunkInputs | None,
        input_ids: torch.Tensor,
        block_size: int,
        warmup: bool = False,
    ) -> None:
        self._pending_sampling_inputs = None
        if not self._needs_sampling_inputs(chunk, warmup):
            return
        rows = input_ids.shape[0] // block_size
        self._pending_sampling_inputs = self._draft_sampling_inputs(rows, chunk)

    def _run_draft_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
        chunk: DraftChunkInputs | None,
        warmup: bool = False,
    ) -> torch.Tensor:
        if not self._needs_sampling_inputs(chunk, warmup):
            # Greedy drafting (config, or an all-greedy batch): same tokens
            # as before, no q to keep.
            return super()._run_draft_forward(
                input_ids, positions, block_size, chunk, warmup
            )
        if warmup:
            # Precompile the greedy graph too; a live step picks one by
            # input_batch.all_greedy.
            super()._run_draft_forward(input_ids, positions, block_size, chunk, warmup)
        # Probabilistic drafting: the draft forward (with its greedy chain,
        # see `_dflash_forward_logits`) runs inside the forward context and
        # also hands out the draft logits; `_finish_draft_forward` samples
        # from them in a separate program after the context block.
        greedy_tokens, _, logits = self._dflash_forward_logits(
            input_ids, positions, block_size
        )
        self._pending_logits = (logits, input_ids)
        # Placeholder: `_finish_draft_forward` replaces it with the sampled drafts.
        return greedy_tokens

    def _finish_draft_forward(
        self,
        draft_tokens_chunk: torch.Tensor,
        chunk: DraftChunkInputs | None,
        warmup: bool = False,
    ) -> torch.Tensor:
        if not self._needs_sampling_inputs(chunk, warmup):
            return draft_tokens_chunk
        pending = self._pending_sampling_inputs
        self._pending_sampling_inputs = None
        assert pending is not None, "_prepare_draft_sampling() must run first"
        temperatures, uniform = pending
        pending_logits = self._pending_logits
        self._pending_logits = None
        assert pending_logits is not None, "_run_draft_forward() must run first"
        logits, input_ids = pending_logits
        block_size = self.block_size
        draft_tokens_chunk, draft_probs = self._markov_sample_probs(
            logits, input_ids, block_size, temperatures, uniform
        )
        if chunk is not None:
            self._record_draft_probs(chunk, draft_probs)
        return draft_tokens_chunk

    def _draft_sampling_inputs(
        self, rows: int, chunk: DraftChunkInputs | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row temperatures `[rows, 1]` (0 past the live requests, so
        padding rows draft greedily) and uniform draws `[K, rows, vocab]`
        for the Gumbel-max draft sample.

        K leads so each Markov step reads a contiguous `[rows, vocab]`
        slice: a `[rows, K, vocab]` layout read as `uniform[:, i]` hangs the
        TPU draft program under EP-DP lockstep (never completes, no error).
        """
        runner = self.runner
        K = self.speculative_config.num_speculative_tokens
        vocab_size = self.vllm_config.model_config.get_vocab_size()
        temps_cpu = torch.zeros(rows, dtype=torch.float32)
        if chunk is not None:
            start, n = chunk.start_index, chunk.num_reqs
            temps_cpu[:n] = runner.input_batch.temperature_cpu_tensor[
                start : start + n
            ].to(torch.float32)
        temperatures = temps_cpu.unsqueeze(1).to(runner.device, non_blocking=True)
        self._host_keepalive.append(temps_cpu)
        get_generator = getattr(runner, "_get_sampling_generator", None)
        generator = get_generator() if get_generator is not None else None
        uniform = torch.rand(
            (K, rows, vocab_size),
            dtype=torch.float32,
            device=runner.device,
            generator=generator,
        )
        return temperatures, uniform

    def _record_draft_probs(
        self, chunk: DraftChunkInputs, draft_probs: torch.Tensor
    ) -> None:
        if self._draft_probs is None:
            self._draft_probs = []
        req_ids = self.runner.input_batch.req_ids
        # This chunk's first row in the concatenation of the recorded chunks.
        base = sum(probs.shape[0] for probs in self._draft_probs)
        for r in range(chunk.num_reqs):
            self._draft_probs_rows[req_ids[chunk.start_index + r]] = base + r
        self._draft_probs.append(draft_probs)

    def gather_draft_probs(
        self,
        req_ids: list[str],
        draft_lengths_cpu: np.ndarray,
        padded_logits_length: int,
    ) -> torch.Tensor | None:
        """q for one verify chunk: `[padded_logits_length, vocab]` rows in
        the rejection sampler's draft-token order (request-major, the first
        `draft_lengths_cpu[i]` slots of request i), or None when the drafts
        were not sampled probabilistically (greedy step, no drafts kept, or
        a request this proposer did not draft), in which case the sampler
        verifies them as one-hot.
        """
        if self._draft_probs is None:
            return None
        K = self.speculative_config.num_speculative_tokens
        index = np.zeros(padded_logits_length, dtype=np.int64)
        pos = 0
        for req_id, n in zip(req_ids, draft_lengths_cpu):
            n = int(n)
            if n == 0:
                continue
            row = self._draft_probs_rows.get(req_id)
            if row is None or n > K or pos + n > padded_logits_length:
                return None
            index[pos : pos + n] = row * K + np.arange(n)
            pos += n
        if pos == 0:
            return None
        if len(self._draft_probs) == 1:
            source = self._draft_probs[0]
        else:
            source = torch.cat(self._draft_probs, dim=0)
        source = source.reshape(-1, source.shape[-1])
        index_host = torch.from_numpy(index)
        index_device = index_host.to(source.device, non_blocking=True)
        self._host_keepalive.append(index_host)
        return source[index_device]

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The greedy draft graph (`_dflash_forward_and_sample`, verbatim)
        that additionally returns the draft logits `[rows * block_size,
        vocab]` for the probabilistic sampler. The greedy Markov chain is
        kept in the program on purpose: without a consumer of the logits
        inside the program, the live-step logits come out NaN on TPU.
        """
        assert self.draft_model is not None
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]
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
        return draft_tokens_chunk, hidden, logits + 0

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _markov_sample_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        block_size: int,
        temperatures: torch.Tensor,
        uniform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential Markov sampling over `_dflash_forward_logits` output.

        `logits` is `[rows * block_size, vocab]`; the chain seeds from each
        request's anchor (slot 0 of `input_ids`). Each step samples from
        softmax((logits + bias) / T) (rows with T == 0 take the argmax) and
        continues from the sampled token. `uniform` is `[K, rows, vocab]`
        (see `_draft_sampling_inputs`). Returns the drafts `[rows, K]` and
        the distributions `[rows, K, vocab]` float32 the tokens were drawn
        from.
        """
        assert self.draft_model is not None
        padded_num_reqs = logits.shape[0] // block_size
        logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])
        prev = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )[:, 0]
        draft_tokens = []
        draft_probs = []
        sample_start = 0 if self.sample_from_anchor else 1
        for i in range(sample_start, block_size):
            bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
            token_ids, probs = sample_draft_tokens(
                logits_3d[:, i] + bias, temperatures, uniform[i - sample_start]
            )
            step = self.draft_model.map_draft_to_target(token_ids)
            draft_tokens.append(step)
            draft_probs.append(probs)
            prev = step
        return torch.stack(draft_tokens, dim=1), torch.stack(draft_probs, dim=1)

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
