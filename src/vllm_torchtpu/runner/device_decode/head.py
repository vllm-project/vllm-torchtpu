# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The device-decode head: sample, gather logprobs, update the slots, pack.

`decode_head` is one compiled program per (rows, SamplerVariant). It turns a
pure-decode step's logits into the sampled tokens and the int32 buffer of the
step's single device-to-host copy, and advances each request's slot in place,
as `post_update` does in vLLM's Model Runner V2
(vllm/v1/worker/gpu/input_batch.py). Sampling and logprobs are the stock TPU
sampler's math (`sampler_math`). Only the uniforms differ: they come from
`rng.counter_uniform`, keyed by (seed, position, token id), so given the same
uniforms the head and the stock sampler pick the same token.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.v1.outputs import LogprobsLists

from vllm_torchtpu.layers.adapter.sample.sampler_math import (
    gather_logprobs_impl,
    sample_from_logits_impl,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import random_sampler_variants
from vllm_torchtpu.runner.device_decode.rng import counter_uniform_impl
from vllm_torchtpu.utils import synchronize_tensors

NO_LOGPROBS = -1


@dataclass(frozen=True)
class SamplerVariant:
    """The static arguments of one head program.

    The first three pick the sampler graph exactly as the stock sampler picks
    it for the same batch. `num_logprobs` is the number of top logprobs after
    the sampled token's: NO_LOGPROBS for none.
    """

    all_greedy: bool
    topk_candidates: int
    apply_topk_topp: bool
    num_logprobs: int

    @staticmethod
    def for_batch(runner: Any, num_reqs: int) -> "SamplerVariant":
        """The variant for batch rows [0, num_reqs), from the stock host
        decisions."""
        input_batch = runner.input_batch
        num_logprobs = _num_logprobs(
            input_batch.max_num_logprobs, runner.model_config.max_logprobs
        )
        if input_batch.all_greedy:
            return SamplerVariant(True, 0, True, num_logprobs)
        return SamplerVariant(
            False,
            runner._sampler_topk_candidates(0, num_reqs, None),
            runner._sampler_applies_topk_topp(),
            num_logprobs,
        )

    @staticmethod
    def all_variants(runner: Any) -> list["SamplerVariant"]:
        """Every variant `for_batch` can return for this runner."""
        samplers = [
            (False, *variant)
            for variant in random_sampler_variants(runner._topk_candidate_blocks())
        ]
        samplers.append((True, 0, True))
        widths = sorted({NO_LOGPROBS, 0, max(runner.model_config.max_logprobs, 0)})
        return [SamplerVariant(*sampler, k) for sampler in samplers for k in widths]


def _num_logprobs(max_num_logprobs: int | None, max_logprobs: int) -> int:
    if max_num_logprobs is None:
        return NO_LOGPROBS
    if max_num_logprobs < 0:
        raise NotImplementedError(
            "TPU runner does not support full logprobs (`logprobs=-1`) "
            "with the merged vLLM v1 sampler path yet."
        )
    if max_num_logprobs == 0:
        # vLLM keeps 1 + requested columns per request, so the sampled
        # token's column alone serves a batch that asks for no top logprobs.
        return 0
    # Stock gather_logprobs width: 1 + model_config.max_logprobs.
    return max(max_logprobs, 0)


@dataclass(frozen=True)
class PackedLayout:
    """Columns of the head's int32 [num_rows, width] device-to-host buffer.

    Column 0 is the sampled token and column 1 the position echo: the slot's
    computed-token count the head read, i.e. the step's input position and
    RNG key. With logprobs, `logprob_cols` = 1 + num_logprobs columns of
    token ids follow (the sampled token's first, as stock gather_logprobs
    orders them), then their float32 logprobs as bit patterns, then the
    sampled token's rank.
    """

    num_rows: int
    num_logprobs: int

    @property
    def logprob_cols(self) -> int:
        return 0 if self.num_logprobs < 0 else 1 + self.num_logprobs

    @property
    def width(self) -> int:
        cols = self.logprob_cols
        return 2 if cols == 0 else 2 * cols + 3

    def unpack(
        self, packed_cpu: torch.Tensor, num_reqs: int
    ) -> tuple[torch.Tensor, np.ndarray, LogprobsLists | None]:
        """Rows [0, num_reqs) of a landed buffer: the sampled tokens int64
        [num_reqs, 1], the position echo int32 [num_reqs], and the logprobs
        with the dtypes of stock `LogprobsTensors.tolists()` (or None)."""
        expected = (self.num_rows, self.width)
        if packed_cpu.dtype != torch.int32 or tuple(packed_cpu.shape) != expected:
            raise ValueError(
                f"device-decode packed buffer: expected int32 {expected}, got "
                f"{packed_cpu.dtype} {tuple(packed_cpu.shape)}"
            )
        if not 0 <= num_reqs <= self.num_rows:
            raise ValueError(
                f"device-decode packed buffer: {num_reqs} requests do not fit "
                f"{self.num_rows} rows"
            )
        rows = packed_cpu[:num_reqs].numpy()
        tokens = torch.from_numpy(rows[:, :1].astype(np.int64))
        echo = rows[:, 1].copy()
        cols = self.logprob_cols
        if cols == 0:
            return tokens, echo, None
        values = 2 + cols
        return (
            tokens,
            echo,
            LogprobsLists(
                logprob_token_ids=rows[:, 2:values].copy(),
                logprobs=rows[:, values : values + cols].copy().view(np.float32),
                sampled_token_ranks=rows[:, values + cols].copy(),
            ),
        )


def decode_head_impl(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed: torch.Tensor,
    last_token: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed_lo: torch.Tensor,
    seed_hi: torch.Tensor,
    all_greedy: bool,
    topk_candidates: int,
    apply_topk_topp: bool,
    num_logprobs: int,
    null_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The plain body of `decode_head`, with the variant unpacked."""
    num_rows, vocab_size = logits.shape
    idx = idx_mapping[:num_rows]
    valid = idx != null_slot
    pos = torch.index_select(num_computed, 0, idx)
    prev_tokens = torch.index_select(last_token, 0, idx)
    temperatures = torch.index_select(temperature, 0, idx).to(logits.dtype)
    if all_greedy:
        u = torch.empty((1, 1), dtype=torch.float32, device=logits.device)
    else:
        u = counter_uniform_impl(
            torch.index_select(seed_lo, 0, idx),
            torch.index_select(seed_hi, 0, idx),
            pos,
            vocab_size,
        )
    tokens = sample_from_logits_impl(
        logits,
        temperatures.unsqueeze(1),
        u,
        torch.index_select(top_k, 0, idx).unsqueeze(1),
        torch.index_select(top_p, 0, idx).unsqueeze(1),
        all_greedy=all_greedy,
        topk_candidates=topk_candidates,
        apply_topk_topp=apply_topk_topp,
    )
    sampled = tokens.view(-1).to(torch.int32)
    columns = [sampled.unsqueeze(1), pos.unsqueeze(1)]
    if num_logprobs >= 0:
        logprobs = gather_logprobs_impl(logits, tokens, num_logprobs)
        columns += [
            logprobs.logprob_token_ids,
            logprobs.logprobs.view(torch.int32),
            logprobs.selected_token_ranks.unsqueeze(1),
        ]
    packed = torch.cat(columns, dim=1)
    # Padded rows all target the null slot and write its own values back.
    slots = idx.long()
    num_computed.index_put_((slots,), torch.where(valid, pos + 1, pos))
    last_token.index_put_((slots,), torch.where(valid, sampled, prev_tokens))
    return sampled, packed


_decode_head_compiled = torch.compile(
    decode_head_impl, backend="tpu", fullgraph=True, dynamic=False
)


def decode_head(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed: torch.Tensor,
    last_token: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed_lo: torch.Tensor,
    seed_hi: torch.Tensor,
    *,
    variant: SamplerVariant,
    null_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per row of `logits` [rows, vocab] and advance the
    rows' slots in place.

    Row i belongs to slot `idx_mapping[i]` ([max_num_reqs] int32; rows past
    the batch hold `null_slot`). The slot tensors are [num_slots]: int32
    num_computed, last_token, top_k (0 = off), seed_lo and seed_hi; float32
    temperature and top_p. Each live row samples at position
    num_computed[slot]; then num_computed[slot] += 1 and last_token[slot] is
    the sampled token. Returns the sampled tokens int32 [rows] and the
    `PackedLayout(rows, variant.num_logprobs)` buffer.
    """
    if logits.shape[0] > idx_mapping.shape[0]:
        raise ValueError(
            f"device-decode head: {logits.shape[0]} logits rows exceed the "
            f"{idx_mapping.shape[0]}-entry idx_mapping"
        )
    fn = _decode_head_compiled if logits.device.type == "tpu" else decode_head_impl
    with torch.no_grad():
        return fn(
            logits,
            idx_mapping,
            num_computed,
            last_token,
            temperature,
            top_k,
            top_p,
            seed_lo,
            seed_hi,
            variant.all_greedy,
            variant.topk_candidates,
            variant.apply_topk_topp,
            variant.num_logprobs,
            null_slot,
        )


def head_key(num_rows: int, variant: SamplerVariant) -> tuple:
    """The precompile registry key of one head program. Head calls always run
    with grad off, so the key records grad=False. It leaves out the caller's
    inference mode: a caller whose mode differs from the warmup's recompiles
    every compiled program at its first call, the stock ones too, so keying
    on it would only turn the device path off."""
    return ("decode_head", num_rows, variant, False)


def precompile_head(
    runner: Any, variants: Iterable[SamplerVariant], row_paddings: Iterable[int]
) -> list[tuple]:
    """Compile the head for every (rows, variant) and return their registry
    keys. Called from `_precompile_sampling_subgraphs`, as the stock sampler
    programs are: in the profile run, when there is one, the memory probe
    sees the programs' HBM before the KV cache is sized. Dummy slot tensors
    have the slots' shapes and dtypes; idx_mapping is all null slots."""
    variants = list(variants)
    null_slot = runner.max_num_reqs
    keys = []
    with runner._precompile_timed("device_decode_head"), torch.no_grad():
        idx_mapping = torch.full((runner.max_num_reqs,), null_slot, dtype=torch.int32)
        idx_mapping = idx_mapping.to(runner.device)
        slots = _null_slot_tensors(null_slot + 1, runner.device)
        for num_rows in row_paddings:
            logits = runner._dummy_logits(num_rows)
            for variant in variants:
                out = decode_head(
                    logits, idx_mapping, *slots, variant=variant, null_slot=null_slot
                )
                synchronize_tensors(out)
                keys.append(head_key(num_rows, variant))
    return keys


def _null_slot_tensors(num_slots: int, device: torch.device) -> list[torch.Tensor]:
    """num_computed, last_token, temperature, top_k, top_p, seed_lo, seed_hi
    for `num_slots` slots, each at the null slot's values."""

    def ints() -> torch.Tensor:
        return torch.zeros(num_slots, dtype=torch.int32).to(device)

    def ones() -> torch.Tensor:
        return torch.ones(num_slots, dtype=torch.float32).to(device)

    return [ints(), ints(), ones(), ints(), ones(), ints(), ints()]
