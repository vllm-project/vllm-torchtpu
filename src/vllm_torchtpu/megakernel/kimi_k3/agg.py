# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 aggregated ("mega-agg"): the prefill and decode programs in one
engine on one TP32 slice, over the same unified pools.

:class:`KimiAggregated` holds a :class:`KimiPrefiller` and a
:class:`KimiDecoder`:

* **Weights.** The decoder loads first; the prefill (EP8: the routed experts
  as stored) adopts the decoder's resident expert tensors instead of a second
  copy, which is what lets both programs fit one core's HBM.
* **Pages.** The prefill writes each request's latent pages where the decoder
  under attention DP reads them (``KimiPrefiller(page_groups=dp)``), and its
  KDA state into the shared state blocks the decoder imports: no transfer.
* **Steps.** The scheduler holds a prompt's last token back (the
  ``_mamba_block_aligned_split`` patch), as the P/D producer does
  (``tpu_connector._maybe_truncate_prompt_for_handoff``): prompt chunks before
  it go to the prefill (as a prompt one token shorter, so it finishes there);
  the last prompt token is the request's first decode step, admitted at
  ``computed = prompt - 1`` exactly like a transferred request. A step runs
  the prefill rows, then the decode rows.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3.program import DCP_SIZE, MAX_STEP_TOKENS
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

logger = init_logger(__name__)


def is_prefill_row(row: RequestInput) -> bool:
    """A scheduled prompt chunk that stops before the prompt's last token
    (the prefill's); every other scheduled row is the decoder's."""
    return row.scheduled > 0 and row.end <= row.num_prompt_tokens - 1


def split_rows(
    rows: Sequence[RequestInput],
) -> tuple[list[RequestInput], list[RequestInput]]:
    """``(prefill rows, decode rows)`` of a step, in batch order.

    Prefill rows are handed over as a prompt one token shorter (the last
    prompt token is the decoder's), so the prefill finishes the request at
    ``prompt - 1``. Decode rows are the whole batch with the prefill rows
    unscheduled, so the decoder's output rows line up with the runner's.
    Raises if a row spans the held-back last prompt token.
    """
    prefill, decode = [], []
    for row in rows:
        if row.scheduled > 0 and row.computed < row.num_prompt_tokens - 1 < row.end:
            raise RuntimeError(
                f"K3 agg: request {row.req_id} schedules [{row.computed}, "
                f"{row.end}) across its last prompt token "
                f"{row.num_prompt_tokens - 1}: the scheduler must hold it back"
            )
        if is_prefill_row(row):
            short = row.num_prompt_tokens - 1
            prefill.append(
                dataclasses.replace(
                    row,
                    num_prompt_tokens=short,
                    num_tokens=min(row.num_tokens, short),
                )
            )
            decode.append(dataclasses.replace(row, scheduled=0))
        else:
            decode.append(row)
    return prefill, decode


def agg_configs(
    config: MegakernelEngineConfig,
) -> tuple[MegakernelEngineConfig, MegakernelEngineConfig]:
    """``(prefill config, decode config)`` of the aggregated ``config``: the
    prefill runs its P/D configuration (DCP 32 compute, one 8192-token
    window per step), the decoder the engine's (attention DP, DCP 32 / dp)."""
    prefill = dataclasses.replace(
        config,
        role="prefill",
        attention_dp=1,
        decode_context_parallel_size=DCP_SIZE,
        max_num_batched_tokens=min(config.max_num_batched_tokens, MAX_STEP_TOKENS),
        num_speculative_tokens=0,
        num_speculative_tokens_per_batch_size=None,
    )
    decode = dataclasses.replace(config, role="decode")
    return prefill, decode


class KimiAggregated:
    """The aggregated role's :class:`MegakernelProgram` (module docstring).
    ``tp_rank`` is unused for page ownership: both programs own pages by
    kernel mesh position, as the decoder's DCP attention reads them."""

    def __init__(
        self, config: MegakernelEngineConfig, presharded: Path, *, tp_rank: int
    ) -> None:
        from vllm_torchtpu.megakernel.kimi_k3 import config as kernel_config
        from vllm_torchtpu.megakernel.kimi_k3.decode import KimiDecoder
        from vllm_torchtpu.megakernel.kimi_k3.decode_ops import (
            DecodeOps,
            decode_batch_sizes,
        )
        from vllm_torchtpu.megakernel.kimi_k3.prefill import KimiPrefiller

        active = kernel_config.active()
        if active.prefill_expert_parallel != 8:
            raise NotImplementedError(
                "K3 agg shares the decoder's routed experts with the prefill: "
                "the kernel config needs prefill_expert_parallel=8 "
                f"(has {active.prefill_expert_parallel})"
            )
        if active.attention_dp != config.attention_dp:
            raise RuntimeError(
                f"K3 agg: kernel config attention_dp {active.attention_dp} != "
                f"the engine's {config.attention_dp}"
            )
        self.config = config
        self.prefill_config, self.decode_config = agg_configs(config)
        self.ops = DecodeOps(
            presharded,
            num_slots=2 * config.max_num_seqs + 1,
            batch_sizes=decode_batch_sizes(config.max_num_seqs),
            mamba_ssm_cache_dtype=config.mamba_ssm_cache_dtype,
        )
        self.rank = self.ops.rank
        # Kernel mesh position for both: pages and KDA heads in mesh order
        # (no permutation between the programs), as ``check handoff`` binds them.
        self.decoder = KimiDecoder(
            self.decode_config, self.ops, None, tp_rank=self.rank
        )
        self.prefiller = KimiPrefiller(
            self.prefill_config,
            presharded,
            None,
            tp_rank=self.rank,
            page_groups=config.attention_dp,
        )
        self.stats = {"prefill_steps": 0, "decode_steps": 0, "prefill_s": 0.0}

    # ------------------------------------------------------------ setup
    def load_weights(self) -> None:
        from vllm_torchtpu.megakernel.bridge import hbm

        started = time.perf_counter()
        self.decoder.load_weights()  # the experts the prefill adopts
        self.prefiller.load_weights(shared_experts=self.ops.weights)
        if self.rank == 0:
            logger.info(
                "K3 agg: decode + prefill weights resident (shared experts) in "
                "%.1f s; HBM %s",
                time.perf_counter() - started,
                hbm(self.rank),
            )

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        self.load_weights()
        self.decoder.bind_kv_caches(caches, groups)
        self.prefiller.bind_kv_caches(caches, groups)

    def warmup(self) -> None:
        from vllm_torchtpu.megakernel.bridge import hbm

        self.decoder.warmup()
        self.prefiller.warmup()
        if self.rank == 0:
            logger.info("K3 agg: both programs warm; HBM %s", hbm(self.rank))

    # ------------------------------------------------------------ steps
    def step(self, step_input: StepInput) -> StepOutput:
        prefill, decode = split_rows(step_input.requests)
        if prefill:
            started = time.perf_counter()
            # Its first tokens are discarded: the decoder produces them.
            self.prefiller.step(
                StepInput(prefill, step_input.finished, step_input.copies)
            )
            self.stats["prefill_steps"] += 1
            self.stats["prefill_s"] += time.perf_counter() - started
        else:
            for req in step_input.finished:  # (a prefill step drops them itself)
                self.prefiller.forget(req)
        self.stats["decode_steps"] += 1
        return self.decoder.step(
            StepInput(decode, step_input.finished, (), step_input.mask)
        )
