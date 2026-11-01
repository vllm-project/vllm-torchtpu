# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 aggregate: the prefill and decode programs on one engine.

:class:`KimiAggregate` (the ``aggregate`` role's :class:`MegakernelProgram`,
selected by an engine without a KV transfer config) holds a
:class:`KimiPrefiller` and a :class:`KimiDecoder` over the same devices and the
same vLLM unified pools, and one copy of the weights they share: the decoder
loads its pre-shard as the decode role does, and the prefill takes the
decoder's tensors for every family that is the same (EP8 experts, attention,
router, embedding and LM head) and reads only its sequence-parallel projections
(``rank_loader.PREFILL_SHARED``).

A step hands each request to one program, as a P/D pair splits it:

* a prompt chunk goes to the prefill; the prompt's last token goes to the
  decoder, which imports the KDA state the prefill left in the pool and
  samples the first token (the KV connector's one-token-prefill flow);
* a decoding request goes to the decoder.

The prefill runs first in a step, so the decoder sees its pages and states.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import hbm
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3.config import active
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

logger = init_logger(__name__)


def role_config(config: MegakernelEngineConfig, role: str) -> MegakernelEngineConfig:
    """The aggregate engine's configuration as program ``role`` sees it."""
    from vllm_torchtpu.megakernel.kimi_k3.program import MAX_STEP_TOKENS

    if role == "decode":
        return dataclasses.replace(config, role="decode")
    return dataclasses.replace(
        config,
        role="prefill",
        max_num_batched_tokens=min(config.max_num_batched_tokens, MAX_STEP_TOKENS),
        num_speculative_tokens=0,
        num_speculative_tokens_per_batch_size=None,
        synthetic_acceptance_rates=None,
        synthetic_acceptance_rates_by_width=None,
    )


def split_rows(
    requests: Sequence[RequestInput], decoding: set[str]
) -> tuple[list[RequestInput], list[RequestInput]]:
    """Each request's rows for the prefill and for the decoder (both in batch
    order; a row with nothing for a program has ``scheduled`` 0).

    A request in ``decoding`` past its prompt decodes. Any other scheduled
    request is computing known tokens: the prefill takes them up to the last
    known one (as a prompt of ``num_tokens - 1`` tokens), and when the step
    reaches it, the decoder takes that token with everything before it computed.
    """
    prefill, decode = [], []
    for row in requests:
        if row.scheduled <= 0 or (
            row.req_id in decoding and row.computed >= row.num_prompt_tokens
        ):
            prefill.append(dataclasses.replace(row, scheduled=0))
            decode.append(row)
            continue
        last = row.num_tokens - 1
        end = min(row.end, last)
        prefill.append(
            dataclasses.replace(
                row,
                scheduled=max(end - row.computed, 0),
                num_tokens=last,
                num_prompt_tokens=last,
            )
        )
        if row.partial:
            decode.append(dataclasses.replace(row, scheduled=0))
        else:
            decode.append(dataclasses.replace(row, computed=last, scheduled=1))
    return prefill, decode


class KimiAggregate:
    """See the module docstring."""

    def __init__(
        self,
        config: MegakernelEngineConfig,
        presharded: Path,
        draft_checkpoint: Path | None = None,
        *,
        tp_rank: int,
    ) -> None:
        from vllm_torchtpu.megakernel.kimi_k3.prefill import KimiPrefiller
        from vllm_torchtpu.megakernel.kimi_k3.program import build_decoder

        kernel = active()
        if kernel.prefill_expert_parallel not in (8, 16):
            raise NotImplementedError(
                "the K3 aggregate engine requires prefill_expert_parallel 8 or 16"
            )
        if kernel.prefill_expert_parallel == 8 and (
            kernel.kd_hot_plan or kernel.kd_replica_plan
        ):
            raise NotImplementedError(
                "the K3 aggregate engine does not run hot-expert plans "
                "(K3_HOT_PLAN, K3_REPLICA_PLAN): they change the shared experts"
            )
        if config.attention_dp != 1:
            raise NotImplementedError(
                "the K3 aggregate engine decodes at attention_dp 1"
            )
        self.config = config
        self.decoder = build_decoder(
            role_config(config, "decode"), presharded, draft_checkpoint, tp_rank=tp_rank
        )
        # In the aggregate engine, the prefill and decoder share the unified
        # pools on each device without a KV connector transfer; the decoder
        # indexes the pools by its kernel mesh position (self.decoder.rank), so
        # the prefill writes pages and KDA states in that same order.
        self.prefiller = KimiPrefiller(
            role_config(config, "prefill"),
            presharded,
            draft_checkpoint,
            tp_rank=self.decoder.rank,
            compute_logits=False,
        )
        self.rank = self.decoder.rank
        self._prefill_ep = kernel.prefill_expert_parallel
        self._loaded = False

    def load_weights(self) -> None:
        if self._loaded:
            return
        from vllm_torchtpu.megakernel.kimi_k3.kernels.rank_loader import (
            EXPERT_FAMILIES,
            PREFILL_SHARED,
        )

        started = time.perf_counter()
        self.decoder.load_weights()
        weights = self.decoder.ops.weights
        kernel = active()
        shared_names = list(PREFILL_SHARED)
        if getattr(self, "_prefill_ep", 8) != 8:
            excluded = set(EXPERT_FAMILIES)
            if kernel.prefill_expert_plan:
                excluded.update(("router", "router_bias"))
            shared_names = [n for n in shared_names if n not in excluded]
        missing = [n for n in shared_names if n not in weights]
        if missing:
            raise RuntimeError(f"K3 aggregate: the decoder holds no {missing}")
        self.prefiller.load_weights({n: weights[n] for n in shared_names})
        self._loaded = True
        if self.rank == 0:
            logger.info(
                "K3 aggregate: weights resident in %.1f s; HBM %s",
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
        self.prefiller.warmup()
        self.decoder.warmup()

    def step(self, step_input: StepInput) -> StepOutput:
        prefill, decode = split_rows(step_input.requests, self.decoder.tracked)
        finished = set(step_input.finished or ())
        # A request computing again (preempted) restarts in the decoder.
        restarted = [
            r.req_id
            for r, d in zip(prefill, decode, strict=True)
            if r.scheduled > 0 and r.req_id in self.decoder.slot_map
        ]
        if restarted:
            self.decoder.forget(restarted)
        if step_input.copies or any(r.scheduled > 0 for r in prefill):
            self.prefiller.step(StepInput(prefill, finished, step_input.copies))
        else:
            for req_id in finished:
                self.prefiller._requests.pop(req_id, None)
        return self.decoder.step(StepInput(decode, finished, (), step_input.mask))
