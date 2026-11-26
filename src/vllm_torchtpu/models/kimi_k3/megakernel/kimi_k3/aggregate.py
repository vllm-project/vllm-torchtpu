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
from vllm_torchtpu.models.kimi_k3.megakernel.bridge import hbm
from vllm_torchtpu.models.kimi_k3.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as _config_mod
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import expert_banks
from vllm_torchtpu.models.kimi_k3.megakernel.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

logger = init_logger(__name__)


def active() -> _config_mod.KernelConfig:
    return _config_mod.active()


def role_config(config: MegakernelEngineConfig, role: str) -> MegakernelEngineConfig:
    """The aggregate engine's configuration as program ``role`` sees it."""
    from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.program import MAX_STEP_TOKENS

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

    _prefill_ep: int = 8
    _decode_ep: int = 8

    def __init__(
        self,
        config: MegakernelEngineConfig,
        presharded: Path,
        draft_checkpoint: Path | None = None,
        *,
        tp_rank: int,
    ) -> None:
        from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.prefill import (
            KimiPrefiller,
        )
        from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.program import (
            build_decoder,
        )

        kernel = active()
        if kernel.prefill_expert_parallel not in (8, 16):
            raise NotImplementedError(
                "the K3 aggregate engine requires prefill_expert_parallel 8 or 16"
            )
        if kernel.decode_expert_parallel == 16 and kernel.prefill_expert_parallel != 16:
            raise NotImplementedError(
                "the K3 aggregate engine requires prefill_expert_parallel 16 "
                "when decode_expert_parallel is 16"
            )
        if kernel.prefill_expert_parallel == 8 and kernel.kd_hot_plan:
            raise NotImplementedError(
                "the K3 aggregate engine does not run kd_hot_plan at EP8; "
                "use kd_replica_plan instead"
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
        self._decode_ep = kernel.decode_expert_parallel
        self._loaded = False

    def load_weights(self) -> None:
        if self._loaded:
            return
        from .kernels.rank_loader import (
            EXPERT_FAMILIES,
            PREFILL_SHARED,
        )

        started = time.perf_counter()
        self.decoder.load_weights()
        weights = self.decoder.ops.weights
        kernel = active()
        shared_names = list(PREFILL_SHARED)
        if self._prefill_ep != self._decode_ep:
            excluded = set(EXPERT_FAMILIES)
            if kernel.prefill_expert_plan:
                excluded.update(("router", "router_bias"))
            shared_names = [n for n in shared_names if n not in excluded]
        elif expert_banks.bank_name(EXPERT_FAMILIES[0], 0) in weights:
            shared_names = [n for n in shared_names if n not in EXPERT_FAMILIES] + [
                expert_banks.bank_name(family, bank)
                for family in EXPERT_FAMILIES
                for bank in range(len(expert_banks.SHARED_BANK_SIZES))
            ]
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
        finished = set(step_input.finished or ())
        if finished:
            self.decoder.forget(finished)
        prefill, decode = split_rows(step_input.requests, self.decoder.tracked)
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
        return self.decoder.step(StepInput(decode, (), (), step_input.mask))
