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
"""Static admission of decode context parallelism (DCP) on TPU.

Mirrors pcp_validation.py for the decode axis. vLLM itself only checks that
DCP divides TP (vllm/config/parallel.py). The TPU batched RPA DCP kernels add
three contracts that nothing else enforces and whose violation is a wrong
answer rather than an error:

1. A DCP group must share one KV head. vLLM gives TP rank r the KV head
   r // (tp / num_kv_heads) and builds DCP groups as contiguous runs of
   dcp ranks inside TP, so dcp must divide tp / num_kv_heads. Qwen3.5-397B
   (2 KV heads): TP8 admits DCP 1, 2, 4; TP16 up to 8; TP32 up to 16. TP8 with
   DCP8 would attend head-1 queries against head-0 pages on ranks 0..3.
2. A decode megakernel serving DCP must place each KV-head group's pages by
   the same page-interleaved rule (page p on DCP rank p % dcp at column
   p // dcp, the rule of the batched RPA kernels and of Raiden's
   tp{T}dcp{D}_decode landing), so the head rule above is the whole admission
   for it as well.
3. The Raiden Stage-3 admission names consumer layouts tp{T}dcp{D}_decode
   only for dp == 1; with attention-DP it returns dp{N}[tp{T}]_decode without
   reading DCP, so the producer and consumer would disagree on the page owner.

PCP together with DCP is handled by pcp_validation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class DcpStaticConfig:
    dcp_size: int
    tp_size: int
    pcp_size: int
    dp_size: int
    total_num_kv_heads: int
    kv_role: str | None

    @property
    def enabled(self) -> bool:
        return self.dcp_size > 1


def admissible_dcp_sizes(tp_size: int, total_num_kv_heads: int) -> tuple[int, ...]:
    """DCP sizes whose groups share one KV head under TP, in ascending order.

    Returns (1,) when every rank holds distinct heads (tp < heads) because a
    group of two such ranks would mix heads.
    """
    if tp_size <= 0 or total_num_kv_heads <= 0:
        return (1,)
    if tp_size < total_num_kv_heads or tp_size % total_num_kv_heads:
        return (1,)
    ranks_per_head = tp_size // total_num_kv_heads
    return tuple(d for d in range(1, ranks_per_head + 1) if ranks_per_head % d == 0)


class DcpStaticSupportValidator:
    @staticmethod
    def from_vllm_config(vllm_config: Any) -> DcpStaticConfig:
        parallel_config = vllm_config.parallel_config
        kv_transfer_config = vllm_config.kv_transfer_config
        # A config built without a model (test fixtures, dry runs) has no KV
        # head count; 0 keeps only the vLLM divisibility rule.
        model_config = vllm_config.model_config
        total_num_kv_heads = (
            0 if model_config is None else int(model_config.get_total_num_kv_heads())
        )
        return DcpStaticConfig(
            dcp_size=int(parallel_config.decode_context_parallel_size),
            tp_size=int(parallel_config.tensor_parallel_size),
            pcp_size=int(parallel_config.prefill_context_parallel_size),
            dp_size=int(parallel_config.data_parallel_size),
            total_num_kv_heads=total_num_kv_heads,
            kv_role=None if kv_transfer_config is None else kv_transfer_config.kv_role,
        )

    @staticmethod
    def validate_platform_config(vllm_config: Any) -> DcpStaticConfig:
        config = DcpStaticSupportValidator.from_vllm_config(vllm_config)
        if not config.enabled:
            return config
        if config.tp_size % config.dcp_size:
            raise ValueError(
                f"tp_size={config.tp_size} must be divisible by "
                f"dcp_size={config.dcp_size}."
            )
        if config.total_num_kv_heads > 0:
            admissible = admissible_dcp_sizes(config.tp_size, config.total_num_kv_heads)
            if config.dcp_size not in admissible:
                raise ValueError(
                    "A DCP group must share one KV head: with "
                    f"{config.total_num_kv_heads} KV heads over TP "
                    f"{config.tp_size}, rank r holds head r // "
                    f"{max(1, config.tp_size // config.total_num_kv_heads)} and "
                    f"the DCP groups are contiguous runs of {config.dcp_size} "
                    f"ranks, so --decode-context-parallel-size must be one of "
                    f"{list(admissible)}."
                )
        if config.dp_size > 1 and config.kv_role is not None and config.pcp_size == 1:
            raise NotImplementedError(
                f"data_parallel_size={config.dp_size} with "
                f"decode_context_parallel_size={config.dcp_size} on a KV "
                f"{config.kv_role}: the Raiden Stage-3 admission names "
                "dp{N}[tp{T}]_decode without the DCP axis, so the peers would "
                "disagree on page ownership. Use dp 1 (tp{T}dcp{D}_decode) until "
                "the admission names the dp x tp x dcp layout."
            )
        logger.info(
            "DCP admitted: dcp=%d tp=%d kv_heads=%d (ranks per head %d), "
            "page-interleaved ownership page %% dcp.",
            config.dcp_size,
            config.tp_size,
            config.total_num_kv_heads,
            max(1, config.tp_size // max(1, config.total_num_kv_heads)),
        )
        return config


def validate_decode_context_parallel_config(vllm_config: Any) -> DcpStaticConfig:
    return DcpStaticSupportValidator.validate_platform_config(vllm_config)
