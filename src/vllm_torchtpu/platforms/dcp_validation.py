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

Mirrors pcp_validation.py for the decode axis. vLLM checks the DCP size
against TP and PCP (vllm/config/parallel.py) and, for models without MLA,
against the KV head count (vllm/config/model.py). The TPU DCP kernels add two
contracts that nothing else enforces and whose violation is a wrong answer
rather than an error:

1. A DCP group must hold one KV head. vLLM orders the ranks TP-major with PCP
   innermost inside a DCP group, gives TP rank r the KV head
   r // (tp / num_kv_heads), and every PCP rank of a TP rank holds that rank's
   heads, so a group holds one head only if dcp divides
   (tp / num_kv_heads) * pcp. Qwen3.5-397B (2 KV heads): TP8 admits DCP 1, 2,
   4; TP16 up to 8; TP32 up to 16. TP8 with DCP8 would attend head-1 queries
   against head-0 pages on ranks 0..3. An MLA cache is one latent that every
   rank holds, so every DCP size vLLM admits holds one head, including the
   PCP manager's DCP = TP x PCP (GLM-5.2 at TP2, PCP4, DCP8).
2. The Raiden connector names a decode engine's layout from its parallel sizes.
   ``TPURaidenConnectorWorker._raiden_qwen35_admission_topology`` (in
   ``distributed/kv_transfer/tpu_connector.py``) names tp{T}dcp{D}_decode only
   for dp == 1; with attention-DP it returns dp{N}[tp{T}]_decode without
   reading DCP, so the producer and consumer would disagree on the page owner.
   (``_raiden_glm_admission_topology`` refuses a consumer with both DP and TP
   above 1, so DCP above 1 cannot reach it there.) The check covers an engine
   one of whose connectors runs that worker: a ``TPUConnector`` whose Raiden
   backend is on (always for ``TPURaidenConnector``), on its own or as a
   MultiConnector child, whatever its role. An engine whose connectors are all
   others, an offload connector for example, never runs that naming. The check
   leaves PCP > 1 alone: pcp_validation.py refuses DCP with PCP on the default
   PCP path, and under the PCP manager every Raiden admission topology refuses
   a PCP > 1 consumer, and a PCP > 1 producer with DP > 1, so no name drops the
   DCP axis.

pcp_validation.py and the PCP manager own the other PCP x DCP rules.
"""

from __future__ import annotations

import copy
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
    use_mla: bool
    kv_role: str | None

    @property
    def enabled(self) -> bool:
        return self.dcp_size > 1

    @property
    def num_kv_heads(self) -> int:
        """The distinct KV heads the ranks hold: one MLA latent, else the
        model's KV heads (0 when unknown)."""
        return 1 if self.use_mla else self.total_num_kv_heads


def admissible_dcp_sizes(
    tp_size: int, total_num_kv_heads: int, pcp_size: int = 1
) -> tuple[int, ...]:
    """DCP sizes whose groups hold one KV head over TP x PCP, ascending.

    A group spans the PCP ranks of consecutive TP ranks, so it holds one head
    when it stays inside the (tp / heads) * pcp ranks of one head. When every
    TP rank holds distinct heads (tp < heads), only the PCP ranks of one TP
    rank share them.
    """
    if tp_size <= 0 or total_num_kv_heads <= 0 or pcp_size <= 0:
        return (1,)
    if tp_size < total_num_kv_heads or tp_size % total_num_kv_heads:
        ranks_per_head = pcp_size
    else:
        ranks_per_head = tp_size // total_num_kv_heads * pcp_size
    return tuple(d for d in range(1, ranks_per_head + 1) if ranks_per_head % d == 0)


def _child_configs(vllm_config: Any) -> list[Any]:
    """The engine configs a MultiConnector builds its children from: per entry
    of ``kv_connector_extra_config["connectors"]``, a shallow copy of the
    engine config whose KV transfer config is that entry with the engine's
    ``engine_id``."""
    from vllm.config import KVTransferConfig

    transfer = vllm_config.kv_transfer_config
    children = []
    for entry in transfer.kv_connector_extra_config.get("connectors", []):
        child = copy.copy(vllm_config)
        child.kv_transfer_config = KVTransferConfig(
            **entry, engine_id=transfer.engine_id
        )
        children.append(child)
    return children


def _raiden_connector(vllm_config: Any) -> str | None:
    """The class name of the engine's first connector that runs
    ``TPURaidenConnectorWorker``, or None.

    The classes are resolved as vLLM resolves them, MultiConnector children
    on the configs MultiConnector builds them from, and a ``TPUConnector`` is
    judged by the backend choice it makes itself.
    """
    # Deferred: the connector modules are heavy, and only an attention-DP
    # engine with DCP and a KV transfer config gets here.
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
        TPUConnector,
        _use_raiden_connector,
    )

    connector_cls = KVConnectorFactory.get_connector_class(
        vllm_config.kv_transfer_config
    )
    if issubclass(connector_cls, MultiConnector):
        for child in _child_configs(vllm_config):
            if (name := _raiden_connector(child)) is not None:
                return name
        return None
    if issubclass(connector_cls, TPUConnector) and (
        connector_cls.force_raiden_connector or _use_raiden_connector(vllm_config)
    ):
        return connector_cls.__name__
    return None


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
            use_mla=model_config is not None and bool(model_config.use_mla),
            kv_role=None if kv_transfer_config is None else kv_transfer_config.kv_role,
        )

    @staticmethod
    def validate_platform_config(vllm_config: Any) -> DcpStaticConfig:
        config = DcpStaticSupportValidator.from_vllm_config(vllm_config)
        if not config.enabled:
            return config
        if (config.tp_size * config.pcp_size) % config.dcp_size:
            raise ValueError(
                f"tp_size={config.tp_size} x pcp_size={config.pcp_size} must be "
                f"divisible by dcp_size={config.dcp_size}."
            )
        if config.num_kv_heads > 0:
            admissible = admissible_dcp_sizes(
                config.tp_size, config.num_kv_heads, config.pcp_size
            )
            if config.dcp_size not in admissible:
                raise ValueError(
                    "A DCP group must share one KV head: with "
                    f"{config.num_kv_heads} KV heads over TP {config.tp_size} x "
                    f"PCP {config.pcp_size}, a DCP group of {config.dcp_size} "
                    "ranks (PCP ranks innermost, then consecutive TP ranks) holds "
                    "more than one head, so --decode-context-parallel-size must "
                    f"be one of {list(admissible)}."
                )
        if config.dp_size > 1 and config.kv_role is not None and config.pcp_size == 1:
            raiden = _raiden_connector(vllm_config)
            if raiden is not None:
                raise NotImplementedError(
                    f"data_parallel_size={config.dp_size} with "
                    f"decode_context_parallel_size={config.dcp_size} on a KV "
                    f"{config.kv_role}: {raiden} runs the Raiden connector "
                    "worker, which names this layout dp{N}[tp{T}]_decode without "
                    "the DCP axis, so the peers would disagree on page "
                    "ownership. Use dp 1 (tp{T}dcp{D}_decode) until the "
                    "connector names the dp x tp x dcp layout."
                )
        logger.info(
            "DCP admitted: dcp=%d tp=%d pcp=%d kv_heads=%d%s, "
            "page-interleaved ownership page %% dcp.",
            config.dcp_size,
            config.tp_size,
            config.pcp_size,
            config.num_kv_heads,
            " (MLA latent)" if config.use_mla else "",
        )
        return config
