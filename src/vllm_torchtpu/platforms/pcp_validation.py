# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any


def _as_int(value: Any, default: int) -> int:
    return value if isinstance(value, int) else default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return default


@dataclass(frozen=True)
class PcpStaticConfig:
    pcp_size: int
    interleave_size: int
    dcp_size: int
    pipeline_parallel_size: int
    async_scheduling: bool
    speculative_enabled: bool
    is_kv_producer: bool | None

    @property
    def enabled(self) -> bool:
        return self.pcp_size > 1


class PcpStaticSupportValidator:

    @staticmethod
    def from_vllm_config(vllm_config: Any) -> PcpStaticConfig:
        if vllm_config is None:
            return PcpStaticConfig(
                pcp_size=1,
                interleave_size=1,
                dcp_size=1,
                pipeline_parallel_size=1,
                async_scheduling=False,
                speculative_enabled=False,
                is_kv_producer=None,
            )

        parallel_config = getattr(vllm_config, "parallel_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)

        is_kv_producer = None
        if kv_transfer_config is not None:
            raw_is_kv_producer = getattr(kv_transfer_config, "is_kv_producer",
                                         None)
            if isinstance(raw_is_kv_producer, bool):
                is_kv_producer = raw_is_kv_producer

        return PcpStaticConfig(
            pcp_size=_as_int(
                getattr(parallel_config, "prefill_context_parallel_size", 1),
                1,
            ),
            interleave_size=_as_int(
                getattr(parallel_config, "cp_kv_cache_interleave_size", 1),
                1,
            ),
            dcp_size=_as_int(
                getattr(parallel_config, "decode_context_parallel_size", 1),
                1,
            ),
            pipeline_parallel_size=_as_int(
                getattr(parallel_config, "pipeline_parallel_size", 1),
                1,
            ),
            async_scheduling=_as_bool(
                getattr(scheduler_config, "async_scheduling", False)),
            speculative_enabled=getattr(vllm_config, "speculative_config",
                                        None) is not None,
            is_kv_producer=is_kv_producer,
        )

    @staticmethod
    def validate_platform_config(
        vllm_config: Any,
        *,
        multihost_backend: str | None,
    ) -> PcpStaticConfig:
        config = PcpStaticSupportValidator.from_vllm_config(vllm_config)
        if not config.enabled:
            return config

        if config.is_kv_producer is False:
            raise NotImplementedError(
                "PCP runner path does not support KV consumer/decode workers "
                "yet. Disable PCP on decode workers.")
        if config.dcp_size > 1:
            raise NotImplementedError(
                "PCP runner path does not support DCP yet.")
        if config.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "PCP runner path does not support pipeline parallelism yet.")
        if config.speculative_enabled:
            raise NotImplementedError(
                "PCP runner path does not support speculative decoding yet.")
        if multihost_backend:
            raise NotImplementedError(
                "PCP runner path does not support TPU multihost yet.")
        if config.interleave_size <= 0:
            raise ValueError(
                "PCP runner path requires cp_kv_cache_interleave_size > 0.")
        return config
