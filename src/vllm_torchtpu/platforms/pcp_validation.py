# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any


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

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        kv_transfer_config = vllm_config.kv_transfer_config

        is_kv_producer = None
        if kv_transfer_config is not None:
            is_kv_producer = kv_transfer_config.is_kv_producer

        return PcpStaticConfig(
            pcp_size=parallel_config.prefill_context_parallel_size,
            interleave_size=parallel_config.cp_kv_cache_interleave_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            async_scheduling=bool(scheduler_config.async_scheduling),
            speculative_enabled=vllm_config.speculative_config is not None,
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
