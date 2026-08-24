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
    speculative_method: str | None
    num_speculative_tokens: int
    is_kv_producer: bool | None
    kv_role: str | None

    @property
    def enabled(self) -> bool:
        return self.pcp_size > 1

    @property
    def pcp_mtp_k1_enabled(self) -> bool:
        return (self.enabled and self.speculative_method == "mtp"
                and self.num_speculative_tokens == 1
                and self.kv_role == "kv_producer")


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
                speculative_method=None,
                num_speculative_tokens=0,
                is_kv_producer=None,
                kv_role=None,
            )

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        kv_transfer_config = vllm_config.kv_transfer_config
        speculative_config = vllm_config.speculative_config

        is_kv_producer = None
        kv_role = None
        if kv_transfer_config is not None:
            is_kv_producer = kv_transfer_config.is_kv_producer
            kv_role = kv_transfer_config.kv_role

        speculative_method = None
        num_speculative_tokens = 0
        if speculative_config is not None:
            speculative_method = speculative_config.method
            num_speculative_tokens = (
                speculative_config.num_speculative_tokens)

        return PcpStaticConfig(
            pcp_size=parallel_config.prefill_context_parallel_size,
            interleave_size=parallel_config.cp_kv_cache_interleave_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            async_scheduling=bool(scheduler_config.async_scheduling),
            speculative_enabled=speculative_config is not None,
            speculative_method=speculative_method,
            num_speculative_tokens=num_speculative_tokens,
            is_kv_producer=is_kv_producer,
            kv_role=kv_role,
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
            if config.speculative_method != "mtp":
                raise NotImplementedError(
                    "PCP speculative decoding currently requires method=mtp.")
            if config.num_speculative_tokens != 1:
                raise NotImplementedError(
                    "PCP MTP currently requires num_speculative_tokens=1.")
            if config.kv_role != "kv_producer":
                raise NotImplementedError(
                    "PCP MTP currently requires kv_role=kv_producer, got "
                    f"{config.kv_role!r}.")
        if multihost_backend:
            raise NotImplementedError(
                "PCP runner path does not support TPU multihost yet.")
        if config.interleave_size <= 0:
            raise ValueError(
                "PCP runner path requires cp_kv_cache_interleave_size > 0.")
        return config
