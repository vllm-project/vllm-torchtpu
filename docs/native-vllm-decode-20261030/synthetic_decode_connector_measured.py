# SPDX-License-Identifier: Apache-2.0

"""Dummy-only TPU decode benchmark: allocate real blocks, skip prompt compute.

The pool is zeroed once before the first request, never per request. This is
intentionally not a correctness mode or a prefix-cache implementation.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorMetadata,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class SyntheticTPUDecodeConnector(DecodeBenchConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        if vllm_config.load_config.load_format != "dummy":
            raise ValueError("Synthetic decode requires load_format=dummy")
        if extra.get("synthetic_decode_benchmark") is not True:
            raise ValueError("Explicit synthetic_decode_benchmark=true required")
        super().__init__(vllm_config, role, kv_cache_config)
        self._runner = None
        self._pool_initialized = False
        self._audit_steps = 0

    def register_kv_caches(self, kv_caches):
        # TPU uses aliased typed FA/Mamba views of the same raw pool. The
        # upstream per-layer fill is not suitable for this shared layout.
        pass

    def register_runner(self, runner):
        if not runner._unified_kv_layout or not runner.kv_cache_raw_tensors:
            raise ValueError("Synthetic TPU benchmark requires unified raw KV pool")
        self._runner = runner

    def start_load_kv(self, forward_context, **kwargs):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DecodeBenchConnectorMetadata):
            raise TypeError("Unexpected synthetic decode metadata")
        if not metadata.reqs_to_fill:
            return
        if self._runner is None:
            raise RuntimeError("Runner not registered")
        if not self._pool_initialized:
            from vllm_torchtpu.utils import synchronize_tensors

            # After startup dummy forwards, before the first real request.
            # Synchronize each raw tensor to bound temporary allocation size.
            for tensor in self._runner.kv_cache_raw_tensors:
                tensor.zero_()
                synchronize_tensors(tensor, wait=True)
            self._pool_initialized = True
            logger.info(
                "SYNTHETIC_DECODE_POOL_READY raw_tensors=%d",
                len(self._runner.kv_cache_raw_tensors),
            )
        # Blocks are allocated by the real HMA scheduler. Future requests use
        # arbitrary existing values; do not clear active requests' aliased state.

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if num_external_tokens:
            logger.info(
                "SYNTHETIC_DECODE_ADMIT request=%s prompt=%d skipped=%d groups=%s",
                request.request_id,
                request.num_prompt_tokens,
                num_external_tokens,
                [len(group) for group in blocks.get_block_ids()],
            )

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        if meta.reqs_to_fill or self._audit_steps < 8:
            logger.info(
                "SYNTHETIC_DECODE_STEP reqs=%d scheduled_tokens=%d "
                "max_per_req=%d new_fills=%d",
                len(scheduler_output.num_scheduled_tokens),
                sum(scheduler_output.num_scheduled_tokens.values()),
                max(scheduler_output.num_scheduled_tokens.values(), default=0),
                len(meta.reqs_to_fill),
            )
        self._audit_steps += 1
        return meta


# Preserve the platform's benchmark connector dispatch; the explicit external
# module path selects this implementation ahead of the built-in registry.
DecodeBenchConnector = SyntheticTPUDecodeConnector
