# SPDX-License-Identifier: Apache-2.0

"""Dummy-only TPU decode benchmark: allocate real blocks, skip prompt compute.

The pool is zeroed once before the first request, never per request. This is
intentionally not a correctness mode or a prefix-cache implementation.

Diagnostics: each worker logs its device memory (SYNTHETIC_DECODE_HBM: free,
runtime reservation, true free) when the pool is allocated, after the
first-request zero and after 1, 50, 300, 1000 and 2000 forward steps.
SYNTHETIC_DECODE_SAMPLING_FALLBACKS=1 also counts the verify steps whose
candidate sampling fell back to the full vocabulary; the count is materialized
every step, which splits the step's deferred sampling ops, so timing runs leave
it off.
"""

import os

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorMetadata,
)
from vllm.logger import init_logger

logger = init_logger("vllm.synthetic_decode_connector")

_GIB = 2**30
# Forward steps (load calls) after which device memory is logged.
_HBM_LOG_CALLS = (1, 50, 300, 1000, 2000)
# Candidate-sampling fallback counter (SYNTHETIC_DECODE_SAMPLING_FALLBACKS=1).
_FB = {"acc": None, "calls": 0, "installed": False}


def _log_hbm(tag):
    """Log this worker's device memory (diagnostic only)."""
    stats = {}
    try:
        free, total = torch.accelerator.get_memory_info()
        stats.update(free_gib=free / _GIB, limit_gib=total / _GIB)
        stats["in_use_gib"] = (total - free) / _GIB
        stats["peak_allocated_gib"] = torch.accelerator.max_memory_allocated() / _GIB
        ms = torch.accelerator.memory_stats()
        stats["reserved_gib"] = ms.get("reserved_bytes.all.current", 0) / _GIB
        stats["peak_reserved_gib"] = ms.get("reserved_bytes.all.peak", 0) / _GIB
        stats["true_free_gib"] = stats["free_gib"] - stats["reserved_gib"]
    except Exception as e:  # noqa: BLE001 - diagnostics must not stop the run
        stats["error"] = repr(e)
    logger.info(
        "SYNTHETIC_DECODE_HBM %s %s",
        tag,
        " ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in stats.items()
        ),
    )


def _install_fallback_counter(runner):
    """Wrap the runner's candidate verify call to count not-exact steps."""
    if _FB["installed"]:
        return
    fn = getattr(type(runner), "_sample_spec_verify_candidates", None)
    g = getattr(fn, "__globals__", None)
    if g is None or "candidate_verify_decisions" not in g:
        logger.info("SYNTHETIC_DECODE_SAMPLING_FALLBACK counter not installed")
        return
    orig = g["candidate_verify_decisions"]

    def counted(*args, **kwargs):
        out = orig(*args, **kwargs)
        from vllm_torchtpu.utils import synchronize_tensors

        miss = (out[3] == 0).to(torch.int32).reshape(())
        _FB["acc"] = miss if _FB["acc"] is None else _FB["acc"] + miss
        # Materialize each step's count (no wait) so the deferred graph does
        # not chain every step.
        synchronize_tensors(_FB["acc"], wait=False)
        _FB["calls"] += 1
        return out

    g["candidate_verify_decisions"] = counted
    _FB["installed"] = True
    logger.info("SYNTHETIC_DECODE_SAMPLING_FALLBACK counter installed")


def _reset_fallback_counter(device):
    if _FB["installed"]:
        _FB["acc"] = torch.zeros((), dtype=torch.int32, device=device)
        _FB["calls"] = 0


def _log_fallbacks(tag):
    if _FB["installed"]:
        n = int(_FB["acc"].item()) if _FB["acc"] is not None else 0
        logger.info(
            "SYNTHETIC_DECODE_SAMPLING_FALLBACK %s calls=%d fallbacks=%d",
            tag,
            _FB["calls"],
            n,
        )


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
        self._load_calls = 0

    def register_kv_caches(self, kv_caches):
        # TPU uses aliased typed FA/Mamba views of the same raw pool. The
        # upstream per-layer fill is not suitable for this shared layout.
        pass

    def register_runner(self, runner):
        if not runner._unified_kv_layout or not runner.kv_cache_raw_tensors:
            raise ValueError("Synthetic TPU benchmark requires unified raw KV pool")
        self._runner = runner
        _log_hbm("pool_allocated")
        if os.environ.get("SYNTHETIC_DECODE_SAMPLING_FALLBACKS") == "1":
            try:
                _install_fallback_counter(runner)
                _reset_fallback_counter(runner.kv_cache_raw_tensors[0].device)
            except Exception as e:  # noqa: BLE001 - diagnostics must not stop the run
                logger.info("SYNTHETIC_DECODE_SAMPLING_FALLBACK install failed: %r", e)

    def start_load_kv(self, forward_context, **kwargs):
        # Unit tests build the connector without __init__.
        self._load_calls = getattr(self, "_load_calls", 0) + 1
        if self._load_calls in _HBM_LOG_CALLS:
            _log_hbm(f"load_call={self._load_calls}")
            _log_fallbacks(f"load_call={self._load_calls}")
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
            if _FB["installed"]:
                _reset_fallback_counter(self._runner.kv_cache_raw_tensors[0].device)
            self._pool_initialized = True
            logger.info(
                "SYNTHETIC_DECODE_POOL_READY raw_tensors=%d",
                len(self._runner.kv_cache_raw_tensors),
            )
            _log_hbm("pool_ready")
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
