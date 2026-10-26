# SPDX-License-Identifier: Apache-2.0
"""Host PLE lookup boundary for vLLM piecewise compilation."""

import time

import torch
from vllm.forward_context import get_forward_context

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

from .ple_storage import HostPLETable

logger = init_logger(__name__)


@torch.library.custom_op("qwen4_exp::host_lookup", mutates_args=())
def host_lookup(
    ids: torch.Tensor,
    local_tokens: int,
    embedding_dim: int,
    dp_size: int,
    fp8: bool,
    prefix: str,
) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[prefix]
    table = layer.ple_embedding.ngram_embedding
    assert isinstance(table, HostPLETable)
    if not envs.VLLM_TPU_PLE_HOST_TIMING:
        raw = table.lookup_host(ids.cpu())
        return raw.reshape(dp_size, local_tokens, embedding_dim).to(
            ids.device, non_blocking=raw.is_pinned()
        )
    t0 = time.perf_counter()
    host_ids = ids.cpu()
    t1 = time.perf_counter()
    raw = table.lookup_host(host_ids)
    t2 = time.perf_counter()
    out = raw.reshape(dp_size, local_tokens, embedding_dim).to(
        ids.device, non_blocking=raw.is_pinned()
    )
    t3 = time.perf_counter()
    _timing_record(ids.numel(), t1 - t0, t2 - t1, t3 - t2)
    return out


_timing = {"n": 0, "rows": 0, "d2h": 0.0, "take": 0.0, "h2d": 0.0}


def _timing_record(rows: int, d2h: float, take: float, h2d: float) -> None:
    t = _timing
    t["n"] += 1
    t["rows"] += rows
    t["d2h"] += d2h
    t["take"] += take
    t["h2d"] += h2d
    if t["n"] % 1000 == 0:
        n = t["n"]
        logger.info(
            "PLE host lookup: %d calls, avg %.0f rows, d2h %.3f ms, take %.3f ms, "
            "h2d %.3f ms",
            n,
            t["rows"] / n,
            t["d2h"] / n * 1e3,
            t["take"] / n * 1e3,
            t["h2d"] / n * 1e3,
        )


@host_lookup.register_fake
def _host_lookup_fake(ids, local_tokens, embedding_dim, dp_size, fp8, prefix):
    return ids.new_empty(
        (dp_size, local_tokens, embedding_dim),
        dtype=torch.int8 if fp8 else torch.int16,
    )
