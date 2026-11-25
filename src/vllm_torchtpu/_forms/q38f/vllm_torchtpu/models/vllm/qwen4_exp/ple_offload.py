# SPDX-License-Identifier: Apache-2.0
"""Host PLE lookup boundary for vLLM piecewise compilation."""

import os
import time

import torch
from vllm.forward_context import get_forward_context

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import l6_fskip as _l6_fskip

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
        host_ids = take_host_ids(prefix, ids)
        raw = table.lookup_host(ids.cpu() if host_ids is None else host_ids)
        # q38 L6 fskip (runner/l6_fskip.py): VLLM_TPU_PLE_HOST_FAST_LOOKUP gathers into a
        # 2-deep reused host ring, which the staging fence also orders; count the upload.
        _l6_fskip.S.n += 1
        return raw.reshape(dp_size, local_tokens, embedding_dim).to(
            ids.device, non_blocking=raw.is_pinned()
        )
    t0 = time.perf_counter()
    host_ids = take_host_ids(prefix, ids)
    if host_ids is None:
        host_ids = ids.cpu()
    t1 = time.perf_counter()
    raw = table.lookup_host(host_ids)
    t2 = time.perf_counter()
    _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py)
    out = raw.reshape(dp_size, local_tokens, embedding_dim).to(
        ids.device, non_blocking=raw.is_pinned()
    )
    t3 = time.perf_counter()
    _timing_record(ids.numel(), t1 - t0, t2 - t1, t3 - t2)
    return out


# VLLM_TPU_PLE_HOST_IDS=1: the step's table rows ("ids") are computed on the host
# from the host copies of the step's inputs (Qwen4ExpAdapter stashes them before the
# forward, ``stash_host_ids``), with the same integer arithmetic as the device graph.
# ``host_lookup`` then reads them instead of copying the device ids back. That copy
# was queued behind the previous forward, so the host waited for it, and the device
# sat idle through the copy, the host gather and the upload: 8.0-9.3 ms per
# back-to-back stage-0 step at FR7 kernels (fr7s j-b6e3865c, plecap j-3de85f34,
# m16 j-61d4e8ad, FR7 c96 j-32b33a72). With host ids, stage 0's host does the gather
# and upload while the previous forward still runs. The device still computes its
# ids (unused), so every compiled program is unchanged.
# VLLM_TPU_PLE_HOST_IDS_CHECK=N compares the first N host ids with a blocking copy of
# the device ids and logs the count that differed. Read from os.environ, not envs,
# so the switch is not a compile-cache factor (no program changes).
_host_ids: dict[str, torch.Tensor] = {}
_host_ids_stats = {"used": 0, "fallback": 0, "checked": 0, "bad": 0}


def host_ids_enabled() -> bool:
    return os.environ.get("VLLM_TPU_PLE_HOST_IDS", "0").strip().lower() in ("1", "true", "on")


def _host_ids_check_budget() -> int:
    try:
        return int(os.environ.get("VLLM_TPU_PLE_HOST_IDS_CHECK", "0") or 0)
    except ValueError:
        return 0


def stash_host_ids(prefix: str, ids: torch.Tensor) -> None:
    _host_ids[prefix] = ids


def clear_host_ids() -> None:
    _host_ids.clear()


def take_host_ids(prefix: str, ids: torch.Tensor) -> torch.Tensor | None:
    """The stashed host ids for this lookup, or None (then the caller copies
    the device ids). A stash is used once; one whose shape or dtype does not
    match the device ids is dropped and counted as a fallback."""
    host = _host_ids.pop(prefix, None)
    if host is None:
        return None
    st = _host_ids_stats
    if tuple(host.shape) != tuple(ids.shape) or host.dtype != ids.dtype:
        st["fallback"] += 1
        if st["fallback"] <= 3:
            logger.warning(
                "PLE host ids: stash %s %s does not match device ids %s %s; "
                "copying the device ids",
                tuple(host.shape), host.dtype, tuple(ids.shape), ids.dtype,
            )
        return None
    budget = _host_ids_check_budget()
    if st["checked"] < budget:
        st["checked"] += 1
        if not torch.equal(host, ids.cpu()):
            st["bad"] += 1
        if st["checked"] == budget:
            logger.warning(
                "PLE host ids check: %d of %d host ids differed from the device ids",
                st["bad"], budget,
            )
    st["used"] += 1
    if st["used"] in (1, 1000) or st["used"] % 10000 == 0:
        logger.info(
            "PLE host ids: %d lookups used host ids (%d fallbacks)",
            st["used"], st["fallback"],
        )
    return host


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


# Split variant of host_lookup (VLLM_TPU_PLE_HOST_OVERLAP): `start` begins the
# ids' device-to-host copy and records an event; the decoder layers before the
# PLE layer are dispatched next; `finish` then waits only for that copy, so the
# host gather and the upload overlap those layers on the device instead of
# draining it. Both are graph splitting ops, run eagerly in order.
_pending: dict[str, tuple[torch.Tensor, object, torch.Tensor]] = {}
_check = {"left": envs.VLLM_TPU_PLE_HOST_OVERLAP_CHECK, "bad": 0}


@torch.library.custom_op("qwen4_exp::host_lookup_start", mutates_args=())
def host_lookup_start(ids: torch.Tensor, prefix: str) -> torch.Tensor:
    host_ids = ids.to("cpu", non_blocking=True)
    event = torch.tpu.Event() if hasattr(torch, "tpu") else None
    if event is not None:
        event.record()
    _pending[prefix] = (host_ids, event, ids)
    return ids.new_empty(0)


@host_lookup_start.register_fake
def _host_lookup_start_fake(ids, prefix):
    return ids.new_empty(0)


@torch.library.custom_op("qwen4_exp::host_lookup_finish", mutates_args=())
def host_lookup_finish(
    token: torch.Tensor,
    after: torch.Tensor,
    local_tokens: int,
    embedding_dim: int,
    dp_size: int,
    fp8: bool,
    prefix: str,
) -> torch.Tensor:
    """``token`` orders this after ``host_lookup_start``; ``after`` (the hidden
    state of the last layer before the PLE layer) orders it after those layers'
    dispatch."""
    host_ids, event, device_ids = _pending.pop(prefix)
    if event is not None:
        event.synchronize()
    if _check["left"] > 0:
        # VLLM_TPU_PLE_HOST_OVERLAP_CHECK: compare the first N async copies with
        # a blocking one, so a copy the event did not cover cannot go unseen.
        _check["left"] -= 1
        if not torch.equal(host_ids, device_ids.cpu()):
            _check["bad"] += 1
        if _check["left"] == 0:
            logger.warning(
                "PLE host overlap check: %d of %d async ids copies differed",
                _check["bad"],
                envs.VLLM_TPU_PLE_HOST_OVERLAP_CHECK,
            )
    layer = get_forward_context().no_compile_layers[prefix]
    table = layer.ple_embedding.ngram_embedding
    assert isinstance(table, HostPLETable)
    raw = table.lookup_host(host_ids)
    _l6_fskip.S.n += 1  # q38 L6 fskip (runner/l6_fskip.py)
    return raw.reshape(dp_size, local_tokens, embedding_dim).to(
        after.device, non_blocking=raw.is_pinned()
    )


@host_lookup_finish.register_fake
def _host_lookup_finish_fake(
    token, after, local_tokens, embedding_dim, dp_size, fp8, prefix
):
    return after.new_empty(
        (dp_size, local_tokens, embedding_dim),
        dtype=torch.int8 if fp8 else torch.int16,
    )
