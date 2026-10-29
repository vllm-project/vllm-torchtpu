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


# VLLM_TPU_PLE_HOST_PRELOOKUP: the lookup runs before the backbone is called
# (Qwen4ExpForCausalLM.forward) and its rows enter the graph as an input, so the
# TP graph has no host boundary: one program launch per step instead of the
# piece before the PLE layer plus the piece after it.
_prelookup_check = {"left": 0, "checked": 0, "bad": 0, "armed": False}


def prelookup_rows(ple, inputs, valid_tokens: torch.Tensor, ids_fn) -> torch.Tensor:
    """``ple``'s raw host rows for ``inputs``: ids on the device by ``ids_fn``
    (the compiled ``_lookup_ids``), one blocking copy of the ids, the host
    gather, and the upload. Same ids, gather and layout as ``host_lookup``."""
    table = ple.ple_embedding.ngram_embedding
    assert isinstance(table, HostPLETable)
    ids = ids_fn(inputs, valid_tokens)
    host_ids = ids.cpu()
    check = envs.VLLM_TPU_PLE_HOST_PRELOOKUP_CHECK
    if check:
        _check_prelookup(ple, table, inputs, valid_tokens, host_ids, check)
    raw = table.lookup_host(host_ids)
    dp_size = ple.parallel.dp_group.world_size if ple.parallel.across_dp else 1
    return raw.reshape(
        dp_size, inputs.input_ids.shape[0], table.embedding_dim * ids.shape[-1]
    ).to(inputs.input_ids.device, non_blocking=raw.is_pinned())


def _check_prelookup(ple, table, inputs, valid_tokens, host_ids, check: str) -> None:
    """VLLM_TPU_PLE_HOST_PRELOOKUP_CHECK=N (``fault:N``, its positive control,
    shifts the reference's tokens by one): for the first N live steps compare
    the gathered rows with the eager in-graph-style lookup of the same inputs."""
    c = _prelookup_check
    if not c["armed"]:
        c["armed"] = True
        c["left"] = int(check.split(":")[-1] or 0)
    if c["left"] <= 0 or not bool((host_ids >= 0).any()):
        return
    c["left"] -= 1
    ref_inputs = inputs
    if check.startswith("fault"):
        ref_inputs = type(inputs)(
            torch.roll(inputs.input_ids, 1), inputs.query_start_loc, inputs.ngram_context
        )
    ref_ids = ple._lookup_ids(ref_inputs, valid_tokens).cpu()
    ref = table.lookup_host(ref_ids).clone()
    got = table.lookup_host(host_ids)
    c["checked"] += 1
    c["bad"] += int(not (torch.equal(ref_ids, host_ids) and torch.equal(ref, got)))
    if c["left"] == 0 or c["checked"] % 500 == 0:
        logger.warning(
            "PLE prelookup check (%s): %d of %d live steps differed",
            check,
            c["bad"],
            c["checked"],
        )
