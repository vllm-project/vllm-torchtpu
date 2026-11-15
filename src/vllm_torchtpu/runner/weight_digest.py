# SPDX-License-Identifier: Apache-2.0
"""Per-tensor digests of a loaded model, to compare two loads byte for byte.

Enabled by VLLM_TPU_WEIGHT_DIGEST_RANKS (comma-separated global ranks, or
"all"). Each listed rank logs one ``weight-digest`` line per parameter and
buffer and one ``weight-digest-summary`` line per model. Buffers are reported
apart from parameters because some are allocated uninitialized and differ
between runs.
"""

import hashlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _digest(host: torch.Tensor) -> str:
    raw = host.reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


# Host copies waiting for their digest at most.
_PENDING = 16


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _log_tensors(pool, rank: int, prefix: str, named) -> None:
    """Logs one digest line per tensor of ``named`` and a combined digest."""
    pending: deque = deque()
    lines: list[str] = []

    def emit(name, dtype, shape, digest) -> None:
        if not isinstance(digest, str):
            digest = digest.result()
        line = f"{prefix} {name} {dtype} {list(shape)} {digest}"
        lines.append(line)
        logger.info("weight-digest rank=%d %s", rank, line)

    # Device-to-host copies run on this thread, digests on the pool.
    for name, tensor in named:
        try:
            host = tensor.detach().contiguous().cpu()
        except Exception as e:  # noqa: BLE001 - a diagnostic never fails the load
            digest = f"unavailable:{type(e).__name__}"
        else:
            digest = pool.submit(_digest, host)
            del host
        pending.append((name, tensor.dtype, tuple(tensor.shape), digest))
        if len(pending) > _PENDING:
            emit(*pending.popleft())
    while pending:
        emit(*pending.popleft())
    combined = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()
    logger.info(
        "weight-digest-summary rank=%d %s count=%d combined=%s",
        rank,
        prefix,
        len(lines),
        combined,
    )


def maybe_log_weight_digests(models: dict[str, nn.Module | None]) -> None:
    """Logs the digest of every parameter and buffer of ``models`` on listed ranks."""
    ranks = envs.VLLM_TPU_WEIGHT_DIGEST_RANKS
    if not ranks:
        return
    rank = _rank()
    if ranks != "all" and rank not in {int(r) for r in ranks.split(",") if r.strip()}:
        return
    with ThreadPoolExecutor(8, thread_name_prefix="weight-digest") as pool:
        for label, model in models.items():
            if model is None:
                continue
            named = model.named_parameters(remove_duplicate=False)
            _log_tensors(pool, rank, f"{label} param", named)
            named = model.named_buffers(remove_duplicate=False)
            _log_tensors(pool, rank, f"{label} buffer", named)
