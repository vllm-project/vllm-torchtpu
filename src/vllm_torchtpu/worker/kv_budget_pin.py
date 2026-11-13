# SPDX-License-Identifier: Apache-2.0
"""Pin the KV-cache byte budget across restarts so compile caches stay valid.

The backbone compile hash (``compute_tpu_compilation_hash``) includes the KV
block count, because TPU AOT graphs embed the KV tensor extent as a literal.
The block count comes from an HBM measurement after ``profile_run`` that moves
by a fraction of a percent from one start to the next. Every start then hashes
to a new compile-cache directory, so a restored cache is never read and every
start compiles from scratch.

This module records the budget a start used, keyed by the compile hash taken
before the KV size is known, under the vLLM cache root (the directory a
persistent compile cache saves and restores). A later start with the same key
reuses the recorded budget when it still fits and is within the tolerance of
what this start measured, so it gets the same block count, the same compile
hash, and the artifacts the earlier start saved.
"""

import hashlib
import json
import os
from pathlib import Path

from vllm.config import VllmConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_SUBDIR = "kv_budget_pin"


def _pin_dir() -> Path:
    import vllm.envs as vllm_envs

    return Path(vllm_envs.VLLM_CACHE_ROOT) / _SUBDIR


def pin_key(vllm_config: VllmConfig) -> str:
    """The compile hash without the KV size: everything else that selects
    the compiled programs, so a pinned budget is reused only by a start that
    would build the same programs apart from the KV extent."""
    from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

    model = str(vllm_config.model_config.model)
    model_tag = hashlib.sha256(model.encode()).hexdigest()[:8]
    return f"{compute_tpu_compilation_hash(vllm_config)}-{model_tag}"


def choose_budget(measured: int, recorded: int | None, tolerance: float) -> int:
    """The budget to use: `recorded` if it fits and is within `tolerance` of
    `measured`, else `measured`."""
    if recorded is None or recorded <= 0 or tolerance <= 0:
        return measured
    if recorded <= measured and recorded >= measured * (1.0 - tolerance):
        return recorded
    return measured


def pin_kv_budget(vllm_config: VllmConfig, measured: int, write: bool) -> int:
    """Return the KV byte budget for this start and record it.

    `measured` is this rank's budget before the cross-rank minimum; pinning
    it there keeps ranks consistent, since the minimum is taken afterwards.
    `write` records the chosen budget (one rank per host is enough).
    Disabled with VLLM_TPU_KV_BUDGET_PIN_TOLERANCE=0.
    """
    tolerance = envs.VLLM_TPU_KV_BUDGET_PIN_TOLERANCE
    if tolerance <= 0 or measured <= 0:
        return measured
    try:
        key = pin_key(vllm_config)
        path = _pin_dir() / f"{key}.json"
        recorded = None
        if path.is_file():
            recorded = int(json.loads(path.read_text()).get("bytes", 0)) or None
    except Exception as e:  # noqa: BLE001 - pinning is an optimization only
        logger.warning("KV budget pin unavailable: %s", e)
        return measured

    chosen = choose_budget(measured, recorded, tolerance)
    logger.info(
        "KV budget pin %s: measured=%d recorded=%s -> using %d (%s)",
        key,
        measured,
        recorded,
        chosen,
        "pinned" if recorded is not None and chosen == recorded else "new",
    )
    if write and chosen != recorded:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"bytes": chosen}))
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not record KV budget pin %s: %s", path, e)
    return chosen
