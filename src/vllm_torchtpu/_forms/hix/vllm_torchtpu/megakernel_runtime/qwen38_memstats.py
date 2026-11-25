# SPDX-License-Identifier: Apache-2.0
"""q38 HIX700 HBM-BUDGET (lead rule 2026-11-23T23:10:56Z, after T54d's device RESOURCE_EXHAUSTED): device memory per leg, logged by the
megakernel driver at bind_and_warmup (before the arm's steps) and with every step-mix line (during / at the end of the arm), in every arm and
with or without VLLM_TPU_MK_GDN_SIDECAR. Log-only: reads `memory_stats()` of the process's local JAX devices (the pattern of
`vllm_torchtpu.utils.hbm_usage_bytes`) and never raises.

  q38 MEMSTATS <where>: peak_bytes_in_use <B> (<GiB>) bytes_in_use <B> bytes_limit <B> (<GiB>) over <n> local devices (max / max / min)
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger("vllm." + __name__)
TAG = "q38 MEMSTATS"


def stats(devices: Any = None) -> dict | None:
    """{n, peak, in_use, limit} over the devices that report memory stats (peak / in_use: max; limit: min), or None."""
    try:
        if devices is None:
            import jax

            devices = jax.local_devices()
        rows = []
        for d in devices:
            try:
                m = d.memory_stats() or {}
            except Exception:  # noqa: BLE001
                m = {}
            if "bytes_limit" in m:
                rows.append(m)
        if not rows:
            return None
        return dict(n=len(rows), peak=max(int(m.get("peak_bytes_in_use", 0)) for m in rows),
                    in_use=max(int(m.get("bytes_in_use", 0)) for m in rows), limit=min(int(m["bytes_limit"]) for m in rows))
    except Exception:  # noqa: BLE001
        return None


def line(where: str, devices: Any = None) -> str:
    s = stats(devices)
    if s is None:
        return "%s %s: unavailable" % (TAG, where)
    g = 2.0 ** 30
    return "%s %s: peak_bytes_in_use %d (%.3f GiB) bytes_in_use %d bytes_limit %d (%.3f GiB) over %d local devices (max / max / min)" % (
        TAG, where, s["peak"], s["peak"] / g, s["in_use"], s["limit"], s["limit"] / g, s["n"])


def log(where: str) -> None:
    try:
        _log.info(line(where))
    except Exception:  # noqa: BLE001
        pass


__all__ = ["TAG", "line", "log", "stats"]
