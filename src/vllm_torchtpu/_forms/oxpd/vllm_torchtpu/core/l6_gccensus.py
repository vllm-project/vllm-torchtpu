# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""q38 L6 gen-2 heap type census (log-only; ``VLLM_TPU_L6_GC_CENSUS_TYPES``; worker process; default off).

Why. T98o (j-ff638d81, M) showed gen-2 collections of 13-19 s on every decode rank, also at engine start with
no long requests, and served PD decode holds only 64-token prompt tails, so what fills the gen-2 heap is
unknown. This census names it.

What. With the flag set to 1, once the worker is warm (``runner.warmup_done``) the module installs a
``gc.callbacks`` hook that only records the start time of each collection and, on a gen-2 collection that
took at least ``VLLM_TPU_L6_GC_CENSUS_MS`` ms (default 1000), marks a census as pending; nothing heavy runs
inside the callback. At the start of the next worker step (``on_step``) a pending census, and one census right
after warm-up, logs the total object count of ``gc.get_objects(generation=2)`` and the top 15 types by count:

``q38 L6 GCCENSUS-TYPES rank=<r> pid=<p> event=<warm|gen2> dur_ms=<d> total=<n> top=<module.qualname:count,...>``

At most ``VLLM_TPU_L6_GC_CENSUS_MAX`` censuses (default 4, warm one included) run per process, so the cost
(one walk of the gen-2 list per census) is bounded. Changes no token, no program; compile-ignored.
"""

from __future__ import annotations

import collections
import gc
import os
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_TPU_L6_GC_CENSUS_TYPES"
MS_ENV = "VLLM_TPU_L6_GC_CENSUS_MS"
MAX_ENV = "VLLM_TPU_L6_GC_CENSUS_MAX"
TOP = 15


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    v = int(raw)
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


ON = False
MS = 1000
MAX = 4
_state: dict = {}


def _resolve() -> None:
    global ON, MS, MAX
    ON = os.environ.get(ENV, "").strip().lower() in ("1", "true", "on")
    MS = _env_int(MS_ENV, 1000)
    MAX = _env_int(MAX_ENV, 4)
    _state.clear()
    _state.update(installed=False, t0=None, pending=None, done=0, warm_done=False)


_resolve()


def _callback(phase: str, info: dict) -> None:
    # Only timestamps and a flag here: no allocation-heavy work inside a collection.
    if phase == "start":
        _state["t0"] = time.perf_counter()
        return
    t0 = _state.get("t0")
    _state["t0"] = None
    if t0 is None or info.get("generation") != 2:
        return
    dur_ms = (time.perf_counter() - t0) * 1e3
    if dur_ms >= MS and _state["pending"] is None:
        _state["pending"] = dur_ms


def _rank() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return -1


def census(event: str, dur_ms: float) -> str:
    objs = gc.get_objects(generation=2)
    # Count by type object at C speed (map(type, ...)), then name only the distinct types: a heap whose gen-2
    # collection takes seconds holds ~10^8 objects, and a per-object f-string would add tens of seconds.
    cnt = collections.Counter()
    for t, v in collections.Counter(map(type, objs)).items():
        cnt[f"{t.__module__}.{t.__qualname__}"] += v
    total = len(objs)
    del objs
    top = ",".join(f"{k}:{v}" for k, v in cnt.most_common(TOP))
    line = (f"q38 L6 GCCENSUS-TYPES rank={_rank()} pid={os.getpid()} event={event} dur_ms={dur_ms:.0f} "
            f"total={total} top={top}")
    logger.info("%s", line)
    _state["done"] += 1
    return line


def on_step(runner) -> None:
    """Install after warm-up (one census then); run a pending gen-2 census; bounded by MAX per process."""
    if not getattr(runner, "warmup_done", False):
        return
    if not _state["installed"]:
        gc.callbacks.append(_callback)
        _state["installed"] = True
    if _state["done"] >= MAX:
        return
    if not _state["warm_done"]:
        _state["warm_done"] = True
        census("warm", 0.0)
        return
    if _state["pending"] is not None:
        dur = _state["pending"]
        _state["pending"] = None
        census("gen2", dur)
