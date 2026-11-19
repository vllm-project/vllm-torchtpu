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
"""q38 L6-HOST lever ``envcache``: cache ``vllm_torchtpu.envs`` reads in the worker.

vLLM caches its own ``vllm.envs`` lookups in every worker after init
(``multiproc_executor.py`` calls ``enable_envs_cache()``), but nothing calls
``vllm_torchtpu.envs.enable_envs_cache()``, so every ``envs.VLLM_TPU_*`` read
on the per-step host path re-runs its parser over ``os.environ``. This lever
swaps the plugin env module's ``__getattr__`` for a memoising copy while the lever is on and back to the raw function while it is off
(clearing the cache), so it can follow the L6 A/B windows. IX's track-3 trims
carried the same idea for its connector's env predicates (``VLLM_TPU_T3_ENV_CACHE``,
Qwen3.5 lane, credited).

Values are unchanged unless the process writes ``os.environ`` after init and
expects a plugin env read to see the write. Every ``os.environ`` writer in
``src`` runs during process or worker init (audited for L6), and
``VLLM_TPU_L6_ENVS_CACHE_CHECK=N`` re-reads every cached name through the raw
parser every N cached steps, counts mismatches in the L6AB window line and
drops the cache on any mismatch.

``VLLM_TPU_L6_ENVS_CENSUS=N`` (0 = off) counts every plugin env read in the
worker and logs, every N steps:
``q38 L6ENVCENSUS steps=<N> reads=<total> reads_per_step=<x> top=<name:count,...>``
"""

from __future__ import annotations

import os

from vllm_torchtpu import envs as _envs
from vllm_torchtpu.logger import init_logger

from . import l6_ab

logger = init_logger(__name__)

_RAW = _envs.__dict__["__getattr__"]
_cache: dict[str, object] = {}


def _CACHED(name: str):
    try:
        return _cache[name]
    except KeyError:
        value = _RAW(name)
        _cache[name] = value
        return value


def _env_int(name: str) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else 0


CENSUS = _env_int("VLLM_TPU_L6_ENVS_CENSUS")
CHECK = _env_int("VLLM_TPU_L6_ENVS_CACHE_CHECK")
_census: dict = {"steps": 0, "reads": {}, "inner": _RAW}
_state: dict = {"cached": False, "applied": False, "check_steps": 0, "checked": 0, "mismatch": 0}


def check_now() -> int:
    """Re-read every cached name through the raw parser; on any difference
    drop the whole cache. Returns the number of mismatching names."""
    bad = [k for k, v in list(_cache.items()) if _RAW(k) != v]
    _state["checked"] += len(_cache)
    if bad:
        _state["mismatch"] += len(bad)
        logger.warning("q38 L6 envcache CHECK mismatch on %s; cache cleared", ",".join(sorted(bad)[:8]))
        _cache.clear()
    return len(bad)


def _counting(name: str):
    reads = _census["reads"]
    reads[name] = reads.get(name, 0) + 1
    return _census["inner"](name)


def _install(inner) -> None:
    """Make `inner` (raw or cached) the module's attribute hook, behind the census counter if on."""
    _census["inner"] = inner
    _envs.__getattr__ = _counting if CENSUS > 0 else inner


def apply(cached: bool) -> None:
    """Switch the plugin env module between cached and raw lookups."""
    if _state["applied"] and _state["cached"] == cached:
        return
    if not cached:
        _cache.clear()
    _install(_CACHED if cached else _RAW)
    if cached != _state["cached"] or not _state["applied"]:
        logger.info("q38 L6 envcache: plugin env reads %s (pid %d)",
                    "cached" if cached else "uncached", os.getpid())
    _state["cached"] = cached
    _state["applied"] = True


def on_step() -> None:
    """Per-step hook from the worker's execute_model, after l6_ab.begin_step()."""
    if l6_ab.ACTIVE or _state["applied"]:
        want = l6_ab.on("envcache")
        apply(want)
        if want:
            l6_ab.note("envcache", "steps")
            if CHECK > 0 and l6_ab.check_allowed():
                _state["check_steps"] += 1
                if _state["check_steps"] >= CHECK:
                    _state["check_steps"] = 0
                    l6_ab.note("envcache", "checked", len(_cache))
                    l6_ab.note("envcache", "mismatch", check_now())
    if CENSUS > 0:
        if not _state["applied"]:
            apply(False)
        # Reads between two calls belong to the step that started at the first
        # call, so close a window of CENSUS complete steps before counting this one.
        if _census["steps"] >= CENSUS:
            reads = _census["reads"]
            total = sum(reads.values())
            top = sorted(reads.items(), key=lambda kv: -kv[1])[:12]
            logger.info(
                "q38 L6ENVCENSUS steps=%d reads=%d reads_per_step=%.1f top=%s",
                _census["steps"], total, total / _census["steps"],
                ",".join(f"{k}:{v}" for k, v in top) or "-",
            )
            _census["steps"] = 0
            _census["reads"] = {}
        _census["steps"] += 1


def restore() -> None:
    """Put the module's original hook back (tests)."""
    _cache.clear()
    _envs.__getattr__ = _RAW
    _census.update(steps=0, reads={}, inner=_RAW)
    _state.update(cached=False, applied=False, check_steps=0, checked=0, mismatch=0)
