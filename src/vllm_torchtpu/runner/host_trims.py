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
"""Host-side trims for batch-1 decode.

At decode batch 1 the step is host-bound: per-step Python work sits on the
critical path between device programs. The trims below remove some of it.
None changes what the device computes.

**One runtime rule** turns them on: the engine decodes (it is not a pure KV
producer) and its ``max_num_seqs`` is 1 (:func:`rule`). The worker applies it
with :func:`configure` when the runner is built:

* the cached forward context (``runner.fast_forward_context``);
* the guard memo of the direct compiled-call dispatch
  (``vllm_torchtpu._qx_fastcall``; it only acts when
  ``tpu_compile.fast_compiled_call`` is on);
* the AOT guard skip of the MTP draft programs (:func:`maybe_skip_aot_guards`);
* the KV connector's stats polled every :data:`CONN_STATS_EVERY_ON` steps
  instead of every step;
* cached answers for process-constant predicates: the KV connector's transport
  checks and ``utils.get_dp_size``.

**Process-wide settings** are separate, because they change the interpreter
for every thread of the process. They come from
``additional_config["tpu_runner"]`` (``runner.runner_options``), both off by
default, and are a documented trade-off:

* ``switch_interval_us``: ``sys.setswitchinterval`` in microseconds instead of
  Python's 5 ms, so a background thread (output, connector or submit thread)
  cannot hold the GIL for long while the step loop waits on it;
* ``gc_freeze``: after :data:`GC_FREEZE_AFTER_STEPS` steps, ``gc.collect()``
  and ``gc.freeze()``, and the gen-0 threshold raised to
  :data:`GC_FREEZE_GEN0_THRESHOLD`, so cyclic GC runs rarely in the decode
  loop. Objects created later are collected less often.

The worker applies them from ``execute_model`` and the prefix-aware scheduler
from ``schedule()``, so they reach the engine-core process too when the
workers run in their own processes.
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

CONN_STATS_EVERY_ON = 16
"""KV connector stats are polled every this many decode steps under the rule."""

GC_FREEZE_AFTER_STEPS = 500
GC_FREEZE_GEN0_THRESHOLD = 50000

# Set by configure(); all off until a runner applies the rule.
ENABLED = False
"""Whether the batch-1 rule holds for this process's engine."""
AOT_SKIP_GUARDS = False
CONN_STATS_EVERY = 1
SWITCH_US = 0
GC_FREEZE = False
STEP_HOOK = False
_configured = False


def rule(vllm_config: Any) -> bool:
    """Whether the batch-1 host trims apply to this engine: it decodes (it is
    not a pure KV producer) and its ``max_num_seqs`` is 1."""
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if getattr(scheduler_config, "max_num_seqs", None) != 1:
        return False
    kv = getattr(vllm_config, "kv_transfer_config", None)
    if kv is None:
        return True
    return not (
        bool(getattr(kv, "is_kv_producer", False))
        and not bool(getattr(kv, "is_kv_consumer", False))
    )


def configure(vllm_config: Any) -> None:
    """Apply the batch-1 rule and the process-wide ``tpu_runner`` settings to
    this process. Called when the runner is built, before any step runs."""
    global ENABLED, AOT_SKIP_GUARDS, CONN_STATS_EVERY, SWITCH_US, GC_FREEZE
    global STEP_HOOK, _configured
    from vllm_torchtpu import _qx_fastcall, utils
    from vllm_torchtpu.runner import fast_forward_context, runner_options

    enabled = rule(vllm_config)
    options = runner_options.get(vllm_config)
    ENABLED = enabled
    AOT_SKIP_GUARDS = enabled
    CONN_STATS_EVERY = CONN_STATS_EVERY_ON if enabled else 1
    fast_forward_context.set_enabled(enabled)
    _qx_fastcall.configure_guard_memo(enabled)
    utils.cache_dp_size(enabled)
    SWITCH_US = options.switch_interval_us
    GC_FREEZE = options.gc_freeze
    STEP_HOOK = bool(SWITCH_US > 0 or GC_FREEZE)
    _hook.update(n=0, applied=False)
    _configured = True
    logger.info(
        "Batch-1 host trims: %s (max_num_seqs=%s); tpu_runner.switch_interval_us=%d "
        "gc_freeze=%s",
        "on" if enabled else "off",
        getattr(getattr(vllm_config, "scheduler_config", None), "max_num_seqs", None),
        SWITCH_US,
        GC_FREEZE,
    )


def on_scheduler_step(vllm_config: Any) -> None:
    """The engine core's per-step hook: configures this process from the
    engine's config on first use (the workers may run in other processes),
    then applies the process-wide settings."""
    if not _configured:
        configure(vllm_config)
    if STEP_HOOK:
        on_step()


_done: set[int] = set()


def _evaluate_guards(module: Any) -> bool:
    """vLLM's evaluate_guards for this module (True when unknown: keep guards)."""
    cfg = getattr(module, "compilation_config", None)
    if cfg is None:
        vcfg = getattr(module, "vllm_config", None)
        cfg = getattr(vcfg, "compilation_config", None)
    ds = getattr(cfg, "dynamic_shapes_config", None)
    if ds is None:
        return True
    return bool(getattr(ds, "evaluate_guards", True))


def maybe_skip_aot_guards(module: Any) -> bool:
    """After a successful call of ``module`` (the MTP draft step and fused-loop
    programs): once its freshly AOT-compiled function has run, skip that
    function's per-call Dynamo guard evaluation, exactly as vLLM already does
    for an AOT function it loads from cache when
    ``dynamic_shapes_config.evaluate_guards`` is False
    (``vllm/compilation/decorators.py``). The compiled callable that runs is
    the same object either way. Returns True when this call disabled it.
    Cheap after the first call per module."""
    if not AOT_SKIP_GUARDS:
        return False
    key = id(module)
    if key in _done:
        return False
    fn = getattr(module, "aot_compiled_fn", None)
    if fn is None:
        return False
    _done.add(key)
    if not getattr(fn, "_guard_check_enabled", False):
        return False
    if _evaluate_guards(module):
        logger.info(
            "Batch-1 host trims: %s keeps its AOT guards (evaluate_guards=True)",
            type(module).__name__,
        )
        return False
    disable = getattr(fn, "disable_guard_check", None)
    if disable is None:
        return False
    disable()
    logger.info(
        "Batch-1 host trims: per-call guard check off for %s "
        "(as vLLM's AOT cache-load path does with evaluate_guards=False)",
        type(module).__name__,
    )
    return True


def _reset_for_tests() -> None:
    _done.clear()


_DEFAULT_GC_THRESH = gc.get_threshold()
_hook = {"n": 0, "applied": False, "frozen": False}


def _apply() -> None:
    if SWITCH_US > 0:
        sys.setswitchinterval(SWITCH_US / 1e6)
    if GC_FREEZE and _hook["frozen"]:
        gc.set_threshold(
            GC_FREEZE_GEN0_THRESHOLD, _DEFAULT_GC_THRESH[1], _DEFAULT_GC_THRESH[2]
        )


def on_step() -> None:
    """Per-step hook of the process-wide settings; only acts when one is set."""
    _hook["n"] += 1
    if GC_FREEZE and not _hook["frozen"] and _hook["n"] >= GC_FREEZE_AFTER_STEPS:
        gc.collect()
        gc.freeze()
        _hook["frozen"] = True
        _hook["applied"] = False  # re-apply the threshold below
        logger.info(
            "tpu_runner.gc_freeze: froze %d objects after %d steps; gen-0 threshold %d",
            gc.get_freeze_count(),
            _hook["n"],
            GC_FREEZE_GEN0_THRESHOLD,
        )
    if not _hook["applied"]:
        _hook["applied"] = True
        _apply()
        if _hook["n"] < 3:
            logger.info(
                "tpu_runner process settings: switch_interval_us=%d gc_freeze=%s "
                "(pid %d)",
                SWITCH_US,
                GC_FREEZE,
                os.getpid(),
            )
