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
"""Pin Pallas kernel operands to HBM so XLA does not stage them in VMEM.

A ``pl.BlockSpec(memory_space=pltpu.HBM)`` only types the ref *inside* the
kernel. XLA's memory-space assignment reads the operand's aval, so it is free
to prefetch the operand into VMEM with a ``copy-start``/``copy-done`` pair
just before the custom call. For a decode step that is pure added latency:
the served DSV4 decode step waits ~0.8 ms on such copies (e.g. the bf16
compressor weight ``[7168, 1024]`` copied whole into VMEM by every
``proj_and_save_state`` call, ~5 us exposed each, plus ~0.6 us per tiny
``s32[8]`` / ``bf16[8, 7168]`` operand). The kernel then DMAs the operand
VMEM->VMEM, so the copy buys nothing.

``pltpu.with_memory_space_constraint(x, pltpu.HBM)`` on a pallas_call input
sets the custom call's operand memory space, which stops the staging (see
``fp8_skinny._call_hbm_pinned``). This patch applies it centrally:

``TPU_PALLAS_HBM_PIN=1``: operands whose BlockSpec says HBM or ANY.
``TPU_PALLAS_HBM_PIN=2``: also blocked operands with no explicit memory space
(the Pallas pipeline DMAs their blocks from HBM itself).
``TPU_PALLAS_HBM_PIN_SKIP=<regex>``: kernel names (pallas_call ``name=``) to leave alone;
default ``_DEFAULT_SKIP``: the prefill compressor (pinning its hidden states
makes XLA add a [T, 7168] relayout copy, +27..43 us per twin mixed step) and
the fused EP MoE family + small collectives (the expert kernel ran +4.6%
slower pinned at TP8 T16, 3 boots per arm, measured). Set it to an empty
string to pin everything.
``TPU_PALLAS_HBM_PIN_ONLY=<regex>``: only kernels whose name matches.

Same kernels, same arithmetic: bit-identical outputs. Only traced operands are
constrained (the primitive cannot run eagerly). Default off.
"""

from __future__ import annotations

import functools
import logging
import os
import re

logger = logging.getLogger(__name__)

_MARK = "_torchtpu_hbm_pin_patch"
_DEFAULT_SKIP = (
    r"^(proj_and_save_state_runs$|fused_ep_moe_v2_|moe_v2_|oneshot_all_gather"
    r"|small_all_|kseam_)"
)


def _level() -> int:
    try:
        return int(os.environ.get("TPU_PALLAS_HBM_PIN", "0") or 0)
    except ValueError:
        return 0


def install_pallas_hbm_pin() -> None:
    level = _level()
    if level <= 0:
        return
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    orig = pl.pallas_call
    if getattr(orig, _MARK, False):
        return
    skip = os.environ.get("TPU_PALLAS_HBM_PIN_SKIP", _DEFAULT_SKIP)
    only = os.environ.get("TPU_PALLAS_HBM_PIN_ONLY", "")
    skip_re = re.compile(skip) if skip else None
    only_re = re.compile(only) if only else None
    hbm = pltpu.HBM
    pinnable = {hbm, pl.ANY}
    tpu_any = getattr(pltpu, "ANY", None)
    if tpu_any is not None:
        pinnable.add(tpu_any)
    no_space = {None}
    no_block_spec = getattr(pl, "no_block_spec", None)

    def _want(spec) -> bool:
        if spec is None or spec is no_block_spec:
            return level >= 2
        ms = getattr(spec, "memory_space", None)
        if ms in pinnable:
            return True
        if level >= 2 and ms in no_space:
            # Blocked operand with no explicit space: the pipeline DMAs it.
            return getattr(spec, "block_shape", None) is not None
        return False

    @functools.wraps(orig)
    def pallas_call(kernel, out_shape, *, grid_spec=None, in_specs=None, **kw):
        if in_specs is None:
            call = orig(kernel, out_shape, grid_spec=grid_spec, **kw)
        else:
            call = orig(kernel, out_shape, grid_spec=grid_spec, in_specs=in_specs, **kw)
        fn = kernel
        while isinstance(fn, functools.partial):
            fn = fn.func
        name = kw.get("name") or getattr(fn, "__name__", "") or ""
        if skip_re is not None and skip_re.search(name):
            return call
        if only_re is not None and not only_re.search(name):
            return call
        specs, nsp = in_specs, 0
        if grid_spec is not None:
            specs = getattr(grid_spec, "in_specs", None)
            nsp = int(getattr(grid_spec, "num_scalar_prefetch", 0) or 0)
        if not isinstance(specs, (list, tuple)):
            return call
        specs = list(specs)

        @functools.wraps(call)
        def wrapped(*args):
            if len(args) != nsp + len(specs):
                return call(*args)
            new = list(args)
            for i, spec in enumerate(specs):
                a = args[nsp + i]
                if isinstance(a, jax.core.Tracer) and _want(spec):
                    new[nsp + i] = pltpu.with_memory_space_constraint(a, hbm)
            return call(*new)

        return wrapped

    setattr(pallas_call, _MARK, True)
    pl.pallas_call = pallas_call
    logger.info(
        "TPU_PALLAS_HBM_PIN=%d: pinning Pallas operands to HBM (skip=%r only=%r)",
        level, skip, only,
    )
