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
"""Smallest compiling ``vmem_limit_bytes`` for a pallas_call (TPU_TIGHT_VMEM=1).

On v7x a pallas_call's VMEM limit costs time even when the kernel does not use
the VMEM: a limit of 56 MiB or more adds ~2.5 us per call, and limits above
32 MiB add up to ~2.5 us more to a kernel with large scratch (measured with a
trivial and a DMA-only kernel, same body, only the limit changed). Kernels that
ask for 48 MiB or 0.9 x capacity "to be safe" pay this on every call.

`pallas_call` is a drop-in for `pl.pallas_call`. With TPU_TIGHT_VMEM=1 and a
limit above the smallest candidate, the first call per (kernel name, argument
shapes, limit) compiles the kernel alone at each smaller candidate limit, in
increasing order, and keeps the first that compiles. Only the limit changes,
so the kernel body, tiling and results are unchanged.
"""

import dataclasses
import os
from typing import Any

import jax
from jax.experimental import pallas as pl

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MIB = 1 << 20
_CANDIDATES = tuple(
    int(v) * _MIB
    for v in os.environ.get("TPU_TIGHT_VMEM_CANDIDATES_MIB", "32,40,48,52").split(",")
    if v
)
_picked: dict[Any, int] = {}


def _enabled() -> bool:
    return os.environ.get("TPU_TIGHT_VMEM", "0") not in ("", "0", "false", "False")


def _aval(a):
    return jax.ShapeDtypeStruct(a.shape, a.dtype) if hasattr(a, "shape") else a


def _pick(kernel, kw, compiler_params, args) -> int:
    limit = compiler_params.vmem_limit_bytes
    avals = jax.tree.map(_aval, args)
    key = (
        kw.get("name"),
        getattr(kernel, "func", kernel).__qualname__,
        str(jax.tree.structure(args)),
        tuple(
            (tuple(a.shape), str(a.dtype)) if hasattr(a, "shape") else repr(a)
            for a in jax.tree.leaves(avals)
        ),
        limit,
    )
    if key in _picked:
        return _picked[key]
    chosen = limit
    for cand in _CANDIDATES:
        if cand >= limit:
            break
        cp = dataclasses.replace(compiler_params, vmem_limit_bytes=cand)
        try:
            jax.jit(lambda *a, cp=cp: pl.pallas_call(kernel, compiler_params=cp, **kw)(*a)).lower(
                *avals
            ).compile()
        except Exception:  # noqa: BLE001 - VMEM OOM at this limit: try the next one
            continue
        chosen = cand
        break
    _picked[key] = chosen
    logger.info(
        "TPU_TIGHT_VMEM=1: %s vmem_limit %d -> %d MiB",
        kw.get("name") or key[1],
        limit // _MIB,
        chosen // _MIB,
    )
    return chosen


def pallas_call(kernel, *, compiler_params=None, **kw):
    limit = getattr(compiler_params, "vmem_limit_bytes", None)
    if not _enabled() or limit is None or limit <= _CANDIDATES[0]:
        return pl.pallas_call(kernel, compiler_params=compiler_params, **kw)

    def run(*args):
        lim = _pick(kernel, kw, compiler_params, args)
        cp = dataclasses.replace(compiler_params, vmem_limit_bytes=lim)
        return pl.pallas_call(kernel, compiler_params=cp, **kw)(*args)

    return run
