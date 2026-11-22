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
"""Helpers shared by the batched RPA tests."""

import jax


def _sub_jaxprs(eqn):
    """The jaxprs in an equation's parameters (jit and cond bodies, loop
    bodies, a kernel's body)."""
    for value in eqn.params.values():
        for sub in value if isinstance(value, (list, tuple)) else [value]:
            sub = getattr(sub, "jaxpr", sub)
            if hasattr(sub, "eqns"):
                yield sub


def _rpa_kernels(fn, *args) -> list:
    """The pallas_call equations of the RPA kernels (not the schedule
    kernels) `fn(*args)` traces, in trace order."""
    kernels = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                if eqn.params["name"].startswith("RPA"):
                    kernels.append(eqn)
                continue
            for sub in _sub_jaxprs(eqn):
                walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return kernels


def rpa_kernel_names(fn, *args) -> list[str]:
    """The names of the RPA kernels (not the schedule kernels) `fn(*args)`
    traces, in trace order. A name carries the kernel's batch, bq and bkv
    tiles as "-b<batch>-q<bq>-k<bkv>", its last field unless the kernel has a
    sliding window ("-sw<window>" follows)."""
    return [eqn.params["name"] for eqn in _rpa_kernels(fn, *args)]


def rpa_kernel_dots(fn, *args, prefix: str) -> list:
    """The matmuls (dot_general equations) in the bodies of the RPA kernels
    `fn(*args)` traces whose names start with `prefix` (e.g. "RPAd" for the
    DECODE kernel)."""
    dots = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "dot_general":
                dots.append(eqn)
            for sub in _sub_jaxprs(eqn):
                walk(sub)

    for kernel in _rpa_kernels(fn, *args):
        if kernel.params["name"].startswith(prefix):
            walk(kernel.params["jaxpr"])
    return dots
