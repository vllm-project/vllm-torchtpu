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
"""Flat, positional form of the Qwen3.8 segment step for runtime bridges.

`make_flat_segment_step` wraps `kernel.make_step` in a function of positional
arrays only — `(pos, hidden, attn, *weights, *states)` (`attn` is ignored by a
head segment, which has no attention input) — returning
`(residual, *new_states)`, with one `PartitionSpec` per input and output, so a
runtime can call it through an SPMD bridge. Weights are the per-layer operands
in `plan.w_names` order with per-layer tuples (routed experts) expanded.
"""

from __future__ import annotations

from typing import Any

from jax.sharding import PartitionSpec as P

from .. import kernel
from ..blocks.hc import RESIDUAL_OUT
from ..exec_spec import ExecSpec
from ..spec import ModelSpec

GDN_STATES = ("conv", "rec")
GDN_CKPTS = ("conv_ckpt", "rec_ckpt")


def flat_weight_layout(spec: ModelSpec, ex: ExecSpec) -> list[tuple[str, int]]:
    """`(name, count)` per plan weight; `count > 0` marks a per-layer tuple."""
    _, info = kernel.build(spec, ex)
    p, wsh = info["plan"], info["wshape"]
    return [
        (n, len(wsh[n]) if kernel.is_tupled_weight(wsh[n]) else 0) for n in p.w_names
    ]


def make_flat_segment_step(mesh: Any, spec: ModelSpec, ex: ExecSpec):
    """Return `(fn, in_specs, out_specs, n_weights)` for one segment bucket."""
    step, info = kernel.make_step(mesh, spec, ex)
    p, wsh = info["plan"], info["wshape"]
    layout = [
        (n, len(wsh[n]) if kernel.is_tupled_weight(wsh[n]) else 0) for n in p.w_names
    ]
    n_w = sum(max(c, 1) for _, c in layout)
    has_gdn = all(s in p.s_names for s in GDN_STATES)
    has_attn = bool(kernel.extra_inputs(spec, info["exec"]))
    states = GDN_STATES if has_gdn else ()
    outs = (RESIDUAL_OUT, *GDN_CKPTS) if has_gdn else (RESIDUAL_OUT,)

    def fn(pos, hidden, attn, *rest):
        flat, st = rest[:n_w], rest[n_w:]
        weights, i = {}, 0
        for name, count in layout:
            if count:
                weights[name] = tuple(flat[i : i + count])
                i += count
            else:
                weights[name] = flat[i]
                i += 1
        extra = (attn,) if has_attn else ()
        _, new = step(pos, hidden, weights, dict(zip(states, st)), *extra)
        return tuple(new[k] for k in outs)

    in_specs = (P(), P("tp"), P("tp")) + (P("tp"),) * (n_w + len(states))
    out_specs = (P("tp"),) * len(outs)
    return fn, in_specs, out_specs, n_w
