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
"""One ``pallas_call`` per decode step, inside a jitted loop.

Asks whether XLA keeps expert weights in VMEM across separate calls:

* ``percall_vmem``: weights are VMEM-space ``pallas_call`` inputs; XLA
  chooses where they live between calls.
* ``percall_constrained``: as above, plus ``pltpu.with_memory_space_constraint``
  on the weights.
* ``percall_streamed``: weights stay in HBM and each call DMAs the experts it
  uses (the per-step reload that serving does today).

``weight_copies_in_loop`` reads the compiled HLO for copies of the weight
buffers inside the while body, which is what decides residency; time alone
cannot.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .kernel import F8, _quant, _rmsnorm, expert_ffn, expert_ffn_q, quant_scratch

PERCALL_ARMS = ("percall_vmem", "percall_constrained", "percall_streamed")


def step_schedule(sched, steps: int, experts: int):
    """Per-step ``count [N, 1]`` and padded expert list ``[N, E]`` from a Schedule."""
    count = np.zeros((steps, 1), np.int32)
    order = np.zeros((steps, experts), np.int32)
    for s, e in zip(np.asarray(sched.step), np.asarray(sched.expert)):
        order[s, count[s, 0]] = e
        count[s, 0] += 1
    return jnp.asarray(count), jnp.asarray(order)


def _quant_once(x, q):
    if q is None:
        return None
    xq, sx = _quant(x[...])
    q[0][...] = xq
    q[1][...] = jnp.broadcast_to(sx, q[1].shape)
    return q


def _ffn(x, q, *w):
    if q is None:
        return expert_ffn(x[...], *w)
    return expert_ffn_q(q[0][...], q[1][:, :1], *w)


def _vmem_step_kernel(cnt, order, x, gate, wg, wu, wd, sg, su, sd, out, acc, *q):
    acc[...] = jnp.zeros_like(acc)
    q = _quant_once(x, q or None)

    def body(i, c):
        e = order[i]
        y = _ffn(x, q, wg[e], wu[e], wd[e], sg[e], su[e], sd[e])
        acc[...] += y * gate[e][:, :1]
        return c

    jax.lax.fori_loop(0, cnt[0], body, 0)
    out[...] = _rmsnorm(acc[...])


def _streamed_step_kernel(cnt, order, x, gate, sg, su, sd, *refs):
    hbm, (out, acc, wg_buf, wu_buf, wd_buf, sem), q = refs[:3], refs[3:9], refs[9:]
    bufs = (wg_buf, wu_buf, wd_buf)
    n = cnt[0]
    acc[...] = jnp.zeros_like(acc)
    q = _quant_once(x, q or None)

    def copies(i, slot):
        e = order[i]
        return [
            pltpu.make_async_copy(src.at[e], dst.at[slot], sem.at[k, slot])
            for k, (src, dst) in enumerate(zip(hbm, bufs))
        ]

    @pl.when(n > 0)
    def _():
        for c in copies(0, 0):
            c.start()

    def body(i, carry):
        slot = i % 2

        @pl.when(i + 1 < n)
        def _():
            for c in copies(i + 1, 1 - slot):
                c.start()

        for c in copies(i, slot):
            c.wait()
        e = order[i]
        w = (wg_buf[slot], wu_buf[slot], wd_buf[slot], sg[e], su[e], sd[e])
        acc[...] += _ffn(x, q, *w) * gate[e][:, :1]
        return carry

    jax.lax.fori_loop(0, n, body, 0)
    out[...] = _rmsnorm(acc[...])


def build_percall(
    arm,
    *,
    experts,
    hidden,
    inter,
    rows,
    vmem_limit_bytes,
    interpret=False,
    hoist=True,
):
    """Jitted ``f(count, order, x0, gate, weights) -> [rows, hidden] f32``."""
    if arm not in PERCALL_ARMS:
        raise ValueError(f"arm must be one of {PERCALL_ARMS}, got {arm!r}")
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    scratch = [pltpu.VMEM((rows, hidden), jnp.float32)]
    if arm == "percall_streamed":
        kernel, in_specs = _streamed_step_kernel, [smem] * 2 + [vmem] * 5 + [hbm] * 3
        scratch += [
            pltpu.VMEM((2, hidden, inter), F8),
            pltpu.VMEM((2, hidden, inter), F8),
            pltpu.VMEM((2, inter, hidden), F8),
            pltpu.SemaphoreType.DMA((3, 2)),
        ]
    else:
        kernel, in_specs = _vmem_step_kernel, [smem] * 2 + [vmem] * 8
    if hoist:
        scratch += quant_scratch(rows, hidden)
    extra = {"interpret": pltpu.InterpretParams()} if interpret else {}
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        in_specs=in_specs,
        out_specs=vmem,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(vmem_limit_bytes)),
        **extra,
    )

    @jax.jit
    def run(count, order, x0, gate, weights):
        wg, wu, wd, sg, su, sd = weights
        # The CPU interpreter rejects the constraint; it only affects placement.
        if arm == "percall_constrained" and not interpret:
            wg, wu, wd = (
                pltpu.with_memory_space_constraint(w, pltpu.VMEM) for w in (wg, wu, wd)
            )

        def step(s, x):
            args = (count[s], order[s], x, gate[s])
            if arm == "percall_streamed":
                return call(*args, sg, su, sd, wg, wu, wd)
            return call(*args, wg, wu, wd, sg, su, sd)

        return jax.lax.fori_loop(0, count.shape[0], step, x0.astype(jnp.float32))

    return run


def weight_copies_in_loop(compiled_text: str, experts: int, hidden: int, inter: int):
    """Count copy ops on expert-weight shapes inside while bodies of compiled HLO."""
    shapes = (f"[{experts},{hidden},{inter}]", f"[{experts},{inter},{hidden}]")
    bodies, cur = {}, None
    for line in compiled_text.splitlines():
        m = re.match(r"^\s*(?:%)?([\w.\-]+)\s.*\{\s*$", line)
        if m and ("body" in m.group(1) or "while" in m.group(1)):
            cur = m.group(1)
            bodies[cur] = []
        elif line.strip() == "}":
            cur = None
        elif cur is not None:
            bodies[cur].append(line)
    hits = []
    for name, lines in bodies.items():
        for ln in lines:
            if re.search(r"\b(copy|copy-start|copy-done)\(", ln) and any(
                s in ln for s in shapes
            ):
                hits.append(f"{name}: {ln.strip()[:200]}")
    vmem_weights = [
        ln.strip()[:200]
        for ln in compiled_text.splitlines()
        if any(s + "{" in ln and "S(1)" in ln for s in shapes)
    ]
    return {"copies_in_loop": hits, "weights_in_vmem_annotations": vmem_weights[:6]}
