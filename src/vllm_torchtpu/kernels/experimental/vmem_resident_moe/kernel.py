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
"""VMEM-resident vs HBM-streamed MoE expert weights over N decode steps in one call.

Both arms run the same per-item body over a flattened (step, expert) work list.
The only difference is where the expert weights are read from:

* ``resident``: every local expert is a whole-array VMEM input, copied in once
  and reused by every step.
* ``streamed``: experts stay in HBM and each work item DMAs its expert into a
  2-slot VMEM ring, prefetching the next item.

Step ``s + 1`` consumes step ``s``'s RMS-normed output, so steps are serially
dependent as in decode. VMEM is not preserved across ``pallas_call``s, so the
loop over steps lives inside the kernel.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F8 = jnp.float8_e4m3fn
F8_MAX = float(jnp.finfo(F8).max)
EPS = 1e-6
ARMS = ("resident", "streamed")


@dataclasses.dataclass(frozen=True)
class Schedule:
    """Flattened (step, expert) work list; ``first``/``last`` mark step edges."""

    step: jax.Array
    expert: jax.Array
    first: jax.Array
    last: jax.Array

    @property
    def items(self) -> int:
        return int(self.step.shape[0])

    def arrays(self):
        return self.step, self.expert, self.first, self.last


def make_schedule(assign: np.ndarray) -> Schedule:
    """Work list from per-step token->local-expert assignments ``[N, B]``."""
    steps, experts = [], []
    for s in range(assign.shape[0]):
        for e in sorted(set(assign[s].tolist())):
            steps.append(s)
            experts.append(e)
    step = np.asarray(steps, np.int32)
    edge = step[1:] != step[:-1]
    first = np.r_[True, edge].astype(np.int32)
    last = np.r_[edge, True].astype(np.int32)
    return Schedule(*(jnp.asarray(a) for a in (step, experts, first, last)))


def _quant(x):
    amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / F8_MAX
    return (x / scale).astype(F8), scale


def expert_ffn(x, wg, wu, wd, sg, su, sd):
    """SwiGLU expert with per-token dynamic fp8 activations, fp32 accumulate."""
    return expert_ffn_q(*_quant(x), wg, wu, wd, sg, su, sd)


def expert_ffn_q(xq, sx, wg, wu, wd, sg, su, sd):
    """``expert_ffn`` on an input already quantized by ``_quant``."""
    g = jnp.dot(xq, wg, preferred_element_type=jnp.float32) * sx * sg
    u = jnp.dot(xq, wu, preferred_element_type=jnp.float32) * sx * su
    hq, sh = _quant(g * jax.nn.sigmoid(g) * u)
    return jnp.dot(hq, wd, preferred_element_type=jnp.float32) * sh * sd


def _rmsnorm(a):
    return a * jax.lax.rsqrt(jnp.mean(a * a, axis=-1, keepdims=True) + EPS)


def _item(t, sched, gate_ref, x_scr, acc_scr, weights, q_scr):
    step_of, expert_of, first_of, last_of = sched

    @pl.when(first_of[t] == 1)
    def _():
        acc_scr[...] = jnp.zeros_like(acc_scr)
        if q_scr is not None:  # quantize the step input once, not per expert
            xq, sx = _quant(x_scr[...])
            q_scr[0][...] = xq
            q_scr[1][...] = jnp.broadcast_to(sx, q_scr[1].shape)

    if q_scr is None:
        y = expert_ffn(x_scr[...], *weights)
    else:
        y = expert_ffn_q(q_scr[0][...], q_scr[1][:, :1], *weights)
    # gate_ref is lane-replicated [N, E, B, 128]; column 0 is the routing weight.
    acc_scr[...] += y * gate_ref[step_of[t], expert_of[t]][:, :1]

    @pl.when(last_of[t] == 1)
    def _():
        x_scr[...] = _rmsnorm(acc_scr[...])


def _resident_kernel(*refs, items, hoist):
    q_scr = tuple(refs[-2:]) if hoist else None
    refs = refs[:-2] if hoist else refs
    sched, (x0, gate, wg, wu, wd, sg, su, sd) = refs[:4], refs[4:12]
    out, x_scr, acc_scr = refs[12:]
    x_scr[...] = x0[...].astype(jnp.float32)

    def body(t, carry):
        e = sched[1][t]
        w = (wg[e], wu[e], wd[e], sg[e], su[e], sd[e])
        _item(t, sched, gate, x_scr, acc_scr, w, q_scr)
        return carry

    jax.lax.fori_loop(0, items, body, 0)
    out[...] = x_scr[...]


def _streamed_kernel(*refs, items, hoist):
    q_scr = tuple(refs[-2:]) if hoist else None
    refs = refs[:-2] if hoist else refs
    sched, (x0, gate, sg, su, sd), hbm = refs[:4], refs[4:9], refs[9:12]
    out, x_scr, acc_scr, wg_buf, wu_buf, wd_buf, sem = refs[12:]
    bufs = (wg_buf, wu_buf, wd_buf)
    x_scr[...] = x0[...].astype(jnp.float32)

    def copies(t, slot):
        e = sched[1][t]
        return [
            pltpu.make_async_copy(src.at[e], dst.at[slot], sem.at[i, slot])
            for i, (src, dst) in enumerate(zip(hbm, bufs))
        ]

    for c in copies(0, 0):
        c.start()

    def body(t, carry):
        slot = t % 2

        @pl.when(t + 1 < items)
        def _():
            for c in copies(t + 1, 1 - slot):
                c.start()

        for c in copies(t, slot):
            c.wait()
        e = sched[1][t]
        w = (wg_buf[slot], wu_buf[slot], wd_buf[slot], sg[e], su[e], sd[e])
        _item(t, sched, gate, x_scr, acc_scr, w, q_scr)
        return carry

    jax.lax.fori_loop(0, items, body, 0)
    out[...] = x_scr[...]


def quant_scratch(rows, hidden):
    """VMEM scratch for the once-per-step quantized input and its row scales."""
    return [pltpu.VMEM((rows, hidden), F8), pltpu.VMEM((rows, 128), jnp.float32)]


def build(
    arm,
    *,
    experts,
    hidden,
    inter,
    rows,
    items,
    vmem_limit_bytes,
    interpret=False,
    hoist=True,
):
    """Jitted ``f(schedule, x0, gate, weights) -> [rows, hidden] f32`` for one arm."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    scratch = [pltpu.VMEM((rows, hidden), jnp.float32)] * 2
    if arm == "resident":
        kernel, in_specs = _resident_kernel, [smem] * 4 + [vmem] * 8
    else:
        kernel, in_specs = _streamed_kernel, [smem] * 4 + [vmem] * 5 + [hbm] * 3
        scratch += [
            pltpu.VMEM((2, hidden, inter), F8),
            pltpu.VMEM((2, hidden, inter), F8),
            pltpu.VMEM((2, inter, hidden), F8),
            pltpu.SemaphoreType.DMA((3, 2)),
        ]
    if hoist:
        scratch += quant_scratch(rows, hidden)
    extra = {"interpret": pltpu.InterpretParams()} if interpret else {}
    call = pl.pallas_call(
        functools.partial(kernel, items=items, hoist=hoist),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), jnp.float32),
        in_specs=in_specs,
        out_specs=vmem,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=int(vmem_limit_bytes)),
        **extra,
    )

    @jax.jit
    def jitted(sched, x0, gate, weights):
        wg, wu, wd, sg, su, sd = weights
        if arm == "resident":
            return call(*sched, x0, gate, wg, wu, wd, sg, su, sd)
        return call(*sched, x0, gate, sg, su, sd, wg, wu, wd)

    return lambda sched, x0, gate, weights: jitted(sched.arrays(), x0, gate, weights)


def reference(sched, x0, gate, weights):
    """Plain-JAX oracle with the same quantization; loops in Python."""
    wg, wu, wd, sg, su, sd = weights
    step, expert, first, last = (np.asarray(a) for a in sched.arrays())
    x, acc = x0.astype(jnp.float32), None
    for t in range(len(step)):
        s, e = int(step[t]), int(expert[t])
        if first[t]:
            acc = jnp.zeros_like(x)
        y = expert_ffn(x, wg[e], wu[e], wd[e], sg[e], su[e], sd[e])
        acc = acc + y * gate[s, e][:, :1]
        if last[t]:
            x = _rmsnorm(acc)
    return x


def make_inputs(key, *, experts, hidden, inter, tokens, rows, steps, seed=0, top_k=1):
    """Synthetic fp8 experts; each real token routes to top_k experts per step."""
    k = jax.random.split(key, 4)

    def weight(kk, shape, fan_in):
        return (
            jax.random.normal(kk, shape, jnp.float32) * (8 / np.sqrt(fan_in))
        ).astype(F8)

    weights = (
        weight(k[0], (experts, hidden, inter), hidden),
        weight(k[1], (experts, hidden, inter), hidden),
        weight(k[2], (experts, inter, hidden), inter),
        jnp.full((experts, 1, inter), 1 / 8, jnp.float32),
        jnp.full((experts, 1, inter), 1 / 8, jnp.float32),
        jnp.full((experts, 1, hidden), 1 / 8, jnp.float32),
    )
    x0 = jax.random.normal(k[3], (rows, hidden), jnp.float32).astype(jnp.bfloat16)
    rng = np.random.default_rng(seed)
    if top_k == 1:
        assign = rng.integers(0, experts, size=(steps, tokens))[..., None]
    else:
        assign = rng.random((steps, tokens, experts)).argsort(-1)[..., :top_k]
    gval = rng.uniform(0.5, 1.0, size=assign.shape).astype(np.float32)
    gate = np.zeros((steps, experts, rows, 128), np.float32)
    for s in range(steps):
        for b in range(tokens):
            for j in range(assign.shape[-1]):
                gate[s, assign[s, b, j], b, :] = gval[s, b, j]
    return make_schedule(assign.reshape(steps, -1)), x0, jnp.asarray(gate), weights
