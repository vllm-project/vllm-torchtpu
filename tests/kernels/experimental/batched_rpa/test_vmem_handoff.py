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

"""Standalone test: hand a VMEM buffer with an in-flight DMA across Pallas calls.

Kernel A starts a DMA HBM->VMEM and returns (VMEM buffer, DMA semaphore)
without waiting. A later kernel waits on the semaphore and reads the buffer.
In between, unrelated XLA work runs. This checks whether XLA keeps the VMEM
buffer in place (no copies) under the default and explicit-prefetch modes,
with and without the gemax-style `xla_prefetch_start/done` tags.
"""

from __future__ import annotations

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu

_SHAPE = (4, 1024, 1024)  # f32: 16 MiB.
_EXPLICIT_FLAGS = {"xla_tpu_enable_explicit_prefetch_scheduler": "true"}


def _tag(**kw):
    return xla_metadata.set_xla_metadata(**kw) if kw else contextlib.nullcontext()


def _prefetch_start(src, *, tagged, hint):
    """Starts HBM->VMEM DMAs for every slot and returns without waiting."""

    def kernel(src_hbm, dst_vmem, sem):
        for i in range(_SHAPE[0]):
            pltpu.make_async_copy(src_hbm.at[i], dst_vmem.at[i], sem).start()

    attrs = {"xla_prefetch_start": True} if tagged else {}
    if hint:
        # The latency-hiding scheduler runs bottom-up: force_delay moves the start
        # as early as possible in program order (gemax does the same).
        attrs["scheduler_hint"] = "force_delay"
    with _tag(**attrs):
        return pl.pallas_call(
            kernel,
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            ],
            out_shape=(
                pltpu.VMEM(_SHAPE, src.dtype),
                pltpu.SemaphoreType.DMA,
            ),
            compiler_params=pltpu.CompilerParams(
                has_side_effects=pltpu.SideEffectType.DATAFLOW_SIDE_EFFECTING
            ),
            name="handoff_start",
        )(src)


def _prefetch_done(src, buf, sem, *, tagged, hint):
    """gemax-style done: waits on all DMAs and aliases the buffer through."""

    def kernel(src_hbm, buf_in, sem_ref, buf_out):
        del buf_out
        # Only the byte count matters for the wait; one full-buffer descriptor
        # covers all per-slot DMAs signalled on this semaphore.
        pltpu.make_async_copy(src_hbm, buf_in, sem_ref).wait()

    attrs = {"xla_prefetch_done": True} if tagged else {}
    if hint:
        attrs["scheduler_hint"] = "force_early"
    with _tag(**attrs):
        return pl.pallas_call(
            kernel,
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            out_shape=pltpu.VMEM(_SHAPE, src.dtype),
            input_output_aliases={1: 0},
            name="handoff_done",
        )(src, buf, sem)


def _consume(buf, *, vmem_limit_bytes=None):
    """Reads the VMEM buffer (already landed) and writes an HBM result."""

    def kernel(buf_vmem, out_ref):
        out_ref[...] = buf_vmem[...] * 2.0 + 1.0

    return pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct(_SHAPE, buf.dtype),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        name="handoff_consume",
    )(buf)


def _consume_and_wait(src, buf, sem, *, vmem_limit_bytes=None):
    """RPA-like consumer: waits on the semaphore itself, then reads."""

    def kernel(src_hbm, buf_vmem, sem_ref, out_ref):
        pltpu.make_async_copy(src_hbm, buf_vmem, sem_ref).wait()
        out_ref[...] = buf_vmem[...] * 2.0 + 1.0

    return pl.pallas_call(
        kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct(_SHAPE, buf.dtype),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        name="handoff_consume_wait",
    )(src, buf, sem)


def _unrelated_work(x):
    # Enough XLA work between start and done that the scheduler has room to
    # move things around.
    y = x
    for _ in range(4):
        y = jnp.tanh(y @ y.T @ y * 1e-3)
    return y.sum()


def _vmem_hog(x, hog_mib):
    """A kernel whose scoped VMEM window competes with the handoff buffer."""
    rows = hog_mib * 256  # f32 rows of 1024 lanes = 4 KiB each.

    def kernel(x_ref, o_ref, scratch):
        @pl.loop(0, rows, step=512)
        def _(r):
            scratch[pl.ds(r, 512), :] = jnp.full((512, 1024), 3.0, jnp.float32)

        acc = jnp.zeros((8, 1024), jnp.float32)
        for r in range(0, rows, rows // 8):
            acc = acc + scratch[pl.ds(r, 8), :]
        o_ref[...] = acc + x_ref[pl.ds(0, 8), :]

    return pl.pallas_call(
        kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        out_shape=jax.ShapeDtypeStruct((8, 1024), jnp.float32),
        scratch_shapes=[pltpu.VMEM((rows, 1024), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=(hog_mib + 8) * 1024 * 1024
        ),
        name="handoff_hog",
    )(x)


def _fn(pattern, tagged, hint, vmem_limit_bytes, hog_mib):
    def f(src, other):
        buf, sem = _prefetch_start(src, tagged=tagged, hint=hint)
        side = _unrelated_work(other)
        if hog_mib:
            side = side + _vmem_hog(other, hog_mib).sum()
        if pattern == "start_done_consume":
            buf = _prefetch_done(src, buf, sem, tagged=tagged, hint=hint)
            out = _consume(buf, vmem_limit_bytes=vmem_limit_bytes)
        else:
            out = _consume_and_wait(src, buf, sem, vmem_limit_bytes=vmem_limit_bytes)
        return out, side

    return f


class VmemHandoffTest(parameterized.TestCase):
    @parameterized.product(
        pattern=("start_done_consume", "start_consume_wait"),
        tagged=(False, True),
        explicit=(False, True),
        hint=(False, True),
        hog_mib=(0, 32, 44, 52),
    )
    def test_handoff(self, pattern, tagged, explicit, hint, hog_mib):
        if pattern == "start_consume_wait" and tagged:
            self.skipTest("Tags require the start/done pair.")
        if explicit and hog_mib:
            self.skipTest("Explicit prefetch rejects per-kernel vmem_limit_bytes.")
        # RPA sets vmem_limit_bytes; mirror that whenever it is allowed.
        vmem_limit = not explicit
        key = jax.random.key(0)
        src = jax.random.normal(key, _SHAPE, jnp.float32)
        other = jax.random.normal(jax.random.key(1), (512, 1024), jnp.float32)
        vmem_limit_bytes = 32 * 1024 * 1024 if vmem_limit else None
        lowered = jax.jit(_fn(pattern, tagged, hint, vmem_limit_bytes, hog_mib)).lower(
            src, other
        )
        name = (
            f"{pattern} tagged={tagged} explicit={explicit} hint={hint}"
            f" hog={hog_mib}MiB"
        )
        try:
            compiled = lowered.compile(_EXPLICIT_FLAGS if explicit else {})
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"RESULT {name}: COMPILE_ERROR {str(e).splitlines()[0][:300]}")
            return
        hlo = compiled.as_text()
        # Copies whose operand is a handoff buffer (start output / done output).
        buf_copies = [
            line.strip()[:250]
            for line in hlo.splitlines()
            if (" copy(" in line or " copy-start(" in line)
            and ("buf" in line.split("copy")[-1] or "handoff" in line.split("copy")[-1])
        ]
        order = [
            line.strip().split(" ")[0].lstrip("%")
            for line in hlo.splitlines()
            if "custom-call(" in line
            or " copy-start(" in line
            or " copy-done(" in line
            or " fusion(" in line
            or " convolution(" in line
        ]
        # Compress unrelated ops into counts so the handoff placement is visible.
        compact = []
        for op in order:
            if "handoff" in op or "copy" in op:
                compact.append(op)
            elif compact and compact[-1].startswith("#"):
                compact[-1] = f"#{int(compact[-1][1:]) + 1}"
            else:
                compact.append("#1")
        order = compact
        out, _ = compiled(src, other)
        expected = np.asarray(src) * 2.0 + 1.0
        max_abs = float(np.max(np.abs(np.asarray(out) - expected)))
        print(f"RESULT {name}: max_abs_diff={max_abs} buf_copies={len(buf_copies)}")
        for copy_line in buf_copies:
            print(f"  BUFCOPY {copy_line}")
        print(f"  ORDER {' -> '.join(order)}")
        self.assertLess(max_abs, 1e-5)


if __name__ == "__main__":
    absltest.main()
