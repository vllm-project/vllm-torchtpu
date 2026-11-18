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
"""The few-token MoE with its DP dispatch and EP combine in the kernel, on a
(dp, tp) = (4, 2) mesh.

The interpret checks run this file as a script in a subprocess on 8 CPU
devices, because the host device count (XLA_FLAGS) must be set before JAX
starts. They use TPU interpret mode with the race detector, so they check
routing, the remote DMA and semaphore protocols and the arithmetic (against a
float32 reference of the kernel's roundings), not Mosaic lowering, VMEM limits
or TPU timing. The multichip checks run the compiled kernel on the 8 TPU
cores of one host against the same reference, at these test shapes and at the
Qwen3.5-397B decode shapes the torch op serves.
"""

import contextlib
import io
import json
import os
import subprocess
import sys

import pytest

DP, TP = 4, 2
# Experts per core, hidden, intermediate, top-k: small enough for the
# interpreter, with every core holding experts some rows route to.
E, H, INTER, TOPK = 2, 512, 256, 2
# The Qwen3.5-397B MoE layer the torch op serves: 512 experts over the 8 cores,
# hidden 4096, expert intermediate 1024, top-10.
Q35_GEOMETRY = (64, 4096, 1024, 10)
REL_L2 = 1e-2
# rows per engine, FP8 dispatch. The first is the torch op's call at a 32-row
# step.
CASES = {
    "served_32": (32, True),
    "bf16_dispatch_32": (32, False),
    "padded_13": (13, False),
    "rows_16": (16, False),
}
# At the 397B shapes, the calls the torch op makes: FP8 rows where a step's
# rows per engine pad to a multiple of 32, bf16 rows otherwise.
Q35_CASES = {
    "q35_served_64": (64, True),
    "q35_served_48": (48, False),
    "q35_served_8": (8, False),
}


def _case_runner(mesh, geometry=(E, H, INTER, TOPK)):
    """``run(*case)`` on ``mesh`` (axes ("dp", "tp")): the kernel's output of
    every core [DP * TP, rows, hidden] bf16 and the float32 reference of this
    engine's rows (float64)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc
    from vllm_torchtpu.kernels.experimental.moe_lowtok import kernel

    e, h, inter, topk = geometry
    n_experts = e * DP * TP
    if geometry == (E, H, INTER, TOPK):
        rng = np.random.default_rng(0)
        w13 = jnp.asarray(
            np.clip(rng.standard_normal((n_experts, h, 2 * inter)) * 40, -448, 448),
            kernel.FP8_DTYPE,
        )
        w2 = jnp.asarray(
            np.clip(rng.standard_normal((n_experts, inter, h)) * 40, -448, 448),
            kernel.FP8_DTYPE,
        )
        s13 = jnp.asarray(
            rng.random((n_experts, 1, 1, 2 * inter)) / (40 * np.sqrt(h)), jnp.float32
        )
        s2 = jnp.asarray(
            rng.random((n_experts, 1, 1, h)) / (40 * np.sqrt(inter)), jnp.float32
        )
    else:
        # The same distributions drawn on the devices, already split over the
        # mesh: the 397B layer's weights are gigabytes.
        keys = jax.random.split(jax.random.PRNGKey(0), 4)

        def draw():
            def fp8(key, shape):
                w = jax.random.normal(key, shape, jnp.float32) * 40
                return jnp.clip(w, -448, 448).astype(kernel.FP8_DTYPE)

            return (
                fp8(keys[0], (n_experts, h, 2 * inter)),
                fp8(keys[1], (n_experts, inter, h)),
                jax.random.uniform(keys[2], (n_experts, 1, 1, 2 * inter))
                / (40 * np.sqrt(h)),
                jax.random.uniform(keys[3], (n_experts, 1, 1, h))
                / (40 * np.sqrt(inter)),
            )

        split = NamedSharding(mesh, P(("dp", "tp")))
        w13, w2, s13, s2 = jax.jit(draw, out_shardings=(split,) * 4)()
    starts = jnp.asarray(np.arange(DP * TP) * e, jnp.int32)
    ranks = jnp.asarray([[i // TP, i % TP] for i in range(DP * TP)], jnp.int32)
    experts = P(("dp", "tp"))

    def inputs(rows):
        r = np.random.default_rng(rows)
        x = jnp.asarray(r.standard_normal((DP * rows, h)), jnp.bfloat16)
        logits = jnp.asarray(
            r.standard_normal((DP * rows, n_experts)) * 2, jnp.bfloat16
        )
        return x, logits

    def reference(x, logits):
        """Plain JAX with the kernel's roundings: per-row FP8 activations, a
        softmax top-k renormalized over every expert, float32 sums."""
        x_q, x_s = kernel.quantize_rows(x.astype(jnp.float32))
        weights, ids = jax.lax.top_k(jax.nn.softmax(logits.astype(jnp.float32)), topk)
        weights = weights / weights.sum(axis=1, keepdims=True)
        out = jnp.zeros((x.shape[0], h), jnp.float32)
        for expert in range(n_experts):
            w = jnp.sum(jnp.where(ids == expert, weights, 0), axis=1)
            hid = jnp.dot(x_q, w13[expert], preferred_element_type=jnp.float32)
            hid = hid * x_s * s13[expert].reshape(1, -1)
            a = jax.nn.silu(hid[:, :inter]) * hid[:, inter:]
            a_q, a_s = kernel.quantize_rows(a)
            o = jnp.dot(a_q, w2[expert], preferred_element_type=jnp.float32)
            out = out + w[:, None] * (o * a_s * s2[expert].reshape(1, -1))
        return np.asarray(out.astype(jnp.bfloat16)).astype(np.float64)

    def run(rows, fp8):
        x, logits = inputs(rows)

        def body(x, logits, w13, w2, s13, s2, start, rank):
            return fc.moe_lowtok_fused(
                x,
                logits,
                w13,
                s13,
                w2,
                s2,
                start[0],
                rank=rank.reshape(2),
                topk=topk,
                dp=DP,
                tp=TP,
                dispatch_fp8=fp8,
            )

        # Each core's output stacked, so both cores of a TP pair are compared.
        f = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("dp"), P("dp")) + (experts,) * 6,
                out_specs=experts,
                check_vma=False,
            )
        )
        with contextlib.redirect_stdout(io.StringIO()):
            out = np.asarray(
                jax.block_until_ready(f(x, logits, w13, w2, s13, s2, starts, ranks))
            )
        return out.reshape(DP * TP, rows, h), reference(x, logits)

    return run


def _summary(cores, want):
    """(bits of every core, the checks on them): both cores of a TP pair hold
    the same bytes, no NaN, and the engines' rows against the reference."""
    import numpy as np

    got = np.concatenate([cores[TP * d] for d in range(DP)]).astype(np.float64)
    bits = cores.view(np.uint16)
    return bits, dict(
        pairs_identical=bool(
            all(np.array_equal(bits[TP * d], bits[TP * d + 1]) for d in range(DP))
        ),
        nan=int(np.isnan(got).sum()),
        rel_l2=float(np.linalg.norm(got - want) / np.linalg.norm(want)),
    )


def _check(case):
    # Both cores of a chip end with the same bytes.
    assert case["pairs_identical"], case
    assert case["nan"] == 0, case
    assert case["rel_l2"] <= REL_L2, case


def _race_control():
    """The detector on a kernel that reads a DMA target before its wait."""
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def make(read_early):
        def body(x_hbm, o_ref, buf, sem):
            copy = pltpu.make_async_copy(x_hbm, buf, sem)
            copy.start()
            if read_early:
                o_ref[...] = buf[...]
            copy.wait()
            if not read_early:
                o_ref[...] = buf[...]

        return pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((8, 128), jnp.float32),
                pltpu.SemaphoreType.DMA(()),
            ],
        )

    found = {}
    for name, read_early in (("racy", True), ("clean", False)):
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            jax.block_until_ready(make(read_early)(jnp.ones((8, 128), jnp.float32)))
        found[name] = bool(interpret_pallas_call.races.races_found)
    return found


def _worker():
    import jax
    import numpy as np
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    pltpu.get_tpu_info = tpu7x
    tpu_info.get_tpu_info = tpu7x

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == DP * TP, devices
    run = _case_runner(Mesh(np.asarray(devices).reshape(DP, TP), ("dp", "tp")))

    result = {"race_control": _race_control(), "cases": {}}
    bits = {}
    for name, case in CASES.items():
        pltpu.reset_tpu_interpret_mode_state()
        cores, want = run(*case)
        bits[name], result["cases"][name] = _summary(cores, want)
        result["cases"][name]["races_found"] = bool(
            interpret_pallas_call.races.races_found
        )
    result["fp8_dispatch_bit_identical"] = bool(
        np.array_equal(bits["served_32"], bits["bf16_dispatch_32"])
    )
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={DP * TP}",
    )
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.cpu_test
def test_race_detector_is_live(on_8_devices):
    assert on_8_devices["race_control"] == {"racy": True, "clean": False}


@pytest.mark.cpu_test
@pytest.mark.parametrize("name", sorted(CASES))
def test_engine_rows_summed_over_every_rank(on_8_devices, name):
    case = on_8_devices["cases"][name]
    assert not case["races_found"], case
    _check(case)


@pytest.mark.cpu_test
def test_fp8_dispatch_moves_the_same_expert_inputs(on_8_devices):
    """Rows pushed as per-row FP8 give the bytes the bf16 push gives: the
    expert pass quantizes the same rows the same way."""
    assert on_8_devices["fp8_dispatch_bit_identical"]


def _call(rows=32, **kwargs):
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc

    statics = dict(rank=jnp.zeros((2,), jnp.int32), topk=TOPK, dp=DP, tp=TP)
    statics.update(kwargs)
    return fc.moe_lowtok_fused(
        jnp.zeros((rows, H), jnp.bfloat16),
        jnp.zeros((rows, E * DP * TP), jnp.bfloat16),
        jnp.zeros((E, H, 2 * INTER), jnp.float8_e4m3fn),
        jnp.zeros((E, 1, 1, 2 * INTER), jnp.float32),
        jnp.zeros((E, INTER, H), jnp.float8_e4m3fn),
        jnp.zeros((E, 1, 1, H), jnp.float32),
        **statics,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(tp=4), "tp == 2"),
        (dict(rows=16, dispatch_fp8=True), "multiple of 32"),
    ],
    ids=["hier_needs_tp2", "fp8_rows"],
)
def test_refuses_unsupported_calls(kwargs, match):
    rows = kwargs.pop("rows", 32)
    with pytest.raises(ValueError, match=match):
        _call(rows=rows, **kwargs)


@pytest.mark.cpu_test
def test_fp8_dispatch_fits_whole_fp8_tiles_of_engine_rows():
    """Rows per engine pad to whole bf16 tiles (16); FP8 rows need whole fp8
    tiles (32): the torch op's dispatch choice and the kernel's refusal share
    this rule."""
    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc

    assert [fc.engine_rows(t) for t in (1, 13, 16, 17, 48, 64)] == [
        16,
        16,
        16,
        32,
        48,
        64,
    ]
    fits = [t for t in range(1, 97) if fc.fp8_dispatch_fits(t)]
    assert fits == list(range(17, 33)) + list(range(49, 65)) + list(range(81, 97))


@pytest.mark.cpu_test
def test_each_collective_of_the_decode_program_has_its_own_barrier_group():
    """The kernels that can run in one decode program on the DP x TP layout:
    the small all-reduce and reduce-scatter, this kernel and the fused EP v3
    kernels."""
    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        all_reduce,
        reduce_scatter,
    )
    from vllm_torchtpu.kernels.fused_moe.v3 import layout

    ids = (
        all_reduce.COLLECTIVE_ID,
        reduce_scatter.COLLECTIVE_ID,
        fc.COLLECTIVE_ID,
        layout.COLLECTIVE_ID_FFN,
        layout.COLLECTIVE_ID_TRANSPORT,
    )
    assert len(set(ids)) == len(ids), ids


@pytest.fixture(scope="module")
def tpu_mesh():
    """The eight cores of one host as the (dp, tp) mesh. A fixture, not a
    `skipif`: asking JAX for devices at collection would claim the chips."""
    import jax
    import numpy as np
    from jax.sharding import Mesh

    if jax.default_backend() != "tpu" or jax.device_count() < DP * TP:
        pytest.skip("needs the 8 TPU cores of one host")
    devices = np.asarray(jax.devices()[: DP * TP]).reshape(DP, TP)
    return Mesh(devices, ("dp", "tp"))


@pytest.mark.multichip
@pytest.mark.parametrize("name", sorted(CASES))
def test_compiled_kernel_sums_the_engine_rows_on_tpu(tpu_mesh, name):
    """The Mosaic-compiled kernel (VMEM budget, remote DMAs, barrier) at the
    interpret checks' cases."""
    _, case = _summary(*_case_runner(tpu_mesh)(*CASES[name]))
    _check(case)


@pytest.mark.multichip
@pytest.mark.parametrize("name", sorted(Q35_CASES))
def test_compiled_kernel_at_the_qwen3_5_decode_shapes_on_tpu(tpu_mesh, name):
    """The forms the torch op runs, at the Qwen3.5-397B MoE layer's shapes."""
    _, case = _summary(*_case_runner(tpu_mesh, Q35_GEOMETRY)(*Q35_CASES[name]))
    _check(case)


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    # Around everything: pallas_call reads the mode when it is built.
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
