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
(dp, tp) = (4, 2) mesh of 8 CPU devices.

The device checks run this file as a script in a subprocess, because the
host device count (XLA_FLAGS) must be set before JAX starts. They use TPU
interpret mode with the race detector, so they check routing, the remote DMA
and semaphore protocols and the arithmetic (against a float32 reference of
the kernel's roundings), not Mosaic lowering, VMEM limits or TPU timing.
"""

import contextlib
import io
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_test

DP, TP = 4, 2
# Experts per core, hidden, intermediate, top-k: small enough for the
# interpreter, with every core holding experts some rows route to.
E, H, I, TOPK = 2, 512, 256, 2
REL_L2 = 1e-2
# rows per engine, dispatch form, combine form, FP8 dispatch, routed-row
# gather, rank as data. The first is the served form at a 32-row step.
CASES = {
    "served_32": (32, "hier", "hier", True, True, True),
    "bf16_dispatch_32": (32, "hier", "hier", False, True, True),
    "no_gather_32": (32, "hier", "hier", True, False, True),
    "rank_from_axis_32": (32, "hier", "hier", True, True, False),
    "padded_13": (13, "hier", "hier", False, True, True),
    "rows_16": (16, "hier", "hier", False, True, True),
    "flat_forms_16": (16, "flat", "flat", False, True, True),
}


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
    import jax.numpy as jnp
    import numpy as np
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc
    from vllm_torchtpu.kernels.experimental.moe_lowtok import kernel

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    pltpu.get_tpu_info = tpu7x
    tpu_info.get_tpu_info = tpu7x

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == DP * TP, devices
    mesh = Mesh(np.asarray(devices).reshape(DP, TP), ("dp", "tp"))
    rng = np.random.default_rng(0)
    n_experts = E * DP * TP
    w13 = jnp.asarray(
        np.clip(rng.standard_normal((n_experts, H, 2 * I)) * 40, -448, 448),
        kernel._FP8,
    )
    w2 = jnp.asarray(
        np.clip(rng.standard_normal((n_experts, I, H)) * 40, -448, 448), kernel._FP8
    )
    s13 = jnp.asarray(
        rng.random((n_experts, 1, 1, 2 * I)) / (40 * np.sqrt(H)), jnp.float32
    )
    s2 = jnp.asarray(rng.random((n_experts, 1, 1, H)) / (40 * np.sqrt(I)), jnp.float32)
    starts = jnp.asarray(np.arange(DP * TP) * E, jnp.int32)
    ranks = jnp.asarray([[i // TP, i % TP] for i in range(DP * TP)], jnp.int32)
    experts = P(("dp", "tp"))

    def inputs(rows):
        r = np.random.default_rng(rows)
        x = jnp.asarray(r.standard_normal((DP * rows, H)), jnp.bfloat16)
        logits = jnp.asarray(r.standard_normal((DP * rows, n_experts)) * 2, jnp.bfloat16)
        return x, logits

    def reference(x, logits):
        """Plain JAX with the kernel's roundings: per-row FP8 activations, a
        softmax top-k renormalized over every expert, float32 sums."""
        x_q, x_s = kernel._quantize_rows(x.astype(jnp.float32))
        weights, ids = jax.lax.top_k(jax.nn.softmax(logits.astype(jnp.float32)), TOPK)
        weights = weights / weights.sum(axis=1, keepdims=True)
        out = jnp.zeros((x.shape[0], H), jnp.float32)
        for e in range(n_experts):
            w = jnp.sum(jnp.where(ids == e, weights, 0), axis=1)
            h = jnp.dot(x_q, w13[e], preferred_element_type=jnp.float32)
            h = h * x_s * s13[e].reshape(1, -1)
            a = jax.nn.silu(h[:, :I]) * h[:, I:]
            a_q, a_s = kernel._quantize_rows(a)
            o = jnp.dot(a_q, w2[e], preferred_element_type=jnp.float32)
            out = out + w[:, None] * (o * a_s * s2[e].reshape(1, -1))
        return np.asarray(out.astype(jnp.bfloat16)).astype(np.float64)

    def run(rows, dform, cform, fp8, gather, rank_operand):
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
                topk=TOPK,
                dp=DP,
                tp=TP,
                dispatch="inkernel",
                combine="inkernel",
                dispatch_form=dform,
                combine_form=cform,
                depth=1,
                dispatch_fp8=fp8,
                rank=rank.reshape(2) if rank_operand else None,
                gather_rows=gather,
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
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out = np.asarray(
                jax.block_until_ready(f(x, logits, w13, w2, s13, s2, starts, ranks))
            )
        races = bool(interpret_pallas_call.races.races_found)
        cores = out.reshape(DP * TP, rows, H)
        got = np.concatenate([cores[TP * d] for d in range(DP)]).astype(np.float64)
        want = reference(x, logits)
        bits = cores.view(np.uint16)
        return bits, dict(
            races_found=races,
            pairs_identical=bool(
                all(np.array_equal(bits[TP * d], bits[TP * d + 1]) for d in range(DP))
            ),
            nan=int(np.isnan(got).sum()),
            rel_l2=float(np.linalg.norm(got - want) / np.linalg.norm(want)),
        )

    result = {"race_control": _race_control(), "cases": {}}
    bits = {}
    for name, case in CASES.items():
        bits[name], result["cases"][name] = run(*case)
    result["fp8_dispatch_bit_identical"] = bool(
        np.array_equal(bits["served_32"], bits["bf16_dispatch_32"])
    )
    result["rank_from_axis_bit_identical"] = bool(
        np.array_equal(bits["served_32"], bits["rank_from_axis_32"])
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


def test_race_detector_is_live(on_8_devices):
    assert on_8_devices["race_control"] == {"racy": True, "clean": False}


@pytest.mark.parametrize("name", sorted(CASES))
def test_engine_rows_summed_over_every_rank(on_8_devices, name):
    case = on_8_devices["cases"][name]
    assert not case["races_found"], case
    # Both cores of a chip end with the same bytes.
    assert case["pairs_identical"], case
    assert case["nan"] == 0, case
    assert case["rel_l2"] <= REL_L2, case


def test_fp8_dispatch_moves_the_same_expert_inputs(on_8_devices):
    """Rows pushed as per-row FP8 give the bytes the bf16 push gives: the
    expert pass quantizes the same rows the same way."""
    assert on_8_devices["fp8_dispatch_bit_identical"]


def test_rank_as_data_matches_the_axis_index(on_8_devices):
    assert on_8_devices["rank_from_axis_bit_identical"]


def _call(rows=32, **kwargs):
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc

    statics = dict(
        topk=TOPK,
        dp=DP,
        tp=TP,
        dispatch="inkernel",
        combine="inkernel",
        dispatch_form="hier",
        combine_form="hier",
        depth=1,
    )
    statics.update(kwargs)
    return fc.moe_lowtok_fused(
        jnp.zeros((rows, H), jnp.bfloat16),
        jnp.zeros((rows, E * DP * TP), jnp.bfloat16),
        jnp.zeros((E, H, 2 * I), jnp.float8_e4m3fn),
        jnp.zeros((E, 1, 1, 2 * I), jnp.float32),
        jnp.zeros((E, I, H), jnp.float8_e4m3fn),
        jnp.zeros((E, 1, 1, H), jnp.float32),
        **statics,
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(tp=4), "tp == 2"),
        (dict(rows=16, dispatch_fp8=True), "multiple of 32"),
        (dict(dispatch="xla", dispatch_fp8=True), "dispatch in the kernel"),
        (dict(dispatch_form="ring"), "forms are one of"),
        (dict(depth=0), "depth is at least 1"),
    ],
    ids=["hier_needs_tp2", "fp8_rows", "fp8_needs_inkernel", "form", "depth"],
)
def test_refuses_unsupported_calls(kwargs, match):
    rows = kwargs.pop("rows", 32)
    with pytest.raises(ValueError, match=match):
        _call(rows=rows, **kwargs)


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    # Around everything: pallas_call reads the mode when it is built.
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
