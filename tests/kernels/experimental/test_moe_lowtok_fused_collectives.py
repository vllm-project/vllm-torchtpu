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
"""The few-token MoE with its DP dispatch and EP combine in the kernel, and the
torch op's program around it, on a (dp, tp) = (4, 2) mesh.

The interpret checks run this file as a script in a subprocess on 8 CPU
devices, because the host device count (XLA_FLAGS) must be set before JAX
starts. They use TPU interpret mode with the race detector, so they check
routing, the remote DMA and semaphore protocols and the arithmetic (against
float32 references that share no code with the kernel), not Mosaic lowering,
VMEM limits or TPU timing. They also run the program the torch op builds
(moe_lowtok_fused_op.program) at, below and above its row cap; above it, the
scores entry's own path (moe_scores_fn) serves the gathered rows, here through
its few-token branch, since the interpreter cannot run the grouped matmul's
DMA pipeline. The multichip checks run the compiled kernel and program on the
8 TPU cores of one host against the same references, at these test shapes and
at the Qwen3.5-397B decode shapes the torch op serves, where the path above
the cap is the grouped matmul.
"""

import contextlib
import inspect
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
# The torch op's program, armed with a cap of 32 rows per engine at these
# shapes: rows per engine, whether the step runs the fused kernel (else the
# all-gather, moe_scores_fn, reduce-scatter path) and with FP8 rows. Each case
# has one row whose router logits are all -inf, as apply() masks a padded row.
OP_CAP = DP * 32
OP_CASES = {
    "op_at_cap_32": (32, True, True),
    "op_below_cap_13": (13, True, False),
    "op_above_cap_33": (33, False, False),
}
# At the 397B shapes with the served cap (448 rows over 4 engines).
Q35_OP_CAP = 448
Q35_OP_CASES = {
    "q35_op_at_cap_112": (112, True, False),
    "q35_op_below_cap_64": (64, True, True),
    "q35_op_above_cap_128": (128, False, False),
}
# The row of each engine-0 block whose logits are all -inf.
PADDED_ROW = 3
# Junk in each engine's padded row (all -inf logits, so no expert takes it),
# which the kernel must not mix into the other rows: rows per engine, FP8
# dispatch, and NaN, +Inf and -Inf across the columns or the largest finite
# bf16 values.
JUNK_CASES = {
    "fp8_32_non_finite": (32, True, "non_finite"),
    "fp8_32_large_finite": (32, True, "large_finite"),
    "bf16_32_non_finite": (32, False, "non_finite"),
    "bf16_32_large_finite": (32, False, "large_finite"),
}
# The same at the 397B shapes, in the forms the torch op runs.
Q35_JUNK_CASES = {
    "q35_fp8_64_non_finite": (64, True, "non_finite"),
    "q35_bf16_48_non_finite": (48, False, "non_finite"),
}


def _reference_quantize_rows(v):
    """Per-row FP8 quantization (e4m3, max 448) for the references, written
    independently of the kernel's helper."""
    import jax.numpy as jnp

    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / 448.0
    return (v / scale).astype(jnp.float8_e4m3fn), scale


def _setup(mesh, geometry=(E, H, INTER, TOPK)):
    """Weights split over ``mesh`` (axes ("dp", "tp")), every core's first
    expert and (dp, tp) index, and the inputs and references of a step."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

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

    def inputs(rows, padded_row=None):
        """Every engine's rows and router logits; row ``padded_row`` of each
        engine's block has all -inf logits."""
        r = np.random.default_rng(rows)
        x = jnp.asarray(r.standard_normal((DP * rows, h)), jnp.bfloat16)
        logits = r.standard_normal((DP * rows, n_experts)) * 2
        if padded_row is not None:
            logits[padded_row::rows] = -np.inf
        return x, jnp.asarray(logits, jnp.bfloat16)

    def reference(x, logits):
        """Plain JAX with the kernel's roundings: per-row FP8 activations, a
        softmax top-k renormalized over every expert, float32 sums, zero for a
        row with no finite logit."""
        x_q, x_s = _reference_quantize_rows(x.astype(jnp.float32))
        lg = logits.astype(jnp.float32)
        routed = jnp.isfinite(lg).any(axis=1, keepdims=True)
        weights, ids = jax.lax.top_k(jax.nn.softmax(jnp.where(routed, lg, 0)), topk)
        weights = jnp.where(routed, weights / weights.sum(axis=1, keepdims=True), 0)
        out = jnp.zeros((x.shape[0], h), jnp.float32)
        for expert in range(n_experts):
            w = jnp.sum(jnp.where(ids == expert, weights, 0), axis=1)
            hid = jnp.dot(x_q, w13[expert], preferred_element_type=jnp.float32)
            hid = hid * x_s * s13[expert].reshape(1, -1)
            a = jax.nn.silu(hid[:, :inter]) * hid[:, inter:]
            a_q, a_s = _reference_quantize_rows(a)
            o = jnp.dot(a_q, w2[expert], preferred_element_type=jnp.float32)
            out = out + w[:, None] * (o * a_s * s2[expert].reshape(1, -1))
        return np.asarray(out.astype(jnp.bfloat16)).astype(np.float64)

    return (w13, w2, s13, s2), starts, ranks, inputs, reference


def _case_runner(mesh, geometry=(E, H, INTER, TOPK)):
    """``run(*case)`` on ``mesh``: the kernel's output of every core
    [DP * TP, rows, hidden] bf16 and the float32 reference of the engines'
    rows (float64)."""
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc

    topk = geometry[3]
    (w13, w2, s13, s2), starts, ranks, inputs, reference = _setup(mesh, geometry)
    h = geometry[1]
    experts = P(("dp", "tp"))

    def run(rows, fp8, edit=None):
        x, logits = inputs(rows)
        if edit is not None:
            x, logits = edit(x, logits)

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


def _junk_summary(run, rows, fp8, junk, geometry=(E, H, INTER, TOPK)):
    """``run`` (a ``_case_runner``) with ``junk`` in every engine's padded
    row, against the same step with that row as drawn: the bits of every other
    row, and the padded rows' output."""
    import jax.numpy as jnp
    import numpy as np

    h = geometry[1]
    padded = np.zeros((DP, rows), bool)
    padded[:, PADDED_ROW] = True

    def masked(x, logits):
        lg = np.array(logits.astype(jnp.float32))
        lg[padded.reshape(-1)] = -np.inf
        return x, jnp.asarray(lg, jnp.bfloat16)

    if junk == "non_finite":
        row = np.resize([np.nan, np.inf, -np.inf, 1.0], h)
    else:
        big = float(jnp.finfo(jnp.bfloat16).max)
        row = np.resize([big, -big], h)

    def planted(x, logits):
        x = np.array(x.astype(jnp.float32))
        x[padded.reshape(-1)] = row
        return masked(jnp.asarray(x, jnp.bfloat16), logits)

    base, want = run(rows, fp8, masked)
    got, _ = run(rows, fp8, planted)
    bits = got.view(np.uint16).reshape(DP, TP, rows, h)
    base_bits = base.view(np.uint16).reshape(DP, TP, rows, h)
    engines = np.concatenate([base[TP * d] for d in range(DP)]).astype(np.float64)
    out = np.concatenate([got[TP * d] for d in range(DP)]).astype(np.float64)
    kept = ~padded.reshape(-1)
    return dict(
        other_rows_identical=all(
            np.array_equal(bits[d, t][~padded[d]], base_bits[d, t][~padded[d]])
            for d in range(DP)
            for t in range(TP)
        ),
        nan=int(np.isnan(out[kept]).sum()),
        padded_rows_zero=bool(np.all(out[~kept] == 0)),
        base_rel_l2=float(np.linalg.norm(engines - want) / np.linalg.norm(want)),
    )


def _op_runner(mesh, cap, geometry=(E, H, INTER, TOPK), scores_cap=None):
    """``run(rows)`` on ``mesh``: the torch op's program (armed with ``cap``)
    on every engine's rows, with operands in apply()'s order. Returns its
    output [DP * rows, hidden] (float64), the float32 reference, the served
    path's sum (each core's moe_scores_fn on every engine's rows, summed over
    the cores), and which forms the traced program holds. ``scores_cap`` is
    the scores entry's cap moe_scores_fn reads (default ``cap``, as the rule
    arms both)."""
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.layers.adapter import moe_lowtok_fused_op
    from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores

    (w13, w2, s13, s2), starts, ranks, inputs, reference = _setup(mesh, geometry)
    moe_lowtok_scores._ARMED_MAX_TOKENS = cap if scores_cap is None else scores_cap
    topk = geometry[3]
    fn = moe_lowtok_fused_op.program(mesh, topk, cap)
    core = P(("dp", "tp"))

    def served_partial(x, logits, w13, w2, s13, s2, start):
        return moe_lowtok_scores.moe_scores_fn(
            x, logits, w13, w2, s13, s2, start[0], topk=topk
        )[None]

    served = jax.jit(
        jax.shard_map(
            served_partial,
            mesh=mesh,
            in_specs=(P(), P()) + (core,) * 5,
            out_specs=core,
            check_vma=False,
        )
    )

    def run(rows):
        x, logits = inputs(rows, padded_row=PADDED_ROW)
        args = (x, logits, w13, w2, s13, s2, starts, ranks.reshape(-1))
        program = str(jax.make_jaxpr(fn)(*args))
        with contextlib.redirect_stdout(io.StringIO()):
            out = np.asarray(jax.block_until_ready(jax.jit(fn)(*args)))
            parts = np.asarray(
                jax.block_until_ready(served(x, logits, w13, w2, s13, s2, starts))
            )
        forms = dict(
            fused_kernel="moe_lowtok_fused_dih" in program,
            fp8_rows="moe_lowtok_fused_dihq" in program,
            all_gather="all_gather" in program,
        )
        return (
            out.astype(np.float64),
            reference(x, logits),
            parts.astype(np.float64).sum(axis=0),
            forms,
        )

    return run


def _built_op(mesh):
    """What moe_lowtok_fused_op._build_op registers, for top-k TOPK with the
    cap armed at OP_CAP, on a one-axis EP mesh of ``mesh``'s devices in mesh
    order: the op's name and specs, its (dp, tp) view, and the forms its
    program traces at the cap and one row per engine above it."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh

    from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
    from vllm_torchtpu.kernels.experimental.moe_lowtok import kernel
    from vllm_torchtpu.layers.adapter import moe_lowtok_fused_op as op
    from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores

    devices = np.asarray(mesh.devices).reshape(-1)
    registered = {}

    def record(name, fn, **kwargs):
        registered.update(kwargs, name=name, fn=fn)
        return object()

    op.build_ep_mesh = lambda: Mesh(devices, (EP_AXIS_NAME,))
    op.sharded_jax_op = record
    op._DP, op._TP = DP, TP
    op._OPS.clear()
    moe_lowtok_scores._ARMED_MAX_TOKENS = OP_CAP
    op._build_op(TOPK)
    view = registered["mesh"]
    n, s = E * DP * TP, jax.ShapeDtypeStruct

    def forms(rows):
        program = str(
            jax.make_jaxpr(registered["fn"])(
                s((DP * rows, H), jnp.bfloat16),
                s((DP * rows, n), jnp.bfloat16),
                s((n, H, 2 * INTER), kernel.FP8_DTYPE),
                s((n, INTER, H), kernel.FP8_DTYPE),
                s((n, 1, 1, 2 * INTER), jnp.float32),
                s((n, 1, 1, H), jnp.float32),
                s((DP * TP,), jnp.int32),
                s((DP * TP * 2,), jnp.int32),
            )
        )
        return dict(
            fused_kernel="moe_lowtok_fused_dih" in program,
            all_gather="all_gather" in program,
        )

    return dict(
        name=registered["name"],
        fake_operands=list(inspect.signature(registered["fake"]).parameters),
        program_operands=list(inspect.signature(registered["fn"]).parameters),
        axes=list(view.axis_names),
        device_ids=[[d.id for d in row] for row in np.asarray(view.devices)],
        mesh_order=[[d.id for d in row] for row in devices.reshape(DP, TP)],
        specs=registered["input_partition_specs"] == op._IN_SPECS
        and registered["output_partition_specs"] == op._OUT_SPEC,
        at_cap=forms(OP_CAP // DP),
        above_cap=forms(OP_CAP // DP + 1),
    )


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


def _op_summary(got, want, want_served, forms, rows):
    """The op program's checks: no NaN, each padded row zero, and the rows
    against the float32 reference and against the served path's sum."""
    import numpy as np

    def rel(ref):
        return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))

    return dict(
        forms,
        nan=int(np.isnan(got).sum()),
        padded_rows_zero=bool(np.all(got[PADDED_ROW::rows] == 0)),
        rel_l2=rel(want),
        rel_l2_served=rel(want_served),
    )


def _check(case):
    # Both cores of a chip end with the same bytes.
    assert case["pairs_identical"], case
    assert case["nan"] == 0, case
    assert case["rel_l2"] <= REL_L2, case


def _check_junk(case):
    assert case["other_rows_identical"], case
    assert case["nan"] == 0, case
    assert case["padded_rows_zero"], case
    assert case["base_rel_l2"] <= REL_L2, case


def _check_op(case, fused_kernel, fp8_rows, served_path_has_kernel_roundings=True):
    """The program took the form its row count selects and summed the
    engines' rows: against the float32 reference of the kernel's roundings
    and, above the cap, also against each core's moe_scores_fn summed over
    the cores (the path's own arithmetic, which isolates the gather,
    reduce-scatter and all-reduce around it). The grouped matmul quantizes
    its activations in blocks, so where it serves above the cap
    (``served_path_has_kernel_roundings=False``) only its own sum is checked
    there."""
    assert case["fused_kernel"] is fused_kernel, case
    assert case["fp8_rows"] is fp8_rows, case
    assert case["all_gather"] is not fused_kernel, case
    assert case["nan"] == 0, case
    assert case["padded_rows_zero"], case
    if fused_kernel or served_path_has_kernel_roundings:
        assert case["rel_l2"] <= REL_L2, case
    if not fused_kernel:
        assert case["rel_l2_served"] <= REL_L2, case


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
    from jax._src import tpu_info as pipeline_tpu_info
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh

    from vllm_torchtpu.kernels.experimental.moe_lowtok import kernel

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    pltpu.get_tpu_info = tpu7x
    tpu_info.get_tpu_info = tpu7x
    # The grouped matmul above the op's cap tiles its pipeline by this one.
    pipeline_tpu_info.get_tpu_info = tpu7x

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == DP * TP, devices
    mesh = Mesh(np.asarray(devices).reshape(DP, TP), ("dp", "tp"))
    run = _case_runner(mesh)

    result = {"race_control": _race_control(), "cases": {}, "op_cases": {}}
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
    result["junk_cases"] = {}
    for name, case in JUNK_CASES.items():
        pltpu.reset_tpu_interpret_mode_state()
        result["junk_cases"][name] = _junk_summary(run, *case)
        result["junk_cases"][name]["races_found"] = bool(
            interpret_pallas_call.races.races_found
        )
    # moe_scores_fn's few-token branch serves the gathered rows above the cap:
    # the interpreter cannot run the grouped matmul's DMA pipeline.
    run_op = _op_runner(mesh, OP_CAP, scores_cap=kernel.MOE_LOWTOK_MAX_TOKENS)
    for name, (rows, _, _) in OP_CASES.items():
        pltpu.reset_tpu_interpret_mode_state()
        result["op_cases"][name] = _op_summary(*run_op(rows), rows)
        result["op_cases"][name]["races_found"] = bool(
            interpret_pallas_call.races.races_found
        )
    result["built_op"] = _built_op(mesh)
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={DP * TP}",
        # As the DP4 x TP2 decode leg runs the path above the op's cap.
        TPU_MOE_SKIP_PADDED_TOKENS="1",
    )
    # Below the CPU lane's 600 s pytest timeout, which this module-scoped
    # setup counts against, so this one fires first and shows the child's
    # output.
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=540,
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


@pytest.mark.cpu_test
@pytest.mark.parametrize("name", sorted(JUNK_CASES))
def test_rows_no_expert_of_a_core_takes_stay_out_of_its_gather(on_8_devices, name):
    """The routed-row gather is a one-hot matmul over every row, so a NaN or
    Inf left in a row no expert takes would turn every gathered row into NaN.
    A padded row is such a row: apply() masks its router logits to -inf and
    passes its activations on. Junk planted there leaves every other row
    bit-identical and the padded rows zero."""
    case = on_8_devices["junk_cases"][name]
    assert not case["races_found"], case
    _check_junk(case)


@pytest.mark.cpu_test
@pytest.mark.parametrize("name", sorted(OP_CASES))
def test_op_program_sums_the_engine_rows_on_either_side_of_its_cap(on_8_devices, name):
    """The torch op's program with operands in apply()'s order: at and below
    its cap the fused kernel (FP8 rows where the rows per engine allow), above
    it the all-gather, moe_scores_fn and reduce-scatter path; a row with no
    finite logit (a padded row) adds nothing."""
    case = on_8_devices["op_cases"][name]
    assert not case["races_found"], case
    _check_op(case, *OP_CASES[name][1:])


@pytest.mark.cpu_test
def test_build_op_registers_the_program_on_the_mesh_view_with_the_armed_cap(
    on_8_devices,
):
    """_build_op, which serves, hands sharded_jax_op the program on the
    (dp, tp) view of the EP mesh in mesh order, so each core's position is
    the one its rank operand names, and with the armed cap: the fused kernel
    at the cap, the all-gather path one row per engine above it."""
    built = on_8_devices["built_op"]
    assert built["name"] == f"pallas::moe_lowtok_fused_k{TOPK}"
    # The fake takes the program's operands, by name.
    assert built["fake_operands"] == built["program_operands"], built
    assert built["axes"] == ["dp", "tp"]
    assert built["device_ids"] == built["mesh_order"]
    assert built["specs"]
    assert built["at_cap"] == {"fused_kernel": True, "all_gather": False}
    assert built["above_cap"] == {"fused_kernel": False, "all_gather": True}


def _call(rows=32, experts=E, logit_cols=None, **kwargs):
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc

    statics = dict(rank=jnp.zeros((2,), jnp.int32), topk=TOPK, dp=DP, tp=TP)
    statics.update(kwargs)
    return fc.moe_lowtok_fused(
        jnp.zeros((rows, H), jnp.bfloat16),
        jnp.zeros((rows, logit_cols or experts * DP * TP), jnp.bfloat16),
        jnp.zeros((experts, H, 2 * INTER), jnp.float8_e4m3fn),
        jnp.zeros((experts, 1, 1, 2 * INTER), jnp.float32),
        jnp.zeros((experts, INTER, H), jnp.float8_e4m3fn),
        jnp.zeros((experts, 1, 1, H), jnp.float32),
        **statics,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(tp=4), "tp == 2"),
        (dict(rows=16, dispatch_fp8=True), "multiple of 32"),
        # 96 experts per rank: the rank starting at expert 96 holds lanes
        # 96..191, across the first 128-lane block of the routing table.
        (dict(experts=96), "do not divide 128"),
        (dict(experts=3), "do not divide 128"),
        (dict(logit_cols=E * DP * TP - 1), "every expert's logit"),
    ],
    ids=[
        "hier_needs_tp2",
        "fp8_rows",
        "experts_cross_a_lane_block",
        "experts_3",
        "logits_short_of_the_experts",
    ],
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
    every id in the registry (small_collectives/common.py) and the fused EP v3
    kernels' two; and each kernel of this package takes its id from the
    registry."""
    from vllm_torchtpu.kernels.experimental.moe_lowtok import fused_collectives as fc
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        all_reduce,
        common,
        reduce_scatter,
    )
    from vllm_torchtpu.kernels.fused_moe.v3 import layout

    registered = {
        name: value
        for name, value in vars(common).items()
        if name.endswith("_COLLECTIVE_ID")
    }
    ids = [
        *registered.values(),
        layout.COLLECTIVE_ID_FFN,
        layout.COLLECTIVE_ID_TRANSPORT,
    ]
    assert len(set(ids)) == len(ids), registered
    assert all_reduce.COLLECTIVE_ID == common.ALL_REDUCE_COLLECTIVE_ID
    assert reduce_scatter.COLLECTIVE_ID == common.REDUCE_SCATTER_COLLECTIVE_ID
    assert fc.COLLECTIVE_ID == common.MOE_LOWTOK_FUSED_COLLECTIVE_ID


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


@pytest.mark.multichip
@pytest.mark.parametrize("name", sorted(JUNK_CASES))
def test_compiled_kernel_keeps_junk_rows_out_of_the_gather_on_tpu(tpu_mesh, name):
    """The interpret check's junk rows on the compiled kernel (the MXU's
    one-hot gather)."""
    _check_junk(_junk_summary(_case_runner(tpu_mesh), *JUNK_CASES[name]))


@pytest.mark.multichip
@pytest.mark.parametrize("name", sorted(Q35_JUNK_CASES))
def test_compiled_kernel_keeps_junk_rows_out_of_the_gather_at_the_qwen3_5_shapes_on_tpu(
    tpu_mesh, name
):
    """Junk rows at the Qwen3.5-397B MoE layer's shapes, in the forms the
    torch op runs."""
    run = _case_runner(tpu_mesh, Q35_GEOMETRY)
    _check_junk(_junk_summary(run, *Q35_JUNK_CASES[name], geometry=Q35_GEOMETRY))


@pytest.fixture
def armed_scores_cap(monkeypatch):
    """The scores entry's cap left as the test found it, and padded-row
    skipping on, as on the DP4 x TP2 decode leg."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.quantization import moe_lowtok_scores

    monkeypatch.setattr(moe_lowtok_scores, "_ARMED_MAX_TOKENS", 0)
    # In the module's dict: it shadows the environment and any attribute
    # another test left, and is removed afterwards.
    monkeypatch.setitem(vars(envs), "TPU_MOE_SKIP_PADDED_TOKENS", True)


@pytest.mark.multichip
@pytest.mark.parametrize("name", sorted(OP_CASES))
def test_compiled_op_program_on_tpu(tpu_mesh, armed_scores_cap, name):
    """The torch op's program compiled, at the interpret checks' cases."""
    from vllm_torchtpu.kernels.experimental.moe_lowtok import kernel

    rows = OP_CASES[name][0]
    run = _op_runner(tpu_mesh, OP_CAP, scores_cap=kernel.MOE_LOWTOK_MAX_TOKENS)
    case = _op_summary(*run(rows), rows)
    _check_op(case, *OP_CASES[name][1:])


@pytest.mark.multichip
@pytest.mark.parametrize("name", sorted(Q35_OP_CASES))
def test_compiled_op_program_at_the_qwen3_5_decode_shapes_on_tpu(
    tpu_mesh, armed_scores_cap, name
):
    """The torch op's program at the Qwen3.5-397B MoE layer's shapes and the
    served cap: the fused kernel at and below 112 rows per engine, the grouped
    matmul path above."""
    rows = Q35_OP_CASES[name][0]
    run = _op_runner(tpu_mesh, Q35_OP_CAP, Q35_GEOMETRY)
    case = _op_summary(*run(rows), rows)
    _check_op(case, *Q35_OP_CASES[name][1:], served_path_has_kernel_roundings=False)


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    # Around everything: pallas_call reads the mode when it is built.
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
