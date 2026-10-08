# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""moe_tp_decode: admission, and the fused kernel and the adapter's op body on
16 CPU devices.

The 16-device checks run this file as a script in a subprocess, because the
host device count (XLA_FLAGS) must be set before JAX starts. They use TPU
interpret mode with the race detector, so they check routing, the DMA and
semaphore protocols and the arithmetic, not Mosaic lowering, VMEM limits or
TPU timing.
"""

import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental import moe_tp_decode as kernel
from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce

N_RANKS = 16
# One rank's slice of GLM-5.2 at TP16.
GLM52_TP16 = dict(h=6144, e=256, i=128, fs=128, topk=8)
# h / 16 = 256: two lane groups per chunk, so the fold path exists (8 rows).
# Few experts keep interpret mode fast; 8 active experts still refill the
# ring (6 slots).
SMALL = dict(h=4096, e=8, i=128, fs=128, topk=4)
ROUTED_SCALE = 2.5
# (rows, n_valid) called directly: zero and full for 16 rows; fold, trim and
# full for 32; fold (8, the most rows that fold), trim (16) and full with
# padding (33, the fewest tokens only this bucket serves) for 64. Zero and
# unpadded full run the same code at every row count, so 64 rows skip them
# (the worker runs under a timeout); arming's all-reduce self-test runs both
# at 64 rows on the device. The 16-row fold path runs through the adapter's
# op body (BODY_CASE), which derives n_valid from a padding mask.
CASES = (
    (16, 0),
    (16, 16),
    (32, 1),
    (32, 12),
    (32, 32),
    (64, 8),
    (64, 16),
    (64, 33),
)
BODY_CASE = (16, 7)
# A token count the kernel does not admit: the op body's psum path.
CHAIN_ROWS = 128
# Kernel vs an f32 reference with dequantized weights and unquantized
# activations, i.e. the kernel's FP8 activations and bf16 roundings: measured
# rel-L2 0.044 to 0.046 and cos >= 0.9989 on these inputs.
TOLERANCE = dict(rel_l2=0.06, cos=0.998)


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _shapes(m, h, e, i, fs, *, w_dtype=jnp.float8_e4m3fn, score_experts=None):
    sds = jax.ShapeDtypeStruct
    f32 = jnp.float32
    return (
        sds((m, h), jnp.bfloat16),
        sds((m, score_experts or e), f32),
        sds((score_experts or e,), f32),
        sds((e, h, 2 * i), w_dtype),
        sds((e, i, h), w_dtype),
        sds((e, 1, 1, 2 * i), f32),
        sds((e, 1, 1, h), f32),
        sds((h, 2 * fs), w_dtype),
        sds((2 * fs,), f32),
        sds((fs, h), w_dtype),
        sds((h,), f32),
    )


def _glm52_shapes(rows, **kwargs):
    c = GLM52_TP16
    return _shapes(rows, c["h"], c["e"], c["i"], c["fs"], **kwargs)


@pytest.mark.parametrize(
    "rows, admitted",
    [(16, True), (32, True), (64, True), (8, False), (48, False), (128, False)],
)
def test_admits_the_decode_buckets_at_glm52_shapes(monkeypatch, rows, admitted):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    shapes = _glm52_shapes(rows)
    assert kernel.can_use_moe_tp_decode(*shapes, topk=8, n_ranks=N_RANKS) is admitted


@pytest.mark.parametrize(
    "info, shapes, n_ranks",
    [
        pytest.param(
            lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1),
            _glm52_shapes(16),
            16,
            id="no-fp8-mxu",
        ),
        pytest.param(
            _tpu7x, _glm52_shapes(16, w_dtype=jnp.bfloat16), 16, id="bf16-weights"
        ),
        pytest.param(
            _tpu7x, _glm52_shapes(16, score_experts=128), 16, id="not-every-expert"
        ),
        pytest.param(_tpu7x, _glm52_shapes(16), 1, id="one-rank"),
        pytest.param(_tpu7x, _glm52_shapes(16), 32, id="chunk-not-lane-aligned"),
    ],
)
def test_refuses(monkeypatch, info, shapes, n_ranks):
    monkeypatch.setattr(pltpu, "get_tpu_info", info)
    assert not kernel.can_use_moe_tp_decode(*shapes, topk=8, n_ranks=n_ranks)


def test_row_paths_at_glm52_shapes(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    h = GLM52_TP16["h"]
    # 6144 / 16 = 3 lane groups per chunk, so up to 16 // 3 = 5 rows fold.
    assert all_reduce.selftest_row_counts(all_reduce.make_spec(16, 16, h)) == [16, 5, 0]
    assert all_reduce.selftest_row_counts(all_reduce.make_spec(16, 32, h)) == [
        32,
        5,
        16,
        0,
    ]
    assert all_reduce.selftest_row_counts(all_reduce.make_spec(16, 64, h)) == [
        64,
        5,
        16,
        0,
    ]


def test_selftest_inputs_have_exact_sums_that_expose_a_wrong_chunk(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    spec = all_reduce.make_spec(N_RANKS, 16, GLM52_TP16["h"])
    x = all_reduce.selftest_inputs(spec)
    total = x.sum(0)
    assert np.array_equal(total.astype(jnp.bfloat16).astype(np.float32), total)
    assert all_reduce.selftest_ok(spec, x, total, 16)
    assert not all_reduce.selftest_ok(spec, x, np.roll(total, 128, axis=1), 16)
    padded = total.copy()
    padded[3:] = -0.0
    assert not all_reduce.selftest_ok(spec, x, padded, 3)


@pytest.fixture(scope="module")
def on_16_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}",
    )
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


def test_race_detector_is_live(on_16_devices):
    assert on_16_devices["race_control"] == {"racy": True, "clean": False}


@pytest.mark.parametrize("rows, n_valid", CASES)
def test_kernel_on_16_devices(on_16_devices, rows, n_valid):
    case = on_16_devices["cases"][f"{rows}:{n_valid}"]
    assert case["ranks_identical"], case
    assert case["padding_rows_positive_zero"], case
    assert not case["races_found"], case
    if n_valid:
        assert case["nan"] == 0, case
        assert case["rel_l2"] <= TOLERANCE["rel_l2"], case
        assert case["cos"] >= TOLERANCE["cos"], case


def test_op_body_runs_the_kernel_with_n_valid_from_the_padding_mask(on_16_devices):
    rows, n_valid = BODY_CASE
    case = on_16_devices["body_kernel"]
    assert case["ranks_identical"], case
    assert case["padding_rows_positive_zero"], case
    assert not case["races_found"], case
    assert case["nan"] == 0, case
    assert case["rel_l2"] <= TOLERANCE["rel_l2"], case
    assert case["cos"] >= TOLERANCE["cos"], case


def test_op_body_psums_the_usual_path_at_other_token_counts(on_16_devices):
    # The usual path is stubbed with ones on real rows: the psum over 16 ranks
    # gives exactly 16 there and 0 on padded rows.
    assert on_16_devices["body_chain_exact"]


def test_all_reduce_selftest_on_16_devices(on_16_devices):
    # 32 rows: one call per row path (full, fold, trim, zero).
    assert on_16_devices["selftest_ok"]


# ---------------------------------------------------------------------------
# The 16-device worker (this file run as a script).
# ---------------------------------------------------------------------------
def _fp8_columns(w):
    """Per-output-channel FP8 of [..., K, N]: (fp8, f32 scale [..., N])."""
    f8 = jnp.float8_e4m3fn
    s = jnp.max(jnp.abs(w), axis=-2) / float(jnp.finfo(f8).max)
    s = jnp.where(s == 0, 1.0, s)
    return (w / s[..., None, :]).astype(f8), s


def _rank_weights(key, c):
    h, e, i, fs = c["h"], c["e"], c["i"], c["fs"]
    k13, k2, ks13, ks2 = jax.random.split(key, 4)
    f32 = jnp.float32
    w13, s13 = _fp8_columns(jax.random.normal(k13, (e, h, 2 * i), f32) / math.sqrt(h))
    w2, s2 = _fp8_columns(jax.random.normal(k2, (e, i, h), f32) / math.sqrt(16 * i))
    sw13, ss13 = _fp8_columns(jax.random.normal(ks13, (h, 2 * fs), f32) / math.sqrt(h))
    sw2, ss2 = _fp8_columns(jax.random.normal(ks2, (fs, h), f32) / math.sqrt(16 * fs))
    return (
        w13,
        w2,
        s13.reshape(e, 1, 1, 2 * i),
        s2.reshape(e, 1, 1, h),
        sw13,
        ss13,
        sw2,
        ss2,
    )


def _reference(x, scores, bias, n_valid, ranks, c):
    """f32 sum over ranks of shared + scale * routed, dequantized weights."""
    f32 = jnp.float32
    x = x.astype(f32)
    _, ids = jax.lax.top_k(scores + bias, c["topk"])
    w = jnp.take_along_axis(scores, ids, axis=-1)
    w = w / jnp.sum(w, axis=-1, keepdims=True)
    coef = jnp.sum(jax.nn.one_hot(ids, c["e"], dtype=f32) * w[..., None], axis=1)
    out = jnp.zeros(x.shape, f32)
    for w13, w2, s13, s2, sw13, ss13, sw2, ss2 in ranks:
        gu = jnp.einsum("nh,ehk->enk", x, w13.astype(f32) * s13[:, 0])
        act = jax.nn.silu(gu[..., : c["i"]]) * gu[..., c["i"] :]
        y = jnp.einsum("eni,eih->enh", act, w2.astype(f32) * s2[:, 0])
        routed = jnp.einsum("ne,enh->nh", coef, y)
        sgu = x @ (sw13.astype(f32) * ss13)
        sact = jax.nn.silu(sgu[:, : c["fs"]]) * sgu[:, c["fs"] :]
        out = out + sact @ (sw2.astype(f32) * ss2) + ROUTED_SCALE * routed
    return np.asarray(out[:n_valid])


def _interpret(fn, *args):
    """fn(*args) with fresh interpreter state; (out, whether races were found)."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    pltpu.reset_tpu_interpret_mode_state()
    with contextlib.redirect_stdout(io.StringIO()):
        out = np.asarray(jax.block_until_ready(fn(*args)))
    return out, bool(interpret_pallas_call.races.races_found)


def _race_control():
    """The detector on a kernel that reads a DMA target before its wait."""
    from jax.experimental import pallas as pl

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

    x = jnp.ones((8, 128), jnp.float32)
    return {
        "racy": _interpret(make(True), x)[1],
        "clean": _interpret(make(False), x)[1],
    }


def _worker():
    from jax._src.pallas.mosaic import tpu_info
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    pltpu.get_tpu_info = _tpu7x
    tpu_info.get_tpu_info = _tpu7x
    c = SMALL
    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == N_RANKS, devices
    mesh = Mesh(np.asarray(devices), ("d",))
    keys = jax.random.split(jax.random.key(0), N_RANKS + 3)
    ranks = [_rank_weights(keys[r], c) for r in range(N_RANKS)]
    stacked = [jnp.concatenate(ws, axis=0) for ws in zip(*ranks, strict=True)]
    gate = jax.random.normal(keys[-1], (c["e"], c["h"])) / math.sqrt(c["h"])
    bias = jax.random.uniform(keys[-2], (c["e"],), jnp.float32, -0.05, 0.05)
    rank_ids = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)
    specs = (P("d"), P("d"), P(), P(), P("d")) + (P("d"),) * 8

    def body(x, scores, b, n_valid, rank, *weights):
        return kernel.moe_tp_decode(
            x,
            scores,
            b,
            n_valid[0],
            rank.reshape(()),
            *weights,
            routed_scale=ROUTED_SCALE,
            topk=c["topk"],
            n_ranks=N_RANKS,
        )

    run = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
        )
    )

    def inputs(rows):
        x = jax.random.normal(keys[-3], (rows, c["h"])).astype(jnp.bfloat16)
        return x, jax.nn.sigmoid(x.astype(jnp.float32) @ gate.T)

    def checks(out, x, scores, rows, n_valid, races, start):
        out = out.reshape(N_RANKS, rows, c["h"]).astype(np.float32)
        bits = out.view(np.uint32)
        pad = out[:, n_valid:]
        case = dict(
            seconds=round(time.monotonic() - start, 1),
            ranks_identical=bool((bits == bits[0]).all()),
            padding_rows_positive_zero=bool(np.all((pad == 0) & ~np.signbit(pad))),
            races_found=races,
            nan=int(np.isnan(out).sum()),
        )
        if n_valid:
            got = out[0, :n_valid].astype(np.float64)
            want = _reference(x, scores, bias, n_valid, ranks, c).astype(np.float64)
            case.update(
                rel_l2=float(np.linalg.norm(got - want) / np.linalg.norm(want)),
                cos=float(
                    np.vdot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want))
                ),
                max_abs=float(np.max(np.abs(got - want))),
            )
        return case

    def put(operands, in_specs):
        return [
            jax.device_put(a, NamedSharding(mesh, s))
            for a, s in zip(operands, in_specs, strict=True)
        ]

    result = {"race_control": _race_control(), "cases": {}}
    for rows, n_valid in CASES:
        x, scores = inputs(rows)
        operands = (
            jnp.tile(x, (N_RANKS, 1)),
            jnp.tile(scores, (N_RANKS, 1)),
            bias,
            jnp.asarray([n_valid], jnp.int32),
            rank_ids,
            *stacked,
        )
        start = time.monotonic()
        out, races = _interpret(run, *put(operands, specs))
        result["cases"][f"{rows}:{n_valid}"] = checks(
            out, x, scores, rows, n_valid, races, start
        )

    # The adapter's op body: the operands moe_tp_decode_forward passes, with a
    # padding mask longer than the token count, as the runner's is.
    from vllm_torchtpu.layers.adapter import moe_tp_decode as adapter

    statics = adapter._OpStatics(
        topk=c["topk"],
        routed_scale=ROUTED_SCALE,
        activation="silu",
        use_sparse_core=False,
        onehot_threshold=0,
    )
    body_specs = (P("d"), P("d"), P(), P()) + (P("d"),) * 9
    run_body = jax.jit(
        jax.shard_map(
            adapter._op_body(statics, N_RANKS, "d"),
            mesh=mesh,
            in_specs=body_specs,
            out_specs=P("d"),
            check_vma=False,
        )
    )

    def body_operands(x, scores, n_valid):
        pad_mask = jnp.arange(CHAIN_ROWS) >= n_valid
        return put(
            (
                jnp.tile(x, (N_RANKS, 1)),
                jnp.tile(scores, (N_RANKS, 1)),
                bias,
                pad_mask,
                *stacked,
                rank_ids,
            ),
            body_specs,
        )

    rows, n_valid = BODY_CASE
    x, scores = inputs(rows)
    start = time.monotonic()
    out, races = _interpret(run_body, *body_operands(x, scores, n_valid))
    result["body_kernel"] = checks(out, x, scores, rows, n_valid, races, start)

    def ones_on_real_rows(x, scores, bias, is_pad, *weights_and_statics):
        return jnp.where(is_pad[:, None], 0.0, 1.0).astype(x.dtype) * jnp.ones_like(x)

    adapter._tp_chain = ones_on_real_rows
    n_valid = CHAIN_ROWS - 5
    x, scores = inputs(CHAIN_ROWS)
    # No Pallas call at this token count, so no interpreter state to reset.
    out = np.asarray(
        jax.block_until_ready(run_body(*body_operands(x, scores, n_valid)))
    )
    out = out.reshape(N_RANKS, CHAIN_ROWS, c["h"]).astype(np.float32)
    want = np.zeros((CHAIN_ROWS, c["h"]), np.float32)
    want[:n_valid] = N_RANKS
    result["body_chain_exact"] = bool(
        np.array_equal(out, np.broadcast_to(want, out.shape))
    )
    spec = all_reduce.make_spec(N_RANKS, 32, c["h"])
    call = all_reduce.standalone(spec)
    partials = all_reduce.selftest_inputs(spec)

    def selftest(x, rank, n_valid):
        return call(jnp.stack([n_valid.reshape(()), rank.reshape(())]), x)

    run_selftest = jax.jit(
        jax.shard_map(
            selftest,
            mesh=mesh,
            in_specs=(P("d"), P("d"), P()),
            out_specs=P("d"),
            check_vma=False,
        )
    )
    ok = True
    for n_valid in all_reduce.selftest_row_counts(spec):
        out, races = _interpret(
            run_selftest,
            jnp.asarray(partials.reshape(N_RANKS * 32, -1), jnp.bfloat16),
            rank_ids,
            jnp.asarray([n_valid], jnp.int32),
        )
        out = out.reshape(N_RANKS, 32, -1)
        ok &= not races and all(
            all_reduce.selftest_ok(spec, partials, out[r], n_valid)
            for r in range(N_RANKS)
        )
    result["selftest_ok"] = bool(ok)
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    # Around everything: pallas_call reads the mode when it is built.
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
