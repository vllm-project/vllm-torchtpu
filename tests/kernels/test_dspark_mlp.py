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
"""Fused decode MLP kernel vs the torch path's math (`reference`), and its
all-reduce epilogue on 8 CPU devices in interpret mode with the race detector
(a subprocess, since the device count is fixed when JAX starts).

Interpret mode runs the kernel body on the host; it does not gate Mosaic
lowering.
"""

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels import dspark_mlp as kernel


@pytest.fixture(autouse=True)
def _tpu7x():
    """get_tpu_info() on the host answers for the abstract device."""
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        yield


def _operands(m: int, h: int, i: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    fp8 = jnp.float8_e4m3fn
    x = jnp.asarray(rng.standard_normal((m, h)), jnp.bfloat16)
    w_gu = jnp.asarray(rng.standard_normal((h, 2 * i)) * 4, fp8)
    w_d = jnp.asarray(rng.standard_normal((i, h)) * 4, fp8)
    # Per-channel powers of two, as requantize_mxfp8_per_channel makes them.
    s_gu = jnp.asarray(2.0 ** rng.integers(-9, -6, (1, 2 * i)), jnp.float32)
    s_d = jnp.asarray(2.0 ** rng.integers(-9, -6, (1, h)), jnp.float32)
    return x, w_gu, s_gu, w_d, s_d


def _err(got, want) -> float:
    """Max abs difference over the reference's max abs value."""
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return float(np.abs(got - want).max() / np.abs(want).max())


# Rows padded to a sublane tile (7, 21), a half bf16 tile, a tile, the cap; a
# multi-chunk K (h = 2048 rows at 2i = 2048: 3 gate_up and 2 down chunks).
@pytest.mark.parametrize("m", [7, 8, 16, 21, 32])
@pytest.mark.parametrize("h,i", [(256, 128), (2048, 1024)])
def test_matches_reference(m: int, h: int, i: int) -> None:
    ops = _operands(m, h, i, seed=m + h)
    got = kernel.fused_mlp(*ops, interpret=True)
    want = kernel.reference(*ops)
    assert got.shape == (m, h) and got.dtype == jnp.bfloat16
    # Summation order only: a few bf16 ulps of the output's scale.
    assert _err(got, want) < 2e-2


def _norm(m: int, h: int, seed: int):
    rng = np.random.default_rng(seed)
    residual = jnp.asarray(rng.standard_normal((m, h)) * 4, jnp.bfloat16)
    weight = jnp.asarray(1 + rng.standard_normal((1, h)) * 0.1, jnp.float32)
    return residual, weight, 1e-6


@pytest.mark.parametrize("m", [8, 21])
def test_norm_prologue(m: int) -> None:
    """The residual add + Gemma RMSNorm in front: the new residual exactly,
    the MLP output to the same bar."""
    ops = _operands(m, 2048, 1024, seed=m)
    norm = _norm(m, 2048, seed=m + 1)
    got, res = kernel.fused_mlp(*ops, interpret=True, norm=norm)
    want, want_res = kernel.reference(*ops, norm=norm)
    assert got.shape == res.shape == (m, 2048)
    np.testing.assert_array_equal(np.asarray(res), np.asarray(want_res))
    assert _err(got, want) < 2e-2
    # Fault: the MLP fed the un-normalized sum.
    assert _err(got, kernel.reference(want_res, *ops[1:])) > 2e-2


def test_rings_wrap(monkeypatch) -> None:
    """128-row chunks: 16 gate_up and 8 down chunks through 7-slot rings."""
    monkeypatch.setattr(kernel, "_CHUNK_BYTES", 1 << 16)
    ops = _operands(16, 2048, 1024, seed=5)
    assert len(kernel._chunks(2048, 2048)) > kernel._SLOTS
    assert len(kernel._chunks(1024, 2048)) > kernel._SLOTS
    got = kernel.fused_mlp(*ops, interpret=True)
    assert _err(got, kernel.reference(*ops)) < 2e-2


def test_faults_are_caught() -> None:
    """The bar above rejects a dropped K chunk and swapped gate / up."""
    m, h, i = 16, 2048, 1024
    x, w_gu, s_gu, w_d, s_d = _operands(m, h, i, seed=3)
    got = kernel.fused_mlp(x, w_gu, s_gu, w_d, s_d, interpret=True)
    dropped = w_gu.at[-128:].set(0)
    swapped = jnp.concatenate([w_gu[:, i:], w_gu[:, :i]], 1)
    for w in (dropped, swapped):
        assert _err(got, kernel.reference(x, w, s_gu, w_d, s_d)) > 2e-2


def test_supported_rows() -> None:
    assert kernel.supported(32, 6144, 1536)
    assert not kernel.supported(40, 6144, 1536)
    assert kernel.supported(7, 6144, 1536)
    assert not kernel.supported(32, 6144, 1000)


N_RANKS = 8
# h / n = 256 lanes: rows 2 and 8 take the epilogue's fold path, 16 trim (of
# 32 padded), 21 and 32 the full path.
AR_ROWS = (2, 8, 16, 21, 32)
AR_H, AR_I = 2048, 128


def _ar_worker():
    import jax
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    pltpu.get_tpu_info = tpu_info.get_tpu_info = tpu7x
    mesh = Mesh(np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id)), ("d",))
    ranks = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)
    run = jax.jit(
        jax.shard_map(
            lambda *a: kernel.fused_mlp(*a[:5], rank=a[5], n=N_RANKS),
            mesh=mesh,
            in_specs=(P("d"),) * 6,
            out_specs=P("d"),
            check_vma=False,
        )
    )
    per_rank = [_operands(32, AR_H, AR_I, seed=10 + r) for r in range(N_RANKS)]
    weights = [jnp.concatenate([ops[k] for ops in per_rank]) for k in range(1, 5)]
    result = {}
    for rows in AR_ROWS:
        xs = [ops[0][:rows] for ops in per_rank]
        partials = [
            np.asarray(kernel.fused_mlp(x, *ops[1:], interpret=True), np.float32)
            for x, ops in zip(xs, per_rank)
        ]
        want = np.zeros_like(partials[0])
        for p in partials:  # f32 in rank order, rounded once
            want = want + p
        want = np.asarray(jnp.asarray(want, jnp.bfloat16), np.float32)
        pltpu.reset_tpu_interpret_mode_state()
        out = run(jnp.concatenate(xs), *weights, ranks)
        out = np.asarray(out, np.float32).reshape(N_RANKS, rows, AR_H)
        result[str(rows)] = dict(
            exact=all(np.array_equal(out[r], want) for r in range(N_RANKS)),
            races=bool(interpret_pallas_call.races.races_found),
        )
    # The norm prologue in front of the all-reduce (24 rows: the full path).
    rows = 24
    norms = [_norm(rows, AR_H, seed=40 + r) for r in range(N_RANKS)]
    run_norm = jax.jit(
        jax.shard_map(
            lambda x, res, w, *a: kernel.fused_mlp(
                x, *a[:4], rank=a[4], n=N_RANKS, norm=(res, w, 1e-6)
            ),
            mesh=mesh,
            in_specs=(P("d"),) * 8,
            out_specs=(P("d"), P("d")),
            check_vma=False,
        )
    )
    xs = [ops[0][:rows] for ops in per_rank]
    local = [
        kernel.fused_mlp(x, *ops[1:], interpret=True, norm=nm)
        for x, ops, nm in zip(xs, per_rank, norms)
    ]
    want = np.zeros((rows, AR_H), np.float32)
    for out, _ in local:
        want = want + np.asarray(out, np.float32)
    want = np.asarray(jnp.asarray(want, jnp.bfloat16), np.float32)
    pltpu.reset_tpu_interpret_mode_state()
    out, res = run_norm(
        jnp.concatenate(xs),
        jnp.concatenate([nm[0] for nm in norms]),
        jnp.concatenate([nm[1] for nm in norms]),
        *weights,
        ranks,
    )
    out = np.asarray(out, np.float32).reshape(N_RANKS, rows, AR_H)
    res = np.asarray(res, np.float32).reshape(N_RANKS, rows, AR_H)
    result["norm"] = dict(
        exact=all(np.array_equal(out[r], want) for r in range(N_RANKS))
        and all(
            np.array_equal(res[r], np.asarray(local[r][1], np.float32))
            for r in range(N_RANKS)
        ),
        races=bool(interpret_pallas_call.races.races_found),
    )
    # pre_ar: x is a partial summed in the kernel ahead of the norm and MLP;
    # every rank gets the same reduced x and residual, its own weights.
    for rows in (8, 21, 24):
        partials = [ops[0][:rows] for ops in per_rank]
        residual, nw, eps = _norm(rows, AR_H, seed=60 + rows)
        x_sum = np.zeros((rows, AR_H), np.float32)
        for p in partials:
            x_sum = x_sum + np.asarray(p, np.float32)
        x_red = jnp.asarray(x_sum, jnp.bfloat16)
        local = [
            kernel.fused_mlp(x_red, *ops[1:], interpret=True, norm=(residual, nw, eps))
            for ops in per_rank
        ]
        run_pre = jax.jit(
            jax.shard_map(
                lambda x, res, w, *a: kernel.fused_mlp(
                    x, *a[:4], rank=a[4], n=N_RANKS, norm=(res, w, 1e-6), pre_ar=True
                ),
                mesh=mesh,
                in_specs=(P("d"),) * 8,
                out_specs=(P("d"), P("d")),
                check_vma=False,
            )
        )
        pltpu.reset_tpu_interpret_mode_state()
        out, res = run_pre(
            jnp.concatenate(partials),
            jnp.concatenate([residual] * N_RANKS),
            jnp.concatenate([nw] * N_RANKS),
            *weights,
            ranks,
        )
        out = np.asarray(out, np.float32).reshape(N_RANKS, rows, AR_H)
        res = np.asarray(res, np.float32).reshape(N_RANKS, rows, AR_H)
        result[f"pre{rows}"] = dict(
            exact=all(
                np.array_equal(out[r], np.asarray(local[r][0], np.float32))
                and np.array_equal(res[r], np.asarray(local[r][1], np.float32))
                for r in range(N_RANKS)
            ),
            races=bool(interpret_pallas_call.races.races_found),
        )
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}",
    )
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=840
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("rows", [*AR_ROWS, "norm", "pre8", "pre21", "pre24"])
def test_all_reduce_epilogue_on_8_devices(on_8_devices, rows):
    case = on_8_devices[str(rows)]
    assert case["exact"] and not case["races"], case


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _ar_worker()
