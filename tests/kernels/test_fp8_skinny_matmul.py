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
"""fp8_skinny: the decode (M <= 16) weight-streaming FP8 matmul (TPU_FP8_LINEAR_PALLAS).

Judged against the float64 product of the SAME quantized operands (x_q, x_scale from
`util.quantize_tensor`, the per-channel weight and scale). The tolerance is set by the
XLA reference's own error on the cell: XLA's fp8 x fp8 dot accumulates below f32 and
this kernel does not, so the kernel must land inside 2x the reference's error.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny, util

F8 = jnp.float8_e4m3fn


def _x(m, k, real, seed=0):
    rng = np.random.default_rng(seed)
    gain = np.exp(rng.normal(0, 0.5, k)).astype(np.float32)
    gain[rng.choice(k, 8, replace=False)] *= 20.0
    x = rng.standard_normal((m, k), dtype=np.float32) * gain[None]
    x[real:] = 0.0
    return jnp.asarray(x, jnp.bfloat16)


def _w(k, n, seed=1):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((k, n), dtype=np.float32) * 0.02
    ws = np.abs(w).max(axis=0) / 448.0
    wq = np.clip(w / ws[None], -448, 448)
    return jnp.asarray(wq, F8), jnp.asarray(ws, jnp.float32)


def _oracle(x, wq, ws):
    xq, xs = util.quantize_tensor(x, F8)
    a = np.asarray(xq).astype(np.float64)
    a = a * np.asarray(xs).astype(np.float64).reshape(-1, 1)
    b = np.asarray(wq).astype(np.float64) * np.asarray(ws).astype(np.float64)[None]
    return a @ b


def _row_err(y, ref):
    y = np.asarray(y).astype(np.float64)
    m = np.abs(ref).max(axis=1)
    d = np.abs(y - ref).max(axis=1)
    return float(np.where(m > 0, d / np.where(m > 0, m, 1), d).max())


def _tol(x, wq, ws, ref):
    y_xla = jax.jit(util.xla_quantized_matmul)(x, wq, ws)
    return max(2 * _row_err(y_xla, ref), 2.0**-8)


# GLM-5.2-FP8 per-rank decode sites (TP16, q-replicate) at their tuned tilings, other
# tilings, and the generic picker.
SITES = [
    (2048, 16384, None),
    (1024, 6144, None),
    (6144, 1536, None),
    (6144, 2688, None),
    (1024, 6144, ("n", 512, 2)),
    (6144, 1536, ("k", 768, 4)),
    (6144, 2688, ("n", 896, 3)),
    (512, 640, None),
]


@pytest.mark.tpu_test
@pytest.mark.parametrize("k,n,config", SITES)
@pytest.mark.parametrize("m,real", [(16, 4), (16, 16), (8, 8)])
def test_matches_f64_of_quantized_operands(k, n, config, m, real):
    x = _x(m, k, real)
    wq, ws = _w(k, n)
    ref = _oracle(x, wq, ws)

    def f(a, b, c):
        return fp8_skinny.fp8_skinny_matmul(a, b, c, config=config)

    y = jax.jit(f)(x, wq, ws)
    assert y.shape == (m, n) and y.dtype == x.dtype
    y32 = np.asarray(y, np.float32)
    assert np.isfinite(y32).all()
    assert np.all(y32[real:] == 0), "pad rows must stay exactly zero"
    assert _row_err(y, ref) <= _tol(x, wq, ws, ref)


@pytest.mark.tpu_test
@pytest.mark.parametrize("config", [("n", 512, 2), ("k", 256, 3)])
@pytest.mark.parametrize("edge", ["none", "zero_row", "outlier", "tiny"])
def test_quantize_bit_identical_to_quantize_block(config, edge):
    """Through W = I the output is bf16(x_q * x_scale): the in-kernel quantize must
    equal util.quantize_tensor bit for bit."""
    k = 1024
    x = np.asarray(_x(16, k, 16), np.float32)
    if edge == "zero_row":
        x[3] = 0.0
    elif edge == "outlier":
        x[0, 5] = 3.0e4
        x[1] = -np.abs(x[1])
    elif edge == "tiny":
        x[2] *= 1e-30
    x = jnp.asarray(x, jnp.bfloat16)
    eye = jnp.asarray(np.eye(k, dtype=np.float32), F8)
    ones = jnp.ones((k,), jnp.float32)
    y = jax.jit(lambda a: fp8_skinny.fp8_skinny_matmul(a, eye, ones, config=config))(x)
    xq, xs = util.quantize_tensor(x, F8)
    ref = (xq.astype(jnp.float32) * xs.reshape(-1, 1)).astype(jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(y, np.float32), np.asarray(ref, np.float32)
    )


@pytest.mark.tpu_test
def test_padded_copy_and_logical_width():
    """q_a + kv_a: 2624 outputs; the kernel runs on a copy padded to 2688 at load and
    the cut output matches the unpadded math."""
    k, n = 6144, 2624
    n_pad = fp8_skinny.padded_n(n)
    assert n_pad == 2688
    x = _x(16, k, 4)
    wq, ws = _w(k, n)
    wq_p = jnp.pad(wq.astype(jnp.float32), ((0, 0), (0, n_pad - n))).astype(F8)
    ws_p = jnp.pad(ws, (0, n_pad - n), constant_values=1.0)

    def f(a, b, c):
        return fp8_skinny.fp8_skinny_matmul(a, b, c, n_logical=n)

    y = jax.jit(f)(x, wq_p, ws_p)
    assert y.shape == (16, n)
    ref = _oracle(x, wq, ws)
    assert _row_err(y, ref) <= _tol(x, wq, ws, ref)


@pytest.mark.tpu_test
@pytest.mark.parametrize("config", [("n", 1024, 3), ("k", 256, 3)])
def test_stacked_weight_index(config):
    """The kernel reads w_stack[idx] itself (the benchmark's cold-weight form): each
    slice gives the same result as passing that weight alone."""
    k, n = 1024, 3072
    x = _x(16, k, 16)
    ws_list = [_w(k, n, seed=s) for s in (3, 4, 5)]
    stack = jnp.stack([w for w, _ in ws_list])
    for i, (wq, ws) in enumerate(ws_list):
        idx = jnp.array([i], jnp.int32)
        y = fp8_skinny.fp8_skinny_matmul_stacked(x, stack, idx, ws, config=config)
        y1 = fp8_skinny.fp8_skinny_matmul(x, wq, ws, config=config)
        np.testing.assert_array_equal(
            np.asarray(y, np.float32), np.asarray(y1, np.float32)
        )


def test_supports():
    wq, ws = _w(1024, 6144)
    assert fp8_skinny.supports(16, wq, ws)
    assert not fp8_skinny.supports(17, wq, ws)
    w2, s2 = _w(512, 640)  # correct, but untuned: stays on XLA
    assert not fp8_skinny.supports(16, w2, s2)
    assert not fp8_skinny.supports(16, wq[:, :2624], ws[:2624])  # pad N at load first
    assert not fp8_skinny.supports(16, wq, ws.reshape(1, 1, 1, -1))  # blockwise scale
    assert not fp8_skinny.supports(16, wq.astype(jnp.bfloat16), ws)


# --------------------------------------------------------------------------------
# scale_dma (TPU_FP8_SKINNY_SCALE_DMA): the scale passed in HBM, DMA'd in-kernel.
# --------------------------------------------------------------------------------

SCALE_DMA_SITES = [
    (2048, 16384, None),
    (2048, 8192, None),
    (1024, 6144, None),
    (6144, 1536, None),
    (6144, 2688, None),
    (1024, 6144, ("n", 512, 2)),
    (6144, 1536, ("k", 768, 4)),
    (512, 640, ("n", 640, 1)),
]


@pytest.mark.tpu_test
@pytest.mark.parametrize("k,n,config", SCALE_DMA_SITES)
@pytest.mark.parametrize("m,real", [(16, 4), (16, 16), (8, 8)])
def test_scale_dma_is_bit_identical(k, n, config, m, real):
    x = _x(m, k, real)
    wq, ws = _w(k, n)

    def f(sd):
        return jax.jit(
            lambda a, b, c: fp8_skinny.fp8_skinny_matmul(
                a, b, c, config=config, scale_dma=sd
            )
        )(x, wq, ws)

    y0, y1 = f(False), f(True)
    np.testing.assert_array_equal(
        np.asarray(y1, np.float32), np.asarray(y0, np.float32)
    )
    # Positive control: the result depends on every scale column, so a scale the
    # kernel never received (or read before its DMA landed) cannot pass the above.
    ws2 = ws.at[n // 2].multiply(2.0)
    y2 = jax.jit(
        lambda a, b, c: fp8_skinny.fp8_skinny_matmul(
            a, b, c, config=config, scale_dma=True
        )
    )(x, wq, ws2)
    assert not np.array_equal(np.asarray(y2, np.float32), np.asarray(y1, np.float32))


@pytest.mark.tpu_test
def test_scale_dma_stacked_eager_bit_identical():
    """The constraint cannot run eagerly; the scale_dma path jits itself."""
    k, n = 1024, 3072
    x = _x(16, k, 16)
    ws_list = [_w(k, n, seed=s) for s in (3, 4)]
    stack = jnp.stack([w for w, _ in ws_list])
    for i, (_, ws) in enumerate(ws_list):
        idx = jnp.array([i], jnp.int32)
        y0 = fp8_skinny.fp8_skinny_matmul_stacked(x, stack, idx, ws)
        y1 = fp8_skinny.fp8_skinny_matmul_stacked(x, stack, idx, ws, scale_dma=True)
        np.testing.assert_array_equal(
            np.asarray(y0, np.float32), np.asarray(y1, np.float32)
        )


def _scale_copy_lines(hlo: str, n: int) -> list[str]:
    """Compiled-HLO instructions that copy or slice an f32[n] / f32[1,n] buffer (the
    scale): the defining op is a copy/slice start and its result is that shape."""
    pat = re.compile(rf"=\s*\(f32\[(?:1,)?{n}\][^=]*\)\s+(?:copy|slice)-start\(")
    return [ln.strip()[:200] for ln in hlo.splitlines() if pat.search(ln)]


@pytest.mark.tpu_test
@pytest.mark.parametrize("k,n", [(1024, 6144), (2048, 16384), (6144, 1536)])
def test_scale_dma_lowers_scale_to_hbm(k, n):
    """scale_dma: the custom call carries `input_memory_space_colors` and the compiled
    module has no copy on the scale buffer. Positive control: the default lowering
    carries no colors."""
    x = _x(16, k, 4)
    wq, ws = _w(k, n)

    def lowered(sd):
        return jax.jit(
            lambda a, b, c: fp8_skinny.fp8_skinny_matmul(a, b, c, scale_dma=sd)
        ).lower(x, wq, ws)

    on, off = lowered(True), lowered(False)
    assert "input_memory_space_colors" in on.as_text()
    assert "input_memory_space_colors" not in off.as_text()
    off_lines = _scale_copy_lines(off.compile().as_text(), n)
    print(f"default scale copies ({k},{n}): {off_lines}")
    assert _scale_copy_lines(on.compile().as_text(), n) == []


def test_scale_dma_applies():
    assert fp8_skinny.scale_dma_applies(1024, 6144)  # o_proj
    assert not fp8_skinny.scale_dma_applies(2048, 16384)  # q_b: copy hidden
    assert not fp8_skinny.scale_dma_applies(6144, 1536)  # gate_up: not a bitcast
    assert not fp8_skinny.scale_dma_applies(6144, 2688)  # q_a + kv_a
    for k, n in fp8_skinny.SCALE_DMA_SITES:
        assert (k, n) in fp8_skinny.TUNED and n % 1024 == 0


# --------------------------------------------------------------------------------
# hbm_pin (TPU_FP8_SKINNY_HBM_PIN_SITES): the weight operand constrained to HBM.
# --------------------------------------------------------------------------------

PIN_SITES = [(2048, 16384), (2048, 8192), (1024, 6144), (6144, 1536), (6144, 2688)]


def _weight_vmem_lines(hlo: str, k: int, n: int) -> list[str]:
    """Compiled-HLO instructions that stage the [1, k, n] fp8 weight into VMEM:
    an instruction whose own result is that shape and is either an async
    copy/slice or laid out in memory space 1 (`S(1)`). The custom call itself
    (a bf16 result) is not one of them."""
    shape = f"f8e4m3fn[1,{k},{n}]"
    out = []
    for ln in hlo.splitlines():
        m = re.match(r"\s*(?:ROOT )?%?\S+ = (.+?) ([a-z][\w-]*)\(", ln)
        if not m:
            continue
        result, opcode = m.group(1), m.group(2)  # result: a shape or a tuple
        mine = [t for t in result.strip("()").split(", ") if t.startswith(shape)]
        if mine and (
            opcode in ("copy-start", "copy-done", "slice-start", "slice-done")
            or any("S(1)" in t for t in mine)
        ):
            out.append(ln.strip())
    return out


@pytest.mark.tpu_test
@pytest.mark.parametrize("k,n", PIN_SITES)
@pytest.mark.parametrize("sd", [False, True])
def test_hbm_pin_is_bit_identical(k, n, sd):
    x = _x(16, k, 4)
    wq, ws = _w(k, n)
    sd = sd and fp8_skinny.scale_dma_applies(k, n)

    def f(pin):
        return jax.jit(
            lambda a, b, c: fp8_skinny.fp8_skinny_matmul(
                a, b, c, hbm_pin=pin, scale_dma=sd
            )
        )(x, wq, ws)

    np.testing.assert_array_equal(
        np.asarray(f(True), np.float32), np.asarray(f(False), np.float32)
    )


@pytest.mark.tpu_test
def test_hbm_pin_stacked_eager_bit_identical():
    """The constraint cannot run eagerly; the pinned path jits itself."""
    k, n = 1024, 6144
    x = _x(16, k, 16)
    ws_list = [_w(k, n, seed=s) for s in (3, 4)]
    stack = jnp.stack([w for w, _ in ws_list])
    for i, (_, ws) in enumerate(ws_list):
        idx = jnp.array([i], jnp.int32)
        y0 = fp8_skinny.fp8_skinny_matmul_stacked(x, stack, idx, ws)
        y1 = fp8_skinny.fp8_skinny_matmul_stacked(x, stack, idx, ws, hbm_pin=True)
        y2 = fp8_skinny.fp8_skinny_matmul_stacked(
            x, stack, idx, ws, hbm_pin=True, scale_dma=True
        )
        for y in (y1, y2):
            np.testing.assert_array_equal(
                np.asarray(y0, np.float32), np.asarray(y, np.float32)
            )


# staged: whether XLA stages the unpinned weight in this standalone module (the
# positive control). q_b's 16 MiB is not staged standalone; it is in the model.
@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "k,n,staged", [(1024, 6144, True), (2048, 8192, True), (2048, 16384, False)]
)
def test_hbm_pin_lowers_to_hbm_color_and_no_vmem_staging(k, n, staged):
    """Pinned: the custom call carries `input_memory_space_colors` and the compiled
    module never stages the weight in VMEM (no copy-start / copy-done / slice-start
    and no S(1) result of its shape). Positive control: the unpinned module does
    stage it (the in_spec's memory_space alone does not reach XLA)."""
    x = _x(16, k, 4)
    wq, ws = _w(k, n)
    sd = fp8_skinny.scale_dma_applies(k, n)

    def lowered(pin):
        return jax.jit(
            lambda a, b, c: fp8_skinny.fp8_skinny_matmul(
                a, b, c, hbm_pin=pin, scale_dma=sd
            )
        ).lower(x, wq, ws)

    pinned, unpinned = lowered(True), lowered(False)
    assert "input_memory_space_colors" in pinned.as_text()
    pinned_hlo = pinned.compile().as_text()
    unpinned_hlo = unpinned.compile().as_text()
    print(f"unpinned VMEM staging of the weight ({k},{n}):")
    for ln in _weight_vmem_lines(unpinned_hlo, k, n):
        print("  ", ln[:200])
    if staged:
        assert _weight_vmem_lines(unpinned_hlo, k, n), "no staging to remove here"
    assert _weight_vmem_lines(pinned_hlo, k, n) == []


@pytest.mark.tpu_test
@pytest.mark.parametrize("m", [4, 8, 16])
def test_window_is_bit_identical_to_the_whole_call(m):
    """q_b's per-rank column windows (TPU_MLA_LOCAL_Q_GATHER): each window's
    output is the whole call's columns bit for bit, and the window index is
    data, so one compiled program serves every rank."""
    k, n, width = 2048, 16384, 1024
    x = _x(m, k, m)
    wq, ws = _w(k, n)
    full = np.asarray(jax.jit(fp8_skinny.fp8_skinny_matmul)(x, wq, ws), np.float32)
    f = jax.jit(
        lambda a, b, c, r: fp8_skinny.fp8_skinny_matmul_window(a, b, c, r, width)
    )
    for r in (0, 5, n // width - 1):
        y = np.asarray(f(x, wq, ws, jnp.array([r], jnp.int32)), np.float32)
        np.testing.assert_array_equal(y, full[:, r * width : (r + 1) * width])


def test_window_supports():
    assert fp8_skinny.window_supports(2048, 16384, 1024)  # q_b at DCP16
    assert not fp8_skinny.window_supports(2048, 16384, 512)  # narrower than a tile
    assert not fp8_skinny.window_supports(6144, 2688, 1344)  # q_a + kv_a tiles K
    assert not fp8_skinny.window_supports(2048, 4096, 1024)  # unmeasured


def test_window_needs_an_n_tiled_config():
    x = _x(16, 6144, 16)
    wq, ws = _w(6144, 2688)  # q_a + kv_a tiles K
    with pytest.raises(ValueError, match="tiles a 1344 window"):
        fp8_skinny.fp8_skinny_matmul_window(x, wq, ws, jnp.zeros((1,), jnp.int32), 1344)
