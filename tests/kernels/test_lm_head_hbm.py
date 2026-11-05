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
"""lm_head_hbm (TPU_LM_HEAD_HBM): the LM head at decode row counts with the weight
streamed from HBM. Judged against the float64 product and against the XLA dot it
replaces (F.linear's lowering): not bit-identical (MXU accumulation order), so every
logit must sit within a few bf16 ulp of XLA's, few of them off, and the error against
f64 must match XLA's own. Each fault-injection variant must fail the same check."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import lm_head_hbm as lh

# Pallas TPU kernel (no CPU interpret mode): off the CPU unit-test lane.
pytestmark = pytest.mark.tpu_test

K = 6144
N = 9680  # GLM-5.2 vocab shard at TP16


def _x(m, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m, K), dtype=np.float32)
    x[:, rng.choice(K, 8, replace=False)] *= 8.0
    return jnp.asarray(x, jnp.bfloat16)


def _w(n, seed=1):
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        rng.standard_normal((n, K), dtype=np.float32) * 0.02, jnp.bfloat16
    )


def _xla(x, w):
    return jax.jit(lambda a, b: jax.lax.dot_general(a, b, (((1,), (1,)), ((), ()))))(
        x, w
    )


def _ulps(a, b):
    a = np.asarray(a).view(np.uint16).astype(np.int32)
    b = np.asarray(b).view(np.uint16).astype(np.int32)
    a = np.where(a & 0x8000, 0x8000 - (a & 0x7FFF), a + 0x8000)
    b = np.where(b & 0x8000, 0x8000 - (b & 0x7FFF), b + 0x8000)
    return np.abs(a - b)


def _judge(y, x, w, tag=""):
    """(ok, info): within 8 ulp of XLA's dot, <= 1e-3 of logits off, and no further
    from the f64 product than XLA (+ one bf16 ulp of the largest logit)."""
    ref = _xla(x, w)
    f64 = np.asarray(x, np.float64) @ np.asarray(w, np.float64).T
    d = _ulps(y, ref)
    y64 = np.asarray(y, np.float64)
    r64 = np.asarray(ref, np.float64)
    # ulps of max(|ref|, 2^-6): near zero a bf16 ulp is tiny, so an unchanged absolute
    # error would read as tens of ulps
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(np.abs(r64), 2.0**-6))) - 7)
    d_rel = np.abs(y64 - r64) / ulp
    err = np.abs(y64 - f64).max()
    ref_err = np.abs(r64 - f64).max()
    ok = bool(
        np.isfinite(y64).all()
        and d_rel.max() <= 8
        and (d > 0).mean() <= 1e-3
        and err <= 1.25 * ref_err + np.abs(f64).max() * 2.0**-8
    )
    info = (
        f"{tag} off={int((d > 0).sum())}/{d.size} "
        f"max_ulp_floored={d_rel.max():.1f} err={err:.3g} ref_err={ref_err:.3g}"
    )
    print("LMHEAD", ok, info)
    return ok, info


@pytest.mark.parametrize("m", [1, 3, 4, 8, 12, 16, 32, 64, 80, 96])
def test_matches_xla_and_f64(m):
    x, w = _x(m, seed=m), _w(N)
    y = lh.lm_head_hbm(x, w)
    assert y.shape == (m, N) and y.dtype == jnp.bfloat16
    ok, info = _judge(y, x, w, f"m={m}")
    assert ok, info
    # first-max argmax agrees with XLA's logits on random data
    ref = np.asarray(_xla(x, w), np.float32)
    assert (np.argmax(np.asarray(y, np.float32), 1) == np.argmax(ref, 1)).all()


@pytest.mark.parametrize(
    "n", [9728, 1040, 464]
)  # no tail tile; a 16-row tail; one short tile
@pytest.mark.parametrize("m", [1, 16])
def test_other_vocab_shards(n, m):
    x, w = _x(m, seed=7 + m), _w(n, seed=n)
    ok, info = _judge(lh.lm_head_hbm(x, w), x, w, f"n={n} m={m}")
    assert ok, info


@pytest.mark.parametrize("variant", ["drop_tail", "off", "stale", "xrow", "kdrop"])
@pytest.mark.parametrize("m", [4, 16])
def test_faults_are_caught(variant, m):
    """Positive control: each injected fault fails the check the kernel passes.
    Fresh inputs per case, so a stale output buffer cannot pass for a right one."""
    x, w = _x(m, seed=100 + m + len(variant)), _w(N, seed=200 + len(variant))
    ok, info = _judge(
        lh.lm_head_hbm(x, w, _variant=variant), x, w, f"fault={variant} m={m}"
    )
    assert not ok, f"fault {variant} not caught: {info}"


@pytest.mark.parametrize("config", [(512, 2), (1024, 3), (256, 3)])
def test_tilings_agree_bitwise(config):
    """Row tiles are independent dots over the full k: any tiling gives the same
    bits."""
    x, w = _x(8, seed=3), _w(N)
    y = lh.lm_head_hbm(x, w, config=config)
    np.testing.assert_array_equal(
        np.asarray(y, np.float32), np.asarray(lh.lm_head_hbm(x, w), np.float32)
    )


def test_dispatch():
    w = _w(N)
    for m in (lh.MAX_M + 1, 128):  # above MAX_M: XLA's dot, bit for bit
        x = _x(m, seed=m)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(lh.lm_head_logits)(x, w), np.float32),
            np.asarray(_xla(x, w), np.float32),
        )
    for m in (4, lh.MAX_M):  # decode row counts: the kernel
        x = _x(m, seed=m)
        np.testing.assert_array_equal(
            np.asarray(jax.jit(lh.lm_head_logits)(x, w), np.float32),
            np.asarray(lh.lm_head_hbm(x, w), np.float32),
        )
    assert lh.supports(64, _x(64), w)
    assert lh.supports(80, _x(80), w)
    assert lh.supports(lh.MAX_M, _x(lh.MAX_M), w)
    assert not lh.supports(lh.MAX_M + 1, _x(lh.MAX_M + 1), w)
    assert not lh.supports(4, _x(4).astype(jnp.float32), w)
    assert not lh.supports(4, _x(4), w[:100])  # 100 rows: not a bf16 tile multiple
    assert not lh.supports(4, _x(4), w.astype(jnp.float32))
    # a vocab shard whose out block would overflow the VMEM cap keeps XLA's dot
    wide = jax.ShapeDtypeStruct((154880, K), jnp.bfloat16)
    assert not lh.supports(64, _x(64), wide)
    assert lh.supports(4, _x(4), wide)


def _weight_vmem_lines(hlo: str, n: int) -> list[str]:
    shape = f"bf16[{n},{K}]"
    out = []
    for ln in hlo.splitlines():
        mo = re.match(r"\s*(?:ROOT )?%?\S+ = (.+?) ([a-z][\w-]*)\(", ln)
        if not mo:
            continue
        result, opcode = mo.group(1), mo.group(2)
        mine = [t for t in result.strip("()").split(", ") if t.startswith(shape)]
        if mine and (
            opcode
            in (
                "copy",
                "copy-start",
                "copy-done",
                "slice-start",
                "slice-done",
                "transpose",
            )
            or any("S(1)" in t for t in mine)
        ):
            out.append(ln.strip())
    return out


def test_no_copy_of_the_weight():
    """Compiled module: the weight reaches the custom call as the HBM parameter --
    no copy, relayout or VMEM staging of a [n, k] array."""
    x, w = _x(4), _w(N)
    lowered = jax.jit(lh.lm_head_logits).lower(x, w)
    assert "input_memory_space_colors" in lowered.as_text()
    hlo = lowered.compile().as_text()
    assert _weight_vmem_lines(hlo, N) == [], _weight_vmem_lines(hlo, N)
    assert hlo.count("custom-call(") >= 1
