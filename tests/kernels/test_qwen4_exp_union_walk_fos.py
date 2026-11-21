# SPDX-License-Identifier: Apache-2.0
"""q38 PSTEP-QSA WALK-FOS prototype (default off; NOT bitwise): BF16 softmax (MM R-B port), FOS lag-1 / static offset
(MM L-A port) and the lean FP8 tile (MM L-B port) in the gathered union walk (credit MM KERN-MM / HUNT-ATTN).

Off TPU (XLA:CPU interpret) these tests check: each variant builds its own kernel (name suffix) only when set and is refused
off the gathered walk; flags off keep the served kernel name; each variant's output is finite and stays within a SMOKE bound
of float64 attention over the exact selected keys (the bound of record is locked later in the WALK-FOS prereg); each variant
really differs from the served walk (engaged). Numerics of record are a TPU question.
"""

import os

_STRICT = "--xla_allow_excess_precision=false"
if _STRICT not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _STRICT).strip()

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from test_qwen4_exp_union_walk_dot16 import _f64_attention  # noqa: E402
from test_qwen4_exp_union_walk_pipe import SERVED_GL, SPARE, _packed_step, _walk, _walk_names, kernels  # noqa: E402,F401
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud  # noqa: E402

VARIANTS = {
    "rb": (dict(bf16_softmax=True), "_rb"),
    "fos1": (dict(bf16_softmax=True, fos=1), "_rb_fos1"),
    "fos2": (dict(bf16_softmax=True, fos=2), "_rb_fos2"),
    "fos2_lt": (dict(bf16_softmax=True, fos=2, lean_tile=True), "_rb_fos2_lt"),
    "rb_lt": (dict(bf16_softmax=True, lean_tile=True), "_rb_lt"),
}
SMOKE_BOUND = 0.03   # max |out - f64| / row rms; a smoke bar only (see the module docstring)


def _err(step, out, rows, ref):
    rms = np.sqrt(np.mean(ref ** 2, axis=-1, keepdims=True)) + 1e-30
    return float(np.max(np.abs(out[rows].astype(np.float64) - ref) / rms))


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_names_bound_and_engaged(kernels, name):
    kw, suffix = VARIANTS[name]
    step = _packed_step(31, SPARE[0], SPARE[1])
    names = [n for n in _walk_names(step, SERVED_GL, **kw) if "_gathered" in n]
    assert names and all(n.endswith(suffix) for n in names), names
    assert not any("_rb" in n for n in _walk_names(step, SERVED_GL)), "flags off must keep the served kernel"
    base = _walk(step, SERVED_GL)
    out = _walk(step, SERVED_GL, **kw)
    assert np.all(np.isfinite(out.astype(np.float32)))
    rows = np.flatnonzero(np.asarray(step[3]) > 0)[::13]
    ref = _f64_attention(step, rows)
    e_base, e_var = _err(step, base, rows, ref), _err(step, out, rows, ref)
    print(f"PQFOS {name} max_rel_err_vs_f64 {e_var:.5f} (served walk {e_base:.5f}, smoke bound {SMOKE_BOUND})")
    assert e_var <= SMOKE_BOUND, (name, e_var, e_base)
    assert not np.array_equal(np.asarray(out).view(np.uint16), np.asarray(base).view(np.uint16)), "variant not engaged"


def test_refused_off_the_gathered_walk_and_bad_combos(kernels):
    step = _packed_step(32, (128,), (2000,), tokens=128)
    q, gate, blocks, lengths, table, pool, starts = step
    with pytest.raises(ValueError, match="WALK-FOS"):
        ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=4, group=64,
                              kv_tile=1024, union_cap=8192, min_context=0, lean=True, listed="reference",
                              bf16_softmax=True)
    for bad in (dict(fos=2), dict(lean_tile=True), dict(bf16_softmax=True, fos=3),
                dict(bf16_softmax=True, walk_pipe=1)):
        with pytest.raises(ValueError):
            _walk(step, SERVED_GL, **bad)


def test_fos2_guard_reruns_the_served_walk_on_overflow(kernels):
    """Logits far outside the exp range: the static-offset walk overflows, the guard re-runs the served walk, so the output is
    bit for bit the served one (finite). With ordinary logits the guard keeps the FOS output (engaged, not the served bits)."""
    ud._FOS_GUARD_TRACED.clear()
    hot = _packed_step(33, SPARE[0], SPARE[1], q_std=60.0)
    served = _walk(hot, SERVED_GL)
    fos = _walk(hot, SERVED_GL, bf16_softmax=True, fos=2)
    assert ud._FOS_GUARD_TRACED, "the fos=2 guard was not traced"
    assert np.all(np.isfinite(fos.astype(np.float32)))
    np.testing.assert_array_equal(np.asarray(fos).view(np.uint16), np.asarray(served).view(np.uint16))
    cool = _packed_step(34, SPARE[0], SPARE[1])
    a, b = _walk(cool, SERVED_GL), _walk(cool, SERVED_GL, bf16_softmax=True, fos=2)
    assert not np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16)), "guard fired on ordinary logits"


@pytest.mark.parametrize("strips", [2, 3, 6])
def test_row_strips_bitwise_served(kernels, strips):
    """Bitwise register-pressure variant: row strips of whole q-heads over the full key range give the served walk's bits
    (CPU interpret with excess precision off; the TPU row decides). Positive control: one ulp reads NOT bitwise."""
    step = _packed_step(35, SPARE[0], SPARE[1])
    names = [n for n in _walk_names(step, SERVED_GL, row_strips=strips) if "_gathered" in n]
    assert names and all(n.endswith(f"_rs{strips}") for n in names), names
    a = np.asarray(_walk(step, SERVED_GL)).view(np.uint16)
    b = np.asarray(_walk(step, SERVED_GL, row_strips=strips)).view(np.uint16)
    np.testing.assert_array_equal(b, a)
    c = b.copy()
    c.reshape(-1)[np.flatnonzero(c.reshape(-1))[0]] ^= 1
    assert not np.array_equal(c, a)


def test_remat_max_close_to_served(kernels):
    """remat_max takes the max from BF16-operand scores: bitwise on TPU only (Mosaic's f32 dot is one BF16 pass, M T83m); on
    XLA:CPU the f32 and BF16 dots differ, so here only the precision is checked (within 2x of the served walk vs float64)."""
    step = _packed_step(36, SPARE[0], SPARE[1])
    names = [n for n in _walk_names(step, SERVED_GL, remat_max=True) if "_gathered" in n]
    assert names and all(n.endswith("_rm") for n in names), names
    base, rm = _walk(step, SERVED_GL), _walk(step, SERVED_GL, remat_max=True)
    rows = np.flatnonzero(np.asarray(step[3]) > 0)[::13]
    ref = _f64_attention(step, rows)
    assert _err(step, rm, rows, ref) <= 2.0 * _err(step, base, rows, ref) + 1e-6
