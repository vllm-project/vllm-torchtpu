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
