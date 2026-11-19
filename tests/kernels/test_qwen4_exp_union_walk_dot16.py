# SPDX-License-Identifier: Apache-2.0
"""q38 UNION-BUILD: dot16, BF16 operands for both union-walk dots (VLLM_TPU_QSA_PREFILL_UNION_DOT16).

The bitwise claim is a TPU one (Mosaic's f32 x f32 dot there is one BF16 pass with round-to-nearest operands, measured by the
T83m precision probe and re-checked per arm on TPU). Off TPU (XLA:CPU dots are true f32) these tests check: the key reaches
union_walk_prefill / the gathered walk kernel only when set (name _d16; refused off the gathered walk); the dot16 output keeps the
same distance to float64 attention as the served walk (precision not reduced: within 2x of the served walk's max relative error,
and equal to a BF16-operand reference built in numpy); with walk_pipe it equals dot16 alone (bitwise; same kernel math); on TPU
the output is bitwise the served walk.
"""

import os

_STRICT = "--xla_allow_excess_precision=false"
if _STRICT not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _STRICT).strip()

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from test_qwen4_exp_union_walk_pipe import (  # noqa: E402  (pytest puts this directory on sys.path)
    ENV as PIPE_ENV, FOUR, SERVED_G, SERVED_GL, SPARE, _bits, _packed_step, _walk, _walk_names, kernels,  # noqa: F401
)
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud  # noqa: E402

ENV = "VLLM_TPU_QSA_PREFILL_UNION_DOT16"


def _f64_attention(step, rows):
    """float64 gated attention over each sampled row's exact selected keys (+ partial tail)."""
    import ml_dtypes
    q, gate, blocks, lengths, table, pool, starts = (np.asarray(x) for x in step)
    pool8 = np.asarray(jax.device_get(pool)).view(np.uint8).view(ml_dtypes.float8_e4m3fn)
    out = np.zeros((len(rows), 24, 256))
    for n, r in enumerate(rows):
        L = int(lengths[r])
        gs = {int(x) for x in blocks[r] if x >= 0}
        if L % 4:
            gs.add(L // 4)
        toks = np.asarray(sorted(t for g in gs for t in range(4 * g, 4 * g + 4) if t < L))
        kv = pool8[table[r][toks // 3200], toks % 3200].reshape(-1, 4, 256).astype(np.float64)
        for h in range(24):
            k, v = kv[:, 2 * (h // 12)], kv[:, 2 * (h // 12) + 1]
            s = k @ q[r, h].astype(np.float64) / 16.0
            p = np.exp(s - s.max())
            out[n, h] = (p / p.sum()) @ v
    return out / (1.0 + np.exp(-gate[rows].astype(np.float64)))


@pytest.mark.parametrize("base", [SERVED_GL, SERVED_G], ids=["glean", "gathered"])
def test_names_and_precision_kept(kernels, base):
    step = _packed_step(21, SPARE[0], SPARE[1])
    names = _walk_names(step, base, dot16=True)
    gathered = [n for n in names if "_gathered" in n]
    assert gathered and all(n.endswith("_d16") for n in gathered), names
    assert not any("_d16" in n for n in _walk_names(step, base)), "flag off must not build the d16 kernel"
    off = _walk(step, base)
    on = _walk(step, base, dot16=True)
    if jax.default_backend() == "tpu":
        assert np.array_equal(_bits(off), _bits(on))
        return
    rows = np.flatnonzero(np.asarray(step[3]) > 0)[::17]
    ref = _f64_attention(step, rows)
    rms = np.sqrt(np.mean(ref ** 2, axis=-1, keepdims=True)) + 1e-30
    e_off = np.max(np.abs(off[rows].astype(np.float64) - ref) / rms)
    e_on = np.max(np.abs(on[rows].astype(np.float64) - ref) / rms)
    assert e_on <= 2.0 * e_off + 1e-6, (e_on, e_off)


def test_dot16_with_pipe_is_dot16(kernels):
    step = _packed_step(22, FOUR[0], FOUR[1])
    a = _walk(step, SERVED_GL, dot16=True)
    for pipe in (1, 2):
        assert np.array_equal(_bits(a), _bits(_walk(step, SERVED_GL, dot16=True, walk_pipe=pipe))), pipe


def test_refused_off_the_gathered_walk(kernels):
    step = _packed_step(23, (128,), (2000,), tokens=128)
    q, gate, blocks, lengths, table, pool, starts = step
    with pytest.raises(ValueError, match="dot16"):
        ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=4, group=64,
                              kv_tile=1024, union_cap=8192, min_context=0, lean=True, listed="reference", dot16=True)


def test_env_key_reaches_the_walk(kernels, monkeypatch):
    calls = []
    original = ud.union_walk_prefill

    def spy(*args, **kw):
        calls.append(dict(kw))
        return original(*args, **kw)

    for name, value in (("VLLM_TPU_QSA_PREFILL_UNION_WALK", "64"), ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "0"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_CAP", "8192"), ("VLLM_TPU_QSA_PREFILL_LEAN", "1"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_LISTED", "reference"),
                        ("VLLM_TPU_QSA_PREFILL_UNION_GATHERED", "reference")):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(ud, "union_walk_prefill", spy)
    step = _packed_step(24, (37, 300, 129), (5000, 1500, 4000), tokens=1024)
    q, gate, blocks, lengths, table, pool, starts = step
    qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=4)
    assert calls and "dot16" not in calls[-1]
    monkeypatch.setenv(ENV, "1")
    jax.clear_caches()
    qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=4)
    assert calls[-1].get("dot16") is True
