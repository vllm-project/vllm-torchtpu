# SPDX-License-Identifier: Apache-2.0
"""q38 UNION-BUILD: vguard_none, the gathered walk without a value guard (VLLM_TPU_QSA_PREFILL_UNION_VGUARD_NONE).

Off TPU, Pallas interpret mode, XLA excess precision off: vguard_none is bitwise the served walk (and QA-1's vguard_lean) on finite
pools across tile counts, packed unaligned steps, both forms and with mask_view / dot16 / walk_pipe; the precondition is real (a
NaN planted in a row the walk reads, the padding source = word 0 of a request's page row, is zeroed by the served guard and reaches
the output under vguard_none: the guard's removal provably bites on non-finite read rows, never on finite ones); the key reaches
the walk only when set; refused off the gathered walk.
"""

import os

_STRICT = "--xla_allow_excess_precision=false"
if _STRICT not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _STRICT).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from test_qwen4_exp_union_walk_pipe import (  # noqa: E402  (pytest puts this directory on sys.path)
    FOUR, SERVED_G, SERVED_GL, SPARE, _bits, _packed_step, _walk, _walk_names, kernels,  # noqa: F401
)
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud  # noqa: E402

ENV = "VLLM_TPU_QSA_PREFILL_UNION_VGUARD_NONE"


@pytest.mark.parametrize("base", [SERVED_GL, SERVED_G], ids=["glean", "gathered"])
def test_names_and_bitwise(kernels, base):
    step = _packed_step(31, SPARE[0], SPARE[1])
    names = [n for n in _walk_names(step, base, vguard_none=True) if "_gathered" in n]
    assert names and all(n.endswith("_vn") for n in names), names
    off = _walk(step, base)
    assert np.array_equal(_bits(off), _bits(_walk(step, base, vguard_none=True)))
    assert np.array_equal(_bits(off), _bits(_walk(step, base, vguard_none=True, mask_view=True)))


@pytest.mark.parametrize("ctx", [600, 2500, 7300], ids=lambda c: "ctx%d" % c)
def test_bitwise_tile_counts(kernels, ctx):
    step = _packed_step(33 + ctx, (192,), (ctx,), tokens=256, local=48)
    assert np.array_equal(_bits(_walk(step, SERVED_GL)), _bits(_walk(step, SERVED_GL, vguard_none=True)))


def test_bitwise_with_pipe_and_dot16_against_their_own_base(kernels):
    step = _packed_step(34, FOUR[0], FOUR[1])
    for extra in (dict(walk_pipe=1), dict(dot16=True), dict(dot16=True, mask_view=True, align_gather=True)):
        a = _walk(step, SERVED_GL, **extra)
        b = _walk(step, SERVED_GL, vguard_none=True, **extra)
        assert np.array_equal(_bits(a), _bits(b)), extra


def test_precondition_bites_on_a_nan_read_row(kernels):
    # The padding ranks of a group's last tile gather word 0 of the group's page row (token 0 of the request); a NaN planted
    # there in a V byte is zeroed by the served guard but reaches the PV dot under vguard_none (0 * NaN): the outputs differ.
    q, gate, blocks, lengths, table, pool, starts = _packed_step(35, (192,), (2500,), tokens=256, local=48)
    raw = np.asarray(jax.device_get(pool)).view(np.uint8).copy()
    page0 = int(np.asarray(table)[0, 0])
    raw[page0, 0:4, 0, 1, :] = 0x7F  # V bytes (KV head 0) of tokens 0-3 = E4M3 NaN
    bad = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    step = (q, gate, blocks, lengths, table, bad, starts)
    served = _walk(step, SERVED_GL)
    none = _walk(step, SERVED_GL, vguard_none=True)
    assert not np.isnan(served.astype(np.float32)).any()
    assert not np.array_equal(_bits(served), _bits(none)), "planting a NaN in a read row changed nothing: the check is vacuous"


def test_refused_off_the_gathered_walk(kernels):
    step = _packed_step(36, (128,), (2000,), tokens=128)
    q, gate, blocks, lengths, table, pool, starts = step
    with pytest.raises(ValueError, match="vguard_none"):
        ud.union_walk_prefill(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=4, group=64,
                              kv_tile=1024, union_cap=8192, min_context=0, lean=True, listed="reference",
                              vguard_none=True)


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
    step = _packed_step(37, (37, 300, 129), (5000, 1500, 4000), tokens=1024)
    q, gate, blocks, lengths, table, pool, starts = step
    want, _ = qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=4)
    assert calls and "vguard_none" not in calls[-1]
    monkeypatch.setenv(ENV, "1")
    jax.clear_caches()
    got, _ = qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=4)
    assert calls[-1].get("vguard_none") is True
    assert np.array_equal(_bits(want), _bits(got))


def test_engagement_marker_names_the_levers(kernels, monkeypatch):
    # one engagement line per trace naming the active union levers; none when no lever is set
    lines = []
    monkeypatch.setattr(ud, "_engagement_log", lambda msg, *args: lines.append(msg % args))
    step = _packed_step(38, FOUR[0], FOUR[1])
    _walk(step, SERVED_GL)
    assert not any("levers traced" in l for l in lines), lines
    _walk(step, SERVED_GL, mask_view=True, vguard_lean=True, align_gather=True, dot16=True, vguard_none=True)
    hit = [l for l in lines if "levers traced" in l]
    assert hit and all(t in hit[-1] for t in ("mask_view=1", "vguard_lean=1", "align_gather=1", "dot16=1", "vguard_none=1",
                                              "walk_pipe=0", "qk16=0")), hit
