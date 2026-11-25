# SPDX-License-Identifier: Apache-2.0
"""q38 HIX-GLUE SL (VLLM_TPU_HXG_SINK_LAYOUT, CPU): the MTP draft's shared-selection buffers padded to a 128-lane width.

Rows:
  S1 flag off: allocated width == logical width; pad_cols / logical_cols return the SAME tensor object (the traced graph is unchanged).
  S2 flag on: 2051 -> 2176; pad_cols pads with -1; logical_cols returns the first `width` columns; strict key ("2" raises).
  S3 the REAL Qwen4ExpMTP.bind_shared_selection on stub modules: flag off -> [max_tokens, width] buffers and no bind line; flag on ->
     [max_tokens, 2176] sink and carried rows, the drafts' _selection_source IS the target's sink, and one per-rank bind line per call.
  S4 chain equivalence (the served order, every pass): target write (the REAL qwen38_l47fold.write_sink) -> draft step 0 gather + carried
     row update -> passes 1..2 read the carried rows -> the logical rows the shared ops receive are BITWISE equal flag on vs off, for
     several steps with changing live rows (stale rows from a longer earlier step included).
  S5 call sites (AST, live calls): the stock sink writer (qsa.py) and write_sink pad; the draft reader (qsa.py, after both branches) and the
     megakernel draft attention (qwen38_draft_attn.py) slice the logical width.
  S6 registration: the key is registered and hashed (not in _TPU_COMPILE_ENV_IGNORED).
Run: PYTHONPATH=src pytest tests_oxpd/megakernel_runtime/test_hxg_sink_layout.py
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu import hixglue

KEY = "VLLM_TPU_HXG_SINK_LAYOUT"
W, MAXT = 2051, 64  # served logical width (budget 512 x 4 + 3); a small max_tokens keeps the test light


def _cpu_update_rows_(buf, rows):
    """The donating row update's semantics (`buf[: rows.shape[0]] = rows`) on CPU."""
    buf[: rows.shape[0]].copy_(rows.to(buf.dtype))


@pytest.fixture(autouse=True)
def _cpu_rows_update(monkeypatch):
    from vllm_torchtpu.layers.adapter.custom_ops import rows_update_op

    monkeypatch.setattr(rows_update_op, "update_rows_", _cpu_update_rows_)


class _Grab(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture
def grab():
    lg = hixglue.logger
    h, old = _Grab(), lg.level
    lg.addHandler(h)
    lg.setLevel(logging.INFO)
    yield h
    lg.removeHandler(h)
    lg.setLevel(old)


def _set(monkeypatch, v):
    monkeypatch.setenv(KEY, v)


# ---------------------------------------------------------------------------------------------------------------- S1 / S2
def test_s1_flag_off_identity(monkeypatch):
    _set(monkeypatch, "0")
    assert hixglue.sink_alloc_width(W) == W
    t = torch.arange(16 * W, dtype=torch.int32).reshape(16, W)
    assert hixglue.pad_cols(t, W) is t
    assert hixglue.logical_cols(t, W) is t


def test_s2_flag_on_pad_and_slice(monkeypatch):
    _set(monkeypatch, "1")
    assert hixglue.sink_alloc_width(W) == 2176
    assert hixglue.sink_alloc_width(2048) == 2048
    t = torch.arange(16 * W, dtype=torch.int32).reshape(16, W)
    p = hixglue.pad_cols(t, 2176)
    assert p.shape == (16, 2176) and torch.equal(p[:, :W], t) and bool((p[:, W:] == -1).all())
    assert torch.equal(hixglue.logical_cols(p, W), t)
    _set(monkeypatch, "2")
    with pytest.raises(Exception):
        hixglue.on(KEY)


# ---------------------------------------------------------------------------------------------------------------- S3
class _Attn(nn.Module):
    def __init__(self, rank=3):
        super().__init__()
        self.max_selected_tokens = W
        self.tp_rank = rank
        self.rope_cache = torch.zeros(1)


def _stubs():
    target_attn = _Attn()
    target = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(layer_type="linear_attention"),
                                                           SimpleNamespace(layer_type="full_attention", self_attn=target_attn)]))
    draft_attn = _Attn()
    me = SimpleNamespace(bind_replicated_embedding=lambda: None, kv_producer=False,
                         model=SimpleNamespace(layers=[SimpleNamespace(self_attn=draft_attn)]))
    return me, target, target_attn, draft_attn


@pytest.mark.parametrize("flag", ["0", "1"])
def test_s3_real_bind(monkeypatch, grab, flag):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    _set(monkeypatch, flag)
    monkeypatch.setenv("VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", "1")
    monkeypatch.setenv("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", "0")
    me, target, ta, da = _stubs()
    assert mtp.Qwen4ExpMTP.bind_shared_selection(me, target, MAXT) is True
    want = 2176 if flag == "1" else W
    assert tuple(ta._selection_sink.shape) == (MAXT, want)
    assert da._selection_source is ta._selection_sink
    assert tuple(da._selection_rows.shape) == (MAXT, want)
    assert bool((ta._selection_sink == -1).all()) and bool((da._selection_rows == -1).all())
    lines = [ln for ln in grab.lines if ln.startswith(hixglue.BIND_PREFIX + KEY + " bound rank 3:")]
    assert len(lines) == (1 if flag == "1" else 0), grab.lines
    if flag == "1":
        assert "[%d, 2176]" % MAXT in lines[0] and "logical width 2051" in lines[0]
        me2, target2, _, _ = _stubs()
        mtp.Qwen4ExpMTP.bind_shared_selection(me2, target2, MAXT)  # every bind prints (every rank process)
        assert len([ln for ln in grab.lines if KEY + " bound rank" in ln]) == 2


# ---------------------------------------------------------------------------------------------------------------- S4
def _chain(flag, monkeypatch, steps):
    """The served order with the REAL write_sink; returns the logical shared rows each draft pass receives."""
    from vllm_torchtpu.megakernel_runtime.qwen38_l47fold import write_sink

    _set(monkeypatch, flag)
    alloc = hixglue.sink_alloc_width(W)
    sink = torch.full((MAXT, alloc), -1, dtype=torch.int32)
    carried = torch.full_like(sink, -1)
    attn = SimpleNamespace(_selection_sink=sink)
    out = []
    for selected, rows in steps:
        assert write_sink(attn, selected)
        # draft step 0 (qsa.py: rows given): gather from the source, carry, then the logical slice
        shared = sink[rows.clamp_min(0)]
        _cpu_update_rows_(carried, shared)
        out.append(hixglue.logical_cols(shared, W).clone())
        for _ in range(2):  # passes 1..2 (qsa.py: rows None): the carried rows
            out.append(hixglue.logical_cols(carried[: rows.shape[0]], W).clone())
    return out


def test_s4_chain_bitwise(monkeypatch):
    g = torch.Generator().manual_seed(0)
    steps = []
    for n_rows, n_req in ((16, 3), (12, 3), (16, 4), (8, 2), (4, 1)):
        sel = torch.randint(-1, 1 << 20, (n_rows, W), generator=g, dtype=torch.int32)
        rows = torch.randint(0, n_rows, (n_req,), generator=g, dtype=torch.int32)
        rows[-1] = -1 if n_req > 2 else rows[-1]  # a dead slot (clamp_min(0) reads row 0)
        steps.append((sel, rows))
    a, b = _chain("0", monkeypatch, steps), _chain("1", monkeypatch, steps)
    assert len(a) == len(b) == 15
    for x, y in zip(a, b):
        assert x.shape == y.shape == (x.shape[0], W) and torch.equal(x, y)
    # positive control: reading the padded width without the slice differs in shape (the slice is load-bearing)
    _set(monkeypatch, "1")
    assert hixglue.sink_alloc_width(W) != W


# ---------------------------------------------------------------------------------------------------------------- S5
def _calls(fn_src: str, name: str) -> int:
    tree = ast.parse(textwrap.dedent(fn_src))
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
               and (getattr(n.func, "id", None) == name or getattr(n.func, "attr", None) == name))


def test_s5_call_sites():
    from vllm_torchtpu.megakernel_runtime import qwen38_draft_attn, qwen38_l47fold
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa

    assert _calls(inspect.getsource(qwen38_l47fold.write_sink), "pad_cols") == 1
    src = inspect.getsource(qsa)
    assert "update_rows_(sink, _hxg_pad_cols(selected, sink.shape[-1]))" in src
    i = src.index("shared = carried[: hidden.shape[0]]")
    assert "shared = _hxg_logical_cols(shared, self.max_selected_tokens)" in src[i: i + 200]
    fwd = [m for n, m in inspect.getmembers(qwen38_draft_attn, inspect.isclass) if "forward" in m.__dict__]
    assert sum(_calls(inspect.getsource(m.forward), "logical_cols") for m in fwd) == 1


# ---------------------------------------------------------------------------------------------------------------- S6
def test_s6_registered_and_hashed():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert KEY in envs.environment_variables
    assert KEY not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert set(hixglue.KEYS) <= set(envs.environment_variables)
