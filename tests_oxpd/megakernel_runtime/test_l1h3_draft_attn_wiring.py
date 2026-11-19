# SPDX-License-Identifier: Apache-2.0
"""q38 L1-H3 (VLLM_TPU_MK_DRAFT_ATTN): the draft attention kind's wiring (CPU, no TPU).

1. The flag is registered, default off, and hashed (the compile-key test row covers the hash).
2. `Qwen4ExpMultiTokenPredictor.forward` with the tail attached: `mk_attn is None` (flag off) -> the stock
   `self_attn.attend` once and the kind never; `mk_attn` set -> the fused-loop pass (rows None) goes to `mk_attn`
   with (block, positions, QSA metadata, the layer's attention) and its output feeds the tail, `attend` is not
   called; draft step 0 (rows given) stays on the stock `attend_with_indices` either way.
3. The runner attaches the kind only under the flag, after `bind_and_warmup` (source check, one call site).
4. `attach_draft_attn` refuses without the tail / without shared selection + no index state / with a context table past
   MAX_TABLE_COLS (warning, stock), and
   attaches otherwise: `mk_attn` holds the packed operands in plan order and the attach line logs once
   (`Qwen3.8 MTP draft attention kernel attached (q38 L1-H3)` at its file:line, printed).
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import os
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import numpy as np
import pytest
import torch


def test_flag_registered_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("VLLM_TPU_MK_DRAFT_ATTN", raising=False)
    assert envs.VLLM_TPU_MK_DRAFT_ATTN is False
    monkeypatch.setenv("VLLM_TPU_MK_DRAFT_ATTN", "1")
    assert envs.VLLM_TPU_MK_DRAFT_ATTN is True


class _Rec:
    def __init__(self, out):
        self.calls, self.out = [], out

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.out


def _stub(mk_attn):
    """A predictor-shaped stub for the unbound `forward` (only the attributes it reads)."""
    T, H = 4, 8
    block = torch.full((T, H), 2.0)
    attend = _Rec(torch.full((T, 6), 3.0))
    rows_attend = _Rec((torch.full((2, 6), 5.0), None))
    attn = types.SimpleNamespace(prefix="mtp.layers.0.self_attn", attend=attend, attend_with_indices=rows_attend)
    layer = types.SimpleNamespace(self_attn=attn, attn_hyper_connection=types.SimpleNamespace(mix=lambda h: (block, None)))
    tail = _Rec(torch.full((T, H), 7.0))
    mixer = types.SimpleNamespace(mix=lambda h: (h, None))
    # q38 L4 ATTNWIRE (composition with DRAFT1-B): the predictor's forward reads DRAFT1-B's trace-time `meta_i32` switch
    # (False by default in Qwen4ExpMultiTokenPredictor.__init__), so the stub carries it too.
    stub = types.SimpleNamespace(fuse_inputs=lambda i, h, e: torch.zeros((T, H)), layers=[layer], mk_tail=tail,
                                 mk_attn=mk_attn, hyper_connection_mixer=mixer, meta_i32=False)
    return stub, block, attend, rows_attend, tail, attn


def _forward(monkeypatch, stub, rows=None):
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    md = types.SimpleNamespace(query_start_loc=torch.arange(5, dtype=torch.int32),
                               seq_lens=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
                               block_tables=torch.zeros((4, 2), dtype=torch.int32))
    ctx = types.SimpleNamespace(attn_metadata={"mtp.layers.0.self_attn": md})
    monkeypatch.setattr(mtp, "get_forward_context", lambda: ctx)
    return mtp.Qwen4ExpMultiTokenPredictor.forward(stub, torch.zeros(4, dtype=torch.int64), torch.arange(4),
                                                   torch.zeros((4, 8)), None, 0, rows=rows)


def test_flag_off_loop_pass_is_stock(monkeypatch):
    stub, block, attend, rows_attend, tail, _ = _stub(None)
    _forward(monkeypatch, stub)
    assert len(attend.calls) == 1 and not rows_attend.calls
    assert torch.equal(tail.calls[0][0][1], attend.out)


def test_flag_on_loop_pass_takes_the_kind(monkeypatch):
    kind = _Rec(torch.full((4, 6), 9.0))
    stub, block, attend, rows_attend, tail, attn = _stub(kind)
    _forward(monkeypatch, stub)
    assert not attend.calls and len(kind.calls) == 1
    a = kind.calls[0][0]
    assert torch.equal(a[0], block) and a[3] is attn and a[2].context_lens.tolist() == [9, 10, 11, 12]
    assert torch.equal(tail.calls[0][0][1], kind.out)


def test_step0_rows_stay_stock(monkeypatch):
    kind = _Rec(torch.full((4, 6), 9.0))
    stub, block, attend, rows_attend, tail, _ = _stub(kind)
    _forward(monkeypatch, stub, rows=torch.tensor([3, 1], dtype=torch.int32))
    assert not kind.calls and not attend.calls and len(rows_attend.calls) == 1


def test_runner_attaches_only_under_the_flag():
    from vllm_torchtpu.runner import tpu_runner

    src = inspect.getsource(tpu_runner)
    assert src.count("attach_draft_attn(self, self._megakernel)") == 1
    i = src.index("attach_draft_attn(self, self._megakernel)")
    head = src[:i]
    assert head.rfind("if envs.VLLM_TPU_MK_DRAFT_ATTN:") > head.rfind("self._megakernel.bind_and_warmup(self)") > 0
    assert head.rfind("\n        if self.enforce_eager:") < head.rfind("self._megakernel.bind_and_warmup(self)")


def _fake(R=2, with_tail=True, no_index=True):
    from jax.sharding import Mesh

    from vllm_torchtpu.megakernel.models import qwen38

    g = dataclasses.replace(qwen38.TINY, NH=3 * R)
    nh, hd = 3, g.HD

    def lin(n_out):
        return types.SimpleNamespace(weight=torch.zeros((n_out, g.H), dtype=torch.bfloat16))

    norm = lambda: types.SimpleNamespace(weight=torch.ones(hd, dtype=torch.bfloat16))  # noqa: E731
    attn = types.SimpleNamespace(
        _shared_ops=(object(), object()), _shared_no_index_state=no_index, _selection_source=torch.zeros(1),
        pooled=True, main_kv_dcp=1, token_topk=256, compress_ratio=4,
        cache_owners=(torch.zeros((4, 128, 1, 4, hd), dtype=torch.uint8),),
        _select_keywords=dict(compress_ratio=4, token_topk=256, norm_eps=1e-6, rotary_dim=64, main_slot=1),
        q_norm=norm(), k_norm=norm(), q_proj=lin(nh * 2 * hd), k_proj=lin(hd), v_proj=lin(hd))
    model = types.SimpleNamespace(mk_tail=object() if with_tail else None, layers=[types.SimpleNamespace(self_attn=attn)],
                                  mk_attn=None)
    runner = types.SimpleNamespace(drafter=types.SimpleNamespace(draft_model=types.SimpleNamespace(model=model)))
    driver = types.SimpleNamespace(g=g, mesh=Mesh(np.array(jax.devices()[:R]), ("tp",)), buckets=(8, 16),
                                   whole_cap=4096)
    return runner, driver, model


def test_attach_refuses_and_attaches(caplog):
    from vllm_torchtpu.megakernel_runtime import qwen38_draft_attn as da

    caplog.set_level(logging.INFO)
    r, d, m = _fake(with_tail=False)
    assert da.attach_draft_attn(r, d) is False and m.mk_attn is None
    r, d, m = _fake(no_index=False)
    assert da.attach_draft_attn(r, d) is False and m.mk_attn is None
    r, d, m = _fake()
    r.max_model_len = 1 << 20  # 8192 columns of 128-token blocks > MAX_TABLE_COLS: stock
    assert da.attach_draft_attn(r, d) is False and m.mk_attn is None
    r, d, m = _fake()
    assert da.attach_draft_attn(r, d) is True
    assert isinstance(m.mk_attn, da.DraftAttn) and m.mk_attn.n_w == 3
    recs = [x for x in caplog.records if "MTP draft attention kernel attached (q38 L1-H3)" in x.getMessage()]
    assert len(recs) == 1, [x.getMessage() for x in caplog.records]
    print(f"L1-H3 ENGAGEMENT LINE {os.path.basename(recs[0].pathname)}:{recs[0].lineno} {recs[0].getMessage()}",
          flush=True)
    assert recs[0].pathname.endswith("megakernel_runtime/qwen38_draft_attn.py")
