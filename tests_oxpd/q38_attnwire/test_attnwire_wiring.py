# SPDX-License-Identifier: Apache-2.0
"""q38 L4 ATTNWIRE T3 / T4: the draft attention kind's attach rules, all-rank agreement and runner hook (CPU, no kernel run).

T4 (one rank):
  1. capture_model's hook (TPUModelRunner.capture_model on a stub): VLLM_TPU_MK_DRAFT_ATTN off -> attach never called; on ->
     attach_draft_attn(runner, megakernel) called once, after bind_and_warmup.
  2. attach sizes the kind by the drafter's loop buckets (eagle3 `_loop_bucket` of every request count up to max_num_reqs):
     max_num_reqs 64 over token paddings [16, 32, 48, ...] -> loop buckets (16, 32, 48, 64) in the attach line and on the kind,
     operands packed at max(driver.buckets); the kind refuses any other rows count (ValueError, nothing built); the
     agreement line counts 1 attached / 0 refused.
  3. max_num_reqs 128 (loop bucket 128 > the 64-row bound) -> stock (mk_attn None), ONE warning naming the bucket and the
     bound, the agreement line counting 0 attached / 1 refused.
T3 (four TP ranks as threads, the CPU-group all_gather as a barrier stand-in):
  4. every rank attaches -> True everywhere; one rank refuses (no index-state-free selection) -> RuntimeError on EVERY rank;
     one rank's fingerprint perturbed (its main slot) -> RuntimeError on every rank; every rank refuses -> stock everywhere.
Flag off:
  5. src imports megakernel_runtime/qwen38_draft_attn.py only under `if envs.VLLM_TPU_MK_DRAFT_ATTN:` (AST), so with the flag
     off no ATTNWIRE code runs.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import logging
import os
import pathlib
import threading
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

_OP_N = itertools.count(1)
_OP_LOCK = threading.Lock()


@pytest.fixture
def da(monkeypatch):
    """qwen38_draft_attn with its once-per-process log reset and a unique custom-op name per kind (one process holds several
    ranks' kinds here)."""
    from vllm_torchtpu.megakernel_runtime import qwen38_draft_attn as mod

    real = torch.library.custom_op

    def custom_op(name, *a, **k):
        if name != "qwen38_mk::draft_attn":
            return real(name, *a, **k)
        name = f"qwen38_mk::draft_attn_wire{next(_OP_N)}"
        deco = real(name, *a, **k)

        def locked(fn):
            with _OP_LOCK:
                return deco(fn)

        return locked

    monkeypatch.setattr(torch.library, "custom_op", custom_op)
    monkeypatch.setattr(mod, "_LOGGED", set())
    return mod


def _pad_fn(paddings):
    return lambda n: next(x for x in paddings if x >= n)


def _fake(R=2, with_tail=True, no_index=True, main_slot=1, max_num_reqs=None, paddings=(16, 32, 48, 64, 80, 96, 112, 128)):
    """L1-H3's wiring-test stub (test_l1h3_draft_attn_wiring._fake) plus the drafter's loop-bucket attributes."""
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
        _select_keywords=dict(compress_ratio=4, token_topk=256, norm_eps=1e-6, rotary_dim=64, main_slot=main_slot),
        q_norm=norm(), k_norm=norm(), q_proj=lin(nh * 2 * hd), k_proj=lin(hd), v_proj=lin(hd))
    model = types.SimpleNamespace(mk_tail=object() if with_tail else None, layers=[types.SimpleNamespace(self_attn=attn)],
                                  mk_attn=None)
    drafter = types.SimpleNamespace(draft_model=types.SimpleNamespace(model=model))
    runner = types.SimpleNamespace(drafter=drafter)
    if max_num_reqs is not None:
        drafter._loop_bucket = _pad_fn(paddings)
        runner.max_num_reqs = max_num_reqs
        runner.num_tokens_paddings = list(paddings)
    driver = types.SimpleNamespace(g=g, mesh=Mesh(np.array(jax.devices()[:R]), ("tp",)), buckets=(8, 16, 32, 48, 64),
                                   whole_cap=4096)
    return runner, driver, model


def _msgs(caplog, text):
    return [x for x in caplog.records if text in x.getMessage()]


# ---------------------------------------------------------------- T4


@pytest.mark.parametrize("flag", ["0", "1"])
def test_runner_hook_attaches_only_under_the_flag(monkeypatch, da, flag):
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    monkeypatch.setenv("VLLM_TPU_MK_DRAFT_ATTN", flag)
    order = []
    mk = types.SimpleNamespace(bind_and_warmup=lambda runner: order.append(("bind", runner)))
    stub = types.SimpleNamespace(_add_pipeline_chunk_buckets=lambda: None, _pp_wave=None, _megakernel=mk,
                                 parallel_config=types.SimpleNamespace(pipeline_parallel_size=1), enforce_eager=True)
    monkeypatch.setattr(da, "attach_draft_attn", lambda runner, driver: order.append(("attach", runner, driver)) or True)
    TPUModelRunner.capture_model(stub)
    attaches = [o for o in order if o[0] == "attach"]
    if flag == "0":
        assert attaches == [], order
    else:
        assert attaches == [("attach", stub, mk)] and order[0][0] == "bind", order
    print(f"ATTNWIRE HOOK flag={flag}: bind_and_warmup then attach calls {len(attaches)}", flush=True)


def test_attach_sizes_the_kind_by_the_loop_buckets(da, caplog):
    caplog.set_level(logging.INFO)
    r, d, m = _fake(max_num_reqs=64)
    assert da.attach_draft_attn(r, d) is True
    assert isinstance(m.mk_attn, da.DraftAttn)
    assert m.mk_attn.kind.buckets == (16, 32, 48, 64), m.mk_attn.kind.buckets
    assert m.mk_attn.pack_rows == max(d.buckets) == 64
    att = _msgs(caplog, "MTP draft attention kernel attached (q38 L1-H3)")
    assert len(att) == 1 and "loop rows buckets (16, 32, 48, 64)" in att[0].getMessage(), [x.getMessage() for x in att]
    agree = _msgs(caplog, "draft attention agreement (q38 ATTNWIRE)")
    assert len(agree) == 1 and "attached 1, refused 0 of 1 TP ranks" in agree[0].getMessage(), agree
    built = []
    da_build = da.build_draft_attn_call
    try:
        da.build_draft_attn_call = lambda *a, **k: built.append(a[2]) or (_ for _ in ()).throw(RuntimeError("stop"))
        for bad in (8, 24, 128):
            with pytest.raises(ValueError, match="loop buckets"):
                m.mk_attn.kind.callable_for(bad)
        with pytest.raises(RuntimeError, match="stop"):
            m.mk_attn.kind.callable_for(48)  # a loop bucket reaches the build
    finally:
        da.build_draft_attn_call = da_build
    assert built == [48], built
    print(f"ATTNWIRE ATTACH max_num_reqs=64: '{att[0].getMessage()}' / '{agree[0].getMessage()}'; kind buckets "
          f"{m.mk_attn.kind.buckets}, packed at {m.mk_attn.pack_rows}; rows 8 / 24 / 128 refused (ValueError), 48 builds",
          flush=True)


def test_attach_refuses_past_64_loop_rows_loudly(da, caplog):
    caplog.set_level(logging.INFO)
    r, d, m = _fake(max_num_reqs=128)
    assert da.attach_draft_attn(r, d) is False and m.mk_attn is None
    warn = [x for x in _msgs(caplog, "REFUSED (q38 ATTNWIRE)") if x.levelno == logging.WARNING]
    assert len(warn) == 1, [x.getMessage() for x in caplog.records]
    w = warn[0].getMessage()
    assert "loop bucket 128 rows exceeds the kernel's 64-row bound" in w and "stays stock" in w, w
    agree = _msgs(caplog, "draft attention agreement (q38 ATTNWIRE)")
    assert len(agree) == 1 and "attached 0, refused 1 of 1 TP ranks" in agree[0].getMessage(), agree
    assert not _msgs(caplog, "MTP draft attention kernel attached (q38 L1-H3)")
    print(f"ATTNWIRE REFUSE max_num_reqs=128: WARNING '{w}' / '{agree[0].getMessage()}'", flush=True)


# ---------------------------------------------------------------- T3


class FakeGroup:
    def __init__(self, world):
        self.barrier = threading.Barrier(world, timeout=120)
        self.slots = [None] * world

    def gather(self, rank, vec):
        self.slots[rank] = list(vec)
        self.barrier.wait()
        out = [list(s) for s in self.slots]
        self.barrier.wait()
        return out


def _ranks(monkeypatch, da, world, make):
    """Each rank thread builds its stub (`make(rank)`) and calls attach_draft_attn; returns per rank (result | exception)."""
    group, tl = FakeGroup(world), threading.local()
    monkeypatch.setattr(da, "_gather", lambda vec: group.gather(tl.rank, vec))
    monkeypatch.setenv("VLLM_TPU_MK_DRAFT_ATTN", "1")
    res = [None] * world

    def go(rk):
        tl.rank = rk
        r, d, m = make(rk)
        try:
            res[rk] = (da.attach_draft_attn(r, d), m)
        except Exception as e:  # noqa: BLE001
            res[rk] = (e, m)

    ts = [threading.Thread(target=go, args=(k,)) for k in range(world)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(300)
    return res


def test_agreement_every_rank_attaches(monkeypatch, da, caplog):
    caplog.set_level(logging.INFO)
    res = _ranks(monkeypatch, da, 4, lambda k: _fake(max_num_reqs=64))
    assert [x for x, _ in res] == [True] * 4, res
    agree = _msgs(caplog, "draft attention agreement (q38 ATTNWIRE)")
    assert agree and "attached 4, refused 0 of 4 TP ranks" in agree[0].getMessage(), agree
    print(f"ATTNWIRE AGREE all attach: {[x for x, _ in res]} '{agree[0].getMessage()}'", flush=True)


def test_agreement_fail_stop_when_one_rank_refuses(monkeypatch, da):
    res = _ranks(monkeypatch, da, 4, lambda k: _fake(max_num_reqs=64, no_index=(k != 2)))
    errs = [x for x, _ in res]
    assert all(isinstance(e, RuntimeError) and "disagree" in str(e) for e in errs), errs
    print(f"ATTNWIRE AGREE rank 2 refuses: RuntimeError on all 4 ranks ({str(errs[0])[:140]})", flush=True)


def test_agreement_fail_stop_on_a_perturbed_fingerprint(monkeypatch, da):
    res = _ranks(monkeypatch, da, 4, lambda k: _fake(max_num_reqs=64, main_slot=(-1 if k == 1 else 1)))
    errs = [x for x, _ in res]
    assert all(isinstance(e, RuntimeError) and "disagree" in str(e) for e in errs), errs
    rows = str(errs[0])
    print(f"ATTNWIRE AGREE rank 1 fingerprint perturbed (main slot): RuntimeError on all 4 ranks ({rows[:160]})", flush=True)


def test_agreement_every_rank_refuses_is_stock(monkeypatch, da, caplog):
    caplog.set_level(logging.INFO)
    res = _ranks(monkeypatch, da, 4, lambda k: _fake(max_num_reqs=64, with_tail=False))
    assert [x for x, _ in res] == [False] * 4 and all(m.mk_attn is None for _, m in res), res
    agree = _msgs(caplog, "draft attention agreement (q38 ATTNWIRE)")
    assert agree and "attached 0, refused 4 of 4 TP ranks" in agree[0].getMessage(), agree
    print(f"ATTNWIRE AGREE all refuse: stock everywhere '{agree[0].getMessage()}'", flush=True)


# ---------------------------------------------------------------- flag off


def test_flag_off_never_imports_the_kind():
    import vllm_torchtpu

    src = pathlib.Path(vllm_torchtpu.__file__).parent
    sites = []
    for f in sorted(src.rglob("*.py")):
        if f.name == "qwen38_draft_attn.py":
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        parents = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                parents[ch] = node
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if not any("qwen38_draft_attn" in n for n in names):
                continue
            up, guarded = node, False
            while up in parents:
                up = parents[up]
                if isinstance(up, ast.If) and "VLLM_TPU_MK_DRAFT_ATTN" in ast.unparse(up.test):
                    guarded = True
                    break
            sites.append((str(f.relative_to(src)), node.lineno, guarded))
    assert sites and all(g for _, _, g in sites), sites
    print(f"ATTNWIRE FLAGOFF import sites of qwen38_draft_attn: {sites} (all under VLLM_TPU_MK_DRAFT_ATTN)", flush=True)
