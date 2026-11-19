# SPDX-License-Identifier: Apache-2.0
"""q38 L2 D73MK: D73's observable engagement markers (``fused_kv_write._d73_traced`` / ``log_d73_state``).

D73 = ``VLLM_TPU_QSA_FUSED_KV_WRITE=1`` (+ ``VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN=1``): the stock QSA cache writes of decode /
verify-sized steps take one Pallas writer. Its served read needs a line that shows the writer was traced, and a startup
line that separates "flag on, never traced" from "flag off". The lines (locked formats):

    q38 D73 fused KV write traced: tokens <T>, requests <R>, lean <0|1>, kernel <name>, site <entry>@<program>
    q38 D73 state: fused_kv_write <0|1|invalid> (raw <'v'|unset>), lean <0|1|invalid> (raw <'v'|unset>),
        stock writer <name|off|invalid>, wholestep gate <present|absent|unknown (..)>, peer replay <0|1|?>, rank <r>

(the state line is one line). Checks:
(a) env on (FUSED + LEAN) at PD's served decode geometry (test_kvf_pd_geometry.make_step: MAIN_KV_DCP 4, 512-row pages,
    64 requests x 4 verify rows = 256 tokens, and 16 x 4 = 64 tokens): the stock writer ``pooled_qsa.qsa_cache_writes``
    traces ``qwen4_exp_qsa_lean_kv_write`` (spied pallas_call names and the jaxpr's pallas_call eqns), and exactly one
    trace line per shape carries the right tokens / requests / lean / kernel. FUSED alone: the fused kernel, lean 0.
(b) env off (LEAN alone set, or both unset), and FUSED on with a step that does not fit the writer (prefill-shaped):
    no trace line and no D73 kernel in the traced program.
(c) a second trace of the same shape re-runs the Python (the record counts 2) and prints nothing more.
(d) the startup line's content for on / fused-only / off / an invalid LEAN value, once per process; the real
    ``Qwen4ExpQSA.bind_pool`` emits it once per process (stub op builders); the whole-step gate is read from the
    imported code (``present`` here; ``absent`` on the pre-gate condition; ``unknown`` on unreadable source).
(e) the site: through ``pooled_qsa.pooled_qsa_step`` (selection and attention stubbed as
    tests/kernels/test_qwen4_exp_fused_kv_write.py does) the entry is the op's step; the program is the vLLM compile
    prefix of the ``TpuCompilerAdaptor`` frame on the stack (``none`` without one, ``?`` without a cache dir).
CPU only: the program is traced (make_jaxpr), not run.
"""

import logging
import os
import re
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kernels"))
import test_qwen4_exp_fused_kv_write as fkw  # noqa: E402  (pooled_qsa_step geometry)
from test_kvf_pd_geometry import (  # noqa: E402
    K_SCALE,
    RATIO,
    ROTARY,
    V_SCALE,
    interpret,  # noqa: F401 (fixture: pallas interpret, TPU7x info, SERVED_ENV)
    make_step,
)

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp import fused_kv_write, pooled_qsa  # noqa: E402

FUSED_ENV, LEAN_ENV = "VLLM_TPU_QSA_FUSED_KV_WRITE", "VLLM_TPU_QSA_FUSED_KV_WRITE_LEAN"
LEAN_K, FUSED_K = "qwen4_exp_qsa_lean_kv_write", "qwen4_exp_qsa_fused_kv_write"
D73_KERNELS = {LEAN_K, FUSED_K}
TRACED = re.compile(
    r"^q38 D73 fused KV write traced: tokens (\d+), requests (\d+), lean ([01]), kernel (\S+), site (\S+)@(\S+)$"
)
STATE = re.compile(
    r"^q38 D73 state: fused_kv_write (0|1|invalid) \(raw ('[^']*'|unset)\), lean (0|1|invalid) \(raw ('[^']*'|unset)\), "
    r"stock writer (\S+), wholestep gate (present|absent|unknown \(\w+\)), peer replay (0|1|\?), rank (\d+)$"
)


@pytest.fixture
def records(monkeypatch):
    """Messages on fused_kv_write's "vllm." logger; fresh marker state (no traces, startup line not yet logged)."""
    recs: list = []
    h = logging.Handler()
    h.emit = recs.append  # type: ignore[method-assign]
    lg = logging.getLogger("vllm." + fused_kv_write.__name__)
    lg.addHandler(h)
    old = lg.level
    lg.setLevel(logging.INFO)
    monkeypatch.setattr(fused_kv_write, "D73_TRACES", {})
    monkeypatch.setattr(fused_kv_write, "_D73_STATE_LOGGED", False)
    try:
        yield recs
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)


def d73_lines(recs, pattern):
    msgs = [r.getMessage() for r in recs]
    assert all(pattern.match(m) for m in msgs if m.startswith(pattern.pattern[1:20])), msgs
    return [m for m in msgs if pattern.match(m)]


def traced(recs):
    return [tuple(TRACED.match(m).groups()) for m in d73_lines(recs, TRACED)]


@pytest.fixture
def names(interpret, monkeypatch):  # noqa: F811
    """Every pallas_call name constructed while tracing (wraps the interpret fixture's pallas_call)."""
    got: list = []
    inner = pl.pallas_call

    def spy(*args, **kwargs):
        got.append(str(kwargs.get("name")))
        return inner(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", spy)
    return got


def jaxpr_kernels(closed) -> set:
    """pallas_call names in a ClosedJaxpr (nested jaxprs included), plus any D73 name in its text."""
    found = set()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                info = eqn.params.get("name_and_src_info")
                found.add(str(getattr(info, "name", None) or eqn.params.get("name")))
            for value in eqn.params.values():
                for sub in value if isinstance(value, (list, tuple)) else (value,):
                    if hasattr(sub, "eqns"):
                        walk(sub)
                    elif hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                        walk(sub.jaxpr)

    walk(closed.jaxpr)
    text = str(closed)
    return found | {k for k in D73_KERNELS if k in text}


def set_mode(monkeypatch, fused, lean):
    monkeypatch.setattr(envs, FUSED_ENV, fused)
    if lean:
        monkeypatch.setenv(LEAN_ENV, "1")
    else:
        monkeypatch.delenv(LEAN_ENV, raising=False)


def trace(step, *, rank=0, dcp=4, cslot=3, max_query_len=None):
    """``qsa_cache_writes`` (the stock QSA layer's writer) over one PD step, traced."""
    mql = step["max_query_len"] if max_query_len is None else max_query_len
    extra = {} if dcp <= 1 else {"dcp": (jnp.asarray([rank], jnp.int32), dcp)}
    keys = ("pool", "k", "v", "raw", "positions", "nik", "trig", "starts", "context", "ring_ids", "main_table",
            "compressed_table")

    def writes(*arrays):
        return pooled_qsa.qsa_cache_writes(
            *arrays, compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROTARY, k_scale=K_SCALE, v_scale=V_SCALE,
            compressed_slot=cslot, main_slot=-1, max_query_len=mql, **extra,
        )

    jax.clear_caches()
    return jax.make_jaxpr(writes)(*(step[k] for k in keys))


STEP64 = partial(make_step, 101, 64, page_rows=512, dcp=4)  # 64 requests x 4 rows: 256 tokens (the c64 bucket)
STEP16 = partial(make_step, 216, 16, page_rows=512, dcp=4)  # 16 requests x 4 rows: 64 tokens


# --- (a) env on: the lean kernel is traced and the line appears once per shape ---------------------------------------

def test_on_lean_traces_the_lean_kernel_and_one_line_per_shape(names, records, monkeypatch):
    set_mode(monkeypatch, True, True)
    s64, s16 = STEP64(), STEP16()
    j64 = trace(s64)
    n64 = list(names)
    j16 = trace(s16, rank=3)
    assert LEAN_K in n64 and FUSED_K not in n64, n64
    assert LEAN_K in jaxpr_kernels(j64) and FUSED_K not in jaxpr_kernels(j64), jaxpr_kernels(j64)
    assert LEAN_K in jaxpr_kernels(j16), jaxpr_kernels(j16)
    got = traced(records)
    print("D73MK (a) lines:", [r.getMessage() for r in records], flush=True)
    assert got == [
        ("256", "64", "1", LEAN_K, "qsa_cache_writes", "none"),
        ("64", "16", "1", LEAN_K, "qsa_cache_writes", "none"),
    ], got
    assert fused_kv_write.D73_TRACES == {
        (256, 64, 1, LEAN_K, "qsa_cache_writes@none"): 1,
        (64, 16, 1, LEAN_K, "qsa_cache_writes@none"): 1,
    }, fused_kv_write.D73_TRACES


def test_on_fused_without_lean_names_the_fused_kernel(names, records, monkeypatch):
    set_mode(monkeypatch, True, False)
    jx = trace(STEP16())
    assert FUSED_K in names and LEAN_K not in names, names
    assert FUSED_K in jaxpr_kernels(jx) and LEAN_K not in jaxpr_kernels(jx), jaxpr_kernels(jx)
    assert traced(records) == [("64", "16", "0", FUSED_K, "qsa_cache_writes", "none")], traced(records)


# --- (b) env off, or a step the writer does not cover: no line, no D73 kernel ------------------------------------------

@pytest.mark.parametrize("lean", [True, False], ids=["lean_alone", "both_unset"])
def test_off_no_line_and_no_d73_kernel(names, records, monkeypatch, lean):
    set_mode(monkeypatch, False, lean)
    jx = trace(STEP64())
    assert not D73_KERNELS & set(names), names
    assert not D73_KERNELS & jaxpr_kernels(jx), jaxpr_kernels(jx)
    assert names, "nothing traced: the separate writers' kernels are missing"
    assert d73_lines(records, TRACED) == [] and fused_kv_write.D73_TRACES == {}, [r.getMessage() for r in records]


def test_on_but_step_not_covered_no_line(names, records, monkeypatch):
    """256 tokens over 64 requests with max_query_len 1: prefill-shaped for the writer (``qsa_cache_writes`` keeps the
    bulk writers), so no D73 kernel and no line even with the env on."""
    set_mode(monkeypatch, True, True)
    jx = trace(STEP64(), max_query_len=1)
    assert not D73_KERNELS & set(names), names
    assert not D73_KERNELS & jaxpr_kernels(jx), jaxpr_kernels(jx)
    assert d73_lines(records, TRACED) == [] and fused_kv_write.D73_TRACES == {}


# --- (c) the same shape again: traced again, printed once --------------------------------------------------------------

def test_second_trace_of_the_same_shape_does_not_print_again(names, records, monkeypatch):
    set_mode(monkeypatch, True, True)
    s64 = STEP64()
    trace(s64)
    trace(s64)
    key = (256, 64, 1, LEAN_K, "qsa_cache_writes@none")
    assert fused_kv_write.D73_TRACES == {key: 2}, fused_kv_write.D73_TRACES  # the Python ran twice
    assert traced(records) == [("256", "64", "1", LEAN_K, "qsa_cache_writes", "none")], traced(records)
    assert names.count(LEAN_K) == 2, names


# --- (d) the startup line -------------------------------------------------------------------------------------------

def _lazy_envs(monkeypatch, *names):
    """Drop real module attributes left on ``envs`` for ``names``, so they are read from os.environ again (envs resolves
    lazily through a module ``__getattr__``; an earlier ``monkeypatch.setattr(envs, ...)`` undoes by setting the old
    value as a real attribute, which then shadows the environment for the rest of the process)."""
    for name in names:
        if name in vars(envs):
            monkeypatch.delattr(envs, name)


def _state(monkeypatch, records, fused, lean, rank, replay=None):
    _lazy_envs(monkeypatch, FUSED_ENV, "TPU_PRECOMPILE_PEER_REPLAY")
    for name, value in ((FUSED_ENV, fused), (LEAN_ENV, lean), ("TPU_PRECOMPILE_PEER_REPLAY", replay)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(fused_kv_write, "_D73_STATE_LOGGED", False)
    before = len(records)
    assert fused_kv_write.log_d73_state(rank) is True
    assert fused_kv_write.log_d73_state(rank) is False  # once per process
    lines = [r.getMessage() for r in records[before:]]
    assert len(lines) == 1 and STATE.match(lines[0]), lines
    return lines[0]


def test_startup_line_on_off_and_invalid(records, monkeypatch):
    assert fused_kv_write.D73_LEAN_KERNEL == LEAN_K and fused_kv_write.D73_FUSED_KERNEL == FUSED_K
    on = _state(monkeypatch, records, "1", "1", 5)
    assert envs.VLLM_TPU_QSA_FUSED_KV_WRITE is True and fused_kv_write.lean_writer_enabled() is True
    assert on == (
        "q38 D73 state: fused_kv_write 1 (raw '1'), lean 1 (raw '1'), stock writer qwen4_exp_qsa_lean_kv_write, "
        "wholestep gate present, peer replay 0, rank 5"
    ), on
    fused_only = _state(monkeypatch, records, "true", None, 2, replay="1")
    assert fused_only == (
        "q38 D73 state: fused_kv_write 1 (raw 'true'), lean 0 (raw unset), stock writer qwen4_exp_qsa_fused_kv_write, "
        "wholestep gate present, peer replay 1, rank 2"
    ), fused_only
    off = _state(monkeypatch, records, None, None, 0)
    assert envs.VLLM_TPU_QSA_FUSED_KV_WRITE is False
    assert off == (
        "q38 D73 state: fused_kv_write 0 (raw unset), lean 0 (raw unset), stock writer off, wholestep gate present, "
        "peer replay 0, rank 0"
    ), off
    lean_alone = _state(monkeypatch, records, "0", "1", 7)
    assert lean_alone == (
        "q38 D73 state: fused_kv_write 0 (raw '0'), lean 1 (raw '1'), stock writer off, wholestep gate present, "
        "peer replay 0, rank 7"
    ), lean_alone
    bad = _state(monkeypatch, records, "1", "2", 1)
    assert bad == (
        "q38 D73 state: fused_kv_write 1 (raw '1'), lean invalid (raw '2'), stock writer invalid, wholestep gate present, "
        "peer replay 0, rank 1"
    ), bad
    # The value as read wins over the raw one (what ``qsa_cache_writes`` acts on): an envs attribute set to False.
    monkeypatch.setenv(FUSED_ENV, "1")
    monkeypatch.setattr(envs, FUSED_ENV, False)
    monkeypatch.setattr(fused_kv_write, "_D73_STATE_LOGGED", False)
    before = len(records)
    assert fused_kv_write.log_d73_state(4) is True
    shadowed = [r.getMessage() for r in records[before:]]
    assert shadowed == [
        "q38 D73 state: fused_kv_write 0 (raw '1'), lean invalid (raw '2'), stock writer off, wholestep gate present, "
        "peer replay 0, rank 4"
    ], shadowed
    print("D73MK (d) startup lines:", on, "|", fused_only, "|", off, "|", lean_alone, "|", bad, "|", shadowed[0],
          flush=True)


def test_wholestep_gate_read_from_the_code():
    import inspect

    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_cache as wc

    src = inspect.getsource(wc)
    assert FUSED_ENV in src  # the docstring / comments name the env; they must not count as a read
    assert fused_kv_write._d73_wholestep_gate() == "present"
    assert fused_kv_write._d73_wholestep_gate(src) == "present"
    # The pre-gate dispatch (q38 L2 D73 gate a4333fb7d reverted): the whole step reads D73's env.
    pre, n = re.subn(r"\bif batched\b", "if batched and not envs.VLLM_TPU_QSA_FUSED_KV_WRITE", src, count=1)
    assert n == 1
    assert fused_kv_write._d73_wholestep_gate(pre) == "absent"
    assert fused_kv_write._d73_wholestep_gate("def broken(:\n") == "unknown (SyntaxError)"


def test_bind_pool_emits_the_startup_line_once_per_process(records, monkeypatch):
    """The real ``Qwen4ExpQSA.bind_pool`` (op builders stubbed: nothing is built) on two layers, the first bound twice
    (the profiling scratch pool, then the real one): one line, with this rank."""
    import torch
    from torch import nn

    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_mod

    flags = ("VLLM_TPU_QSA_DRAFT_SHARED_SELECTION", "VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", "VLLM_TPU_MEGAKERNEL",
             "VLLM_TPU_QSA_INDEXER_SHARD", "TPU_PRECOMPILE_PEER_REPLAY")
    _lazy_envs(monkeypatch, FUSED_ENV, *flags)
    for name in flags:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(FUSED_ENV, "1")
    monkeypatch.setenv(LEAN_ENV, "1")
    built: list = []
    for name in ("build_pooled_qsa_op", "build_pooled_qsa_rows_op"):
        monkeypatch.setattr(qsa_op, name, lambda *a, _n=name, **k: built.append(_n) or (lambda *x, **y: None))

    def layer(rank):
        m = qsa_mod.Qwen4ExpQSA.__new__(qsa_mod.Qwen4ExpQSA)
        nn.Module.__init__(m)
        m.prefix = f"model.layers.{3 + rank}.self_attn"
        m.pooled, m.single_graph, m.token_shard = True, False, False
        m.tp_size, m.tp_rank, m.main_kv_dcp, m.max_query_len = 8, rank, 1, 4
        m._op = partial(lambda *a, **k: None, compress_ratio=RATIO)
        return m

    first = layer(6)
    first.bind_pool(torch.zeros(4, 8))
    first.bind_pool(torch.zeros(4, 8))
    layer(6).bind_pool(torch.zeros(4, 8))
    assert built.count("build_pooled_qsa_op") == 3, built  # bind_pool ran to its op builds each time
    lines = d73_lines(records, STATE)
    assert lines == [
        "q38 D73 state: fused_kv_write 1 (raw '1'), lean 1 (raw '1'), stock writer qwen4_exp_qsa_lean_kv_write, "
        "wholestep gate present, peer replay 0, rank 6"
    ], lines


# --- (e) the site ----------------------------------------------------------------------------------------------------

class TpuCompilerAdaptor:
    """Stands in for vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor: the marker matches the class name on a
    ``_run_backend`` / ``compile`` / ``load`` frame and reads ``cache_dir`` (``<hash>/rank_<r>_<dp>/<prefix>``)."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def _run_backend(self, build):
        return build()


def test_site_names_the_op_step_and_the_program(records, monkeypatch):
    original = pl.pallas_call  # tests/kernels/test_qwen4_exp_fused_kv_write.py's fixture, without SERVED_ENV
    monkeypatch.setattr(pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True}))
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(pooled_qsa, "select_blocks",
                        lambda query, *a, block_topk, **k: jnp.zeros((len(query), block_topk), jnp.int32))
    monkeypatch.setattr(pooled_qsa, "qsa_attention",
                        lambda q, gate, blocks, lengths, table, pool, *a, **k: (jnp.zeros(q.shape, jnp.bfloat16), pool))
    set_mode(monkeypatch, True, True)
    args = fkw.make_step(np.random.default_rng(3), [4, 4], [5, 9], jnp.float8_e4m3fn)

    def step(*a):
        return pooled_qsa.pooled_qsa_step(*a, compress_ratio=fkw.RATIO, token_topk=fkw.TOPK, norm_eps=1e-6,
                                          rotary_dim=fkw.ROTARY, main_slot=1, max_query_len=4)

    def build():
        jax.clear_caches()
        return jax.make_jaxpr(step)(*args)

    assert LEAN_K in jaxpr_kernels(build())
    TpuCompilerAdaptor("/root/vllm-torchtpu-cache/vllm/torch_compile_cache/0a1b/rank_0_0/backbone")._run_backend(build)
    TpuCompilerAdaptor("/c/rank_3_0/model_mk_whole/")._run_backend(build)
    TpuCompilerAdaptor("/c/rank_3_0/model_mk_whole")._run_backend(build)  # same program again: no new line
    TpuCompilerAdaptor(None)._run_backend(build)
    got = traced(records)
    print("D73MK (e) lines:", [r.getMessage() for r in records], flush=True)
    assert got == [
        ("8", "2", "1", LEAN_K, "pooled_qsa_step", "none"),
        ("8", "2", "1", LEAN_K, "pooled_qsa_step", "backbone"),
        ("8", "2", "1", LEAN_K, "pooled_qsa_step", "model_mk_whole"),
        ("8", "2", "1", LEAN_K, "pooled_qsa_step", "?"),
    ], got
    assert fused_kv_write.D73_TRACES[(8, 2, 1, LEAN_K, "pooled_qsa_step@model_mk_whole")] == 2
