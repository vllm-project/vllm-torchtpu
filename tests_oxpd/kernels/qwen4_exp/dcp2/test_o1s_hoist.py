# SPDX-License-Identifier: Apache-2.0
"""q38 L2 O1s-HOIST: shared rank operands for O1s (port of MM DSTOCK-B HOIST H-A) and the decode
indexer without its early-exit cond.

Port 1, VLLM_TPU_QSA_SHARED_RANK_OPERANDS (MM q38mm-dstockb 581cfffd3 + 2c9309247; int-5 97b9b6e17 +
d1e888ee1). Each QSA layer registers its own int32 [1] ``request_shard_index`` / ``dcp_rank`` buffer
today, so O1s's per-step selection metadata (``request_shard_step`` and the shared-stream
``_request_window``) runs once per layer: 12x per step, 0.183 ms/step at 64 rows (M, G2r j-92d0cc68
rank 0, FR7 tree). With the flag every layer on a rank registers the same tensor:
  (a) ``rank_operand`` shares one object per (kind, value, device) only when on; ``qsa.py`` builds all
      three rank buffers through it;
  (a2) Dynamo then gives the shared buffer ONE graph input (12 when not shared);
  (b) a two-layer O1s module still gives the decode (request-shard) and prefill (plain) traces one
      input list (the shape_variants hand-over, test_o1s_every_compile_range_takes_one_input_list),
      with one request_shard_index / dcp_rank input when shared, two when not;
  (c) CSE probe: 12 layers of the real ``pooled_qsa_request_select_dcp_step`` in one jit on XLA:CPU.
      Shared rank operands leave the request_shard_step / _request_window instructions at about one
      layer's count; distinct operands keep 12 copies. Outputs are identical either way.

Port 2, VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT: the decode shared-stream indexer calls StreamIndex with
``enable_early_exit=False`` (pooled_qsa.py ``_select_blocks`` decode branch). Step-level check here
(dcp2 harness, the O1s stand-in honours the cond and returns a batch-dependent order without it):
O1 and O1s steps with the flag on equal the flag-off steps (pools byte-equal, partial attention and
lse bit-identical, selections equal as sets) at short contexts (every row inside the top-k budget, the
cond's fast path today) and at served-like contexts. The real StreamIndex kernel check is in
tests/kernels/test_qwen4_exp_decode_no_early_exit.py.
"""

import ast
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dcp2_harness import rank_operand as jrank
from test_o1s_request_shard import (  # noqa: F401 (standin is a fixture)
    D,
    HID_,
    K_SCALE,
    RATIO,
    ROTARY,
    ROWS,
    TOPK,
    TP,
    V_SCALE,
    _Captured,
    _args,
    _bits,
    _bound_qsa,
    _inputs,
    _qsa_ops,
    _seats,
    _u8,
    make_step,
    standin,
)

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
QSA_PY = os.path.join(ROOT, "src", "vllm_torchtpu", "models", "vllm", "qwen4_exp", "qsa.py")
FLAG = "VLLM_TPU_QSA_SHARED_RANK_OPERANDS"
NO_EXIT = "VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT"
RANK_BUFFERS = ("dcp_rank", "token_shard_index", "request_shard_index")


_MISSING = object()


@pytest.fixture
def envset(monkeypatch):
    """Set envs flags for one test, order-proof and leak-free: os.environ (monkeypatch) AND the module attribute,
    restored on teardown by deleting it (or putting back an attribute that was already there). Never
    monkeypatch.setattr(envs, ...): its undo writes the old value back as a frozen module attribute that shadows
    os.environ for every later test in the session (j-3307bbed / j-cab937b3)."""
    saved = {}

    def set_(name, value):
        if name not in saved:
            saved[name] = envs.__dict__.get(name, _MISSING)
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(value))
        envs.__dict__[name] = envs.environment_variables[name]()

    yield set_
    for name, old in saved.items():
        if old is _MISSING:
            envs.__dict__.pop(name, None)
        else:
            envs.__dict__[name] = old


@pytest.fixture
def fresh_shared(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa_rank_operands as ro

    monkeypatch.setattr(ro, "_SHARED", {})
    return ro


# ------------------------------------------------------------------------------------- (a) / (a2)


def test_rank_operand_shares_only_when_on(monkeypatch, fresh_shared, envset):
    import torch

    ro = fresh_shared
    envset(FLAG, None)
    a, b = ro.rank_operand("dcp_rank", 1), ro.rank_operand("dcp_rank", 1)
    assert a is not b, "flag off: a fresh tensor per layer (unchanged behaviour)"
    for t in (a, b):
        assert t.dtype == torch.int32 and tuple(t.shape) == (1,) and int(t) == 1
    envset(FLAG, 0)
    assert ro.rank_operand("dcp_rank", 1) is not ro.rank_operand("dcp_rank", 1)
    envset(FLAG, 1)
    a, b = ro.rank_operand("dcp_rank", 1), ro.rank_operand("dcp_rank", 1)
    assert a is b and int(a) == 1 and a.dtype == torch.int32 and tuple(a.shape) == (1,)
    c = ro.rank_operand("dcp_rank", 2)
    d = ro.rank_operand("request_shard_index", 1)
    assert c is not a and int(c) == 2, "a different value must be a different tensor"
    assert d is not a and int(d) == 1, "a different kind must be a different tensor"
    assert ro.rank_operand("request_shard_index", 1) is d


def _register_buffer_sites():
    """(buffer name, the value expression's source) of every self.register_buffer(<name>, ...) call
    in qsa.py whose name is a rank buffer."""
    src = open(QSA_PY).read()
    out = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register_buffer" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value in RANK_BUFFERS):
            out.append((node.args[0].value, ast.get_source_segment(src, node.args[1])))
    return out


def test_qsa_builds_every_rank_buffer_through_rank_operand():
    sites = _register_buffer_sites()
    assert sorted(n for n, _ in sites) == sorted(RANK_BUFFERS), sites
    for name, expr in sites:
        assert re.fullmatch(rf'rank_operand\("{name}", [^)]+\)', expr.strip()), (name, expr)


def test_shared_buffers_are_one_dynamo_input(monkeypatch, fresh_shared, envset):
    """12 layers whose rank buffer comes from rank_operand: one int32 [1] graph input when the flag
    is on (shared), 12 when off. Dynamo graph and the AOT forward graph (what a torch_tpu backend
    receives); MM's q38mm_hoist_dynamo_dedup.py probe as a test."""
    import torch
    from torch._dynamo.backends.common import aot_autograd

    ro = fresh_shared

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("request_shard_index", ro.rank_operand("request_shard_index", 3),
                                 persistent=False)

        def forward(self, x, starts):
            base = self.request_shard_index.reshape(()) * 32
            return x + (starts - base).clamp(0, 32).sum()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer() for _ in range(12)])

        def forward(self, x, starts):
            for layer in self.layers:
                x = layer(x, starts)
            return x

    def int_inputs(on):
        envset(FLAG, 1 if on else None)
        seen = {}

        def dyn(gm, ex):
            seen["dynamo"] = sum(1 for n in gm.graph.nodes if n.op == "placeholder"
                                 and getattr(n.meta.get("example_value"), "dtype", None) == torch.int32
                                 and tuple(n.meta["example_value"].shape) == (1,))
            return gm

        def fw(gm, ex):
            seen["aot"] = sum(1 for e in ex if isinstance(e, torch.Tensor) and e.dtype == torch.int32
                              and tuple(e.shape) == (1,))
            return gm

        x, starts = torch.zeros(4), torch.arange(33, dtype=torch.int32)
        torch._dynamo.reset()
        torch.compile(Model(), backend=dyn, fullgraph=True)(x, starts)
        torch._dynamo.reset()
        torch.compile(Model(), backend=aot_autograd(fw_compiler=fw), fullgraph=True)(x, starts)
        torch._dynamo.reset()
        return seen

    assert int_inputs(False) == {"dynamo": 12, "aot": 12}
    assert int_inputs(True) == {"dynamo": 1, "aot": 1}


# ------------------------------------------------------------------------------------- (b)


def _traced_two_layer(m1, m2, buckets):
    """test_o1s_request_shard._traced_subgraphs for a step of two QSA layers (the dedup only shows
    with more than one layer)."""
    import torch
    from torch import nn
    from vllm.compilation.backends import split_graph

    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import QSAMetadata

    class Step(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.a, self.b = a, b

        def forward(self, hidden, positions, query_start_loc, context_lens):
            meta = QSAMetadata(query_start_loc, context_lens, None, None, None)
            return (self.a.attend_with_indices(hidden, positions, meta),
                    self.b.attend_with_indices(hidden, positions, meta))

    graphs = []

    def backend(gm, example_inputs):
        _, items = split_graph(gm, [])
        assert len(items) == 1, [i.submod_name for i in items]
        graphs.append(items[0].graph)

        def run(*args):
            raise _Captured()

        return run

    torch._dynamo.reset()
    compiled = torch.compile(Step(m1, m2), backend=backend, fullgraph=True, dynamic=False)
    starts = torch.zeros(65, dtype=torch.int32)
    context = torch.zeros(64, dtype=torch.int32)
    for tokens in buckets:
        hidden = torch.zeros(tokens, HID_, dtype=torch.bfloat16)
        positions = torch.zeros(tokens, dtype=torch.int64)
        torch._dynamo.maybe_mark_dynamic(hidden, 0)
        torch._dynamo.maybe_mark_dynamic(positions, 0)
        try:
            with torch.no_grad():
                compiled(hidden, positions, starts, context)
        except Exception as exc:
            if not isinstance(exc, _Captured) and not isinstance(exc.__cause__, _Captured):
                raise
    torch._dynamo.reset()
    return graphs


@pytest.mark.parametrize("on", [False, True])
def test_o1s_two_layers_one_input_list_with_shared_rank(monkeypatch, fresh_shared, envset, on):
    """Two O1s layers (TP rank 5, DCP 4) whose rank buffers come from rank_operand, as __init__ builds
    them. Decode (16 tokens, request shard) and prefill (512 tokens, plain DCP op) traces give vLLM's
    subgraph the same inputs in the same order (shape_variants hands buckets between them); shared:
    one request_shard_index and one dcp_rank input; not shared: one per layer."""
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.compilation import shape_variants

    ro = fresh_shared
    envset(FLAG, 1 if on else None)
    layers = [_bound_qsa(monkeypatch, True) for _ in range(2)]
    for m in layers:  # as Qwen4ExpQSA.__init__ registers them (qsa.py, rank_operand)
        m._buffers["dcp_rank"] = ro.rank_operand("dcp_rank", 5 % 4)
        m._buffers["request_shard_index"] = ro.rank_operand("request_shard_index", 5)
    assert (layers[0].request_shard_index is layers[1].request_shard_index) is on
    assert (layers[0].dcp_rank is layers[1].dcp_rank) is on
    decode, prefill = _traced_two_layer(layers[0], layers[1], (16, 512))
    tracked = shape_variants.BucketExecutables()
    key = "artifact_compile_range_16_16_subgraph_0"
    runnable = object()
    tracked.put("cache_dir", key, decode, runnable)
    assert tracked.get("cache_dir", key, prefill) is runnable
    assert _inputs(decode) == _inputs(prefill)
    names = [n for n, _ in _inputs(decode)]
    want = 1 if on else 2
    assert sum("request_shard_index" in n for n in names) == want, names
    assert sum("dcp_rank" in n for n in names) == want, names
    assert _qsa_ops(decode) == ["pooled_qsa_request_attend_dcp", "pooled_qsa_request_select_dcp",
                                "qsa_dcp_combine"], _qsa_ops(decode)
    assert _qsa_ops(prefill) == ["pooled_qsa_request_plain_dcp", "qsa_dcp_combine"], _qsa_ops(prefill)


# ------------------------------------------------------------------------------------- (c) CSE probe

LAYERS = 12
SCOPES = {"q38h_rss": "request_shard_step", "q38h_rsw": "request_shard_width", "q38h_win": "_request_window"}


def _scope_metadata(monkeypatch):
    """Wrap the O1s per-step metadata functions in named scopes (module globals, read at call time)."""
    for scope, name in SCOPES.items():
        original = getattr(pooled_qsa, name)

        def scoped(*a, _f=original, _s=scope, **k):
            with jax.named_scope(_s):
                return _f(*a, **k)

        monkeypatch.setattr(pooled_qsa, name, scoped)


def _probe(monkeypatch, *, layers, shared, tokens=48, seed=21):
    """layers x pooled_qsa_request_select_dcp_step (TP rank 3, DCP rank 1) in one jit, each layer with
    its own qgate / key / index rows (so the selections cannot merge) and the same step metadata.
    Returns (outputs, {scope: optimized-HLO instruction count}, conditional count)."""
    query_lens, contexts = _seats(9, 7, rng_seed=seed)
    layout, pools, qgate, s, m, _ = make_step(seed, query_lens, contexts, tokens)
    r = 1
    rng = np.random.default_rng(seed + 1)
    per = [dict(qgate=jnp.asarray(rng.normal(0, 1, qgate.shape), jnp.bfloat16),
                index_qk=jnp.asarray(rng.normal(0, 1, s["index_qk"].shape), jnp.bfloat16))
           for _ in range(layers)]
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=0, max_query_len=ROWS, shard_size=TP, dcp_size=D)
    shard_ops = [jrank(3) for _ in range(layers)]
    dcp_ops = [jrank(r) for _ in range(layers)]

    def model(qgs, iqks, shard_ops, dcp_ops, pool):
        outs = []
        for i in range(layers):
            si = shard_ops[0] if shared else shard_ops[i]
            dr = dcp_ops[0] if shared else dcp_ops[i]
            ss = dict(s, index_qk=iqks[i])
            res = pooled_qsa.pooled_qsa_request_select_dcp_step(
                *_args(qgs[i], ss, pool, m, layout.dcp_table), si, dr, **opts)
            outs.append(res[-1])
        return jnp.stack(outs)

    args = ([p["qgate"] for p in per], [p["index_qk"] for p in per], shard_ops, dcp_ops, pools[r])
    jax.clear_caches()
    f = jax.jit(model)
    txt = f.lower(*args).compile().as_text()
    instr = [l for l in txt.split("\n") if " = " in l and "metadata=" in l]
    counts = {scope: sum(1 for l in instr if scope in l) for scope in SCOPES}
    out = np.asarray(f(*args))
    return out, counts, txt.count(" conditional(")


def test_o1s_shared_rank_merges_step_metadata_across_layers(monkeypatch, interpret, rpa_stand_in, standin):
    """The CSE probe (XLA:CPU optimized HLO; a proxy for the TPU program, I): with the rank operands
    shared, the O1s step metadata of 12 layers compiles to about one layer's instructions; with one
    operand per layer it stays 12x. Same selections either way. One summary line for the reader."""
    _scope_metadata(monkeypatch)
    _, one, _ = _probe(monkeypatch, layers=1, shared=True)
    out_d, distinct, cond_d = _probe(monkeypatch, layers=LAYERS, shared=False)
    out_s, shared, cond_s = _probe(monkeypatch, layers=LAYERS, shared=True)
    total = {k: sum(v.values()) for k, v in (("one", one), ("distinct", distinct), ("shared", shared))}
    print("Q38_L2_HOIST_PROBE " + " ".join(f"{k}={v}" for k, v in total.items())
          + f" per_scope_one={one} per_scope_distinct={distinct} per_scope_shared={shared}"
          + f" conditionals_distinct={cond_d} conditionals_shared={cond_s}")
    assert total["one"] > 0, "the named scopes did not reach the compiled HLO"
    assert np.array_equal(out_d, out_s), "shared rank operands changed a selection"
    assert total["distinct"] >= (LAYERS - 2) * total["one"], total
    assert total["shared"] <= 2 * total["one"], total


# ------------------------------------------------------------------------------------- port 2, step level


def _step_outputs(query_lens, contexts, tokens, seed, flag_on, monkeypatch, envset):
    """O1 (pooled_qsa_dcp_step) and O1s (every TP shard's select + the attend) for DCP rank 1, with the
    no-early-exit flag on or off."""
    envset(NO_EXIT, 1 if flag_on else None)
    seen = []
    inner = pooled_qsa.streamindex_topk  # the standin fixture's stand-in

    def spy(*a, **k):
        seen.append(k.get("enable_early_exit"))
        return inner(*a, **k)

    monkeypatch.setattr(pooled_qsa, "streamindex_topk", spy)
    jax.clear_caches()
    layout, pools, qgate, s, m, starts = make_step(seed, query_lens, contexts, tokens)
    opts = dict(compress_ratio=RATIO, token_topk=TOPK, norm_eps=1e-6, rotary_dim=ROTARY,
                k_scale=K_SCALE, v_scale=V_SCALE, main_slot=0, max_query_len=ROWS)
    static = ("compress_ratio", "token_topk", "norm_eps", "rotary_dim", "dcp_size", "k_scale",
              "v_scale", "main_slot", "max_query_len")
    r = 1
    args = _args(qgate, s, pools[r], m, layout.dcp_table)
    o1 = jax.jit(pooled_qsa.pooled_qsa_dcp_step, static_argnames=static)(*args, jrank(r), dcp_size=D, **opts)
    sel = jax.jit(pooled_qsa.pooled_qsa_request_select_dcp_step, static_argnames=static + ("shard_size",))
    att = jax.jit(pooled_qsa.pooled_qsa_request_attend_dcp_step,
                  static_argnames=("kv_heads", "compress_ratio", "token_topk", "dcp_size", "k_scale",
                                   "v_scale", "main_slot", "max_query_len"))
    blocks = []
    for sh in range(TP):
        res = sel(*args, jrank(sh), jrank(r), shard_size=TP, dcp_size=D, **opts)
        first = res if sh == 0 else first
        blocks.append(res[-1])
    pool, q, gate, lengths, req, st, _ = first
    o1s = att(q, gate, lengths, req, st, jnp.asarray(layout.dcp_table), m["context"], pool,
              jnp.stack(blocks), jrank(r), kv_heads=1, compress_ratio=RATIO, token_topk=TOPK,
              dcp_size=D, k_scale=K_SCALE, v_scale=V_SCALE, main_slot=0, max_query_len=ROWS)
    monkeypatch.setattr(pooled_qsa, "streamindex_topk", inner)
    # Engagement: the decode indexer saw the cond switch as set (never vacuous).
    assert seen and all(v is (not flag_on) for v in seen), (flag_on, seen)
    return int(starts[-1]), o1, o1s


def _same_step(a, b, live, what):
    assert bool(jnp.array_equal(_u8(a[0]), _u8(b[0]))), f"{what}: pool differs"
    assert np.array_equal(_bits(a[1])[:live], _bits(b[1])[:live]), f"{what}: partial attention differs"
    assert np.array_equal(_bits(a[2])[:live], _bits(b[2])[:live]), f"{what}: lse differs"
    sa, sb = np.asarray(a[3]), np.asarray(b[3])
    for row in range(live):
        assert np.array_equal(np.sort(sa[row]), np.sort(sb[row])), f"{what}: row {row} selection differs"
    return bool(np.array_equal(sa[:live], sb[:live]))


@pytest.mark.parametrize("contexts_kind", ["short", "served_like"])
def test_no_early_exit_step_equals_today(monkeypatch, interpret, rpa_stand_in, standin, envset, contexts_kind):
    """48-token bucket (> 32 rows, so the stand-in's order differs without the cond): short contexts
    (every row inside the 256-token top-k budget: today's cond takes the fast path) and served-like
    contexts (37..4100 tokens: today's cond takes the kernel path). Flag on equals flag off for O1
    and O1s: pools byte-equal, out / lse bit-identical, selections equal as sets."""
    query_lens, contexts = _seats(10, 2, rng_seed=31, short=contexts_kind == "short")
    live, o1_off, o1s_off = _step_outputs(query_lens, contexts, 48, 31, False, monkeypatch, envset)
    _, o1_on, o1s_on = _step_outputs(query_lens, contexts, 48, 31, True, monkeypatch, envset)
    exact = {"o1": _same_step(o1_off, o1_on, live, "O1"), "o1s": _same_step(o1s_off, o1s_on, live, "O1s")}
    print(f"Q38_L2_NOEXIT_STEP {contexts_kind} live={live} exact_order={exact}")


# ------------------------------------------------------------------------------------- engagement lines


def test_engagement_lines(monkeypatch, fresh_shared, envset):
    """One plain logger.info line per process for each flag (every TP rank, as the O1s bind line): H-A at the first
    reuse of a shared rank tensor (two layers hold one object), never when off; the no-early-exit line only when set."""
    pytest.importorskip("torch_tpu")
    from vllm_torchtpu.models.vllm.qwen4_exp import qsa as qsa_mod

    ro = fresh_shared
    seen = []

    class Rec:
        def info(self, msg, *args):
            seen.append(msg % args)

    monkeypatch.setattr(ro, "logger", Rec())
    monkeypatch.setattr(ro, "_LOGGED", False)
    envset(FLAG, None)
    ro.rank_operand("dcp_rank", 1)
    ro.rank_operand("dcp_rank", 1)
    assert seen == []
    envset(FLAG, 1)
    ro.rank_operand("dcp_rank", 1)
    assert seen == [], "the first (creating) call is not a reuse"
    ro.rank_operand("dcp_rank", 1)
    ro.rank_operand("request_shard_index", 5)
    ro.rank_operand("request_shard_index", 5)
    assert seen == ["q38 O1s-HOIST shared rank operands: on (dcp_rank 1 reused across QSA layers)"]

    seen.clear()
    monkeypatch.setattr(qsa_mod, "logger", Rec())
    monkeypatch.setattr(qsa_mod, "_Q38_NO_EXIT_LOGGED", False)
    envset(NO_EXIT, None)
    qsa_mod._log_no_early_exit()
    assert seen == []
    envset(NO_EXIT, 1)
    qsa_mod._log_no_early_exit()
    qsa_mod._log_no_early_exit()
    assert seen == ["q38 decode indexer early-exit cond: off (VLLM_TPU_QSA_DECODE_NO_EARLY_EXIT)"]
    import inspect

    assert "_log_no_early_exit()" in inspect.getsource(qsa_mod.Qwen4ExpQSA.__init__)
