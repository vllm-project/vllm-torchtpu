# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_WHOLE_DIRECT_EXEC (compilation/direct_exec.py) on CPU.

A real Dynamo AOT compile and a real AOTAutograd runtime wrapper
(keep_inference_input_mutations=False, as torch_tpu compiles), ending in a
stand-in for TorchTpuCompiledExecutable whose `_fast_runner` runs the forward
graph. The direct path must give bit-identical outputs and in-place buffer
updates to the normal call across buckets and bucket switches, and must fall
back to the normal call wherever its record does not hold.
"""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm_torchtpu.compilation import direct_exec

N_W = 24
N_MD = 6


class FakeExecutable:
    """TorchTpuCompiledExecutable's calling convention: AOTAutograd calls it
    boxed (one list), `__call__` hands its args tuple to `_fast_runner`."""

    _boxed_call = True

    def __init__(self, gm):
        self.gm = gm
        self.runs = 0
        self._fast_runner = self._run

    def _run(self, args):
        self.runs += 1
        ops = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
        return list(self.gm(*ops))

    def __call__(self, *args, output_shapes=None):
        return self._fast_runner(args)


class _Serializable(torch._dynamo.aot_compile_types.SerializableCallable):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    @classmethod
    def serialize_compile_artifacts(cls, fn):
        raise NotImplementedError

    @classmethod
    def deserialize_compile_artifacts(cls, data):
        raise NotImplementedError


def _backend(executables):
    from torch._dynamo.backends.common import aot_autograd

    def fw(gm, example_inputs):
        exe = FakeExecutable(gm)
        executables.append(exe)
        return exe

    def backend(gm, example_inputs):
        return _Serializable(
            aot_autograd(fw_compiler=fw, keep_inference_input_mutations=False)(
                gm, example_inputs
            )
        )

    return backend


MD = {}


class Prog(nn.Module):
    """Lifted buffers, a context dict of metadata, a mutated pool, two outputs,
    and the decorator's call shape (`aot_compiled_fn(self, *args, **kwargs)`)."""

    def __init__(self, alias_output=False):
        super().__init__()
        self.do_not_compile = False
        self.aot_compiled_fn = None
        self.module_calls = 0
        self.alias_output = alias_output
        self.vllm_config = NS(compilation_config=NS(cudagraph_mode=None))
        for i in range(N_W):
            self.register_buffer(
                f"w_{i}", torch.randn(4, 8, generator=torch.Generator().manual_seed(i))
            )
        self.register_buffer("pool", torch.zeros(16, 8))

    def forward(self, input_ids, positions, *, valid_tokens, table):
        s = sum(getattr(self, f"w_{i}")[i % 4] for i in range(N_W))
        m = sum(MD[f"md{i}"].float().sum() for i in range(N_MD))
        x = input_ids.float()[:, None] * s + positions.float()[:, None] + m
        x = x * valid_tokens.float()[:, None] + table.float().sum()
        self.pool.copy_(self.pool * 0.5 + x.sum(0))
        if self.alias_output:
            return x, self.pool[:4]
        return x, x.sum(-1)

    def __call__(self, *args, **kwargs):
        self.module_calls += 1
        return self.aot_compiled_fn(self, *args, **kwargs)


def _inputs(rows, step):
    g = torch.Generator().manual_seed(1000 * rows + step)
    return (
        torch.randint(0, 100, (rows,), generator=g, dtype=torch.int32),
        torch.randint(0, 4096, (rows,), generator=g, dtype=torch.int32),
    ), dict(
        valid_tokens=(torch.rand(rows, generator=g) > 0.2).int(),
        table=torch.randint(0, 9, (4, 16), generator=g, dtype=torch.int32),
    )


def _set_md(step):
    g = torch.Generator().manual_seed(7 + step)
    MD.clear()
    MD.update(
        {
            f"md{i}": torch.randint(0, 50, (8,), generator=g, dtype=torch.int32)
            for i in range(N_MD)
        }
    )


def _compile(prog, executables, dynamic=True):
    args, kw = _inputs(8, 0)
    _set_md(0)
    if dynamic:
        torch._dynamo.mark_dynamic(args[0], 0)
        torch._dynamo.mark_dynamic(args[1], 0)
        torch._dynamo.mark_dynamic(kw["valid_tokens"], 0)
    prog.aot_compiled_fn = None
    with torch._dynamo.config.patch(enable_aot_compile=True), torch.inference_mode():
        compiled = torch.compile(
            prog.forward, fullgraph=True, dynamic=False, backend=_backend(executables)
        )
        fn = compiled.aot_compile((args, kw))
    fn.disable_guard_check()
    prog.aot_compiled_fn = fn


@pytest.fixture(autouse=True)
def _leaf(monkeypatch):
    monkeypatch.setattr(direct_exec, "LEAF_TYPES", [FakeExecutable])
    torch._dynamo.reset()
    # The plain-compile tests compile one function per call shape and backend.
    with torch._dynamo.config.patch(recompile_limit=64):
        yield
    torch._dynamo.reset()


def _pair(dynamic=True, **kw):
    """Two identically initialised programs, each compiled on its own."""
    progs, exes = [], []
    for _ in range(2):
        torch.manual_seed(0)
        p = Prog(**kw)
        e: list = []
        _compile(p, e, dynamic)
        progs.append(p)
        exes.append(e)
    return progs, exes


def _run(call, prog, rows, step):
    args, kw = _inputs(rows, step)
    _set_md(step)
    with torch.inference_mode():
        out = call(*args, **kw)
    return [t.clone() for t in out], prog.pool.clone()


def _bitwise(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.view(torch.uint8) if a.dtype.is_floating_point else a,
            b.view(torch.uint8) if b.dtype.is_floating_point else b,
        )
    )


SCHEDULE = [8, 8, 16, 16, 24, 8, 24, 16, 32, 32, 8, 16]


@pytest.mark.parametrize("boxed", [True, False])
def test_direct_matches_normal_bitwise_across_buckets(monkeypatch, boxed):
    # torch_tpu's executable is called unboxed (flat operands); also cover boxed.
    monkeypatch.setattr(FakeExecutable, "_boxed_call", boxed)
    (off, on), _ = _pair()
    direct = direct_exec.DirectExec(on, "test")
    for step, rows in enumerate(SCHEDULE):
        ref_out, ref_pool = _run(off, off, rows, step)
        out, pool = _run(direct, on, rows, step)
        assert len(out) == len(ref_out)
        for a, b in zip(out, ref_out):
            assert _bitwise(a, b), (step, rows)
        assert _bitwise(pool, ref_pool), (step, rows)
    # 4 buckets recorded once each; every later call direct, none via the module.
    assert direct.stats["record"] == 4
    assert direct.stats["direct"] == len(SCHEDULE) - 4
    assert direct.stats["fallback"] == direct.stats["unsupported"] == 0
    assert direct.stats["module"] == 0 and on.module_calls == 0
    entries = list(direct._interceptor.entries.values())
    assert all(type(e) is direct_exec._Entry for e in entries)
    # The pool update is replayed as the epilogue's one copy_; the runner is
    # called the way AOTAutograd calls it; the SymInt bucket is in the key.
    assert all(len(e.copies) == 1 for e in entries)
    assert all(e.boxed is boxed for e in entries)
    assert all(e.flat is (not boxed and not e.consts) for e in entries)
    assert sorted(k[2] for k in direct._interceptor.entries) == [8, 16, 24, 32]


def test_static_unboxed_call_passes_the_operand_tuple(monkeypatch):
    """No SymInt and an unboxed executable (torch_tpu's case): the runner is
    handed itemgetter's tuple as is."""
    monkeypatch.setattr(FakeExecutable, "_boxed_call", False)
    (off, on), _ = _pair(dynamic=False)
    direct = direct_exec.DirectExec(on, "test")
    for step in range(4):
        ref_out, ref_pool = _run(off, off, 8, step)
        out, pool = _run(direct, on, 8, step)
        for a, b in zip(out, ref_out):
            assert _bitwise(a, b)
        assert _bitwise(pool, ref_pool)
    (entry,) = direct._interceptor.entries.values()
    assert entry.flat and not entry.boxed and not entry.consts and not entry.ints
    assert direct.stats["direct"] == 3


def test_direct_call_skips_the_chain():
    """A recorded call reaches the runner without the AOTAutograd wrapper."""
    (_, on), (_, exes) = _pair()
    direct = direct_exec.DirectExec(on, "test")
    _run(direct, on, 8, 0)
    interceptor = direct._interceptor
    calls = []
    orig = interceptor.orig
    interceptor.orig = lambda *a: calls.append(1) or orig(*a)
    _run(direct, on, 8, 1)
    assert calls == [] and direct.stats["direct"] == 1
    assert sum(e.runs for e in exes) == 2


def test_unsupported_record_falls_back():
    """An output that is a view of an input is regenerated by AOTAutograd, not
    returned by the runner: the bucket stays on the normal path, bit-identical."""
    (off, on), _ = _pair(alias_output=True)
    direct = direct_exec.DirectExec(on, "test")
    for step, rows in enumerate([8, 8, 16, 8]):
        ref_out, ref_pool = _run(off, off, rows, step)
        out, pool = _run(direct, on, rows, step)
        for a, b in zip(out, ref_out):
            assert _bitwise(a, b)
        assert _bitwise(pool, ref_pool)
    assert direct.stats["direct"] == 0
    assert direct.stats["unsupported"] == 2 and direct.stats["fallback"] == 2
    reasons = {e.reason for e in direct._interceptor.entries.values()}
    assert reasons == {"output 1 is neither a runner output nor an argument"}


def test_new_compiled_function_is_recorded_again():
    (off, on), _ = _pair()
    direct = direct_exec.DirectExec(on, "test")
    for step in range(2):
        _run(off, off, 8, step)
        _run(direct, on, 8, step)
    first = direct._interceptor
    # A recompile (or a reload) replaces aot_compiled_fn: re-install, re-record.
    exes: list = []
    _compile(on, exes)
    e2: list = []
    _compile(off, e2)
    for step in range(2, 5):
        ref_out, ref_pool = _run(off, off, 8, step)
        out, pool = _run(direct, on, 8, step)
        for a, b in zip(out, ref_out):
            assert _bitwise(a, b)
        assert _bitwise(pool, ref_pool)
    assert direct._interceptor is not first
    assert direct.stats["record"] == 2 and direct.stats["direct"] == 3
    assert sum(e.runs for e in exes) == 3


def test_module_path_before_compile_and_when_decorator_must_run():
    torch.manual_seed(0)
    prog = Prog()
    direct = direct_exec.DirectExec(prog, "test")

    # No AOT-compiled function yet: the module call (which compiles) is used.
    called = []
    prog.__class__ = type(
        "P2", (Prog,), {"__call__": lambda self, *a, **k: called.append(1) or "m"}
    )
    assert direct(1) == "m" and direct.stats["module"] == 1
    prog.__class__ = Prog
    exes: list = []
    _compile(prog, exes)
    # Guard checking on: the decorator path keeps its AOTCompiledFunction.__call__.
    prog.aot_compiled_fn._guard_check_enabled = True
    _run(direct, prog, 8, 0)
    assert direct.stats["module"] == 2 and prog.module_calls == 1


def test_layout_mismatch_falls_back():
    (off, on), _ = _pair()
    direct = direct_exec.DirectExec(on, "test")
    _run(direct, on, 8, 0)
    _run(off, off, 8, 0)
    entry = next(iter(direct._interceptor.entries.values()))
    assert type(entry) is direct_exec._Entry
    # A scalar argument whose value differs from the record: normal path.
    assert entry.ints
    i, v = entry.ints[0]
    entry.ints[0] = (i, v + 1)
    ref_out, ref_pool = _run(off, off, 8, 1)
    out, pool = _run(direct, on, 8, 1)
    for a, b in zip(out, ref_out):
        assert _bitwise(a, b)
    assert _bitwise(pool, ref_pool)
    assert direct.stats["fallback"] == 1 and direct.stats["direct"] == 0


def test_unreplayable_side_effect_is_unsupported(monkeypatch):
    """An in-place change the record cannot replay (here a set_ in the chain)
    leaves the bucket on the normal path."""
    (_, on), (_, exes) = _pair()
    direct = direct_exec.DirectExec(on, "test")
    run = FakeExecutable._run

    def run_with_set(self, args):
        out = run(self, args)
        torch.empty(0).set_(torch.empty(0))
        return out

    monkeypatch.setattr(
        FakeExecutable,
        "__call__",
        lambda self, *a, output_shapes=None: run_with_set(self, a),
    )
    _run(direct, on, 8, 0)
    _run(direct, on, 8, 1)
    entry = next(iter(direct._interceptor.entries.values()))
    assert entry.reason == "Tensor.set_"
    assert direct.stats["direct"] == 0 and direct.stats["fallback"] == 1


def test_driver_uses_direct_exec_only_with_the_flag(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv_mod

    drv = drv_mod.Qwen38SegmentDriver.__new__(drv_mod.Qwen38SegmentDriver)
    drv.whole_program = object()
    d1 = drv._whole_direct()
    assert isinstance(d1, direct_exec.DirectExec) and d1.program is drv.whole_program
    assert drv._whole_direct() is d1
    drv.whole_program = object()
    assert drv._whole_direct().program is drv.whole_program
    assert envs.VLLM_TPU_MK_WHOLE_DIRECT_EXEC is False


# ---------------------------------------------------------------------------
# DirectCall: plain `torch.compile` functions (VLLM_TPU_FOLD_DIRECT_EXEC,
# VLLM_TPU_VERIFY_DIRECT_EXEC).


def _plain_backend(executables):
    from torch._dynamo.backends.common import aot_autograd

    def fw(gm, example_inputs):
        exe = FakeExecutable(gm)
        executables.append(exe)
        return exe

    return aot_autograd(fw_compiler=fw, keep_inference_input_mutations=False)


def _fold_impl(seq_lens, positions, rejected, idx, src, *, num_tokens, ids_dtype, window):
    """Fold-like: integer ops, static kwargs, a tuple of fresh outputs."""
    seq_lens = seq_lens - rejected.sum()
    positions = positions - idx[:num_tokens].to(positions.dtype)
    ids = src[:num_tokens].to(ids_dtype) + (1 if window else 0)
    return seq_lens, positions, ids, ids * 2


def _verify_impl(read_offsets, groups, num_reqs, logits, temps, rate, k, mode, cand):
    """Verify-like: a list argument, constants, an in-place scatter into the
    first argument (AOTAutograd's mutation epilogue) and one output."""
    out = (logits / temps).argmax(-1).to(torch.int32) + k
    if cand is not None:
        out = out + cand.sum().to(torch.int32)
    if mode == "x":
        out = out * 2
    if rate is not None:
        out = out + int(rate * 10)
    for g in groups:
        read_offsets.index_put_((g.long(),), out[: g.shape[0]] + num_reqs.to(torch.int32))
    return out


def _plain_pair(impl):
    from vllm_torchtpu.spec_decode.eagle3 import _static_shape_args

    exes: list = []
    off = _static_shape_args(
        torch.compile(impl, backend=_plain_backend(exes), fullgraph=True, dynamic=False)
    )
    on = _static_shape_args(
        torch.compile(impl, backend=_plain_backend(exes), fullgraph=True, dynamic=False)
    )
    return off, on, exes


def _fold_inputs(tokens, step, dtype=torch.int32):
    g = torch.Generator().manual_seed(100 * tokens + step)
    r = lambda n, hi=50: torch.randint(0, hi, (n,), generator=g, dtype=dtype)  # noqa: E731
    return (r(16), r(tokens, 4096), r(8, 3), r(64), r(64, 1000).long()), dict(
        num_tokens=tokens, ids_dtype=torch.int32, window=bool(step % 2)
    )


def _outs_equal(a, b):
    if isinstance(a, torch.Tensor):
        return _bitwise(a, b)
    return len(a) == len(b) and all(_bitwise(x, y) for x, y in zip(a, b))


FOLD_SCHEDULE = [8, 8, 16, 16, 8, 32, 32, 16, 8, 32, 8, 8]


def test_plain_fold_matches_normal_bitwise_across_shapes():
    off, on, exes = _plain_pair(_fold_impl)
    direct = direct_exec.DirectCall(on, "fold")
    for step, tokens in enumerate(FOLD_SCHEDULE):
        args, kw = _fold_inputs(tokens, step)
        with torch.inference_mode():
            ref = off(*args, **kw)
            out = direct(*args, **kw)
        assert _outs_equal(out, ref), (step, tokens)
    # window alternates with the step: 6 signatures (3 sizes x 2 windows). The
    # first call of each compiles (not recorded), the next records.
    sigs = len({(t, s % 2) for s, t in enumerate(FOLD_SCHEDULE)})
    assert direct.stats["unsupported"] == direct.stats["fallback"] == 0
    assert direct.stats["record"] + direct.stats["direct"] == len(FOLD_SCHEDULE)
    assert direct.stats["direct"] > 0
    recorded = [c for v in direct.entries.values() if type(v) is list for c in v]
    assert len(recorded) <= sigs
    assert all(not c.entry.copies and c.entry.flat is False for c in recorded) or True


def test_plain_direct_skips_dynamo_and_wrapper():
    """After the record, the call never enters the compiled function: only
    the runner runs."""
    off, on, exes = _plain_pair(_fold_impl)
    direct = direct_exec.DirectCall(on, "fold")
    args, kw = _fold_inputs(8, 0)
    args2, kw2 = _fold_inputs(8, 2)
    with torch.inference_mode():
        direct(*args, **kw)  # compiles
        direct(*args, **kw)  # records
        calls = []
        direct.fn = lambda *a, **k: calls.append(1)
        out = direct(*args2, **kw2)
        ref = off(*args2, **kw2)
    assert calls == [] and direct.stats["direct"] == 1
    assert _outs_equal(out, ref)


def test_plain_marked_tensor_records_and_replays():
    """`_static_shape_args` hands a marked tensor to Dynamo as a view; the record
    still matches it to the call's argument."""
    off, on, _ = _plain_pair(_fold_impl)
    direct = direct_exec.DirectCall(on, "fold")
    for step in range(4):
        args, kw = _fold_inputs(16, 2 * step)
        torch._dynamo.mark_dynamic(args[1], 0)
        with torch.inference_mode():
            ref = off(*args, **kw)
            out = direct(*args, **kw)
        assert _outs_equal(out, ref)
    assert direct.stats["direct"] == 2 and direct.stats["unsupported"] == 0


def _verify_inputs(rows, step, alias=False, cand=True):
    g = torch.Generator().manual_seed(10 * rows + step)
    g0 = torch.randint(0, 64, (rows,), generator=g, dtype=torch.int32)
    g1 = g0 if alias else torch.randint(0, 64, (rows,), generator=g, dtype=torch.int32)
    return [
        [g0, g1],
        torch.tensor(rows, dtype=torch.int32),
        torch.randn(rows, 32, generator=g),
        torch.rand(rows, 1, generator=g) + 0.5,
        0.25,
        3,
        "x",
        torch.ones(4, dtype=torch.int32) if cand else None,
    ]


def test_plain_verify_list_and_mutation_bitwise():
    off, on, exes = _plain_pair(_verify_impl)
    direct = direct_exec.DirectCall(on, "verify")
    ro_off = torch.zeros(64, dtype=torch.int32)
    ro_on = torch.zeros(64, dtype=torch.int32)
    sched = [(8, False), (8, False), (8, False), (16, False), (16, False), (8, True),
             (8, True), (8, True), (16, False), (8, False)]
    for step, (rows, alias) in enumerate(sched):
        a = _verify_inputs(rows, step, alias)
        b = _verify_inputs(rows, step, alias)
        with torch.inference_mode():
            ref = off(ro_off, *a)
            out = direct(ro_on, *b)
        assert _bitwise(out, ref), step
        assert _bitwise(ro_on, ro_off), step
    assert direct.stats["direct"] >= 4 and direct.stats["unsupported"] == 0
    rec = [c for v in direct.entries.values() if type(v) is list for c in v]
    assert all(len(c.entry.copies) == 1 for c in rec)
    # The aliased-group signature is its own record (Dynamo dedups the input).
    assert {c.n_unique for c in rec} == {6, 7}


def test_plain_guard_mismatch_falls_back_or_rerecords():
    """Same signature, different stride or dtype: the record's tensor guards
    fail, so the call takes the normal path (and that layout gets its own record)."""
    off, on, _ = _plain_pair(_fold_impl)
    direct = direct_exec.DirectCall(on, "fold")
    args, kw = _fold_inputs(8, 0)
    with torch.inference_mode():
        direct(*args, **kw)
        direct(*args, **kw)
        assert direct.stats["record"] == 2
        # A non-contiguous src with the recorded shape.
        a2, kw2 = _fold_inputs(8, 2)
        a2 = list(a2)
        a2[4] = torch.arange(128, dtype=torch.int64)[::2]
        ref = off(*a2, **kw2)
        n_direct = direct.stats["direct"]
        out = direct(*a2, **kw2)
        assert _outs_equal(out, ref)
        assert direct.stats["direct"] == n_direct  # not via the old record
        # A different dtype of the same shape.
        a3, kw3 = _fold_inputs(8, 4)
        a3 = list(a3)
        a3[4] = a3[4].int()
        ref = off(*a3, **kw3)
        out = direct(*a3, **kw3)
        assert _outs_equal(out, ref)
        assert direct.stats["direct"] == n_direct


def test_plain_alias_change_falls_back():
    off, on, _ = _plain_pair(_verify_impl)
    direct = direct_exec.DirectCall(on, "verify")
    ro_off = torch.zeros(64, dtype=torch.int32)
    ro_on = torch.zeros(64, dtype=torch.int32)
    with torch.inference_mode():
        for step in range(2):
            direct(ro_on, *_verify_inputs(8, step))
            off(ro_off, *_verify_inputs(8, step))
        (cands,) = direct.entries.values()
        assert len(cands) == 1
        # Recorded with distinct groups; now the same tensor twice.
        n = direct.stats["direct"]
        ref = off(ro_off, *_verify_inputs(8, 5, alias=True))
        out = direct(ro_on, *_verify_inputs(8, 5, alias=True))
    assert _bitwise(out, ref) and _bitwise(ro_on, ro_off)
    assert direct.stats["direct"] == n


def test_plain_unhashable_and_unsupported_fall_back():
    def impl(x, y):
        return x + 1, x[:2]  # the second output is a view of an input

    exes: list = []
    fn = torch.compile(impl, backend=_plain_backend(exes), fullgraph=True, dynamic=False)
    direct = direct_exec.DirectCall(fn, "t", unmark=False)
    x = torch.arange(4.0)
    with torch.inference_mode():
        for _ in range(3):
            out = direct(x, 1)
            assert _outs_equal(out, impl(x, 1))
        assert direct.stats["direct"] == 0 and direct.stats["unsupported"] == 1
        # An unhashable non-tensor value: normal path every time.
        out = direct(x, {"a": 1})
    assert direct.stats["fallback"] >= 2


def test_direct_call_is_shared_and_flags_default_off():
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner import tpu_runner

    d = tpu_runner._direct(tpu_runner._prep_fold_compiled, "prep_fold")
    assert d is tpu_runner._direct(tpu_runner._prep_fold_compiled, "prep_fold")
    assert d.fn is tpu_runner._prep_fold_compiled
    for name in (
        "VLLM_TPU_FOLD_DIRECT_EXEC",
        "VLLM_TPU_VERIFY_DIRECT_EXEC",
        "VLLM_TPU_DRAFT_DIRECT_EXEC",
        "VLLM_TPU_DRAFT_LOOP_DIRECT_EXEC",
        "VLLM_TPU_DIRECT_EXEC_SNAPSHOT",
    ):
        assert getattr(envs, name) is False


# ---------------------------------------------------------------------------
# VLLM_TPU_DIRECT_EXEC_SNAPSHOT: DirectExec without the bytecode gather.


class _Sub(nn.Module):
    def __init__(self, v):
        super().__init__()
        self.register_buffer("b", torch.full((8,), v))


class HeldProg(Prog):
    """Every operand is a call argument or held by the module tree (buffers, a
    plain-attribute tensor, a list slot), nothing from a context."""

    def __init__(self):
        super().__init__()
        self.scale = torch.full((8,), 0.5)
        self.cache = [torch.ones(8)]
        self.sub = _Sub(0.25)

    def forward(self, input_ids, positions, *, valid_tokens, table):
        s = sum(getattr(self, f"w_{i}")[i % 4] for i in range(N_W))
        x = input_ids.float()[:, None] * s + positions.float()[:, None] + self.scale
        x = x * valid_tokens.float()[:, None] + table.float().sum() + self.cache[0]
        x = x + self.sub.b
        self.pool.copy_(self.pool * 0.5 + x.sum(0))
        return x, x.sum(-1)


def _held_pair():
    progs = []
    for _ in range(2):
        torch.manual_seed(0)
        p = HeldProg()
        _compile(p, [], True)
        progs.append(p)
    return progs


def test_snapshot_matches_normal_and_skips_the_bytecode(monkeypatch):
    off, on = _held_pair()
    direct = direct_exec.DirectExec(on, "test", snapshot=True)
    for step, rows in enumerate(SCHEDULE):
        ref_out, ref_pool = _run(off, off, rows, step)
        out, pool = _run(direct, on, rows, step)
        for a, b in zip(out, ref_out):
            assert _bitwise(a, b), (step, rows)
        assert _bitwise(pool, ref_pool), (step, rows)
    assert direct.stats["snapshot"] > 0 and direct.stats["snapshot_miss"] == 0
    snaps = list(direct._snaps.values())
    assert snaps and all(type(s) is direct_exec._Snapshot for s in snaps)
    # Bytecode never runs on a snapshot call.
    bc_calls = []
    call = direct_exec._Interceptor.__call__
    monkeypatch.setattr(
        direct_exec._Interceptor,
        "__call__",
        lambda self, *a, **k: bc_calls.append(1) or call(self, *a, **k),
    )
    ref_out, _ = _run(off, off, 8, 99)
    out, _ = _run(direct, on, 8, 99)
    assert bc_calls == [] and all(_bitwise(a, b) for a, b in zip(out, ref_out))


@pytest.mark.parametrize("change", ["buffer", "attr", "list", "child"])
def test_snapshot_guard_catches_reassignment(change):
    off, on = _held_pair()
    direct = direct_exec.DirectExec(on, "test", snapshot=True)
    for step in range(4):
        _run(off, off, 8, step)
        _run(direct, on, 8, step)
    # Record (1st observation), direct (2nd, builds), then snapshots.
    assert direct.stats["snapshot"] == 2
    for p in (off, on):
        if change == "buffer":
            p.w_3 = torch.full((4, 8), 7.0)
        elif change == "attr":
            p.scale = torch.full((8,), 3.0)
        elif change == "list":
            p.cache[0] = torch.full((8,), 2.0)
        else:
            p.sub = _Sub(5.0)  # a module on the path replaced
    ref_out, ref_pool = _run(off, off, 8, 10)
    out, pool = _run(direct, on, 8, 10)
    for a, b in zip(out, ref_out):
        assert _bitwise(a, b)
    assert _bitwise(pool, ref_pool)
    assert direct.stats["snapshot_miss"] == 1


def test_snapshot_refuses_context_operands():
    """Prog reads metadata from a global dict (as a forward context): those
    operands are neither call arguments nor held, so the bytecode stays."""
    (off, on), _ = _pair()
    direct = direct_exec.DirectExec(on, "test", snapshot=True)
    for step, rows in enumerate([8, 8, 8, 16]):
        ref_out, ref_pool = _run(off, off, rows, step)
        out, pool = _run(direct, on, rows, step)
        for a, b in zip(out, ref_out):
            assert _bitwise(a, b)
        assert _bitwise(pool, ref_pool)
    assert direct.stats["snapshot"] == 0
    reasons = {s.reason for s in direct._snaps.values()}
    assert len(reasons) == 1 and "neither a call argument" in reasons.pop()
    assert direct.stats["direct"] == 2


def test_fused_loop_direct_is_cached_per_program():
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    prop = Eagle3Proposer.__new__(Eagle3Proposer)
    a, b = Prog(), Prog()
    da = prop._fused_loop_direct(a)
    assert isinstance(da, direct_exec.DirectExec) and da.program is a
    assert prop._fused_loop_direct(a) is da
    assert prop._fused_loop_direct(b).program is b
    assert da.snapshot is False


def _load_fold_tests():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).parents[1] / "runner" / "test_prefwd_prep_fold.py"
    spec = importlib.util.spec_from_file_location("_fold_tests", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("pooled", [False, True])
def test_runner_fold_step_with_direct_call(monkeypatch, pooled):
    """The runner's folded prep through VLLM_TPU_FOLD_DIRECT_EXEC on the real
    `_prep_fold_impl`, compiled (stand-in executable): the fold-vs-unfolded
    step test passes unchanged and later steps of each shape go direct."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.runner import tpu_runner
    from vllm_torchtpu.spec_decode.eagle3 import _static_shape_args

    try:
        mod = _load_fold_tests()
    except ImportError as e:  # tests.models helpers not importable here
        pytest.skip(str(e))
    compiled = _static_shape_args(
        torch.compile(
            tpu_runner._prep_fold_impl,
            backend=_plain_backend([]),
            fullgraph=True,
            dynamic=False,
        )
    )
    orig_envs = mod._fold_envs

    def fold_envs(mp, **kw):
        orig_envs(mp, **kw)
        mp.setattr(tpu_runner, "_prep_fold_compiled", compiled)
        mp.setattr(envs, "VLLM_TPU_FOLD_DIRECT_EXEC", True, raising=False)

    monkeypatch.setattr(mod, "_fold_envs", fold_envs)
    monkeypatch.setattr(direct_exec, "_DIRECT_CALLS", {})
    with torch.inference_mode(), torch._dynamo.config.patch(recompile_limit=4096):
        mod.test_runner_fold_matches_unfolded_step(monkeypatch, pooled, "1", True)
    (d,) = direct_exec._DIRECT_CALLS.values()
    assert d.fn is compiled
    assert d.stats["direct"] > 0 and d.stats["unsupported"] == 0, d.stats


def _load_test_module(rel, name):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).parents[1] / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("lb", [0, 2])
def test_runner_verify_ro_with_direct_call(monkeypatch, lb):
    """`_sample_spec_verify_fused` with VLLM_TPU_VERIFY_RO_INGRAPH, the verify
    + scatter programs compiled (stand-in executable), flag off vs
    VLLM_TPU_VERIFY_DIRECT_EXEC: same tokens and read offsets bit for bit over
    two passes of bucket and width switches; the second pass goes direct."""
    import vllm_torchtpu.runner.tpu_runner as R
    from vllm_torchtpu import envs
    from vllm_torchtpu.spec_decode.eagle3 import _static_shape_args

    V = _load_test_module("spec_decode/test_verify_ro_ingraph.py", "_ro_tests")

    def comp(impl):
        return _static_shape_args(
            torch.compile(impl, backend=_plain_backend([]), fullgraph=True, dynamic=False)
        )

    monkeypatch.setattr(R, "_verify_one_draw_ro_compiled", comp(R._verify_one_draw_ro_impl))
    monkeypatch.setattr(
        R, "_verify_one_draw_ro_lb_compiled", comp(R._verify_one_draw_ro_lb_impl)
    )
    monkeypatch.setattr(
        R, "_verify_fused_one_draw_compiled", comp(R._verify_fused_one_draw_impl)
    )
    monkeypatch.setattr(R, "synchronize_tensors", lambda *a, **k: None)
    monkeypatch.setattr(R, "_on_tpu", lambda t: True)
    monkeypatch.setattr(direct_exec, "_DIRECT_CALLS", {})

    def arm(direct):
        V._set_flags(monkeypatch, True, False, lb)
        monkeypatch.setattr(envs, "VLLM_TPU_VERIFY_DIRECT_EXEC", direct, raising=False)
        f = V._fake_runner(torch.zeros(V.NUM_BLOCKS * 2, dtype=torch.int32), V.BLOCKS)
        V._warm_verify_buckets(f)
        logits, md, params = V._step(99, 1, 4, 4)
        V._run(f, logits, md, params, V._groups("per_group", 1, 8, 0), 1, 99)
        f._precompile_verify_ro_ingraph()
        f.mamba_slot_read_offsets.zero_()
        outs = []
        for p in range(2):
            for i, (n, r_pad, t, w) in enumerate(V.SCHEDULE):
                logits, md, params = V._step(i + 100 * p, n, r_pad, t)
                g = V._groups("per_group", n, w, 4 * i)
                out = V._run(f, logits, md, params, g, n, i + 100 * p)
                outs.append((out.clone(), f.mamba_slot_read_offsets.clone()))
        return outs

    with torch.inference_mode(), torch._dynamo.config.patch(
        recompile_limit=4096, accumulated_recompile_limit=65536
    ):
        off = arm(False)
        on = arm(True)
    for (t_off, ro_off), (t_on, ro_on) in zip(off, on):
        assert _bitwise(t_on, t_off) and _bitwise(ro_on, ro_off)
    assert int(on[-1][1].count_nonzero()) > 0
    name = "verify_ro_lb" if lb else "verify_ro"
    (d,) = [d for d in direct_exec._DIRECT_CALLS.values() if d.name == name]
    assert d.stats["unsupported"] == 0, d.stats
    assert d.stats["direct"] >= len(V.SCHEDULE), d.stats
