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


def _compile(prog, executables):
    args, kw = _inputs(8, 0)
    _set_md(0)
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
    yield
    torch._dynamo.reset()


def _pair(**kw):
    """Two identically initialised programs, each compiled on its own."""
    progs, exes = [], []
    for _ in range(2):
        torch.manual_seed(0)
        p = Prog(**kw)
        e: list = []
        _compile(p, e)
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


def test_direct_matches_normal_bitwise_across_buckets():
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
    # called boxed, as AOTAutograd calls it; the SymInt bucket is in the key.
    assert all(len(e.copies) == 1 and e.boxed for e in entries)
    assert sorted(k[2] for k in direct._interceptor.entries) == [8, 16, 24, 32]


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
