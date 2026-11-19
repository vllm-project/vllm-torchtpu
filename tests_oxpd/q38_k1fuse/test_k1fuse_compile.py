# SPDX-License-Identifier: Apache-2.0
"""q38 L4 T1-K1FUSE: the program is guard-free (CDK CPU pod only).

The product build (``k1fuse.build_k1fuse_program``, captured before any harness patch) is instantiated under a CPU
vLLM config in VLLM_COMPILE mode, with a counting eager backend in place of VllmBackend (``init_backend``): the
real vLLM 0.31 ``support_torch_compile`` / ``TorchCompileWithNoGuardsWrapper`` path (guard_filter_fn drops every
guard; T marked dynamic on every token-dim input). It is then called on four serving steps' real K1 arguments
(fresh tensors, fresh forward-context metadata objects) while a Python value the draft reads changes per step (as a
per-step forward-context object or field would): ONE Dynamo trace, and each step's outputs equal the uncompiled
body's. The positive control, a guarded ``torch.compile`` of the same program class, retraces on those steps.
"""

import pytest
import torch

import k1fuse_harness as kh
from vllm_torchtpu.spec_decode import k1fuse

ORIG_BUILD = k1fuse.build_k1fuse_program
STEP = {"n": 0}


class FakeMtpStep(kh.FakeMtpKV):
    """FakeMtpKV that also reads a per-step Python value (it changes no number: x + 0 * n)."""

    def forward(self, *a, **kw):
        hh, nxt = super().forward(*a, **kw)
        return hh + 0 * STEP["n"], nxt


def _capture(monkeypatch, steps=4, R=20, F=8):
    kh.install(monkeypatch, k1=True)
    kp = kh.make_pair_proposer("k1", F=F, mtp_cls=FakeMtpStep)
    kh.warm(kp, [R])
    kh.agree(kp)
    caps = []
    ctl = kp._draft_asm
    real = ctl._k1_call

    def grab(module, md, args, T, serves):
        caps.append((md, args, T, module.statics))
        return real(module, md, args, T, serves)

    monkeypatch.setattr(ctl, "_k1_call", grab)
    for i in range(steps):
        kh.serve(kp, kh.make_vstep(50000 + i, R), F, 50000 + i)
    assert len(caps) == steps
    return kp, caps


def _cpu_config(monkeypatch, backend):
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, DeviceConfig, VllmConfig

    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    cfg.compilation_config.mode = CompilationMode.VLLM_COMPILE
    cfg.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    monkeypatch.setattr(CompilationConfig, "init_backend",
                        lambda self, vllm_config, prefix="", is_encoder=False: backend)
    return cfg


def _run(kp, prog, md, args, i):
    """One call with step value i, the draft KV and the buffers restored afterwards (the caller compares)."""
    snap = [b.clone() for b in args[-3:]]
    kv = kp.draft_model.kv.clone()
    STEP["n"] = i
    with kh.h.fake_forward_context(md):
        out = tuple(prog(*args))
    for b, s in zip(args[-3:], snap):
        b.copy_(s)
    kp.draft_model.kv.copy_(kv)
    return out


def test_k1_program_compiles_once_guard_free_across_steps(monkeypatch):
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
    from vllm.config import set_current_vllm_config

    kp, caps = _capture(monkeypatch)
    traces = []

    def backend(gm, example_inputs):
        traces.append(len(example_inputs))
        return gm.forward

    cfg = _cpu_config(monkeypatch, backend)
    statics = caps[0][3]
    with set_current_vllm_config(cfg):
        prog = ORIG_BUILD(vllm_config=cfg, prefix="k1fuse_cpu_test_0", mtp=kp.draft_model, num_steps=kh.K - 1,
                          dvp=False, vh=kh.VH, statics=statics, index=900)
        assert isinstance(prog, TorchCompileWithNoGuardsWrapper) and not prog.do_not_compile
        assert prog._dynamic_arg_dims == k1fuse.K1_DYNAMIC_ARG_DIMS
        eager = k1fuse._program_class(901)(vllm_config=None, prefix="k1fuse_cpu_ref", mtp=kp.draft_model,
                                           num_steps=kh.K - 1, dvp=False, vh=kh.VH, statics=statics)
        for i, (md, args, T, st) in enumerate(caps):
            assert st == statics
            want = _run(kp, eager, md, args, i)
            got = _run(kp, prog, md, args, i)
            for a, b in zip(got, want):
                kh.same(a, b, f"compiled K1 step {i}")
    assert len(traces) == 1, traces
    print(f"\nK1FUSE_GUARDFREE support_torch_compile (vLLM no-guards wrapper, T dynamic): {len(caps)} steps with fresh "
          f"metadata objects and a per-step Python value -> {len(traces)} trace; outputs equal the uncompiled body")


def test_a_guarded_compile_of_the_same_program_retraces(monkeypatch):
    kp, caps = _capture(monkeypatch)
    traces = []

    def backend(gm, example_inputs):
        traces.append(len(example_inputs))
        return gm.forward

    statics = caps[0][3]
    m = k1fuse._program_class(902)(vllm_config=None, prefix="k1fuse_cpu_guarded", mtp=kp.draft_model,
                                   num_steps=kh.K - 1, dvp=False, vh=kh.VH, statics=statics)
    guarded = torch.compile(m, backend=backend, fullgraph=True, dynamic=False)
    for i, (md, args, T, st) in enumerate(caps):
        _run(kp, guarded, md, args, i)
    assert len(traces) == len(caps), traces
    print(f"\nK1FUSE_GUARDED positive control: a guarded torch.compile retraced on every one of {len(caps)} steps "
          f"({len(traces)} traces)")


def test_build_failure_stops(monkeypatch):
    kh.install(monkeypatch, k1=True)
    kp = kh.make_pair_proposer("k1", F=0)

    def boom(**kw):
        raise ValueError("no program")

    monkeypatch.setattr(k1fuse, "build_k1fuse_program", boom)
    with pytest.raises(RuntimeError, match="could not build the verify"):
        kh.warm(kp, [8])
