# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_FORWARD_CTX_PREBUILT: the main forward must see the same vLLM
forward context and model wrapper context as under the original context
stack, and both globals must be restored the same way, every step."""

import contextlib
import dataclasses
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.forward_context as vfc

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm import vllm_model_wrapper_context as wctx
from vllm_torchtpu.runner import tpu_runner as tr
from vllm_torchtpu.runner.stage_timing import RunnerStageTiming


def _runner(dp=1, sp_moe=False):
    r = object.__new__(tr.TPUModelRunner)
    parallel = SimpleNamespace(
        data_parallel_size=dp,
        use_sequence_parallel_moe=sp_moe,
        is_moe_model=True,
        enable_expert_parallel=True,
    )
    r.vllm_config = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            fast_moe_cold_start=False,
            static_forward_context={"layer": object()},
            static_all_moe_layers=["moe.0"],
        ),
        model_config=None,
        additional_config={},
    )
    r.parallel_config = parallel
    r.mesh = object()
    r._kv_cache_bundle = torch.zeros(4)
    # The real class without its device-timing thread (no TPU here); only
    # `_has_sink` / `stage` are exercised, and only without a sink.
    timing = object.__new__(RunnerStageTiming)
    timing._sink = False
    r._stage_timing = timing
    r.position_ids = torch.arange(8)
    return r


def _observe(r, attn_metadata, num_tokens, on, trace_kwargs=None, raise_in=False):
    """Run the forward under the flag's path; return what the forward saw."""
    seen = {}

    def forward_model(**kwargs):
        fc = vfc.get_forward_context()
        wc = wctx.get_vllm_model_wrapper_context()
        seen["fc"] = fc
        seen["fc_fields"] = {
            f.name: getattr(fc, f.name) for f in dataclasses.fields(fc)
        }
        seen["wc"] = wc
        seen["wc_fields"] = vars(wc).copy()
        seen["kwargs"] = kwargs
        if raise_in:
            raise RuntimeError("forward failed")
        return "hidden", None

    r.forward_model = forward_model
    trace_kwargs = {} if trace_kwargs is None else trace_kwargs
    env = {"VLLM_TPU_FORWARD_CTX_PREBUILT": "1" if on else "0"}
    with patch.dict(os.environ, env):
        scope = (
            r._prebuilt_forward_scope(attn_metadata, num_tokens, trace_kwargs)
            if envs.VLLM_TPU_FORWARD_CTX_PREBUILT
            else None
        )
        seen["prebuilt"] = scope is not None
        args = (16, num_tokens, trace_kwargs, "ids", None, None, {"k": 1})
        try:
            if scope is not None:
                with scope:
                    seen["out"] = r.forward_model(
                        input_ids="ids",
                        positions=r.position_ids,
                        inputs_embeds=None,
                        intermediate_tensors=None,
                        model_kwargs={"k": 1},
                    )
            else:
                seen["out"] = r._forward_in_contexts(attn_metadata, *args)
        except RuntimeError as e:
            seen["raised"] = str(e)
    del r.forward_model
    return seen


def _assert_same_view(a, b, md):
    assert a.get("out") == b.get("out")
    assert a.get("raised") == b.get("raised")
    fa, fb = a["fc_fields"], b["fc_fields"]
    assert fa.keys() == fb.keys()
    for name in fa:
        if name in ("slot_mapping", "additional_kwargs"):
            # Fresh empty dicts per forward on both paths.
            assert fa[name] == fb[name] == {} and fa[name] is not fb[name]
        else:
            assert fa[name] is fb[name] or fa[name] == fb[name], name
    assert fa["attn_metadata"] is md and fb["attn_metadata"] is md
    assert a["wc_fields"].keys() == b["wc_fields"].keys()
    for name in a["wc_fields"]:
        assert a["wc_fields"][name] is b["wc_fields"][name], name
    assert type(a["fc"]) is type(b["fc"]) and a["fc"] is not b["fc"]
    assert type(a["wc"]) is type(b["wc"])
    assert a["kwargs"].keys() == b["kwargs"].keys()


@pytest.fixture
def stage_timing_only_with_sink():
    with patch.dict(os.environ, {"VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK": "1"}):
        yield


def test_same_contexts_every_step(stage_timing_only_with_sink):
    r = _runner()
    outer_fc = vfc.create_forward_context({"outer": 1}, r.vllm_config)
    outer_wc = wctx.VllmModelWrapperContext(mesh="outer")
    prev_fcs = []
    with vfc.override_forward_context(outer_fc):
        wctx._vllm_model_wrapper_context = outer_wc
        try:
            for step in range(20):
                md = {f"layer.{i}": object() for i in range(5)}
                n = (16, 32, 64)[step % 3]
                raise_in = step % 7 == 6
                off = _observe(r, md, n, on=False, raise_in=raise_in)
                assert vfc._forward_context is outer_fc
                assert wctx._vllm_model_wrapper_context is outer_wc
                on = _observe(r, md, n, on=True, raise_in=raise_in)
                assert on["prebuilt"] and not off["prebuilt"]
                assert vfc._forward_context is outer_fc
                assert wctx._vllm_model_wrapper_context is outer_wc
                _assert_same_view(off, on, md)
                # A new ForwardContext every step (the forward may write
                # into it, e.g. vLLM's MoE layer index).
                assert all(on["fc"] is not p for p in prev_fcs)
                prev_fcs.append(on["fc"])
                # The wrapper is rebuilt when its inputs change.
                if step == 10:
                    r._kv_cache_bundle = torch.ones(4)
        finally:
            wctx._vllm_model_wrapper_context = None
    assert vfc._forward_context is None


def test_falls_back_when_a_dropped_piece_would_act(stage_timing_only_with_sink):
    md = {"layer.0": object()}
    r = _runner()
    assert _observe(r, md, 16, on=True)["prebuilt"]
    # Trace kwargs (phase profiler stats) or an active profiler.
    assert not _observe(r, md, 16, on=True, trace_kwargs={"phase": "x"})["prebuilt"]
    with patch.object(tr.TraceAnnotation, "is_enabled", classmethod(lambda c: True)):
        assert not _observe(r, md, 16, on=True)["prebuilt"]
    # Batch-size tracking.
    with patch.object(vfc, "track_batchsize", True):
        assert not _observe(r, md, 16, on=True)["prebuilt"]
    # A stage-timing sink.
    with (
        patch.object(r._stage_timing, "_has_sink", lambda: True),
        patch.object(r._stage_timing, "stage", lambda name: contextlib.nullcontext()),
    ):
        assert not _observe(r, md, 16, on=True)["prebuilt"]
    # Patched context functions.
    patched = contextlib.contextmanager(vfc.set_forward_context.__wrapped__)
    with patch.object(tr, "set_forward_context", patched):
        assert not _observe(r, md, 16, on=True)["prebuilt"]
    # DP metadata, sequence-parallel MoE, a different vLLM source.
    # (Only the decision: the stack's DP branch needs a real DP setup.)
    for kw in (dict(dp=2), dict(sp_moe=True)):
        assert _runner(**kw)._prebuilt_forward_scope(md, 16, {}) is None
    with patch.dict(tr._FORWARD_CTX_SOURCE_SHA, {"set_forward_context": "0" * 16}):
        assert not _observe(_runner(), md, 16, on=True)["prebuilt"]


def test_stage_timing_always_on_keeps_the_stack():
    """Without VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK every forward records a
    stage timing, so the stack stays."""
    r = _runner()
    with (
        patch.dict(os.environ, {"VLLM_TPU_STAGE_TIMING_ONLY_WITH_SINK": "0"}),
        patch.object(r._stage_timing, "stage", lambda name: contextlib.nullcontext()),
    ):
        assert not _observe(r, {"l": object()}, 16, on=True)["prebuilt"]


def test_source_hash_matches_installed_vllm():
    assert tr._forward_ctx_source_matches()


def test_flag_off_is_a_no_op(stage_timing_only_with_sink):
    assert envs.VLLM_TPU_FORWARD_CTX_PREBUILT is False
    r = _runner()
    seen = _observe(r, {"l": object()}, 16, on=False)
    assert not seen["prebuilt"]
    assert "_forward_scope_ok" not in r.__dict__
