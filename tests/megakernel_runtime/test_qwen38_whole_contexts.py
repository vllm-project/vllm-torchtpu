# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PREFLAT_FWD_ARGS: the whole-step call's contexts, resolved once,
set the same model tag and vLLM config as `_contexts(runner, "model_mk_whole")`."""

from types import SimpleNamespace

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv


def _seen(ctx_manager):
    import vllm.compilation.backends as backends
    from vllm.config import get_current_vllm_config

    with ctx_manager:
        return backends.model_tag, get_current_vllm_config()


def test_whole_contexts_match_per_step_contexts(monkeypatch):
    from vllm.config import VllmConfig

    whole_cfg, cfg = VllmConfig(), VllmConfig()
    d = drv.Qwen38SegmentDriver.__new__(drv.Qwen38SegmentDriver)
    d.whole_cfg, d.cfg = whole_cfg, cfg
    runner = SimpleNamespace(vllm_config=VllmConfig())
    ref = _seen(d._contexts(runner, "model_mk_whole"))
    monkeypatch.setattr(envs, "VLLM_TPU_PREFLAT_FWD_ARGS", True, raising=False)
    first = d._whole_contexts(runner)
    assert isinstance(first, drv._StackedContexts)
    assert _seen(first) == ref
    assert _seen(d._whole_contexts(runner)) == ref  # reused factories
    assert ref[0] == "model_mk_whole" and ref[1] is whole_cfg
    monkeypatch.setattr(envs, "VLLM_TPU_PREFLAT_FWD_ARGS", False, raising=False)
    assert not isinstance(d._whole_contexts(runner), drv._StackedContexts)
