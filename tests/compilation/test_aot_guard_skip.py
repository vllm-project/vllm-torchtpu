# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS


def test_fresh_aot_compile_disables_guard_check(monkeypatch):
    """VLLM_TPU_AOT_SKIP_GUARD_CHECK: a freshly AOT-compiled function skips
    the per-call guard check when guards are not evaluated, and keeps it when
    they are."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import aot_guard_skip

    class Compiled:
        def __init__(self):
            self.checking = True

        def disable_guard_check(self):
            self.checking = False

    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper, "aot_compile", lambda self, *a, **k: Compiled()
    )
    monkeypatch.setattr(envs, "VLLM_TPU_AOT_SKIP_GUARD_CHECK", True, raising=False)
    aot_guard_skip.patch_aot_guard_skip()
    for evaluate, checking in ((False, False), (True, True)):
        owner = NS(
            compilation_config=NS(dynamic_shapes_config=NS(evaluate_guards=evaluate))
        )
        compiled = TorchCompileWithNoGuardsWrapper.aot_compile(owner, 1)
        assert compiled.checking is checking
