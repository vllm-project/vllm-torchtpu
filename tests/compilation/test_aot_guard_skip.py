# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace as NS


def test_fresh_aot_compile_disables_guard_check(monkeypatch):
    """A freshly AOT-compiled function skips the per-call guard check when
    guards are not evaluated, and keeps it when they are (or when config is
    absent)."""
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    from vllm_torchtpu.compilation import aot_guard_skip

    class Compiled:
        def __init__(self):
            self.checking = True

        def disable_guard_check(self):
            self.checking = False

    monkeypatch.setattr(
        TorchCompileWithNoGuardsWrapper, "aot_compile", lambda self, *a, **k: Compiled()
    )
    aot_guard_skip.patch_aot_guard_skip()
    patched = TorchCompileWithNoGuardsWrapper.aot_compile
    aot_guard_skip.patch_aot_guard_skip()
    assert TorchCompileWithNoGuardsWrapper.aot_compile is patched
    for evaluate, checking in ((False, False), (True, True)):
        owner = NS(
            compilation_config=NS(dynamic_shapes_config=NS(evaluate_guards=evaluate))
        )
        compiled = TorchCompileWithNoGuardsWrapper.aot_compile(owner, 1)
        assert compiled.checking is checking
    # Negative control: missing compilation_config defaults to keeping guard check.
    compiled_default = TorchCompileWithNoGuardsWrapper.aot_compile(NS(), 1)
    assert compiled_default.checking is True
