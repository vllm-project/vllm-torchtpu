# SPDX-License-Identifier: Apache-2.0
"""Record every source file a trace was keyed on in its AOT artifact.

vLLM keys the piecewise compile cache dir on the source files of a trace. A
cold start collects them on entry to every inline attempt
(``InliningInstructionTranslator.inline_call_``); a start that loads the AOT
artifact reads them back from ``source_info()``, which torch fills only with
inlines that completed. A function whose inlining raised into a caller that
handled it (``getattr(obj, name, default)`` on a missing attribute, a caught
``KeyError``) is in the first set and not the second, so the two starts hash
different files for the same source and never share a cache dir. Recording
every attempted inline in the artifact gives both starts the same set, and the
artifact's source check on load then covers those files too.
"""

from types import CodeType
from typing import Any
from unittest.mock import patch


def record_inline_attempts() -> None:
    """Make ``TorchCompileWithNoGuardsWrapper.aot_compile`` record failed inlines."""
    from torch._dynamo.symbolic_convert import InliningInstructionTranslator
    from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

    original = TorchCompileWithNoGuardsWrapper.aot_compile
    if getattr(original, "_tpu_records_inline_attempts", False):
        return

    def aot_compile(self, *args: Any, **kwargs: Any) -> Any:
        attempted: set[CodeType] = set()
        # The same hook vLLM's decorator uses to collect a cold start's files.
        inline_call = InliningInstructionTranslator.inline_call_

        def recording_inline_call(tx: Any) -> Any:
            attempted.add(tx.f_code)
            return inline_call(tx)

        with patch.object(
            InliningInstructionTranslator, "inline_call_", recording_inline_call
        ):
            compiled = original(self, *args, **kwargs)
        source_info = compiled.source_info()
        for code in attempted:
            source_info.add_code(code)
        return compiled

    aot_compile._tpu_records_inline_attempts = True
    TorchCompileWithNoGuardsWrapper.aot_compile = aot_compile
