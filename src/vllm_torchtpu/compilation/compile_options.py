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
"""Options of compiled-function dispatch, from ``additional_config["tpu_compile"]``.

    --additional-config '{"tpu_compile": {"fast_compiled_call": true,
        "drop_global_state_guard": true, "skip_aot_guard_checks": true}}'

Every option defaults to off, which is main's behaviour. The options change
how compiled programs are guarded and called, so the section stays in the
compile-cache key. The worker applies them with :func:`apply` when it is
built, before any compiled function runs; the torch.compile hook in
``_qx_fastcall`` is installed at import but does nothing until then.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

SECTION = "tpu_compile"


@dataclass(frozen=True)
class CompileOptions:
    drop_global_state_guard: bool = False
    """Compile ``torch.compile(backend="tpu", fullgraph=True, dynamic=False)``
    functions without Dynamo's GLOBAL_STATE guard (grad mode, CPU thread
    count, default dtype), so entries compiled at warmup are reused by serving
    threads whose thread count or grad mode differs. Tensor and value guards
    are kept."""

    fast_compiled_call: bool = False
    """Call those functions' compiled code directly when one of their cached
    entries' guards accepts the arguments, instead of going through Dynamo's
    frame evaluation on every call (``_qx_fastcall``)."""

    skip_aot_guard_checks: bool = False
    """After warmup, turn off the per-call guard check of the target's and the
    MTP drafter's AOT-compiled forward functions; serving only calls them at
    the warmed shapes."""


def parse(raw: Any) -> CompileOptions:
    """The ``tpu_compile`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError``.
    """
    if raw is None:
        return CompileOptions()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{SECTION} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    fields = {f.name: f.type for f in dataclasses.fields(CompileOptions)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            f"additional_config.{SECTION} has unknown keys {unknown}; "
            f"valid keys: {sorted(fields)}"
        )
    for name, value in raw.items():
        want = fields[name]
        assert isinstance(want, type)  # the annotations are classes, not strings
        if type(value) is not want:
            raise ValueError(
                f"additional_config.{SECTION}.{name} must be a "
                f"{want.__name__}, got {value!r}"
            )
    return CompileOptions(**raw)


def get(vllm_config: Any) -> CompileOptions:
    """The engine's compile options."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return CompileOptions()
    return parse(additional.get(SECTION))


def apply(vllm_config: Any) -> CompileOptions:
    """Arm the torch.compile hook of this process with the engine's options."""
    from vllm_torchtpu import _qx_fastcall

    options = get(vllm_config)
    _qx_fastcall.configure(
        fast=options.fast_compiled_call,
        drop_global_state=options.drop_global_state_guard,
    )
    return options
