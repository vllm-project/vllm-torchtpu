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

    --additional-config '{"tpu_compile": {"fast_compiled_call": true}}'

Every option defaults to off, which is main's behaviour. The section stays in
the compile-cache key, as every ``additional_config`` section that is not
diagnostics-only does. The options act on the functions compiled with
``decorators.tpu_compile``. The worker applies them with :func:`apply` when
it is built, before any compiled function runs; until then those functions
behave as plain ``torch.compile`` ones.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_torchtpu import config_section

if TYPE_CHECKING:
    from vllm.config import VllmConfig

SECTION = "tpu_compile"


@dataclass(frozen=True)
class CompileOptions:
    fast_compiled_call: bool = False
    """Call those functions' compiled code directly when one of their cached
    entries' guards accepts the arguments, instead of going through Dynamo's
    frame evaluation on every call (``fast_dispatch``)."""


def parse(raw: Any) -> CompileOptions:
    """The ``tpu_compile`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError`` (``config_section.parse``).
    """
    return config_section.parse(CompileOptions, SECTION, raw)


def get(vllm_config: "VllmConfig | None") -> CompileOptions:
    """The engine's compile options. ``TpuPlatform.check_and_update_config``
    calls this, so a malformed section fails at startup. None (a worker built
    without a config, as the spawned-process patch test builds one) gives the
    defaults, and so does an ``additional_config`` that is not a dict
    (``config_section.get``)."""
    if vllm_config is None:
        return CompileOptions()
    return config_section.get(CompileOptions, SECTION, vllm_config)


def apply(vllm_config: "VllmConfig | None") -> CompileOptions:
    """Arm this process's ``tpu_compile`` functions with the engine's options."""
    # Imported here, not at the top: parsing and checking the section (the
    # platform does it at startup) needs no torch; only arming does.
    from vllm_torchtpu.compilation import fast_dispatch

    options = get(vllm_config)
    fast_dispatch.configure(fast=options.fast_compiled_call)
    return options


def set_guard_memo(enabled: bool) -> None:
    """Turn ``fast_dispatch``'s guard memo on or off for this process.
    ``runner.host_trims`` calls it under its batch-1 rule; like :func:`apply`,
    it keeps ``fast_dispatch`` out of every other module's imports."""
    from vllm_torchtpu.compilation import fast_dispatch

    fast_dispatch.configure_guard_memo(enabled)
