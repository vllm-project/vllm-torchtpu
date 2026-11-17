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
"""Options of the API-server patches, from ``additional_config["tpu_frontend"]``.

    --additional-config '{"tpu_frontend": {"tokenize_prefix_cache": true,
        "lazy_reasoning_count": true, "detok_prompt_tail": true}}'

Each option installs one patch of vLLM's API-server path. Every option
defaults to off, which is vLLM's own behaviour. The patches run on the host and
never change a compiled program, so the section stays out of the compile-cache
key (``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``).

Each patch has an installer registered in ``patch_registry.PATCHES`` for the
``platform_activation`` stage, which ``check_and_update_config`` applies with
the engine config in every process that builds it, including each API server.
The reasoning-count and detokenizer patches only apply in the API server
processes; the tokenization cache is only reached through vLLM's renderer.
"""

import importlib
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import config_section
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SECTION = "tpu_frontend"


@dataclass(frozen=True)
class FrontendOptions:
    detok_prompt_tail: bool = False
    """Prime vLLM's incremental detokenizer with the last 8 prompt tokens
    instead of the whole prompt when that suffix ends on a character boundary
    (byte-identical output; see ``frontend_fix.safe_prompt_tail``). Saves
    decoding the prompt twice on the first emitted token."""

    lazy_reasoning_count: bool = False
    """Skip recounting reasoning tokens on intermediate streaming chunks when
    continuous usage stats are off, and return 0 when the reasoning parser's
    state machine can never emit a reasoning chunk."""

    tokenize_prefix_cache: bool = False
    """Tokenize chat prompts incrementally: reuse the token ids of the longest
    cached prefix that ends at a ``<|im_start|>`` boundary and encode only the
    new suffix. Verified against a full encode for the first prompts; any
    mismatch disables the cache."""


def parse(raw: Any) -> FrontendOptions:
    """The ``tpu_frontend`` section as options (see ``config_section.parse``)."""
    return config_section.parse(FrontendOptions, SECTION, raw)


def get(vllm_config: Any) -> FrontendOptions:
    """The engine's frontend options."""
    return config_section.get(FrontendOptions, SECTION, vllm_config)


def install_tokenize_prefix_cache(vllm_config: Any = None) -> bool | None:
    """patch_registry installer of ``tokenize_prefix_cache``."""
    return _install(
        vllm_config,
        "tokenize_prefix_cache",
        "vllm_torchtpu.frontend_tokcache",
        "maybe_install_tokenize_prefix_cache",
    )


def install_lazy_reasoning_count(vllm_config: Any = None) -> bool | None:
    """patch_registry installer of ``lazy_reasoning_count``."""
    return _install(
        vllm_config,
        "lazy_reasoning_count",
        "vllm_torchtpu.frontend_fix",
        "maybe_install_lazy_reasoning_count",
    )


def install_detok_prompt_tail(vllm_config: Any = None) -> bool | None:
    """patch_registry installer of ``detok_prompt_tail``."""
    return _install(
        vllm_config,
        "detok_prompt_tail",
        "vllm_torchtpu.frontend_fix",
        "maybe_install_detok_prompt_tail",
    )


def _install(vllm_config: Any, name: str, module: str, installer: str) -> bool | None:
    """Install option ``name``'s patch with ``module.installer`` when the engine
    config turns it on.

    Off, this returns False, which keeps the registry entry eligible: a later
    engine config built in the same process may still turn it on. A patch that
    cannot be installed logs a warning and leaves vLLM's own path in place; it
    never stops the server from starting.
    """
    if vllm_config is None or not getattr(get(vllm_config), name):
        return False
    try:
        getattr(importlib.import_module(module), installer)()
    except Exception as e:
        logger.warning("%s.%s: install failed (%r)", SECTION, name, e)
    return None
