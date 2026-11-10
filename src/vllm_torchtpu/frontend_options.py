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

The platform calls :func:`install` from ``check_and_update_config``, which runs
in every process that builds the engine config, including each API server.
The reasoning-count and detokenizer patches only apply in the API server
processes; the tokenization cache is only reached through vLLM's renderer.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any

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
    """The ``tpu_frontend`` section as options; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError``.
    """
    if raw is None:
        return FrontendOptions()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{SECTION} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    fields = {f.name: f.type for f in dataclasses.fields(FrontendOptions)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            f"additional_config.{SECTION} has unknown keys {unknown}; "
            f"valid keys: {sorted(fields)}"
        )
    for name, value in raw.items():
        if type(value) is not fields[name]:
            raise ValueError(
                f"additional_config.{SECTION}.{name} must be a "
                f"{fields[name].__name__}, got {value!r}"
            )
    return FrontendOptions(**raw)


def get(vllm_config: Any) -> FrontendOptions:
    """The engine's frontend options."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        return FrontendOptions()
    return parse(additional.get(SECTION))


def install(vllm_config: Any) -> FrontendOptions:
    """Install the patches the options turn on. Idempotent per process.

    A patch that cannot be installed logs a warning and leaves vLLM's own path
    in place; it never stops the server from starting.
    """
    options = get(vllm_config)
    if options.tokenize_prefix_cache:
        from vllm_torchtpu.frontend_tokcache import maybe_install_tokenize_prefix_cache

        _install_or_warn("tokenize_prefix_cache", maybe_install_tokenize_prefix_cache)
    if options.lazy_reasoning_count:
        from vllm_torchtpu.frontend_fix import maybe_install_lazy_reasoning_count

        _install_or_warn("lazy_reasoning_count", maybe_install_lazy_reasoning_count)
    if options.detok_prompt_tail:
        from vllm_torchtpu.frontend_fix import maybe_install_detok_prompt_tail

        _install_or_warn("detok_prompt_tail", maybe_install_detok_prompt_tail)
    return options


def _install_or_warn(name: str, install_fn) -> None:
    try:
        install_fn()
    except Exception as e:
        logger.warning("%s.%s: install failed (%r)", SECTION, name, e)
