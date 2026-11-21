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
        "detok_prompt_tail": true}}'

Each option installs one patch of vLLM's API-server path. Every option
defaults to off, which is vLLM's own behaviour. The patches run on the host and
never change a compiled program, so the section stays out of the compile-cache
key (``HASH_IGNORED_ADDITIONAL_CONFIG_KEYS``).

Each patch has an installer registered in ``patch_registry.PATCHES`` for the
``platform_activation`` stage, which ``check_and_update_config`` applies with
the engine config in every process that builds it, including each API server.
The detokenizer patch only applies in the API server processes; the
tokenization cache is only reached through vLLM's renderer.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import config_section
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SECTION = "tpu_frontend"


@dataclass(frozen=True)
class FrontendOptions:
    detok_prompt_tail: bool = False
    """Prime vLLM's incremental detokenizer with the last
    ``frontend_fix.DETOK_PROMPT_TAIL_TOKENS`` prompt tokens instead of the
    whole prompt when that suffix ends on a character boundary (byte-identical
    output; see ``frontend_fix.safe_prompt_tail``). Saves decoding the prompt
    twice on the first emitted token."""

    tokenize_prefix_cache: bool = False
    """Tokenize chat prompts incrementally: reuse the token ids of the longest
    cached prefix that ends at a ``<|im_start|>`` boundary and encode only the
    new suffix. Verified against a full encode for the first prompts; any
    mismatch disables the cache."""

    tokenize_prefix_cache_max_tokens: int = 256_000_000
    """Most token ids the tokenization cache keeps per API server; the least
    recently used prompts are evicted first. Read only with
    ``tokenize_prefix_cache``. This trades API-server host memory for
    tokenization CPU time. A cached token id holds about 15 bytes when
    conversations reuse their earlier turns (measured with the Qwen3.5
    tokenizer on multi-turn chats) and up to about 40 bytes with little
    prefix reuse, so the default allows about 4 to 10 GB per API server, and
    each DP API server has its own cache. A smaller limit evicts more
    conversations, whose next turn is then encoded in full (the token ids are
    the same either way). Lower it to bound an API server's host memory."""

    def __post_init__(self) -> None:
        if self.tokenize_prefix_cache_max_tokens <= 0:
            raise ValueError(
                f"additional_config.{SECTION}.tokenize_prefix_cache_max_tokens "
                f"must be positive, got {self.tokenize_prefix_cache_max_tokens}"
            )


def parse(raw: Any) -> FrontendOptions:
    """The ``tpu_frontend`` section as options (see ``config_section.parse``)."""
    return config_section.parse(FrontendOptions, SECTION, raw)


def get(vllm_config: Any) -> FrontendOptions:
    """The engine's frontend options."""
    return config_section.get(FrontendOptions, SECTION, vllm_config)


def install_tokenize_prefix_cache(vllm_config: Any = None) -> bool | None:
    """patch_registry installer of ``tokenize_prefix_cache``. Returns False
    while the option is off, which keeps the registry entry eligible: a later
    engine config built in the same process may still turn it on."""
    options = None if vllm_config is None else get(vllm_config)
    if options is None or not options.tokenize_prefix_cache:
        return False

    def install() -> None:
        from vllm_torchtpu import frontend_tokcache

        frontend_tokcache.maybe_install_tokenize_prefix_cache(
            max_tokens=options.tokenize_prefix_cache_max_tokens
        )

    return _install("tokenize_prefix_cache", install)


def install_detok_prompt_tail(vllm_config: Any = None) -> bool | None:
    """patch_registry installer of ``detok_prompt_tail``. Returns False while
    the option is off, as ``install_tokenize_prefix_cache`` does."""
    options = None if vllm_config is None else get(vllm_config)
    if options is None or not options.detok_prompt_tail:
        return False

    def install() -> None:
        from vllm_torchtpu import frontend_fix

        frontend_fix.maybe_install_detok_prompt_tail()

    return _install("detok_prompt_tail", install)


def _install(name: str, install: Callable[[], None]) -> None:
    """Run the patch installer of option ``name``, which the engine config
    turns on; it imports its module only then.

    A patch that cannot be installed logs a warning and leaves vLLM's own path
    in place; it never stops the server from starting.
    """
    try:
        install()
    except Exception as e:
        logger.warning("%s.%s: install failed (%r)", SECTION, name, e)
