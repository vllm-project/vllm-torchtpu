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
"""Qwen3.5 model-scoped settings, read from ``--additional-config``.

The settings live in one section of vLLM's ``additional_config``::

    vllm serve Qwen/Qwen3.5-397B-A17B-FP8 ... \\
        --additional-config '{"qwen3_5": {"<setting>": <value>}}'

Each setting is one field of :class:`Qwen35Settings`. A setting is added by
the change that implements it, together with a test that its consumer honours
it, so every accepted key changes something. ``parse`` derives the accepted
keys and their types from the dataclass and rejects anything else, so a typo
fails at engine start instead of being ignored.

The settings apply to Qwen3.5 targets only (``ARCHITECTURES``). ``get``
returns the defaults for any other model and rejects a ``qwen3_5`` section
given to one. ``additional_config`` is part of vLLM's config hash, so a
setting that changes a compiled program also changes the compile-cache key.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

SECTION = "qwen3_5"

# The architectures the Qwen3.5 settings and config hook apply to: the
# released Qwen3.5-397B-A17B checkpoints declare the first.
ARCHITECTURES = frozenset({"Qwen3_5MoeForConditionalGeneration"})


@dataclasses.dataclass(frozen=True)
class Qwen35Settings:
    """The Qwen3.5 settings of one engine. Fields are added by the change that
    implements them; keep them sorted by name."""


def _field_types() -> dict[str, Any]:
    hints = typing.get_type_hints(Qwen35Settings)
    return {f.name: hints[f.name] for f in dataclasses.fields(Qwen35Settings)}


def _check_value(key: str, value: Any, expected: Any) -> None:
    origin = get_origin(expected)
    if origin is Literal:
        choices = get_args(expected)
        # `True == 1` and `1 == True`, so compare the type as well.
        if not any(type(value) is type(c) and value == c for c in choices):
            raise ValueError(
                f"additional_config[{SECTION!r}][{key!r}] must be one of "
                f"{list(choices)}, got {value!r}"
            )
        return
    if origin in (typing.Union, types.UnionType):
        alternatives = get_args(expected)
    else:
        alternatives = (expected,)
    for alt in alternatives:
        if alt is type(None) and value is None:
            return
        if alt is bool and type(value) is bool:
            return
        # A bool is an int in Python; an int or float field must not take one.
        if alt is int and type(value) is int:
            return
        if alt is float and type(value) in (int, float):
            return
        if alt is str and isinstance(value, str):
            return
    names = " | ".join(getattr(a, "__name__", repr(a)) for a in alternatives)
    raise ValueError(
        f"additional_config[{SECTION!r}][{key!r}] must be {names}, got "
        f"{type(value).__name__} {value!r}"
    )


def parse(raw: Any) -> Qwen35Settings:
    """The settings in one ``additional_config["qwen3_5"]`` value.

    ``None`` or an empty mapping gives the defaults. Unknown keys and values of
    the wrong type raise ``ValueError``.
    """
    if raw is None:
        return Qwen35Settings()
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"additional_config[{SECTION!r}] must be a JSON object, got "
            f"{type(raw).__name__} {raw!r}"
        )
    types_by_key = _field_types()
    unknown = sorted(str(k) for k in raw if k not in types_by_key)
    if unknown:
        known = ", ".join(sorted(types_by_key)) or "none yet"
        raise ValueError(
            f"Unknown Qwen3.5 setting(s) {unknown} in additional_config"
            f"[{SECTION!r}]; known settings: {known}"
        )
    for key, value in raw.items():
        _check_value(key, value, types_by_key[key])
    return Qwen35Settings(**dict(raw))


def is_qwen3_5(model_config: ModelConfig | None) -> bool:
    """Whether the engine's target model is a Qwen3.5 architecture."""
    if model_config is None:
        return False
    return not ARCHITECTURES.isdisjoint(model_config.architectures or ())


def _section(vllm_config: VllmConfig) -> Any:
    additional = vllm_config.additional_config
    if isinstance(additional, Mapping):
        return additional.get(SECTION)
    return None


def get(vllm_config: VllmConfig) -> Qwen35Settings:
    """The Qwen3.5 settings of this engine.

    For a Qwen3.5 target, the parsed ``additional_config["qwen3_5"]``. For any
    other model, the defaults; a ``qwen3_5`` section on another model raises
    ``ValueError``, since it would otherwise be silently ignored.
    """
    raw = _section(vllm_config)
    if is_qwen3_5(vllm_config.model_config):
        return parse(raw)
    if raw is not None:
        archs = (
            list(vllm_config.model_config.architectures or ())
            if vllm_config.model_config is not None
            else []
        )
        raise ValueError(
            f"additional_config[{SECTION!r}] is set, but the model "
            f"architectures {archs} are not Qwen3.5 ({sorted(ARCHITECTURES)})"
        )
    return Qwen35Settings()


def current() -> Qwen35Settings:
    """The settings of the engine whose config vLLM is currently building.

    For code that runs under vLLM's config context (model construction,
    kernel builders). Code on the step path should read the settings once at
    init and keep them.
    """
    from vllm.config import get_current_vllm_config

    return get(get_current_vllm_config())
