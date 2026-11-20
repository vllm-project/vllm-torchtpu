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
"""Strict parsing of one ``additional_config`` section into a frozen dataclass.

Shared by the option modules of the TPU sections. Every field's annotation
must be a class, not a string, so the option modules do not use
``from __future__ import annotations``.
"""

import dataclasses
from typing import Any


def parse[T](options_cls: type[T], section: str, raw: Any) -> T:
    """The section ``raw`` as ``options_cls``; None gives the defaults.

    Strict: an unknown key, or a value whose type is not exactly the field's
    (``1`` for a bool, say), raises ``ValueError``.
    """
    if raw is None:
        return options_cls()
    if not isinstance(raw, dict):
        raise ValueError(
            f"additional_config.{section} must be a JSON object, "
            f"got {type(raw).__name__}"
        )
    # dataclasses.fields is typed for DataclassInstance, which a type
    # parameter cannot name; every caller passes a frozen dataclass.
    # pyrefly: ignore[bad-argument-type]
    fields = {f.name: f.type for f in dataclasses.fields(options_cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            f"additional_config.{section} has unknown keys {unknown}; "
            f"valid keys: {sorted(fields)}"
        )
    for name, value in raw.items():
        want = fields[name]
        assert isinstance(want, type)  # the annotations are classes, not strings
        if type(value) is not want:
            raise ValueError(
                f"additional_config.{section}.{name} must be a "
                f"{want.__name__}, got {value!r}"
            )
    return options_cls(**raw)


def get[T](options_cls: type[T], section: str, vllm_config: Any) -> T:
    """The section of ``vllm_config.additional_config`` as ``options_cls``."""
    additional = vllm_config.additional_config
    if not isinstance(additional, dict):
        return options_cls()
    return parse(options_cls, section, additional.get(section))
