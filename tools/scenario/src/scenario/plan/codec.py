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
"""The Plan as canonical JSON, and as the env value a pod receives.

JSON: sorted keys, no spaces, ASCII, a Ref as {"ref": kind, ...} and a
Concat as {"concat": [...]}. Encoded: base64 of that JSON gzipped with a
zero timestamp, so the same Plan always gives the same text, and the text
has no `$` for a template pass to expand. A pod refuses a Plan of another
format instead of misreading it.
"""

import base64
import binascii
import dataclasses
import gzip
import hashlib
import json
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from scenario import PLAN_FORMAT
from scenario.plan.types import Concat, Plan, Ref

# An env value larger than this risks the pod spec limits; checked per Plan.
MAX_ENCODED_BYTES = 32 * 1024


class PlanFormatError(ValueError):
    """Text that is not an encoded Plan this tool can read."""


def _dump(value: Any) -> Any:
    if isinstance(value, Ref):
        out: dict[str, Any] = {"ref": value.kind}
        if value.role is not None:
            out["role"] = value.role
        if value.sep != ",":
            out["sep"] = value.sep
        if value.fallback is not None:
            out["fallback"] = value.fallback
        return out
    if isinstance(value, Concat):
        return {"concat": [_dump(part) for part in value.parts]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _dump(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, tuple | list):
        return [_dump(item) for item in value]
    return value


def to_data(plan: Plan) -> dict[str, Any]:
    return _dump(plan)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def to_json(plan: Plan) -> str:
    return canonical_json(to_data(plan))


def digest(plan: Plan) -> str:
    """A short id of the Plan's content, for logs and the driver status."""
    return hashlib.sha256(to_json(plan).encode()).hexdigest()[:12]


def encode_data(data: Any) -> str:
    """base64(gzip(canonical JSON)), the same text for the same data."""
    raw = canonical_json(data).encode()
    return base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode()


def decode_data(text: str) -> Any:
    try:
        return json.loads(gzip.decompress(base64.b64decode(text, validate=True)))
    except (binascii.Error, OSError, EOFError, ValueError) as error:
        raise PlanFormatError(f"not an encoded Plan: {error}") from None


def encode(plan: Plan) -> str:
    return encode_data(to_data(plan))


def decode(text: str) -> Plan:
    """The Plan in an encoded env value.

    Raises:
        PlanFormatError: the text is not an encoded Plan, or one of another
            format than this tool's.
    """
    return from_data(decode_data(text.strip()))


def from_json(text: str) -> Plan:
    try:
        data = json.loads(text)
    except ValueError as error:
        raise PlanFormatError(f"not Plan JSON: {error}") from None
    return from_data(data)


def from_data(data: Any) -> Plan:
    if not isinstance(data, dict) or "format" not in data:
        raise PlanFormatError("not a Plan: no format field")
    if data["format"] != PLAN_FORMAT:
        raise PlanFormatError(
            f"plan format {data['format']}, this tool reads {PLAN_FORMAT}: the pod "
            "image and the submitting checkout differ"
        )
    try:
        return _load(Plan, data)
    except (KeyError, TypeError, ValueError) as error:
        raise PlanFormatError(f"not a Plan of format {PLAN_FORMAT}: {error}") from None


def _load(hint: Any, data: Any) -> Any:
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        return _load_union(get_args(hint), data)
    if origin is Literal:
        if data not in get_args(hint):
            raise ValueError(f"{data!r} is not one of {get_args(hint)}")
        return data
    if origin is tuple:
        args = get_args(hint)
        if not isinstance(data, list):
            raise TypeError(f"expected a list, got {data!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_load(args[0], item) for item in data)
        return tuple(_load(arg, item) for arg, item in zip(args, data, strict=True))
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return _load_dataclass(hint, data)
    if hint is float and isinstance(data, int | float) and not isinstance(data, bool):
        return data
    if isinstance(hint, type) and isinstance(data, hint):
        return data
    raise TypeError(f"expected {getattr(hint, '__name__', hint)}, got {data!r}")


def _load_dataclass(cls: type, data: Any) -> Any:
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {data!r}")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__} has no field {sorted(unknown)[0]}")
    return cls(**{name: _load(hints[name], value) for name, value in data.items()})


def _load_union(members: tuple[Any, ...], data: Any) -> Any:
    if data is None:
        if type(None) in members:
            return None
        raise TypeError("unexpected null")
    if isinstance(data, dict):
        if "ref" in data and Ref in members:
            fields = dict(data)
            return _load_dataclass(Ref, {"kind": fields.pop("ref"), **fields})
        if "concat" in data and Concat in members:
            return Concat(tuple(_load(str | Ref, part) for part in data["concat"]))
        classes = [
            m for m in members if isinstance(m, type) and dataclasses.is_dataclass(m)
        ]
        if len(classes) == 1:
            return _load_dataclass(classes[0], data)
        for cls in classes:
            kind = {f.name: f.default for f in dataclasses.fields(cls)}.get("kind")
            if kind is not None and data.get("kind") == kind:
                return _load_dataclass(cls, data)
        raise ValueError(f"cannot tell which of {members} this is: {data!r}")
    for member in members:
        if member is not type(None):
            try:
                return _load(member, data)
            except (TypeError, ValueError):
                continue
    raise TypeError(f"{data!r} matches none of {members}")
