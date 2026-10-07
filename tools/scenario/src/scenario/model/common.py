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
"""Field types shared by the scenario models.

Each type validates one value and says what it expected in words, since the
message ends up in front of whoever wrote the YAML. Types whose YAML form
differs from the stored form (durations, sizes) also carry their JSON schema,
so editors validate what people actually write.
"""

import re
from dataclasses import dataclass
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    JsonValue,
    WithJsonSchema,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema

from scenario.paths import PathSyntaxError, parse_path


def fail(kind: str, message: str) -> PydanticCustomError:
    """A validation error whose message is shown as written."""
    return PydanticCustomError(kind, message)


class Model(BaseModel):
    """Base of every scenario model.

    Unknown keys fail and values are never coerced ("8" is not 8). YAML keys
    that clash with Python names (`schema`, `from`) are aliases, and only the
    YAML spelling is accepted.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_by_alias=True,
        validate_by_name=False,
        serialize_by_alias=True,
    )


@dataclass(frozen=True)
class Matches:
    """`Annotated` metadata: the string must match `pattern` in full.

    `what` states the rule in words; it is the error message.
    """

    kind: str
    pattern: str
    what: str

    def __get_pydantic_core_schema__(
        self, source: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        regex = re.compile(self.pattern)

        def check(value: str) -> str:
            if not regex.fullmatch(value):
                raise fail(self.kind, f"{self.what}; got {value!r}")
            return value

        return core_schema.no_info_after_validator_function(check, handler(source))

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        json_schema["pattern"] = f"^{self.pattern}$"
        return json_schema


# Durations ------------------------------------------------------------------

MAX_DURATION_S = 24 * 3600
_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


def _duration(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        seconds = value
    elif isinstance(value, str) and value and (match := _DURATION.fullmatch(value)):
        hours, minutes, secs = (int(group or 0) for group in match.groups())
        seconds = hours * 3600 + minutes * 60 + secs
    else:
        raise fail(
            "duration",
            "expected a duration such as 2h, 90m, 4h10m or 30s, or whole "
            f"seconds; got {value!r}",
        )
    if not 0 < seconds <= MAX_DURATION_S:
        raise fail("duration", f"must be more than 0s and at most 24h; got {value!r}")
    return seconds


def format_duration(seconds: int) -> str:
    """The short form of a duration in seconds: 7800 -> "2h10m"."""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    text = "".join(
        f"{amount}{unit}"
        for amount, unit in ((hours, "h"), (minutes, "m"), (secs, "s"))
        if amount
    )
    return text or "0s"


Duration = Annotated[
    int,
    BeforeValidator(_duration),
    WithJsonSchema(
        {
            "anyOf": [
                {"type": "string", "pattern": r"^(\d+h)?(\d+m)?(\d+s)?$"},
                {"type": "integer", "minimum": 1, "maximum": MAX_DURATION_S},
            ]
        }
    ),
]
"""`2h`, `90m`, `4h10m`, `30s` or whole seconds; stored as seconds."""


# Sizes ----------------------------------------------------------------------

_BYTES = re.compile(r"(\d+)(Ki|Mi|Gi|Ti)")
_UNITS = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}


def _byte_size(value: object) -> int:
    if isinstance(value, str) and (match := _BYTES.fullmatch(value)):
        count = int(match.group(1))
        if count > 0:
            return count * _UNITS[match.group(2)]
    raise fail(
        "byte_size", f"expected a size such as 64Gi (Ki, Mi, Gi or Ti); got {value!r}"
    )


ByteSize = Annotated[
    int,
    BeforeValidator(_byte_size),
    WithJsonSchema({"type": "string", "pattern": r"^[1-9]\d*(Ki|Mi|Gi|Ti)$"}),
]
"""`64Gi` and the like; stored as bytes."""


# Hardware and images --------------------------------------------------------

Accelerator = Annotated[
    str,
    Matches(
        "accelerator",
        r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,61}[A-Za-z0-9])?",
        "an accelerator is a cloud.google.com/gke-tpu-accelerator label value such "
        "as tpu7x: letters, digits, '.', '_' and '-', starting and ending with a "
        "letter or digit, at most 63 characters",
    ),
]
"""GKE's `cloud.google.com/gke-tpu-accelerator` label, e.g. `tpu7x`."""

Topology = Annotated[
    str,
    Matches(
        "topology",
        r"[1-9][0-9]*(?:x[1-9][0-9]*){1,2}",
        "a topology is a cloud.google.com/gke-tpu-topology label value: two or "
        "three positive integers joined by x, e.g. 2x2x2 or 2x4",
    ),
]
"""GKE's `cloud.google.com/gke-tpu-topology` label, e.g. `2x2x2`."""

_IMAGE = re.compile(
    r"(?:localhost|(?:[a-z0-9-]+\.)+[a-z0-9-]+)(?::\d+)?"
    r"/[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?"
    r"(?:@sha256:[0-9a-f]{64})?"
)


def _image(value: str) -> str:
    if value == "ci":
        return value
    last = value.rsplit("/", 1)[-1]
    if _IMAGE.fullmatch(value) and (":" in last or "@" in last):
        return value
    raise fail(
        "image",
        "expected ci (this build's CI image) or a full image URI with a registry "
        "host and a tag or @sha256 digest, e.g. "
        f"us-docker.pkg.dev/project/repo/image:tag; got {value!r}",
    )


ImageRef = Annotated[
    str,
    AfterValidator(_image),
    WithJsonSchema({"type": "string", "pattern": r"^(ci|[^\s]+/[^\s]+)$"}),
]
"""`ci` or a full image URI."""


# Names ----------------------------------------------------------------------

Name = Annotated[
    str,
    Matches(
        "name",
        r"[a-z0-9][a-z0-9._-]{0,62}",
        "names are lowercase letters, digits, '.', '_' and '-', start with a "
        "letter or digit, and are at most 63 characters",
    ),
]

RESERVED_ROLE_NAMES = ("server", "client")


def _role_name(value: str) -> str:
    if value in RESERVED_ROLE_NAMES:
        raise fail(
            "role_name",
            "server is the name of the single role of a scenario without roles, "
            f"and client the chipless driver; pick another name than {value!r}",
        )
    return value


RoleName = Annotated[
    str,
    Matches(
        "role_name",
        r"[a-z][a-z0-9-]{0,14}",
        "role names are lowercase letters, digits and '-', start with a letter, "
        "and are at most 15 characters, since they become host names",
    ),
    AfterValidator(_role_name),
]

PointName = Annotated[
    str,
    Matches(
        "point_name",
        r"[a-z][a-z0-9-]{0,62}",
        "point names are lowercase letters, digits and '-', start with a "
        "letter, and are at most 63 characters",
    ),
]

TaskName = Annotated[
    str,
    Matches(
        "task_name",
        r"[a-z0-9][a-z0-9._-]{0,62}",
        "task names are lowercase letters, digits, '.', '_' and '-', start "
        "with a letter or digit, and are at most 63 characters",
    ),
]

Label = Annotated[
    str,
    Matches(
        "label",
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}",
        "labels are letters, digits, '.', '_' and '-', start with a letter or "
        "digit, and are at most 63 characters",
    ),
]

ServedModelName = Annotated[
    str, Matches("model_name", r"\S+", "model names cannot be empty or contain spaces")
]

_REF_ID = r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*"


def _ref_id(value: str) -> str:
    if value.endswith((".yaml", ".yml")):
        stem = value.rsplit(".", 1)[0]
        raise fail(
            "ref_id",
            "a scenario reference is its path under scenarios/ without .yaml; "
            f"write {stem!r}",
        )
    return value


RefId = Annotated[
    str,
    Matches(
        "ref_id",
        _REF_ID,
        "a scenario reference is its path under scenarios/ without .yaml, "
        "e.g. kimi-k3/tp32-ep",
    ),
    AfterValidator(_ref_id),
]

_RELATIVE_YAML = re.compile(r"\.\.?/\S*\.ya?ml")
_REF_ID_RE = re.compile(_REF_ID)


def _ref_or_path(value: str) -> str:
    if _RELATIVE_YAML.fullmatch(value):
        return value
    if _REF_ID_RE.fullmatch(value):
        return _ref_id(value)
    raise fail(
        "ref_id",
        "expected a scenario reference (its path under scenarios/ without "
        f".yaml) or a relative path such as ./tp32-ep.yaml; got {value!r}",
    )


RefOrPath = Annotated[
    str,
    AfterValidator(_ref_or_path),
    WithJsonSchema({"type": "string"}),
]
"""A reference id, or a path relative to the file, normalized by resolution."""


# Env ------------------------------------------------------------------------

EnvName = Annotated[
    str,
    Matches(
        "env_name",
        r"[A-Za-z_][A-Za-z0-9_]*",
        "env names are letters, digits and '_', and do not start with a digit",
    ),
]


def _env_value(value: object) -> object:
    if isinstance(value, bool):
        raise fail("env_value", 'env values are strings; quote it: "1" or "true"')
    if isinstance(value, float):
        # YAML reads 0.90 as 0.9, so the value would not be what was written.
        raise fail(
            "env_value",
            f'env values are strings or integers; quote it as written, e.g. "{value}"',
        )
    if isinstance(value, int):
        return str(value)
    if value is None:
        raise fail(
            "env_value",
            'null removes an inherited key; write "" for an empty value',
        )
    return value


EnvValue = Annotated[
    str,
    BeforeValidator(_env_value),
    WithJsonSchema({"type": ["string", "integer"]}),
]
"""A string; an integer is accepted and stored as its decimal text."""

EnvMap = dict[EnvName, EnvValue]
"""Env name to value."""

RoleEnvMap = dict[EnvName, EnvValue | None]
"""Env over `server.env`; null removes the inherited entry for this role."""


# vLLM-style args ------------------------------------------------------------

_ARG_KEY = re.compile(r"[a-z0-9][a-z0-9-]*(?:\.[A-Za-z0-9_-]+)*")


def _arg_key(value: str) -> str:
    if value.startswith("-"):
        raise fail("V-15", f"write the key without dashes: {value.lstrip('-')}")
    head, dot, rest = value.partition(".")
    if "_" in head:
        raise fail("V-15", f"use {head.replace('_', '-')}{dot}{rest}")
    if not _ARG_KEY.fullmatch(value):
        raise fail(
            "V-15",
            "arg keys are flag names without the leading dashes, lowercase with "
            f"'-' between words, e.g. max-model-len; got {value!r}",
        )
    return value


ArgKey = Annotated[
    str,
    AfterValidator(_arg_key),
    WithJsonSchema({"type": "string", "pattern": f"^{_ARG_KEY.pattern}$"}),
]

ArgValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
"""true -> --k, false -> --no-k, scalar -> --k v, list or mapping -> --k <json>."""

ArgMap = dict[ArgKey, ArgValue]
RoleArgMap = dict[ArgKey, ArgValue | None]
"""Args over `server.args`; null removes the inherited key for this role."""


# Numbers --------------------------------------------------------------------

PosInt = Annotated[int, Field(gt=0)]
NonNegInt = Annotated[int, Field(ge=0)]
Tolerance = Annotated[float, Field(gt=0, lt=1)]
UnitInterval = Annotated[float, Field(ge=0, le=1)]


# Paths ----------------------------------------------------------------------


def _key_path(value: str) -> str:
    try:
        parse_path(value)
    except PathSyntaxError as error:
        raise fail("key_path", str(error)) from None
    return value


KeyPathText = Annotated[
    str, AfterValidator(_key_path), WithJsonSchema({"type": "string"})
]
"""A dotted key path such as `server.args.max-num-seqs`."""
