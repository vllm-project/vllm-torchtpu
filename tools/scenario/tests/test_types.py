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
"""Field types: what each accepts, what it stores, and what it rejects."""

from typing import Any

import pytest
from pydantic import ConfigDict, TypeAdapter, ValidationError
from scenario.model import common

STRICT = ConfigDict(strict=True)


def check(type_: Any, value: object) -> object:
    return TypeAdapter(type_, config=STRICT).validate_python(value)


def error(type_: Any, value: object) -> dict[str, Any]:
    with pytest.raises(ValidationError) as info:
        check(type_, value)
    (first, *_) = info.value.errors()
    return dict(first)


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("2h", 7200), ("90m", 5400), ("4h10m", 15000), ("30s", 30), (600, 600)],
)
def test_duration_accepts(value: object, seconds: int) -> None:
    assert check(common.Duration, value) == seconds


@pytest.mark.parametrize(
    "value", ["", "2", "2 h", "1d", "h", True, 0, "0s", "25h", 1.5]
)
def test_duration_rejects(value: object) -> None:
    assert error(common.Duration, value)["type"] == "duration"


def test_format_duration() -> None:
    assert common.format_duration(7800) == "2h10m"
    assert common.format_duration(14400) == "4h"
    assert common.format_duration(61) == "1m1s"


@pytest.mark.parametrize(("value", "size"), [("64Gi", 64 * 2**30), ("1Ki", 1024)])
def test_byte_size_accepts(value: str, size: int) -> None:
    assert check(common.ByteSize, value) == size


@pytest.mark.parametrize("value", ["64G", "64GB", "0Gi", 64, "Gi", "1.5Gi"])
def test_byte_size_rejects(value: object) -> None:
    assert error(common.ByteSize, value)["type"] == "byte_size"


def test_slice() -> None:
    assert check(common.Slice, "tpu7x/2x2x2") == "tpu7x/2x2x2"
    failure = error(common.Slice, "tpu7x/3x3x3")
    assert failure["type"] == "slice"
    assert "known: 1x1x1" in failure["msg"]


@pytest.mark.parametrize(
    "value",
    [
        "ci",
        "us-docker.pkg.dev/proj/repo/vllm-torchtpu:abc123",
        "us-docker.pkg.dev/proj/repo/image@sha256:" + "a" * 64,
        "us-docker.pkg.dev/proj/repo/image:tag@sha256:" + "0" * 64,
        "localhost:5000/image:dev",
        "docker.io/library/ubuntu:22.04",
    ],
)
def test_image_accepts(value: str) -> None:
    assert check(common.ImageRef, value) == value


@pytest.mark.parametrize(
    "value",
    [
        "latest",
        "ubuntu:22.04",
        "us-docker.pkg.dev/proj/repo/image",
        "us-docker.pkg.dev/proj/repo/image@sha256:abc",
        "CI",
    ],
)
def test_image_rejects(value: str) -> None:
    failure = error(common.ImageRef, value)
    assert failure["type"] == "image"
    assert failure["msg"].startswith("expected ci (this build's CI image)")


@pytest.mark.parametrize(
    ("type_name", "good", "bad"),
    [
        ("Name", ["glm-5.2-fp8-dp16-ep", "a", "x_1"], ["GLM", "-a", "a/b", "a" * 64]),
        ("RoleName", ["prefill", "decode-2"], ["Prefill", "p_1", "2p", "a" * 16]),
        ("PointName", ["c1", "isl8192-osl1024-c64"], ["1c", "c.1", "C1", "c_1"]),
        ("TaskName", ["correctness", "check.completions"], ["", "A", "a b"]),
        ("Label", ["kimik3", "fp4", "MXFP4"], ["", "-x", "a b"]),
        ("ServedModelName", ["zai-org/GLM-5.2-FP8"], ["", "a b"]),
        ("EnvName", ["TPU_X", "_A1", "lower"], ["1A", "A-B", ""]),
        ("RefId", ["kimi-k3/tp32-ep", "glm-5.2/fp8-dp16-ep", "x"], ["/abs", "a//b"]),
    ],
)
def test_patterns(type_name: str, good: list[str], bad: list[str]) -> None:
    type_ = getattr(common, type_name)
    for value in good:
        assert check(type_, value) == value
    for value in bad:
        failure = error(type_, value)
        assert failure["msg"].endswith(f"; got {value!r}"), failure


@pytest.mark.parametrize("value", ["server", "client"])
def test_role_name_reserved(value: str) -> None:
    assert "pick another name" in error(common.RoleName, value)["msg"]


def test_ref_id_suffix_hint() -> None:
    assert error(common.RefId, "kimi-k3/tp32-ep.yaml")["msg"].endswith(
        "write 'kimi-k3/tp32-ep'"
    )


@pytest.mark.parametrize("value", ["kimi-k3/tp32-ep", "./tp32-ep.yaml", "../m/x.yml"])
def test_ref_or_path_accepts(value: str) -> None:
    assert check(common.RefOrPath, value) == value


@pytest.mark.parametrize("value", ["/abs/x.yaml", "./x", "tp32-ep.yaml"])
def test_ref_or_path_rejects(value: str) -> None:
    assert error(common.RefOrPath, value)["type"] == "ref_id"


def test_env_map_coerces_numbers() -> None:
    assert check(common.EnvMap, {"A": 8, "B": 0.5, "C": "x"}) == {
        "A": "8",
        "B": "0.5",
        "C": "x",
    }


def test_env_map_rejects_bool_and_null() -> None:
    assert "quote it" in error(common.EnvMap, {"A": True})["msg"]
    assert 'write ""' in error(common.EnvMap, {"A": None})["msg"]


def test_role_env_map_keeps_null() -> None:
    assert check(common.RoleEnvMap, {"A": None, "B": 1}) == {"A": None, "B": "1"}


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("--max-model-len", "write the key without dashes: max-model-len"),
        ("max_model_len", "use max-model-len"),
        ("compilation_config.x_y", "use compilation-config.x_y"),
        ("Max-Len", "arg keys are flag names without the leading dashes"),
    ],
)
def test_arg_key_form(key: str, message: str) -> None:
    failure = error(common.ArgMap, {key: 1})
    assert failure["type"] == "V-15"
    assert failure["msg"].startswith(message)


def test_arg_map_values() -> None:
    args = {
        "enable-expert-parallel": True,
        "max-model-len": 9216,
        "gpu-memory-utilization": 0.9,
        "reasoning-parser": "glm45",
        "compilation-config": {"compile_sizes": [1, 8]},
        "profiler-config.profiler": "torch",
        "x": [1, "a"],
    }
    assert check(common.ArgMap, args) == args
    assert error(common.ArgMap, {"x": None})["loc"][0] == "x"
    assert check(common.RoleArgMap, {"x": None}) == {"x": None}


def test_numbers() -> None:
    assert check(common.PosInt, 1) == 1
    for bad in (0, True, "1", 1.0):
        error(common.PosInt, bad)
    assert check(common.Tolerance, 0.05) == 0.05
    for bad in (0, 1, 1.5):
        error(common.Tolerance, bad)
    assert check(common.UnitInterval, 0) == 0
    assert check(common.NonNegInt, 0) == 0


def test_key_path_text() -> None:
    assert check(common.KeyPathText, "server.args.max-num-seqs")
    assert error(common.KeyPathText, "a..b")["type"] == "key_path"


def test_json_schema_reflects_yaml_form() -> None:
    def schema(type_: Any) -> dict[str, Any]:
        return TypeAdapter(type_).json_schema()

    assert {"type": "integer", "minimum": 1, "maximum": 86400} in schema(
        common.Duration
    )["anyOf"]
    assert "tpu7x/2x2x2" in schema(common.Slice)["enum"]
    assert schema(common.Name)["pattern"] == "^[a-z0-9][a-z0-9._-]{0,62}$"
    assert schema(common.ByteSize)["type"] == "string"
