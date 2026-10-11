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
"""The Plan's JSON and encoded forms (R-PLAN-7, R-NF-3, R-NF-7)."""

import base64
import gzip
import json

import pytest
from plan_fixtures import VARIANTS, Data, build
from scenario.plan import codec
from scenario.plan.types import Concat, Ref


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_round_trip(name: str, variants: dict[str, Data]) -> None:
    plan = build(variants[name])
    assert codec.decode(codec.encode(plan)) == plan
    assert codec.from_json(codec.to_json(plan)) == plan


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_deterministic_and_small(name: str, variants: dict[str, Data]) -> None:
    plan = build(variants[name])
    again = build(variants[name])
    encoded = codec.encode(plan)
    assert encoded == codec.encode(again) == codec.encode(codec.decode(encoded))
    assert "$" not in encoded
    assert len(encoded) <= codec.MAX_ENCODED_BYTES
    resolved = codec.encode_data({"point": None, "scenario": variants[name]})
    assert len(resolved) <= codec.MAX_ENCODED_BYTES
    assert len(codec.digest(plan)) == 12 and codec.digest(plan) == codec.digest(again)


def test_json_is_canonical(variants: dict[str, Data]) -> None:
    text = codec.to_json(build(variants["pd"]))
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":"))
    assert '"$5"' in text and text.isascii()


def test_refs_are_tagged() -> None:
    value = Concat(("http://", Ref("role_head", role="prefill"), ":8000"))
    assert codec._dump(value) == {
        "concat": ["http://", {"ref": "role_head", "role": "prefill"}, ":8000"]
    }
    assert codec._dump(Ref("pulled_model_dir", fallback="m")) == {
        "ref": "pulled_model_dir",
        "fallback": "m",
    }


def _encode_raw(data: object) -> str:
    raw = json.dumps(data).encode()
    return base64.b64encode(gzip.compress(raw, mtime=0)).decode()


def test_other_format_is_refused(variants: dict[str, Data]) -> None:
    data = codec.to_data(build(variants["single"]))
    data["format"] = 2
    with pytest.raises(codec.PlanFormatError) as info:
        codec.decode(_encode_raw(data))
    assert str(info.value) == (
        "plan format 2, this tool reads 1: the pod image and the submitting "
        "checkout differ"
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("not base64!", "not an encoded Plan"),
        (base64.b64encode(b"plain").decode(), "not an encoded Plan"),
        (_encode_raw({"a": 1}), "no format field"),
        (_encode_raw({"format": 1, "roles": []}), "not a Plan of format 1"),
    ],
)
def test_garbage_is_refused(text: str, message: str) -> None:
    with pytest.raises(codec.PlanFormatError, match=message):
        codec.decode(text)
