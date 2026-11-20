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
"""Tests for the Qwen3.5 settings parser (models/qwen3_5/settings.py)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Literal

import pytest

from vllm_torchtpu.models.qwen3_5 import settings

QWEN35 = "Qwen3_5MoeForConditionalGeneration"


@dataclasses.dataclass(frozen=True)
class _Schema:
    """A stand-in schema with one field of every supported kind."""

    a_bool: bool = False
    a_float: float = 1.0
    a_int: int = 0
    a_mode: Literal["off", "pvs"] = "off"
    a_str: str = ""
    an_optional: int | None = None


@pytest.fixture
def schema(monkeypatch):
    monkeypatch.setattr(settings, "Qwen35Settings", _Schema)
    return _Schema


def _vllm_config(arch: str | None, additional) -> SimpleNamespace:
    model_config = None if arch is None else SimpleNamespace(architectures=[arch])
    return SimpleNamespace(model_config=model_config, additional_config=additional)


def test_empty_section_gives_defaults():
    assert settings.parse(None) == settings.Qwen35Settings()
    assert settings.parse({}) == settings.Qwen35Settings()


@pytest.mark.parametrize("raw", ["x", ["a"], 3, True])
def test_section_must_be_an_object(raw):
    with pytest.raises(ValueError, match="must be a JSON object"):
        settings.parse(raw)


def test_unknown_key_is_rejected_with_the_known_keys():
    with pytest.raises(ValueError, match=r"Unknown Qwen3.5 setting\(s\) \['typo'\]"):
        settings.parse({"typo": True})


def test_unknown_key_lists_the_schema(schema):
    with pytest.raises(ValueError, match="known settings: a_bool, a_float"):
        settings.parse({"a_boll": True})


def test_valid_values_are_accepted(schema):
    parsed = settings.parse(
        {
            "a_bool": True,
            "a_float": 2,
            "a_int": 3,
            "a_mode": "pvs",
            "a_str": "x",
            "an_optional": None,
        }
    )
    assert parsed == _Schema(
        a_bool=True, a_float=2, a_int=3, a_mode="pvs", a_str="x", an_optional=None
    )
    assert settings.parse({"an_optional": 7}).an_optional == 7
    assert settings.parse({"a_float": 0.5}).a_float == 0.5


@pytest.mark.parametrize(
    "key,value",
    [
        ("a_bool", 1),
        ("a_bool", "true"),
        ("a_bool", None),
        ("a_int", True),
        ("a_int", 1.5),
        ("a_int", "3"),
        ("a_float", False),
        ("a_float", "1.0"),
        ("a_str", 3),
        ("a_mode", "qk"),
        ("a_mode", True),
        ("an_optional", "7"),
        ("an_optional", False),
    ],
)
def test_wrongly_typed_value_is_rejected(schema, key, value):
    with pytest.raises(ValueError, match=f"\\['{key}'\\]"):
        settings.parse({key: value})


@pytest.mark.parametrize(
    "archs,expected",
    [
        ([QWEN35], True),
        (["SomethingElse", QWEN35], True),
        (["Qwen3_5ForConditionalGeneration"], False),
        (["Qwen3_5MoeForCausalLM"], False),
        (["Qwen3_5ForCausalLM"], False),
        ([], False),
        (None, False),
    ],
)
def test_is_qwen3_5(archs, expected):
    assert settings.is_qwen3_5(SimpleNamespace(architectures=archs)) is expected


def test_is_qwen3_5_without_a_model_config():
    assert settings.is_qwen3_5(None) is False


def test_get_parses_the_section_for_qwen3_5(schema):
    cfg = _vllm_config(QWEN35, {"qwen3_5": {"a_bool": True}, "other": 1})
    assert settings.get(cfg) == _Schema(a_bool=True)


def test_get_validates_the_section_for_qwen3_5():
    with pytest.raises(ValueError, match="Unknown Qwen3.5 setting"):
        settings.get(_vllm_config(QWEN35, {"qwen3_5": {"typo": 1}}))


@pytest.mark.parametrize("additional", [None, {}, {"other": 1}])
def test_get_defaults_without_a_section(additional):
    for arch in (QWEN35, "LlamaForCausalLM", None):
        assert settings.get(_vllm_config(arch, additional)) == settings.Qwen35Settings()


@pytest.mark.parametrize("arch", ["LlamaForCausalLM", "Qwen3_5ForCausalLM", None])
def test_get_rejects_a_section_on_another_model(arch):
    with pytest.raises(ValueError, match="are not Qwen3.5"):
        settings.get(_vllm_config(arch, {"qwen3_5": {}}))


def _engine(num_experts: int, additional) -> SimpleNamespace:
    """A Qwen3.5 engine of Qwen3.5-397B-A17B's text config, or another
    checkpoint's expert count."""
    text = SimpleNamespace(
        num_experts=num_experts, hidden_size=4096, num_hidden_layers=60
    )
    model_config = SimpleNamespace(architectures=[QWEN35], hf_text_config=text)
    return SimpleNamespace(model_config=model_config, additional_config=additional)


@pytest.mark.parametrize(
    ("chip", "num_experts", "accepted"),
    [
        ("7x", 512, True),
        ("v6e", 512, False),
        (None, 512, False),
        ("7x", 256, False),
    ],
)
def test_quantize_target_lm_head_is_refused_where_nothing_runs_it(
    monkeypatch, chip, num_experts, accepted
):
    """The target-head quantization runs with the Qwen3.5 model patch, only for
    Qwen3.5-397B-A17B's checkpoint shape on TPU7x. Elsewhere (another chip, a
    chip that cannot be read, another Qwen3.5 checkpoint) the setting would be
    accepted and change nothing, so the engine refuses to start; the default
    value is accepted everywhere."""
    from vllm_torchtpu.models.qwen3_5 import hooks

    monkeypatch.setattr(hooks, "_chip_family", lambda: chip)
    on = _engine(num_experts, {"qwen3_5": {"quantize_target_lm_head": True}})
    if accepted:
        assert settings.get(on).quantize_target_lm_head is True
    else:
        with pytest.raises(NotImplementedError, match="quantize_target_lm_head"):
            settings.get(on)
    off = _engine(num_experts, {"qwen3_5": {"quantize_target_lm_head": False}})
    assert settings.get(off).quantize_target_lm_head is False


def test_validation_entries_keep_quantize_target_lm_head(monkeypatch):
    """Every leg of every validation entry sets quantize_target_lm_head for
    Qwen3.5-397B-A17B on TPU7x, where it is accepted."""
    import pathlib

    import yaml

    from vllm_torchtpu.models.qwen3_5 import hooks

    monkeypatch.setattr(hooks, "_chip_family", lambda: "7x")
    entries = sorted(
        (pathlib.Path(settings.__file__).parent / "validation").glob(
            "p[0-9][0-9]-*.yaml"
        )
    )
    assert entries
    for path in entries:
        legs = yaml.safe_load(path.read_text())["legs"]
        for leg in ("prefill", "decode"):
            additional = legs[leg]["additional_config"]
            assert additional["qwen3_5"]["quantize_target_lm_head"] is True
            assert settings.get(_engine(512, additional)).quantize_target_lm_head
