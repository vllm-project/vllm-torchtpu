# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SKIP_TOKENIZER_ID_RANGE_CHECK on the real renderer and input processor."""

from types import SimpleNamespace

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.renderers.base import BaseRenderer
from vllm.v1.engine.input_processor import InputProcessor

from vllm_torchtpu import input_range_check

MAX_MODEL_LEN = 16
VOCAB = 100


class ModelConfig:
    """The model config fields the checks read; the range scan alone reads the vocab
    size, so its calls count the scans."""

    max_model_len = MAX_MODEL_LEN
    runner_type = "generate"

    def __init__(self):
        self.scans = 0

    def get_vocab_size(self):
        self.scans += 1
        return VOCAB


def _processor():
    processor = object.__new__(InputProcessor)
    processor.model_config = ModelConfig()
    processor.supports_mm_inputs = False
    processor.mm_encoder_cache_size = 0
    processor.skip_prompt_length_check = False
    processor.renderer = SimpleNamespace(
        tokenizer=SimpleNamespace(max_token_id=VOCAB - 1)
    )
    return processor


def _validate(processor, ids):
    processor._validate_model_input(
        {"type": "token", "prompt_token_ids": ids}, "decoder"
    )


def _tokenized(ids):
    return BaseRenderer._build_tokens_prompt(ids, {"prompt": "text"})[
        "prompt_token_ids"
    ]


@pytest.fixture
def upstream():
    return InputProcessor._validate_model_input


@pytest.fixture
def patched(monkeypatch, upstream):
    monkeypatch.setattr(
        BaseRenderer,
        "_build_tokens_prompt",
        BaseRenderer.__dict__["_build_tokens_prompt"],
    )
    monkeypatch.setattr(InputProcessor, "_validate_model_input", upstream)
    monkeypatch.setattr(
        InputProcessor, "_tpu_skip_tokenizer_id_range_check", False, raising=False
    )
    monkeypatch.setenv("VLLM_TPU_SKIP_TOKENIZER_ID_RANGE_CHECK", "1")
    input_range_check.patch_skip_tokenizer_id_range_check()


def test_patch_installs_only_when_enabled(monkeypatch, upstream):
    build = BaseRenderer.__dict__["_build_tokens_prompt"]
    assert isinstance(build, staticmethod)
    monkeypatch.setattr(
        InputProcessor, "_tpu_skip_tokenizer_id_range_check", False, raising=False
    )
    monkeypatch.setenv("VLLM_TPU_SKIP_TOKENIZER_ID_RANGE_CHECK", "0")
    input_range_check.patch_skip_tokenizer_id_range_check()
    assert BaseRenderer.__dict__["_build_tokens_prompt"] is build
    assert InputProcessor._validate_model_input is upstream


def test_tokenizer_ids_skip_only_the_range_scan(patched):
    processor = _processor()
    _validate(processor, _tokenized([1, 2, 3]))
    assert processor.model_config.scans == 0
    with pytest.raises(VLLMValidationError, match="longer than the maximum"):
        _validate(processor, _tokenized(list(range(MAX_MODEL_LEN + 1))))
    with pytest.raises(VLLMValidationError, match="cannot be empty"):
        _validate(processor, _tokenized([]))


def test_offsets_pass_through(patched):
    prompt = BaseRenderer._build_tokens_prompt(
        [1, 2], {"prompt": "ab"}, offset_mapping=[(0, 1), (1, 2)]
    )
    assert prompt["prompt_token_offsets"] == [(0, 1), (1, 2)]
    processor = _processor()
    _validate(processor, prompt["prompt_token_ids"])
    assert processor.model_config.scans == 0


@pytest.mark.parametrize("ids", [[1, VOCAB], [-1, 2]])
def test_ids_a_user_sent_are_checked(patched, ids):
    processor = _processor()
    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        _validate(processor, ids)
    assert processor.model_config.scans == 1


def test_a_replaced_or_validated_list_is_checked_again(patched):
    processor = _processor()
    tokenized = _tokenized([1, 2, 3])
    _validate(processor, tokenized[:2])
    _validate(processor, list(tokenized))
    assert processor.model_config.scans == 2
    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        _validate(processor, tokenized + [VOCAB])
    _validate(processor, tokenized)
    _validate(processor, tokenized)
    assert processor.model_config.scans == 4


@pytest.mark.parametrize("length", [0, 1, MAX_MODEL_LEN - 1, MAX_MODEL_LEN, 99])
def test_tokenizer_ids_get_every_check_but_the_range_scan(patched, upstream, length):
    def outcome(validate, ids):
        try:
            validate(
                _processor(), {"type": "token", "prompt_token_ids": ids}, "decoder"
            )
        except VLLMValidationError as error:
            return str(error)
        return None

    ids = list(range(length))
    assert outcome(InputProcessor._validate_model_input, _tokenized(ids)) == outcome(
        upstream, ids
    )
