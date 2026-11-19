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
"""PromptTailDecodeStream emits what a stream primed with the whole prompt does."""

import random

import pytest
import tokenizers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from vllm_torchtpu import prompt_tail_decode_stream as ptds

CORPUS = (
    "To be, or not to be, that is the question:\n"
    "  def f(x):\n\treturn x  # 123 4567 89\r\n"
    "naïve café — 東京都 😀🎉 中文字符 e\u0301  \n\n"
)
SPECIALS = ["]~b]", "[e~[", "<mm:think>"]
# The class the patch replaces, whatever install() did in this process.
DecodeStream = ptds.PromptTailDecodeStream._decode_stream or decoders.DecodeStream


def _tokenizer(decoder):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoder
    trainer = trainers.BpeTrainer(
        vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator([CORPUS] * 20, trainer=trainer)
    tok.add_special_tokens([AddedToken(s, normalized=False) for s in SPECIALS])
    return tok


def _streams_agree(tok, prompt, continuation, skip):
    full = DecodeStream(ids=list(prompt), skip_special_tokens=skip)
    tail = ptds.PromptTailDecodeStream(ids=list(prompt), skip_special_tokens=skip)
    tail._decode_stream = DecodeStream
    for token in continuation:
        assert tail.step(tok, token) == full.step(tok, token)


def test_byte_level_streams_emit_the_same_text():
    tok = _tokenizer(decoders.ByteLevel())
    ids = tok.encode(CORPUS * 4, add_special_tokens=False).ids
    vocab = tok.get_vocab_size()
    rng = random.Random(0)
    tails = 0
    for _ in range(2000):
        start = rng.randrange(len(ids))
        # Prompts that end inside a multi-byte character or on special tokens.
        prompt = ids[start : start + rng.randrange(1, 200)] + [
            rng.randrange(vocab) for _ in range(rng.randrange(3))
        ]
        skip = rng.random() < 0.5
        tails += ptds.prompt_tail(tok, prompt, skip) is not prompt
        continuation = [rng.randrange(vocab) for _ in range(rng.randrange(1, 30))]
        _streams_agree(tok, prompt, continuation, skip)
    assert tails > 1000


def test_tail_only_on_a_character_boundary():
    tok = _tokenizer(decoders.ByteLevel())
    prompt = tok.encode("hello world, more words here 😀", add_special_tokens=False).ids
    assert ptds.prompt_tail(tok, prompt, False) == prompt[-ptds._TAIL_TOKENS :]
    # Drop the emoji's last byte token: the tail ends mid-character.
    cut = prompt[:-1]
    assert tok.decode(cut[-ptds._TAIL_TOKENS :]).endswith("\ufffd")
    assert ptds.prompt_tail(tok, cut, False) is cut
    # A tail of skipped special tokens decodes to nothing.
    specials = prompt + [tok.token_to_id("]~b]")] * ptds._TAIL_TOKENS
    assert ptds.prompt_tail(tok, specials, True) is specials


def test_other_decoders_keep_the_whole_prompt():
    tok = _tokenizer(decoders.Metaspace())
    seen = []

    def recording_stream(ids=None, skip_special_tokens=False):
        seen.append(ids)
        return DecodeStream(ids=ids, skip_special_tokens=skip_special_tokens)

    stream = ptds.PromptTailDecodeStream(ids=list(range(20)))
    stream._decode_stream = recording_stream
    stream.step(tok, 5)
    assert seen == [list(range(20))]


def test_install_replaces_the_module_class_once(monkeypatch):
    monkeypatch.setattr(decoders, "DecodeStream", DecodeStream)
    monkeypatch.setattr(ptds.PromptTailDecodeStream, "_decode_stream", None)
    ptds.install()
    ptds.install()
    assert decoders.DecodeStream is ptds.PromptTailDecodeStream
    assert ptds.PromptTailDecodeStream._decode_stream is DecodeStream


def test_vllm_detokenizer_output_is_unchanged(monkeypatch):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("vllm")
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=_tokenizer(decoders.ByteLevel())
    )
    prompt = tok(CORPUS * 3, add_special_tokens=False)["input_ids"]
    output = tok("中文 😀 x = f(1) 東京都", add_special_tokens=False)["input_ids"]

    def detokenize():
        request = EngineCoreRequest(
            request_id="r",
            prompt_token_ids=prompt,
            mm_features=None,
            sampling_params=SamplingParams(skip_special_tokens=True),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )
        detok = FastIncrementalDetokenizer(tok, request)
        return [detok.decode_next(t) for t in output]

    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", DecodeStream)
    want = detokenize()
    monkeypatch.setattr(ptds.PromptTailDecodeStream, "_decode_stream", None)
    ptds.install()
    got = detokenize()
    assert got == want
    assert "".join(got) == tok.decode(output)
