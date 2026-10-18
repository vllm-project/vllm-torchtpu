# SPDX-License-Identifier: Apache-2.0
"""Incremental prompt tokenization at ChatML turn boundaries.

Every result is compared with a full tokenization by the same tokenizer: a
byte-level BPE trained here on a small corpus, with Qwen2's normalizer,
pre-tokenizer split pattern and added-token settings, so the tests need no
network.
"""

import inspect
import multiprocessing
import types

import pytest
from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from transformers import PreTrainedTokenizerFast
from vllm.renderers.base import BaseRenderer
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import TokenizeParams

from vllm_torchtpu.frontend import tokenize_prefix_cache
from vllm_torchtpu.frontend.tokenize_prefix_cache import (
    BOUNDARY,
    TurnPrefixTokenCache,
    build_cache,
)

# Qwen2's pre-tokenizer split pattern (tokenizer.json of the Qwen2/3 models).
_QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 4,
    "def tokenize(text):\n    return [ord(c) for c in text]\n" * 3,
    "Résumé naïve café — façade, über straße. " * 3,
    "日本語のテキストと中文文本混合。 " * 3,
    "Emoji 🙂🚀 and numbers 1234567890, tabs\tand\r\nnewlines. " * 3,
    '{"tool": "bash", "args": {"cmd": "ls -la /tmp"}} ' * 3,
]


def _make_tokenizer(**boundary_flags) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(_QWEN2_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        _CORPUS,
        trainers.BpeTrainer(
            vocab_size=600,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    flags = {"lstrip": False, "rstrip": False, "normalized": False}
    flags.update(boundary_flags)
    tok.add_special_tokens(
        [
            AddedToken(BOUNDARY, special=True, **flags),
            AddedToken("<|im_end|>", special=True, normalized=False),
        ]
    )
    tok.add_tokens([AddedToken(t, normalized=False) for t in ("<think>", "</think>")])
    return PreTrainedTokenizerFast(tokenizer_object=tok)


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    return _make_tokenizer()


def _full(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _cache(tokenizer, **kwargs) -> TurnPrefixTokenCache:
    kwargs.setdefault("min_cached_tokens", 0)
    return TurnPrefixTokenCache(
        lambda s: _full(tokenizer, s),
        tokenizer.convert_tokens_to_ids(BOUNDARY),
        **kwargs,
    )


def _chatml(messages: list[tuple[str, str]]) -> str:
    turns = "".join(f"{BOUNDARY}{role}\n{text}<|im_end|>\n" for role, text in messages)
    return turns + f"{BOUNDARY}assistant\n"


_MESSAGES = [
    ("system", "You are a coding agent. Use the tools."),
    ("user", "List the files in /tmp and explain them."),
    ("assistant", '<think>\n\n</think>\n\n{"tool": "bash", "args": {"cmd": "ls"}}'),
    ("tool", "a.txt\nb.py\n  c dir/\n\n\n"),
    ("assistant", "Résumé: two files, one dir. 日本語 🙂"),
    ("user", f"Why does my log contain {BOUNDARY} in the middle of a line?"),
    ("assistant", "e\u0301 combining marks and \u212b (angstrom) normalize under NFC."),
    ("user", "   leading spaces, trailing spaces   "),
    ("assistant", "<|im_end|> typed inside a message"),
    ("user", "def f():\n\treturn 1\r\n"),
]


def _check(tokenizer, cache: TurnPrefixTokenCache, prompts: list[str]) -> None:
    for prompt in prompts:
        assert cache.encode(prompt) == _full(tokenizer, prompt), prompt


def test_multi_turn_matches_full_tokenization(tokenizer):
    cache = _cache(tokenizer)
    prompts = [_chatml(_MESSAGES[:k]) for k in range(1, len(_MESSAGES) + 1)]

    _check(tokenizer, cache, prompts)

    assert cache.hits == len(prompts) - 1
    assert not cache.disabled
    # Each turn extends the previous one, which leaves nothing to match it.
    assert cache.num_entries == 1
    assert cache.cached_tokens == len(_full(tokenizer, prompts[-1]))


def _edited(i: int, text: str) -> list[tuple[str, str]]:
    return [(r, text if j == i else t) for j, (r, t) in enumerate(_MESSAGES)]


@pytest.mark.parametrize(
    "prompts",
    [
        pytest.param(
            [_chatml(_MESSAGES), _chatml(_edited(2, "edited history"))],
            id="edited_history",
        ),
        pytest.param([_chatml(_MESSAGES), _chatml(_MESSAGES[:4])], id="prefix_shrink"),
        pytest.param(
            [_chatml(_MESSAGES), _chatml(_edited(len(_MESSAGES) - 1, "regenerated"))],
            id="last_turn_replaced",
        ),
        pytest.param(
            [_chatml(_MESSAGES), _chatml(_MESSAGES)[:-3]], id="cut_inside_boundary"
        ),
        pytest.param(["no boundary at all", "no boundary at all"], id="no_boundary"),
        pytest.param([BOUNDARY, BOUNDARY + "user\n"], id="boundary_at_start"),
        pytest.param(
            ["x" + BOUNDARY * 3 + "y", "x" + BOUNDARY * 4], id="adjacent_boundaries"
        ),
        pytest.param(
            ["e\u0301" + BOUNDARY + "a", "e\u0301" + BOUNDARY + "b"],
            id="combining_mark_before_boundary",
        ),
        pytest.param(
            [" \n\n" + BOUNDARY + "a", " \n\n" + BOUNDARY + " b"],
            id="whitespace_around_boundary",
        ),
        pytest.param(
            ["<|im_" + BOUNDARY + "start|>", "<|im_" + BOUNDARY + "end|>"],
            id="partial_boundary_text",
        ),
    ],
)
def test_edge_cases_match_full_tokenization(tokenizer, prompts):
    cache = _cache(tokenizer)
    _check(tokenizer, cache, prompts + prompts)
    assert not cache.disabled


def test_cache_is_bounded(tokenizer):
    max_entries = 3
    max_cached_tokens = 4 * len(_full(tokenizer, _chatml(_MESSAGES)))
    cache = _cache(
        tokenizer, max_entries=max_entries, max_cached_tokens=max_cached_tokens
    )
    prompts = [
        _chatml([("user", f"conversation {n}")] + _MESSAGES[:k])
        for n in range(8)
        for k in range(1, len(_MESSAGES) + 1, 3)
    ]

    _check(tokenizer, cache, prompts)

    assert cache.num_entries == max_entries
    assert cache.cached_tokens <= max_cached_tokens
    # Keys of evicted prompts are dropped too.
    assert cache.num_keys <= max_entries * (len(_MESSAGES) + 2)


def test_prompt_larger_than_the_cache_is_not_kept(tokenizer):
    cache = _cache(tokenizer, max_cached_tokens=10)
    prompt = _chatml(_MESSAGES)
    _check(tokenizer, cache, [prompt, prompt])
    assert cache.num_entries == 0
    assert cache.num_keys == 0
    assert cache.hits == 0


def test_short_prompts_are_not_cached(tokenizer):
    prompt = _chatml(_MESSAGES)
    cache = _cache(tokenizer, min_cached_tokens=len(_full(tokenizer, prompt)) + 1)
    _check(tokenizer, cache, [prompt, prompt])
    assert cache.num_entries == 0


def test_split_mismatch_disables_the_cache(tokenizer):
    """A tokenizer whose output depends on the whole input must not be split:
    the verified hits catch it and the full tokenization is returned."""

    def encode(text: str) -> list[int]:
        return _full(tokenizer, text) + [10_000 + len(text)]

    cache = TurnPrefixTokenCache(
        encode, tokenizer.convert_tokens_to_ids(BOUNDARY), min_cached_tokens=0
    )
    for k in (3, 6, 9):
        prompt = _chatml(_MESSAGES[:k])
        assert cache.encode(prompt) == encode(prompt)
    assert cache.disabled


@pytest.mark.parametrize(
    "flags",
    [{"lstrip": True}, {"single_word": True}, {"normalized": True}],
    ids=lambda f: next(iter(f)),
)
def test_boundary_token_that_can_absorb_text_is_rejected(flags):
    assert build_cache(_make_tokenizer(**flags)) is None


def test_tokenizer_without_boundary_token_is_rejected():
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    assert build_cache(PreTrainedTokenizerFast(tokenizer_object=tok)) is None


def test_qwen_style_tokenizer_is_accepted(tokenizer):
    assert build_cache(tokenizer) is not None


def test_pinned_vllm_supports_the_patch():
    """Fails when a vLLM bump renames what the patch relies on."""
    assert tokenize_prefix_cache.find_unsupported_vllm_api() is None


@pytest.fixture
def stock_renderer(monkeypatch):
    """Stock HfRenderer tokenization, restored after the test."""
    monkeypatch.setattr(
        HfRenderer,
        "_tokenize_prompt",
        inspect.unwrap(HfRenderer._tokenize_prompt),
    )
    monkeypatch.setattr(tokenize_prefix_cache, "_installed", False)


def _renderer(tokenizer) -> HfRenderer:
    # Only the tokenizer is used on this path; a full HfRenderer needs a
    # VllmConfig.
    renderer = object.__new__(HfRenderer)
    renderer.tokenizer = tokenizer
    return renderer


def _long_conversation(turns: int) -> list[str]:
    messages = [(r, t * 20) for r, t in _MESSAGES]
    return [_chatml(messages[:k]) for k in range(1, turns + 1)]


@pytest.mark.parametrize(
    "params",
    [
        TokenizeParams(max_total_tokens=None, add_special_tokens=False),
        TokenizeParams(max_total_tokens=1_000_000, add_special_tokens=True),
        # Truncating prompts fall back to vLLM.
        TokenizeParams(max_total_tokens=3000, add_special_tokens=False),
    ],
    ids=["no_limit", "special_tokens", "truncated"],
)
def test_installed_renderer_matches_stock(
    stock_renderer, monkeypatch, tokenizer, params
):
    monkeypatch.setenv("VLLM_TPU_TOKENIZE_PREFIX_CACHE", "1")
    renderer = _renderer(tokenizer)
    prompts = _long_conversation(len(_MESSAGES))
    stock = [
        BaseRenderer._tokenize_prompt(renderer, {"prompt": p}, params) for p in prompts
    ]

    tokenize_prefix_cache.maybe_install_tokenize_prefix_cache()
    patched = [renderer._tokenize_prompt({"prompt": p}, params) for p in prompts]

    assert patched == stock
    cache = renderer.__dict__[tokenize_prefix_cache._CACHE_ATTR]
    assert cache.hits > 0


def test_env_off_leaves_renderer_unpatched(stock_renderer, monkeypatch):
    monkeypatch.delenv("VLLM_TPU_TOKENIZE_PREFIX_CACHE", raising=False)
    stock = HfRenderer._tokenize_prompt

    tokenize_prefix_cache.maybe_install_tokenize_prefix_cache()

    assert HfRenderer._tokenize_prompt is stock
    assert not tokenize_prefix_cache._installed


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("MainProcess", True),
        ("ApiServer_0", True),
        ("APIServer_DPRank_1", True),
        ("EngineCore_DP0", False),
    ],
)
def test_api_server_process_names(monkeypatch, name, expected):
    monkeypatch.setattr(
        multiprocessing, "current_process", lambda: types.SimpleNamespace(name=name)
    )
    assert tokenize_prefix_cache._is_api_server_process() is expected


def test_repeated_prompt_keeps_one_entry(tokenizer):
    cache = _cache(tokenizer)
    prompt = _chatml(_MESSAGES)

    _check(tokenizer, cache, [prompt] * 3)

    assert cache.num_entries == 1
    assert cache.cached_tokens == len(_full(tokenizer, prompt))


def test_edited_history_keeps_the_entry_it_still_shares(tokenizer):
    cache = _cache(tokenizer)
    original = _chatml(_MESSAGES)
    edited = _chatml(_edited(len(_MESSAGES) - 2, "edited"))

    _check(tokenizer, cache, [original, edited, original])

    # The edit forks after the shared prefix, so both entries stay reachable
    # and the original is a hit again.
    assert cache.num_entries == 2
    assert cache.hits == 2
