# SPDX-License-Identifier: Apache-2.0
"""q38 ADM-P (engine-core half): NewRequestData ships only the prompt tail on a
KV-consumer engine when VLLM_TPU_PD_PROMPT_TAIL=W > 0.

The prompt becomes PromptTail(length, start, tail) with start =
num_computed_tokens - W; the engine core's own prompt list is never changed;
ineligible requests (embeddings, prompt_is_token_ids, multimodal, penalties,
prompt logprobs), short offsets, W = 0 and non-consumer or ineligible engines
keep the full list exactly as today.

PromptTail itself is DEC-HOST-B's (vllm_torchtpu/runner/prompt_tail.py); a
stand-in with the agreed constructor is injected here, and the integration test
with the real class is added when it lands.
"""

import pickle
from types import SimpleNamespace

import pytest

from vllm_torchtpu.core import pd_prompt_tail as pt


class FakeTail:
    def __init__(self, length, start, tail):
        self.length, self.start, self.tail = length, start, tail

    def __len__(self):
        return self.length


def _sp(**kw):
    base = dict(presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, prompt_logprobs=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _nrd(n_prompt, n_computed, **kw):
    prompt = list(range(1000, 1000 + n_prompt))
    d = dict(prompt_token_ids=prompt, num_computed_tokens=n_computed, prompt_embeds=None,
             prompt_is_token_ids=None, mm_features=[], sampling_params=_sp())
    d.update(kw)
    return SimpleNamespace(**d), prompt


def test_trims_to_prompt_tail_from_num_computed_minus_window():
    nrd, prompt = _nrd(100_001, 100_000)
    pt.tail_new_request_data(nrd, 64, FakeTail)
    t = nrd.prompt_token_ids
    assert isinstance(t, FakeTail)
    assert (t.length, t.start) == (100_001, 100_000 - 64)
    assert t.tail == prompt[100_000 - 64:] and len(t.tail) == 65
    assert len(t) == len(prompt)  # len() keeps the full length
    assert len(prompt) == 100_001 and prompt[0] == 1000  # the Request's list is untouched


@pytest.mark.parametrize(
    "n_computed,window,kw",
    [
        (64, 64, {}),  # start 0: nothing to trim
        (10, 64, {}),  # start < 0
        (999, 0, {}),  # off
        (999, 64, {"prompt_embeds": "E"}),
        (999, 64, {"prompt_is_token_ids": [True] * 1000}),
        (999, 64, {"mm_features": ["img"]}),
        (999, 64, {"sampling_params": None}),
        (999, 64, {"sampling_params": _sp(presence_penalty=0.5)}),
        (999, 64, {"sampling_params": _sp(frequency_penalty=0.5)}),
        (999, 64, {"sampling_params": _sp(repetition_penalty=1.1)}),
        (999, 64, {"sampling_params": _sp(prompt_logprobs=1)}),
        (999, 64, {"sampling_params": _sp(logits_processors=["p"])}),
    ],
)
def test_ineligible_or_short_keeps_the_full_list(n_computed, window, kw):
    nrd, prompt = _nrd(1000, n_computed, **kw)
    pt.tail_new_request_data(nrd, window, FakeTail)
    assert nrd.prompt_token_ids is prompt


def test_trim_shrinks_what_is_pickled():
    nrd, prompt = _nrd(100_001, 100_000)
    full = len(pickle.dumps(prompt))
    pt.tail_new_request_data(nrd, 64, FakeTail)
    assert len(pickle.dumps(nrd.prompt_token_ids)) * 100 < full


def _cfg(role="kv_consumer", lp=None, spec=None):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=role) if role else None,
        model_config=SimpleNamespace(logits_processors=lp),
        speculative_config=SimpleNamespace(method=spec) if spec else None,
    )


@pytest.fixture
def clean_class(monkeypatch):
    from vllm.v1.core.sched import output as so

    cls = so.NewRequestData
    monkeypatch.setattr(cls, "from_request", cls.from_request)
    monkeypatch.setattr(cls, "_q38_prompt_tail_patch", False, raising=False)
    monkeypatch.setattr(pt, "_prompt_tail_cls", lambda: FakeTail)
    return cls


def _request(n_prompt, n_computed):
    return SimpleNamespace(
        request_id="r1", prompt_token_ids=list(range(n_prompt)), mm_features=[], sampling_params=_sp(),
        pooling_params=None, num_computed_tokens=n_computed, lora_request=None, prompt_embeds=None,
        prompt_is_token_ids=None, replay_start=0,
    )


def test_patch_on_consumer_trims_real_new_request_data(monkeypatch, clean_class):
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", "64")
    assert pt.patch_new_request_data_for_prompt_tail(_cfg(spec="mtp"))
    req = _request(50_000, 49_999)
    nrd = clean_class.from_request(req, ([1, 2, 3],))
    assert isinstance(nrd, clean_class) and isinstance(nrd.prompt_token_ids, FakeTail)
    assert nrd.prompt_token_ids.start == 49_999 - 64 and len(nrd.prompt_token_ids) == 50_000
    assert len(req.prompt_token_ids) == 50_000


@pytest.mark.parametrize(
    "cfg,window",
    [
        (_cfg(role="kv_producer"), "64"),
        (_cfg(role=None), "64"),
        (_cfg(lp=["my.Proc"]), "64"),
        (_cfg(spec="ngram"), "64"),
        (_cfg(), "0"),
    ],
)
def test_no_patch_when_ineligible_engine_or_off(monkeypatch, clean_class, cfg, window):
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", window)
    assert not pt.patch_new_request_data_for_prompt_tail(cfg)
    nrd = clean_class.from_request(_request(5_000, 4_999), ([1],))
    assert isinstance(nrd.prompt_token_ids, list) and len(nrd.prompt_token_ids) == 5_000


@pytest.mark.parametrize("window", ["1", "15"])
def test_no_patch_below_the_workers_minimum_window(monkeypatch, clean_class, window):
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", window)
    assert not pt.patch_new_request_data_for_prompt_tail(_cfg())


def test_min_window_matches_the_worker():
    from vllm_torchtpu.runner import prompt_tail

    assert pt.MIN_WINDOW == prompt_tail.MIN_TOKENS_BELOW_COMPUTED


def test_window_16_installs(monkeypatch, clean_class):
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", "16")
    assert pt.patch_new_request_data_for_prompt_tail(_cfg())


def test_no_patch_without_the_worker_half(monkeypatch, clean_class):
    def missing():
        raise ImportError("no prompt_tail")

    monkeypatch.setattr(pt, "_prompt_tail_cls", missing)
    monkeypatch.setenv("VLLM_TPU_PD_PROMPT_TAIL", "64")
    assert not pt.patch_new_request_data_for_prompt_tail(_cfg())


def test_compile_neutral():
    from vllm_torchtpu.compilation import tpu_compiler

    assert "VLLM_TPU_PD_PROMPT_TAIL" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
