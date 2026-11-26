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

"""Token-parallel prefill keeps every step inside its gather buffer.

Under prefix caching the scheduler admits prefill chunks only while their pages
fit, so the runner's `check_step_capacity` stays within budget.
"""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.core import token_parallel_page_budget as tpb
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op

# Example serving shape: block 256 x dcp 16 = 4096-token virtual pages.
MAX_MODEL_LEN = 163840
BLOCK_SIZE = 256
DCP = 16
MAX_NUM_SEQS = 64
PADDED_STEP_TOKENS = 16384


def _config(*, prefix_caching=True, dcp=DCP, arch="GlmMoeDsaForCausalLM"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=prefix_caching, block_size=BLOCK_SIZE
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp, cp_kv_cache_interleave_size=4
        ),
        model_config=SimpleNamespace(
            max_model_len=MAX_MODEL_LEN,
            architectures=[arch],
            hf_text_config=SimpleNamespace(architectures=[arch]),
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=PADDED_STEP_TOKENS
        ),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        compilation_config=None,
    )


@pytest.fixture
def scheduler(monkeypatch):
    """A stand-in carrying what the patched methods read.

    vLLM's own chunk-fit check (no MTP lookahead) runs underneath the patch.
    """
    monkeypatch.setattr(tpb, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        dcp_token_parallel_op, "glm_dsa_v7x_active", lambda *_a, **_kw: True
    )
    for name in ("schedule", tpb.CHUNK_FIT_HOOK):
        monkeypatch.setattr(Scheduler, name, inspect.unwrap(getattr(Scheduler, name)))
    monkeypatch.setattr(Scheduler, "schedule", lambda self, *args, **kwargs: None)
    tpb.install_token_parallel_page_budget()
    return SimpleNamespace(vllm_config=_config(), num_prefill_lookahead=0)


def _request(num_tokens):
    return SimpleNamespace(num_tokens=num_tokens)


def _offer(scheduler, num_computed_tokens, num_new_tokens):
    request = _request(num_computed_tokens + num_new_tokens)
    return getattr(Scheduler, tpb.CHUNK_FIT_HOOK)(
        scheduler, request, num_computed_tokens, num_new_tokens
    )


def test_budget_should_match_the_gather_buffer_the_runner_builds(monkeypatch):
    monkeypatch.setattr(tpb, "glm_dsa_v7x_active", lambda *_a, **_kw: True)
    layout = dcp_token_parallel_op.layout_for(_config())

    budget = tpb.budget_for(_config())

    assert budget == tpb.PageBudget(
        virtual_page_size=layout.virtual_page_size, max_pages=layout.max_pages
    )
    assert (budget.virtual_page_size, budget.max_pages) == (4096, 108)


@pytest.mark.parametrize(
    "config",
    [
        _config(prefix_caching=False),
        _config(dcp=1),
        _config(arch="Gemma4ForConditionalGeneration"),
    ],
    ids=["no_pc", "dcp1", "non_glm"],
)
def test_budget_should_be_off_where_the_buffer_cannot_overflow(config, monkeypatch):
    monkeypatch.setattr(
        tpb,
        "glm_dsa_v7x_active",
        lambda mcfg=None: getattr(mcfg, "architectures", [None])[0]
        == "GlmMoeDsaForCausalLM",
    )
    assert tpb.budget_for(config) is None


def test_budget_should_be_off_without_glm_dsa_active(monkeypatch):
    monkeypatch.setattr(tpb, "glm_dsa_v7x_active", lambda *_a, **_kw: False)
    assert tpb.budget_for(_config()) is None


def test_should_admit_chunks_until_the_step_fills_the_buffer_exactly(scheduler):
    # 40 + 40 pages, then a 28-page chunk lands on 108 = max_pages exactly.
    assert _offer(scheduler, 160 * 1024 - 1000, 1000) == 1000
    assert _offer(scheduler, 160 * 1024 - 1000, 1000) == 1000
    assert _offer(scheduler, 0, 28 * 4096) == 28 * 4096


def test_should_defer_the_chunk_one_page_past_the_buffer(scheduler):
    assert _offer(scheduler, 160 * 1024 - 1000, 1000) == 1000
    assert _offer(scheduler, 160 * 1024 - 1000, 1000) == 1000

    assert _offer(scheduler, 0, 28 * 4096 + 1) == 0


def test_should_not_count_single_token_rows(scheduler):
    assert _offer(scheduler, 160 * 1024 - 1000, 1000) == 1000  # 40 pages
    assert _offer(scheduler, 160 * 1024 - 1000, 1000) == 1000  # 40 pages
    for _ in range(MAX_NUM_SEQS):
        assert _offer(scheduler, 150 * 1024, 1) == 1
    assert _offer(scheduler, 0, 28 * 4096) == 28 * 4096  # 80 + 28 = 108


def test_should_start_every_step_with_an_empty_buffer(scheduler):
    assert _offer(scheduler, 160 * 1024 - 1000, 1000) == 1000  # 40 pages
    assert _offer(scheduler, 160 * 1024 - 1000, 1000) == 1000  # 40 pages
    assert _offer(scheduler, 80 * 1024 - 1000, 1000) == 1000  # +20 = 100
    assert _offer(scheduler, 80 * 1024 - 1000, 1000) == 0  # +20 > 108

    Scheduler.schedule(scheduler)

    assert _offer(scheduler, 80 * 1024 - 1000, 1000) == 1000


def test_should_pass_chunks_through_without_prefix_caching(scheduler):
    scheduler.vllm_config = _config(prefix_caching=False)
    for _ in range(4):
        assert _offer(scheduler, 160 * 1024, 1000) == 1000


def test_install_should_be_idempotent(scheduler):
    patched = (Scheduler.schedule, getattr(Scheduler, tpb.CHUNK_FIT_HOOK))

    tpb.install_token_parallel_page_budget()

    assert (Scheduler.schedule, getattr(Scheduler, tpb.CHUNK_FIT_HOOK)) == patched


def _runner_step(chunks):
    """Lay out a [decode][prefill] batch the way the runner does."""
    decodes = [(150 * 1024, 1)] * 8
    rows = decodes + list(chunks)
    q_lens = np.array([new for _, new in rows], dtype=np.int32)
    seq_lens = np.array([done + new for done, new in rows], dtype=np.int32)
    query_start_loc = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    return query_start_loc, seq_lens, len(rows)


def _check_runner_capacity(chunks):
    query_start_loc, seq_lens, num_reqs = _runner_step(chunks)
    return dcp_token_parallel_op.check_step_capacity(
        _config(), PADDED_STEP_TOKENS, query_start_loc, seq_lens, num_reqs
    )


# Six requests, each with its own 80K cached history, each sending a 1K-token
# turn: 6 x 21 pages = 126 > 108.
PREFIX_HIT_TURNS = [(80 * 1024, 1024)] * 6


def test_positive_control_unbudgeted_prefix_hits_overflow_the_runner(monkeypatch):
    monkeypatch.setattr(
        dcp_token_parallel_op, "glm_dsa_v7x_active", lambda *_a, **_kw: True
    )
    dcp_token_parallel_op._state.over_capacity = False  # noqa: SLF001
    assert _check_runner_capacity(PREFIX_HIT_TURNS) is False
    assert dcp_token_parallel_op._state.over_capacity is True  # noqa: SLF001


def test_admitted_prefix_hits_should_pass_the_runner_capacity_check(scheduler):
    admitted = [
        (done, new) for done, new in PREFIX_HIT_TURNS if _offer(scheduler, done, new)
    ]

    assert len(admitted) == 5
    dcp_token_parallel_op._state.over_capacity = False  # noqa: SLF001
    assert _check_runner_capacity(admitted) is True
    assert dcp_token_parallel_op._state.over_capacity is False  # noqa: SLF001


def test_supported_for_config_should_accept_prefix_caching_for_glm_only(monkeypatch):
    monkeypatch.setattr(
        dcp_token_parallel_op,
        "glm_dsa_v7x_active",
        lambda mcfg=None: getattr(mcfg, "architectures", [None])[0]
        == "GlmMoeDsaForCausalLM",
    )
    assert dcp_token_parallel_op.supported_for_config(_config()) is True
    assert (
        dcp_token_parallel_op.supported_for_config(
            _config(arch="Gemma4ForConditionalGeneration")
        )
        is False
    )
