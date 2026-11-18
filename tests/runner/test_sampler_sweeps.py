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
"""The sampler-side precompile sweeps (sample_from_logits, gather_logprobs,
structured_decoding's num_reqs sweep, and thinking_budget's num_reqs +
verify_logits sweeps): shape order with and without TPU_PARALLEL_PRECOMPILE,
and why rotating them is safe."""

import contextlib
import types
from types import SimpleNamespace

import pytest
import torch
from vllm.config.reasoning import ReasoningConfig

from tests.runner.precompile_sweep_checks import ops_outside_aten as _ops_outside_aten
from vllm_torchtpu.layers.adapter.sample import scatter_ops, thinking_budget
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import CANDIDATE_BLOCK
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.thinking_budget_tracker import ThinkingBudgetTracker
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.structured_decoding import StructuredDecodingManager

FLAG = "TPU_PARALLEL_PRECOMPILE"
PADDINGS = [8, 16, 32, 64, 128, 256, 512, 1024]
VERIFY_PADDINGS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _reasoning_config() -> ReasoningConfig:
    cfg = ReasoningConfig(reasoning_parser="qwen3")
    cfg._reasoning_start_token_ids = [105, 2364, 107]
    cfg._reasoning_end_token_ids = [106, 107]
    cfg._natural_reasoning_end_token_ids = [106]
    cfg._enabled = True
    return cfg


def _runner(seen, speculative=False):
    def logits(num_reqs):
        seen.append(num_reqs)
        return torch.zeros(num_reqs, 4)

    manager = SimpleNamespace(
        structured_decode_arange=torch.arange(32, dtype=torch.int32),
        require_structured_out_cpu=torch.ones(PADDINGS[-1], 1, dtype=torch.bool),
        grammar_bitmask_cpu=torch.zeros(PADDINGS[-1], 1, dtype=torch.int32),
        target_grammar_bitmask_cpu=torch.zeros(
            VERIFY_PADDINGS[-1], 1, dtype=torch.int32
        ),
        structured_decode=lambda *args: args[2],
    )
    runner = SimpleNamespace(
        num_reqs_paddings=PADDINGS,
        num_tokens_paddings=VERIFY_PADDINGS,
        device=torch.device("cpu"),
        _hidden_states_dtype=torch.float32,
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _dummy_logits=logits,
        _topk_candidate_blocks=lambda: 0,
        sample_from_logits_func=lambda logits, *args, **kwargs: logits,
        gather_logprobs=lambda logits, ids: SimpleNamespace(logprobs=logits),
        structured_decoding_manager=manager,
        speculative_config=(
            SimpleNamespace(num_speculative_tokens=3) if speculative else None
        ),
        _mtp_prefill_only=False,
        _greedy_head_owner=None,
        _thinking_budget=ThinkingBudgetTracker(
            _reasoning_config(), max_num_reqs=PADDINGS[-1], device=torch.device("cpu")
        ),
    )
    # The real helper; without a speculative config it yields no verify rows.
    runner._verify_logits_paddings = types.MethodType(
        TPUModelRunner._verify_logits_paddings, runner
    )
    return runner


SWEEPS = [
    TPUModelRunner._precompile_sample_from_logits,
    TPUModelRunner._precompile_gather_logprobs,
    TPUModelRunner._precompile_structured_decoding,
    TPUModelRunner._precompile_thinking_budget,
]


def _sweep(monkeypatch, sweep, flag, rank=3, speculative=False):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    # Through the environment, as test_rejection_sampler_sweep does: envs reads
    # it lazily, and undoing a setattr on envs would leave a module attribute
    # that hides later changes to the environment from every other test.
    if flag:
        monkeypatch.setenv(FLAG, "1")
    else:
        monkeypatch.delenv(FLAG, raising=False)
    seen: list[int] = []
    with torch.compiler.set_stance("force_eager"):
        sweep(_runner(seen, speculative=speculative))
    return seen


@pytest.mark.parametrize("sweep", SWEEPS)
def test_order_is_the_bucket_order_without_the_flag(monkeypatch, sweep):
    assert _sweep(monkeypatch, sweep, False, rank=0) == PADDINGS
    assert _sweep(monkeypatch, sweep, False, rank=5) == PADDINGS


@pytest.mark.parametrize("sweep", SWEEPS)
@pytest.mark.parametrize("rank", [0, 3, 7, 11])
def test_flag_starts_each_local_rank_at_its_share_of_the_sweep(
    monkeypatch, sweep, rank
):
    # 8 local ranks per host; a local rank past 8 wraps around.
    start = rank % 8 * len(PADDINGS) // 8
    assert _sweep(monkeypatch, sweep, True, rank=rank) == (
        PADDINGS[start:] + PADDINGS[:start]
    )


@pytest.mark.parametrize("rank", [0, 3, 7, 11])
def test_thinking_budget_verify_logits_sweep_rotation(monkeypatch, rank):
    # Plain rows force with the [max_num_reqs] slot-width IDs and verify target
    # rows with [num_tokens] ones, so a verify bucket reuses a plain bucket's
    # force_logits_rows graph only at num_tokens == max_num_reqs.
    max_reqs = PADDINGS[-1]
    verify_only = [n for n in VERIFY_PADDINGS if n != max_reqs]
    assert _sweep(
        monkeypatch,
        TPUModelRunner._precompile_thinking_budget,
        False,
        rank=rank,
        speculative=True,
    ) == (PADDINGS + verify_only)
    req_start = rank % 8 * len(PADDINGS) // 8
    ver_start = rank % 8 * len(VERIFY_PADDINGS) // 8
    rot_ver = VERIFY_PADDINGS[ver_start:] + VERIFY_PADDINGS[:ver_start]
    assert _sweep(
        monkeypatch,
        TPUModelRunner._precompile_thinking_budget,
        True,
        rank=rank,
        speculative=True,
    ) == (
        PADDINGS[req_start:]
        + PADDINGS[:req_start]
        + [n for n in rot_ver if n != max_reqs]
    )


def _trace(fn, *args, **kwargs):
    graphs: list[torch.fx.GraphModule] = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    try:
        torch.compile(
            getattr(fn, "_torchdynamo_orig_callable", fn),
            backend=record,
            fullgraph=True,
            dynamic=False,
        )(*args, **kwargs)
    finally:
        # Drop the recording backend's entries so later tests compile afresh.
        torch._dynamo.reset()
    assert graphs
    return graphs


def test_rotated_sweeps_hold_no_collective():
    """Rotation lets ranks reach each shape at different times, so a
    collective in any of these graphs would deadlock them. Trace the real
    functions the sweeps warm, with the variants they warm (greedy, sampling
    with and without top-k/top-p, and top-k candidate sampling), using the
    check whose positive controls are in test_rejection_sampler_sweep.py."""
    vocab, n = 2 * CANDIDATE_BLOCK, 8
    logits = torch.randn(n, vocab)
    runner = SimpleNamespace(model_config=SimpleNamespace(max_logprobs=5))
    runner._apply_temperature = lambda lg, t: TPUModelRunner._apply_temperature(
        runner, lg, t
    )
    graphs = []
    for all_greedy, topk_candidates, apply_topk_topp in [
        (False, 0, True),
        (False, 0, False),
        (True, 0, True),
        (False, 2, True),
    ]:
        graphs += _trace(
            TPUModelRunner.sample_from_logits,
            runner,
            logits,
            torch.ones(n, 1),
            torch.rand_like(logits),
            torch.zeros(n, 1, dtype=torch.int32),
            torch.ones(n, 1),
            all_greedy=all_greedy,
            topk_candidates=topk_candidates,
            apply_topk_topp=apply_topk_topp,
        )
    graphs += _trace(
        TPUModelRunner.gather_logprobs,
        runner,
        logits,
        torch.zeros(n, 1, dtype=torch.int64),
    )
    manager = SimpleNamespace(vocab_size=vocab)
    manager._structured_decode = lambda *a: (
        StructuredDecodingManager._structured_decode(manager, *a)
    )
    manager.apply_grammar_bitmask = lambda *a: (
        StructuredDecodingManager.apply_grammar_bitmask(manager, *a)
    )
    graphs += _trace(
        StructuredDecodingManager.structured_decode,
        manager,
        torch.ones(n, 1, dtype=torch.bool),
        torch.zeros(n, vocab // 32, dtype=torch.int32),
        logits,
        torch.arange(32, dtype=torch.int32),
    )

    markers = thinking_budget.ThinkingBudgetMarkers.from_reasoning_config(
        _reasoning_config()
    )
    assert markers is not None
    kernel_args = markers.kernel_args()
    state_table = torch.full(
        (n + 1, thinking_budget.NUM_STATE_COLS), -1, dtype=torch.int32
    )
    slot_ids = torch.zeros(n, dtype=torch.int32)
    forced_ids = torch.full((n,), -1, dtype=torch.int32)
    selected = torch.zeros((n, 1), dtype=torch.int64)
    tokens_1d = torch.zeros(n, dtype=torch.int64)
    draft_ids = torch.zeros(16, dtype=torch.int32)
    forced_tgt = torch.full((16,), -1, dtype=torch.int32)
    did_force = torch.zeros(16, dtype=torch.bool)
    seg_ids = torch.full((16,), n, dtype=torch.int64)
    grp_ids = torch.zeros(16, dtype=torch.int32)
    next_toks = torch.full((n, 4), -1, dtype=torch.int32)

    graphs += _trace(
        scatter_ops.seed_slot_rows,
        state_table,
        slot_ids,
        torch.zeros((n, thinking_budget.NUM_STATE_COLS), dtype=torch.int32),
    )
    graphs += _trace(
        thinking_budget.query_forced_tokens,
        state_table,
        slot_ids,
        markers.natural_end_ids,
        markers.end_ids,
    )
    graphs += _trace(thinking_budget.force_logits_rows, logits, forced_ids)
    graphs += _trace(thinking_budget.force_token_ids, tokens_1d, forced_ids)
    graphs += _trace(
        thinking_budget.invalidate_forced_drafts, draft_ids, forced_tgt, did_force
    )
    graphs += _trace(
        thinking_budget.advance_plain_states,
        state_table,
        slot_ids,
        selected,
        *kernel_args,
    )
    graphs += _trace(
        thinking_budget.override_plain_top_tokens,
        state_table,
        slot_ids,
        selected,
        markers.natural_end_ids,
        markers.end_ids,
    )
    graphs += _trace(
        thinking_budget.prepare_spec_verify_drafts,
        state_table,
        slot_ids,
        draft_ids,
        seg_ids,
        grp_ids,
        3,
        *kernel_args,
    )
    graphs += _trace(
        thinking_budget.advance_spec_states,
        state_table,
        slot_ids,
        next_toks,
        3,
        *kernel_args,
    )
    assert [op for gm in graphs for op in _ops_outside_aten(gm)] == []
