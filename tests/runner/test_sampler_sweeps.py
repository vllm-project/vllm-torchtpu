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
"""The sampler-side precompile sweeps (sample_from_logits, gather_logprobs and
structured_decoding's num_reqs sweep): shape order with and without
TPU_PARALLEL_PRECOMPILE, and why rotating them is safe."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tests.runner.precompile_sweep_checks import ops_outside_aten as _ops_outside_aten
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import CANDIDATE_BLOCK
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.structured_decoding import StructuredDecodingManager

FLAG = "TPU_PARALLEL_PRECOMPILE"
PADDINGS = [8, 16, 32, 64, 128, 256, 512, 1024]


def _runner(seen):
    def logits(num_reqs):
        seen.append(num_reqs)
        return torch.zeros(num_reqs, 4)

    manager = SimpleNamespace(
        structured_decode_arange=torch.arange(32, dtype=torch.int32),
        require_structured_out_cpu=torch.ones(PADDINGS[-1], 1, dtype=torch.bool),
        grammar_bitmask_cpu=torch.zeros(PADDINGS[-1], 1, dtype=torch.int32),
        structured_decode=lambda *args: args[2],
    )
    return SimpleNamespace(
        num_reqs_paddings=PADDINGS,
        device="cpu",
        _hidden_states_dtype=torch.float32,
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _dummy_logits=logits,
        _topk_candidate_blocks=lambda: 0,
        sample_from_logits_func=lambda logits, *args, **kwargs: logits,
        gather_logprobs=lambda logits, ids: SimpleNamespace(logprobs=logits),
        structured_decoding_manager=manager,
        speculative_config=None,
    )


SWEEPS = [
    TPUModelRunner._precompile_sample_from_logits,
    TPUModelRunner._precompile_gather_logprobs,
    TPUModelRunner._precompile_structured_decoding,
]


def _sweep(monkeypatch, sweep, flag, rank=3):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    # Set the attribute, not the environment (see test_rejection_sampler_sweep).
    monkeypatch.setattr(tpu_runner.envs, FLAG, flag)
    seen: list[int] = []
    sweep(_runner(seen))
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


def _trace(fn, *args, **kwargs):
    graphs: list[torch.fx.GraphModule] = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    torch.compile(
        getattr(fn, "_torchdynamo_orig_callable", fn),
        backend=record,
        fullgraph=True,
        dynamic=False,
    )(*args, **kwargs)
    assert graphs
    return graphs


def test_rotated_sweeps_hold_no_collective():
    """Rotation lets ranks reach each shape at different times, so a
    collective in any of these graphs would deadlock them. Trace the real
    functions the sweeps warm, with the variants they warm (greedy, sampling
    and top-k candidate sampling), using the check whose positive controls
    are in test_rejection_sampler_sweep.py."""
    vocab, n = 2 * CANDIDATE_BLOCK, 8
    logits = torch.randn(n, vocab)
    runner = SimpleNamespace(model_config=SimpleNamespace(max_logprobs=5))
    runner._apply_temperature = lambda lg, t: TPUModelRunner._apply_temperature(
        runner, lg, t
    )
    graphs = []
    for all_greedy, topk_candidates in [(False, 0), (True, 0), (False, 2)]:
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
        )
    graphs += _trace(
        TPUModelRunner.gather_logprobs,
        runner,
        logits,
        torch.zeros(n, 1, dtype=torch.int64),
    )
    manager = SimpleNamespace(vocab_size=vocab)
    manager._structured_decode = (
        lambda *a: StructuredDecodingManager._structured_decode(manager, *a)
    )
    manager.apply_grammar_bitmask = (
        lambda *a: StructuredDecodingManager.apply_grammar_bitmask(manager, *a)
    )
    graphs += _trace(
        StructuredDecodingManager.structured_decode,
        manager,
        torch.ones(n, 1, dtype=torch.bool),
        torch.zeros(n, vocab // 32, dtype=torch.int32),
        logits,
        torch.arange(32, dtype=torch.int32),
    )
    assert [op for gm in graphs for op in _ops_outside_aten(gm)] == []
