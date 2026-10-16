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
"""The rejection-sampler precompile sweep: shape order with and without
TPU_PARALLEL_PRECOMPILE, and why rotating it is safe."""

import contextlib
import functools
from types import SimpleNamespace

import pytest
import torch

from tests.runner.precompile_sweep_checks import ops_outside_aten as _ops_outside_aten
from vllm_torchtpu.layers.adapter.sample import rejection_sampler as rs
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import CANDIDATE_BLOCK
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

FLAG = "TPU_PARALLEL_PRECOMPILE"
K = 3


class _Sampler:
    """Records the (num_tokens, num_reqs) of each warmup call, read off the
    tensors the sweep built for it."""

    synthetic_mode = False

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def __call__(self, *, draft_token_ids, num_draft_tokens, **kwargs):
        self.calls.append((draft_token_ids.shape[0], num_draft_tokens.shape[0]))
        return draft_token_ids


def _logits(dummy_logits, bonus_logits_indices, target_logits_indices):
    return bonus_logits_indices, dummy_logits


def _runner():
    runner = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        rejection_sampler=_Sampler(),
        num_tokens_paddings=[32, 64, 128, 256, 384, 512, 768],
        num_reqs_paddings=[8, 16, 32, 64, 128, 192],
        max_num_reqs=192,
        vocab_size=4,
        _hidden_states_dtype=torch.float32,
        device="cpu",
        spec_bonus_and_target_logits=_logits,
        spec_gather_bonus_and_target_logits=_logits,
        _precompile_timed=lambda name: contextlib.nullcontext(),
        # The greedy top-token verify and top-k candidate warmups are off.
        _greedy_head_owner=None,
        _topk_candidate_blocks=lambda: 0,
    )
    runner._spec_verify_bucket_reachable = functools.partial(
        TPUModelRunner._spec_verify_bucket_reachable, runner
    )
    return runner, runner.rejection_sampler.calls


def _sweep(monkeypatch, flag, rank=3):
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    runner, calls = _runner()
    TPUModelRunner._precompile_rejection_sampler(runner)  # type: ignore[arg-type]
    # Each warmup shape runs the greedy and the non-greedy verify path.
    assert calls[::2] == calls[1::2]
    return calls[::2]


# The reachable (padded_logits_length, padded_num_reqs) pairs for these
# paddings with K=3, in runtime bucket order; max_logits_len is 768.
PAIRS = [
    (32, 8),
    (32, 16),
    (32, 32),
    (64, 16),
    (64, 32),
    (64, 64),
    (128, 32),
    (128, 64),
    (128, 128),
    (256, 64),
    (256, 128),
    (256, 192),
    (384, 128),
    (384, 192),
    (512, 128),
    (512, 192),
    (768, 192),
]


@pytest.mark.parametrize("flag", [None, "0"])
def test_order_is_the_bucket_order_without_the_flag(monkeypatch, flag):
    assert _sweep(monkeypatch, flag) == PAIRS


@pytest.mark.parametrize("rank", [0, 3, 7, 11])
def test_flag_starts_each_local_rank_at_its_share_of_the_sweep(monkeypatch, rank):
    # 8 local ranks per host; a local rank past 8 wraps around.
    start = rank % 8 * len(PAIRS) // 8
    assert _sweep(monkeypatch, "1", rank=rank) == PAIRS[start:] + PAIRS[:start]


def _recording(fn, graphs):
    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    return torch.compile(
        fn._torchdynamo_orig_callable, backend=record, fullgraph=True, dynamic=False
    )


def _no_collectives(*args, **kwargs):
    raise AssertionError("eager collective in the rejection-sampler sweep")


@pytest.mark.parametrize("optional_paths", [False, True])
@pytest.mark.parametrize("synthetic", [False, True])
def test_rotated_sweep_holds_no_collective(monkeypatch, synthetic, optional_paths):
    """Rotation lets ranks reach each shape at different times, so a collective
    anywhere in the sweep would deadlock them. Run the real sweep on CPU, with
    the real rejection sampler and verify preludes recorded as they compile,
    and with torch.distributed's collectives raising if called eagerly.
    optional_paths also warms the greedy top-token verify and the top-k
    candidate sampling path."""
    graphs: list[torch.fx.GraphModule] = []
    for name in (
        "_greedy_rejection_sample_with_segment",
        "_greedy_rejection_sample_from_argmax",
        "_random_rejection_sample_with_segment",
    ):
        monkeypatch.setattr(rs, name, _recording(getattr(rs, name), graphs))
    for name in (
        "all_reduce",
        "all_gather",
        "all_gather_into_tensor",
        "reduce_scatter_tensor",
        "all_to_all_single",
        "broadcast",
    ):
        monkeypatch.setattr(torch.distributed, name, _no_collectives)
    monkeypatch.setattr(tpu_runner, "synchronize_tensors", lambda out: None)
    monkeypatch.delenv(FLAG, raising=False)

    spec_config = (
        SimpleNamespace(
            rejection_sample_method="synthetic", synthetic_acceptance_rates=[0.5] * K
        )
        if synthetic
        else None
    )
    runner = SimpleNamespace(
        _is_async_drafter=True,
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        rejection_sampler=rs.RejectionSampler(spec_config, torch.device("cpu")),
        num_tokens_paddings=[8, 16],
        num_reqs_paddings=[2, 4],
        max_num_reqs=4,
        # The top-k candidate path needs whole CANDIDATE_BLOCKs of vocab.
        vocab_size=2 * CANDIDATE_BLOCK if optional_paths else 16,
        _greedy_head_owner=object() if optional_paths else None,
        _hidden_states_dtype=torch.float32,
        device="cpu",
        _precompile_timed=lambda name: contextlib.nullcontext(),
    )
    runner.spec_bonus_and_target_logits = functools.partial(
        _recording(TPUModelRunner.spec_bonus_and_target_logits, graphs), None
    )
    runner.spec_gather_bonus_and_target_logits = functools.partial(
        _recording(TPUModelRunner.spec_gather_bonus_and_target_logits, graphs), None
    )
    runner._spec_verify_bucket_reachable = functools.partial(
        TPUModelRunner._spec_verify_bucket_reachable, runner
    )
    candidate_calls = []

    def topk_candidate_blocks():
        candidate_calls.append(1)
        return 2 if optional_paths else 0

    runner._topk_candidate_blocks = topk_candidate_blocks
    token_graphs: list[torch.fx.GraphModule] = []
    runner.spec_bonus_and_target_tokens = functools.partial(
        _recording(TPUModelRunner.spec_bonus_and_target_tokens, token_graphs), None
    )
    runner._precompile_top_token_verify = functools.partial(
        TPUModelRunner._precompile_top_token_verify, runner
    )
    torch._dynamo.reset()
    TPUModelRunner._precompile_rejection_sampler(runner)  # type: ignore[arg-type]

    assert len(graphs) >= 4
    # The optional paths ran only when enabled.
    assert bool(token_graphs) == optional_paths
    assert candidate_calls
    assert [op for gm in graphs + token_graphs for op in _ops_outside_aten(gm)] == []


@torch.library.custom_op("rotation_sweep_test::exchange", mutates_args=())
def _exchange(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_exchange.register_fake
def _(x):
    return torch.empty_like(x)


def test_check_flags_a_traced_collective(tmp_path):
    """Positive control on the thing the check exists for: a collective call
    compiled into a graph (one-rank gloo group, so it can trace on CPU). An
    earlier test in the same session may have left a default group up; the
    trace needs only some default group, so reuse it rather than init twice."""
    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1
        )
    try:
        graphs: list[torch.fx.GraphModule] = []

        def record(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        def reduce(x):
            torch.distributed.all_reduce(x)
            return x + 1

        torch.compile(reduce, backend=record, fullgraph=True)(torch.ones(2))
        assert "_c10d_functional.all_reduce" in _ops_outside_aten(graphs[0])
    finally:
        if owns_group:
            torch.distributed.destroy_process_group()


def test_check_flags_an_op_outside_aten():
    """Positive control: the check would see a custom op slipped into a graph."""
    graphs: list[torch.fx.GraphModule] = []

    def record(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch.compile(lambda x: _exchange(x) + 1, backend=record, fullgraph=True)(
        torch.ones(2)
    )
    assert _ops_outside_aten(graphs[0]) == ["rotation_sweep_test.exchange.default"]
