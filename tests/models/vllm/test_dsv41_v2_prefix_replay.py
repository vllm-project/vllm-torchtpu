# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1 v2 CED with prefix caching through SWA bounded replay.

The scheduler resumes a prefix hit at its replay start (the hit minus the SWA
window). The model must bound SWA visibility by that start and leave the
prefix-cached KV of the replayed tokens untouched; the runner keeps the start
per request.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.ced import (
    plan_decoder_replay,
    prefix_replay_enabled,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

WINDOW = 128


@pytest.mark.parametrize("local_rows,peer_rows", [(0, 128), (128, 0), (0, 0)])
def test_dp_decoder_bucket_preserves_empty_local_plan(
    monkeypatch, local_rows, peer_rows
):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import ced

    group = SimpleNamespace(world_size=8, cpu_group=object())
    monkeypatch.setattr(ced, "get_dp_group", lambda: group)

    def reduce(count, op, group):
        count.fill_(max(local_rows, peer_rows))

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    plan = ced.DecoderReplayPlan(
        tuple(range(local_rows)), (0, local_rows), (65536,), local_rows
    )
    coordinated = ced.coordinate_decoder_replay(plan, 8192)
    assert coordinated.padded_num_tokens == max(local_rows, peer_rows)
    assert coordinated.indices == plan.indices
    assert coordinated.query_start_loc == plan.query_start_loc


def _plan_metadata(bounds, lengths, prompts, hits=None):
    return SimpleNamespace(
        query_start_loc_cpu=tuple(bounds),
        seq_lens_cpu=tuple(lengths),
        prompt_lens_cpu=tuple(prompts),
        replay_starts_cpu=None if hits is None else tuple(hits),
    )


def test_plan_without_hits_keeps_prompt_tail():
    # One 1024-token prompt in one chunk: the decoder runs the last window.
    plan = plan_decoder_replay(
        _plan_metadata([0, 1024], [1024], [1024]), WINDOW, 1024, ()
    )
    assert plan.replay_starts == (1024 - WINDOW,)
    assert plan.indices == tuple(range(1024 - WINDOW, 1024))


def test_plan_fresh_hit_resumes_inside_the_scheduled_chunk():
    # A 1024-token prompt hit at 640 resumes at 512. The decoder still starts
    # at the prompt tail, which the scheduled chunk [512, 1024) covers.
    metadata = _plan_metadata([0, 512], [1024], [1024], hits=[512])
    plan = plan_decoder_replay(metadata, WINDOW, 512, ())
    assert plan.replay_starts == (1024 - WINDOW,)
    assert plan.indices == tuple(range(512 - WINDOW, 512))


def test_plan_resumed_hit_past_the_prompt_bounds_the_decoder():
    # A preempted request (prompt 256, 1000 tokens so far) resumes at 768:
    # no decoder SWA row exists below it.
    metadata = _plan_metadata([0, 232], [1000], [256], hits=[768])
    plan = plan_decoder_replay(metadata, WINDOW, 256, ())
    assert plan.replay_starts == (768,)
    assert plan.indices == tuple(range(232))


def test_plan_rejects_mismatched_hits():
    metadata = _plan_metadata([0, 4, 8], [4, 8], [4, 8], hits=[0])
    with pytest.raises(ValueError, match="Invalid decoder replay"):
        plan_decoder_replay(metadata, WINDOW, 8, ())


def test_replayed_marks_only_the_hit_window():
    # Two requests: the first resumes at 256 (tokens 256..300 scheduled),
    # the second has no hit (replay start 0).
    positions = torch.cat((torch.arange(256, 400), torch.arange(0, 16)))
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 144, 160]))
    replayed = v2_attn._replayed(metadata, positions, torch.tensor([256, 0]), WINDOW)
    expected = torch.cat(
        (
            torch.arange(256, 400) < 256 + WINDOW,
            torch.zeros(16, dtype=torch.bool),
        )
    )
    assert torch.equal(replayed, expected)


@pytest.mark.parametrize(
    "ced,caching,bounded,pcp,expected",
    [
        (True, True, True, 1, True),
        (False, True, True, 1, False),
        (True, False, True, 1, False),
        (True, True, False, 1, False),
        (True, True, True, 2, False),
    ],
)
def test_prefix_replay_enabled(monkeypatch, ced, caching, bounded, pcp, expected):
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", ced, raising=False)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=caching, swa_bounded_replay=bounded
        ),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp),
    )
    assert prefix_replay_enabled(config) is expected


def _runner(replay_tokens=WINDOW):
    runner = object.__new__(TPUModelRunner)
    spec = SimpleNamespace(prefix_replay_tokens=replay_tokens)
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]
    )
    runner._replay_starts = {}
    return runner


def _scheduler_output(new=(), cached=(), resumed=()):
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, replay_start=s, num_computed_tokens=s)
            for r, s in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _ in cached],
            num_computed_tokens=[n for _, n in cached],
            resumed_req_ids=set(resumed),
        ),
    )


def test_record_replay_starts_for_new_and_resumed_requests():
    runner = _runner()
    output = _scheduler_output(
        new=[("hit", 512), ("cold", 0)],
        cached=[("running", 900), ("resumed", 768), ("resumed_cold", 0)],
        resumed=["resumed", "resumed_cold"],
    )
    runner._record_replay_starts(output)
    assert runner._replay_starts == {"hit": 512, "resumed": 768}


def test_record_replay_starts_is_off_without_a_replay_group():
    runner = _runner(replay_tokens=0)
    runner._record_replay_starts(_scheduler_output(new=[("hit", 512)]))
    assert runner._replay_starts == {}


class _Buffer:
    def __init__(self, rows, depth):
        self.np = np.zeros((rows, depth), dtype=np.int64)

    def copy_to_gpu(self):
        return torch.from_numpy(self.np.copy())


def test_lookback_reads_committed_tokens_on_a_replayed_admission():
    runner = _runner()
    runner.lookback_token_ids = _Buffer(rows=3, depth=2)
    tokens = np.arange(1000, dtype=np.int64)[None, :].repeat(3, axis=0)
    runner.input_batch = SimpleNamespace(
        req_ids=["resumed", "later", "fresh"],
        num_computed_tokens_cpu=np.array([768, 800, 512]),
        num_prompt_tokens=np.array([256, 256, 1024]),
        num_tokens_no_spec=np.array([1000, 1000, 1024]),
        token_ids_cpu=tokens,
    )
    # "later" resumed at 768 earlier, so this step is not its admission.
    runner._replay_starts = {"resumed": 768, "later": 768, "fresh": 512}
    ids = runner._prepare_lookback_token_ids(3)
    assert ids[0].tolist() == [767, 766]
    # Generated positions on later steps may hold async placeholders.
    assert ids[1].tolist() == [-1, -1]
    assert ids[2].tolist() == [511, 510]
