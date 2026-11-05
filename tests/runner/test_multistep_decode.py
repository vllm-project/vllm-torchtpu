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
"""K-step decode backends: loading, registration and which chunks take them."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu.runner import multistep_decode
from vllm_torchtpu.runner import tpu_runner as runner_module
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class _Backend:
    def __init__(self, runner=None, rows=64):
        self.runner = runner
        self.rows = rows

    def supports(self, num_reqs, padded_num_reqs, steps):
        return padded_num_reqs <= self.rows

    def decode(self, **kwargs):
        raise AssertionError("not called in these tests")


def _runner(**overrides):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    attrs = dict(
        _greedy_head_owner=object(),
        speculative_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        is_pooling_model=False,
        lora_config=None,
        _pcp_enabled=False,
        uses_mrope=False,
        supports_mm_inputs=False,
        k_step_decode_backend=None,
        input_batch=SimpleNamespace(req_ids=["a", "b", "c"]),
    )
    attrs.update(overrides)
    for name, value in attrs.items():
        setattr(runner, name, value)
    runner._dp_lockstep_enabled = lambda: False
    return runner


def test_load_backend_unset_is_none():
    assert multistep_decode.load_backend("", object()) is None


@pytest.mark.parametrize("spec", ["no_colon", ":factory", "module:"])
def test_load_backend_rejects_malformed_spec(spec):
    with pytest.raises(ValueError, match="module:factory"):
        multistep_decode.load_backend(spec, object())


def test_load_backend_calls_the_factory_with_the_runner():
    module = types.ModuleType("_k_step_backend_for_test")
    module.build = lambda runner: _Backend(runner)
    runner = object()
    with patch.dict(sys.modules, {module.__name__: module}):
        backend = multistep_decode.load_backend(f"{module.__name__}:build", runner)
    assert isinstance(backend, _Backend) and backend.runner is runner


def test_register_sets_the_backend():
    runner = _runner()
    backend = _Backend()
    with patch.object(runner_module, "has_kv_transfer_group", return_value=False):
        runner.register_k_step_decode_backend(backend)
    assert runner.k_step_decode_backend is backend


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"_greedy_head_owner": None}, "TPU_GREEDY_TOP_TOKENS is off"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"_pp_is_last": False}, "pipeline parallelism"),
        ({"lora_config": object()}, "LoRA"),
        ({"supports_mm_inputs": True}, "multimodal inputs"),
    ],
)
def test_register_refuses_what_the_backend_cannot_serve(overrides, reason):
    runner = _runner(**overrides)
    with (
        patch.object(runner_module, "has_kv_transfer_group", return_value=False),
        pytest.raises(ValueError, match=reason),
    ):
        runner.register_k_step_decode_backend(_Backend())
    assert runner.k_step_decode_backend is None


def _reason(runner, tokens, top_tokens=True, has_drafts=False, padded=8):
    scheduler_output = SimpleNamespace(num_scheduled_tokens=dict(zip("abc", tokens)))
    return runner._k_step_chunk_reason(
        scheduler_output, 0, len(tokens), padded, 1, top_tokens, has_drafts
    )


def test_chunk_reason():
    runner = _runner(k_step_decode_backend=_Backend(rows=8))
    assert _reason(runner, [1, 1, 1]) is None
    assert _reason(runner, [1, 5, 1]) == "prefill"
    assert _reason(runner, [1, 1, 1], has_drafts=True) == "drafts"
    assert _reason(runner, [1, 1, 1], top_tokens=False).startswith("not top-token")
    assert _reason(runner, [1, 1, 1], padded=16).startswith("backend refused")


def test_path_counter_counts_served_chunks_only(caplog):
    counter = multistep_decode.KStepPathCounter("test")
    counter.record(None, ["__warmup_0__"])
    counter.record("drafts", ["__warmup_0__", None])
    assert counter.chunks == 0 and not counter.counts
    counter.record(None, ["r1"])
    counter.record("drafts", ["r1", None])
    counter.record("drafts", ["r2"])
    assert counter.chunks == 3 and counter.counts == {"drafts": 2}


def test_advance_decode_inputs_moves_real_rows_only():
    import torch

    # 3 real requests in a bucket of 4 requests and 8 tokens; seq_lens has 6.
    tokens = torch.tensor([[11], [12], [13], [13]])
    input_ids = torch.tensor([1, 2, 3, 0, 0, 0, 0, 0], dtype=torch.int32)
    positions = torch.tensor([40, 7, 9, 0, 5, 0, 0, 0], dtype=torch.int32)
    seq_lens = torch.tensor([41, 8, 10, 0, 0, 0], dtype=torch.int32)
    ids, pos, lens = multistep_decode.advance_decode_inputs(
        tokens, input_ids, positions, seq_lens
    )
    assert ids.tolist() == [11, 12, 13, 0, 0, 0, 0, 0]
    assert pos.tolist() == [41, 8, 10, 0, 5, 0, 0, 0]
    assert lens.tolist() == [42, 9, 11, 0, 0, 0]
    assert ids.dtype == torch.int32


def test_stack_tokens_returns_the_block_and_its_last_column():
    import torch

    block, last = multistep_decode.stack_tokens(
        [torch.tensor([[1], [2]]), torch.tensor([[3], [4]])]
    )
    assert block.tolist() == [[1, 3], [2, 4]] and last.tolist() == [3, 4]


def test_advance_metadata_keeps_shared_objects_shared():
    md_a, md_b = (
        SimpleNamespace(seq_lens=0, input_positions=0, block_tables="a"),
        (SimpleNamespace(seq_lens=0, input_positions=0, block_tables="b")),
    )
    out = multistep_decode.advance_metadata(
        {"l0": md_a, "l1": md_a, "l2": md_b}, "pos", "lens"
    )
    assert out["l0"] is out["l1"] and out["l0"] is not md_a
    assert (out["l2"].seq_lens, out["l2"].input_positions) == ("lens", "pos")
    assert out["l2"].block_tables == "b" and md_b.seq_lens == 0


@pytest.mark.parametrize(
    "scheduled, drafts",
    [
        ({"a": 3, "b": 2}, {"a": [-1, -1], "b": [-1]}),
        ({"a": 3}, {"a": [-1, 5]}),
        ({"a": 3, "b": 3}, {"a": [-1, -1]}),
    ],
)
def test_burst_len_refuses_ragged_bursts(scheduled, drafts):
    out = SimpleNamespace(
        num_scheduled_tokens=scheduled,
        scheduled_spec_decode_tokens=drafts,
        scheduled_new_reqs=[],
    )
    with pytest.raises(RuntimeError, match="uniform burst"):
        multistep_decode.burst_len(out)


def test_burst_len_of_a_plain_step_is_one():
    out = SimpleNamespace(
        num_scheduled_tokens={"a": 1, "b": 7},
        scheduled_spec_decode_tokens={},
        scheduled_new_reqs=[],
    )
    assert multistep_decode.burst_len(out) == 1


def _burst_runner(tokens, burst):
    """The runner state sample_tokens reads for one greedy request (row 0 of
    a 2-row bucket) whose step was a `burst`-token burst, async scheduling."""
    from unittest.mock import MagicMock

    import numpy as np

    from vllm_torchtpu.runner import min_tokens

    runner = SimpleNamespace(
        execute_model_state=runner_module.ExecuteModelState(
            scheduler_output=SimpleNamespace(
                num_scheduled_tokens={"r0": 1},
                finished_req_ids=set(),
                has_structured_output_requests=False,
            ),
            logits_list=[tokens],
            pooler_output_list=[],
            num_reqs_list=[1],
            spec_decode_metadata_list=[None],
            top_tokens=True,
            burst_len=burst,
            burst_last_tokens=[tokens[:, -1].contiguous()],
        ),
        input_batch=SimpleNamespace(
            max_num_logprobs=None,
            num_reqs=1,
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            all_greedy=True,
            num_tokens_no_spec=np.array([17]),
            token_ids_cpu=np.zeros((1, 64), dtype=np.int32),
        ),
        requests={
            "r0": SimpleNamespace(
                req_id="r0", num_computed_tokens=16, num_tokens=17, output_token_ids=[5]
            )
        },
        _pre_async_results=None,
        _pp_wave=None,
        _dp_step_other_decode=False,
        _min_tokens=min_tokens.MinTokensTracker(),
        speculative_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=True),
        _is_async_drafter=False,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        is_pooling_model=False,
        max_num_reqs=8,
        _get_sampling_generator=lambda: None,
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
        _modify_prev_results=lambda: None,
        _stage_timing=MagicMock(),
    )
    for name in (
        "_mask_min_tokens",
        "_min_tokens_batch_state",
        "_settle_draft_placeholders_for_min_tokens",
        "_num_output_tokens",
        "maybe_wait_for_kv_save",
        "get_finished_kv_transfers",
        "_update_placeholder",
        "_sync_replicated_drafts_across_tp",
    ):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def test_burst_tokens_reach_the_output_and_the_next_steps_source():
    import torch

    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

    tokens = torch.tensor([[21, 22, 23], [9, 9, 9]])  # row 1 is padding
    runner = _burst_runner(tokens, burst=3)

    def host_copy(chunks, lengths, spec_logprobs=None):
        return AsyncTPUCopyState(
            chunks_cpu=chunks,
            chunk_real_lens=lengths,
            copy_ready_event=SimpleNamespace(synchronize=lambda: None),
            spec_logprobs=spec_logprobs,
        )

    with (
        patch.object(runner_module, "has_kv_transfer_group", return_value=False),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
        torch.compiler.set_stance("force_eager"),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
    assert output.get_output().sampled_token_ids == [[21, 22, 23]]
    pre = runner._pre_async_results
    # The next step's input token of each row is the burst's last.
    assert pre.next_tokens_tpu.tolist() == [23, 9]
    assert pre.req_id_to_index_copy == {"r0": 0}
    assert pre.num_draft_per_req == {0: 2}
    # Three optimistic placeholders, then the real tokens replace them.
    assert runner.requests["r0"].output_token_ids == [5, 0, 0, 0]
    assert runner.input_batch.num_tokens_no_spec[0] == 20
    TPUModelRunner._modify_prev_results(runner)
    assert runner.requests["r0"].output_token_ids == [5, 21, 22, 23]
    assert runner.input_batch.token_ids_cpu[0, 17:20].tolist() == [21, 22, 23]
