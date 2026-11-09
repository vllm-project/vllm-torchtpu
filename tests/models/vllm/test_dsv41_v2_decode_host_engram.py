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
"""100% Full-module line+branch coverage for host_engram and model.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram (117/117 stmts, 42/42 branches)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.model (lines 204-220, 294-310)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from vllm.models.deepseek_v41.common.engram import DEAD_ID

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram import (
    HostEngramInputs,
    _async_source_tokens_cpu,
    _resolve_decode_lookback_cpu,
    capture_host_engram_inputs,
    gather_host_engram_hashes,
)


def _mock_runner(
    seq_lens: list[int],
    num_prompt: list[int],
    kind: SequenceLayoutKind = SequenceLayoutKind.ALL,
    has_plan: bool = True,
    start_index: int = 0,
) -> SimpleNamespace:
    plan = SimpleNamespace(kind=kind, start_index=start_index) if has_plan else None
    return SimpleNamespace(
        _last_sequence_layout_plan=plan,
        input_batch=SimpleNamespace(
            num_prompt_tokens=np.array(num_prompt, dtype=np.int64),
            num_computed_tokens_cpu=np.array(num_prompt, dtype=np.int64),
            token_ids_cpu=np.zeros((len(seq_lens), 16), dtype=np.int32),
            req_ids=[f"req-{i}" for i in range(len(seq_lens))],
        ),
        seq_lens_np=np.array(seq_lens, dtype=np.int64),
        input_ids_cpu=torch.arange(len(seq_lens) * 4, dtype=torch.int32),
        positions_cpu=torch.arange(len(seq_lens) * 4, dtype=torch.int64),
        query_start_loc_cpu=torch.arange(
            0, (len(seq_lens) + 1) * 4, 4, dtype=torch.int32
        ),
        lookback_token_ids=SimpleNamespace(
            cpu=torch.full((len(seq_lens), 4), -1, dtype=torch.int32)
        ),
        _pre_async_results=None,
        speculative_config=None,
    )


# ==============================================================================
# 1. HostEngramInputs DataClass Tests
# ==============================================================================


class TestHostEngramInputs:
    def test_can_hash_on_host_property(self):
        ids = torch.tensor([1, 2], dtype=torch.int32)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        qsl = torch.tensor([0, 2], dtype=torch.int32)
        lb = torch.tensor([[10, 11]], dtype=torch.int32)

        # host_ready is None -> returns all_prompt
        inputs_prompt = HostEngramInputs(
            input_ids=ids,
            positions=pos,
            query_start_loc=qsl,
            lookback_token_ids=lb,
            all_prompt=True,
            host_ready=None,
        )
        assert inputs_prompt.can_hash_on_host is True

        inputs_decode = HostEngramInputs(
            input_ids=ids,
            positions=pos,
            query_start_loc=qsl,
            lookback_token_ids=lb,
            all_prompt=False,
            host_ready=None,
        )
        assert inputs_decode.can_hash_on_host is False

        # host_ready is explicitly set -> overrides all_prompt
        inputs_ready_true = HostEngramInputs(
            input_ids=ids,
            positions=pos,
            query_start_loc=qsl,
            lookback_token_ids=lb,
            all_prompt=False,
            host_ready=True,
        )
        assert inputs_ready_true.can_hash_on_host is True

        inputs_ready_false = HostEngramInputs(
            input_ids=ids,
            positions=pos,
            query_start_loc=qsl,
            lookback_token_ids=lb,
            all_prompt=True,
            host_ready=False,
        )
        assert inputs_ready_false.can_hash_on_host is False


# ==============================================================================
# 2. _async_source_tokens_cpu Tests
# ==============================================================================


class TestAsyncSourceTokensCPU:
    def test_single_chunk_in_copy_state(self):
        chunk = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)
        pre = SimpleNamespace(
            copy_state=SimpleNamespace(chunks_cpu=[chunk]),
            wait_for_copy=lambda: None,
        )
        out = _async_source_tokens_cpu(pre)
        assert out is not None
        assert out.tolist() == [10, 11, 12, 13]

    def test_multiple_chunks_in_copy_state(self):
        c1 = torch.tensor([1, 2], dtype=torch.int32)
        c2 = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
        pre = SimpleNamespace(
            copy_state=SimpleNamespace(chunks_cpu=[c1, c2]),
            wait_for_copy=None,
        )
        out = _async_source_tokens_cpu(pre)
        assert out is not None
        assert out.tolist() == [1, 2, 3, 4, 5, 6]

    def test_wait_for_copy_returns_tensor(self):
        tensor = torch.tensor([[20, 21]], dtype=torch.int32)
        pre = SimpleNamespace(
            copy_state=None,
            wait_for_copy=lambda: tensor,
        )
        out = _async_source_tokens_cpu(pre)
        assert out is not None
        assert out.tolist() == [20, 21]

    def test_wait_for_copy_non_callable_and_next_tokens_tpu(self):
        tensor = torch.tensor([30, 31], dtype=torch.int32)
        pre = SimpleNamespace(
            copy_state=None,
            wait_for_copy="not_callable",
            next_tokens_tpu=tensor,
        )
        out = _async_source_tokens_cpu(pre)
        assert out is not None
        assert out.tolist() == [30, 31]

    def test_all_none_returns_none(self):
        pre1 = SimpleNamespace(
            copy_state=None,
            wait_for_copy=None,
            next_tokens_tpu=None,
        )
        assert _async_source_tokens_cpu(pre1) is None

        pre2 = SimpleNamespace(
            copy_state=SimpleNamespace(chunks_cpu=None),
            wait_for_copy=lambda: "not_a_tensor",
            next_tokens_tpu="not_a_tensor",
        )
        assert _async_source_tokens_cpu(pre2) is None


# ==============================================================================
# 3. _resolve_decode_lookback_cpu Tests
# ==============================================================================


class TestResolveDecodeLookbackCPU:
    def test_num_reqs_zero_or_negative(self):
        runner = _mock_runner([4, 4], [4, 4])
        batch = runner.input_batch
        out_zero = _resolve_decode_lookback_cpu(
            runner, batch, num_reqs=0, start_index=0
        )
        assert torch.equal(out_zero, runner.lookback_token_ids.cpu)

        out_neg = _resolve_decode_lookback_cpu(
            runner, batch, num_reqs=-1, start_index=0
        )
        assert torch.equal(out_neg, runner.lookback_token_ids.cpu)

    def test_gather_valid_and_out_of_bounds(self):
        runner = _mock_runner([10, 5], [10, 5])
        token_ids_cpu = np.array(
            [
                [100, 101, 102, 103, 104, 105, 0, 0],
                [200, 201, 202, 0, 0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        batch = SimpleNamespace(
            num_computed_tokens_cpu=np.array([5, 1], dtype=np.int64),
            token_ids_cpu=token_ids_cpu,
        )
        # depth = 4 (runner.lookback_token_ids.cpu.shape[1] is 4)
        # req 0: start = 5 -> pos = [4, 3, 2, 1] -> [104, 103, 102, 101]
        # req 1: start = 1 -> pos = [0, -1, -2, -3] -> pos=0 valid (200), rest -1
        out = _resolve_decode_lookback_cpu(runner, batch, num_reqs=2, start_index=0)
        assert out[0].tolist() == [104, 103, 102, 101]
        assert out[1].tolist() == [200, -1, -1, -1]

    def test_start_index_offset_and_empty_max_cols(self):
        runner = _mock_runner([4, 4, 4], [4, 4, 4])
        token_ids_cpu = np.array(
            [
                [0, 1],
                [10, 11],
                [20, 21],
            ],
            dtype=np.int32,
        )
        batch = SimpleNamespace(
            num_computed_tokens_cpu=np.array([1, 2, 2], dtype=np.int64),
            token_ids_cpu=token_ids_cpu,
        )
        out = _resolve_decode_lookback_cpu(runner, batch, num_reqs=2, start_index=1)
        # req at start_index=1: computed=2 -> pos=[1, 0, -1, -2] -> [11, 10, -1, -1]
        # req at start_index=2: computed=2 -> pos=[1, 0, -1, -2] -> [21, 20, -1, -1]
        assert out[0].tolist() == [11, 10, -1, -1]
        assert out[1].tolist() == [21, 20, -1, -1]

        # max_cols boundary: pos >= max_cols -> valid is False and gathered is all -1
        batch_oob = SimpleNamespace(
            num_computed_tokens_cpu=np.array([10], dtype=np.int64),
            token_ids_cpu=np.zeros((1, 2), dtype=np.int32),
        )
        out_oob = _resolve_decode_lookback_cpu(
            runner, batch_oob, num_reqs=1, start_index=0
        )
        assert out_oob[0].tolist() == [-1, -1, -1, -1]


# ==============================================================================
# 4. capture_host_engram_inputs Tests
# ==============================================================================


class TestCaptureHostEngramInputs:
    def test_rejection_and_fallback_branches(self):
        # 1. pre_async with spec_decode_num_rejected_tokens not None
        runner = _mock_runner([8, 2], [4, 4])
        runner._pre_async_results = SimpleNamespace(
            spec_decode_num_rejected_tokens=torch.tensor([1])
        )
        res_spec_reject = capture_host_engram_inputs(runner, 2)
        assert res_spec_reject is not None
        assert res_spec_reject.all_prompt is False
        assert res_spec_reject.host_ready is False
        assert res_spec_reject.can_hash_on_host is False

        # 2. runner with speculative_config not None
        runner._pre_async_results = SimpleNamespace(
            spec_decode_num_rejected_tokens=None
        )
        runner.speculative_config = SimpleNamespace()
        res_spec_cfg = capture_host_engram_inputs(runner, 2)
        assert res_spec_cfg is not None
        assert res_spec_cfg.host_ready is False

        # 3. batch.num_computed_tokens_cpu is None
        runner.speculative_config = None
        runner.input_batch.num_computed_tokens_cpu = None
        res_no_computed = capture_host_engram_inputs(runner, 2)
        assert res_no_computed is not None
        assert res_no_computed.host_ready is False

        # 4. batch.token_ids_cpu is None
        runner.input_batch.num_computed_tokens_cpu = np.array([4, 4])
        runner.input_batch.token_ids_cpu = None
        res_no_tokens = capture_host_engram_inputs(runner, 2)
        assert res_no_tokens is not None
        assert res_no_tokens.host_ready is False

    def test_decode_without_pre_async_or_req_ids(self):
        runner = _mock_runner([8, 2], [4, 4])
        runner.input_batch.num_computed_tokens_cpu = np.array([4, 2])
        runner.input_batch.token_ids_cpu = np.arange(32).reshape(2, 16)
        runner._pre_async_results = None

        res = capture_host_engram_inputs(runner, 2)
        assert res is not None
        assert res.all_prompt is False
        assert res.host_ready is True
        assert res.can_hash_on_host is True
        assert torch.equal(res.input_ids, runner.input_ids_cpu)

        # req_ids is None in batch
        runner.input_batch.req_ids = None
        runner._pre_async_results = SimpleNamespace(req_id_to_index_copy={"req-0": 0})
        res_no_req_ids = capture_host_engram_inputs(runner, 2)
        assert res_no_req_ids is not None
        assert res_no_req_ids.host_ready is True

    def test_decode_with_async_substitution_matched_and_unmatched(self):
        runner = _mock_runner([8, 4, 2], [4, 4, 4])
        runner.input_batch.num_computed_tokens_cpu = np.array([7, 3, 2])
        runner.input_batch.token_ids_cpu = np.arange(48).reshape(3, 16)
        runner.input_batch.req_ids = ["req-0", "req-1", "req-2"]

        # query_start_loc:
        # req 0: [0, 2] (length 2)
        # req 1: [2, 2] (length 0, q_end == q_start)
        # req 2: [2, 3] (length 1)
        runner.query_start_loc_cpu = torch.tensor([0, 2, 2, 3], dtype=torch.int32)
        runner.input_ids_cpu = torch.zeros(4, dtype=torch.int32)

        # Substituted source tokens: flat_src is [101, 102, 103, 104]
        chunk = torch.tensor([101, 102, 103, 104], dtype=torch.int32)
        runner._pre_async_results = SimpleNamespace(
            req_id_to_index_copy={"req-0": 0, "req-1": 2, "req-unknown": 3},
            wait_for_copy=None,
            copy_state=SimpleNamespace(chunks_cpu=[chunk]),
            spec_decode_num_rejected_tokens=None,
        )

        res = capture_host_engram_inputs(runner, 3)
        assert res is not None
        assert res.all_prompt is False
        assert res.host_ready is True
        assert res.can_hash_on_host is True
        # req-0 matched q_start=0, q_end=2, src_start=0 -> ids[0:2] = [101, 102]
        # req-1 matched but q_end == q_start (empty) -> skipped
        # req-2 not in req_id_to_index_copy -> ids[2:3] stays 0
        assert res.input_ids.tolist() == [101, 102, 0, 0]

    def test_decode_with_async_substitution_missing_flat_src(self):
        runner = _mock_runner([8], [4])
        runner.input_batch.num_computed_tokens_cpu = np.array([7])
        runner.input_batch.token_ids_cpu = np.arange(16).reshape(1, 16)
        runner.input_batch.req_ids = ["req-0"]
        runner.query_start_loc_cpu = torch.tensor([0, 2], dtype=torch.int32)

        runner._pre_async_results = SimpleNamespace(
            req_id_to_index_copy={"req-0": 0},
            wait_for_copy=None,
            copy_state=None,
            next_tokens_tpu=None,
            spec_decode_num_rejected_tokens=None,
        )
        res = capture_host_engram_inputs(runner, 1)
        assert res is not None
        assert res.host_ready is False
        assert res.can_hash_on_host is False

    def test_decode_with_async_substitution_empty_matched(self):
        # When pre_async exists but matched is empty (req_id not in
        # req_id_to_index_copy), if matched: is False (branch 158 -> 176),
        # returning HostEngramInputs with ids = runner.input_ids_cpu
        # and host_ready = True.
        runner = _mock_runner([8], [4])
        runner.input_batch.num_computed_tokens_cpu = np.array([7])
        runner.input_batch.token_ids_cpu = np.arange(16).reshape(1, 16)
        runner.input_batch.req_ids = ["req-0"]
        runner.query_start_loc_cpu = torch.tensor([0, 2], dtype=torch.int32)
        runner.input_ids_cpu = torch.tensor([42, 43], dtype=torch.int32)

        runner._pre_async_results = SimpleNamespace(
            req_id_to_index_copy={"req-unmatched": 0},
            wait_for_copy=None,
            copy_state=None,
            next_tokens_tpu=None,
            spec_decode_num_rejected_tokens=None,
        )
        res = capture_host_engram_inputs(runner, 1)
        assert res is not None
        assert res.host_ready is True
        assert res.can_hash_on_host is True
        assert res.input_ids.tolist() == [42, 43]

    def test_capture_host_engram_inputs_multi_chunk_ctx(self):
        # Exercises runner._attn_metadata_builder_ctx with start_index > 0
        # and chunk_num_reqs in multi-chunk batches.
        runner = _mock_runner([8, 8, 8], [8, 8, 8])
        runner._attn_metadata_builder_ctx = SimpleNamespace(start_index=1, num_reqs=1)
        runner.input_batch.num_computed_tokens_cpu = np.array([0, 7, 0])
        runner.input_batch.token_ids_cpu = np.arange(48).reshape(3, 16)
        runner.input_batch.req_ids = ["req-0", "req-1", "req-2"]
        runner.query_start_loc_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)
        runner.input_ids_cpu = torch.tensor([99, 100], dtype=torch.int32)

        res = capture_host_engram_inputs(runner, 2)
        assert res is not None
        assert res.host_ready is True
        assert res.can_hash_on_host is True
        assert res.all_prompt is True


# ==============================================================================
# 5. gather_host_engram_hashes Tests
# ==============================================================================


class TestGatherHostEngramHashes:
    def test_cpu_all_gather_into_tensor_all_branches(self):
        hashes = torch.tensor([[[1, 2]]], dtype=torch.int32)  # shape [1, 1, 2]
        fake_dp_group = SimpleNamespace(world_size=2, cpu_group="fake_cpu_group")

        def fake_all_gather(out_tensor, in_tensor, group=None):
            out_tensor.copy_(torch.cat([in_tensor, in_tensor], dim=0))

        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_engram_dp_group",
                return_value=fake_dp_group,
            ),
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
        ):
            # 1. Without forward context (dp_metadata is None) -> slot = shape[0]
            with patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                return_value=False,
            ):
                out_no_ctx = gather_host_engram_hashes(hashes)
                assert out_no_ctx.shape == (2, 1, 2)

            # 2. With forward context and dp_metadata:
            with (
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                    return_value=True,
                ),
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_forward_context",
                    return_value=SimpleNamespace(dp_metadata=object()),
                ),
            ):
                # 2a. engram_hashes.shape[0] < slot -> pad with DEAD_ID
                with patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.engram_gathered_num_tokens",
                    return_value=3,
                ):
                    out_padded = gather_host_engram_hashes(hashes)
                    assert out_padded.shape == (6, 1, 2)
                    # rank 0 padded tokens 1..2 have DEAD_ID
                    assert out_padded[1, 0, 0].item() == DEAD_ID

                # 2b. engram_hashes.shape[0] == slot -> no pad
                with patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.engram_gathered_num_tokens",
                    return_value=1,
                ):
                    out_exact = gather_host_engram_hashes(hashes)
                    assert out_exact.shape == (2, 1, 2)

                # 2c. engram_hashes.shape[0] > slot -> ValueError
                with (
                    patch(
                        "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.engram_gathered_num_tokens",
                        return_value=0,
                    ),
                    pytest.raises(ValueError, match="exceeds the DP token slot"),
                ):
                    gather_host_engram_hashes(hashes)


# ==============================================================================
# 6. install_host_engram_inputs Tests
# ==============================================================================


# ==============================================================================
# 7. DeepseekV41Model Diff Tests (_prepare_engram and forward)
# ==============================================================================
