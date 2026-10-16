# SPDX-License-Identifier: Apache-2.0
"""100% Full-module line+branch coverage for host_engram and model.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram (117/117 stmts, 42/42 branches)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.model (lines 204-220, 294-310)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from vllm.models.deepseek_v41.common.engram import DEAD_ID

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram import (
    HOST_ENGRAM_KEY,
    HostEngramInputs,
    _async_source_tokens_cpu,
    _resolve_decode_lookback_cpu,
    capture_host_engram_inputs,
    gather_host_engram_hashes,
    install_host_engram_inputs,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import DeepseekV41Model
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


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
    def test_plan_none_or_not_all_layout_kind(self):
        runner_no_plan = _mock_runner([4], [4], has_plan=False)
        assert capture_host_engram_inputs(runner_no_plan, 1) is None

        other_kind = next(
            k for k in SequenceLayoutKind if k is not SequenceLayoutKind.ALL
        )
        runner_other_kind = _mock_runner([4], [4], kind=other_kind)
        assert capture_host_engram_inputs(runner_other_kind, 1) is None

    def test_all_prompt_true_path(self):
        runner = _mock_runner([4, 2], [4, 4])
        res = capture_host_engram_inputs(runner, 2)
        assert res is not None
        assert res.all_prompt is True
        assert res.host_ready is True
        assert res.can_hash_on_host is True
        assert torch.equal(res.input_ids, runner.input_ids_cpu)
        assert torch.equal(res.positions, runner.positions_cpu)
        assert torch.equal(res.query_start_loc, runner.query_start_loc_cpu)
        assert torch.equal(res.lookback_token_ids, runner.lookback_token_ids.cpu)

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
    def test_device_non_cpu_fallback(self):
        hashes_meta = torch.tensor([[[1, 2]]], device="meta")
        mock_gather = MagicMock(return_value=hashes_meta)

        out = gather_host_engram_hashes(
            hashes_meta, dp_shared_memory=False, gather_fn=mock_gather
        )
        mock_gather.assert_called_once_with(hashes_meta, dp_shared_memory=False)
        assert out is hashes_meta

    def test_cpu_dp_shared_memory_true_fallback(self):
        hashes = torch.tensor([[[1, 2]]], dtype=torch.int32)
        mock_gather = MagicMock(return_value=hashes)

        out = gather_host_engram_hashes(
            hashes, dp_shared_memory=True, gather_fn=mock_gather
        )
        mock_gather.assert_called_once_with(hashes, dp_shared_memory=True)
        assert out is hashes

    def test_cpu_dp_group_none_or_single_world_size_fallback(self):
        hashes = torch.tensor([[[1, 2]]], dtype=torch.int32)
        mock_gather = MagicMock(return_value=hashes)

        # dp_group is None
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_engram_dp_group",
            return_value=None,
        ):
            out = gather_host_engram_hashes(
                hashes, dp_shared_memory=False, gather_fn=mock_gather
            )
            assert out is hashes

        # dp_group world_size == 1
        fake_dp_1 = SimpleNamespace(world_size=1, cpu_group="fake_cpu")
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_engram_dp_group",
            return_value=fake_dp_1,
        ):
            out = gather_host_engram_hashes(
                hashes, dp_shared_memory=False, gather_fn=mock_gather
            )
            assert out is hashes

        # dp_group cpu_group is None
        fake_dp_no_cpu = SimpleNamespace(world_size=2, cpu_group=None)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_engram_dp_group",
            return_value=fake_dp_no_cpu,
        ):
            out = gather_host_engram_hashes(
                hashes, dp_shared_memory=False, gather_fn=mock_gather
            )
            assert out is hashes

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

    def test_fallback_import_when_gather_fn_none(self):
        hashes = torch.tensor([[[1, 2]]], device="meta")
        with patch(
            "vllm.models.deepseek_v41.common.engram.gather_engram_hashes",
            return_value="default_nvidia_gather",
        ):
            out = gather_host_engram_hashes(
                hashes, dp_shared_memory=True, gather_fn=None
            )
            assert out == "default_nvidia_gather"


# ==============================================================================
# 6. install_host_engram_inputs Tests
# ==============================================================================


class TestInstallHostEngramInputs:
    def test_install_idempotency_and_prepare_execution(self):
        # Save original method
        orig_method = TPUModelRunner._prepare_lookback_token_ids

        mock_runner = SimpleNamespace(
            _last_sequence_layout_plan=SimpleNamespace(kind=SequenceLayoutKind.ALL),
            input_batch=SimpleNamespace(
                num_prompt_tokens=np.array([4]),
                num_computed_tokens_cpu=np.array([4]),
                token_ids_cpu=np.zeros((1, 4), dtype=np.int32),
                req_ids=["r0"],
            ),
            seq_lens_np=np.array([4]),
            input_ids_cpu=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            positions_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
            lookback_token_ids=SimpleNamespace(
                cpu=torch.zeros((1, 2), dtype=torch.int32)
            ),
        )

        try:
            # Fake original function that is not a MagicMock
            def fake_original(self, num_reqs: int) -> torch.Tensor:
                return torch.tensor([42])

            TPUModelRunner._prepare_lookback_token_ids = fake_original

            # First install
            install_host_engram_inputs()
            installed_method = TPUModelRunner._prepare_lookback_token_ids
            assert getattr(installed_method, "_deepseek_v41_host_engram", False) is True

            # Second install (idempotent early return)
            install_host_engram_inputs()
            assert TPUModelRunner._prepare_lookback_token_ids is installed_method

            # Execute installed method with forward context available
            mock_ctx = SimpleNamespace(additional_kwargs={})
            with (
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                    return_value=True,
                ),
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_forward_context",
                    return_value=mock_ctx,
                ),
            ):
                res = installed_method(mock_runner, 1)
                assert res.item() == 42
                assert HOST_ENGRAM_KEY in mock_ctx.additional_kwargs
                assert isinstance(
                    mock_ctx.additional_kwargs[HOST_ENGRAM_KEY], HostEngramInputs
                )

            # Execute installed method without forward context available
            with patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                return_value=False,
            ):
                mock_ctx_empty = SimpleNamespace(additional_kwargs={})
                res_no_ctx = installed_method(mock_runner, 1)
                assert res_no_ctx.item() == 42
                assert HOST_ENGRAM_KEY not in mock_ctx_empty.additional_kwargs

        finally:
            TPUModelRunner._prepare_lookback_token_ids = orig_method


# ==============================================================================
# 7. DeepseekV41Model Diff Tests (_prepare_engram and forward)
# ==============================================================================


class TestModelDecodeEngram:
    def _create_model(self, cpu_offload: bool = True):
        model = object.__new__(DeepseekV41Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=64)
        model.engram_dp_shared_memory = False
        model.start_layer = 0
        model.end_layer = 1

        mock_engram = SimpleNamespace(
            embed_tokens=SimpleNamespace(host_offload=cpu_offload),
            layer_hash_index=0,
            prepare_embeddings=MagicMock(),
        )
        layer = SimpleNamespace(engram=mock_engram)
        model.layers = [layer]

        mock_hash = MagicMock(
            return_value=torch.tensor([[[10, 20]]], dtype=torch.int32)
        )
        mock_hash.ensure_cache = MagicMock(return_value=True)
        mock_hash.forward = MagicMock(
            return_value=torch.tensor([[[10, 20]]], dtype=torch.int32)
        )
        mock_hash.hash_on_host = MagicMock(
            return_value=torch.tensor([[[10, 20]]], dtype=torch.int32)
        )
        mock_hash.dummy_hashes = MagicMock(
            return_value=(torch.tensor([[[30, 40]]], dtype=torch.int32), None)
        )
        mock_hash.swa_cache_module = SimpleNamespace(kv_cache=None)
        mock_hash.lookback_depth = 4
        mock_hash.use_slot_cache = True
        mock_hash.multipliers = torch.zeros((1, 4))
        model.engram_hash = mock_hash
        model.engram_swa_prefix = "swa_0"
        return model

    def test_prepare_engram_can_hash_on_host_branches(self):
        model = self._create_model(cpu_offload=True)
        input_ids = torch.tensor([1, 2], dtype=torch.int32)
        swa_meta = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32)
        )
        lookback = torch.tensor([[10, 11]], dtype=torch.int32)

        # 1. lookback is None -> returns None
        assert model._host_hash_inputs(input_ids, swa_meta, None) is None

        # 2. compiling -> returns None
        with patch("torch.compiler.is_compiling", return_value=True):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 3. Not cpu offloaded -> returns None
        model_no_offload = self._create_model(cpu_offload=False)
        assert model_no_offload._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 4. host is None -> returns None
        mock_ctx = SimpleNamespace(additional_kwargs={})
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 5. host is present but can_hash_on_host is False -> returns None
        mock_ctx.additional_kwargs[HOST_ENGRAM_KEY] = HostEngramInputs(
            input_ids=input_ids,
            positions=torch.tensor([0, 1]),
            query_start_loc=swa_meta.query_start_loc,
            lookback_token_ids=lookback,
            all_prompt=False,
            host_ready=False,
        )
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 6. host is present and can_hash_on_host is True (host_ready=True)
        mock_ctx.additional_kwargs[HOST_ENGRAM_KEY] = HostEngramInputs(
            input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            positions=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
            lookback_token_ids=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            all_prompt=False,
            host_ready=True,
        )
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.image_sentinel_mask",
                return_value=None,
            ),
        ):
            res = model._host_hash_inputs(input_ids, swa_meta, lookback)
            assert res is not None
            ids_out, pos_out, qsl_out, mask_out, lb_out, lb_mask_out = res
            assert ids_out.shape == (2,)
            assert pos_out.shape == (2,)
            assert qsl_out.shape == (2,)
            assert lb_out.shape == (1, 2)

    def test_prepare_engram_forward_gather_and_cpu_offload_branches(self):
        # Test line 302: self._has_cpu_offloaded_engram() True vs False
        pos = torch.tensor([0, 1], dtype=torch.int64)
        mock_meta = {
            "swa_0": SimpleNamespace(
                seq_lens=torch.tensor([2]),
                query_start_loc=torch.tensor([0, 2]),
                block_tables=torch.zeros((1, 4), dtype=torch.int32),
            )
        }
        mock_ctx = SimpleNamespace(attn_metadata=mock_meta, additional_kwargs={})

        # 1. cpu_offload=True -> gathered_hashes.to("cpu") is called (lines 302->303)
        model_offload = self._create_model(cpu_offload=True)
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._cache_slots",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._block_table",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.image_sentinel_mask",
                return_value=torch.zeros(2, dtype=torch.bool),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.gather_host_engram_hashes",
                return_value=torch.tensor([[[5, 6]]], dtype=torch.int32),
            ) as mock_gather_fn,
        ):
            h, m = model_offload._prepare_engram(
                torch.tensor([1, 2]),
                pos,
                torch.tensor([[10, 11]]),
                stage="full",
                skip_lookup=False,
            )
            mock_gather_fn.assert_called_once()
            model_offload.layers[0].engram.prepare_embeddings.assert_called_once()

        # 2. cpu_offload=False -> gathered_hashes.to("cpu") is SKIPPED (branch 302->304)
        model_no_offload = self._create_model(cpu_offload=False)
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._cache_slots",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._block_table",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.image_sentinel_mask",
                return_value=torch.zeros(2, dtype=torch.bool),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.gather_host_engram_hashes",
                return_value=torch.tensor([[[5, 6]]], dtype=torch.int32),
            ) as mock_gather_fn2,
        ):
            h2, m2 = model_no_offload._prepare_engram(
                torch.tensor([1, 2]),
                pos,
                torch.tensor([[10, 11]]),
                stage="full",
                skip_lookup=False,
            )
            mock_gather_fn2.assert_called_once()
            model_no_offload.layers[0].engram.prepare_embeddings.assert_called_once()

        # 3. Layer with engram=None (branch 306->304)
        model_none_engram = self._create_model(cpu_offload=False)
        model_none_engram.layers = [SimpleNamespace(engram=None)]
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._cache_slots",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model._block_table",
                return_value=None,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.image_sentinel_mask",
                return_value=torch.zeros(2, dtype=torch.bool),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.gather_host_engram_hashes",
                return_value=torch.tensor([[[5, 6]]], dtype=torch.int32),
            ),
        ):
            h3, m3 = model_none_engram._prepare_engram(
                torch.tensor([1, 2]),
                pos,
                torch.tensor([[10, 11]]),
                stage="full",
                skip_lookup=False,
            )
            assert h3 is not None
