# SPDX-License-Identifier: Apache-2.0
"""100% Diff coverage test suite for host Engram hashing and staging.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram (host_engram.py:1-82)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.engram (engram.py:66-175, 288-445)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.model (model.py:180-213, 261-326, 357-358)
"""

from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy  # noqa: F401
import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import (
    _UNIT_SCALE,
    Engram,
    NgramHashState,
    ParallelEngramEmbedding,
    _gather_engram_rows,
    dequant_fp8_rows,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram import (
    HOST_ENGRAM_KEY,
    HostEngramInputs,
    _async_source_tokens_cpu,
    _resolve_decode_lookback_cpu,
    capture_host_engram_inputs,
    gather_host_engram_hashes,
    install_host_engram_inputs,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import (
    DeepseekV41Model,
)
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

# ==============================================================================
# 1. host_engram.py Tests
# ==============================================================================


def _mock_runner(
    seq_lens: list[int],
    num_prompt: list[int],
    kind: SequenceLayoutKind = SequenceLayoutKind.ALL,
    has_plan: bool = True,
):
    plan = types.SimpleNamespace(kind=kind) if has_plan else None
    return types.SimpleNamespace(
        _last_sequence_layout_plan=plan,
        input_batch=types.SimpleNamespace(num_prompt_tokens=np.array(num_prompt)),
        seq_lens_np=np.array(seq_lens),
        input_ids_cpu=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        positions_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        lookback_token_ids=types.SimpleNamespace(
            cpu=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        ),
    )


class TestHostEngram:
    def test_capture_host_engram_inputs_none_when_no_plan(self):
        runner = _mock_runner([2, 2], [5, 5], has_plan=False)
        assert capture_host_engram_inputs(runner, 2) is None

    def test_capture_host_engram_inputs_none_when_kind_not_all(self):
        other_kind = next(
            k for k in SequenceLayoutKind if k is not SequenceLayoutKind.ALL
        )
        runner = _mock_runner([2, 2], [5, 5], kind=other_kind)
        assert capture_host_engram_inputs(runner, 2) is None

    def test_capture_host_engram_inputs_all_prompt_true_and_false(self):
        # All prompt: seq_lens <= num_prompt_tokens
        runner_prompt = _mock_runner([2, 3], [5, 5])
        captured = capture_host_engram_inputs(runner_prompt, 2)
        assert isinstance(captured, HostEngramInputs)
        assert captured.all_prompt is True
        assert torch.equal(captured.input_ids, runner_prompt.input_ids_cpu)
        assert torch.equal(captured.positions, runner_prompt.positions_cpu)
        assert torch.equal(captured.query_start_loc, runner_prompt.query_start_loc_cpu)
        assert torch.equal(
            captured.lookback_token_ids, runner_prompt.lookback_token_ids.cpu
        )

        # Decoding request: seq_lens > num_prompt_tokens
        runner_decode = _mock_runner([6, 3], [5, 5])
        captured_decode = capture_host_engram_inputs(runner_decode, 2)
        assert isinstance(captured_decode, HostEngramInputs)
        assert captured_decode.all_prompt is False

    def test_install_host_engram_inputs_hook_lifecycle_and_idempotency(self):
        orig_fn = TPUModelRunner._prepare_lookback_token_ids
        try:
            # Set dummy plain function so execution won't invoke real
            # TPUModelRunner internals
            TPUModelRunner._prepare_lookback_token_ids = lambda self, n: torch.ones(
                (n, 2)
            )

            # 1. First installation
            install_host_engram_inputs()
            installed_fn = TPUModelRunner._prepare_lookback_token_ids
            assert getattr(installed_fn, "_deepseek_v41_host_engram", False) is True

            # 2. Second installation is a no-op (idempotency branch lines 69-70)
            install_host_engram_inputs()
            assert TPUModelRunner._prepare_lookback_token_ids is installed_fn

            # 3. Execution without forward context
            runner = _mock_runner([2], [5])
            with patch(
                "vllm.forward_context.is_forward_context_available",
                return_value=False,
            ):
                out = TPUModelRunner._prepare_lookback_token_ids(runner, 1)
                assert out is not None

            # 4. Execution with forward context sets additional_kwargs
            mock_fwd_ctx = SimpleNamespace(additional_kwargs={})
            with patch("vllm.forward_context._forward_context", mock_fwd_ctx):
                out = TPUModelRunner._prepare_lookback_token_ids(runner, 1)
                assert HOST_ENGRAM_KEY in mock_fwd_ctx.additional_kwargs
                assert isinstance(
                    mock_fwd_ctx.additional_kwargs[HOST_ENGRAM_KEY],
                    HostEngramInputs,
                )
        finally:
            TPUModelRunner._prepare_lookback_token_ids = orig_fn

    def test_async_source_tokens_cpu_all_branches(self):
        # 1. Single chunk in chunks_cpu
        pre1 = SimpleNamespace(
            wait_for_copy=lambda: None,
            copy_state=SimpleNamespace(
                chunks_cpu=[torch.tensor([[10, 11]], dtype=torch.int32)]
            ),
        )
        assert _async_source_tokens_cpu(pre1).tolist() == [10, 11]

        # 2. Multiple chunks in chunks_cpu
        pre2 = SimpleNamespace(
            wait_for_copy=lambda: None,
            copy_state=SimpleNamespace(
                chunks_cpu=[
                    torch.tensor([10], dtype=torch.int32),
                    torch.tensor([11], dtype=torch.int32),
                ]
            ),
        )
        assert _async_source_tokens_cpu(pre2).tolist() == [10, 11]

        # 3. No chunks_cpu, wait_for_copy returns Tensor
        pre3 = SimpleNamespace(
            wait_for_copy=lambda: torch.tensor([[30, 31]], dtype=torch.int32),
            copy_state=None,
        )
        assert _async_source_tokens_cpu(pre3).tolist() == [30, 31]

        # 4. No chunks_cpu, wait_for_copy non-callable, next_tokens_tpu is Tensor
        pre4 = SimpleNamespace(
            wait_for_copy=None,
            copy_state=None,
            next_tokens_tpu=torch.tensor([[40, 41]], dtype=torch.int32),
        )
        assert _async_source_tokens_cpu(pre4).tolist() == [40, 41]

        # 5. All None -> returns None
        pre5 = SimpleNamespace(
            wait_for_copy=None, copy_state=None, next_tokens_tpu=None
        )
        assert _async_source_tokens_cpu(pre5) is None

    def test_resolve_decode_lookback_and_capture_decode_branches(self):
        runner = _mock_runner([6, 3], [5, 5])
        batch = SimpleNamespace(
            num_prompt_tokens=np.array([5, 5]),
            num_computed_tokens_cpu=np.array([5, 1]),
            token_ids_cpu=np.array([[10, 11, 12, 13, 14, 15], [20, 21, 22, 0, 0, 0]]),
            req_ids=["r0", "r1"],
        )
        runner.input_batch = batch

        # num_reqs <= 0 early return
        lb0 = _resolve_decode_lookback_cpu(runner, batch, 0, 0)
        assert torch.equal(lb0, runner.lookback_token_ids.cpu)

        # Spec decode active -> host_ready=False
        runner._pre_async_results = SimpleNamespace(
            spec_decode_num_rejected_tokens=torch.tensor([0])
        )
        cap_spec = capture_host_engram_inputs(runner, 2)
        assert cap_spec.can_hash_on_host is False

        # No pre_async -> resolves lookback and host_ready=True
        runner._pre_async_results = None
        cap_ok = capture_host_engram_inputs(runner, 2)
        assert cap_ok.can_hash_on_host is True
        assert cap_ok.lookback_token_ids[0].tolist() == [14, 13]
        assert cap_ok.lookback_token_ids[1].tolist() == [20, -1]

        # pre_async with matched request (q_end > q_start), empty span,
        # and unmatched request
        runner.query_start_loc_cpu = torch.tensor([0, 1, 1], dtype=torch.int32)
        runner._pre_async_results = SimpleNamespace(
            req_id_to_index_copy={"r0": 0, "r1": 1},
            wait_for_copy=None,
            copy_state=None,
            next_tokens_tpu=None,
            spec_decode_num_rejected_tokens=None,
        )
        # Source tokens missing -> host_ready=False
        cap_missing_src = capture_host_engram_inputs(runner, 2)
        assert cap_missing_src.can_hash_on_host is False

        # Source tokens present -> substitutes r0 and skips empty-span r1
        runner._pre_async_results.next_tokens_tpu = torch.tensor(
            [99, 88], dtype=torch.int32
        )
        runner.query_start_loc_cpu = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
        runner.seq_lens_np = np.array([6, 3, 2])
        batch.num_prompt_tokens = np.array([5, 5, 5])
        batch.num_computed_tokens_cpu = np.array([5, 1, 1])
        batch.token_ids_cpu = np.zeros((3, 8), dtype=np.int32)
        batch.req_ids = ["r0", "r1", "unmatched"]
        runner.lookback_token_ids = SimpleNamespace(
            cpu=torch.zeros((3, 2), dtype=torch.int32)
        )
        cap_sub = capture_host_engram_inputs(runner, 3)
        assert cap_sub.can_hash_on_host is True
        assert cap_sub.input_ids[0].item() == 99

        # pre_async with req_id_to_index_copy where no request matches
        runner._pre_async_results.req_id_to_index_copy = {"other": 0}
        cap_no_match = capture_host_engram_inputs(runner, 3)
        assert cap_no_match.can_hash_on_host is True

    def test_gather_host_engram_hashes_all_branches(self):
        hashes = torch.tensor([[[1, 2]]], dtype=torch.int32)  # [1, 1, 2]

        # 1. Fallback to default gather_engram_hashes when dp_shared_memory=True
        with patch(
            "vllm.models.deepseek_v41.common.engram.gather_engram_hashes",
            side_effect=lambda x, **kw: x,
        ) as mock_default_gather:
            out = gather_host_engram_hashes(hashes, dp_shared_memory=True)
            assert torch.equal(out, hashes)
            mock_default_gather.assert_called_once()

        # 2. CPU group all_gather with dp_metadata (padding, exact, and ValueError)
        fake_dp_group = SimpleNamespace(world_size=2, cpu_group="fake_cpu_group")

        def fake_all_gather(out_t, in_t, group=None):
            out_t.copy_(torch.cat([in_t, in_t], dim=0))

        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_engram_dp_group",
                return_value=fake_dp_group,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_forward_context",
                return_value=SimpleNamespace(dp_metadata=object()),
            ),
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
        ):
            # 2a. Padding when shape[0] < slot
            with patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.engram_gathered_num_tokens",
                return_value=3,
            ):
                out_pad = gather_host_engram_hashes(hashes)
                assert out_pad.shape == (6, 1, 2)

            # 2b. Exact slot when shape[0] == slot
            with patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.engram_gathered_num_tokens",
                return_value=1,
            ):
                out_exact = gather_host_engram_hashes(hashes)
                assert out_exact.shape == (2, 1, 2)

            # 2c. ValueError when shape[0] > slot
            with (
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.engram_gathered_num_tokens",
                    return_value=0,
                ),
                pytest.raises(ValueError, match="exceeds the DP token slot"),
            ):
                gather_host_engram_hashes(hashes)

        # 3. CPU group all_gather without forward context (dp_metadata is None)
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_engram_dp_group",
                return_value=fake_dp_group,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                return_value=False,
            ),
            patch(
                "torch.distributed.all_gather_into_tensor",
                side_effect=fake_all_gather,
            ),
        ):
            out_no_ctx = gather_host_engram_hashes(hashes)
            assert out_no_ctx.shape == (2, 1, 2)


# ==============================================================================
# 2. v2/engram.py Tests
# ==============================================================================


def _build_hash_state(use_slot_cache: bool = True, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    state = object.__new__(NgramHashState)
    nn.Module.__init__(state)
    state.token_map = torch.randint(1, 50, (100,), generator=g)
    state.multipliers = torch.randint(1, 1 << 20, (2, 3), generator=g)
    state.primes = torch.randint(1000, 5000, (2, 2, 4), generator=g)
    state.offsets = torch.randint(0, 9999, (2, 8), generator=g)
    state.lookback_depth = 4
    state.pad_id = 0
    state.use_slot_cache = use_slot_cache
    state.block_size = 8
    state._cache = torch.randint(1, 50, (64,), generator=g)
    state.swa_cache_module = SimpleNamespace(
        kv_cache=torch.randint(1, 50, (8, 8), generator=g)
    )
    return state


class TestEngramHashState:
    def test_hash_on_host_empty_tokens(self):
        state = _build_hash_state()
        empty_ids = torch.empty((0,), dtype=torch.int32)
        pos = torch.empty((0,), dtype=torch.int64)
        qsl = torch.tensor([0], dtype=torch.int32)
        dead = torch.empty((0,), dtype=torch.bool)
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)

        res = state.hash_on_host(empty_ids, pos, qsl, dead, lookback, dead_lb)
        assert res.shape == (0, 2, 8)
        assert res.dtype == torch.int32

    def test_hash_on_host_caching_and_execution(self):
        state = _build_hash_state()
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        pos = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        qsl = torch.tensor([0, 4], dtype=torch.int32)
        dead = torch.zeros(4, dtype=torch.bool)
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)

        assert not hasattr(state, "_host_tables")
        out1 = state.hash_on_host(ids, pos, qsl, dead, lookback, dead_lb)
        assert hasattr(state, "_host_tables")
        cached_tables = state._host_tables

        out2 = state.hash_on_host(ids, pos, qsl, dead, lookback, dead_lb)
        assert state._host_tables is cached_tables
        assert torch.equal(out1, out2)

    def test_forward_fallback_without_and_with_slot_cache(self):
        # 1. Without slot cache
        state_no_cache = _build_hash_state(use_slot_cache=False)
        ids = torch.tensor([1, 2], dtype=torch.int32)
        pos = torch.tensor([10, 11], dtype=torch.int64)
        qsl = torch.tensor([0, 2], dtype=torch.int32)
        dead = torch.zeros(2, dtype=torch.bool)
        # Lookback empty (-1) forces fallback call for past tokens
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)
        block_table = torch.zeros((1, 4), dtype=torch.int32)

        out_no_cache = state_no_cache.forward(
            ids, pos, qsl, dead, lookback, dead_lb, None, block_table
        )
        assert out_no_cache.shape == (2, 2, 8)

        # 2. With slot cache
        state_with_cache = _build_hash_state(use_slot_cache=True)
        slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        out_with_cache = state_with_cache.forward(
            ids, pos, qsl, dead, lookback, dead_lb, slot_mapping, block_table
        )
        assert out_with_cache.shape == (2, 2, 8)

    def test_hash_dead_mask_and_window_branches(self):
        state = _build_hash_state()
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        # Shift past > chunk_positions and past < chunk_positions
        pos = torch.tensor([0, 1, 5, 6], dtype=torch.int64)
        qsl = torch.tensor([0, 2, 4], dtype=torch.int32)
        dead = torch.tensor([True, False, False, True], dtype=torch.bool)
        # Valid lookback tokens in window
        lookback = torch.tensor([[10, 11, -1, 13], [20, -1, 22, 23]], dtype=torch.int32)
        dead_lb = torch.tensor(
            [
                [False, True, False, False],
                [True, False, False, False],
            ],
            dtype=torch.bool,
        )
        out = state.hash_on_host(ids, pos, qsl, dead, lookback, dead_lb)
        assert out.shape == (4, 2, 8)


class TestParallelEngramEmbeddingGatherRawAndDequant:
    def test_gather_raw_and_device_transfer(self):
        embed = object.__new__(ParallelEngramEmbedding)
        embed.head_start = 0
        embed.part_n_hash_cols = 2
        embed.n_hash_cols = 4
        embed.vocab_start_idx = 0
        embed.vocab_end_idx = 100
        embed.part_num_embeddings = 100
        embed.dim = 32
        embed.block_size = 16

        weight = torch.ones((100, 32), dtype=torch.uint8)
        scale = torch.full((100, 2), 127, dtype=torch.uint8)
        embed._storage = lambda: (weight, scale)

        # Test with CPU indices (matching weight device)
        indices = torch.tensor([[10, 20, 200, 300]], dtype=torch.int32)
        raw_out = torch.zeros((1, 2, 32), dtype=torch.uint8)
        scale_out = torch.zeros((1, 2, 2), dtype=torch.uint8)

        embed.gather_raw(indices, raw_out, scale_out)
        # First 2 heads are within vocab: values are copied
        assert torch.all(raw_out[0, 0] == 1)
        assert torch.all(raw_out[0, 1] == 1)
        assert torch.all(scale_out == 127)

        # Test invalid indices masked_fill / sentinel scale
        invalid_indices = torch.tensor([[200, 300, 400, 500]], dtype=torch.int32)
        embed.gather_raw(invalid_indices, raw_out, scale_out)
        if hasattr(v2_engram, "_INVALID_SCALE"):
            assert torch.all(scale_out == v2_engram._INVALID_SCALE)
        else:
            assert torch.all(raw_out == 0)
            assert torch.all(scale_out == _UNIT_SCALE)

        # Test indices device transfer branch (line 300)
        mock_indices = MagicMock(
            wraps=torch.tensor([[10, 20, 200, 300]], dtype=torch.int32)
        )
        mock_indices.device = torch.device("meta")
        embed.gather_raw(mock_indices, raw_out, scale_out)
        mock_indices.to.assert_called_once_with(weight.device)

    def test_dequant_fp8_rows_numerical(self):
        # 1.0 in float8_e4m3fn is byte 0x38 (binary 00111000)
        raw_val = torch.tensor([1.0], dtype=torch.float8_e4m3fn).view(torch.uint8)
        raw = raw_val.repeat(4).unsqueeze(0)  # [1, 4]
        # scale byte 127 corresponds to 2^(127-127) = 1.0; 128 corresponds to 2.0
        scale = torch.tensor([[127, 128]], dtype=torch.uint8)  # [1, 2]
        dequant = dequant_fp8_rows(raw, scale, block_size=2)
        assert dequant.dtype == torch.bfloat16
        assert dequant.shape == (1, 4)
        assert torch.isclose(
            dequant[0, :2].float(), torch.tensor([1.0, 1.0]), atol=1e-3
        ).all()
        assert torch.isclose(
            dequant[0, 2:].float(), torch.tensor([2.0, 2.0]), atol=1e-3
        ).all()

        if hasattr(v2_engram, "_INVALID_SCALE"):
            scale_sentinel = torch.tensor(
                [[v2_engram._INVALID_SCALE, 127]], dtype=torch.uint8
            )
            dequant_sentinel = dequant_fp8_rows(raw, scale_sentinel, block_size=2)
            assert torch.all(dequant_sentinel[0, :2] == 0.0)
            assert torch.isclose(
                dequant_sentinel[0, 2:].float(), torch.tensor([1.0, 1.0]), atol=1e-3
            ).all()

        if hasattr(v2_engram, "_unpack_bytes"):
            words = torch.tensor([0x04030201], dtype=torch.int32)
            unpacked = v2_engram._unpack_bytes(words, num_bytes=4)
            assert torch.equal(unpacked, torch.tensor([1, 2, 3, 4], dtype=torch.uint8))

    def test_gather_engram_rows_padding(self):
        # 1. Width already a multiple of 128
        rows_128 = torch.zeros((2, 128), dtype=torch.bfloat16)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.engram.upstream.tensor_model_parallel_all_gather",
            side_effect=lambda x, dim: x,
        ):
            res_128 = _gather_engram_rows(rows_128, dim=0)
            assert res_128.shape == (2, 128)

        # 2. Width not a multiple of 128 (padding branch)
        rows_100 = torch.zeros((2, 100), dtype=torch.bfloat16)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.engram.upstream.tensor_model_parallel_all_gather",
            side_effect=lambda x, dim: x,
        ):
            res_100 = _gather_engram_rows(rows_100, dim=0)
            assert res_100.shape == (2, 100)


class TestEngramModuleStagingAndEmbed:
    def _create_mock_engram(self, cpu_offload: bool):
        engram = object.__new__(Engram)
        embed_tokens = SimpleNamespace(
            cpu_offload=cpu_offload,
            host_offload=cpu_offload,
            dp_size=1,
            part_n_hash_cols=2,
            n_hash_cols=2,
            head_dim=32,
            block_size=16,
            tp_size=1,
            gather_raw=MagicMock(),
        )
        engram.embed_tokens = embed_tokens
        return engram

    def test_dp_staging_uploads_only_local_tokens(self, monkeypatch):
        from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common

        engram = self._create_mock_engram(cpu_offload=True)
        engram.embed_tokens.dp_size = 8
        engram.embed_tokens.n_hash_cols = 16
        engram._init_staging(max_tokens=16, head_dim=32)
        assert engram._host_raw.shape == (16, 16, 32)
        assert not hasattr(engram, "staged_rows")
        group = SimpleNamespace(world_size=8, cpu_group=object())
        monkeypatch.setattr(common.upstream, "get_engram_dp_group", lambda: group)

        def gather(ids, raw, scales):
            assert raw.shape == (ids.shape[0], 2, 32)
            raw.fill_(56)  # E4M3 1.0
            scales.fill_(127)

        def exchange(raw, scales, cpu_group, size):
            assert cpu_group is group.cpu_group and size == 8
            # Exchange correctness is covered by the real Gloo regression.
            tokens = raw.shape[0] // size
            return SimpleNamespace(
                wait=lambda: (
                    raw[:tokens].repeat(1, size, 1),
                    scales[:tokens].repeat(1, size, 1),
                )
            )

        engram.embed_tokens.gather_raw = gather
        monkeypatch.setattr(common, "start_exchange_rows", exchange)
        for tokens in (16, 8, 16):
            engram.prepare_embeddings(torch.zeros(tokens * 8, 16, dtype=torch.int32))
            rows = engram._ready_rows(tokens)
            assert rows.shape == (tokens, 16, 32)
            assert torch.equal(rows, torch.ones_like(rows))
            assert engram._active_raw.numel() * 4 == tokens * 16 * (32 + 2)
        assert len(engram._staged_by_tokens) == 2

    def test_init_staging_cpu_offload_false_and_true(self):
        # cpu_offload = False
        engram_no_offload = self._create_mock_engram(cpu_offload=False)
        with patch.object(
            v2_engram.upstream.Engram, "_init_staging"
        ) as mock_super_init:
            engram_no_offload._init_staging(max_tokens=16, head_dim=32)
            mock_super_init.assert_called_once_with(16, 32)
            assert engram_no_offload._staged_raw is None

        # cpu_offload = True
        engram_offload = self._create_mock_engram(cpu_offload=True)
        with patch.object(
            v2_engram.upstream.Engram, "_init_staging"
        ) as mock_super_init:
            engram_offload._init_staging(max_tokens=16, head_dim=32)
            mock_super_init.assert_not_called()
            assert engram_offload._staged_raw is not None
            assert engram_offload._host_raw.shape == (16, 2, 32)
            assert engram_offload._host_scale.shape == (16, 2, 2)
            raw_words = 16 * 2 * 32 // 4
            scale_words = engram_offload._host_scale_buf.numel() // 4
            assert engram_offload._staged_raw.shape == (raw_words + scale_words,)
            assert engram_offload._staged_raw.dtype == torch.int32
            assert torch.all(engram_offload._staged_raw[:raw_words] == 0)
            assert torch.all(engram_offload._staged_raw[raw_words:] == -1)

    def test_prepare_embeddings_staging_fast_path_and_fallback(self):
        engram = self._create_mock_engram(cpu_offload=True)
        with patch.object(v2_engram.upstream.Engram, "_init_staging"):
            engram._init_staging(max_tokens=16, head_dim=32)

        # 1. Fast path: num_tokens <= max_tokens (and tail reset)
        hash_ids = torch.zeros((8, 2), dtype=torch.int32)
        engram.prepare_embeddings(hash_ids)
        engram.embed_tokens.gather_raw.assert_called_once()
        if hasattr(v2_engram, "_INVALID_SCALE"):
            assert torch.all(engram._host_scale[8:] == v2_engram._INVALID_SCALE)
        else:
            assert torch.all(engram._host_raw[8:] == 0)
            assert torch.all(engram._host_scale[8:] == _UNIT_SCALE)

        # 1b. Fast path with exact capacity (skips tail zeroing branch)
        hash_ids_exact = torch.zeros((16, 2), dtype=torch.int32)
        engram.prepare_embeddings(hash_ids_exact)

        # Oversized host requests must not fall back to a BF16 device lookup.
        hash_ids_large = torch.zeros((32, 2), dtype=torch.int32)
        with pytest.raises(ValueError, match="staging slots"):
            engram.prepare_embeddings(hash_ids_large)

    def test_ready_rows_and_sp_embed(self):
        # 1. _staged_raw is None calls super()._ready_rows
        engram_none = self._create_mock_engram(cpu_offload=False)
        engram_none._staged_raw = None
        with patch.object(
            v2_engram.upstream.Engram,
            "_ready_rows",
            return_value=torch.zeros((4, 2, 32)),
        ) as mock_super_ready:
            res = engram_none._ready_rows(4)
            mock_super_ready.assert_called_once_with(4)
            assert res.shape == (4, 2, 32)

        # 2. _staged_raw is populated calls dequant_fp8_rows
        engram = self._create_mock_engram(cpu_offload=True)
        with patch.object(v2_engram.upstream.Engram, "_init_staging"):
            engram._init_staging(max_tokens=16, head_dim=32)
        engram.prepare_embeddings(torch.zeros((4, 2), dtype=torch.int32))
        ready = engram._ready_rows(4)
        assert ready.shape == (4, 2, 32)
        assert ready.dtype == torch.bfloat16

        # 3. _sp_embed returns ready rows for hash_ids
        with patch.object(
            v2_engram.upstream, "get_tensor_model_parallel_rank", return_value=0
        ):
            # tp_size = 1
            rows = engram._sp_embed(torch.zeros((4, 2), dtype=torch.int32))
            assert rows.shape == (4, 2, 32)

            # tp_size = 2 with padding (num_tokens=3, tp=2 -> chunk=2, pad=1)
            engram.embed_tokens.tp_size = 2
            engram.prepare_embeddings(torch.zeros((3, 2), dtype=torch.int32))
            with patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.common.engram._gather_engram_rows",
                side_effect=lambda x, dim: torch.cat([x, x], dim=dim),
            ):
                rows_tp2 = engram._sp_embed(torch.zeros((3, 2), dtype=torch.int32))
                assert rows_tp2.shape == (2, 2, 32)

                # tp_size = 2 without padding (num_tokens=4, tp=2 -> chunk=2, pad=0)
                engram.prepare_embeddings(torch.zeros((4, 2), dtype=torch.int32))
                rows_tp2_exact = engram._sp_embed(
                    torch.zeros((4, 2), dtype=torch.int32)
                )
                assert rows_tp2_exact.shape == (2, 2, 32)

    def test_create_embedding(self):
        engram = object.__new__(Engram)
        layout = SimpleNamespace(
            num_embeddings=[100],
            head_dim=32,
            primes=[[(3, 5)]],
            layer_ids=[0],
        )
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.common.engram.register_row_slice"
            ) as mock_reg,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.engram.upstream.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.engram.upstream.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch("torch.accelerator.current_device_index", return_value=0),
        ):
            emb = engram._create_embedding(layout, 0)
            assert isinstance(emb, ParallelEngramEmbedding)
            assert mock_reg.call_count == 2


# ==============================================================================
# 3. v2/model.py Tests
# ==============================================================================


class TestModelEngramPreparation:
    def test_prepare_all_host_tables_before_waiting_and_uploading(self, monkeypatch):
        from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common
        from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as model_module

        model = self._create_mock_model(use_slot_cache=False)
        model.end_layer = 2
        model.engram_dp_shared_memory = False
        model.engram_hash.ensure_cache = lambda: False
        issued = []
        group = SimpleNamespace(world_size=2, cpu_group=object())
        monkeypatch.setattr(common.upstream, "get_engram_dp_group", lambda: group)
        monkeypatch.setattr(model_module.upstream, "get_engram_dp_size", lambda: 2)
        monkeypatch.setattr(
            model_module, "gather_host_engram_hashes", lambda hashes, **kw: hashes
        )
        monkeypatch.setattr(
            model_module.upstream, "is_forward_context_available", lambda: True
        )
        monkeypatch.setattr(
            model_module.upstream,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={}),
        )

        def gather(ids, raw, scales):
            raw.fill_(56 + 8 * int(ids[0, 0]))  # FP8 1.0 or 2.0.
            scales.fill_(127)

        def exchange(raw, scales, cpu_group, size):
            issued.append(int(raw[0, 0, 0]))

            def wait():
                assert issued == [56, 64]
                tokens = raw.shape[0] // size
                return (
                    raw[:tokens].repeat(1, size, 1),
                    scales[:tokens].repeat(1, size, 1),
                )

            return SimpleNamespace(wait=wait)

        monkeypatch.setattr(common, "start_exchange_rows", exchange)
        stagers = []
        for index in range(2):
            engram = TestEngramModuleStagingAndEmbed()._create_mock_engram(True)
            engram.layer_hash_index = index
            engram.embed_tokens.dp_size = 2
            engram.embed_tokens.gather_raw = gather
            engram._init_staging(7, 32)
            stagers.append(engram)
        model.layers = [SimpleNamespace(engram=e) for e in stagers]

        for tokens in (7, 2, 7):
            issued.clear()
            hashes = torch.tensor([0, 1]).view(1, 2, 1).expand(tokens * 2, 2, 4)
            model.engram_hash.dummy_hashes = lambda ids, hashes=hashes: (hashes, None)
            model._prepare_engram(torch.arange(tokens), torch.arange(tokens), None)
            for index, engram in enumerate(stagers):
                rows = engram._ready_rows(tokens)
                assert rows.shape == (tokens, 4, 32)
                assert torch.equal(rows, torch.full_like(rows, index + 1))

    def _create_mock_model(
        self,
        cpu_offload: bool = True,
        use_slot_cache: bool = True,
        ced_enabled: bool = False,
    ):
        model = object.__new__(DeepseekV41Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=64)
        model._ced_enabled = ced_enabled
        model._ced_window = 128
        model._ced_compile_sizes = [16, 2048]
        model.use_native_mega_moe = False
        model.start_layer = 0
        model.end_layer = 1
        model.engram_dp_shared_memory = True

        engram_mock = SimpleNamespace(
            embed_tokens=SimpleNamespace(host_offload=cpu_offload),
            layer_hash_index=0,
            prepare_embeddings=MagicMock(),
        )
        layer_mock = SimpleNamespace(engram=engram_mock)
        model.layers = [layer_mock]

        state = _build_hash_state(use_slot_cache=use_slot_cache)
        state.ensure_cache = MagicMock(return_value=True)
        model.engram_hash = state
        model.engram_swa_prefix = "swa_0"
        return model

    def test_has_cpu_offloaded_engram_true_and_false(self):
        model_true = self._create_mock_model(cpu_offload=True)
        assert model_true._has_cpu_offloaded_engram() is True

        model_false = self._create_mock_model(cpu_offload=False)
        assert model_false._has_cpu_offloaded_engram() is False

        # Layer without engram
        model_no_engram = self._create_mock_model(cpu_offload=False)
        model_no_engram.layers = [SimpleNamespace()]
        assert model_no_engram._has_cpu_offloaded_engram() is False

    def test_host_hash_inputs_branches(self):
        model = self._create_mock_model(cpu_offload=True)
        input_ids = torch.tensor([1, 2], dtype=torch.int32)
        swa_meta = SimpleNamespace(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32)
        )
        lookback = torch.tensor([[10, 11]], dtype=torch.int32)

        # 1. lookback_token_ids is None -> returns None
        assert model._host_hash_inputs(input_ids, swa_meta, None) is None

        # 2. torch.compiler.is_compiling() -> returns None
        with patch("torch.compiler.is_compiling", return_value=True):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 3. Not cpu offloaded -> returns None
        with patch.object(model, "_has_cpu_offloaded_engram", return_value=False):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 4. Forward context missing HOST_ENGRAM_KEY -> returns None
        mock_ctx = SimpleNamespace(additional_kwargs={})
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 5. Host inputs present but all_prompt is False -> returns None
        mock_ctx.additional_kwargs[HOST_ENGRAM_KEY] = HostEngramInputs(
            input_ids=input_ids,
            positions=torch.tensor([0, 1]),
            query_start_loc=swa_meta.query_start_loc,
            lookback_token_ids=lookback,
            all_prompt=False,
        )
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            assert model._host_hash_inputs(input_ids, swa_meta, lookback) is None

        # 6. Host inputs present and all_prompt is True -> returns full sliced tuple
        mock_ctx.additional_kwargs[HOST_ENGRAM_KEY] = HostEngramInputs(
            input_ids=input_ids,
            positions=torch.tensor([0, 1]),
            query_start_loc=swa_meta.query_start_loc,
            lookback_token_ids=lookback,
            all_prompt=True,
        )
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
            return_value=mock_ctx,
        ):
            res = model._host_hash_inputs(input_ids, swa_meta, lookback)
            assert res is not None
            assert len(res) == 6
            assert torch.equal(res[0], input_ids)

    def test_prepare_engram_early_exits_and_skip_lookup(self):
        model = self._create_mock_model(cpu_offload=True)
        pos = torch.tensor([0, 1], dtype=torch.int64)

        # 1. stage == "decoder" -> early returns (None, None)
        assert model._prepare_engram(
            torch.tensor([1, 2]), pos, None, stage="decoder"
        ) == (None, None)

        # 2. input_ids is None -> early returns (None, None)
        assert model._prepare_engram(None, pos, None, stage="full") == (None, None)

        # 3. is_forward_context_available is False -> early returns (None, None)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
            return_value=False,
        ):
            assert model._prepare_engram(
                torch.tensor([1, 2]), pos, None, stage="full"
            ) == (None, None)

        # 4. skip_lookup=True expands input_ids
        mock_meta = {
            "swa_0": SimpleNamespace(
                seq_lens=torch.tensor([2]),
                query_start_loc=torch.tensor([0, 2]),
            )
        }
        mock_ctx = SimpleNamespace(attn_metadata=mock_meta, additional_kwargs={})
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
        ):
            hashes, mask = model._prepare_engram(
                torch.tensor([1, 2]), pos, None, stage="full", skip_lookup=True
            )
            assert hashes.shape == (2, 2, 1)
            assert mask.shape == (2,)

        # 5. DP dummy hashes branch (lines 281-284)
        model.engram_dp_shared_memory = False
        model.engram_hash.ensure_cache = MagicMock(return_value=False)
        model.engram_hash.dummy_hashes = MagicMock(
            return_value=(torch.zeros((2, 2, 8)), None)
        )
        mock_ctx_dp = SimpleNamespace(attn_metadata=mock_meta, additional_kwargs={})
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx_dp,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_engram_dp_size",
                return_value=2,
            ),
        ):
            h, m = model._prepare_engram(
                torch.tensor([1, 2]), pos, None, stage="encoder"
            )
            model.engram_hash.dummy_hashes.assert_called_once()

            # DP dummy hashes skipped when engram_dp_shared_memory is True
            # (line 281->285)
            model.engram_dp_shared_memory = True
            model.engram_hash.dummy_hashes.reset_mock()
            h_skip, m_skip = model._prepare_engram(
                torch.tensor([1, 2]), pos, None, stage="encoder"
            )
            model.engram_hash.dummy_hashes.assert_not_called()

    def test_prepare_engram_missing_lookback_not_implemented(self):
        # 1. When use_slot_cache is False and lookback_token_ids is None
        # -> raises NotImplementedError
        model = self._create_mock_model(cpu_offload=True, use_slot_cache=False)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        mock_meta = {
            "swa_0": SimpleNamespace(
                seq_lens=torch.tensor([2]),
                query_start_loc=torch.tensor([0, 2]),
                block_tables=torch.zeros((1, 4), dtype=torch.int32),
            )
        }
        mock_ctx = SimpleNamespace(attn_metadata=mock_meta, additional_kwargs={})
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            pytest.raises(NotImplementedError, match="lookback_token_ids"),
        ):
            model._prepare_engram(
                torch.tensor([1, 2]), pos, None, stage="full", skip_lookup=False
            )

        # 2. When use_slot_cache is True and lookback_token_ids is None
        # -> allocates lookback_token_ids (lines 257-260)
        model_cache = self._create_mock_model(cpu_offload=True, use_slot_cache=True)
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch.object(
                model_cache.engram_hash,
                "forward",
                wraps=model_cache.engram_hash.forward,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                side_effect=lambda h, **kwargs: h.unsqueeze(1),
            ),
        ):
            hashes, mask = model_cache._prepare_engram(
                torch.tensor([1, 2], dtype=torch.int32),
                pos,
                None,
                stage="full",
                skip_lookup=False,
            )
            assert hashes is not None

    def test_prepare_engram_host_and_device_dispatch(self):
        model = self._create_mock_model(cpu_offload=True, use_slot_cache=True)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        ids = torch.tensor([1, 2], dtype=torch.int32)
        lookback = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
        mock_meta = {
            "swa_0": SimpleNamespace(
                seq_lens=torch.tensor([2]),
                query_start_loc=torch.tensor([0, 2]),
                block_tables=torch.zeros((1, 4), dtype=torch.int32),
            )
        }

        # 1. Host hash dispatch (host is not None)
        host_inputs = HostEngramInputs(
            input_ids=ids,
            positions=pos,
            query_start_loc=mock_meta["swa_0"].query_start_loc,
            lookback_token_ids=lookback,
            all_prompt=True,
        )
        mock_ctx = SimpleNamespace(
            attn_metadata=mock_meta,
            additional_kwargs={HOST_ENGRAM_KEY: host_inputs},
        )
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch.object(
                model.engram_hash, "hash_on_host", wraps=model.engram_hash.hash_on_host
            ) as mock_host_hash,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                side_effect=lambda h, **kwargs: h.unsqueeze(1),
            ),
        ):
            hashes, mask = model._prepare_engram(ids, pos, lookback, stage="full")
            mock_host_hash.assert_called_once()
            assert hashes.shape == (2, 2, 8)
            model.layers[0].engram.prepare_embeddings.assert_called_once()

        # 2. Device hash dispatch (host is None, e.g. decode step)
        host_inputs_decode = HostEngramInputs(
            input_ids=ids,
            positions=pos,
            query_start_loc=mock_meta["swa_0"].query_start_loc,
            lookback_token_ids=lookback,
            all_prompt=False,
        )
        mock_ctx_decode = SimpleNamespace(
            attn_metadata=mock_meta,
            additional_kwargs={HOST_ENGRAM_KEY: host_inputs_decode},
        )
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx_decode,
            ),
            patch.object(
                model.engram_hash, "forward", wraps=model.engram_hash.forward
            ) as mock_dev_hash,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                side_effect=lambda h, **kwargs: h.unsqueeze(1),
            ),
        ):
            hashes, mask = model._prepare_engram(ids, pos, lookback, stage="full")
            mock_dev_hash.assert_called_once()
            assert hashes.shape == (2, 2, 8)

    def test_forward_offload_engram_lifecycle_and_mega_moe(self):
        model = self._create_mock_model(cpu_offload=True, ced_enabled=False)
        ids = torch.tensor([1, 2], dtype=torch.int32)
        pos = torch.tensor([0, 1], dtype=torch.int64)

        model._prepare_engram = MagicMock()
        model.full_stage = MagicMock(return_value=(torch.zeros((2, 64)), None))

        # use_native_mega_moe=True converts input_ids to int64
        model.use_native_mega_moe = True
        model.forward(ids, pos)
        model._prepare_engram.assert_called_once()
        call_ids = model._prepare_engram.call_args[0][0]
        assert call_ids.dtype == torch.int64
        assert model._engram_prepared_outside_stage is False


class TestDeepseekV41ForCausalLMInit:
    def test_init_engram_offload_hook_installed(self, monkeypatch: pytest.MonkeyPatch):
        def fake_super_init(self, *args, **kwargs):
            self.config = SimpleNamespace(expert_dtype="fp4")

        monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", "1")
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.install_host_engram_inputs"
            ) as mock_install,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.DeepseekV41LLMForCausalLM.__init__",
                fake_super_init,
            ),
        ):
            from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import (
                DeepseekV41ForCausalLM,
            )

            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                ),
                speculative_config=None,
                scheduler_config=SimpleNamespace(async_scheduling=False),
            )
            DeepseekV41ForCausalLM(vllm_config=vllm_config)
            mock_install.assert_called_once()

    def test_init_engram_offload_hook_not_installed_when_false(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def fake_super_init(self, *args, **kwargs):
            self.config = SimpleNamespace(expert_dtype="fp4")

        monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", "0")
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.install_host_engram_inputs"
            ) as mock_install,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.DeepseekV41LLMForCausalLM.__init__",
                fake_super_init,
            ),
        ):
            from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import (
                DeepseekV41ForCausalLM,
            )

            vllm_config = SimpleNamespace(
                parallel_config=SimpleNamespace(
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                ),
                speculative_config=None,
                scheduler_config=SimpleNamespace(async_scheduling=False),
            )
            DeepseekV41ForCausalLM(vllm_config=vllm_config)
            mock_install.assert_not_called()
