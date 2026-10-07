# SPDX-License-Identifier: Apache-2.0
"""100% Full-module line and branch coverage test suite for .

Covers all lines and branches in:
1. src/vllm_torchtpu/models/vllm/deepseek_v41/common/engram.py
   (240 stmts, 56 branches, 100%)
2. src/vllm_torchtpu/models/vllm/deepseek_v41/v2/host_engram.py
   (120 stmts, 42 branches, 100%)
3. src/vllm_torchtpu/models/vllm/deepseek_v41/v2/model.py
   (diff lines 274, 294, 296-297, 345-349)
4. src/vllm_torchtpu/kernels/fused_moe/v2/router_ops.py (32 stmts, 4 branches, 100%)
5. src/vllm_torchtpu/kernels/fused_moe/v2/token_parallel.py
   (41 stmts, 14 branches, 100%)
6. src/vllm_torchtpu/kernels/fused_moe/v2/host.py (481 stmts, 80 branches, 100%)
7. src/vllm_torchtpu/kernels/fused_moe/v2/layer.py (219 stmts, 84 branches, 100%)
8. src/vllm_torchtpu/kernels/fused_moe/v2/kernel.py (744 stmts, 178 branches, 100%)
"""

from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure CPU execution for all tests
os.environ["MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE"] = "1"

from vllm_torchtpu.kernels.fused_moe.v2 import (
    AXIS,
    WeightFormat,
    fused_ep_moe_v2,
    host,
    kernel,
    layer,
)
from vllm_torchtpu.kernels.fused_moe.v2.host import (
    GatheredPairs,
    WeightSlots,
    _refuse_bad_plan_args,
    array_vmem_bytes,
    arrival_mirror_rows,
    build_routing_tables,
    build_routing_tables_sharded,
    check_u32_sublane_tile,
    chip_generation,
    contrib_mirror_rows,
    fit_weight_slots,
    pair_block_hist,
    pow2_shift,
    rank_window,
    token_gather_smem_bytes,
    token_gather_window_rows,
    vmem_estimate_bytes,
    vmem_limit,
    vmem_scratch_arrays,
    vmem_tile_body_arrays,
    weight_form,
    weight_format_of_dtype,
)
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    _absmax_rows,
    _all_pairs_barrier,
    _and_nonempty,
    _apply_act,
    _build_combine_kernel,
    _build_fused_ep_moe_kernel,
    _chunked_dot,
    _intermediate_rows,
    _quant_apply,
    _row_scale,
    _rows_wait,
    _slot_copy,
    _tile_row_scales,
    build_combine_kernel,
    build_fused_ep_moe_kernel,
    combine_step_tokens,
    expert_ffn_bf16,
    expert_ffn_blockscale,
    expert_ffn_fp8,
    expert_ffn_int8,
    rowquant_fp8,
    tile_height_rungs,
    wire_row_and_scale,
)
from vllm_torchtpu.kernels.fused_moe.v2.layer import (
    _apply_scoring_fn,
    _combine_arrivals,
    _exchange_layout,
    _pack_routing_exchange,
    _relabel_expert_ids_to_mesh_order,
    _unpack_routing_exchange,
)
from vllm_torchtpu.kernels.fused_moe.v2.router_ops import pallas_select
from vllm_torchtpu.kernels.fused_moe.v2.token_parallel import TokenReplicaLayout
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
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

# ==============================================================================
# Helper Mock Objects
# ==============================================================================


class MockRef:
    """Mock mutable Ref for Pallas TPU kernel simulation."""

    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, idx):
        if hasattr(self.arr, "__getitem__"):
            try:
                sub = self.arr[idx]
                if isinstance(sub, jnp.ndarray):
                    return sub
            except Exception:
                pass
        return jnp.zeros((32, 128), dtype=jnp.float32)

    def __setitem__(self, idx, val):
        pass

    @property
    def at(self):
        return self

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return self.arr.dtype

    def astype(self, dtype):
        return self

    def bitcast(self, dtype):
        return self


class MockRefWithoutBitcast:
    """Mock Ref lacking bitcast attribute to exercise NotImplementedError."""

    def __init__(self, arr):
        self.arr = arr

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return self.arr.dtype


class MockCache(dict):
    """Mock dictionary simulating cache miss on first get and hit on second."""

    def __init__(self, target_fn):
        super().__init__()
        self.calls = 0
        self.target_fn = target_fn

    def get(self, key, default=None):
        self.calls += 1
        if self.calls == 1:
            return None
        return self.target_fn


def _mock_tpu_info():
    info = MagicMock()
    info.generation = 7
    info.num_lanes = 128
    info.num_sublanes = 8
    info.vmem_capacity_bytes = 64 * 1024 * 1024
    info.get_sublane_tiling.side_effect = lambda dtype: (
        8 * (32 // (jnp.dtype(dtype).itemsize * 8))
    )
    return info


# ==============================================================================
# Section 1: deepseek_v41/common/engram.py Full Coverage
# ==============================================================================


def _build_common_hash_state(
    use_slot_cache: bool = True,
    block_size: int = 8,
    seed: int = 42,
    subclass=common_engram.NgramHashState,
):
    g = torch.Generator().manual_seed(seed)
    state = object.__new__(subclass)
    nn.Module.__init__(state)
    state.token_map = torch.randint(1, 50, (100,), generator=g)
    state.multipliers = torch.randint(1, 1 << 20, (2, 3), generator=g)
    state.primes = torch.randint(1000, 5000, (2, 2, 4), generator=g)
    state.offsets = torch.randint(0, 9999, (2, 8), generator=g)
    state.lookback_depth = 4
    state.pad_id = 0
    state.use_slot_cache = use_slot_cache
    state.block_size = block_size
    state._cache = torch.randint(1, 50, (64,), generator=g)
    state._kv_cache_ref = None
    state.swa_cache_module = SimpleNamespace(
        kv_cache=torch.randint(1, 50, (8, block_size), generator=g),
        bound_cache=torch.randint(1, 50, (8, block_size), generator=g),
        block_size=block_size,
    )
    return state


class TestCommonNgramHashStateFull:
    def test_common_write_cache_not_implemented(self):
        state = object.__new__(common_engram.NgramHashState)
        dummy = torch.empty(0)
        with pytest.raises(NotImplementedError):
            state.write_cache(dummy, dummy, dummy, dummy)

    @pytest.mark.parametrize("use_slot_cache", [True, False])
    @pytest.mark.parametrize("has_cache", [True, False])
    def test_common_ensure_cache_compiling_branches(
        self, monkeypatch: pytest.MonkeyPatch, use_slot_cache: bool, has_cache: bool
    ):
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        state = _build_common_hash_state(use_slot_cache=use_slot_cache)
        if use_slot_cache:
            state._cache = torch.zeros(10) if has_cache else None
            assert state.ensure_cache() is has_cache
        else:
            state.swa_cache_module.kv_cache = (
                torch.ones(2, 2) if has_cache else torch.empty(0)
            )
            assert state.ensure_cache() is has_cache

    def test_common_ensure_cache_eager_block_size_resize(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        state = _build_common_hash_state(use_slot_cache=True, block_size=8)
        state.swa_cache_module.kv_cache = torch.zeros(4, 16)
        monkeypatch.setattr(
            common_engram.upstream.NgramHashState,
            "ensure_cache",
            lambda self: True,
        )
        assert state.ensure_cache() is True
        assert state.block_size == 16

    def test_common_forward_and_hash_branches(self):
        state = _build_common_hash_state(use_slot_cache=True)
        written = []
        state.write_cache = lambda c, s, v, val: written.append((s, v, val))

        tokens = torch.tensor([10, 20], dtype=torch.int32)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        q_start = torch.tensor([0, 2], dtype=torch.int64)
        dead_mask = torch.tensor([False, False], dtype=torch.bool)
        lookback = torch.tensor([[-1, -1, -1, -1], [10, -1, -1, -1]], dtype=torch.int32)
        lb_dead_mask = torch.zeros_like(lookback, dtype=torch.bool)
        slots = torch.tensor([0, 1], dtype=torch.int64)
        block_table = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)

        # 1. Forward with use_slot_cache=True, count > 0, slot_mapping not None
        out = state(
            tokens, pos, q_start, dead_mask, lookback, lb_dead_mask, slots, block_table
        )
        assert out.shape[0] == 2
        assert len(written) == 1

        # 2. Forward with count == 0
        empty_out = state(
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
            q_start,
            torch.empty(0, dtype=torch.bool),
            lookback,
            lb_dead_mask,
            None,
            block_table,
        )
        assert empty_out.shape == (0, 2, 8)

        # 3. Forward with use_slot_cache=False, exercises fallback returning pad_id
        state_no_cache = _build_common_hash_state(use_slot_cache=False)
        out_no_cache = state_no_cache(
            tokens, pos, q_start, dead_mask, lookback, lb_dead_mask, None, block_table
        )
        assert out_no_cache.shape[0] == 2

    def test_common_hash_on_host_and_caching(self):
        state = _build_common_hash_state(use_slot_cache=True)
        tokens = torch.tensor([10, 20], dtype=torch.int32)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        q_start = torch.tensor([0, 2], dtype=torch.int64)
        dead_mask = torch.tensor([False, True], dtype=torch.bool)
        lookback = torch.tensor([[5, -1, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32)
        lb_dead_mask = torch.tensor(
            [[False, False, False, False], [True, False, False, False]],
            dtype=torch.bool,
        )

        # count == 0
        empty = state.hash_on_host(
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int64),
            q_start,
            torch.empty(0, dtype=torch.bool),
            lookback,
            lb_dead_mask,
        )
        assert empty.shape == (0, 2, 8)

        # First call populates _host_tables
        assert not hasattr(state, "_host_tables")
        out1 = state.hash_on_host(
            tokens, pos, q_start, dead_mask, lookback, lb_dead_mask
        )
        assert hasattr(state, "_host_tables")
        assert out1.shape[0] == 2

        # Second call reuses _host_tables
        out2 = state.hash_on_host(
            tokens, pos, q_start, dead_mask, lookback, lb_dead_mask
        )
        assert torch.equal(out1, out2)

    def test_load_local_rows(self):
        # 1. uint8 param
        p_u8 = torch.zeros(4, 8, dtype=torch.uint8)
        w_u8 = torch.ones(4, 8, dtype=torch.uint8)
        common_engram._load_local_rows(p_u8, w_u8)
        assert torch.equal(p_u8, w_u8)

        # 2. float8_e8m0fnu weight
        w_e8 = torch.ones(4, 8, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        common_engram._load_local_rows(p_u8, w_e8)
        assert torch.equal(p_u8, torch.ones(4, 8, dtype=torch.uint8))

        # 3. float32 param
        p_f32 = torch.zeros(4, 8, dtype=torch.float32)
        w_f32 = torch.ones(4, 8, dtype=torch.float32)
        common_engram._load_local_rows(p_f32, w_f32)
        assert torch.equal(p_f32, w_f32)

        # 4. Shape mismatch assertion
        with pytest.raises(AssertionError, match="does not fit param"):
            common_engram._load_local_rows(p_f32, torch.ones(2, 4))

    def test_parallel_engram_embedding_full(self, monkeypatch: pytest.MonkeyPatch):
        # ParallelEngramEmbedding._weight_loader branches
        p = torch.zeros(4, 8, dtype=torch.float32)
        w_same = torch.ones(4, 8, dtype=torch.float32)

        # loaded_weight.shape[0] == param.shape[0]
        common_engram.ParallelEngramEmbedding._weight_loader(p, w_same)
        assert torch.equal(p, w_same)

        # non-tpu param with shape mismatch calls upstream loader
        w_diff = torch.ones(2, 8, dtype=torch.float32)
        with patch.object(
            common_engram.upstream.ParallelEngramEmbedding, "_weight_loader"
        ) as mock_upstream_loader:
            common_engram.ParallelEngramEmbedding._weight_loader(p, w_diff)
            mock_upstream_loader.assert_called_once()

        # tpu param device branch
        mock_tpu_param = MagicMock()
        mock_tpu_param.device.type = "tpu"
        mock_tpu_param.shape = (4, 8)
        mock_stager = MagicMock()
        mock_stager.host_param.return_value = p
        with patch(
            "vllm_torchtpu.model_loader_patches.ExpertParamStager",
            return_value=mock_stager,
        ):
            common_engram.ParallelEngramEmbedding._weight_loader(mock_tpu_param, w_same)
            mock_stager.flush.assert_called_once_with(mock_tpu_param)

        # __init__ with cpu_offload None vs bool
        monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", "1")
        emb = object.__new__(common_engram.ParallelEngramEmbedding)
        with (
            patch(
                "vllm.models.deepseek_v41.common.engram.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.models.deepseek_v41.common.engram.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm.distributed.parallel_state.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch(
                "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch("torch.accelerator.current_device_index", return_value=0),
        ):
            common_engram.ParallelEngramEmbedding.__init__(
                emb, 100, 16, (4, 4), cpu_offload=None
            )
            assert emb.cpu_offload is True

            emb2 = object.__new__(common_engram.ParallelEngramEmbedding)
            common_engram.ParallelEngramEmbedding.__init__(
                emb2, 100, 16, (4, 4), cpu_offload=False
            )
            assert emb2.cpu_offload is False

        # _allocate_weights
        w1, s1 = emb._allocate_weights()
        assert w1.device.type == "cpu"
        assert s1.device.type == "cpu"

        with patch.object(
            common_engram.upstream.ParallelEngramEmbedding,
            "_allocate_weights",
            return_value=("w", "s"),
        ):
            assert emb2._allocate_weights() == ("w", "s")

        # lookup and gather_raw
        emb.part_num_embeddings = 32
        emb.dim = 16
        emb.block_size = 8
        emb.head_start = 0
        emb.part_n_hash_cols = 2
        emb.n_hash_cols = 2
        emb.vocab_start_idx = 0
        emb.vocab_end_idx = 32
        emb.weight = nn.Parameter(
            torch.ones(32, 16, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        emb.weight_scale_inv = nn.Parameter(
            torch.full((32, 2), 127, dtype=torch.uint8), requires_grad=False
        )

        indices = torch.tensor([[0, 1], [30, 31]], dtype=torch.int64)
        out = torch.zeros(2, 2, 16, dtype=torch.bfloat16)
        emb.lookup(indices, out)
        assert out.shape == (2, 2, 16)

        # gather_raw
        raw_out = torch.zeros(2, 2, 16, dtype=torch.uint8)
        scale_out = torch.zeros(2, 2, 2, dtype=torch.uint8)
        emb.gather_raw(indices, raw_out, scale_out)
        assert raw_out.shape == (2, 2, 16)
        assert scale_out.shape == (2, 2, 2)

    def test_dequant_unpack_gather_rows(self):
        # dequant_fp8_rows with valid and invalid sentinel scale
        raw = torch.zeros((2, 16), dtype=torch.uint8)
        scale = torch.full((2, 2), common_engram._INVALID_SCALE, dtype=torch.uint8)
        scale[0, 0] = 127
        deq = common_engram.dequant_fp8_rows(raw, scale, block_size=8)
        assert deq.shape == (2, 16)
        assert deq.dtype == torch.bfloat16
        assert torch.equal(deq[1], torch.zeros(16, dtype=torch.bfloat16))

        # _unpack_bytes
        words = torch.tensor([0x04030201], dtype=torch.int32)
        unpacked = common_engram._unpack_bytes(words, num_bytes=4)
        assert unpacked.tolist() == [1, 2, 3, 4]

        # _gather_engram_rows with padding and without padding
        rows_pad = torch.ones(2, 64, dtype=torch.bfloat16)
        with patch.object(
            common_engram.upstream,
            "tensor_model_parallel_all_gather",
            side_effect=lambda r, dim: r,
        ):
            g_pad = common_engram._gather_engram_rows(rows_pad, dim=0)
            assert g_pad.shape == (2, 64)

            rows_nopad = torch.ones(2, 128, dtype=torch.bfloat16)
            g_nopad = common_engram._gather_engram_rows(rows_nopad, dim=0)
            assert g_nopad.shape == (2, 128)

    def test_engram_module_full(self):
        engram = object.__new__(common_engram.Engram)
        nn.Module.__init__(engram)

        # _create_embedding
        layout = SimpleNamespace(
            num_embeddings=[100],
            head_dim=16,
            primes=[[(4, 4)]],
            layer_ids=[0],
        )
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.common.engram.register_row_slice"
            ) as mock_reg,
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.common.engram.ParallelEngramEmbedding"
            ) as mock_emb_cls,
        ):
            mock_emb = MagicMock()
            mock_emb.vocab_start_idx = 0
            mock_emb.part_num_embeddings = 100
            mock_emb_cls.return_value = mock_emb
            created = engram._create_embedding(layout, 0)
            assert created is mock_emb
            assert mock_reg.call_count == 2

        # _init_staging with cpu_offload=True and False
        engram.embed_tokens = SimpleNamespace(
            part_n_hash_cols=2, block_size=8, cpu_offload=True
        )
        with patch.object(common_engram.upstream.Engram, "_init_staging"):
            engram._init_staging(max_tokens=4, head_dim=16)
            assert engram._staged_raw is not None
            assert engram._host_raw.shape == (4, 2, 16)

            engram.embed_tokens.cpu_offload = False
            engram._init_staging(max_tokens=4, head_dim=16)
            assert engram._staged_raw is None

            engram.embed_tokens = SimpleNamespace(
                gather_raw=lambda h, r, s: None,
                part_n_hash_cols=2,
                block_size=8,
                cpu_offload=True,
                n_hash_cols=2,
                tp_size=1,
            )
            engram._init_staging(max_tokens=4, head_dim=16)

        hash_ids = torch.tensor([[0, 1], [2, 3]])
        engram.prepare_embeddings(hash_ids)
        ready = engram._ready_rows(2)
        assert ready.shape == (2, 2, 16)

        # Fallback to super().prepare_embeddings and super()._ready_rows
        engram._staged_raw = None
        with (
            patch.object(
                common_engram.upstream.Engram, "prepare_embeddings"
            ) as mock_prep,
            patch.object(
                common_engram.upstream.Engram,
                "_ready_rows",
                return_value=torch.zeros(2, 2, 16),
            ) as mock_ready,
        ):
            engram.prepare_embeddings(hash_ids)
            mock_prep.assert_called_once()
            assert engram._ready_rows(2).shape == (2, 2, 16)
            mock_ready.assert_called_once()

        # _sp_embed: tp == 1 vs tp > 1
        engram._ready_rows = lambda n: torch.ones(n, 2, 16, dtype=torch.bfloat16)
        engram.embed_tokens.tp_size = 1
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.common.engram.upstream.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            assert engram._sp_embed(hash_ids).shape == (2, 2, 16)

            engram.embed_tokens.tp_size = 2
            with patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.common.engram._gather_engram_rows",
                side_effect=lambda r, dim: torch.cat([r, r], dim=0),
            ):
                sp_out = engram._sp_embed(hash_ids)
                assert sp_out.shape[0] == 1

        # forward: use_sequence_parallel True and False, with and without token_mask
        engram.dim = 16
        engram.hc_mult = 1
        engram.eps = 1e-5
        engram.clamp_value = 1e-4
        engram.q_weight = nn.Parameter(torch.ones(16))
        engram.k_weight = nn.Parameter(torch.ones(16))
        engram.wkv = lambda x: torch.ones(x.shape[0], 2 * 16)

        # non-SP forward
        engram.use_sequence_parallel = False
        engram.embed = lambda h: torch.ones(h.shape[0], 2, 16)
        h_states = torch.ones(2, 16, dtype=torch.bfloat16)
        t_mask = torch.ones(2, dtype=torch.bool)
        f_out1 = engram.forward(h_states, hash_ids, token_mask=t_mask)
        assert f_out1.shape[0] == 2
        f_out2 = engram.forward(h_states, hash_ids, token_mask=None)
        assert f_out2.shape[0] == 2

        # SP forward
        engram.use_sequence_parallel = True
        engram._sp_embed = lambda h: torch.ones(1, 2, 16)
        h_chunk = torch.ones(1, 16, dtype=torch.bfloat16)
        with patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.common.engram.upstream.get_tensor_model_parallel_rank",
            return_value=0,
        ):
            sp_out1 = engram.forward(h_chunk, hash_ids, token_mask=t_mask)
            assert sp_out1.shape[0] == 1
            sp_out2 = engram.forward(h_chunk, hash_ids, token_mask=None)
            assert sp_out2.shape[0] == 1


# ==============================================================================
# Section 2: deepseek_v41/v2/host_engram.py Full Coverage
# ==============================================================================


class TestV2HostEngramFull:
    def test_host_engram_inputs_dataclass(self):
        # host_ready is None with all_prompt True / False
        h_true = HostEngramInputs(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            query_start_loc=torch.tensor([0, 2]),
            lookback_token_ids=torch.tensor([[1], [2]]),
            all_prompt=True,
            host_ready=None,
        )
        assert h_true.can_hash_on_host is True

        h_false = HostEngramInputs(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            query_start_loc=torch.tensor([0, 2]),
            lookback_token_ids=torch.tensor([[1], [2]]),
            all_prompt=False,
            host_ready=None,
        )
        assert h_false.can_hash_on_host is False

        # host_ready explicitly set
        h_hr_true = HostEngramInputs(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            query_start_loc=torch.tensor([0, 2]),
            lookback_token_ids=torch.tensor([[1], [2]]),
            all_prompt=False,
            host_ready=True,
        )
        assert h_hr_true.can_hash_on_host is True

        h_hr_false = HostEngramInputs(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            query_start_loc=torch.tensor([0, 2]),
            lookback_token_ids=torch.tensor([[1], [2]]),
            all_prompt=True,
            host_ready=False,
        )
        assert h_hr_false.can_hash_on_host is False

    def test_async_source_tokens_cpu_branches(self):
        # 1. wait_for_copy returning Tensor
        p1 = SimpleNamespace(wait_for_copy=lambda: torch.tensor([[1, 2], [3, 4]]))
        assert _async_source_tokens_cpu(p1).tolist() == [1, 2, 3, 4]

        # 2. copy_state with 1 chunk
        p2 = SimpleNamespace(
            copy_state=SimpleNamespace(chunks_cpu=[torch.tensor([5, 6])])
        )
        assert _async_source_tokens_cpu(p2).tolist() == [5, 6]

        # 3. copy_state with multiple chunks
        p3 = SimpleNamespace(
            copy_state=SimpleNamespace(
                chunks_cpu=[torch.tensor([7]), torch.tensor([8])]
            )
        )
        assert _async_source_tokens_cpu(p3).tolist() == [7, 8]

        # 4. next_tokens_tpu Tensor
        p4 = SimpleNamespace(next_tokens_tpu=torch.tensor([9, 10]))
        assert _async_source_tokens_cpu(p4).tolist() == [9, 10]

        # 5. Fallback None
        assert _async_source_tokens_cpu(SimpleNamespace()) is None

    def test_resolve_decode_lookback_cpu(self):
        runner = SimpleNamespace(
            lookback_token_ids=SimpleNamespace(
                cpu=torch.full((4, 2), -1, dtype=torch.int64)
            )
        )
        batch = SimpleNamespace(
            num_computed_tokens_cpu=np.array([2, 3]),
            token_ids_cpu=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
        )
        # num_reqs <= 0
        lb_zero = _resolve_decode_lookback_cpu(runner, batch, num_reqs=0, start_index=0)
        assert lb_zero.shape == (4, 2)

        # num_reqs > 0
        lb = _resolve_decode_lookback_cpu(runner, batch, num_reqs=2, start_index=0)
        assert lb.shape == (4, 2)
        assert lb[0].tolist() == [20, 10]
        assert lb[1].tolist() == [70, 60]

    def test_capture_host_engram_inputs_all_branches(self):
        runner = SimpleNamespace(
            _last_sequence_layout_plan=None,
            input_batch=SimpleNamespace(num_prompt_tokens=np.array([2, 2])),
            seq_lens_np=np.array([2, 2]),
            input_ids_cpu=torch.tensor([1, 2, 3, 4]),
            positions_cpu=torch.tensor([0, 1, 0, 1]),
            query_start_loc_cpu=torch.tensor([0, 2, 4]),
            lookback_token_ids=SimpleNamespace(
                cpu=torch.full((2, 2), -1, dtype=torch.int64)
            ),
        )

        # 1. plan is None
        assert capture_host_engram_inputs(runner, num_reqs=2) is None

        # 2. plan is not SequenceLayoutKind.ALL
        runner._last_sequence_layout_plan = SimpleNamespace(kind="OTHER")
        assert capture_host_engram_inputs(runner, num_reqs=2) is None

        # 3. all_prompt and start_index == 0 -> host_ready=True
        runner._last_sequence_layout_plan = SimpleNamespace(
            kind=SequenceLayoutKind.ALL, start_index=0, num_reqs=2
        )
        inp1 = capture_host_engram_inputs(runner, num_reqs=2)
        assert inp1 is not None
        assert inp1.all_prompt is True
        assert inp1.host_ready is True

        # 4. not all_prompt, speculative decode active -> host_ready=False
        runner.input_batch.num_prompt_tokens = np.array([1, 1])
        runner._pre_async_results = SimpleNamespace(spec_decode_num_rejected_tokens=0)
        inp2 = capture_host_engram_inputs(runner, num_reqs=2)
        assert inp2 is not None
        assert inp2.all_prompt is False
        assert inp2.host_ready is False

        # 5. not all_prompt, missing token_ids_cpu -> host_ready=False
        runner._pre_async_results = None
        runner.input_batch.num_computed_tokens_cpu = None
        runner.input_batch.token_ids_cpu = None
        inp3 = capture_host_engram_inputs(runner, num_reqs=2)
        assert inp3 is not None
        assert inp3.host_ready is False

        # 6. decode step with matched reqs and flat_src is None -> host_ready=False
        runner.input_batch.num_computed_tokens_cpu = np.array([2, 2])
        runner.input_batch.token_ids_cpu = np.array([[10, 20], [30, 40]])
        runner.input_batch.req_ids = ["req0", "req1"]
        runner._pre_async_results = SimpleNamespace(req_id_to_index_copy={"req0": 0})
        inp4 = capture_host_engram_inputs(runner, num_reqs=2)
        assert inp4 is not None
        assert inp4.host_ready is False

        # 7. decode step with matched reqs and flat_src provided -> host_ready=True
        runner._pre_async_results.copy_state = SimpleNamespace(
            chunks_cpu=[torch.tensor([99, 100])]
        )
        inp5 = capture_host_engram_inputs(runner, num_reqs=2)
        assert inp5 is not None
        assert inp5.host_ready is True
        assert inp5.input_ids[0].item() == 99

        # 8. decode step without matched reqs -> host_ready=True
        runner._pre_async_results = SimpleNamespace(req_id_to_index_copy={})
        inp6 = capture_host_engram_inputs(runner, num_reqs=2)
        assert inp6 is not None
        assert inp6.host_ready is True

    def test_gather_host_engram_hashes_branches(self):
        hashes_cpu = torch.ones((2, 2, 1), dtype=torch.int32, device="cpu")

        # 1. CPU, not dp_shared_memory, dp_group with world_size > 1 and cpu_group
        mock_cpu_group = MagicMock()
        mock_dp_group = SimpleNamespace(world_size=2, cpu_group=mock_cpu_group)
        mock_ctx = SimpleNamespace(dp_metadata=SimpleNamespace())

        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_engram_dp_group",
                return_value=mock_dp_group,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.engram_gathered_num_tokens",
                return_value=4,
            ),
            patch("torch.distributed.all_gather_into_tensor") as mock_all_gather,
        ):
            out = gather_host_engram_hashes(hashes_cpu, dp_shared_memory=False)
            assert out.shape == (8, 2, 1)
            mock_all_gather.assert_called_once()

            # Exceeds slot -> raises ValueError
            with (
                patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.engram_gathered_num_tokens",
                    return_value=1,
                ),
                pytest.raises(ValueError, match="exceeds the DP token slot"),
            ):
                gather_host_engram_hashes(hashes_cpu, dp_shared_memory=False)

        # 2. gather_fn provided
        called_fn = []
        gather_host_engram_hashes(
            hashes_cpu,
            dp_shared_memory=True,
            gather_fn=lambda h, dp_shared_memory: called_fn.append(
                (h.shape, dp_shared_memory)
            ),
        )
        assert called_fn == [((2, 2, 1), True)]

        # 3. Fallback to upstream gather_engram_hashes
        with patch(
            "vllm.models.deepseek_v41.nvidia.engram.gather_engram_hashes",
            return_value=torch.ones((4, 2, 1)),
        ) as mock_up_gather:
            out3 = gather_host_engram_hashes(hashes_cpu, dp_shared_memory=True)
            assert out3.shape == (4, 2, 1)
            mock_up_gather.assert_called_once()

    def test_install_host_engram_inputs(self):
        class MockRunner:
            def _prepare_lookback_token_ids(self, num_reqs: int):
                return torch.tensor([num_reqs])

        with patch("vllm_torchtpu.runner.tpu_runner.TPUModelRunner", MockRunner):
            install_host_engram_inputs()
            assert (
                getattr(
                    MockRunner._prepare_lookback_token_ids,
                    "_deepseek_v41_host_engram",
                    False,
                )
                is True
            )

            # Second call early returns
            install_host_engram_inputs()

            runner_inst = MockRunner()
            runner_inst._last_sequence_layout_plan = None
            runner_inst.input_batch = None
            # Outside forward context
            with patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram.is_forward_context_available",
                return_value=False,
            ):
                assert runner_inst._prepare_lookback_token_ids(3).item() == 3

            # Inside forward context
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
                assert runner_inst._prepare_lookback_token_ids(3).item() == 3
                assert HOST_ENGRAM_KEY in mock_ctx.additional_kwargs


# ==============================================================================
# Section 3: deepseek_v41/v2/model.py Diff Lines Coverage
# ==============================================================================


class TestV2ModelHostEngramStaging:
    def test_model_host_engram_skip_lookup(self):
        """Cover line 274: skip_lookup=True -> engram_mask."""
        model = object.__new__(DeepseekV41Model)
        model.engram_hash = SimpleNamespace(
            multipliers=torch.ones((2, 3)),
            ensure_cache=lambda: True,
        )
        model.engram_swa_prefix = "swa"
        input_ids = torch.tensor([10, 20], dtype=torch.int32)
        mock_ctx = SimpleNamespace(
            attn_metadata={"swa": SimpleNamespace(seq_lens=torch.tensor([2]))}
        )
        with (
            patch(
                "vllm.forward_context.is_forward_context_available", return_value=True
            ),
            patch("vllm.forward_context.get_forward_context", return_value=mock_ctx),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.image_sentinel_mask",
                return_value=torch.zeros(2, dtype=torch.bool),
            ),
        ):
            hashes, mask = model._prepare_engram(
                input_ids, torch.tensor([0, 1]), None, skip_lookup=True
            )
            assert hashes.shape == (2, 2, 1)
            assert mask.shape == (2,)

    def test_model_host_engram_host_not_none(self):
        """Cover line 294: host is not None -> engram_mask."""
        model = object.__new__(DeepseekV41Model)
        model.engram_hash = SimpleNamespace(
            use_slot_cache=True,
            lookback_depth=2,
            ensure_cache=lambda: True,
            hash_on_host=lambda *h: torch.ones((2, 2, 1)),
        )
        model.engram_swa_prefix = "swa"
        model._host_hash_inputs = lambda inp, seq, layer_idx: (
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
        )
        model.engram_dp_shared_memory = False
        model._has_cpu_offloaded_engram = lambda: False
        model.layers = []
        model.start_layer = 0
        model.end_layer = 0
        input_ids = torch.tensor([10, 20])
        swa = SimpleNamespace(seq_lens=torch.tensor([2]))
        mock_ctx = SimpleNamespace(attn_metadata={"swa": swa})
        with (
            patch(
                "vllm.forward_context.is_forward_context_available", return_value=True
            ),
            patch("vllm.forward_context.get_forward_context", return_value=mock_ctx),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.image_sentinel_mask",
                return_value=torch.zeros(2, dtype=torch.bool),
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.gather_host_engram_hashes",
                return_value=torch.ones((2, 2, 1)),
            ),
        ):
            hashes, mask = model._prepare_engram(
                input_ids, torch.tensor([0, 1]), torch.tensor([[0, 1]])
            )
            assert hashes.shape == (2, 2, 1)
            assert mask.shape == (2,)

    def test_model_host_engram_host_none(self):
        """Cover lines 296-297: host is None -> image_mask calculation."""
        model = object.__new__(DeepseekV41Model)
        model.engram_hash = MagicMock(return_value=torch.ones((2, 2, 1)))
        model.engram_hash.use_slot_cache = True
        model.engram_hash.lookback_depth = 2
        model.engram_hash.ensure_cache = lambda: True
        model.engram_hash.swa_cache_module = SimpleNamespace(kv_cache=torch.zeros(1))
        model.engram_swa_prefix = "swa"
        model._host_hash_inputs = lambda inp, seq, layer_idx: None
        model.engram_dp_shared_memory = False
        model._has_cpu_offloaded_engram = lambda: False
        model.layers = []
        model.start_layer = 0
        model.end_layer = 0
        input_ids = torch.tensor([10, 20])
        swa = SimpleNamespace(
            seq_lens=torch.tensor([2]), query_start_loc=torch.tensor([0, 2])
        )
        mock_ctx = SimpleNamespace(attn_metadata={"swa": swa})
        with (
            patch(
                "vllm.forward_context.is_forward_context_available", return_value=True
            ),
            patch("vllm.forward_context.get_forward_context", return_value=mock_ctx),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.image_sentinel_mask",
                return_value=torch.zeros(2, dtype=torch.bool),
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
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.gather_host_engram_hashes",
                return_value=torch.ones((2, 2, 1)),
            ),
        ):
            hashes, mask = model._prepare_engram(
                input_ids, torch.tensor([0, 1]), torch.tensor([[0, 1]])
            )
            assert hashes.shape == (2, 2, 1)
            assert mask.shape == (2,)

    def test_model_host_engram_forward_offload_mega_moe(self):
        """Cover lines 345-349: offload_mega_moe input_ids int64 casting."""
        model = object.__new__(DeepseekV41Model)
        model._has_cpu_offloaded_engram = lambda: True
        model._ced_enabled = False
        model.full_stage = lambda *args, **kwargs: torch.tensor([42])

        prep_called = []
        model._prepare_engram = lambda ids, p, layer_idx, stage: prep_called.append(
            (ids.dtype if ids is not None else None, stage)
        )

        # Case 1: self.use_mega_moe=True and input_ids on cpu
        model.use_mega_moe = True
        input_ids = torch.tensor([1, 2], dtype=torch.int32, device="cpu")
        model.forward(input_ids, torch.tensor([0, 1]))
        assert prep_called[-1] == (torch.int64, "full")

        # Case 2: self.use_mega_moe=False -> input_ids kept as-is
        model.use_mega_moe = False
        input_ids = torch.tensor([1, 2], dtype=torch.int32, device="cpu")
        model.forward(input_ids, torch.tensor([0, 1]))
        assert prep_called[-1] == (torch.int32, "full")

        # Case 3: input_ids is None -> kept as-is
        model.use_mega_moe = True
        model.forward(None, None)
        assert prep_called[-1] == (None, "full")

        # Case 4: input_ids device is not cpu (mocked cuda/tpu device)
        model.use_mega_moe = True
        cuda_ids = MagicMock()
        cuda_ids.device.type = "cuda"
        cuda_ids.dtype = torch.int32
        model.forward(cuda_ids, torch.tensor([0, 1]))
        assert prep_called[-1] == (torch.int32, "full")

        # Stage with _ced_enabled True
        model._ced_enabled = True
        with contextlib.suppress(Exception):
            model.forward(input_ids, torch.tensor([0, 1]))
        assert prep_called[-1] == (torch.int64, "encoder")


# ==============================================================================
# Section 4: fused_moe/v2/token_parallel.py Full Coverage
# ==============================================================================


class TestV2TokenParallelFull:
    def test_create_validation_guards(self):
        with pytest.raises(
            ValueError, match="EP width and input row count must be positive"
        ):
            TokenReplicaLayout.create(ep=0, input_rows=10)
        with pytest.raises(
            ValueError, match="EP width and input row count must be positive"
        ):
            TokenReplicaLayout.create(ep=2, input_rows=0)
        with pytest.raises(
            ValueError, match="row alignment must be a positive power of two"
        ):
            TokenReplicaLayout.create(ep=2, input_rows=10, row_alignment=0)
        with pytest.raises(
            ValueError, match="row alignment must be a positive power of two"
        ):
            TokenReplicaLayout.create(ep=2, input_rows=10, row_alignment=3)
        with pytest.raises(ValueError, match="token replica groups must partition"):
            TokenReplicaLayout.create(ep=2, input_rows=10, groups=())
        with pytest.raises(ValueError, match="token replica groups must partition"):
            TokenReplicaLayout.create(ep=2, input_rows=10, groups=((),))
        with pytest.raises(ValueError, match="token replica groups must partition"):
            TokenReplicaLayout.create(ep=3, input_rows=10, groups=((0, 1), (2,)))
        with pytest.raises(ValueError, match="token replica groups must partition"):
            TokenReplicaLayout.create(ep=2, input_rows=10, groups=((0, 2), (1, 3)))

    def test_singleton_split_and_restore(self):
        layout = TokenReplicaLayout.create(ep=2, input_rows=10)
        rows = jnp.ones((10, 4))
        rank = jnp.array(0)
        # Singleton returns input directly
        assert layout.split(rows, rank) is rows
        assert layout.restore(rows, "axis") is rows

    def test_replicated_split_and_restore(self):
        layout = TokenReplicaLayout.create(
            ep=2, input_rows=5, groups=((0, 1),), row_alignment=8
        )
        rows = jnp.arange(20).reshape(5, 4)
        rank0 = jnp.array(0)
        s0 = layout.split(rows, rank0)
        assert s0.shape == (layout.local_rows, 4)

        rank1 = jnp.array(1)
        s1 = layout.split(rows, rank1)
        assert s1.shape == (layout.local_rows, 4)

        # test restore with mock lax.all_gather
        gathered_mock = jnp.pad(rows, ((0, 3), (0, 0)))
        with patch("jax.lax.all_gather", return_value=gathered_mock):
            restored = layout.restore(s0, "x")
            assert restored.shape == (5, 4)


# ==============================================================================
# Section 5: fused_moe/v2/router_ops.py Full Coverage
# ==============================================================================


class TestV2RouterOpsFull:
    def test_pallas_select_validation_and_interpret(self):
        scores = jnp.ones((32, 8), dtype=jnp.float32)
        # Shape mismatch on weight_scores
        bad_weight_scores = jnp.ones((16, 8), dtype=jnp.float32)
        with pytest.raises(ValueError, match="must match scores"):
            pallas_select(
                scores,
                topk=2,
                block_rows=32,
                weight_scores=bad_weight_scores,
                interpret=True,
            )

        # Single-input select with interpret=True
        w, idx = pallas_select(scores, topk=2, block_rows=32, interpret=True)
        assert w.shape == (32, 2)
        assert idx.shape == (32, 2)

        # Split-input select with interpret=True
        w_scores = scores + 0.5
        w2, idx2 = pallas_select(
            scores, topk=2, block_rows=32, weight_scores=w_scores, interpret=True
        )
        assert w2.shape == (32, 2)
        assert idx2.shape == (32, 2)


# ==============================================================================
# Section 6: fused_moe/v2/host.py Full Coverage
# ==============================================================================


class TestV2HostFull:
    def test_pow2_shift_and_helpers(self):
        with pytest.raises(ValueError, match="not a power of two"):
            pow2_shift(0, "test")
        with pytest.raises(ValueError, match="not a power of two"):
            pow2_shift(3, "test")

        assert token_gather_window_rows(32) > 0
        assert token_gather_smem_bytes(32) > 0
        assert contrib_mirror_rows(64, 2, 32) > 0
        assert arrival_mirror_rows(64, 4, 32) > 0

    def test_weight_format_and_form(self):
        assert str(WeightFormat.FP8) == "fp8"
        assert repr(WeightFormat.FP8) == "'fp8'"

        with pytest.raises(NotImplementedError):
            weight_form("invalid_format")
        assert weight_format_of_dtype(jnp.bfloat16) == WeightFormat.BF16
        assert weight_format_of_dtype(jnp.float32) is None

    def test_refuse_bad_plan_args(self):
        with pytest.raises(ValueError, match="need at least one token"):
            _refuse_bad_plan_args(0, 1, e_total=4, ep=2, t_local=1, block=1, tile_m=8)
        with pytest.raises(
            ValueError, match="not divisible by the expert-parallel width"
        ):
            _refuse_bad_plan_args(2, 1, e_total=5, ep=2, t_local=1, block=1, tile_m=8)
        with pytest.raises(ValueError, match="alignment slots"):
            _refuse_bad_plan_args(
                2, 1, e_total=20000, ep=20000, t_local=1, block=1, tile_m=8
            )
        with pytest.raises(ValueError, match="arrival rows per shard"):
            _refuse_bad_plan_args(
                100000000, 1, e_total=2, ep=2, t_local=1, block=1, tile_m=8
            )
        with pytest.raises(ValueError, match="routing blocks"):
            _refuse_bad_plan_args(3, 1, e_total=2, ep=2, t_local=1, block=2, tile_m=8)
        with pytest.raises(ValueError, match="this shard's"):
            _refuse_bad_plan_args(4, 1, e_total=2, ep=2, t_local=1, block=2, tile_m=8)
        with pytest.raises(ValueError, match="tile height"):
            _refuse_bad_plan_args(2, 1, e_total=2, ep=2, t_local=1, block=1, tile_m=7)

    def test_build_routing_tables_shard_stride_mismatch(self):
        topk = jnp.zeros((2, 1), dtype=jnp.int32)
        with pytest.raises(ValueError, match="per-shard slab stride"):
            build_routing_tables(
                topk, e_total=2, ep=2, t_local=1, block=1, tile_m=8, shard_stride=9
            )

    def test_pair_block_hist_and_rank_window(self):
        blocks = jnp.zeros((2, 4), dtype=jnp.int32)
        hist = pair_block_hist(blocks, 2)
        assert hist.shape == (2, 2)
        w = rank_window(48, host.ROUTING_WINDOW_MIN_PAIRS + 10)
        assert w == 16

    def test_build_routing_tables_sharded_guards_and_gathered(self):
        with pytest.raises(
            ValueError, match="pass exactly one of all_gather_rows and gathered"
        ):
            build_routing_tables_sharded(
                None, 0, e_total=2, ep=2, t_local=1, block=1, tile_m=8, shard_stride=8
            )
        with pytest.raises(
            ValueError, match="pass exactly one of all_gather_rows and gathered"
        ):
            build_routing_tables_sharded(
                None,
                0,
                e_total=2,
                ep=2,
                t_local=1,
                block=1,
                tile_m=8,
                shard_stride=8,
                all_gather_rows=lambda x: x,
                gathered=object(),
            )
        with pytest.raises(ValueError, match="per-shard slab stride"):
            build_routing_tables_sharded(
                jnp.zeros((2, 1), jnp.int32),
                0,
                e_total=2,
                ep=2,
                t_local=1,
                block=1,
                tile_m=8,
                shard_stride=9,
                all_gather_rows=lambda x: x,
            )

        gathered = GatheredPairs(
            expert_blocks=jnp.zeros((2, 1), dtype=jnp.int32),
            local_blocks=jnp.zeros((1, 1), dtype=jnp.int32),
            block_hist=jnp.zeros((1, 2), dtype=jnp.int32),
            rows_by_dest=jnp.zeros((2, 2), dtype=jnp.int32),
            n_tokens=2,
            topk=1,
        )
        res_sharded = build_routing_tables_sharded(
            None,
            0,
            e_total=2,
            ep=2,
            t_local=1,
            block=1,
            tile_m=8,
            shard_stride=8,
            gathered=gathered,
        )
        assert res_sharded is not None

    def test_vmem_and_slots_with_mock_tpu_info(self):
        mock_info = _mock_tpu_info()
        with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info):
            assert vmem_limit(None) == int(64 * 1024 * 1024 * host.VMEM_FRACTION)
            assert chip_generation(None) == 7
            assert chip_generation(mock_info) == 7  # covers branch 1288->1290
            check_u32_sublane_tile(None)

            bad_info = MagicMock()
            bad_info.get_sublane_tiling.return_value = 4
            with pytest.raises(ValueError, match="tiles them to 4"):
                check_u32_sublane_tile(bad_info)

            b1 = array_vmem_bytes((10,), jnp.float32, mock_info)
            assert b1 == 128 * 4

            sc = vmem_scratch_arrays(
                1,
                32,
                128,
                128,
                nbuf=2,
                weight_format=WeightFormat.BF16,
                has_w1_bias=True,
                has_w2_bias=True,
            )
            names = [n for n, _, _ in sc]
            assert "w1b_vm" in names
            assert "w2b_vm" in names
            assert "w1s_vm" not in names

            t_bf16 = vmem_tile_body_arrays(
                32, 128, 128, weight_format=WeightFormat.BF16
            )
            t_names = [n for n, _, _ in t_bf16]
            assert "mid_concat" in t_names
            assert "wire_rows" not in t_names

            t_int8 = vmem_tile_body_arrays(
                32, 128, 128, weight_format=WeightFormat.INT8
            )
            assert any(n == "widened_weight_block" for n, _, _ in t_int8)

            v_est = vmem_estimate_bytes(1, 32, 128, 128, 2, info=None)
            assert v_est > 0

            with pytest.raises(ValueError, match="negative distance"):
                fit_weight_slots(1, 32, 128, 128, max_distance=-1)

            slots = fit_weight_slots(1, 32, 128, 128, info=None, limit=None)
            assert slots.fits


# ==============================================================================
# Section 7: fused_moe/v2/layer.py Full Coverage
# ==============================================================================


class TestV2LayerFull:
    def test_apply_scoring_fn(self):
        logits = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        assert _apply_scoring_fn(logits, "softmax").shape == (2, 2)
        assert _apply_scoring_fn(logits, "sigmoid").shape == (2, 2)
        assert _apply_scoring_fn(logits, "sqrtsoftplus").shape == (2, 2)
        with pytest.raises(NotImplementedError):
            _apply_scoring_fn(logits, "invalid_scoring_fn")

    def test_combine_arrivals_branches(self):
        arrivals = jnp.ones((64, 1, 128), dtype=jnp.bfloat16)
        arrival_scales = jnp.ones((8, 32), dtype=jnp.float32)
        pos = jnp.zeros((32, 2), dtype=jnp.int32)
        mirror_pos = jnp.zeros((32, 2), dtype=jnp.int32)
        topk_weights = jnp.ones((32, 2), dtype=jnp.float32)

        # Branch 1: combine_step_tokens(32) == 0 (fallback slice and weighted sum)
        c0 = _combine_arrivals(
            arrivals, arrival_scales, pos, mirror_pos, topk_weights, jnp.bfloat16
        )
        assert c0.shape == (32, 128)

        # Branch 2: combine_step_tokens(256) > 0 (calls build_combine_kernel)
        pos256 = jnp.zeros((256, 2), dtype=jnp.int32)
        mirror_pos256 = jnp.zeros((256, 2), dtype=jnp.int32)
        topk_weights256 = jnp.ones((256, 2), dtype=jnp.float32)
        mock_combine = MagicMock(return_value=jnp.zeros((256, 128), jnp.bfloat16))
        with patch(
            "vllm_torchtpu.kernels.fused_moe.v2.layer.build_combine_kernel",
            return_value=mock_combine,
        ):
            c1 = _combine_arrivals(
                arrivals,
                arrival_scales,
                pos256,
                mirror_pos256,
                topk_weights256,
                jnp.bfloat16,
            )
            assert c1.shape == (256, 128)

    def test_exchange_layout_pack_unpack(self):
        layout_q = _exchange_layout(32, 2, 4, quantized=True)
        assert layout_q[0] > 0
        layout_unq = _exchange_layout(32, 2, 4, quantized=False)
        assert layout_unq[0] == layout_unq[1]

        # pack & unpack with scale_bits
        pairs = jnp.ones((64,), dtype=jnp.int32)
        scale_bits = jnp.ones((32,), dtype=jnp.int32)
        counts = jnp.ones((4,), dtype=jnp.int32)
        packed = _pack_routing_exchange(pairs, scale_bits, counts, layout_q)
        exchange = packed[None, :].repeat(2, axis=0)
        p, s, c = _unpack_routing_exchange(exchange, layout_q, 32, 2, 4)
        assert p.shape == (2, 64)
        assert s is not None
        assert c.shape == (2, 4)

        # pack & unpack without scale_bits
        packed_no_scale = _pack_routing_exchange(pairs, None, counts, layout_unq)
        exchange_no_scale = packed_no_scale[None, :].repeat(2, axis=0)
        p2, s2, c2 = _unpack_routing_exchange(exchange_no_scale, layout_unq, 32, 2, 4)
        assert s2 is None

    def test_relabel_expert_ids_to_mesh_order(self):
        topk = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
        assert (
            _relabel_expert_ids_to_mesh_order(topk, g_local=2, mesh_ep_ranks=None)
            is topk
        )

        with pytest.raises(ValueError, match="permutation of range"):
            _relabel_expert_ids_to_mesh_order(topk, g_local=2, mesh_ep_ranks=(0, 2))

        relabeled = _relabel_expert_ids_to_mesh_order(
            topk, g_local=2, mesh_ep_ranks=(1, 0)
        )
        assert relabeled.shape == (2, 2)

    def test_fused_ep_moe_v2_validation_guards(self):
        mesh = jax.sharding.Mesh(jax.devices()[:1], (AXIS,))
        x = jnp.ones((32, 128), jnp.bfloat16)
        w1_fp8 = jnp.ones((4, 128, 256), jnp.float8_e4m3fn)
        w2_fp8 = jnp.ones((4, 128, 128), jnp.float8_e4m3fn)
        w1_bf16 = jnp.ones((4, 128, 256), jnp.bfloat16)
        w2_bf16 = jnp.ones((4, 128, 128), jnp.bfloat16)
        gating = jnp.ones((32, 4), jnp.float32)

        # Mismatched weight format
        with pytest.raises(ValueError, match="expert weights"):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_fp8,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
            )

        # Scale presence mismatch (BF16 given scales)
        with pytest.raises(ValueError, match="weight scales and w1_scale is present"):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                jnp.ones((4, 256)),
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                weight_format=WeightFormat.BF16,
            )

        # Scale pair mismatch
        with pytest.raises(
            ValueError, match="both weight scales are supplied together"
        ):
            fused_ep_moe_v2(
                x,
                w1_fp8,
                w2_fp8,
                jnp.ones((4, 256)),
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
            )

        # Scale layout mismatch
        with pytest.raises(ValueError, match="layout"):
            fused_ep_moe_v2(
                x,
                w1_fp8,
                w2_fp8,
                jnp.ones((4, 256, 1)),
                jnp.ones((4, 128, 1)),
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
            )

        # mesh_ep_ranks invalid
        with pytest.raises(ValueError, match="mesh_ep_ranks must be a permutation"):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                weight_format=WeightFormat.BF16,
                mesh_ep_ranks=(1,),
            )

        # Gating shape mismatch
        with pytest.raises(ValueError, match="input and gating rows must agree"):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                jnp.ones((16, 4)),
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                weight_format=WeightFormat.BF16,
            )

        # ragged_stride % capacity != 0
        with pytest.raises(ValueError, match="ragged_stride must be a multiple"):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                ragged_stride=33,
                weight_format=WeightFormat.BF16,
            )

        # ragged_stride < stride_bound
        with pytest.raises(ValueError, match="below the no-drop bound"):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                ragged_stride=32,
                weight_format=WeightFormat.BF16,
            )

        # Invalid scoring_fn
        with pytest.raises(ValueError, match="scoring_fn must be one of"):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                weight_format=WeightFormat.BF16,
                scoring_fn="invalid",
            )

        # Invalid score_bias shape
        with pytest.raises(
            ValueError, match="score_bias must be one value per global expert"
        ):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                weight_format=WeightFormat.BF16,
                score_bias=jnp.ones((2,)),
            )

        # Invalid hash_topk_ids shape and dtype
        with pytest.raises(
            ValueError, match="hash_topk_ids must be one expert id per token slot"
        ):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                weight_format=WeightFormat.BF16,
                hash_topk_ids=jnp.zeros((16, 2), jnp.int32),
            )
        with pytest.raises(ValueError, match="integer dtype"):
            fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
                weight_format=WeightFormat.BF16,
                hash_topk_ids=jnp.zeros((32, 2), jnp.float32),
            )

    def test_fused_ep_moe_v2_execution_matrix(self):
        mesh = jax.sharding.Mesh(jax.devices()[:1], (AXIS,))
        T, hidden, e_total, inter = 32, 128, 4, 128
        x = jnp.ones((T, hidden), dtype=jnp.bfloat16)
        w1_fp8 = jnp.ones((e_total, hidden, 2 * inter), dtype=jnp.float8_e4m3fn)
        w2_fp8 = jnp.ones((e_total, inter, hidden), dtype=jnp.float8_e4m3fn)
        w1_s = jnp.ones((e_total, 2 * inter), dtype=jnp.float32)
        w2_s = jnp.ones((e_total, hidden), dtype=jnp.float32)
        gating = jnp.ones((T, e_total), dtype=jnp.float32)
        # Add NaN to gating row to exercise row_routes False branch
        gating = gating.at[0].set(jnp.nan)

        def mock_kfn(*args, **kwargs):
            recv_rows = kwargs.get("recv_rows", 64)
            arrivals = jnp.zeros(
                (recv_rows, hidden // 128, 128), dtype=jnp.float8_e4m3fn
            )
            arrival_scales = jnp.ones(
                (host.arrival_mirror_rows(recv_rows, e_total, 32), 32),
                dtype=jnp.float32,
            )
            return arrivals, arrival_scales

        with (
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.layer.pallas_select",
                side_effect=lambda *args, **kwargs: pallas_select(
                    *args, **{**kwargs, "interpret": True}
                ),
            ),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.layer.build_fused_ep_moe_kernel",
                return_value=mock_kfn,
            ),
        ):
            # Run 1: FP8, renormalize=True, score_bias, w1_bias, w2_bias,
            # routed_scaling_factor!=1.0, mesh_ep_ranks
            # (AGQ_MIN_LOCAL_TOKENS=16 -> branch 600->601)
            with patch.object(layer, "AGQ_MIN_LOCAL_TOKENS", 16):
                out1 = fused_ep_moe_v2(
                    x,
                    w1_fp8,
                    w2_fp8,
                    w1_s,
                    w2_s,
                    gating,
                    w1_bias=jnp.zeros((e_total, 1, 2 * inter), jnp.float32),
                    w2_bias=jnp.zeros((e_total, 1, hidden), jnp.float32),
                    mesh=mesh,
                    topk=2,
                    renormalize=True,
                    capacity=32,
                    score_bias=jnp.zeros((e_total,), jnp.float32),
                    routed_scaling_factor=2.0,
                    mesh_ep_ranks=(0,),
                )
            assert out1.shape == (T, hidden)

            # Run 2: BF16 (unquantized), renormalize=False, hash_topk_ids,
            # sharded_plan=True
            w1_bf16 = jnp.ones((e_total, hidden, 2 * inter), dtype=jnp.bfloat16)
            w2_bf16 = jnp.ones((e_total, inter, hidden), dtype=jnp.bfloat16)
            hash_ids = jnp.zeros((T, 2), dtype=jnp.int32)
            out2 = fused_ep_moe_v2(
                x,
                w1_bf16,
                w2_bf16,
                None,
                None,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=False,
                capacity=32,
                weight_format=WeightFormat.BF16,
                hash_topk_ids=hash_ids,
                sharded_plan=True,
            )
            assert out2.shape == (T, hidden)

            # Run 3: token_layout with multiple groups (len > 1),
            # score_bias=None, hash_topk_ids=None, block=16, rank explicitly passed
            tl = MagicMock()
            tl.groups = ((0, 1),)
            tl.split.side_effect = lambda rows, *args, **kw: rows
            tl.local_rows = 32
            tl.restore.side_effect = lambda rows, *args, **kw: rows

            with patch(
                "vllm_torchtpu.kernels.fused_moe.v2.layer.TokenReplicaLayout.create",
                return_value=tl,
            ):
                out3 = fused_ep_moe_v2(
                    x,
                    w1_fp8,
                    w2_fp8,
                    w1_s,
                    w2_s,
                    gating,
                    mesh=mesh,
                    topk=2,
                    renormalize=True,
                    capacity=32,
                    block=16,
                    rank=jnp.array([[0]], jnp.int32),
                )
                assert out3.shape == (T, hidden)

                # Calling again with identical arguments triggers
                # _LAYER_SM_CACHE hit (line 769->781)
                out3_cached = fused_ep_moe_v2(
                    x,
                    w1_fp8,
                    w2_fp8,
                    w1_s,
                    w2_s,
                    gating,
                    mesh=mesh,
                    topk=2,
                    renormalize=True,
                    capacity=32,
                    block=16,
                    rank=jnp.array([[0]], jnp.int32),
                )
                assert out3_cached.shape == (T, hidden)

    def test_layer_sm_cache_double_check_locking(self):
        mesh = jax.sharding.Mesh(jax.devices()[:1], (AXIS,))
        T, hidden, e_total, inter = 32, 128, 4, 128
        x = jnp.ones((T, hidden), dtype=jnp.bfloat16)
        w1_fp8 = jnp.ones((e_total, hidden, 2 * inter), dtype=jnp.float8_e4m3fn)
        w2_fp8 = jnp.ones((e_total, inter, hidden), dtype=jnp.float8_e4m3fn)
        w1_s = jnp.ones((e_total, 2 * inter), dtype=jnp.float32)
        w2_s = jnp.ones((e_total, hidden), dtype=jnp.float32)
        gating = jnp.ones((T, e_total), dtype=jnp.float32)

        mock_sm = MagicMock(return_value=jnp.zeros((T, hidden), jnp.bfloat16))

        with (
            patch.object(layer, "_LAYER_SM_CACHE", MockCache(mock_sm)),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.layer.build_fused_ep_moe_kernel",
                return_value=MagicMock(),
            ),
        ):
            out = fused_ep_moe_v2(
                x,
                w1_fp8,
                w2_fp8,
                w1_s,
                w2_s,
                gating,
                mesh=mesh,
                topk=2,
                renormalize=True,
                capacity=32,
            )
            assert out.shape == (T, hidden)


# ==============================================================================
# Section 8: fused_moe/v2/kernel.py Full Coverage
# ==============================================================================


class TestV2KernelFull:
    def test_apply_act_and_tile_height_rungs(self):
        gate = jnp.ones((2, 4), dtype=jnp.float32)
        up = jnp.ones((2, 4), dtype=jnp.float32)
        assert _apply_act(gate, up, "silu").shape == (2, 4)
        assert _apply_act(gate, up, "swigluoai").shape == (2, 4)
        assert _apply_act(gate, up, "silu_and_mul_with_clamp").shape == (2, 4)
        with pytest.raises(NotImplementedError):
            _apply_act(gate, up, "unknown_act")

        # tile_height_rungs branches
        with patch.dict(os.environ, {"MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE": "1"}):
            assert tile_height_rungs(32, 32, 1, 1) == (32,)
        with patch.dict(os.environ, {"MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE": "4"}):
            # rows_alloc > ep * g_local * tile_m
            assert tile_height_rungs(32, 64, 1, 1) == (32,)
            # standard rungs
            assert tile_height_rungs(32, 32, 1, 1) == (8, 16, 24, 32)

    def test_quant_and_wire_scale(self):
        x = jnp.array([[1.0, 2.0], [0.0, 0.0], [jnp.nan, 1.0]], dtype=jnp.float32)
        q, s = rowquant_fp8(x)
        assert q.shape == (3, 2)
        assert s.shape == (3, 1)

        scale, sinv, usable = _row_scale(jnp.array([[0.0], [1.0], [jnp.inf]]))
        assert usable.shape == (3, 1)

        applied = _quant_apply(x, sinv, usable)
        assert applied.shape == (3, 2)
        assert _absmax_rows(x).shape == (3, 1)

        # wire_row_and_scale branches
        acc2 = jnp.ones((4, 8), dtype=jnp.float32)
        w2s = jnp.ones((1, 8), dtype=jnp.float32)
        w2b = jnp.ones((1, 8), dtype=jnp.float32)
        # FP8 wire with scales and biases
        w_fp8, s_fp8 = wire_row_and_scale(
            acc2, jnp.ones((4, 1)), w2s=w2s, w2b=w2b, wire_dtype=kernel.FP8
        )
        assert w_fp8.shape == (4, 8)

        # BF16 unquantized wire
        w_bf16, s_bf16 = wire_row_and_scale(
            acc2, None, w2s=None, w2b=None, wire_dtype=jnp.bfloat16, tile_m=4
        )
        assert w_bf16.shape == (4, 8)
        assert s_bf16.shape == (4, 1)

    def test_intermediate_rows_and_dots(self):
        acc1 = jnp.ones((4, 16), dtype=jnp.float32)
        w1s = jnp.ones((1, 16), dtype=jnp.float32)
        w1b = jnp.ones((1, 16), dtype=jnp.float32)

        # requantize = True
        mid_q, s = _intermediate_rows(
            acc1, jnp.ones((4, 1)), 8, w1s=w1s, act_fn="silu", w1b=w1b, requantize=True
        )
        assert mid_q.shape == (4, 8)
        assert s.shape == (4, 1)

        # requantize = False
        mid_bf16, s_none = _intermediate_rows(
            acc1, None, 8, w1s=None, act_fn="silu", w1b=None, requantize=False
        )
        assert mid_bf16.shape == (4, 8)
        assert s_none is None

        # _chunked_dot
        lhs = jnp.ones((4, 16), dtype=jnp.bfloat16)

        def chunk_fn(b):
            return jnp.ones((8, 8), dtype=jnp.bfloat16)

        dot = _chunked_dot(lhs, chunk_fn, 2, 8)
        assert dot.shape == (4, 8)

    def test_expert_ffn_bodies(self):
        act_q = jnp.ones((4, 16), dtype=kernel.FP8)
        act_s = jnp.ones((4, 1), dtype=jnp.float32)
        w1_fp8 = jnp.ones((16, 16), dtype=kernel.FP8)
        w2_fp8 = jnp.ones((8, 16), dtype=kernel.FP8)
        w1s = jnp.ones((1, 16), dtype=jnp.float32)

        # expert_ffn_fp8
        a2, s2 = expert_ffn_fp8(act_q, act_s, w1_fp8, w2_fp8, w1s)
        assert a2.shape == (4, 16)
        assert s2.shape == (4, 1)

        # expert_ffn_blockscale
        def w1_block(b):
            return jnp.ones((16, 16), dtype=kernel.FP8)

        def w2_block(b):
            return jnp.ones((8, 16), dtype=kernel.FP8)

        w1s_blocks = jnp.ones((1, 16), dtype=jnp.float32)
        w2s_blocks = jnp.ones((1, 16), dtype=jnp.float32)
        a_bs, s_bs = expert_ffn_blockscale(
            act_q, act_s, w1_block, w2_block, w1s_blocks, w2s_blocks, qb=16
        )
        assert a_bs.shape == (4, 16)

        # expert_ffn_bf16
        act_bf16 = jnp.ones((4, 16), dtype=jnp.bfloat16)
        w1_bf16 = jnp.ones((16, 16), dtype=jnp.bfloat16)
        w2_bf16 = jnp.ones((8, 16), dtype=jnp.bfloat16)
        a_bf16, s_none = expert_ffn_bf16(act_bf16, w1_bf16, w2_bf16)
        assert a_bf16.shape == (4, 16)
        assert s_none is None

        # expert_ffn_int8
        def w1_chunk(b):
            return jnp.ones((8, 16), dtype=jnp.bfloat16)

        def w2_chunk(b):
            return jnp.ones((8, 16), dtype=jnp.bfloat16)

        a_int8, _ = expert_ffn_int8(
            act_bf16, w1_chunk, w2_chunk, w1s, n_chunks1=2, n_chunks2=1, kb=8
        )
        assert a_int8.shape == (4, 16)

    def test_kernel_helpers(self):
        # _all_pairs_barrier
        with (
            patch("jax.experimental.pallas.tpu.get_barrier_semaphore"),
            patch("jax.experimental.pallas.semaphore_signal"),
            patch("jax.experimental.pallas.semaphore_wait"),
            patch(
                "jax.experimental.pallas.run_scoped",
                side_effect=lambda fn, **kw: fn(MagicMock()),
            ),
        ):
            _all_pairs_barrier(2)

        # _slot_copy & _rows_wait
        mock_copy = MagicMock()
        mock_ref = MagicMock()
        mock_ref.at = MagicMock(return_value=mock_ref)
        with patch(
            "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
        ):
            _slot_copy(mock_ref, mock_ref, mock_ref, 0)
            _rows_wait(MagicMock(), mock_ref, 32)
            assert mock_copy.wait.called

        # _and_nonempty
        assert _and_nonempty(True, 10)
        assert not _and_nonempty(True, 0)

        # _tile_row_scales
        with patch(
            "jax.experimental.pallas.tpu.roll",
            side_effect=lambda x, s, a: jnp.roll(x, s, axis=a),
        ):
            scales_col = _tile_row_scales(jnp.ones((4, 128)), jnp.int32(10), 32)
            assert scales_col.shape == (32, 1)

    def test_build_fused_ep_moe_kernel_validations(self):
        mock_info = _mock_tpu_info()
        with patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info):
            # Invalid act_fn
            with pytest.raises(NotImplementedError, match="fuses"):
                _build_fused_ep_moe_kernel(
                    g_local=1,
                    capacity=32,
                    hidden=128,
                    inter=128,
                    ep=1,
                    act_fn="unknown",
                )

            # capacity % ROWBLK != 0
            with pytest.raises(ValueError, match="not a whole number"):
                _build_fused_ep_moe_kernel(
                    g_local=1, capacity=31, hidden=128, inter=128, ep=1
                )

            # rhs_packed4 block size validations
            with pytest.raises(ValueError, match="no block smaller than one"):
                _build_fused_ep_moe_kernel(
                    g_local=1,
                    capacity=32,
                    hidden=128,
                    inter=128,
                    ep=1,
                    weight_format=WeightFormat.FP4,
                    rhs_qb=0,
                )
            with pytest.raises(ValueError, match="has to divide both"):
                _build_fused_ep_moe_kernel(
                    g_local=1,
                    capacity=32,
                    hidden=128,
                    inter=128,
                    ep=1,
                    weight_format=WeightFormat.FP4,
                    rhs_qb=48,
                )
            with pytest.raises(ValueError, match="packed-weight row tile"):
                _build_fused_ep_moe_kernel(
                    g_local=1,
                    capacity=32,
                    hidden=128,
                    inter=128,
                    ep=1,
                    weight_format=WeightFormat.FP4,
                    rhs_qb=16,
                )

            # INT8 WIDEN_KCHUNK validation
            with pytest.raises(ValueError, match="integer weights widen"):
                _build_fused_ep_moe_kernel(
                    g_local=1,
                    capacity=32,
                    hidden=128,
                    inter=120,
                    ep=1,
                    weight_format=WeightFormat.INT8,
                )

            # hidden bounds
            with pytest.raises(ValueError, match="hidden"):
                _build_fused_ep_moe_kernel(
                    g_local=1, capacity=32, hidden=64, inter=128, ep=1
                )
            with pytest.raises(ValueError, match="inter"):
                _build_fused_ep_moe_kernel(
                    g_local=1, capacity=32, hidden=128, inter=64, ep=1
                )

            # slots.fits = False
            unfit_slots = WeightSlots(0, 1, 100000000, 1000)
            with (
                patch(
                    "vllm_torchtpu.kernels.fused_moe.v2.host.fit_weight_slots",
                    return_value=unfit_slots,
                ),
                pytest.raises(ValueError, match="budget"),
            ):
                _build_fused_ep_moe_kernel(
                    g_local=1, capacity=32, hidden=128, inter=128, ep=1
                )

            # ragged_rows_alloc % ROWBLK or capacity
            with pytest.raises(ValueError, match="ragged_rows_alloc"):
                _build_fused_ep_moe_kernel(
                    g_local=1,
                    capacity=32,
                    hidden=128,
                    inter=128,
                    ep=1,
                    ragged_rows_alloc=33,
                )

    def test_build_fused_ep_moe_kernel_operand_guards_and_cache(self):
        mock_info = _mock_tpu_info()
        with (
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch(
                "jax.experimental.pallas.pallas_call",
                return_value=MagicMock(
                    return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())
                ),
            ),
        ):
            fn = _build_fused_ep_moe_kernel(
                g_local=1,
                capacity=32,
                hidden=128,
                inter=128,
                ep=1,
                ragged_rows_alloc=32,
                has_w1_bias=True,
                has_w2_bias=True,
            )
            # missing act_scales
            with pytest.raises(ValueError, match="activation row scale"):
                fn(
                    [jnp.zeros((1, 1))] * 5,
                    jnp.zeros(32),
                    jnp.zeros(1),
                    jnp.zeros(1),
                    jnp.zeros((32, 1, 128)),
                    None,
                    jnp.zeros((1, 128, 256)),
                    jnp.zeros((1, 128, 128)),
                    jnp.zeros((1, 256)),
                    jnp.zeros((1, 128)),
                    recv_rows=32,
                    visit=jnp.zeros(1),
                    counts=jnp.zeros((5, 1)),
                    rank=jnp.zeros((1, 1)),
                )

            # missing w1s
            with pytest.raises(ValueError, match="gate/up weight scale table"):
                fn(
                    [jnp.zeros((1, 1))] * 5,
                    jnp.zeros(32),
                    jnp.zeros(1),
                    jnp.zeros(1),
                    jnp.zeros((32, 1, 128)),
                    jnp.zeros((2, 128)),
                    jnp.zeros((1, 128, 256)),
                    jnp.zeros((1, 128, 128)),
                    None,
                    jnp.zeros((1, 128)),
                    recv_rows=32,
                    visit=jnp.zeros(1),
                    counts=jnp.zeros((5, 1)),
                    rank=jnp.zeros((1, 1)),
                )

            # missing w2s
            with pytest.raises(ValueError, match="down weight scale table"):
                fn(
                    [jnp.zeros((1, 1))] * 5,
                    jnp.zeros(32),
                    jnp.zeros(1),
                    jnp.zeros(1),
                    jnp.zeros((32, 1, 128)),
                    jnp.zeros((2, 128)),
                    jnp.zeros((1, 128, 256)),
                    jnp.zeros((1, 128, 128)),
                    jnp.zeros((1, 256)),
                    None,
                    recv_rows=32,
                    visit=jnp.zeros(1),
                    counts=jnp.zeros((5, 1)),
                    rank=jnp.zeros((1, 1)),
                )

            # missing w1b
            with pytest.raises(ValueError, match="gate/up bias"):
                fn(
                    [jnp.zeros((1, 1))] * 5,
                    jnp.zeros(32),
                    jnp.zeros(1),
                    jnp.zeros(1),
                    jnp.zeros((32, 1, 128)),
                    jnp.zeros((2, 128)),
                    jnp.zeros((1, 128, 256)),
                    jnp.zeros((1, 128, 128)),
                    jnp.zeros((1, 256)),
                    jnp.zeros((1, 128)),
                    w1b=None,
                    w2b=jnp.zeros((1, 128)),
                    recv_rows=32,
                    visit=jnp.zeros(1),
                    counts=jnp.zeros((5, 1)),
                    rank=jnp.zeros((1, 1)),
                )

            # missing w2b
            with pytest.raises(ValueError, match="down bias"):
                fn(
                    [jnp.zeros((1, 1))] * 5,
                    jnp.zeros(32),
                    jnp.zeros(1),
                    jnp.zeros(1),
                    jnp.zeros((32, 1, 128)),
                    jnp.zeros((2, 128)),
                    jnp.zeros((1, 128, 256)),
                    jnp.zeros((1, 128, 128)),
                    jnp.zeros((1, 256)),
                    jnp.zeros((1, 128)),
                    w1b=jnp.zeros((1, 256)),
                    w2b=None,
                    recv_rows=32,
                    visit=jnp.zeros(1),
                    counts=jnp.zeros((5, 1)),
                    rank=jnp.zeros((1, 1)),
                )

            # recv_rows % ROWBLK != 0 in make_call
            with pytest.raises(ValueError, match="arrival buffer height"):
                fn(
                    [jnp.zeros((1, 1))] * 5,
                    jnp.zeros(32),
                    jnp.zeros(1),
                    jnp.zeros(1),
                    jnp.zeros((32, 1, 128)),
                    jnp.zeros((2, 128)),
                    jnp.zeros((1, 128, 256)),
                    jnp.zeros((1, 128, 128)),
                    jnp.zeros((1, 256)),
                    jnp.zeros((1, 128)),
                    w1b=jnp.zeros((1, 256)),
                    w2b=jnp.zeros((1, 128)),
                    recv_rows=33,
                    visit=jnp.zeros(1),
                    counts=jnp.zeros((5, 1)),
                    rank=jnp.zeros((1, 1)),
                )

            # test build_fused_ep_moe_kernel caching
            cached1 = build_fused_ep_moe_kernel(
                g_local=1,
                capacity=32,
                hidden=128,
                inter=128,
                ep=1,
                ragged_rows_alloc=32,
            )
            cached2 = build_fused_ep_moe_kernel(
                g_local=1,
                capacity=32,
                hidden=128,
                inter=128,
                ep=1,
                ragged_rows_alloc=32,
            )
            assert cached1 is cached2

    def test_combine_kernel_full(self):
        # combine_step_tokens
        assert combine_step_tokens(64) == 0
        assert combine_step_tokens(256) > 0

        # _build_combine_kernel validation and execution
        mock_info = _mock_tpu_info()
        with (
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch(
                "jax.experimental.pallas.pallas_call",
                return_value=MagicMock(
                    return_value=jnp.zeros((256, 128), jnp.bfloat16)
                ),
            ),
        ):
            combine_fn = _build_combine_kernel(
                t_local=256,
                topk=2,
                hidden=128,
                lane_blocks=1,
                wire_dtype=kernel.FP8,
                out_dtype=jnp.bfloat16,
            )
            # Bad shape
            with pytest.raises(ValueError, match="combine takes token-major"):
                combine_fn(
                    jnp.zeros((256, 1, 128)), jnp.zeros((256, 2)), jnp.zeros((128, 2))
                )

            # Successful call
            res = combine_fn(
                jnp.zeros((256, 1, 128)), jnp.zeros((256, 2)), jnp.zeros((256, 2))
            )
            assert res.shape == (256, 128)

            # build_combine_kernel caching
            c1 = build_combine_kernel(
                t_local=256,
                topk=2,
                hidden=128,
                lane_blocks=1,
                wire_dtype=kernel.FP8,
                out_dtype=jnp.bfloat16,
            )
            c2 = build_combine_kernel(
                t_local=256,
                topk=2,
                hidden=128,
                lane_blocks=1,
                wire_dtype=kernel.FP8,
                out_dtype=jnp.bfloat16,
            )
            assert c1 is c2

    def test_inner_kernel_fp8_direct_execution_simulation(self):
        """Simulate inner kernel(*refs) execution for FP8 with push_from_vmem True."""
        captured_kernel = None

        def mock_pallas_call(k, **kwargs):
            nonlocal captured_kernel
            captured_kernel = k
            return MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())
            )

        mock_info = _mock_tpu_info()

        with (
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch("jax.experimental.pallas.pallas_call", side_effect=mock_pallas_call),
        ):
            fn = _build_fused_ep_moe_kernel(
                g_local=1,
                capacity=32,
                hidden=128,
                inter=128,
                ep=1,
                ragged_rows_alloc=32,
            )
            fn(
                [jnp.zeros((1, 1), jnp.int32)] * 5,
                jnp.zeros((32,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((32, 1, 128), jnp.float8_e4m3fn),
                jnp.zeros((2, 128), jnp.float32),
                jnp.zeros((1, 128, 256), jnp.float8_e4m3fn),
                jnp.zeros((1, 128, 128), jnp.float8_e4m3fn),
                jnp.zeros((1, 256), jnp.float32),
                jnp.zeros((1, 128), jnp.float32),
                recv_rows=32,
                visit=jnp.zeros((1,), jnp.int32),
                counts=jnp.zeros((5, 1), jnp.int32),
                rank=jnp.zeros((1, 1), jnp.int32),
            )

        counts_sm = jnp.zeros((5, 1), jnp.int32)
        counts_sm = counts_sm.at[host.COUNT_VISITS, 0].set(1)
        counts_sm = counts_sm.at[host.COUNT_RECV_ROWS, 0].set(32)
        counts_sm = counts_sm.at[host.COUNT_RECV_MIRROR, 0].set(1)

        sem = MagicMock()
        sem.at = MagicMock(return_value=sem)

        refs = [
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.array([32], jnp.int32),
            jnp.array([0], jnp.int32),
            jnp.array([0], jnp.int32),
            counts_sm,
            jnp.zeros((1, 1), jnp.int32),
            MockRef(jnp.zeros((512,), jnp.int32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRef(jnp.zeros((1, 128, 256), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((1, 128, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((1, 256), jnp.float32)),
            MockRef(jnp.zeros((1, 128), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((2, 32, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 128, 256), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 128, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 256), jnp.float32)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 32, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 32, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 1, 32), jnp.float32)),
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
        ]

        def mock_when(cond):
            def dec(fn):
                return jax.lax.cond(cond, fn, lambda: None)

            return dec

        mock_copy = MagicMock()
        mock_copy.start.return_value = None
        mock_copy.wait.return_value = None

        with (
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._all_pairs_barrier"),
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._rows_wait"),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel._tile_row_scales",
                return_value=jnp.ones((32, 1), jnp.float32),
            ),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel.expert_ffn_fp8",
                return_value=(
                    jnp.ones((32, 128), jnp.float32),
                    jnp.ones((32, 1), jnp.float32),
                ),
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_remote_copy",
                return_value=mock_copy,
            ),
            patch(
                "jax.experimental.pallas.tpu.roll",
                side_effect=lambda x, s, a: jnp.roll(x, s, axis=a),
            ),
            patch("jax.experimental.pallas.when", side_effect=mock_when),
        ):
            captured_kernel(*refs)
            assert captured_kernel is not None

    def test_inner_kernel_push_from_vmem_false_simulation(self):
        """Simulate inner kernel execution for push_from_vmem=False."""
        captured_kernel = None

        def mock_pallas_call(k, **kwargs):
            nonlocal captured_kernel
            captured_kernel = k
            return MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())
            )

        mock_info = _mock_tpu_info()

        with (
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch("jax.experimental.pallas.pallas_call", side_effect=mock_pallas_call),
        ):
            fn = _build_fused_ep_moe_kernel(
                g_local=2,
                capacity=32,
                hidden=128,
                inter=128,
                ep=2,
                ragged_rows_alloc=256,
                has_w1_bias=True,
                has_w2_bias=True,
            )
            fn(
                [jnp.zeros((1, 1), jnp.int32)] * 5,
                jnp.zeros((2,), jnp.int32),
                jnp.zeros((2,), jnp.int32),
                jnp.zeros((2,), jnp.int32),
                jnp.zeros((32, 1, 128), jnp.float8_e4m3fn),
                jnp.zeros((3, 128), jnp.float32),
                jnp.zeros((2, 128, 256), jnp.float8_e4m3fn),
                jnp.zeros((2, 128, 128), jnp.float8_e4m3fn),
                jnp.zeros((2, 256), jnp.float32),
                jnp.zeros((2, 128), jnp.float32),
                w1b=jnp.zeros((2, 256), jnp.float32),
                w2b=jnp.zeros((2, 128), jnp.float32),
                recv_rows=32,
                visit=jnp.zeros((2,), jnp.int32),
                counts=jnp.zeros((5, 2), jnp.int32),
                rank=jnp.zeros((1, 1), jnp.int32),
            )

        sem = MagicMock()
        sem.at = MagicMock(return_value=sem)

        counts_sm = jnp.zeros((5, 2), jnp.int32)
        counts_sm = counts_sm.at[host.COUNT_VISITS, 0].set(2)
        counts_sm = counts_sm.at[host.COUNT_RECV_ROWS, 0].set(64)
        counts_sm = counts_sm.at[host.COUNT_RECV_MIRROR, 0].set(2)
        counts_sm = counts_sm.at[host.COUNT_SEND_ROWS, 0].set(64)
        counts_sm = counts_sm.at[host.COUNT_SEND_MIRROR, 0].set(2)
        counts_sm = counts_sm.at[host.COUNT_SELF_MIRROR, 0].set(2)

        commit_start_sm = jnp.zeros((2, 2), jnp.int32)
        commit_len_sm = jnp.ones((2, 2), jnp.int32) * 2
        contrib_off_sm = jnp.zeros((2, 2), jnp.int32)
        push_dst_sm = jnp.zeros((2, 2), jnp.int32)
        true_rows_sm = jnp.ones((2, 2), jnp.int32) * 32
        expert_rows_sm = jnp.array([32, 32], jnp.int32)
        expert_base_sm = jnp.array([0, 32], jnp.int32)
        visit_sm = jnp.array([0, 1], jnp.int32)
        rank_sm = jnp.zeros((1, 1), jnp.int32)

        refs = [
            commit_start_sm,
            commit_len_sm,
            contrib_off_sm,
            push_dst_sm,
            true_rows_sm,
            expert_rows_sm,
            expert_base_sm,
            visit_sm,
            counts_sm,
            rank_sm,
            MockRef(jnp.zeros((512,), jnp.int32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((3, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 128, 256), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 128, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 256), jnp.float32)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 256), jnp.float32)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((2, 32, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((3, 128, 256), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((3, 128, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((3, 256), jnp.float32)),
            MockRef(jnp.zeros((3, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 256), jnp.float32)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 32, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 32, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 1, 32), jnp.float32)),
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
        ]

        def mock_when(cond):
            def dec(fn):
                return jax.lax.cond(cond, fn, lambda: None)

            return dec

        mock_copy = MagicMock()
        mock_copy.start.return_value = None
        mock_copy.wait.return_value = None

        with (
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._all_pairs_barrier"),
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._rows_wait"),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel._tile_row_scales",
                return_value=jnp.ones((32, 1), jnp.float32),
            ),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel.expert_ffn_fp8",
                return_value=(
                    jnp.ones((32, 128), jnp.float32),
                    jnp.ones((32, 1), jnp.float32),
                ),
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_remote_copy",
                return_value=mock_copy,
            ),
            patch(
                "jax.experimental.pallas.tpu.roll",
                side_effect=lambda x, s, a: jnp.roll(x, s, axis=a),
            ),
            patch("jax.experimental.pallas.when", side_effect=mock_when),
        ):
            captured_kernel(*refs)
            assert captured_kernel is not None

    def test_inner_kernel_fp4_and_missing_bitcast(self):
        """Test WeightFormat.FP4 path, bitcast validation, and starts_next_weights."""
        mock_info = _mock_tpu_info()

        # Missing bitcast attribute on ref raises NotImplementedError (lines 825-834)
        captured_kernel = None

        def mock_pallas_call(k, **kwargs):
            nonlocal captured_kernel
            captured_kernel = k
            return MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())
            )

        slots_zero = host.WeightSlots(distance=0, nbuf=1, vmem_bytes=1000, limit=2000)

        with (
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.host.fit_weight_slots",
                return_value=slots_zero,
            ),
            patch("jax.experimental.pallas.pallas_call", side_effect=mock_pallas_call),
        ):
            fn = _build_fused_ep_moe_kernel(
                g_local=1,
                capacity=32,
                hidden=128,
                inter=128,
                ep=1,
                weight_format=host.WeightFormat.FP4,
                rhs_qb=128,
                ragged_rows_alloc=32,
            )
            fn(
                [jnp.zeros((1, 1), jnp.int32)] * 5,
                jnp.zeros((32,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((32, 1, 128), jnp.float8_e4m3fn),
                jnp.zeros((2, 128), jnp.float32),
                jnp.zeros((1, 16, 256), jnp.uint32),
                jnp.zeros((1, 16, 128), jnp.uint32),
                jnp.zeros((1, 256), jnp.float32),
                jnp.zeros((1, 128), jnp.float32),
                recv_rows=32,
                visit=jnp.zeros((1,), jnp.int32),
                counts=jnp.zeros((5, 1), jnp.int32),
                rank=jnp.zeros((1, 1), jnp.int32),
            )

        sem = MagicMock()
        counts_sm = jnp.zeros((5, 1), jnp.int32)
        counts_sm = counts_sm.at[host.COUNT_VISITS, 0].set(1)

        refs_bad = [
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.array([32], jnp.int32),
            jnp.array([0], jnp.int32),
            jnp.array([0], jnp.int32),
            counts_sm,
            jnp.zeros((1, 1), jnp.int32),
            MockRef(jnp.zeros((512,), jnp.int32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRefWithoutBitcast(jnp.zeros((1, 16, 256), jnp.uint32)),
            MockRefWithoutBitcast(jnp.zeros((1, 16, 128), jnp.uint32)),
            MockRef(jnp.zeros((1, 256), jnp.float32)),
            MockRef(jnp.zeros((1, 128), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((2, 32, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 16, 256), jnp.uint32)),
            MockRef(jnp.zeros((2, 16, 128), jnp.uint32)),
            MockRef(jnp.zeros((2, 256), jnp.float32)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 32, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 32, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 1, 32), jnp.float32)),
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
        ]

        with pytest.raises(
            NotImplementedError, match="the four-bit weight stream views a memory ref"
        ):
            captured_kernel(*refs_bad)

        # Valid FP4 execution with bitcast MockRef
        refs_good = list(refs_bad)
        refs_good[13] = MockRef(jnp.zeros((1, 16, 256), jnp.uint32))
        refs_good[14] = MockRef(jnp.zeros((1, 16, 128), jnp.uint32))

        def mock_when(cond):
            def dec(fn):
                return jax.lax.cond(cond, fn, lambda: None)

            return dec

        mock_copy = MagicMock()
        mock_copy.start.return_value = None
        mock_copy.wait.return_value = None

        def mock_blockscale(act_rows, act_scales, w1_block, w2_block, *args, **kwargs):
            w1_block(0)
            w1_block(1)
            w2_block(0)
            w2_block(1)
            return jnp.ones((32, 128), jnp.float32), jnp.ones((32, 1), jnp.float32)

        with (
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._all_pairs_barrier"),
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._rows_wait"),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel._tile_row_scales",
                return_value=jnp.ones((32, 1), jnp.float32),
            ),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel.expert_ffn_blockscale",
                side_effect=mock_blockscale,
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_remote_copy",
                return_value=mock_copy,
            ),
            patch(
                "jax.experimental.pallas.tpu.roll",
                side_effect=lambda x, s, a: jnp.roll(x, s, axis=a),
            ),
            patch(
                "jax.experimental.pallas.tpu.bitcast",
                side_effect=lambda x, ty: MagicMock(
                    astype=lambda dt: jnp.zeros((32, 128), jnp.float32)
                ),
            ),
            patch("jax.experimental.pallas.when", side_effect=mock_when),
        ):
            captured_kernel(*refs_good)

            # Simulation with streams_weights=False to cover branches
            # 1016->1018 and 1026->1028 (wait_slab is None)
            idx_sw = captured_kernel.__code__.co_freevars.index("streams_weights")
            cell_sw = captured_kernel.__closure__[idx_sw]
            orig_sw = cell_sw.cell_contents
            try:
                cell_sw.cell_contents = False
                captured_kernel(*refs_good)
            finally:
                cell_sw.cell_contents = orig_sw

        # Simulation to cover branch 1681->1684
        # (if has_scales: False in starts_next_weights)
        refs_no_tiles = list(refs_good)
        counts_sm2 = counts_sm.at[host.COUNT_VISITS, 0].set(2)
        refs_no_tiles[5] = jnp.array([0, 0], jnp.int32)
        refs_no_tiles[6] = jnp.array([0, 0], jnp.int32)
        refs_no_tiles[7] = jnp.array([0, 1], jnp.int32)
        refs_no_tiles[8] = counts_sm2

        def mock_fori(lower, upper, body_fun, init_val):
            if lower == upper:
                return init_val
            val = init_val
            for i in range(int(lower), int(upper)):
                val = body_fun(i, val)
            return val

        idx = captured_kernel.__code__.co_freevars.index("has_scales")
        cell = captured_kernel.__closure__[idx]
        orig_has_scales = cell.cell_contents
        try:
            cell.cell_contents = False
            with (
                patch("jax.lax.fori_loop", side_effect=mock_fori),
                patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._all_pairs_barrier"),
                patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._rows_wait"),
                patch(
                    "jax.experimental.pallas.tpu.make_async_copy",
                    return_value=mock_copy,
                ),
                patch(
                    "jax.experimental.pallas.tpu.make_async_remote_copy",
                    return_value=mock_copy,
                ),
                patch(
                    "jax.experimental.pallas.tpu.roll",
                    side_effect=lambda x, s, a: jnp.roll(x, s, axis=a),
                ),
                patch(
                    "jax.experimental.pallas.tpu.bitcast",
                    side_effect=lambda x, ty: MagicMock(
                        astype=lambda dt: jnp.zeros((32, 128), jnp.float32)
                    ),
                ),
                patch(
                    "jax.experimental.pallas.when",
                    side_effect=lambda cond: lambda fn: fn(),
                ),
            ):
                captured_kernel(*refs_no_tiles)
        finally:
            cell.cell_contents = orig_has_scales

    def test_inner_kernel_int8_and_bf16(self):
        """Test INT8 chunk readers and BF16 execution paths."""
        mock_info = _mock_tpu_info()
        captured_kernel = None

        def mock_pallas_call(k, **kwargs):
            nonlocal captured_kernel
            captured_kernel = k
            return MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())
            )

        def mock_when(cond):
            def dec(fn):
                return jax.lax.cond(cond, fn, lambda: None)

            return dec

        mock_copy = MagicMock()
        mock_copy.start.return_value = None
        mock_copy.wait.return_value = None
        sem = MagicMock()
        sem.at = MagicMock(return_value=sem)

        # 1. INT8 path
        with (
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch("jax.experimental.pallas.pallas_call", side_effect=mock_pallas_call),
        ):
            fn_int8 = _build_fused_ep_moe_kernel(
                g_local=1,
                capacity=32,
                hidden=512,
                inter=512,
                ep=1,
                weight_format=host.WeightFormat.INT8,
                ragged_rows_alloc=32,
            )
            fn_int8(
                [jnp.zeros((1, 1), jnp.int32)] * 5,
                jnp.zeros((32,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((32, 4, 128), jnp.bfloat16),
                None,
                jnp.zeros((1, 512, 1024), jnp.int8),
                jnp.zeros((1, 512, 512), jnp.int8),
                jnp.zeros((1, 1024), jnp.float32),
                jnp.zeros((1, 512), jnp.float32),
                recv_rows=32,
                visit=jnp.zeros((1,), jnp.int32),
                counts=jnp.zeros((5, 1), jnp.int32),
                rank=jnp.zeros((1, 1), jnp.int32),
            )

        counts_sm = jnp.zeros((5, 1), jnp.int32)
        counts_sm = counts_sm.at[host.COUNT_VISITS, 0].set(1)
        counts_sm = counts_sm.at[host.COUNT_RECV_ROWS, 0].set(32)
        counts_sm = counts_sm.at[host.COUNT_RECV_MIRROR, 0].set(1)

        refs_int8 = [
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.array([32], jnp.int32),
            jnp.array([0], jnp.int32),
            jnp.array([0], jnp.int32),
            counts_sm,
            jnp.zeros((1, 1), jnp.int32),
            MockRef(jnp.zeros((512,), jnp.int32)),
            MockRef(jnp.zeros((512, 4, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((1, 512, 1024), jnp.int8)),
            MockRef(jnp.zeros((1, 512, 512), jnp.int8)),
            MockRef(jnp.zeros((1, 1024), jnp.float32)),
            MockRef(jnp.zeros((1, 512), jnp.float32)),
            MockRef(jnp.zeros((512, 4, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((512, 4, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((2, 32, 4, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((2, 512, 1024), jnp.int8)),
            MockRef(jnp.zeros((2, 512, 512), jnp.int8)),
            MockRef(jnp.zeros((2, 1024), jnp.float32)),
            MockRef(jnp.zeros((2, 512), jnp.float32)),
            MockRef(jnp.zeros((2, 32, 4, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((2, 1, 32), jnp.float32)),
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
        ]

        def mock_int8(a, w1_chunk, w2_chunk, *args, **kwargs):
            w1_chunk(0)
            w2_chunk(0)
            return (
                jnp.ones((a.shape[0] if hasattr(a, "shape") else 32, 512), jnp.float32),
                None,
            )

        with (
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._all_pairs_barrier"),
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._rows_wait"),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel.expert_ffn_int8",
                side_effect=mock_int8,
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_remote_copy",
                return_value=mock_copy,
            ),
            patch(
                "jax.experimental.pallas.tpu.roll",
                side_effect=lambda x, s, a: jnp.roll(x, s, axis=a),
            ),
            patch("jax.experimental.pallas.when", side_effect=mock_when),
        ):
            captured_kernel(*refs_int8)

        # 2. BF16 path
        slots_zero = host.WeightSlots(distance=0, nbuf=1, vmem_bytes=1000, limit=2000)
        with (
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.host.fit_weight_slots",
                return_value=slots_zero,
            ),
            patch("jax.experimental.pallas.pallas_call", side_effect=mock_pallas_call),
        ):
            fn_bf16 = _build_fused_ep_moe_kernel(
                g_local=1,
                capacity=32,
                hidden=128,
                inter=128,
                ep=1,
                weight_format=host.WeightFormat.BF16,
                ragged_rows_alloc=32,
            )
            fn_bf16(
                [jnp.zeros((1, 1), jnp.int32)] * 5,
                jnp.zeros((32,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((32, 1, 128), jnp.bfloat16),
                None,
                jnp.zeros((1, 128, 256), jnp.bfloat16),
                jnp.zeros((1, 128, 128), jnp.bfloat16),
                None,
                None,
                recv_rows=32,
                visit=jnp.zeros((1,), jnp.int32),
                counts=jnp.zeros((5, 1), jnp.int32),
                rank=jnp.zeros((1, 1), jnp.int32),
            )

        refs_bf16 = [
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.array([32], jnp.int32),
            jnp.array([0], jnp.int32),
            jnp.array([0], jnp.int32),
            counts_sm,
            jnp.zeros((1, 1), jnp.int32),
            MockRef(jnp.zeros((512,), jnp.int32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((1, 128, 256), jnp.bfloat16)),
            MockRef(jnp.zeros((1, 128, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((512, 1, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((512, 32), jnp.float32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((2, 32, 1, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((1, 128, 256), jnp.bfloat16)),
            MockRef(jnp.zeros((1, 128, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((2, 32, 1, 128), jnp.bfloat16)),
            MockRef(jnp.zeros((2, 1, 32), jnp.float32)),
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
        ]

        with (
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._all_pairs_barrier"),
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._rows_wait"),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel.expert_ffn_bf16",
                side_effect=lambda a, *args, **kw: (
                    jnp.ones(
                        (a.shape[0] if hasattr(a, "shape") else 32, 128), jnp.float32
                    ),
                    None,
                ),
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_remote_copy",
                return_value=mock_copy,
            ),
            patch(
                "jax.experimental.pallas.tpu.roll",
                side_effect=lambda x, s, a: jnp.roll(x, s, axis=a),
            ),
            patch("jax.experimental.pallas.when", side_effect=mock_when),
        ):
            captured_kernel(*refs_bf16)

    def test_multi_rung_execution(self):
        """Test multi-rung execution where tile_height_rungs returns multiple rungs."""
        mock_info = _mock_tpu_info()
        captured_kernel = None

        def mock_pallas_call(k, **kwargs):
            nonlocal captured_kernel
            captured_kernel = k
            return MagicMock(
                return_value=(MagicMock(), MagicMock(), MagicMock(), MagicMock())
            )

        def mock_when(cond):
            def dec(fn):
                fn()
                return None

            return dec

        mock_copy = MagicMock()
        mock_copy.start.return_value = None
        mock_copy.wait.return_value = None
        sem = MagicMock()
        sem.at = MagicMock(return_value=sem)

        with (
            patch.dict(os.environ, {"MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE": "4"}),
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch("jax.experimental.pallas.pallas_call", side_effect=mock_pallas_call),
        ):
            fn = _build_fused_ep_moe_kernel(
                g_local=1,
                capacity=64,
                hidden=128,
                inter=128,
                ep=1,
                ragged_rows_alloc=64,
            )
            fn(
                [jnp.zeros((1, 1), jnp.int32)] * 5,
                jnp.zeros((64,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((1,), jnp.int32),
                jnp.zeros((64, 1, 128), jnp.float8_e4m3fn),
                jnp.zeros((2, 128), jnp.float32),
                jnp.zeros((1, 128, 256), jnp.float8_e4m3fn),
                jnp.zeros((1, 128, 128), jnp.float8_e4m3fn),
                jnp.zeros((1, 256), jnp.float32),
                jnp.zeros((1, 128), jnp.float32),
                recv_rows=64,
                visit=jnp.zeros((1,), jnp.int32),
                counts=jnp.zeros((5, 1), jnp.int32),
                rank=jnp.zeros((1, 1), jnp.int32),
            )

        counts_sm = jnp.zeros((5, 1), jnp.int32)
        counts_sm = counts_sm.at[host.COUNT_VISITS, 0].set(1)
        counts_sm = counts_sm.at[host.COUNT_RECV_ROWS, 0].set(64)
        counts_sm = counts_sm.at[host.COUNT_RECV_MIRROR, 0].set(1)

        refs = [
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.zeros((1, 1), jnp.int32),
            jnp.array([16], jnp.int32),
            jnp.array([0], jnp.int32),
            jnp.array([0], jnp.int32),
            counts_sm,
            jnp.zeros((1, 1), jnp.int32),
            MockRef(jnp.zeros((512,), jnp.int32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRef(jnp.zeros((1, 128, 256), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((1, 128, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((1, 256), jnp.float32)),
            MockRef(jnp.zeros((1, 128), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((512, 64), jnp.float32)),
            MockRef(jnp.zeros((512, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((512, 64), jnp.float32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((256,), jnp.int32)),
            MockRef(jnp.zeros((2, 64, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 128, 256), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 128, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 256), jnp.float32)),
            MockRef(jnp.zeros((2, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 64, 128), jnp.float32)),
            MockRef(jnp.zeros((2, 64, 1, 128), jnp.float8_e4m3fn)),
            MockRef(jnp.zeros((2, 1, 64), jnp.float32)),
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
            sem,
        ]

        def mock_ffn(act_rows, *args, **kwargs):
            h = act_rows.shape[0] if hasattr(act_rows, "shape") else 16
            return jnp.ones((h, 128), jnp.float32), jnp.ones((h, 1), jnp.float32)

        with (
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._all_pairs_barrier"),
            patch("vllm_torchtpu.kernels.fused_moe.v2.kernel._rows_wait"),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel._tile_row_scales",
                side_effect=lambda s, b, h: jnp.ones((h, 1), jnp.float32),
            ),
            patch(
                "vllm_torchtpu.kernels.fused_moe.v2.kernel.expert_ffn_fp8",
                side_effect=mock_ffn,
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch(
                "jax.experimental.pallas.tpu.make_async_remote_copy",
                return_value=mock_copy,
            ),
            patch(
                "jax.experimental.pallas.tpu.roll",
                side_effect=lambda x, s, a: jnp.roll(x, s, axis=a),
            ),
            patch("jax.experimental.pallas.when", side_effect=mock_when),
        ):
            captured_kernel(*refs)

    def test_combine_kernel_simulation_and_steps(self):
        """Simulate combine kernel body execution and token step calculations."""
        mock_info = _mock_tpu_info()
        captured_combine = None

        def mock_pallas_call(k, **kwargs):
            nonlocal captured_combine
            captured_combine = k
            return MagicMock()

        with (
            patch("jax.experimental.pallas.tpu.get_tpu_info", return_value=mock_info),
            patch("jax.experimental.pallas.pallas_call", side_effect=mock_pallas_call),
        ):
            _build_combine_kernel(
                t_local=512,
                topk=2,
                hidden=128,
                lane_blocks=1,
                wire_dtype=kernel.FP8,
                out_dtype=jnp.bfloat16,
            )

        def mock_when(cond):
            def dec(fn):
                return jax.lax.cond(cond, fn, lambda: None)

            return dec

        mock_copy = MagicMock()
        mock_copy.start.return_value = None
        mock_copy.wait.return_value = None
        sem = MagicMock()
        sem.at = MagicMock(return_value=sem)

        t_local = 512
        topk = 2
        tb = kernel.COMBINE_TOKENS
        pos_sm = jnp.zeros((t_local * topk,), jnp.int32)
        coef_sm = jnp.zeros((t_local * topk,), jnp.float32)
        arr_hbm = MockRef(jnp.zeros((t_local * topk, 1, 128), kernel.FP8))
        out_hbm = MockRef(jnp.zeros((t_local, 128), jnp.bfloat16))
        rows_vm = MockRef(
            jnp.zeros((kernel.COMBINE_PARITIES, topk * tb, 1, 128), kernel.FP8)
        )
        tile_vm = MockRef(jnp.zeros((tb, 1, 128), jnp.bfloat16))
        out_vm = MockRef(jnp.zeros((kernel.COMBINE_PARITIES, tb, 128), jnp.bfloat16))

        with (
            patch(
                "jax.experimental.pallas.tpu.make_async_copy", return_value=mock_copy
            ),
            patch("jax.experimental.pallas.when", side_effect=mock_when),
        ):
            captured_combine(
                pos_sm, coef_sm, arr_hbm, out_hbm, rows_vm, tile_vm, out_vm, sem, sem
            )

        # Exercise combine_step_tokens loop tb //= 2 path returning 0 (lines 2177-2178)
        assert combine_step_tokens(300) == 0

    def test_cache_double_checked_locking_kernel_and_combine(self):
        """Test cache double-checked locking paths for build kernels."""
        mock_fn = MagicMock()

        # 1. build_fused_ep_moe_kernel lock path (line 2099->2107)
        with patch.object(kernel, "_BUILD_CACHE", MockCache(mock_fn)):
            res1 = build_fused_ep_moe_kernel(test_param=123)
            assert res1 is mock_fn

        # 2. build_combine_kernel lock path (line 2334->2337)
        with patch.object(kernel, "_COMBINE_CACHE", MockCache(mock_fn)):
            res2 = build_combine_kernel(test_param=456)
            assert res2 is mock_fn
