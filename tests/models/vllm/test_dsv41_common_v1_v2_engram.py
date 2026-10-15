# SPDX-License-Identifier: Apache-2.0
"""100% Line and branch coverage test suite for DeepSeek-V4.1 Engram.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.common.engram
- vllm_torchtpu.models.vllm.deepseek_v41.v1.engram
- vllm_torchtpu.models.vllm.deepseek_v41.v2.engram
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v1 import engram as v1_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram


@pytest.fixture(autouse=True)
def setup_torch_accelerator(monkeypatch: pytest.MonkeyPatch):
    """Enable CPU execution for upstream Engram constructor checks."""
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda: 0, raising=False
    )


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


# ==============================================================================
# 1. common.NgramHashState tests (lines 21-209)
# ==============================================================================


class TestCommonNgramHashState:
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

    def test_common_ensure_cache_eager_block_size_resize_and_match(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        state = _build_common_hash_state(use_slot_cache=True, block_size=8)
        # kv_cache has shape (4, 16) -> block_size = 16 != self.block_size (8)
        state.swa_cache_module.kv_cache = torch.zeros(4, 16)

        super_called = []
        monkeypatch.setattr(
            common_engram.upstream.NgramHashState,
            "ensure_cache",
            lambda self: super_called.append(True) or True,
        )

        # Mismatch branch: line 51 True -> updates block_size, resets _kv_cache_ref
        res = state.ensure_cache()
        assert res is True
        assert state.block_size == 16
        assert state._kv_cache_ref is None
        assert len(super_called) == 1

        # Match branch: line 51 False -> block_size already 16
        res2 = state.ensure_cache()
        assert res2 is True
        assert state.block_size == 16
        assert len(super_called) == 2

    def test_common_ensure_cache_eager_no_slot_cache_and_empty_kv(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        super_called = []
        monkeypatch.setattr(
            common_engram.upstream.NgramHashState,
            "ensure_cache",
            lambda self: super_called.append(True) or False,
        )

        # 1. use_slot_cache is False
        state_no_slot = _build_common_hash_state(use_slot_cache=False, block_size=8)
        state_no_slot.swa_cache_module.kv_cache = torch.zeros(4, 16)
        assert state_no_slot.ensure_cache() is False
        assert state_no_slot.block_size == 8  # Not modified

        # 2. kv_cache is empty
        state_empty_kv = _build_common_hash_state(use_slot_cache=True, block_size=8)
        state_empty_kv.swa_cache_module.kv_cache = torch.empty(0)
        assert state_empty_kv.ensure_cache() is False
        assert state_empty_kv.block_size == 8  # Not modified
        assert len(super_called) == 2

    def test_common_forward_empty_input_ids(self):
        state = _build_common_hash_state()
        input_ids = torch.empty(0, dtype=torch.int32)
        pos = torch.empty(0, dtype=torch.int64)
        qsl = torch.tensor([0], dtype=torch.int32)
        dead = torch.empty(0, dtype=torch.bool)
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)

        out = state.forward(
            input_ids, pos, qsl, dead, lookback, dead_lb, None, torch.zeros(1, 4)
        )
        layers, ngram = state.multipliers.shape
        expected_shape = (0, layers, (ngram - 1) * state.primes.shape[-1])
        assert out.shape == expected_shape
        assert out.dtype == torch.int32

    def test_common_forward_slot_mapping_none_and_not_none(self):
        class DummyState(common_engram.NgramHashState):
            def __init__(self, use_slot_cache=True):
                super().__init__()
                self.written = []

            def write_cache(self, cache, slots, values, valid):
                self.written.append((cache, slots, values, valid))

        # 1. slot_mapping is not None and use_slot_cache is True
        state = _build_common_hash_state(use_slot_cache=True, subclass=DummyState)
        state.written = []
        ids = torch.tensor([1, 2], dtype=torch.int32)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        qsl = torch.tensor([0, 2], dtype=torch.int32)
        dead = torch.zeros(2, dtype=torch.bool)
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)
        slots = torch.tensor(
            [10, -1, 99], dtype=torch.int64
        )  # > count to test slicing [:count]
        block_table = torch.zeros((1, 4), dtype=torch.int32)

        out = state.forward(ids, pos, qsl, dead, lookback, dead_lb, slots, block_table)
        assert out.shape == (2, 2, 8)
        assert len(state.written) == 1
        assert torch.equal(state.written[0][1], slots[:2])

        # 2. slot_mapping is None and use_slot_cache is False
        state_no_slot = _build_common_hash_state(
            use_slot_cache=False, subclass=DummyState
        )
        state_no_slot.written = []
        out2 = state_no_slot.forward(
            ids, pos, qsl, dead, lookback, dead_lb, None, block_table
        )
        assert out2.shape == (2, 2, 8)
        assert len(state_no_slot.written) == 0

    def test_common_forward_fallback_without_and_with_slot_cache(self):
        # Without slot cache: fallback returns full_like(window, self.pad_id)
        state_no_cache = _build_common_hash_state(use_slot_cache=False)
        ids = torch.tensor([1, 2], dtype=torch.int32)
        pos = torch.tensor([10, 11], dtype=torch.int64)
        qsl = torch.tensor([0, 2], dtype=torch.int32)
        dead = torch.zeros(2, dtype=torch.bool)
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)
        block_table = torch.zeros((1, 4), dtype=torch.int32)

        out_no_cache = state_no_cache.forward(
            ids, pos, qsl, dead, lookback, dead_lb, None, block_table
        )
        assert out_no_cache.shape == (2, 2, 8)

        # With slot cache: fallback returns self._cache[slots]
        class WriteMockState(common_engram.NgramHashState):
            def write_cache(self, cache, slots, values, valid):
                pass

        state_with_cache = _build_common_hash_state(
            use_slot_cache=True, subclass=WriteMockState
        )
        slots = torch.tensor([0, 1], dtype=torch.int64)
        out_with_cache = state_with_cache.forward(
            ids, pos, qsl, dead, lookback, dead_lb, slots, block_table
        )
        assert out_with_cache.shape == (2, 2, 8)

    def test_common_hash_on_host_empty_tokens(self):
        state = _build_common_hash_state()
        ids = torch.empty(0, dtype=torch.int32)
        pos = torch.empty(0, dtype=torch.int64)
        qsl = torch.tensor([0], dtype=torch.int32)
        dead = torch.empty(0, dtype=torch.bool)
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)

        res = state.hash_on_host(ids, pos, qsl, dead, lookback, dead_lb)
        layers, ngram = state.multipliers.shape
        assert res.shape == (0, layers, (ngram - 1) * state.primes.shape[-1])
        assert res.dtype == torch.int32

    def test_common_hash_on_host_caching_and_execution(self):
        state = _build_common_hash_state()
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

    def test_common_hash_dead_mask_and_window_branches(self):
        state = _build_common_hash_state()
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


# ==============================================================================
# 2. ParallelEngramEmbedding and helpers tests (lines 210-391)
# ==============================================================================


class TestParallelEngramEmbedding:
    def test_load_local_rows_dtypes_and_shape_mismatch(self):
        # 1. uint8 param
        param_u8 = torch.zeros(10, 4, dtype=torch.uint8)
        loaded_u8 = torch.arange(40, dtype=torch.uint8).reshape(10, 4)
        common_engram._load_local_rows(param_u8, loaded_u8)
        assert torch.equal(param_u8, loaded_u8)

        # 2. float8_e8m0fnu loaded weight (param.dtype != uint8 to test
        # the second branch of or)
        param_f32 = torch.zeros(2, 2, dtype=torch.float32)
        loaded_e8 = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        common_engram._load_local_rows(param_f32, loaded_e8)
        assert torch.equal(param_f32, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        # 3. float dtype conversion
        param_bf16 = torch.zeros(4, 4, dtype=torch.bfloat16)
        loaded_f32 = torch.full((4, 4), 3.5, dtype=torch.float32)
        common_engram._load_local_rows(param_bf16, loaded_f32)
        assert torch.equal(param_bf16, loaded_f32.to(torch.bfloat16))

        # 4. Shape mismatch assertion
        param_mismatch = torch.zeros(4, 4, dtype=torch.uint8)
        loaded_mismatch = torch.zeros(5, 4, dtype=torch.uint8)
        with pytest.raises(AssertionError, match="does not fit param"):
            common_engram._load_local_rows(param_mismatch, loaded_mismatch)

    def test_weight_loader_branches(self, monkeypatch: pytest.MonkeyPatch):
        # 1. Shape match on CPU -> calls _load_local_rows
        param = nn.Parameter(
            torch.zeros(6, 8, dtype=torch.bfloat16), requires_grad=False
        )
        loaded = torch.full((6, 8), 3.0, dtype=torch.float32)
        common_engram.ParallelEngramEmbedding._weight_loader(param, loaded)
        assert torch.equal(param.data, loaded.to(torch.bfloat16))

        # 2. Shape mismatch on CPU -> calls upstream loader
        mock_upstream_loader = MagicMock()
        monkeypatch.setattr(
            common_engram.upstream.ParallelEngramEmbedding,
            "_weight_loader",
            mock_upstream_loader,
        )
        loaded_diff = torch.full((12, 8), 1.0, dtype=torch.float32)
        common_engram.ParallelEngramEmbedding._weight_loader(param, loaded_diff)
        mock_upstream_loader.assert_called_once_with(param, loaded_diff)

        # 3. TPU device path -> uses ExpertParamStager
        mock_stager = MagicMock()
        mock_stager_cls = MagicMock(return_value=mock_stager)
        host_tensor = torch.zeros(4, 4, dtype=torch.bfloat16)
        mock_stager.host_param.return_value = host_tensor
        monkeypatch.setattr(
            "vllm_torchtpu.model_loader_patches.ExpertParamStager", mock_stager_cls
        )

        class MockTPUParam:
            def __init__(self):
                self.shape = (4, 4)
                self.dtype = torch.bfloat16
                self.device = SimpleNamespace(type="tpu")

        tpu_param = MockTPUParam()
        loaded_tpu = torch.full((4, 4), 7.0, dtype=torch.bfloat16)
        common_engram.ParallelEngramEmbedding._weight_loader(tpu_param, loaded_tpu)
        mock_stager_cls.assert_called_once()
        mock_stager.host_param.assert_called_once_with(tpu_param)
        mock_stager.flush.assert_called_once_with(tpu_param)
        assert torch.equal(host_tensor, loaded_tpu)

    @pytest.mark.parametrize(
        ("env_setting", "init_arg", "expected_offload"),
        [
            (None, None, False),
            ("0", None, False),
            ("1", None, True),
            ("0", True, True),
            ("1", False, False),
        ],
    )
    def test_init_cpu_offload_and_weight_scale_inv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_setting: str | None,
        init_arg: bool | None,
        expected_offload: bool,
    ):
        if env_setting is None:
            monkeypatch.delenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", raising=False)
        else:
            monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", env_setting)

        monkeypatch.setattr(
            common_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
        )
        monkeypatch.setattr(
            common_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0
        )

        kwargs = {}
        if init_arg is not None:
            kwargs["cpu_offload"] = init_arg

        emb = common_engram.ParallelEngramEmbedding(100, 32, (10, 10), **kwargs)
        assert emb.host_offload is expected_offload
        assert getattr(emb.weight, "dummy_weight_value", None) == 1.0
        if emb.weight_scale_inv is not None:
            assert getattr(emb.weight_scale_inv, "dummy_weight_value", None) == 127

    def test_allocate_weights_cpu_and_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            common_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
        )
        monkeypatch.setattr(
            common_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0
        )

        # Offload True
        emb_cpu = common_engram.ParallelEngramEmbedding(
            64, 32, (8, 8), cpu_offload=True
        )
        w, s = emb_cpu._allocate_weights()
        assert w.device.type == "cpu"
        assert w.dtype == torch.float8_e4m3fn
        assert s.device.type == "cpu"
        assert s.dtype == torch.uint8

        # Offload False (delegates to super)
        emb_def = common_engram.ParallelEngramEmbedding(
            64, 32, (8, 8), cpu_offload=False
        )
        w2, s2 = emb_def._allocate_weights()
        assert w2 is not None

    def test_init_weight_scale_inv_none(self, monkeypatch: pytest.MonkeyPatch):
        def fake_upstream_init(self, *args, **kwargs):
            nn.Module.__init__(self)
            self.weight = nn.Parameter(torch.empty(1))
            self.weight_scale_inv = None
            self.vocab_start_idx = 0
            self.part_num_embeddings = 1

        monkeypatch.setattr(
            common_engram.upstream.ParallelEngramEmbedding,
            "__init__",
            fake_upstream_init,
        )
        emb = common_engram.ParallelEngramEmbedding(64, 32, (8, 8))
        assert emb.weight_scale_inv is None

    def test_lookup_device_transfer_and_computation(self):
        emb = object.__new__(common_engram.ParallelEngramEmbedding)
        emb.head_start = 0
        emb.part_n_hash_cols = 2
        emb.n_hash_cols = 4
        emb.vocab_start_idx = 0
        emb.vocab_end_idx = 100
        emb.part_num_embeddings = 100
        emb.dim = 16
        emb.block_size = 8

        # Storage
        weight = torch.ones((100, 16), dtype=torch.float8_e4m3fn)
        scales = torch.full((100, 2), 127, dtype=torch.uint8)  # 2^(127-127) = 1.0
        emb._storage = lambda: (weight, scales)

        # 1. Matching device
        indices = torch.tensor([[10, 20, 200, 300]], dtype=torch.int32)
        out = torch.zeros(1, 2, 16, dtype=torch.bfloat16)
        emb.lookup(indices, out)
        assert torch.all(out == 1.0)

        # 2. Device mismatch: tests indices.to(work_device) branch (line 286)
        mock_indices = MagicMock(
            wraps=torch.tensor([[10, 20, 200, 300]], dtype=torch.int32)
        )
        mock_indices.device = torch.device("meta")
        emb.lookup(mock_indices, out)
        mock_indices.to.assert_called_once_with(weight.device)

    def test_gather_raw_device_transfer_and_computation(self):
        emb = object.__new__(common_engram.ParallelEngramEmbedding)
        emb.head_start = 0
        emb.part_n_hash_cols = 2
        emb.n_hash_cols = 4
        emb.vocab_start_idx = 0
        emb.vocab_end_idx = 100
        emb.part_num_embeddings = 100
        emb.dim = 32
        emb.block_size = 16

        weight = torch.ones((100, 32), dtype=torch.uint8)
        scale = torch.full((100, 2), 127, dtype=torch.uint8)
        emb._storage = lambda: (weight, scale)

        indices = torch.tensor([[10, 20, 200, 300]], dtype=torch.int32)
        raw_out = torch.zeros((1, 2, 32), dtype=torch.uint8)
        scale_out = torch.zeros((1, 2, 2), dtype=torch.uint8)

        # Valid heads
        emb.gather_raw(indices, raw_out, scale_out)
        assert torch.all(raw_out[0, 0] == 1)
        assert torch.all(raw_out[0, 1] == 1)
        assert torch.all(scale_out == 127)

        # Invalid indices (masked with _INVALID_SCALE)
        invalid_indices = torch.tensor([[200, 300, 400, 500]], dtype=torch.int32)
        emb.gather_raw(invalid_indices, raw_out, scale_out)
        assert torch.all(scale_out == common_engram._INVALID_SCALE)

        # Device mismatch: tests indices.to(work_device) branch (line 319)
        mock_indices = MagicMock(
            wraps=torch.tensor([[10, 20, 200, 300]], dtype=torch.int32)
        )
        mock_indices.device = torch.device("meta")
        emb.gather_raw(mock_indices, raw_out, scale_out)
        mock_indices.to.assert_called_once_with(weight.device)

    def test_dequant_fp8_rows_and_unpack_bytes(self):
        # 1. dequant_fp8_rows with valid scale (127 -> 1.0, 128 -> 2.0)
        raw_val = torch.tensor([1.0], dtype=torch.float8_e4m3fn).view(torch.uint8)
        raw = raw_val.repeat(4).unsqueeze(0)  # [1, 4]
        scale = torch.tensor([[127, 128]], dtype=torch.uint8)  # [1, 2]
        dequant = common_engram.dequant_fp8_rows(raw, scale, block_size=2)
        assert dequant.dtype == torch.bfloat16
        assert torch.isclose(
            dequant[0, :2].float(), torch.tensor([1.0, 1.0]), atol=1e-3
        ).all()
        assert torch.isclose(
            dequant[0, 2:].float(), torch.tensor([2.0, 2.0]), atol=1e-3
        ).all()

        # 2. _INVALID_SCALE (255) maps to +0.0
        scale_invalid = torch.tensor(
            [[common_engram._INVALID_SCALE, common_engram._INVALID_SCALE]],
            dtype=torch.uint8,
        )
        dequant_invalid = common_engram.dequant_fp8_rows(
            raw, scale_invalid, block_size=2
        )
        assert torch.all(dequant_invalid == 0.0)

        # 3. _unpack_bytes
        words = torch.tensor([0x04030201], dtype=torch.int32)
        unpacked = common_engram._unpack_bytes(words, num_bytes=4)
        assert torch.equal(unpacked, torch.tensor([1, 2, 3, 4], dtype=torch.uint8))
        unpacked_partial = common_engram._unpack_bytes(words, num_bytes=2)
        assert torch.equal(unpacked_partial, torch.tensor([1, 2], dtype=torch.uint8))

    def test_gather_engram_rows_padding_branches(self, monkeypatch: pytest.MonkeyPatch):
        def fake_all_gather(t, dim):
            return t

        monkeypatch.setattr(
            common_engram.upstream, "tensor_model_parallel_all_gather", fake_all_gather
        )

        # 1. width % 128 != 0 (e.g. 100) -> padding branch (lines 387-390)
        rows_unpadded = torch.ones(2, 3, 100, dtype=torch.bfloat16)
        res_padded = common_engram._gather_engram_rows(rows_unpadded, dim=-1)
        assert res_padded.shape == (2, 3, 100)
        assert torch.equal(res_padded, rows_unpadded)

        # 2. width % 128 == 0 (e.g. 128) -> no padding branch (line 390 else)
        rows_aligned = torch.ones(2, 3, 128, dtype=torch.bfloat16)
        res_aligned = common_engram._gather_engram_rows(rows_aligned, dim=-1)
        assert res_aligned.shape == (2, 3, 128)


# ==============================================================================
# 3. common.Engram module tests (lines 393-542)
# ==============================================================================


class TestEngramModule:
    def test_engram_create_embedding(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            common_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
        )
        monkeypatch.setattr(
            common_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0
        )

        registered_slices = []
        monkeypatch.setattr(
            common_engram, "register_row_slice", lambda *a: registered_slices.append(a)
        )

        layout = SimpleNamespace(
            num_embeddings=[64],
            head_dim=32,
            primes=[[(16, 16), (16, 16)]],
            layer_ids=[7],
        )

        engram = object.__new__(common_engram.Engram)
        emb = common_engram.Engram._create_embedding(engram, layout, 0)
        assert isinstance(emb, common_engram.ParallelEngramEmbedding)
        assert len(registered_slices) == 2
        assert registered_slices[0] == (
            "layers.7.engram.embed.weight",
            emb.vocab_start_idx,
            emb.part_num_embeddings,
        )
        assert registered_slices[1] == (
            "layers.7.engram.embed.scale",
            emb.vocab_start_idx,
            emb.part_num_embeddings,
        )

    def test_engram_staging_init_offload_true_and_false(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        engram = object.__new__(common_engram.Engram)
        mock_super_init = MagicMock()
        monkeypatch.setattr(
            common_engram.upstream.Engram, "_init_staging", mock_super_init
        )

        # 1. cpu_offload is False
        engram.embed_tokens = SimpleNamespace(host_offload=False)
        common_engram.Engram._init_staging(engram, 16, 32)
        mock_super_init.assert_called_once_with(16, 32)
        assert engram._staged_raw is None

        # 2. cpu_offload is True
        engram.embed_tokens = SimpleNamespace(
            host_offload=True, part_n_hash_cols=2, block_size=16, dp_size=1
        )
        common_engram.Engram._init_staging(engram, 16, 32)
        assert engram._staged_raw is not None
        assert engram._host_raw.shape == (16, 2, 32)

    def test_engram_prepare_embeddings_branches(self, monkeypatch: pytest.MonkeyPatch):
        cols, dim, block_size = 1, 16, 16
        max_tokens = 8

        def make_engram(cpu_offload: bool):
            engram = object.__new__(common_engram.Engram)
            mock_emb = MagicMock()
            mock_emb.host_offload = cpu_offload
            mock_emb.dp_size = 1
            mock_emb.part_n_hash_cols = cols
            mock_emb.block_size = block_size
            mock_emb.dim = dim
            mock_emb.gather_raw = lambda hids, r_out, s_out: (
                r_out.fill_(1),
                s_out.fill_(129),
            )
            engram.embed_tokens = mock_emb
            monkeypatch.setattr(
                common_engram.upstream.Engram, "_init_staging", lambda self, m, h: None
            )
            common_engram.Engram._init_staging(engram, max_tokens, dim)
            return engram

        engram = make_engram(cpu_offload=True)

        # 1. Partial tokens with scale_bytes % 4 != 0 (num_tokens=3, cols=1, blocks=1)
        engram._host_scale_buf.fill_(99)
        hash_ids_partial = torch.zeros(3, cols, dtype=torch.int32)
        engram.prepare_embeddings(hash_ids_partial)
        assert torch.all(engram._host_scale[:3] == 129)
        assert engram._host_scale_buf[3].item() == common_engram._INVALID_SCALE
        staged_raw_3 = engram._staged_tensors(3)
        assert staged_raw_3.shape == (3 * (cols * dim // 4 + 1),)
        assert engram._active_raw.shape == (3, cols * dim // 4 + 1)

        # Calling again with num_tokens=3 reuses the cached staging tensor and view
        engram.prepare_embeddings(hash_ids_partial)
        assert engram._staged_tensors(3) is staged_raw_3
        assert engram._staged_views(3) is engram._active_raw
        assert engram._staged_tensors(max_tokens) is engram._staged_raw

        # 1b. Single token (num_tokens=1 < 2 skips mark_dynamic)
        engram.prepare_embeddings(torch.zeros(1, cols, dtype=torch.int32))
        assert engram._active_raw.shape == (1, cols * dim // 4 + 1)

        # 2. Exact tokens: scale_bytes % 4 == 0 (skips tail word byte pad)
        hash_ids_full = torch.zeros(max_tokens, cols, dtype=torch.int32)
        engram.prepare_embeddings(hash_ids_full)
        assert torch.all(engram._host_scale == 129)

        # 3. Fallback: hash_ids tokens > max_tokens or cpu_offload is False
        fallback_calls = []
        monkeypatch.setattr(
            common_engram.upstream.Engram,
            "prepare_embeddings",
            lambda self, hids: fallback_calls.append(self),
        )
        with pytest.raises(ValueError, match="staging slots"):
            engram.prepare_embeddings(
                torch.zeros(max_tokens + 2, cols, dtype=torch.int32)
            )
        engram_no_offload = make_engram(cpu_offload=False)
        engram_no_offload.prepare_embeddings(hash_ids_partial)
        assert fallback_calls == [engram_no_offload]

    def test_engram_ready_rows_staged_and_unstaged(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        cols, dim, block_size = 2, 16, 8
        max_tokens = 4

        engram = object.__new__(common_engram.Engram)
        # 1. unstaged -> delegates to super
        engram._staged_raw = None
        super_ready = []
        monkeypatch.setattr(
            common_engram.upstream.Engram,
            "_ready_rows",
            lambda self, n: super_ready.append(n) or torch.zeros(n, cols, dim),
        )
        out_unstaged = engram._ready_rows(max_tokens)
        assert out_unstaged.shape == (max_tokens, cols, dim)
        assert super_ready == [max_tokens]

        # 2. staged -> unpacks and dequants
        mock_emb = MagicMock()
        mock_emb.host_offload = True
        mock_emb.dp_size = 1
        mock_emb.part_n_hash_cols = cols
        mock_emb.block_size = block_size
        mock_emb.dim = dim
        mock_emb.gather_raw = lambda hids, r_out, s_out: (
            r_out.fill_(0x38),  # 1.0 in float8_e4m3fn
            s_out.fill_(127),  # 1.0 multiplier
        )
        engram.embed_tokens = mock_emb
        common_engram.Engram._init_staging(engram, max_tokens, dim)
        engram.prepare_embeddings(torch.zeros(max_tokens, cols, dtype=torch.int32))
        rows = engram._ready_rows(max_tokens)
        assert rows.shape == (max_tokens, cols, dim)
        assert rows.dtype == torch.bfloat16
        assert torch.isclose(
            rows.float(), torch.ones_like(rows.float()), atol=1e-3
        ).all()

    @pytest.mark.parametrize("num_tokens", [6, 7])
    @pytest.mark.parametrize("tp_size", [1, 2])
    def test_engram_sp_embed_and_forward(
        self, monkeypatch: pytest.MonkeyPatch, num_tokens: int, tp_size: int
    ):
        cols, dim, hc_mult, rank = 2, 8, 2, 0
        monkeypatch.setattr(
            common_engram.upstream,
            "get_tensor_model_parallel_world_size",
            lambda: tp_size,
        )
        monkeypatch.setattr(
            common_engram.upstream, "get_tensor_model_parallel_rank", lambda: rank
        )

        chunk = -(-num_tokens // tp_size)
        engram = object.__new__(common_engram.Engram)
        nn.Module.__init__(engram)
        engram.use_sequence_parallel = True
        engram.hc_mult = hc_mult
        engram.dim = dim
        engram.eps = 1e-6
        engram.clamp_value = 1e-4

        mock_embed = MagicMock()
        mock_embed.tp_size = tp_size
        mock_embed.n_hash_cols = cols
        engram.embed_tokens = mock_embed

        staged = torch.randn(num_tokens, cols, dim, dtype=torch.bfloat16)
        engram._ready_rows = lambda n: staged[:n]

        def fake_gather(rows, dim=0):
            assert dim == 0
            return torch.cat([rows for _ in range(tp_size)], dim=0)

        monkeypatch.setattr(common_engram, "_gather_engram_rows", fake_gather)

        hash_ids = torch.zeros(num_tokens, cols, dtype=torch.int32)
        sp_out = engram._sp_embed(hash_ids)
        assert sp_out.shape == (chunk, cols, dim)

        engram.wkv = nn.Linear(
            cols * dim, (hc_mult + 1) * dim, bias=False, dtype=torch.bfloat16
        )
        engram.q_weight = nn.Parameter(torch.ones(hc_mult, dim, dtype=torch.bfloat16))
        engram.k_weight = nn.Parameter(torch.ones(hc_mult, dim, dtype=torch.bfloat16))

        hidden_states = torch.randn(chunk, hc_mult, dim, dtype=torch.bfloat16)
        token_mask = torch.ones(num_tokens, dtype=torch.bfloat16)

        # 1. SP=True with token_mask (lines 517-522)
        out_sp_mask = engram.forward(hidden_states, hash_ids, token_mask=token_mask)
        assert out_sp_mask.shape == (chunk, hc_mult, dim)

        # 2. SP=True without token_mask
        out_sp_nomask = engram.forward(hidden_states, hash_ids, token_mask=None)
        assert out_sp_nomask.shape == (chunk, hc_mult, dim)

        # 3. SP=False with and without token_mask (lines 525, 539-540)
        engram.use_sequence_parallel = False
        engram.embed = lambda hids: staged[: hids.shape[0]]
        hidden_full = torch.randn(num_tokens, hc_mult, dim, dtype=torch.bfloat16)

        out_nonsp_mask = engram.forward(hidden_full, hash_ids, token_mask=token_mask)
        assert out_nonsp_mask.shape == (num_tokens, hc_mult, dim)

        out_nonsp_nomask = engram.forward(hidden_full, hash_ids, token_mask=None)
        assert out_nonsp_nomask.shape == (num_tokens, hc_mult, dim)


# ==============================================================================
# 4. v1.engram tests (lines 1-53)
# ==============================================================================


class TestV1Engram:
    def test_v1_write_cache(self, monkeypatch: pytest.MonkeyPatch):
        mock_write = MagicMock()
        monkeypatch.setattr(v1_engram, "write_cache", mock_write)

        state = object.__new__(v1_engram.NgramHashState)
        cache = torch.zeros(10)
        slots = torch.zeros(2)
        values = torch.zeros(2)
        valid = torch.ones(2, dtype=torch.bool)

        state.write_cache(cache, slots, values, valid)
        mock_write.assert_called_once_with(cache, slots, values, valid)

    def test_v1_ensure_cache_compiling(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        super_called = []
        monkeypatch.setattr(
            common_engram.NgramHashState,
            "ensure_cache",
            lambda self: super_called.append(True) or True,
        )

        state = object.__new__(v1_engram.NgramHashState)
        assert state.ensure_cache() is True
        assert len(super_called) == 1

    def test_v1_ensure_cache_eager_empty_kv_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        state = object.__new__(v1_engram.NgramHashState)
        state._cache = torch.zeros(10)
        state._kv_cache_ref = "dummy_ref"
        state.swa_cache_module = SimpleNamespace(
            bound_cache=torch.empty(0),
            block_size=8,
        )

        # Empty bound_cache -> resets _cache and _kv_cache_ref,
        # returns False (lines 30-33)
        assert state.ensure_cache() is False
        assert state._cache is None
        assert state._kv_cache_ref is None

    def test_v1_ensure_cache_eager_no_slot_cache(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        state = object.__new__(v1_engram.NgramHashState)
        state.use_slot_cache = False
        state.swa_cache_module = SimpleNamespace(
            bound_cache=torch.ones(4, 8),
            block_size=8,
        )

        # not use_slot_cache -> returns True (lines 34-35)
        assert state.ensure_cache() is True

    def test_v1_ensure_cache_eager_block_size_resize_and_cache_hit_and_miss(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
        state = object.__new__(v1_engram.NgramHashState)
        state.use_slot_cache = True
        state.block_size = 8
        state._kv_cache_ref = None
        state._cache = None

        kv_cache = torch.zeros(4, 16)
        state.swa_cache_module = SimpleNamespace(
            bound_cache=kv_cache,
            block_size=16,
        )

        # 1. First call: block_size resized from 8 to 16, cache miss, allocates _cache
        assert state.ensure_cache() is True
        assert state.block_size == 16
        assert state._cache is not None
        assert state._cache.shape == (4 * 16,)
        assert state._kv_cache_ref is not None
        assert state._kv_cache_ref() is kv_cache
        cached_obj = state._cache

        # 2. Second call with same kv_cache: block_size matches (line 41 False),
        # cache hit (line 44 True), returns True without reallocating
        assert state.ensure_cache() is True
        assert state._cache is cached_obj

        # 3. Third call with different kv_cache object (same block_size 16):
        # cache miss (line 44 False), allocates fresh _cache
        new_kv_cache = torch.zeros(4, 16)
        state.swa_cache_module.bound_cache = new_kv_cache
        assert state.ensure_cache() is True
        assert state._cache is not cached_obj
        assert state._kv_cache_ref() is new_kv_cache

    def test_v1_all_exports(self):
        assert set(v1_engram.__all__) == {"Engram", "NgramHashState"}
        assert v1_engram.Engram is common_engram.Engram
        assert issubclass(v1_engram.NgramHashState, common_engram.NgramHashState)


# ==============================================================================
# 5. v2.engram tests (lines 1-39)
# ==============================================================================


class TestV2Engram:
    def test_v2_write_cache(self, monkeypatch: pytest.MonkeyPatch):
        mock_write = MagicMock()
        monkeypatch.setattr(v2_engram, "write_cache", mock_write)

        state = object.__new__(v2_engram.NgramHashState)
        cache = torch.zeros(10)
        slots = torch.zeros(2)
        values = torch.zeros(2)
        valid = torch.ones(2, dtype=torch.bool)

        state.write_cache(cache, slots, values, valid)
        mock_write.assert_called_once_with(cache, slots, values, valid)

    def test_v2_all_exports(self):
        expected_exports = {
            "Engram",
            "NgramHashState",
            "ParallelEngramEmbedding",
            "_INVALID_SCALE",
            "_UNIT_SCALE",
            "_gather_engram_rows",
            "_load_local_rows",
            "_unpack_bytes",
            "dequant_fp8_rows",
            "register_row_slice",
            "upstream",
        }
        assert set(v2_engram.__all__) == expected_exports
        for name in expected_exports:
            assert hasattr(v2_engram, name)
