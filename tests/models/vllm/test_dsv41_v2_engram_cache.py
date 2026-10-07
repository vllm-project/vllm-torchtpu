# SPDX-License-Identifier: Apache-2.0
"""100% Diff coverage test suite for DeepSeek-V4.1 Engram CPU offload and cache sizing.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.common.engram
- vllm_torchtpu.models.vllm.deepseek_v41.v2.cache (lines 20-40)
- vllm_torchtpu.envs (TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import vllm_torchtpu.envs as envs
from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as v2_engram
from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import (
    Engram,
    ParallelEngramEmbedding,
    _gather_engram_rows,
    _load_local_rows,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    SPARSECORE_MAX_POOL_BYTES,
    reserve_rope_companion,
    uses_fp8_rope_companion,
)


@pytest.fixture(autouse=True)
def setup_torch_accelerator(monkeypatch: pytest.MonkeyPatch):
    """Enable CPU execution for upstream Engram constructor checks."""
    monkeypatch.setattr(
        torch.accelerator, "current_device_index", lambda: 0, raising=False
    )


# ==============================================================================
# 1. Tests for _load_local_rows
# ==============================================================================


def test_load_local_rows_uint8_dtype():
    """Verify _load_local_rows executes .view(torch.uint8) for uint8 param."""
    param = torch.zeros(10, 4, dtype=torch.uint8)
    loaded_weight = torch.arange(40, dtype=torch.uint8).reshape(10, 4)
    _load_local_rows(param, loaded_weight)
    assert torch.equal(param, loaded_weight)


def test_load_local_rows_non_uint8_dtype():
    """Verify _load_local_rows executes .to(param.dtype) for float params."""
    param = torch.zeros(8, 4, dtype=torch.bfloat16)
    loaded_weight = torch.ones(8, 4, dtype=torch.float32) * 2.5
    _load_local_rows(param, loaded_weight)
    assert torch.equal(param, loaded_weight.to(torch.bfloat16))


# ==============================================================================
# 2. Tests for ParallelEngramEmbedding._weight_loader
# ==============================================================================


def test_weight_loader_local_rows_cpu():
    """Verify _weight_loader dispatches to _load_local_rows when shape matches."""
    param = torch.nn.Parameter(
        torch.zeros(6, 8, dtype=torch.bfloat16), requires_grad=False
    )
    loaded = torch.full((6, 8), 3.0, dtype=torch.float32)
    ParallelEngramEmbedding._weight_loader(param, loaded)
    assert torch.equal(param.data, loaded.to(torch.bfloat16))


def test_weight_loader_upstream_dispatch_cpu(monkeypatch: pytest.MonkeyPatch):
    """Verify _weight_loader dispatches to upstream loader when shape differs."""
    mock_upstream_loader = MagicMock()
    monkeypatch.setattr(
        v2_engram.upstream.ParallelEngramEmbedding,
        "_weight_loader",
        mock_upstream_loader,
    )

    param = torch.nn.Parameter(
        torch.zeros(6, 8, dtype=torch.bfloat16), requires_grad=False
    )
    loaded = torch.full((12, 8), 1.0, dtype=torch.float32)
    ParallelEngramEmbedding._weight_loader(param, loaded)
    mock_upstream_loader.assert_called_once_with(param, loaded)


def test_weight_loader_tpu_device_path(monkeypatch: pytest.MonkeyPatch):
    """Verify _weight_loader executes ExpertParamStager when device is TPU."""
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

    param = MockTPUParam()
    loaded = torch.full((4, 4), 7.0, dtype=torch.bfloat16)
    ParallelEngramEmbedding._weight_loader(param, loaded)

    mock_stager_cls.assert_called_once()
    mock_stager.host_param.assert_called_once_with(param)
    mock_stager.flush.assert_called_once_with(param)
    assert torch.equal(host_tensor, loaded)


# ==============================================================================
# 3. Tests for ParallelEngramEmbedding.__init__ and _allocate_weights
# ==============================================================================


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
def test_parallel_engram_embedding_init_offload_flag(
    monkeypatch: pytest.MonkeyPatch,
    env_setting: str | None,
    init_arg: bool | None,
    expected_offload: bool,
):
    """Verify cpu_offload resolution from envs and explicit constructor parameter."""
    if env_setting is None:
        monkeypatch.delenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", raising=False)
    else:
        monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", env_setting)

    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0)

    kwargs = {}
    if init_arg is not None:
        kwargs["cpu_offload"] = init_arg

    emb = ParallelEngramEmbedding(32, 16, (8, 8, 8, 8), block_size=16, **kwargs)
    assert emb.cpu_offload is expected_offload
    if expected_offload:
        assert emb.weight.device.type == "cpu"
        assert emb.weight_scale_inv is not None
        assert emb.weight_scale_inv.device.type == "cpu"
        assert getattr(emb.weight, "dummy_weight_value", None) == 1.0
        assert getattr(emb.weight_scale_inv, "dummy_weight_value", None) == 127


def test_parallel_engram_embedding_allocate_weights_branching(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify _allocate_weights branches on cpu_offload setting."""
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0)

    # Offload True
    emb_offload = ParallelEngramEmbedding(
        16, 32, (8, 8), block_size=16, cpu_offload=True
    )
    w_cpu, s_cpu = emb_offload._allocate_weights()
    assert w_cpu.device.type == "cpu"
    assert w_cpu.dtype == torch.float8_e4m3fn
    assert s_cpu.device.type == "cpu"
    assert s_cpu.dtype == torch.uint8

    # Offload False (device allocation)
    emb_device = ParallelEngramEmbedding(
        16, 32, (8, 8), block_size=16, cpu_offload=False
    )
    w_dev, s_dev = emb_device._allocate_weights()
    assert w_dev.shape == (emb_device.part_num_embeddings, emb_device.dim)
    assert s_dev.shape == (emb_device.part_num_embeddings, emb_device.dim // 16)


def test_parallel_engram_embedding_none_scale_inv(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify initialization safely handles None weight_scale_inv."""
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0)

    orig_init = v2_engram.upstream.ParallelEngramEmbedding.__init__

    def fake_upstream_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.weight_scale_inv = None

    monkeypatch.setattr(
        v2_engram.upstream.ParallelEngramEmbedding, "__init__", fake_upstream_init
    )
    emb = ParallelEngramEmbedding(16, 32, (8, 8), block_size=16, cpu_offload=True)
    assert emb.weight_scale_inv is None
    assert getattr(emb.weight, "dummy_weight_value", None) == 1.0


# ==============================================================================
# 4. Tests for ParallelEngramEmbedding.lookup
# ==============================================================================


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_parallel_engram_embedding_lookup_computation(
    monkeypatch: pytest.MonkeyPatch, tp_rank: int
):
    """Verify lookup mathematical correctness, head clamping, and sync."""
    tp_size = 2
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: tp_rank
    )

    head_sizes = (16, 16, 16, 16)
    num_embeddings = sum(head_sizes)
    dim = 32
    block_size = 16

    emb = ParallelEngramEmbedding(
        num_embeddings, dim, head_sizes, block_size=block_size, cpu_offload=True
    )

    emb.weight.fill_(1.0)
    emb.weight_scale_inv.fill_(127)

    indices = torch.tensor(
        [
            [0, 16, 32, 48],
            [-1, 10, 999, 20],
        ],
        dtype=torch.int32,
    )

    out = torch.empty(
        (indices.shape[0], emb.part_n_hash_cols, dim), dtype=torch.bfloat16
    )

    emb.lookup(indices, out)
    assert out.shape == (2, emb.part_n_hash_cols, dim)

    heads = torch.arange(emb.part_n_hash_cols) + emb.head_start
    for t in range(indices.shape[0]):
        for h_idx in range(emb.part_n_hash_cols):
            head_col = heads[h_idx].item()
            sel_val = indices[t, min(head_col, emb.n_hash_cols - 1)].item()
            is_valid = (
                head_col < emb.n_hash_cols
                and sel_val >= emb.vocab_start_idx
                and sel_val < emb.vocab_end_idx
            )
            expected_val = 1.0 if is_valid else 0.0
            torch.testing.assert_close(
                out[t, h_idx],
                torch.full((dim,), expected_val, dtype=torch.bfloat16),
            )


def test_parallel_engram_embedding_lookup_cross_device_transfer(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify lookup converts indices.device to work_device when they differ."""
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0)

    emb = ParallelEngramEmbedding(16, 16, (8, 8), block_size=16, cpu_offload=True)
    emb.weight.fill_(1.0)
    emb.weight_scale_inv.fill_(127)

    indices = torch.zeros((2, emb.n_hash_cols), dtype=torch.int32)

    device_transferred = False

    class MockTensorWithDevice:
        def __init__(self, tensor):
            self._t = tensor
            self.device = SimpleNamespace(type="fake_dev")

        def to(self, dev):
            nonlocal device_transferred
            device_transferred = True
            return self._t

        def __getattr__(self, name):
            return getattr(self._t, name)

        def __getitem__(self, item):
            return self._t[item]

    wrapped_indices = MockTensorWithDevice(indices)

    out = torch.empty((2, emb.part_n_hash_cols, 16), dtype=torch.bfloat16)
    emb.lookup(wrapped_indices, out)
    assert device_transferred is True


# ==============================================================================
# 5. Tests for _gather_engram_rows
# ==============================================================================


@pytest.mark.parametrize("width", [64, 128])
def test_gather_engram_rows_padding(monkeypatch: pytest.MonkeyPatch, width: int):
    """Verify _gather_engram_rows pads width to 128 and strips padding."""
    rows = torch.randn(4, 3, width, dtype=torch.bfloat16)

    def fake_all_gather(t, dim):
        assert dim == 0
        if width % 128 != 0:
            assert t.shape[-1] == 128
        else:
            assert t.shape[-1] == width
        return torch.cat([t, t], dim=0)

    monkeypatch.setattr(
        v2_engram.upstream, "tensor_model_parallel_all_gather", fake_all_gather
    )

    out = _gather_engram_rows(rows, dim=0)
    assert out.shape == (8, 3, width)
    torch.testing.assert_close(out[:4], rows)
    torch.testing.assert_close(out[4:], rows)


# ==============================================================================
# 6. Tests for Engram staging, embedding, and preparation
# ==============================================================================


def test_engram_create_embedding_and_row_slice_registration(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify _create_embedding instantiates embedding and registers slices."""
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: 0)

    registered_slices = []

    def fake_register(name, start, count):
        registered_slices.append((name, start, count))

    monkeypatch.setattr(v2_engram, "register_row_slice", fake_register)

    layout = SimpleNamespace(
        num_embeddings=[64],
        head_dim=32,
        primes=[[(16, 16), (16, 16)]],
        layer_ids=[14],
    )

    engram_obj = object.__new__(Engram)
    emb = Engram._create_embedding(engram_obj, layout, 0)

    assert isinstance(emb, ParallelEngramEmbedding)
    assert len(registered_slices) == 2
    assert registered_slices[0] == (
        "layers.14.engram.embed.weight",
        emb.vocab_start_idx,
        emb.part_num_embeddings,
    )
    assert registered_slices[1] == (
        "layers.14.engram.embed.scale",
        emb.vocab_start_idx,
        emb.part_num_embeddings,
    )


def test_engram_init_staging_branches():
    """Verify _init_staging allocates raw/scale staging only with cpu_offload."""
    # Branch 1: cpu_offload = True
    engram_on = object.__new__(Engram)
    nn.Module.__init__(engram_on)
    engram_on.embed_tokens = SimpleNamespace(
        cpu_offload=True, part_n_hash_cols=4, block_size=8
    )
    Engram._init_staging(engram_on, max_tokens=10, head_dim=16)
    for name, last in (("_host_raw", 16), ("_host_scale", 2)):
        buf = getattr(engram_on, name)
        assert buf.shape == (10, 4, last)
        assert buf.dtype == torch.uint8
        assert buf.device.type == "cpu"
    assert engram_on._staged_raw.shape == (10, 4, 16)
    assert engram_on._staged_scale.shape == (10, 4, 2)
    assert torch.all(engram_on._host_scale == v2_engram._UNIT_SCALE)
    assert torch.all(engram_on._staged_scale == v2_engram._UNIT_SCALE)
    assert engram_on.staged_rows.shape == (10, 4, 16)

    # Branch 2: cpu_offload = False
    engram_off = object.__new__(Engram)
    nn.Module.__init__(engram_off)
    engram_off.embed_tokens = SimpleNamespace(cpu_offload=False, part_n_hash_cols=4)
    Engram._init_staging(engram_off, max_tokens=10, head_dim=16)
    assert engram_off._staged_raw is None
    assert engram_off.staged_rows.shape == (10, 4, 16)


def test_engram_prepare_embeddings_branching(monkeypatch: pytest.MonkeyPatch):
    """Verify the raw-gather offload path, tail reset, and both fallbacks."""
    dim = 16
    block = 8
    max_tokens = 8
    cols = 3
    one_fp8 = torch.tensor(1.0, dtype=torch.float8_e4m3fn).view(torch.uint8).item()

    def make_engram(cpu_offload: bool):
        engram = object.__new__(Engram)
        nn.Module.__init__(engram)
        embed = MagicMock()
        embed.cpu_offload = cpu_offload
        embed.part_n_hash_cols = cols
        embed.block_size = block

        def mock_gather_raw(hash_ids, raw_out, scale_out):
            raw_out.fill_(one_fp8)
            scale_out.fill_(v2_engram._UNIT_SCALE + 2)  # 2**2

        embed.gather_raw = mock_gather_raw
        engram.embed_tokens = embed
        Engram._init_staging(engram, max_tokens=max_tokens, head_dim=dim)
        return engram

    engram = make_engram(cpu_offload=True)

    # Case A: Partial tokens -> tail reset to FP8 zero with unit scale.
    engram._host_raw.fill_(99)
    engram._host_scale.fill_(99)
    engram.prepare_embeddings(torch.zeros(5, cols, dtype=torch.int32))
    assert torch.all(engram._staged_raw[:5] == one_fp8)
    assert torch.all(engram._staged_scale[:5] == v2_engram._UNIT_SCALE + 2)
    assert torch.count_nonzero(engram._staged_raw[5:]) == 0
    assert torch.all(engram._staged_scale[5:] == v2_engram._UNIT_SCALE)
    rows = Engram._ready_rows(engram, max_tokens)
    torch.testing.assert_close(
        rows[:5], torch.full((5, cols, dim), 4.0, dtype=torch.bfloat16)
    )
    assert torch.count_nonzero(rows[5:]) == 0

    # Case B: Exact tokens -> no tail reset.
    engram.prepare_embeddings(torch.zeros(max_tokens, cols, dtype=torch.int32))
    torch.testing.assert_close(
        Engram._ready_rows(engram, max_tokens),
        torch.full((max_tokens, cols, dim), 4.0, dtype=torch.bfloat16),
    )

    # Case C/D: Exceeding max_tokens, or no offload staging -> upstream path.
    fallback_calls = []
    monkeypatch.setattr(
        v2_engram.upstream.Engram,
        "prepare_embeddings",
        lambda self, hash_ids: fallback_calls.append(self),
    )
    engram.prepare_embeddings(torch.zeros(12, cols, dtype=torch.int32))
    engram_off = make_engram(cpu_offload=False)
    engram_off.prepare_embeddings(torch.zeros(5, cols, dtype=torch.int32))
    assert fallback_calls == [engram, engram_off]


# ==============================================================================
# 7. Tests for Engram._sp_embed and Engram.forward
# ==============================================================================


@pytest.mark.parametrize("num_tokens", [6, 7])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_engram_sp_embed_and_forward(
    monkeypatch: pytest.MonkeyPatch, num_tokens: int, tp_size: int
):
    """Verify _sp_embed and forward across SP, non-SP, and token_mask."""
    cols = 3
    dim = 8
    hc_mult = 2
    rank = 0

    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: rank
    )

    chunk = -(-num_tokens // tp_size)

    engram = object.__new__(Engram)
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

    monkeypatch.setattr(v2_engram, "_gather_engram_rows", fake_gather)

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

    # Test SP=True with token_mask
    out_sp_mask = engram.forward(hidden_states, hash_ids, token_mask=token_mask)
    assert out_sp_mask.shape == (chunk, hc_mult, dim)

    # Test SP=True without token_mask
    out_sp_nomask = engram.forward(hidden_states, hash_ids, token_mask=None)
    assert out_sp_nomask.shape == (chunk, hc_mult, dim)

    # Test SP=False with and without token_mask
    engram.use_sequence_parallel = False
    engram.embed = lambda hids: staged[: hids.shape[0]]
    hidden_full = torch.randn(num_tokens, hc_mult, dim, dtype=torch.bfloat16)

    out_nonsp_mask = engram.forward(hidden_full, hash_ids, token_mask=token_mask)
    assert out_nonsp_mask.shape == (num_tokens, hc_mult, dim)

    out_nonsp_nomask = engram.forward(hidden_full, hash_ids, token_mask=None)
    assert out_nonsp_nomask.shape == (num_tokens, hc_mult, dim)


# ==============================================================================
# 8. Tests for cache.py and envs.py
# ==============================================================================


def test_uses_fp8_rope_companion_architectures():
    """Verify uses_fp8_rope_companion covers all architecture and dtype branches."""
    # Architecture matches with fp8 cache
    cfg_match = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    assert uses_fp8_rope_companion(cfg_match) is True

    # Architecture matches but cache_dtype is not fp8
    cfg_non_fp8 = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="bfloat16"),
    )
    assert uses_fp8_rope_companion(cfg_non_fp8) is False

    # Architecture does not match
    cfg_diff = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    assert uses_fp8_rope_companion(cfg_diff) is False

    # Architectures is empty
    cfg_empty = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[])),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    assert uses_fp8_rope_companion(cfg_empty) is False

    # Architectures is None (exercises `or []` branch)
    cfg_none = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=None)),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    assert uses_fp8_rope_companion(cfg_none) is False


def test_reserve_rope_companion_capping():
    """Verify reserve_rope_companion non-V41 bypass and 16 GiB cap."""
    cfg_non_v41 = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    assert reserve_rope_companion(cfg_non_v41, 20 << 30) == 20 << 30

    cfg_v41 = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
        ),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
    )
    # Under limit: 10 GiB * 4 // 5 = 8 GiB < SPARSECORE_MAX_POOL_BYTES
    avail_small = 10 << 30
    assert reserve_rope_companion(cfg_v41, avail_small) == avail_small * 4 // 5

    # Over limit: 30 GiB * 4 // 5 = 24 GiB > SPARSECORE_MAX_POOL_BYTES
    avail_large = 30 << 30
    capped = reserve_rope_companion(cfg_v41, avail_large)
    assert capped == SPARSECORE_MAX_POOL_BYTES
    assert capped == (16 << 30) - (64 << 20)


@pytest.mark.parametrize(
    ("val_str", "expected_bool"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("0", False),
        ("false", False),
        ("False", False),
    ],
)
def test_envs_engram_cpu_offload_parsing(
    monkeypatch: pytest.MonkeyPatch, val_str: str, expected_bool: bool
):
    """Verify TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD parses booleans."""
    monkeypatch.setenv("TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD", val_str)
    resolver = envs.environment_variables["TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD"]
    assert resolver() is expected_bool
