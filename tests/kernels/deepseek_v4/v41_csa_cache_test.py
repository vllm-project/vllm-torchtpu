"""V4.1 FP8 CSA cache encoding and physical-address contracts."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41 import cache, csa

pytestmark = pytest.mark.cpu_test


def values(count=17):
    rng = np.random.default_rng(121)
    x = rng.normal(size=(count, 512)).astype(np.float32)
    x[0] = 0
    if count > 1:
        x[1] *= 0.000001
    if count > 2:
        x[2] *= 1000
    return torch.from_numpy(x).to(torch.bfloat16)


def reference_pack(v):
    # Independent NumPy bit packing with exact power-of-two scaling. The native
    # JAX reference uses an approximate CPU exp2; actual Pallas parity is a TPU check.
    import ml_dtypes

    x = v.float().numpy()
    groups = x[:, :448].reshape(-1, 7, 64)
    scales = np.exp2(np.ceil(np.log2(np.maximum(np.abs(groups).max(1), 1e-4) / 448.0)))
    q = (
        (groups / scales[:, None])
        .astype(ml_dtypes.float8_e4m3fn)
        .view(np.uint8)
        .reshape(-1, 448)
    )
    scale_bytes = (scales.view(np.uint32) >> 23).astype(np.uint8)
    slabs = (
        np.concatenate((q, scale_bytes), axis=1).reshape(-1, 4, 128).astype(np.uint32)
    )
    nope = (
        slabs[:, 0] | (slabs[:, 1] << 8) | (slabs[:, 2] << 16) | (slabs[:, 3] << 24)
    ).view(np.int32)
    bits = x[:, 448:].astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.uint32)
    rope = (bits[:, :32] | (bits[:, 32:] << 16)).view(np.int32)
    return nope, rope


def write_cpu_cache(owner, slots, data):
    """Drive the production record mapping through a CPU Torch/JAX bridge."""

    def write(pool, nope, rope, values):
        result = jax.jit(csa.write_csa_values)(
            jnp.asarray(pool.numpy()),
            jnp.asarray(nope.numpy()),
            jnp.asarray(rope.numpy()),
            jnp.asarray(values.float().numpy(), jnp.bfloat16),
        )
        return torch.from_numpy(np.asarray(result).copy())

    with patch.object(csa, "_write_op", write):
        owner.write(slots, data)


def read_cpu_cache(owner, slots):
    """Decode packed records for numerical assertions, outside serving code."""
    nope = owner.raw[owner.row_indices(slots).long()].long()
    slab = torch.stack([(nope >> (8 * i)) & 255 for i in range(4)], dim=-2)
    record = slab.to(torch.uint8).flatten(-2)
    fp8 = record[..., :448].contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.pow(2.0, record[..., 448:].int() - 127)
    nope = (fp8.unflatten(-1, (7, 64)) * scales[..., None, :]).flatten(-2)
    rope = owner.raw.reshape(-1, 32)[owner.rope_indices(slots).long()].long()
    bits = torch.cat((rope & 65535, (rope >> 16) & 65535), dim=-1)
    rope = bits.to(torch.int16).view(torch.bfloat16)
    return torch.cat((nope.to(torch.bfloat16), rope), dim=-1)


@pytest.mark.parametrize("page_tokens", [4, 8, 16, 32])
def test_packed_cache_bit_exact_and_isolated(page_tokens):
    pages = 5
    stride = page_tokens * 640 + 1024
    offset = 1536
    raw = torch.full(
        ((offset + pages * stride + 1024) // 512, 128), 0x5A5A5A5A, dtype=torch.int32
    )
    owner = csa.PackedCSACache(
        raw, ((pages, page_tokens, 640), torch.uint8, stride, offset)
    )
    slots = torch.tensor(
        [0, 1, 2, 3, page_tokens - 1, page_tokens + 2, 4 * page_tokens + 1, -1, -1],
        dtype=torch.int32,
    )
    # Avoid duplicate valid writes for the smallest page.
    slots[4] = 2 * page_tokens
    v = values(len(slots))
    old = raw.clone()
    write_cpu_cache(owner, slots, v)
    expected_nope, expected_rope = reference_pack(v[:-2])
    np.testing.assert_array_equal(
        raw[owner.row_indices(slots[:-2]).long()].numpy(), expected_nope
    )
    np.testing.assert_array_equal(
        raw.reshape(-1, 32)[owner.rope_indices(slots[:-2]).long()].numpy(),
        expected_rope,
    )
    changed = torch.zeros_like(raw, dtype=torch.bool)
    changed[owner.row_indices(slots[:-2]).long()] = True
    changed.reshape(-1, 32)[owner.rope_indices(slots[:-2]).long()] = True
    torch.testing.assert_close(raw[~changed], old[~changed], rtol=0, atol=0)
    gpu_words = jax.jit(csa.write_csa_values)(
        jnp.asarray(old.numpy()),
        jnp.asarray(
            owner.row_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1).numpy()
        ),
        jnp.asarray(
            owner.rope_indices(slots.clamp(min=0)).masked_fill(slots < 0, -1).numpy()
        ),
        jnp.asarray(v.float().numpy(), jnp.bfloat16),
    )
    np.testing.assert_array_equal(gpu_words, raw.numpy())
    decoded = read_cpu_cache(owner, slots[:-2])
    torch.testing.assert_close(decoded[:, 448:], v[:-2, 448:], rtol=0, atol=0)
    assert decoded.isfinite().all()


def test_holey_reindex_compacts_without_changing_selected_pairs():
    nope = jnp.array(
        [[40, -1, 91, -1, 3], [8, 8, -1, 7, -1], [-1, -1, -1, -1, -1]], jnp.int32
    )
    rope = jnp.array(
        [[201, -1, 311, -1, 19], [50, 50, -1, 11, -1], [-1, -1, -1, -1, -1]], jnp.int32
    )
    n, r = jax.jit(csa.compact_csa_records)(nope, rope)
    assert n.shape == r.shape == (3, 128)
    np.testing.assert_array_equal(n[0, :3], [40, 91, 3])
    np.testing.assert_array_equal(r[0, :3], [201, 311, 19])
    np.testing.assert_array_equal(n[1, :3], [8, 8, 7])
    np.testing.assert_array_equal(r[1, :3], [50, 50, 11])
    assert np.all(np.asarray(n[:2, 3:]) == -1) and np.all(np.asarray(n[2]) == -1)


def test_large_pool_uses_record_indices_without_byte_overflow():
    raw = torch.empty((20 * 2**30 // 512, 128), device="meta", dtype=torch.int32)
    stride = 16 * 640 + 1024
    offset = 9 * 2**30
    owner = csa.PackedCSACache(raw, ((1024, 16, 640), torch.uint8, stride, offset))
    slots = torch.tensor([0, 1, 16, 16383], dtype=torch.int32)
    for slot, n, r in zip(
        slots.tolist(),
        owner.row_indices(slots).tolist(),
        owner.rope_indices(slots).tolist(),
    ):
        assert n == (offset + slot // 16 * stride) // 512 + slot % 16
        assert r == (offset + slot // 16 * stride + 16 * 512) // 128 + slot % 16
    assert (
        owner.row_indices(slots).dtype == owner.rope_indices(slots).dtype == torch.int32
    )


@pytest.mark.parametrize(
    "offset,stride,page_tokens", [(128, 5120, 8), (0, 5136, 8), (0, 4096, 3)]
)
def test_reject_incompatible_page_geometry(offset, stride, page_tokens):
    with pytest.raises(ValueError, match="alignment"):
        csa.PackedCSACache(
            torch.empty((64, 128), dtype=torch.int32),
            ((2, page_tokens, 640), torch.uint8, stride, offset),
        )


def test_csa_wiring_keeps_swa_state_and_shared_record_pairs():
    rng = np.random.default_rng(672)
    rows = jnp.asarray(rng.normal(size=(128, 512)), jnp.bfloat16)
    pool = cache.scatter_cache_values(
        jnp.zeros((256, 128), jnp.int32), jnp.arange(128, dtype=jnp.int32), rows
    )
    q = jnp.asarray(rng.normal(size=(3, 8, 512)), jnp.bfloat16)
    swa = jnp.tile(jnp.arange(64, dtype=jnp.int32), (3, 1))
    nope = jnp.array([[1, -1, 8], [7, 8, -1], [-1, -1, -1]], jnp.int32)
    rope = jnp.array([[200, -1, 206], [311, 312, -1], [-1, -1, -1]], jnp.int32)
    sink = jnp.zeros((8,), jnp.float32)
    maximum, denominator, acc = csa.swa_attention_state(q, pool, swa, 0.125)
    captured = []

    def native(*args, **kwargs):
        captured.append((args, kwargs))
        return args[0]

    with patch.object(csa.sparse_mla, "sparse_ragged_paged_attention", native):
        result = csa.csa_attention_jax(q, pool, pool, swa, nope, rope, sink, 0.125)
    args, kw = captured[0]
    assert kw["sm_scale"] == 0.125
    np.testing.assert_array_equal(args[8], acc.astype(q.dtype))
    np.testing.assert_array_equal(args[9], denominator)
    np.testing.assert_array_equal(args[10], maximum)
    n, r = kw["physical_indices"]
    assert n.shape == r.shape == (3, 128)
    np.testing.assert_array_equal(n[0, :2], [1, 8])
    np.testing.assert_array_equal(r[0, :2], [200, 206])
    np.testing.assert_array_equal(result, q)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("largest_page", [16384, 65536])
def test_upstream_page_unification_preserves_reserved_geometry(ratio, largest_page):
    from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    main = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=ratio,
        alignment=512,
        model_version="deepseek_v4",
    )
    other = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=largest_page // 32,
        dtype=torch.bfloat16,
        alignment=512,
    )
    baseline = unify_kv_cache_spec_page_size({"main": main, "other": other})
    packed = unify_kv_cache_spec_page_size(
        {"main": csa.packed_csa_spec(main), "other": other}
    )
    for name in baseline:
        assert packed[name].page_size_bytes == baseline[name].page_size_bytes
        assert packed[name].block_size == baseline[name].block_size
        assert packed[name].num_states == baseline[name].num_states
    assert packed["main"].dtype == torch.uint8
    assert packed["main"].state_content_size_bytes == 1024
    assert packed["main"].head_size == 512


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("block_compact", [False, True])
def test_upstream_allocator_binds_packed_main_beside_indexer(ratio, block_compact):
    from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MLAAttentionSpec,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout

    specs = {
        "main": csa.packed_csa_spec(
            MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.bfloat16,
                tokens_per_state=ratio,
                alignment=512,
            )
        ),
        "indexer": MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            tokens_per_state=ratio,
            alignment=512,
        ),
    }
    specs = unify_kv_cache_spec_page_size(specs)
    blocks, page = 3, specs["main"].page_size_bytes
    layout = KVCacheLayout.BLHNC if block_compact else KVCacheLayout.LBHNC
    stride = page * 2 if block_compact else page
    offset = page if block_compact else blocks * page
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=blocks * page * 2,
                layers=[name],
                layer_stride=blocks * page,
                block_stride=stride,
                offset=position,
            )
            for name, position in [("main", 0), ("indexer", offset)]
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec)
            for name, spec in specs.items()
        ],
    )
    actual = cache.materialize_upstream_kv_cache(
        config,
        torch.device("cpu"),
        layout,
        word_rows=True,
    )
    owner = csa.PackedCSACache(
        actual.native_raw_cache["main"],
        actual.native_cache_layouts["main"],
    )
    assert owner.shape == (blocks, specs["main"].num_states, 1024)
    assert (owner.raw is actual.native_raw_cache["indexer"]) == block_compact
    indexer_before = actual.kv_caches["indexer"].clone()
    slots = torch.tensor(
        [0, owner.shape[1] - 1, owner.shape[1], blocks * owner.shape[1] - 1],
        dtype=torch.int32,
    )
    data = values(len(slots))
    write_cpu_cache(owner, slots, data)
    nope, rope = reference_pack(data)
    np.testing.assert_array_equal(owner.raw[owner.row_indices(slots).long()], nope)
    np.testing.assert_array_equal(
        owner.raw.reshape(-1, 32)[owner.rope_indices(slots).long()],
        rope,
    )
    torch.testing.assert_close(
        actual.kv_caches["indexer"], indexer_before, rtol=0, atol=0
    )


@pytest.mark.parametrize("ratio", [1, 2])
def test_source_owns_packed_cache_without_feature_switch(ratio):
    from types import SimpleNamespace

    from vllm_torchtpu.models.vllm.deepseek_v41.attention import DeepseekV41Attention

    layer = DeepseekV41Attention.__new__(DeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    layer.is_kv_source = True
    layer.compress_ratio, layer.head_dim = ratio, 512
    layer.kv_cache_dtype, layer.kv_cache_torch_dtype = "auto", torch.bfloat16
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))
    spec = layer.get_kv_cache_spec(config)
    assert spec.dtype == torch.uint8 and spec.num_states == 16 // ratio
    assert spec.state_content_size_bytes == 1024
    layer._tpu_raw_cache = torch.empty((64, 128), dtype=torch.int32)
    layer._tpu_cache_layout = ((2, spec.num_states, 1024), torch.uint8, 16384, 0)
    assert isinstance(layer.kv_cache, csa.PackedCSACache)
    layer.is_kv_source = False
    assert layer.get_kv_cache_spec(config) is None


@pytest.mark.parametrize("replay", [False, True])
def test_swa_only_model_path_preserves_request_boundaries(replay):
    from types import SimpleNamespace

    from vllm_torchtpu.models.vllm.deepseek_v41 import attention

    torch.manual_seed(12)
    dim, heads, window, block = 512, 2, 4, 4
    layer = attention.DeepseekV41Attention.__new__(attention.DeepseekV41Attention)
    torch.nn.Module.__init__(layer)
    layer._ced_decoder = replay
    layer.n_local_heads, layer.head_dim, layer.window_size = heads, dim, window
    layer.scale = dim**-0.5
    layer.compress_ratio, layer.compressor, layer.indexer = 0, None, None
    layer.wq_b = lambda x: x
    layer._rotate = lambda x, positions: x
    layer.attn_sink = torch.randn(heads)
    raw = torch.zeros((48, 128), dtype=torch.int32)
    swa = cache.PagedCacheRows(raw, ((6, block, dim), torch.bfloat16, 4096, 0))
    layer.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=swa)
    tables = torch.tensor([[4, 1, 3], [5, 0, 2]])
    history = [torch.randn(9, dim).bfloat16(), torch.randn(5, dim).bfloat16()]
    for req, size in enumerate([6, 3]):
        ids = torch.arange(size)
        slots = tables[req, ids // block] * block + ids % block
        swa.write(slots, history[req][:size])
    positions = torch.tensor([6, 7, 8, 3, 4, 0])
    meta = SimpleNamespace(
        seq_lens=torch.tensor([9, 5]),
        block_tables=tables.flatten(),
        query_start_loc=torch.tensor([0, 3, 5]),
    )
    kv = torch.cat((history[0][6:], history[1][3:], torch.zeros(1, dim).bfloat16()))
    qr = torch.randn(6, heads * dim).bfloat16()
    output = torch.empty(6, heads, dim, dtype=torch.bfloat16)

    def run(q, sw, main, sr, nr, rr, sink, scale):
        assert nr.shape == rr.shape == (6, 0)
        result = csa.csa_attention_jax(
            jnp.asarray(q.float().numpy(), jnp.bfloat16),
            jnp.asarray(sw.raw.numpy()),
            jnp.asarray(main.raw.numpy()),
            jnp.asarray(sr.numpy()),
            jnp.asarray(nr.numpy()),
            jnp.asarray(rr.numpy()),
            jnp.asarray(sink.numpy()),
            scale,
        )
        return torch.from_numpy(np.asarray(result, dtype=np.float32)).bfloat16()

    context = SimpleNamespace(
        attn_metadata={"swa": meta},
        additional_kwargs={"deepseek_v41_replay_starts": torch.tensor([6, 3])},
    )
    with (
        patch.object(attention, "get_forward_context", return_value=context),
        patch.object(attention, "csa_attention", run),
    ):
        layer._prepare_tpu(kv, qr, kv, None, None, None, positions, output)
    q = qr.reshape(6, heads, dim).float()
    for row, (req, pos) in enumerate([(0, 6), (0, 7), (0, 8), (1, 3), (1, 4)]):
        start = max([6, 3][req] if replay else 0, pos - window + 1)
        selected = history[req][start : pos + 1].float()
        scores = q[row] @ selected.T * layer.scale
        probs = torch.cat((scores, layer.attn_sink[:, None]), dim=-1).softmax(-1)
        expected = probs[:, :-1] @ selected
        torch.testing.assert_close(output[row].float(), expected, atol=0.008, rtol=0.03)
    assert output[-1].count_nonzero() == 0
