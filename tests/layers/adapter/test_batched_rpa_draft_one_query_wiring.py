"""Wiring of TPU_BATCHED_RPA_DRAFT_ONE_QUERY onto the speculative draft layers.

The real-kernel equivalence is in test_batched_rpa_draft_one_query.py (TPU).
"""

import functools
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.layers.adapter import attention as adapter

KVLayout = batched_rpa_configs.KVLayout
BlockSizes = batched_rpa_configs.BlockSizes
_LANE = KVLayout.SEQ_ALONG_LANE


@pytest.mark.cpu_test
def test_one_query_steps_wrap_only_verify_window_kernels(monkeypatch):
    entries = []

    def impl(*args, rpa_func, **kwargs):
        entries.append(rpa_func)

    monkeypatch.setattr(adapter, "_pallas_rpa_kernel_impl", impl)
    monkeypatch.delenv("TPU_BATCHED_RPA_DECODE_BLOCK", raising=False)
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", "window")
    run = functools.partial(
        adapter._pallas_rpa_kernel_batched,
        *([None] * 12),
        mesh=None,
        sliding_window=None,
        skip_kv_update=False,
        kv_layout=_LANE,
    )
    run(decode_query_size=4)
    assert entries[-1].__qualname__.startswith("_with_verify_block_override")
    run(decode_query_size=4, one_query_steps=True)
    assert entries[-1].__qualname__.startswith("_with_one_query_steps")
    # Layers without verify windows keep the one-query decode path.
    run(one_query_steps=True)
    assert entries[-1] is adapter.ragged_paged_attention_batched
    # The page-table bound still wraps the switch.
    run(decode_query_size=4, one_query_steps=True, max_model_len=1048576)
    assert entries[-1].__qualname__.startswith("_with_addressable_page_table")

    monkeypatch.delenv("TPU_BATCHED_RPA_VERIFY_BLOCK")
    with pytest.raises(ValueError, match="needs TPU_BATCHED_RPA_VERIFY_BLOCK"):
        run(decode_query_size=4, one_query_steps=True)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("verify_block", "kv_block"),
    [("window", (3072, 8, 3)), ("8192,4,2", (8192, 4, 2))],
)
def test_one_query_steps_switch_on_the_query_counts(
    monkeypatch, verify_block, kv_block
):
    seen = {}

    def verify(q, k, v, cache, kv_lens, table, cu, dist, **kwargs):
        seen["verify"] = kwargs
        return jnp.zeros(q.shape, q.dtype), cache

    def one_query(q, k, v, cache, kv_lens, table, cu, dist, **kwargs):
        seen["one_query"] = kwargs
        seen["one_query_table"] = table.shape
        return jnp.ones(q.shape, q.dtype), cache

    def blocks(model_cfgs, serve_cfgs, vmem_limit_bytes, decode_query_size):
        assert decode_query_size == 4
        assert (serve_cfgs.num_seqs, serve_cfgs.page_size) == (12, 128)
        return BlockSizes(4, 2, 3072, 8, 3), None

    monkeypatch.setattr(adapter, "ragged_paged_attention_batched", one_query)
    monkeypatch.setattr(adapter.rpa_batched_wrapper, "calculate_block_sizes", blocks)
    monkeypatch.setattr(
        adapter.pltpu, "get_tpu_info", lambda: SimpleNamespace(num_lanes=128)
    )
    monkeypatch.setenv("TPU_BATCHED_RPA_VERIFY_BLOCK", verify_block)
    fn = adapter._with_one_query_steps(verify, 4)
    num_seqs = 12
    cache = jnp.zeros((10, 2, 64, 4, 128), jnp.uint8)
    table = jnp.zeros(num_seqs * 8192, jnp.int32)

    def ran_one_query(q_lens, decode=None, tokens=num_seqs, kv_len=131072):
        """True if the call ran the one-query kernel."""
        live = len(q_lens)
        lens = np.zeros(num_seqs, np.int32)
        lens[:live] = q_lens
        kv_lens = jnp.asarray(np.where(np.arange(num_seqs) < live, kv_len, 0))
        cu = jnp.asarray(np.concatenate(([0], np.cumsum(lens))), jnp.int32)
        dist = jnp.asarray([live if decode is None else decode, live, live], jnp.int32)
        q = jnp.zeros((tokens, 16, 256), jnp.bfloat16)
        k = jnp.zeros((tokens, 1, 256), jnp.bfloat16)
        out, _ = fn(
            q, k, k, cache, kv_lens, table, cu, dist,
            kv_layout=_LANE, vmem_limit_bytes=1 << 26,
        )  # fmt: skip
        return bool(out[0, 0, 0] == 1)

    assert ran_one_query([1] * 12)
    assert ran_one_query([1] * 5)  # padded slots carry no query
    assert seen["one_query"] == {
        "decode_block_sizes": BlockSizes(1, 1, *kv_block),
        "kv_layout": _LANE,
        "vmem_limit_bytes": 1 << 26,
    }
    assert seen["verify"] == {"kv_layout": _LANE, "vmem_limit_bytes": 1 << 26}
    # The one-query call gets the compact table: 160 KiB of 12 rows, 3328 columns.
    assert seen["one_query_table"] == (num_seqs * 3328,)
    assert ran_one_query([1] * 12, kv_len=3328 * 128)
    assert not ran_one_query([1] * 12, kv_len=3328 * 128 + 1)
    assert not ran_one_query([4] * 3)  # verify windows
    assert not ran_one_query([1, 1, 2, 1, 1, 1])
    assert not ran_one_query([1, 0, 1])  # a live sequence without a query
    assert not ran_one_query([1] * 5, decode=4)  # a prefill or mixed sequence
    # A program with room for full verify windows never traces the switch.
    seen.clear()
    assert not ran_one_query([1] * 12, tokens=4 * num_seqs)
    assert "one_query" not in seen


@pytest.mark.cpu_test
def test_compact_page_columns_follow_the_kernel_rule(monkeypatch):
    monkeypatch.setattr(
        adapter.pltpu, "get_tpu_info", lambda: SimpleNamespace(num_lanes=128)
    )
    # 1M-token table at 128-token pages: 160 KiB of rows.
    assert adapter._compact_page_columns(12, 8320, 4) == 3328
    assert adapter._compact_page_columns(10, 8192, 4) == 4096
    assert adapter._compact_page_columns(256, 8192, 4) == 128
    # Tables the kernel keeps in SMEM whole.
    assert adapter._compact_page_columns(12, 1152, 4) == 0
    assert adapter._compact_page_columns(10, 4096, 4) == 0


@pytest.mark.cpu_test
def test_build_rpa_kernel_binds_one_query_steps(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu.layers.adapter.attention import (
        PallasAttentionBackendImpl,
        PallasBatchedRPAAttentionBackendImpl,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    captured = []

    class FakeOp:
        def register_fake(self, _fake_impl):
            pass

    def fake_jax_op(_name, wrapped_fn, **_kwargs):
        captured.append(wrapped_fn)
        return FakeOp()

    monkeypatch.setattr(PallasAttentionBackendImpl, "_kernel_registry", {})
    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_select_kernel_mesh",
        staticmethod(lambda _mesh, _pcp: (object(), object(), None)),
    )
    monkeypatch.setattr(
        "vllm_torchtpu.layers.adapter.attention.pallas.jax_op", fake_jax_op
    )
    mesh = SimpleNamespace(shape={"attn_dp": 1, "expert": 1, "model": 1})
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    )
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])

    def bound(one_query_steps):
        with set_current_vllm_config(config):
            impl = PallasBatchedRPAAttentionBackendImpl(
                num_heads=2,
                head_size=128,
                scale=1.0,
                num_kv_heads=1,
                alibi_slopes=None,
                sliding_window=None,
                kv_cache_dtype="bfloat16",
            )
        impl.decode_query_size = 4
        impl.one_query_steps = one_query_steps
        with set_vllm_model_wrapper_context(mesh=mesh, vllm_config=None):
            impl._build_rpa_kernel(None, None, None)
        return captured[-1].keywords

    assert "one_query_steps" not in bound(False)
    assert bound(True)["one_query_steps"] is True
    # Draft and target layers of one geometry get separate kernels.
    assert len(captured) == 2


@pytest.mark.cpu_test
def test_proposer_marks_only_verify_window_draft_layers():
    from vllm_torchtpu.layers.adapter.attention import (
        PallasAttentionBackendImpl,
        PallasBatchedRPAAttentionBackendImpl,
    )
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    def batched(decode_query_size):
        impl = object.__new__(PallasBatchedRPAAttentionBackendImpl)
        impl.decode_query_size = decode_query_size
        return impl

    draft, one_query, target = batched(4), batched(1), batched(4)
    plain = object.__new__(PallasAttentionBackendImpl)
    layers = {
        "mtp.layers.0.self_attn.attn": SimpleNamespace(impl=draft),
        "mtp.layers.1.self_attn.attn": SimpleNamespace(impl=one_query),
        "mtp.layers.2.self_attn.attn": SimpleNamespace(impl=plain),
        "model.layers.3.self_attn.attn": SimpleNamespace(impl=target),
    }
    names = {name for name in layers if name.startswith("mtp.")}
    Eagle3Proposer._mark_one_query_draft_layers(layers, names)
    assert draft.one_query_steps
    assert not one_query.one_query_steps
    assert not plain.one_query_steps
    assert not target.one_query_steps
