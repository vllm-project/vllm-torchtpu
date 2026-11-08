# SPDX-License-Identifier: Apache-2.0
"""Cache summaries retain round ownership without retaining live cache data."""

import logging
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
import torch

from vllm_torchtpu.runner.kv_cache_nan_flags import CacheNaNLayer, KVCacheNaNDetector


def test_round_accumulates_chunks_and_snapshots_ids_before_next_round(caplog):
    def inspect(value, armed):
        return torch.tensor(value) & armed[0]

    detector = KVCacheNaNDetector(
        (CacheNaNLayer("a", 7, inspect), CacheNaNLayer("b", 19, inspect)),
        torch.device("cpu"),
    )
    assert detector.readback() is None
    detector.arm()
    detector.begin_round(["req-old"])
    detector.record_chunk({"a": False, "b": True})
    detector.record_chunk({"a": False, "b": False})
    assert detector.flags.tolist() == [False, True]
    # Device-to-host copies own distinct destinations, unlike CPU Tensor.to.
    with (
        patch.object(torch.Tensor, "to", lambda tensor, *a, **kw: tensor.clone()),
        patch("vllm_torchtpu.runner.kv_cache_nan_flags.TpuEvent"),
    ):
        bad = detector.readback()
        detector.begin_round(["req-new"])
        detector.record_chunk({"a": False, "b": False})
        good = detector.readback()
    with caplog.at_level(
        logging.ERROR, logger="vllm_torchtpu.runner.kv_cache_nan_flags"
    ):
        good.consume()
        bad.consume()
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "round=1" in message and "req-old" in message and "req-new" not in message
    assert "layer_ids=[7, 19]" in message and "isnonfinite=[false, true]" in message


def test_multiple_cache_modules_in_one_decoder_layer_share_one_flag():
    def inspect(value, armed):
        return torch.tensor(value) & armed[0]

    detector = KVCacheNaNDetector(
        (CacheNaNLayer("a", 7, inspect), CacheNaNLayer("b", 7, inspect)),
        torch.device("cpu"),
    )
    detector.arm()
    detector.begin_round(["r"])
    detector.record_chunk({"a": True, "b": False})
    assert detector.layer_ids == (7,)
    assert detector.flags.tolist() == [True]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("nested_config", [False, True])
def test_install_binds_target_layers_and_reads_current_device_metadata(
    monkeypatch, batched, nested_config
):
    from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_torchtpu.layers.adapter.attention import (
        PallasAttentionBackend,
        PallasAttentionBackendImpl,
        PallasBatchedRPAAttentionBackend,
    )
    from vllm_torchtpu.layers.adapter.custom_ops.kv_cache_nan_flags_op import (
        inspect_attention_cache,
    )
    from vllm_torchtpu.runner import kv_cache_nan_flags

    # Real registered layer types with CPU storage; only the TPU bridge is
    # replaced. The replacement runs the production JAX page reader/reducer.
    def make_attention():
        module = Attention.__new__(Attention)
        torch.nn.Module.__init__(module)
        module.impl = PallasAttentionBackendImpl.__new__(PallasAttentionBackendImpl)
        module.impl._bm_region = None
        module.impl._pool_is_seq_along_lane = False
        module.attn_backend = (
            PallasBatchedRPAAttentionBackend if batched else PallasAttentionBackend
        )
        module.kv_cache = torch.zeros((4, 4, 1, 2, 128), dtype=torch.bfloat16)
        return module

    def cpu_op(shape, heads, width, seq_along_lane, block_major):
        inspect = jax.jit(
            partial(
                inspect_attention_cache,
                native_shape=shape,
                num_kv_heads=heads,
                head_size=width,
                seq_along_lane=seq_along_lane,
                block_major=block_major,
            )
        )

        def call(cache, tables, lengths, starts, armed):
            # Torch BF16 cannot use numpy directly; conversion here is only
            # CPU test transport. Production passes the original device dtype.
            arrays = [
                jnp.array(t.float().numpy())
                if t.dtype == torch.bfloat16
                else jnp.array(t.numpy())
                for t in (cache, tables, lengths, starts)
            ]
            return torch.tensor(bool(armed[0]) and bool(inspect(*arrays)))

        return call

    monkeypatch.setattr(kv_cache_nan_flags, "attention_nan_op", cpu_op)
    attention, draft = make_attention(), make_attention()
    model = torch.nn.Module()
    model.add_module("attn", attention)
    name, draft_name = "model.layers.7.self_attn.attn", "draft_model.layers.0.attn"
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    config = VllmConfig()
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    config.compilation_config.static_forward_context = {
        name: attention,
        draft_name: draft,
    }
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        use_spmd=False,
        is_pooling_model=False,
        device=torch.device("cpu"),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[name, draft_name],
                    kv_cache_spec=spec,
                )
            ]
        ),
        vllm_config=config,
        model=model,
        _kv_cache_bundle=None,
        _kv_cache_bundle_layer_index={},
    )
    # capture_model installs the detector outside a current-config scope. The
    # real batched backend reads that scope even for its ordinary LBNHC shape.
    outer_config = VllmConfig() if nested_config else None
    with set_current_vllm_config(outer_config) if nested_config else nullcontext():
        detector = kv_cache_nan_flags.install_kv_cache_nan_detector(runner)
        if nested_config:
            assert get_current_vllm_config() is outer_config
        else:
            with pytest.raises(AssertionError, match="Current vLLM config is not set"):
                get_current_vllm_config()
    assert detector.layer_ids == (7,)
    detector.arm()
    detector.begin_round(["r"])
    attention.kv_cache[2, 0, 0, 1, 127] = float("nan")
    meta = SimpleNamespace(
        block_tables=torch.tensor([2, 0]),
        seq_lens=torch.tensor([2]),
        query_start_loc=torch.tensor([0, 1]),
    )
    detector.record_chunk({name: meta})
    assert detector.flags.tolist() == [True]
    detector.begin_round(["r"])
    meta.seq_lens = torch.tensor([1])  # Actual corrected prefix is empty.
    detector.record_chunk({name: meta})
    assert detector.flags.tolist() == [False]

    for field in (
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        setattr(runner.parallel_config, field, 2)
        with pytest.raises(ValueError, match="PP=PCP=DCP=1"):
            kv_cache_nan_flags.install_kv_cache_nan_detector(runner)
        setattr(runner.parallel_config, field, 1)

    runner._two_slot_ckpts = True
    with pytest.raises(ValueError, match="TPU_GDN_TWO_SLOT_CKPT"):
        kv_cache_nan_flags.install_kv_cache_nan_detector(runner)
    runner._two_slot_ckpts = False

    runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec = replace(
        spec, dtype=torch.uint8
    )
    with pytest.raises(ValueError, match="floating FA cache storage"):
        kv_cache_nan_flags.install_kv_cache_nan_detector(runner)
