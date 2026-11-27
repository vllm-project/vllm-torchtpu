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
"""TPU numerics parity for in-graph RoPE against CPU-built reference tables."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers import rotary_embedding as rope_lib

from vllm_torchtpu import rope_in_graph, rope_in_graph_flag, rope_patch
from vllm_torchtpu.layers.adapter import rotary_embedding as tpu_rope

pytestmark = pytest.mark.tpu_test

_CHUNK_SIZE = 32768


@pytest.mark.parametrize(
    ("name", "head_size", "max_pos", "rope_params"),
    [
        (
            "standard",
            128,
            262144,
            {"rope_type": "default", "rope_theta": 10000.0},
        ),
        (
            "proportional",
            256,
            262144,
            {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        ),
    ],
    ids=["standard", "proportional"],
)
def test_patched_forward_matches_cpu_reference_table_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    head_size: int,
    max_pos: int,
    rope_params: dict[str, Any],
) -> None:
    """Compiled in-graph cos/sin reproduce the CPU-built bf16 table on TPU.

    Covers both standard RoPE (where device `pow` ~8-ulp drift flips several
    percent of bf16 entries by 262,144 positions) and zero-padded proportional
    RoPE (Gemma-4, where recomputing `1 / base ** (2i / d)` on device would
    overwrite the zero frequencies).
    """
    import torch_tpu  # noqa: F401

    # Flag off: the module keeps its table until the post-load sweep converts it.
    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        additional_config={rope_in_graph_flag.KEY: False},
    )
    # Multi-threaded CPU `cos`/`sin` can perturb one thread's slice of the
    # reference table by more than the threshold below.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with set_current_vllm_config(config), torch.device("cpu"):
            rope = rope_lib.get_rope(
                head_size,
                max_position=max_pos,
                is_neox_style=True,
                rope_parameters=dict(rope_params),
                dtype=torch.bfloat16,
            )
            # `_maybe_compact_neox_rope_cache` narrows `rope.cos_sin_cache` on
            # proportional NeoX; compare against the full CPU-built table.
            ref_table = rope._compute_cos_sin_cache().to(torch.bfloat16)
    finally:
        torch.set_num_threads(previous_threads)
    assert ref_table.shape == (max_pos, head_size)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(rope, allow_table_recovery=False)
        == 1
    )

    dev = torch.device("tpu")
    # A NeoX rotation of `[1, ..., 1 | 0, ..., 0]` returns `[cos | sin]`.
    half = head_size // 2
    probe = torch.cat(
        (torch.ones(_CHUNK_SIZE, half), torch.zeros(_CHUNK_SIZE, half)), dim=-1
    )
    probe = probe.to(torch.bfloat16).to(dev)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def cos_sin_rows(positions: torch.Tensor) -> torch.Tensor:
        return rope(positions, probe)[0]

    rows = torch.cat(
        [
            cos_sin_rows(
                torch.arange(start, start + _CHUNK_SIZE, dtype=torch.int32, device=dev)
            ).cpu()
            for start in range(0, max_pos, _CHUNK_SIZE)
        ]
    )
    mismatch = float((rows != ref_table).float().mean())
    assert mismatch < 1e-4, f"{name}: bf16 mismatch {mismatch:.6g} >= 1e-4"


@pytest.mark.parametrize(
    ("name", "head_size", "max_pos", "rope_params"),
    [
        (
            "standard",
            128,
            262144,
            {"rope_type": "default", "rope_theta": 10000.0},
        ),
        (
            "proportional",
            256,
            262144,
            {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        ),
        (
            "llama3",
            128,
            262144,
            {
                "rope_type": "llama3",
                "rope_theta": 500000.0,
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        ),
        (
            "yarn",
            128,
            65536,
            {
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": 4.0,
                "original_max_position_embeddings": 65536,
                "attention_factor": 1.2,
            },
        ),
        (
            "linear",
            64,
            32768,
            {
                "rope_type": "linear",
                "rope_theta": 10000.0,
                "factor": [1.0, 2.0],
            },
        ),
    ],
    ids=["standard", "proportional", "llama3", "yarn", "linear"],
)
def test_in_graph_cos_sin_matches_cpu_reference_table_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    head_size: int,
    max_pos: int,
    rope_params: dict[str, Any],
) -> None:
    import torch_tpu  # noqa: F401

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    dev = torch.device("tpu")

    with set_current_vllm_config(config):
        # Build the CPU reference table single-threaded: on 112-thread hosts,
        # multi-threaded `einsum` + `cos`/`sin` inside `_compute_cos_sin_cache`
        # intermittently perturbs one OpenMP worker slice by ~1.5e-4, flipping
        # >1e-4 of bf16 rounding ties independently of the TPU path under test.
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            config.additional_config[rope_in_graph_flag.KEY] = False
            with torch.device("cpu"):
                ref = rope_lib.get_rope(
                    head_size,
                    max_position=max_pos,
                    is_neox_style=True,
                    rope_parameters=dict(rope_params),
                    dtype=torch.float32,
                )
                # `_maybe_compact_neox_rope_cache` narrows `ref.cos_sin_cache`
                # on proportional NeoX; compare against the full upstream table.
                ref_table = ref._compute_cos_sin_cache().to(torch.float32)
        finally:
            torch.set_num_threads(previous_threads)
        assert isinstance(ref.cos_sin_cache, torch.Tensor)

        config.additional_config[rope_in_graph_flag.KEY] = True
        rope_lib._ROPE_DICT.clear()
        with dev:
            rope = rope_lib.get_rope(
                head_size,
                max_position=max_pos,
                is_neox_style=True,
                rope_parameters=dict(rope_params),
                dtype=torch.bfloat16,
            )

    assert type(rope).__module__ == tpu_rope.__name__
    assert sum(b.numel() for b in rope.buffers()) == 0
    assert isinstance(rope.cos_sin_cache, rope_in_graph.InGraphCosSinCache)
    assert rope.cos_sin_cache.device.type == "tpu"
    assert (
        rope_patch.patch_in_graph_rotary_embedding(rope, allow_table_recovery=False)
        == 0
    )

    spec: rope_in_graph.RopeInGraphSpec = rope._in_graph_spec

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def eval_chunk(pos: torch.Tensor) -> torch.Tensor:
        return rope_in_graph.eval_spec_cos_sin(spec, pos, torch.bfloat16)

    outs = [
        eval_chunk(
            torch.arange(s, s + _CHUNK_SIZE, dtype=torch.int32, device=dev)
        ).cpu()
        for s in range(0, ref_table.shape[0], _CHUNK_SIZE)
    ]
    ingraph_bf16 = torch.cat(outs, dim=0)

    mismatch = float((ingraph_bf16 != ref_table.to(torch.bfloat16)).float().mean())
    max_abs = float((ingraph_bf16.float() - ref_table).abs().max())
    half_bf16_ulp = 2**-8 if float(ref_table.abs().max()) > 1.0 else 2**-9
    bound = half_bf16_ulp + 1e-5
    assert mismatch < 1e-4, f"{name}: bf16 mismatch {mismatch:.6g} >= 1e-4"
    assert max_abs <= bound, f"{name}: max abs diff {max_abs:.6g} > {bound:.6g}"


@pytest.mark.parametrize(
    "case",
    [
        "static",
        "mrope",
        "linear_multi",
        "deepseek_scaling",
        "deepseek_v4_scaling",
    ],
)
def test_compiled_tier1_forward_matches_table_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """Compiled Tier-1 forward matches flag-off TPU table across rotary families."""
    import torch_tpu  # noqa: F401
    from vllm.model_executor.layers.rotary_embedding import (
        DeepseekScalingRotaryEmbedding,
        LinearScalingRotaryEmbedding,
        MRotaryEmbedding,
        RotaryEmbedding,
    )
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekV4ScalingRotaryEmbedding,
    )

    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: F401

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    dev = torch.device("tpu")

    def _build() -> torch.nn.Module:
        if case == "static":
            return RotaryEmbedding(64, 64, 512, 10000.0, True, torch.bfloat16)
        if case == "mrope":
            return MRotaryEmbedding(
                64,
                64,
                128,
                10000.0,
                True,
                torch.bfloat16,
                mrope_section=[8, 12, 12],
            )
        if case == "linear_multi":
            return LinearScalingRotaryEmbedding(
                64, 64, 128, 10000.0, True, [1.0, 2.0], torch.bfloat16
            )
        if case == "deepseek_v4_scaling":
            return DeepseekV4ScalingRotaryEmbedding(
                128,
                64,
                128,
                10000.0,
                False,
                4.0,
                torch.bfloat16,
                beta_fast=32,
                beta_slow=1,
                mscale=0.707,
                mscale_all_dim=0,
            )
        return DeepseekScalingRotaryEmbedding(
            64,
            64,
            128,
            10000.0,
            False,
            4.0,
            torch.bfloat16,
            beta_fast=32,
            beta_slow=1,
            mscale=0.707,
            mscale_all_dim=0,
        )

    with set_current_vllm_config(config):
        config.additional_config[rope_in_graph_flag.KEY] = False
        with dev:
            ref = _build()
        config.additional_config[rope_in_graph_flag.KEY] = True
        with dev:
            uut = _build()

    assert sum(b.numel() for b in uut.buffers()) == 0
    assert isinstance(uut.cos_sin_cache, rope_in_graph.InGraphCosSinCache)

    g = torch.Generator().manual_seed(7)
    n_tok = 16
    if case == "mrope":
        pos = torch.stack(
            [
                torch.arange(n_tok, dtype=torch.int32),
                torch.arange(n_tok, dtype=torch.int32) * 2,
                torch.arange(n_tok, dtype=torch.int32) * 3,
            ],
            dim=0,
        ).to(dev)
        offsets: torch.Tensor | None = None
    elif case == "linear_multi":
        pos = torch.arange(n_tok, dtype=torch.int32, device=dev)
        offsets = torch.cat(
            [
                torch.zeros(n_tok // 2, dtype=torch.int32),
                torch.full((n_tok // 2,), 128, dtype=torch.int32),
            ]
        ).to(dev)
    else:
        pos = torch.arange(n_tok, dtype=torch.int32, device=dev)
        offsets = None

    if case == "deepseek_scaling":
        q = torch.randn(n_tok, 4, 64, generator=g, dtype=torch.bfloat16).to(dev)
        k = torch.randn(n_tok, 1, 64, generator=g, dtype=torch.bfloat16).to(dev)
    elif case == "deepseek_v4_scaling":
        q = torch.randn(n_tok, 4, 128, generator=g, dtype=torch.bfloat16).to(dev)
        k = torch.randn(n_tok, 1, 128, generator=g, dtype=torch.bfloat16).to(dev)
    else:
        q = torch.randn(n_tok, 4 * 64, generator=g, dtype=torch.bfloat16).to(dev)
        k = torch.randn(n_tok, 2 * 64, generator=g, dtype=torch.bfloat16).to(dev)

    if offsets is not None:
        ref_fn = torch.compile(
            lambda p, q_, k_, o: ref(p, q_, k_, offsets=o),
            backend="tpu",
            dynamic=False,
        )
        uut_fn = torch.compile(
            lambda p, q_, k_, o: uut(p, q_, k_, offsets=o),
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        ref_q, ref_k = ref_fn(pos, q.clone(), k.clone(), offsets)
        uut_q, uut_k = uut_fn(pos, q.clone(), k.clone(), offsets)
    else:
        ref_fn = torch.compile(
            lambda p, q_, k_: ref(p, q_, k_), backend="tpu", dynamic=False
        )
        uut_fn = torch.compile(
            lambda p, q_, k_: uut(p, q_, k_),
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        ref_q, ref_k = ref_fn(pos, q.clone(), k.clone())
        uut_q, uut_k = uut_fn(pos, q.clone(), k.clone())

    torch.testing.assert_close(uut_q.cpu(), ref_q.cpu(), rtol=2e-2, atol=2e-2)
    assert ref_k is not None and uut_k is not None
    torch.testing.assert_close(uut_k.cpu(), ref_k.cpu(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("in_graph", [False, True])
def test_mla_wrapper_process_weights_after_loading_tracks_tpu_device(
    monkeypatch: pytest.MonkeyPatch,
    in_graph: bool,
) -> None:
    """`process_weights_after_loading` moves `cos_sin_cache` and its device."""
    import torch_tpu  # noqa: F401
    from vllm.model_executor.layers.rotary_embedding import (
        DeepseekScalingRotaryEmbedding,
    )

    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        additional_config={rope_in_graph_flag.KEY: in_graph},
    )

    wrapper = object.__new__(mla_attention_op.VllmTPUMultiHeadLatentAttentionWrapper)
    torch.nn.Module.__init__(wrapper)
    with set_current_vllm_config(config), torch.device("cpu"):
        wrapper.rotary_emb = DeepseekScalingRotaryEmbedding(
            64,
            64,
            128,
            10000.0,
            False,
            4.0,
            torch.bfloat16,
            beta_fast=32,
            beta_slow=1,
            mscale=0.707,
            mscale_all_dim=0,
        )

    wrapper.process_weights_after_loading(torch.bfloat16)

    if in_graph:
        cache = wrapper.rotary_emb.cos_sin_cache
        assert isinstance(cache, rope_in_graph.InGraphCosSinCache)
        assert cache.device.type == "tpu"
        assert wrapper.rotary_emb._in_graph_rope_device == cache.device
        cos, sin = wrapper.rotary_emb.get_cos_sin(8)
        assert cos.device.type == "tpu"
        assert sin.device.type == "tpu"
    else:
        assert isinstance(wrapper.rotary_emb.cos_sin_cache, torch.Tensor)
        assert wrapper.rotary_emb.cos_sin_cache.device.type == "tpu"


def test_deepseek_v4_cos_sin_at_matches_cpu_table_on_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cos_sin_at` on TPU reproduces DSv4's fp32 CPU table."""
    import torch_tpu  # noqa: F401
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekV4ScalingRotaryEmbedding,
    )

    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: F401

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    dev = torch.device("tpu")

    def _make() -> DeepseekV4ScalingRotaryEmbedding:
        return DeepseekV4ScalingRotaryEmbedding(
            128,
            64,
            128,
            10000.0,
            False,
            4.0,
            torch.bfloat16,
            beta_fast=32,
            beta_slow=1,
            mscale=0.707,
            mscale_all_dim=0,
        )

    with set_current_vllm_config(config):
        config.additional_config[rope_in_graph_flag.KEY] = False
        with torch.device("cpu"):
            ref = _make()
        config.additional_config[rope_in_graph_flag.KEY] = True
        with dev:
            uut = _make()

    assert sum(b.numel() for b in uut.buffers()) == 0
    assert isinstance(uut.cos_sin_cache, rope_in_graph.InGraphCosSinCache)
    ref_table = ref.cos_sin_cache
    assert isinstance(ref_table, torch.Tensor) and ref_table.dtype == torch.float32

    pos = torch.arange(ref_table.shape[0], dtype=torch.int32, device=dev)
    compiled_rows = torch.compile(
        lambda p: rope_in_graph.cos_sin_at(uut, p),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
    )(pos)
    assert compiled_rows.dtype == torch.float32
    torch.testing.assert_close(compiled_rows.cpu(), ref_table, rtol=1e-4, atol=1e-4)
