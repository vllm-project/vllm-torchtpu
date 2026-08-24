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

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.flash_attention import kernel, tuned_params
from vllm_torchtpu.kernels.flash_attention.kernel import (BlockSizes,
                                                          SegmentIds,
                                                          flash_attention)
from vllm_torchtpu.kernels.flash_attention.tuned_params import (
    TunableParams, get_tuned_params, make_tuning_key, tuned_params_mapping)

_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def _qkv(num_heads: int, seq_len: int, head_dim: int):
    qkv = jax.ShapeDtypeStruct((1, num_heads, seq_len, head_dim), jnp.bfloat16)
    segment = jax.ShapeDtypeStruct((1, seq_len), jnp.int32)
    return qkv, SegmentIds(q=segment, kv=segment)


def _tuning_params(num_heads: int,
                   seq_len: int,
                   head_dim: int,
                   *,
                   has_segment_ids: bool = True) -> TunableParams | None:
    qkv, _ = _qkv(num_heads, seq_len, head_dim)
    key = make_tuning_key(
        qkv,
        qkv,
        qkv,
        causal=False,
        has_segment_ids=has_segment_ids,
        has_attention_bias=False,
        vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        device_kind="TPU7x",
    )
    return get_tuned_params(key)


def test_tuned_entries_use_valid_blocks() -> None:
    for key, params in tuned_params_mapping.items():
        assert params.block_q <= key.q_seq_len
        assert key.kv_seq_len % params.block_k_major == 0
        assert params.block_k_major % params.block_k == 0


def test_tuning_table_has_requested_vision_grids() -> None:
    kimi_seq_lens = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                     67328)
    qwen_seq_lens = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
    expected = ({(12, 128, seq_len)
                 for seq_len in kimi_seq_lens}
                | {(16, 80, seq_len)
                   for seq_len in qwen_seq_lens})
    actual = {(key.num_heads, key.head_dim, key.q_seq_len)
              for key in tuned_params_mapping}

    assert actual == expected


@pytest.mark.parametrize(
    ("num_heads", "seq_len", "head_dim", "expected"),
    (
        (12, 256, 128, TunableParams(256, 256, 256, 1)),
        (12, 4096, 128, TunableParams(1024, 4096, 512, 1)),
        (12, 16384, 128, TunableParams(1024, 8192, 512, 1)),
        (12, 65536, 128, TunableParams(512, 16384, 512, 1)),
        (12, 67328, 128, TunableParams(8192, 256, 256, 1)),
        (16, 256, 80, TunableParams(256, 256, 256, 1)),
        (16, 4096, 80, TunableParams(1024, 4096, 512, 1)),
        (16, 16384, 80, TunableParams(512, 16384, 512, 1)),
        (16, 65536, 80, TunableParams(512, 16384, 512, 1)),
    ),
)
def test_representative_vision_tuning(num_heads: int, seq_len: int,
                                      head_dim: int,
                                      expected: TunableParams) -> None:
    assert _tuning_params(num_heads, seq_len, head_dim) == expected


def test_tuning_lookup_requires_exact_setup() -> None:
    assert _tuning_params(12, 65536, 128, has_segment_ids=False) is None
    assert _tuning_params(12, 44032, 128) is None
    assert _tuning_params(12, 67200, 128) is None


def _selected_block_sizes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seq_len: int,
    explicit: BlockSizes | None = None,
    vmem_limit_bytes: int | None = _VMEM_LIMIT_BYTES,
) -> BlockSizes:
    qkv, segment_ids = _qkv(1, seq_len, 128)
    selected = []

    def capture(*args, **_kwargs):
        selected.append(args[8])
        return args[0]

    monkeypatch.setattr(tuned_params, "current_device_kind", lambda: "TPU7x")
    monkeypatch.setattr(
        kernel.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(num_lanes=128,
                                num_sublanes=8,
                                vmem_capacity_bytes=_VMEM_LIMIT_BYTES),
    )
    monkeypatch.setattr(kernel, "_flash_attention", capture)
    kernel.flash_attention.__wrapped__(
        qkv,
        qkv,
        qkv,
        segment_ids=segment_ids,
        block_sizes=explicit,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return selected[0]


def test_fitting_unknown_shape_uses_single_step(
        monkeypatch: pytest.MonkeyPatch) -> None:
    assert _selected_block_sizes(monkeypatch, seq_len=21760) == BlockSizes(
        128, 21760, 21760, 1)


def test_unspecified_vmem_limit_uses_device_capacity(
        monkeypatch: pytest.MonkeyPatch) -> None:
    assert _selected_block_sizes(monkeypatch,
                                 seq_len=21760,
                                 vmem_limit_bytes=None) == BlockSizes(
                                     128, 21760, 21760, 1)


def test_oversized_unknown_shape_uses_safe_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    assert _selected_block_sizes(monkeypatch,
                                 seq_len=44032) == BlockSizes.get_default(
                                     1, 1, 44032, 44032, 128)


def test_explicit_block_sizes_take_precedence(
        monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = BlockSizes(256, 512, 256, 1)

    assert _selected_block_sizes(monkeypatch, seq_len=21760,
                                 explicit=explicit) == explicit


@pytest.mark.parametrize(
    ("num_heads", "seq_len", "head_dim", "has_tuning"),
    (
        (12, 21760, 128, False),
        (12, 44032, 128, False),
        (12, 65536, 128, True),
        (12, 67200, 128, False),
        (12, 67328, 128, True),
        (16, 32768, 80, True),
        (16, 65536, 80, True),
    ),
)
def test_profile_compiles(num_heads: int, seq_len: int, head_dim: int,
                          has_tuning: bool) -> None:
    devices = jax.local_devices()
    if not devices or devices[0].platform != "tpu":
        pytest.skip("requires a TPU compiler")
    if devices[0].device_kind != "TPU7x":
        pytest.skip("tuned for TPU7x")

    qkv, segment_ids = _qkv(num_heads, seq_len, head_dim)
    assert (_tuning_params(num_heads, seq_len, head_dim)
            is not None) is has_tuning

    try:
        flash_attention.lower(
            qkv,
            qkv,
            qkv,
            segment_ids=segment_ids,
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ).compile()
    finally:
        flash_attention.clear_cache()
