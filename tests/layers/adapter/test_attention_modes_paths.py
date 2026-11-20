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
"""Reduced-precision attention modes reach only kernels that run them.

The PCP streaming kernel, kernel entries without `attention_modes` and the
bundled block-major kernel refuse a layer that asks for modes instead of
dropping them, and the export path builds the MTP draft's attention kernels
with every mode off.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

from vllm_torchtpu.experimental.export import universal_adapter
from vllm_torchtpu.kernels.experimental.batched_rpa.configs import (
    Fp8Operand,
    NativeFp8,
)
from vllm_torchtpu.layers.adapter import attention, native_fp8_scope
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

_MODES = native_fp8_scope.AttentionModes(
    native_fp8=NativeFp8(pv=Fp8Operand.STATIC), decode_exp2=True
)


@pytest.fixture(autouse=True)
def cpu_vllm_config_context():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


def _impl(cls, modes=_MODES, draft=False, head_size=256):
    impl = cls(
        num_heads=8,
        head_size=head_size,
        scale=1 / 16,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8",
    )
    impl.attention_modes = modes
    impl.is_attention_draft = draft
    return impl


def _context():
    return set_vllm_model_wrapper_context(
        mesh=None,
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)),
    )


def test_pcp_streaming_refuses_a_layer_with_modes():
    impl = _impl(attention.PallasBatchedRPAAttentionBackendImpl)
    with _context(), pytest.raises(NotImplementedError, match="PCP streaming"):
        impl._build_rpa_kernel(None, None, None, use_pcp_streaming=True)


def test_head_dim_64_refuses_a_layer_with_modes():
    """head_dim 64 runs ragged_paged_attention_hd64 whatever the entry, and
    that kernel has no modes."""
    impl = _impl(attention.PallasBatchedRPAAttentionBackendImpl, head_size=64)
    with _context(), pytest.raises(NotImplementedError, match="hd64"):
        impl._build_rpa_kernel(None, None, None)


def test_an_entry_without_modes_refuses_a_layer_with_modes():
    impl = _impl(attention.PallasAttentionBackendImpl)
    with _context(), pytest.raises(NotImplementedError, match="no reduced-precision"):
        impl._build_rpa_kernel(None, None, None)


def test_the_bundled_kernel_refuses_a_layer_with_modes():
    impl = _impl(attention.PallasAttentionBackendImpl)
    with _context(), pytest.raises(NotImplementedError, match="BLOCK_MAJOR"):
        impl._build_bundled_kernel(None, None, None)


class _PastTheRefusal(Exception):
    """Raised by the step after the modes check."""


def _past_the_refusal(*args, **kwargs):
    raise _PastTheRefusal


@pytest.mark.parametrize(
    ("cls", "head_size", "build", "next_step"),
    [
        (
            attention.PallasAttentionBackendImpl,
            256,
            lambda impl: impl._build_rpa_kernel(
                None, None, None, use_pcp_streaming=True
            ),
            "_select_kernel_mesh",
        ),
        (
            attention.PallasBatchedRPAAttentionBackendImpl,
            64,
            lambda impl: impl._build_rpa_kernel(None, None, None),
            "_select_kernel_mesh",
        ),
        (
            attention.PallasAttentionBackendImpl,
            256,
            lambda impl: impl._build_bundled_kernel(None, None, None),
            "get_vllm_model_wrapper_context",
        ),
    ],
    ids=["pcp-streaming", "head-dim-64", "bundled"],
)
def test_a_draft_layer_with_target_modes_is_not_refused(
    monkeypatch, cls, head_size, build, next_step
):
    """A draft runs every mode off, so the refusal does not apply to it: the
    build goes on to its next step."""
    impl = _impl(cls, draft=True, head_size=head_size)
    if next_step == "_select_kernel_mesh":
        monkeypatch.setattr(impl, next_step, _past_the_refusal)
    else:
        monkeypatch.setattr(attention, next_step, _past_the_refusal)
    with _context(), pytest.raises(_PastTheRefusal):
        build(impl)


def test_export_marks_the_mtp_drafts_attention(monkeypatch):
    class FakeAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.impl = object()

    monkeypatch.setattr(
        "vllm.model_executor.layers.attention_layer_base.AttentionLayerBase",
        FakeAttention,
    )
    target_attn, draft_attn = FakeAttention(), FakeAttention()
    draft = torch.nn.Sequential(torch.nn.Linear(2, 2), draft_attn)
    impls = universal_adapter._draft_attention_impls(draft)
    assert id(draft_attn.impl) in impls
    assert id(target_attn.impl) not in impls
    assert universal_adapter._draft_attention_impls(None) == set()


def test_export_builds_the_mtp_drafts_kernels_with_modes_off(monkeypatch):
    """load_vllm_model marks the draft's attention before building its
    kernels, so the draft's kernels run every mode off and the target's keep
    the target's modes."""
    from vllm.model_executor.layers.attention import Attention
    from vllm.platforms import current_platform

    def attention_layer(impl):
        layer = object.__new__(Attention)
        torch.nn.Module.__init__(layer)
        layer.impl = impl
        return layer

    target_attn = attention_layer(_impl(attention.PallasAttentionBackendImpl))
    draft_attn = attention_layer(_impl(attention.PallasAttentionBackendImpl))
    target, draft = torch.nn.Sequential(target_attn), torch.nn.Sequential(draft_attn)
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.compilation_config.static_forward_context.update(
        {
            "model.layers.3.self_attn.attn": target_attn,
            "mtp.layers.0.self_attn.attn": draft_attn,
        }
    )
    built = {}

    def initialize_kernel(impl, layer):
        built[id(layer)] = native_fp8_scope.effective(
            impl.attention_modes, impl.is_attention_draft
        )

    monkeypatch.setattr(
        attention.PallasAttentionBackendImpl, "initialize_kernel", initialize_kernel
    )
    monkeypatch.setattr(
        universal_adapter, "init_export_distributed_environment", lambda cfg: None
    )
    monkeypatch.setattr(
        "vllm.distributed.ensure_model_parallel_initialized", lambda *args: None
    )
    monkeypatch.setattr(
        universal_adapter, "_create_single_device_tpu_mesh", lambda: None
    )
    monkeypatch.setattr(universal_adapter, "run_model_loader", lambda cfg: target)
    monkeypatch.setattr(universal_adapter, "_load_mtp_draft", lambda cfg, t: draft)
    monkeypatch.setattr(
        current_platform, "update_block_size_for_backend", lambda cfg: None
    )

    model = universal_adapter.load_vllm_model(config, load_format="dummy")

    assert built == {id(target_attn): _MODES, id(draft_attn): native_fp8_scope.OFF}
    assert getattr(model, universal_adapter.MTP_DRAFT_ATTR) is draft
