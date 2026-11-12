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
"""A Qwen3.5 prefill engine and its decode engine page their seq-along-lane
KV caches alike, so their Raiden layout fingerprints agree.

The fingerprint of the seq-along-lane FA pool pins the batched-RPA kernel
page. A TP1 prefill runs 2304-token blocks and a TP8 decode 640-token blocks;
vLLM's own choice, the largest supported page that divides the block, gives
256 and 128. The fingerprints then differ and the source controller refuses
every transfer ("Layout fingerprint mismatch between source and
destination"), which is what the pair below reproduces without the rule.
"""

import contextlib
import os
from types import SimpleNamespace

import pytest
from vllm.v1.worker.utils import select_common_block_size

from vllm_torchtpu.distributed.kv_transfer.raiden import pool_manifest as rpm
from vllm_torchtpu.distributed.kv_transfer.raiden import seq_on_lane as sol
from vllm_torchtpu.layers.adapter.attention import PallasBatchedRPAAttentionBackend

from .raiden_test_utils import FakeTensor

BATCHED = PallasBatchedRPAAttentionBackend
QWEN35 = "Qwen3_5MoeForConditionalGeneration"
OTHER = "Qwen3NextForCausalLM"

# Block sizes on main's GDN state sizing (bf16 SSM, fp32 conv).
PREFILL_BLOCK = 2304  # TP1 (pipeline-parallel) prefill
DECODE_BLOCK = 640  # TP8 decode

# Qwen3.5-397B: 2 KV heads of head_dim 256 in fp8; GDN 16 key and 64 value
# heads of dim 128. A TP1 prefill holds every head, a TP8 decode an eighth of
# the GDN heads and one KV head.
TP1_GDN = rpm.GdnHeadGeometry(
    local_key_heads=16, local_value_heads=64, key_head_dim=128, value_head_dim=128
)
TP8_GDN = rpm.GdnHeadGeometry(
    local_key_heads=2, local_value_heads=8, key_head_dim=128, value_head_dim=128
)
SOL_LAYOUT = ((4, 3, 2, 1, 0), ((4, 128), (4, 1)), 8)


@contextlib.contextmanager
def _engine(monkeypatch, *, kv_role, arch=QWEN35, layout="HND", pcp=1):
    """A current vLLM config for one engine: the KV cache layout resolved as
    the runner resolves it, the model's architecture, and a KV transfer
    config with ``kv_role`` (None: no KV transfer)."""
    from vllm import envs as vllm_envs
    from vllm.config import (
        DeviceConfig,
        KVTransferConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.attention import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from vllm_torchtpu import patch_registry
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    was_cached = vllm_envs._is_envs_cache_enabled()
    vllm_envs.disable_envs_cache()
    try:
        monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", layout)
        # Building a VllmConfig runs the platform's check_and_update_config.
        # That applies the platform patches for the rest of the process (one
        # replaces torch_tpu's Pallas shape conversion, which later tests rely
        # on) and edits the TorchTPU bootstrap environment. The page selection
        # needs neither, so stub both, as the platform tests do, and let
        # monkeypatch restore the one variable the check pops.
        monkeypatch.setattr(patch_registry, "apply", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            TpuPlatform, "_prepare_singlehost_tpu_env", lambda *args, **kwargs: None
        )
        monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
        config = VllmConfig(
            device_config=DeviceConfig(device="cpu"),
            attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
        )
        resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
        config.parallel_config.prefill_context_parallel_size = pcp
        config.model_config = SimpleNamespace(architectures=[arch])
        if kv_role is not None:
            config.kv_transfer_config = KVTransferConfig(
                kv_connector="TPURaidenConnector", kv_role=kv_role
            )
        with set_current_vllm_config(config):
            yield config
    finally:
        monkeypatch.undo()
        if was_cached:
            vllm_envs.enable_envs_cache()


def _kernel_page(monkeypatch, block, **engine):
    """The kernel page vLLM's selection gives this engine's blocks."""
    with _engine(monkeypatch, **engine):
        return select_common_block_size(block, [BATCHED])


def _leg_fingerprint(*, block, kernel_page, num_kv_heads, gdn):
    """The seq-along-lane layout fingerprint of one engine whose FA cache is
    paged at ``kernel_page`` (two blocks, one FA and one GDN layer)."""
    blocks = 2
    fa_name = "model.layers.3.self_attn.attn"
    gdn_name = "model.layers.0.linear_attn"
    fa_shape = (blocks * block // kernel_page, 2 * num_kv_heads, 64, 4, kernel_page)
    conv = FakeTensor((blocks, 3, gdn.conv_dim), 4, dtype="torch.float32")
    ssm = FakeTensor(
        (blocks, gdn.local_value_heads, 128, 128), 2, dtype="torch.bfloat16"
    )
    manifest = sol.build_qwen35_pool_manifest_sol(
        named_kv_caches={
            fa_name: FakeTensor(fa_shape, 1, dtype="torch.float8_e4m3fn"),
            gdn_name: (conv, ssm),
        },
        kv_cache_groups=(
            SimpleNamespace(
                layer_names=(fa_name,),
                kv_cache_spec=SimpleNamespace(
                    block_size=block, num_kv_heads=num_kv_heads, head_size=256
                ),
            ),
            SimpleNamespace(layer_names=(gdn_name,), kv_cache_spec=SimpleNamespace()),
        ),
        raw_tensors=(),
        gdn_geometry=gdn,
    )
    assert sol.sol_fa_page_tokens(manifest) == block
    return sol.measured_sol_layout_fingerprint(
        manifest,
        layout_getter=lambda tensor: SOL_LAYOUT,
        package_version=lambda name: f"{name}-test",
    )


def test_prefill_and_decode_register_the_same_layout(monkeypatch):
    prefill_page = _kernel_page(monkeypatch, PREFILL_BLOCK, kv_role="kv_producer")
    decode_page = _kernel_page(monkeypatch, DECODE_BLOCK, kv_role="kv_consumer")
    prefill_fp, prefill_payload = _leg_fingerprint(
        block=PREFILL_BLOCK, kernel_page=prefill_page, num_kv_heads=2, gdn=TP1_GDN
    )
    decode_fp, decode_payload = _leg_fingerprint(
        block=DECODE_BLOCK, kernel_page=decode_page, num_kv_heads=1, gdn=TP8_GDN
    )
    assert prefill_payload == decode_payload
    assert prefill_fp == decode_fp
    assert prefill_payload["fa_kernel_page_tokens"] == 128


def test_differing_pages_are_refused_by_the_fingerprint():
    # What the pair registered without the rule: the payloads differ only in
    # the kernel page.
    fp_256, payload_256 = _leg_fingerprint(
        block=PREFILL_BLOCK, kernel_page=256, num_kv_heads=2, gdn=TP1_GDN
    )
    fp_128, payload_128 = _leg_fingerprint(
        block=DECODE_BLOCK, kernel_page=128, num_kv_heads=1, gdn=TP8_GDN
    )
    assert fp_256 != fp_128
    assert {k for k in payload_256 if payload_256[k] != payload_128[k]} == {
        "fa_kernel_page_tokens"
    }


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer", "kv_both"])
def test_qwen3_5_engines_with_kv_transfer_page_at_128(monkeypatch, kv_role):
    # kv_both is the role an offloading connector inside a prefill engine sees
    # in its copy of the engine config: it must resolve the runner's page.
    with _engine(monkeypatch, kv_role=kv_role):
        assert BATCHED.get_supported_kernel_block_sizes() == [128]
    for block in (PREFILL_BLOCK, DECODE_BLOCK, 1280, 2560):
        assert _kernel_page(monkeypatch, block, kv_role=kv_role) == 128


def test_other_models_keep_vllms_page(monkeypatch):
    with _engine(monkeypatch, kv_role="kv_producer", arch=OTHER):
        assert BATCHED.get_supported_kernel_block_sizes() == [128, 256]
    assert (
        _kernel_page(monkeypatch, PREFILL_BLOCK, kv_role="kv_producer", arch=OTHER)
        == 256
    )
    assert (
        _kernel_page(monkeypatch, DECODE_BLOCK, kv_role="kv_consumer", arch=OTHER)
        == 128
    )


def test_qwen3_5_without_kv_transfer_keeps_vllms_page(monkeypatch):
    with _engine(monkeypatch, kv_role=None):
        assert BATCHED.get_supported_kernel_block_sizes() == [128, 256]
    assert _kernel_page(monkeypatch, PREFILL_BLOCK, kv_role=None) == 256


def test_head_along_sublane_keeps_vllms_page(monkeypatch):
    with _engine(monkeypatch, kv_role="kv_producer", layout="NHD"):
        assert BATCHED.get_supported_kernel_block_sizes() == [128, 256]


def test_pcp_streaming_keeps_its_pages(monkeypatch):
    with _engine(monkeypatch, kv_role="kv_producer", pcp=8):
        assert BATCHED.get_supported_kernel_block_sizes() == [
            128,
            256,
            512,
            1024,
            2048,
            4096,
        ]


def test_engine_configs_leave_no_global_state(monkeypatch):
    # Later tests in the process must see torch_tpu's own Pallas shape
    # conversion and an unchanged environment. Run first or alone, this fails
    # if building an engine config applies the platform patches.
    from torch_tpu._internal.pallas import pallas

    shapes = (pallas.get_global_shape, pallas.get_local_shape)
    environ = dict(os.environ)
    with _engine(monkeypatch, kv_role="kv_producer"):
        pass
    assert (pallas.get_global_shape, pallas.get_local_shape) == shapes
    assert dict(os.environ) == environ
