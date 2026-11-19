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
"""TPU tests for Mooncake KV and hidden-state caching on AR stages."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

BLOCK_SIZE = 16
HIDDEN_SIZE = 8
HIDDEN_LAYER = 1
MAX_HS_BYTES = 256 * 1024 * 1024


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_mooncake_kv_and_hidden_state_caching(tmp_path: Path) -> None:
    """Mooncake KV & Hidden-State Caching (AR Stages).

    - TPU path: import ``MooncakeConfig`` and ``thinker_captures``, validate
      config parsing, produce bf16 Thinker embedding and hidden-state captures
      on ``torch.device("tpu")``, and round-trip encode/decode + ``RequestHS``
      prefix assembly.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``MooncakeConfig.from_dict`` preserves explicit identity and defaults.
    - Outcome check: decoded block tensors and restored+appended ``RequestHS``
      prefix+tail captures match CPU reference tensors bit-for-bit.
    """
    from vllm_omni.distributed.kv_transfer.mooncake.config import MooncakeConfig
    from vllm_omni.distributed.kv_transfer.mooncake.thinker_captures import (
        HSBlockSpec,
        RequestHS,
        decode_hs,
        encode_hs,
        hs_key,
        prefix_digest,
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    cfg_file = tmp_path / "mooncake_store.json"
    cfg_file.write_text('{"local_hostname": "127.0.0.1"}\n', encoding="utf-8")
    cfg = MooncakeConfig.from_dict(
        {
            "config_file": str(cfg_file),
            "namespace": "tpu-qwen3-omni",
            "model_revision": "rev-0",
        }
    )
    assert cfg.config_file == str(cfg_file), (
        f"config_file mismatch: {cfg.config_file!r} != {str(cfg_file)!r}"
    )
    assert cfg.namespace == "tpu-qwen3-omni", f"namespace mismatch: {cfg.namespace!r}"
    assert cfg.max_hs_bytes == MAX_HS_BYTES, (
        f"max_hs_bytes default {cfg.max_hs_bytes} != {MAX_HS_BYTES}"
    )

    torch.manual_seed(0)
    tpu_device = torch.device("tpu")
    prefix_emb_ref = torch.randn(BLOCK_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
    prefix_hid_ref = torch.randn(BLOCK_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
    tail_emb_ref = torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)
    tail_hid_ref = torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)

    prefix_emb_tpu = prefix_emb_ref.to(tpu_device)
    prefix_hid_tpu = prefix_hid_ref.to(tpu_device)
    tail_emb_tpu = tail_emb_ref.to(tpu_device)
    tail_hid_tpu = tail_hid_ref.to(tpu_device)

    block_hash = b"tpu-block-0"
    key = hs_key(cfg.namespace, block_hash)
    assert key.startswith("omni-hs-v1:"), f"Unexpected HS key format: {key!r}"

    spec = HSBlockSpec(
        namespace=cfg.namespace,
        prefix_hash=prefix_digest(block_hash),
        start=0,
        end=BLOCK_SIZE,
        hidden_layer=HIDDEN_LAYER,
        hidden_size=HIDDEN_SIZE,
        dtype=torch.bfloat16,
    )
    encoded = encode_hs(
        spec,
        {
            "embedding": prefix_emb_tpu.cpu(),
            "hidden": prefix_hid_tpu.cpu(),
        },
    )
    restored = decode_hs(spec, encoded)

    req_hs = RequestHS(
        prompt_tokens=BLOCK_SIZE + 4,
        block_size=BLOCK_SIZE,
        max_bytes=cfg.max_hs_bytes,
    )
    req_hs.restore(restored)
    merged = req_hs.append(
        BLOCK_SIZE,
        {"embedding": tail_emb_tpu.cpu(), "hidden": tail_hid_tpu.cpu()},
    )
    assert_parity(
        merged["embedding"],
        torch.cat([prefix_emb_ref, tail_emb_ref], dim=0),
        name="mooncake_hs_embedding",
        max_rel_l2=0.0,
        max_abs=0.0,
    )
    assert_parity(
        merged["hidden"],
        torch.cat([prefix_hid_ref, tail_hid_ref], dim=0),
        name="mooncake_hs_hidden",
        max_rel_l2=0.0,
        max_abs=0.0,
    )
