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
"""TPU tests for MRv2 AR and Talker speculative decoding."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_validation import _resolve_worker_cls

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


@pytest.mark.xfail(
    strict=True,
    raises=ImportError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_ar_talker_speculative_decoding_mrv2() -> None:
    """AR / Talker Speculative Decoding (MRv2).

    - TPU path: import MRv2 ``DraftInputs`` and ``OmniARModelRunner``, verify
      that the active TPU AR worker's model runner exposes MRv2 speculative
      draft proposal (``_propose_draft_tokens``), and verify draft-token
      publishing on ``torch.device("tpu")``.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``issubclass(worker_cls.model_runner_cls, OmniARModelRunner)``.
    - Outcome check: ``_propose_draft_tokens`` populates and publishes
      ``num_spec_tokens`` draft tokens per request on TPU.
    """
    from vllm_omni.worker_v2.omni_ar_model_runner import (
        DraftInputs,
        OmniARModelRunner,
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    worker_cls_path = current_omni_platform.get_omni_ar_worker_cls()
    worker_cls = _resolve_worker_cls(worker_cls_path)
    assert issubclass(worker_cls.model_runner_cls, OmniARModelRunner), (
        f"{worker_cls.__name__}.model_runner_cls "
        f"({worker_cls.model_runner_cls.__name__}) does not inherit "
        "MRv2 OmniARModelRunner"
    )

    tpu_device = torch.device("tpu")
    draft = DraftInputs(
        num_spec_tokens=2,
        hidden_states=torch.zeros(2, 8, dtype=torch.bfloat16, device=tpu_device),
        aux_hidden_states=None,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        dp_sync_state=None,
    )
    runner = object.__new__(worker_cls.model_runner_cls)
    runner._kstep_frames = 0
    runner.speculator = None
    runner.pp_handler = None
    runner.req_states = SimpleNamespace(
        draft_tokens=torch.tensor(
            [[11, 12, 13], [21, 22, 23]], dtype=torch.int32, device=tpu_device
        )
    )
    published: list[torch.Tensor] = []
    runner.draft_tokens_handler = SimpleNamespace(
        set_draft_tokens=lambda _batch, tokens: published.append(tokens)
    )
    batch = SimpleNamespace(
        idx_mapping=torch.tensor([0, 1], dtype=torch.int64, device=tpu_device)
    )

    runner._propose_draft_tokens(
        batch,
        draft,
        torch.tensor([1, 1], dtype=torch.int32, device=tpu_device),
        torch.tensor([0, 0], dtype=torch.int32, device=tpu_device),
        None,
    )
    assert len(published) == 1, f"Expected 1 draft publication, got {len(published)}"
    expected = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)
    assert torch.equal(published[0].cpu(), expected), (
        f"Published drafts {published[0].cpu().tolist()} != {expected.tolist()}"
    )
