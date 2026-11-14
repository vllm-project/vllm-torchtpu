# SPDX-License-Identifier: Apache-2.0
"""The MK draft sizes its DCP split from the draft layer pool, not the target."""

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime import qwen38_draft


@pytest.mark.parametrize(
    ("main_dcp", "mtp_dcp", "expected"),
    [(0, False, 1), (2, False, 1), (2, True, 2), (0, True, 1)],
)
def test_draft_kv_dcp_follows_the_draft_layer(monkeypatch, main_dcp, mtp_dcp, expected):
    # qsa.py: the draft layer splits its main KV only with VLLM_TPU_QSA_MTP_KV_DCP.
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MAIN_KV_DCP", main_dcp)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_MTP_KV_DCP", mtp_dcp)
    assert qwen38_draft.draft_kv_dcp() == expected
