# SPDX-License-Identifier: Apache-2.0
"""TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER is recognized and reports its libtpu flags."""

from unittest import mock

import pytest

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import vllm_adapter

FLAGS = " ".join(vllm_adapter.ASYNC_ROW_GATHER_FLAGS)


@pytest.mark.parametrize(
    "flag, libtpu, info, warning",
    [
        ("0", FLAGS, 0, 0),
        ("1", "--foo=1 " + FLAGS, 1, 0),
        ("1", "--xla_enable_async_all_gather=true", 0, 1),
    ],
)
def test_arming_log(monkeypatch, flag, libtpu, info, warning):
    monkeypatch.setenv("TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER", flag)
    monkeypatch.setenv("LIBTPU_INIT_ARGS", libtpu)
    with mock.patch.object(vllm_adapter, "logger") as logger:
        vllm_adapter._log_async_row_gather()
    assert logger.info_once.call_count == info
    assert logger.warning_once.call_count == warning
