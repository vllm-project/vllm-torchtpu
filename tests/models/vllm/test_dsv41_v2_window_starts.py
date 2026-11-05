# SPDX-License-Identifier: Apache-2.0
"""CPU contract: reused window starts equal a fresh upload."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import _with_window_starts

pytestmark = pytest.mark.cpu_test


@dataclass(frozen=True)
class _Context:
    additional_kwargs: dict = field(default_factory=dict)


def _first(starts, requests):
    return SimpleNamespace(
        seq_lens=torch.zeros(requests, dtype=torch.int32), replay_starts_cpu=starts
    )


def _starts(context):
    return context.additional_kwargs["deepseek_v41_window_starts"]


def test_reuse_keeps_values_and_uploads_only_on_change():
    positions = torch.zeros(288, dtype=torch.int64)
    context = _Context(additional_kwargs={"other": 1})
    reuse = {}
    first = _first((97, 0, 5), 4)
    fresh = _starts(_with_window_starts(context, first, positions))
    reused = _with_window_starts(context, first, positions, reuse)
    assert reused.additional_kwargs["other"] == 1
    assert torch.equal(_starts(reused), fresh) and _starts(reused).dtype == fresh.dtype
    again = _starts(
        _with_window_starts(context, _first((97, 0, 5), 4), positions, reuse)
    )
    assert again is _starts(reused)
    changed = _starts(
        _with_window_starts(context, _first((97, 1, 5), 4), positions, reuse)
    )
    assert changed is not again
    assert changed.tolist() == [97, 1, 5, 0]
    # A different request count is a different upload even with equal prefixes.
    wider = _starts(
        _with_window_starts(context, _first((97, 1, 5), 5), positions, reuse)
    )
    assert wider.tolist() == [97, 1, 5, 0, 0]
