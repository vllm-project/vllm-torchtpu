# SPDX-License-Identifier: Apache-2.0
"""gather_step_reports: per-rank outcomes, tagged by the step they answer."""

import zlib
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.distributed.rank_agreement import gather_step_reports

GROUP = SimpleNamespace(world_size=3, rank_in_group=0, cpu_group=None)


def _all_gather(peers):
    """all_gather that answers for ranks 1.. with `peers`, each (tag, *values)
    or just values, in which case the caller's tag is used."""

    def all_gather(reports, mine, group):
        tag = int(mine[0])
        rows = [mine.tolist()] + [
            p if len(p) == len(mine) else (tag, *p) for p in peers
        ]
        for report, row in zip(reports, rows, strict=True):
            report.copy_(torch.tensor(row, dtype=torch.int64))

    return all_gather


def test_returns_each_ranks_values_in_rank_order(monkeypatch):
    monkeypatch.setattr(torch.distributed, "all_gather", _all_gather([(0, 1), (1, 0)]))
    assert gather_step_reports(GROUP, "step", (1, 1), on_timeout="") == [
        (1, 1),
        (0, 1),
        (1, 0),
    ]


def test_a_rank_at_another_step_raises(monkeypatch):
    other = zlib.crc32(b"another step") & 0x7FFFFFFF
    monkeypatch.setattr(
        torch.distributed, "all_gather", _all_gather([(1, 1), (other, 1, 1)])
    )
    with pytest.raises(RuntimeError, match="step: ranks reported from different"):
        gather_step_reports(GROUP, "step", (1, 1), on_timeout="")


def test_a_rank_that_never_reports_names_what_to_check(monkeypatch):
    def all_gather(reports, mine, group):
        raise RuntimeError("Connection closed by peer")

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    with pytest.raises(RuntimeError, match="stopped waiting") as e:
        gather_step_reports(GROUP, "step", (1,), on_timeout="Check rank 2.")
    assert str(e.value).endswith("Check rank 2.")
    assert isinstance(e.value.__cause__, RuntimeError)
