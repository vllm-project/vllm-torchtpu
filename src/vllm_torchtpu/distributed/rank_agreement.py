# SPDX-License-Identifier: Apache-2.0
"""Ranks reporting a local outcome to each other before their paths diverge.

A rank that decides alone whether to retry, skip or recompile can take a
different next step from its peers, and the collectives in that step then no
longer match. The callers gather each rank's outcome first and act on the
whole set, so every rank takes the same step.
"""

import time
import zlib
from collections.abc import Sequence
from typing import Any

import torch


def gather_step_reports(
    group: Any, step: str, values: Sequence[int], on_timeout: str
) -> list[tuple[int, ...]]:
    """All-gather `values` from every rank of `group`; one tuple per rank.

    `group` is a vLLM GroupCoordinator, gathered over its CPU group. `step`
    must be the same string on every rank that is meant to meet here: each
    report carries a tag derived from it, so a rank at another step, or at
    another all_gather on the same group, raises on every rank instead of being
    read as an answer. A rank that never reports raises after the CPU group's
    timeout; `on_timeout` tells the user what to check.
    """
    tag = zlib.crc32(step.encode()) & 0x7FFFFFFF
    mine = torch.tensor([tag, *values], dtype=torch.int64)
    reports = [torch.zeros_like(mine) for _ in range(group.world_size)]
    t0 = time.monotonic()
    try:
        torch.distributed.all_gather(reports, mine, group=group.cpu_group)
    except RuntimeError as e:
        raise RuntimeError(
            f"{step}: rank {group.rank_in_group} stopped waiting for the other "
            f"{group.world_size - 1} ranks after {time.monotonic() - t0:.0f} s. "
            f"{on_timeout}"
        ) from e
    if {int(r[0]) for r in reports} != {tag}:
        raise RuntimeError(
            f"{step}: ranks reported from different steps, so they are out of "
            "step in their collectives. Look for a rank that skipped or repeated "
            "a step in the logs above."
        )
    return [tuple(int(v) for v in r[1:]) for r in reports]
