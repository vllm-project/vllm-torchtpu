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
"""Rank-uniform megakernel declines across a spec's collective domain.

Provides the vote table, the host Gloo process group for a spec's collective
domain, the single all-reduce agreement, and the declined driver stand-in.
"""

from __future__ import annotations

import importlib
import zlib
from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.runtime import stats as _stats
from vllm_torchtpu.megakernel.runtime.contract import (
    COLLECTIVE_DOMAINS,
    DOMAIN_DCP,
    DOMAIN_TP,
    DOMAIN_TP_X_DCP,
    EligibilityResult,
)
from vllm_torchtpu.megakernel_runtime.errors import MegakernelPluginError
from vllm_torchtpu.megakernel_runtime.guards import ParallelTopology
from vllm_torchtpu.megakernel_runtime.hooks import StepContext, is_suppressed

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = init_logger(__name__)

# Decline codes a rank can vote with in `agree_on_decline` (index + 1 is the
# vote; 0 means "can serve"). Unlisted codes vote as `other`.
DECLINE_VOTE_CODES: tuple[str, ...] = (
    "pallas_mesh_unavailable",
    "pallas_mesh_mismatch",
    "cache_layout_incompatible",
    "pool_layout_unknown",
    "pool_layout_mismatch",
    "conv_state_dtype_unsupported",
    "bind_error",
    "plugin_mismatch",
    "create_error",
    "draft_declined",
    "mesh_size_mismatch",
    "other",
)


def vote_codes(extra: Sequence[str] = ()) -> tuple[str, ...]:
    """Return the runtime vote codes, any plugin codes, and trailing `other`."""
    base = DECLINE_VOTE_CODES[:-1]
    return (*base, *(c for c in extra if c not in base and c != "other"), "other")


def _decline_vote(
    result: EligibilityResult | None, codes: Sequence[str] = DECLINE_VOTE_CODES
) -> int:
    """Encode `result` as a 1-based index into `codes`, or 0 when eligible."""
    if result is None:
        return 0
    code = result.code if result.code in codes else "other"
    return list(codes).index(code) + 1


def _coordinator_group(getter_name: str) -> object | None:
    """Return a live vLLM `GroupCoordinator` when distributed is initialized."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    try:
        vllm_dist = importlib.import_module("vllm.distributed")
    except ImportError:
        return None
    getter = getattr(vllm_dist, getter_name)
    try:
        return getter()
    except AssertionError:
        return None


def _coordinator_cpu_group(getter_name: str, label: str) -> object | None:
    """Return a vLLM `GroupCoordinator`'s CPU Gloo group, or `None` for 1 rank."""
    coord = _coordinator_group(getter_name)
    if coord is None or int(getattr(coord, "world_size", 1)) <= 1:
        return None
    cpu_group = getattr(coord, "cpu_group", None)
    if cpu_group is None:
        msg = f"{label} group cpu_group is not initialized"
        raise RuntimeError(msg)
    return cpu_group


def _tp_cpu_group() -> object | None:
    """Return the TP group's CPU (Gloo) process group, or `None` on a single rank."""
    return _coordinator_cpu_group("get_tp_group", "TP")


def _dcp_cpu_group() -> object | None:
    """Return the DCP group's CPU (Gloo) process group, or `None` on a single rank."""
    return _coordinator_cpu_group("get_dcp_group", "DCP")


def _world_cpu_group() -> object | None:
    """Return the world group's CPU (Gloo) process group, or `None` on a single rank."""
    return _coordinator_cpu_group("get_world_group", "World")


def _verify_nested_tp_dcp_ranks() -> None:
    """Verify live TP and DCP coordinator rank sets nest when `.ranks` exist."""
    tp_coord = _coordinator_group("get_tp_group")
    dcp_coord = _coordinator_group("get_dcp_group")
    tp_ranks = getattr(tp_coord, "ranks", None)
    dcp_ranks = getattr(dcp_coord, "ranks", None)
    if tp_ranks is None or dcp_ranks is None:
        return
    tp_set = set(tp_ranks)
    dcp_set = set(dcp_ranks)
    if not (dcp_set <= tp_set or tp_set <= dcp_set):
        msg = (
            f"TP ranks {sorted(tp_set)} and DCP ranks {sorted(dcp_set)} are not "
            "nested; no single coordinator spans tp_x_dcp"
        )
        raise MegakernelPluginError(msg)


def agreement_group(domain: str, runner: object) -> object | None:
    """Return the CPU process group spanning `domain` (`None` on a single rank).

    Validates topology compatibility via `ParallelTopology.domain_error` so
    pure `dcp` requires `TP=1` and `tp_x_dcp` requires nested `(TP, PCP, DCP)`
    groups before selecting the outer coordinator.
    """
    if domain not in COLLECTIVE_DOMAINS:
        msg = f"collective_domain {domain!r} has no agreement group"
        raise MegakernelPluginError(msg)
    topo = ParallelTopology.from_runner(runner)
    if err := topo.domain_error(domain):
        raise MegakernelConfigError(err)
    if domain == DOMAIN_TP:
        return _tp_cpu_group()
    if domain == DOMAIN_DCP:
        return _dcp_cpu_group()
    if domain == DOMAIN_TP_X_DCP:
        _verify_nested_tp_dcp_ranks()
        return _dcp_cpu_group() if topo.dcp > topo.tp else _tp_cpu_group()
    return _world_cpu_group()


def _all_reduce_max(t: torch.Tensor, group: Any) -> None:
    """Run an in-place MAX all-reduce of `t` across `group`."""
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)


def buckets_token(keys: Sequence[tuple[int, int]], salt: str = "") -> int:
    """Return a rank-independent 32-bit digest of `(T, B)` bucket keys."""
    return zlib.crc32((salt + repr(sorted(keys))).encode("ascii"))


def agree_on_decline(
    local: EligibilityResult | None,
    *,
    token: int | None = None,
    group: object | None = None,
    mismatch_code: str | None = None,
    codes: Sequence[str] = DECLINE_VOTE_CODES,
) -> EligibilityResult | None:
    """Make a megakernel decline rank-uniform across the agreement group.

    Every rank calls this at the same point with its own decline (`None` when
    it can serve) and, optionally, a `token` that must match on every rank
    (e.g. `buckets_token` of the compiled buckets). One MAX all-reduce of
    `(vote, token, -token)` ensures that if any rank declines, every rank
    declines, and if tokens differ, every rank declines with `mismatch_code`
    or `bucket_set_mismatch`.
    """
    if group is None:
        group = _tp_cpu_group()
    if group is None:
        return local
    tok = int(token or 0)
    t = torch.tensor([_decline_vote(local, codes), tok, -tok], dtype=torch.int64)
    _all_reduce_max(t, group)
    vote, tok_max, tok_min = int(t[0]), int(t[1]), -int(t[2])
    if mismatch_code is not None and tok_max != tok_min:
        msg = (
            "Megakernel plugin or runtime source differs across ranks "
            f"({mismatch_code}); skipping."
        )
        return EligibilityResult(False, mismatch_code, msg)
    if vote != 0:
        if local is not None:
            return local
        code = codes[vote - 1] if vote <= len(codes) else "other"
        msg = f"Megakernel declined on another rank ({code}); skipping."
        return EligibilityResult(False, f"peer_declined:{code}", msg)
    if tok_max != tok_min:
        msg = "Megakernel compiled bucket sets differ across ranks; skipping."
        return EligibilityResult(False, "bucket_set_mismatch", msg)
    return None


class DeclinedDriver:
    """Stand-in for a megakernel the adapter enabled but had to decline.

    Every forward falls through to the baseline model and is counted in the
    periodic step-mix stats so declines remain visible during serving.
    """

    serving = False

    def __init__(
        self,
        result: EligibilityResult,
        *,
        tp_rank: int = 0,
        display_name: str = "megakernel",
        stats_period_s: float = 60.0,
        codes: Sequence[str] = DECLINE_VOTE_CODES,
        domain: str = DOMAIN_TP,
    ) -> None:
        """Initialize declined driver state for `domain`."""
        self.decline = result
        self.tp_rank, self._display_name = int(tp_rank), display_name
        self._codes = tuple(codes)
        self._domain = domain
        self.step_mix_tracker = _stats.StepMixTracker(period_s=float(stats_period_s))
        self._step_key = f"stock_{result.code}"

    def bind_and_warmup(self, runner: Any) -> None:
        """Join the bind-time agreement run by serving peers in `domain`."""
        group = agreement_group(self._domain, runner)
        agree_on_decline(self.decline, group=group, codes=self._codes)
        logger.warning(
            "%s megakernel declined (%s): %s",
            self._display_name,
            self.decline.code,
            self.decline.message,
        )

    def reset_stats(self) -> None:
        """Reset periodic step-mix counters."""
        self.step_mix_tracker.reset()

    def on_warmup_done(self, runner: Any) -> None:
        """No-op post-warmup hook matching `ProgramDriver`."""
        _ = runner

    def try_forward(self, runner: Any, ctx: StepContext) -> tuple[bool, Any]:
        """Record a fallback step when unsuppressed and return `(False, None)`."""
        _ = ctx
        if is_suppressed(runner):
            return False, None
        _, summary = self.step_mix_tracker.note_step(self._step_key, False)
        if summary is not None and self.tp_rank == 0:
            elapsed, mk_cnt, total, pct, snap = summary
            logger.info(
                "%s megakernel step mix over %.0fs: %d/%d MK (%.1f%%) %s",
                self._display_name,
                elapsed,
                mk_cnt,
                total,
                pct,
                snap,
            )
        return False, None
