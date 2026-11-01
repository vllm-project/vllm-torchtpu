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
"""Rank-uniform megakernel declines: the vote table, the agreement group of a
spec's collective domain, the one-all-reduce agreement and the declined
driver that serves the stock path."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.runtime import stats as _stats
from vllm_torchtpu.megakernel.runtime.contract import DOMAIN_TP, EligibilityResult
from vllm_torchtpu.megakernel_runtime.errors import MegakernelPluginError
from vllm_torchtpu.megakernel_runtime.hooks import StepContext, is_suppressed

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
    """The vote table: the runtime's codes, a plugin's own codes, then `other`."""
    base = DECLINE_VOTE_CODES[:-1]
    return (*base, *(c for c in extra if c not in base and c != "other"), "other")


def _decline_vote(
    result: EligibilityResult | None, codes: Sequence[str] = DECLINE_VOTE_CODES
) -> int:
    if result is None:
        return 0
    code = result.code if result.code in codes else "other"
    return list(codes).index(code) + 1


def _tp_cpu_group() -> Any:
    """The TP group's CPU (gloo) process group, or `None` on a single rank."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return None
    from vllm.distributed import get_tp_group

    tp_group = get_tp_group()
    if int(getattr(tp_group, "world_size", 1)) <= 1:
        return None
    return tp_group.cpu_group


def agreement_group(domain: str, runner: Any) -> Any:
    """The CPU process group a spec's agreements run over for its
    `collective_domain` (`None` on a single rank): the ranks its kernel
    communicates with. Only "tp" (the TP group) resolves today;
    `check_spec` declines the other domains before any collective."""
    del runner
    if domain == DOMAIN_TP:
        return _tp_cpu_group()
    raise MegakernelPluginError(f"collective_domain {domain!r} has no agreement group")


def _all_reduce_max(t: torch.Tensor, group: Any) -> None:
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=group)


def buckets_token(keys: Sequence[tuple[int, int]], salt: str = "") -> int:
    """A rank-independent 32-bit digest of a set of `(T, B)` bucket keys
    (salted, e.g. with the plugin fingerprint)."""
    return zlib.crc32((salt + repr(sorted(keys))).encode("ascii"))


def agree_on_decline(
    local: EligibilityResult | None,
    *,
    token: int | None = None,
    group: Any = None,
    mismatch_code: str | None = None,
    codes: Sequence[str] = DECLINE_VOTE_CODES,
) -> EligibilityResult | None:
    """Make a megakernel decline rank-uniform across the TP group.

    Every rank calls this at the same point with its own decline (`None` when
    it can serve) and, optionally, a `token` that must be equal on every rank
    (e.g. `buckets_token` of the compiled buckets). One MAX all-reduce of
    `(vote, token, -token)`: if any rank declined, every rank declines, and
    ranks that could serve get `peer_declined:<code>` with the declining
    rank's code (the highest vote when several decline); if the tokens differ,
    every rank declines with `bucket_set_mismatch`. A rank-local decline while
    the other ranks run the megakernel's collectives would hang the job.

    With `mismatch_code` (startup agreement #1: the token is the plugin
    fingerprint), differing tokens win over votes, since the ranks may not
    even share a vote table, and every rank declines with that code.
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
            "Megakernel plugin or runtime source differs across TP ranks "
            f"({mismatch_code}); skipping."
        )
        return EligibilityResult(False, mismatch_code, msg)
    if vote != 0:
        if local is not None:
            return local
        code = codes[vote - 1] if vote <= len(codes) else "other"
        msg = f"Megakernel declined on another TP rank ({code}); skipping."
        return EligibilityResult(False, f"peer_declined:{code}", msg)
    if tok_max != tok_min:
        msg = "Megakernel compiled bucket sets differ across TP ranks; skipping."
        return EligibilityResult(False, "bucket_set_mismatch", msg)
    return None


class DeclinedDriver:
    """Stands in for a megakernel the adapter enabled but had to decline.

    Every forward falls through to the stock model and is counted as
    `stock_<code>` in the same periodic step-mix stats a serving driver logs,
    so the decline shows up in serving stats and not only in the startup log.
    It holds no weights or compiled programs.
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
        self.decline = result
        self.tp_rank, self._display_name = int(tp_rank), display_name
        self._codes = tuple(codes)
        self._domain = domain
        self.step_mix_tracker = _stats.StepMixTracker(period_s=float(stats_period_s))
        self._step_key = f"stock_{result.code}"

    def bind_and_warmup(self, runner: Any) -> None:
        # Join the bind-time agreement the serving drivers on other ranks run.
        group = agreement_group(self._domain, runner)
        agree_on_decline(self.decline, group=group, codes=self._codes)
        logger.warning(
            "%s megakernel declined (%s): %s",
            self._display_name,
            self.decline.code,
            self.decline.message,
        )

    def reset_stats(self) -> None:
        self.step_mix_tracker.reset()

    def on_warmup_done(self, runner: Any) -> None:
        del runner

    def try_forward(self, runner: Any, ctx: StepContext) -> tuple[bool, Any]:
        del ctx
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
