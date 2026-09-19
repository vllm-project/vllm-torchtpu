# SPDX-License-Identifier: Apache-2.0
"""Bookkeeping records for Stage-3 (Raiden) KV transfers.

Plain dataclasses shared by the connector halves: the scheduler half emits a
``_Stage3LoadMeta`` per request it wants pulled, and the worker half tracks
each submission and producer registration with the remaining records. No
behaviour lives here.
"""

from dataclasses import dataclass
from typing import Any, Optional


class _DoneFuture:

    def wait(self) -> None:
        return


_DONE_FUTURE = _DoneFuture()


@dataclass
class _Stage3LoadMeta:
    """Decode-local declaration for one controller-driven FA transfer.

    Source physical block IDs deliberately do not appear here.  They stay in
    the source controller's D5 registry; the only source-side routing datum
    crossing the P/D boundary is its controller address.
    """

    uuid: int
    source_req_id: str
    local_block_ids: list[int]
    num_tokens: int
    src_controller_address: str
    src_job_name: str
    src_engine_id: str
    src_data_replica_idx: int
    src_parallelism: int
    # Uniform-mamba-layout destination state slots, one per mamba kv-cache
    # group ordinal: the block holding the recurrent state for the resumed
    # request (state follows each group's block table; None for FA-only
    # models).
    mamba_state_block_ids: Optional[list[int]] = None
    # Prefix-aware load: tokens satisfied by the decode-local prefix cache
    # and excluded from the transfer. local_block_ids then holds only the
    # suffix pages; the source store clips its plan at
    # skip_tokens * token_bytes of the FA global destination byte space.
    skip_tokens: int = 0
    # Full local hit: nothing to transfer; the worker only releases the
    # producer's request-block registration instead of pulling.
    release_only: bool = False


@dataclass
class _Stage3PendingSubmit:
    """One consumer load parked because the producer's request-block
    registrations were not yet visible at the source controller.

    The registration travels out-of-band and structurally lags the producer's
    finish response by at least one producer scheduler step, so a consumer
    query can legitimately arrive first. Instead of spinning the worker step
    in an inline poll, the request is re-attempted once per scheduler step
    from get_finished() until defer_deadline.
    """

    req_meta: Any
    source_req_id: str
    destination_req_id: str
    uuid: int
    num_tokens: int
    local_blocks: list
    dst_controller_address: str
    dst_units: list
    # None = legacy inline-poll mode (the entry is then never parked).
    defer_deadline: Optional[float]
    first_attempt_s: float
    wait_logged: bool = False
    # RPC rate limiting for re-attempts.
    last_attempt_s: float = 0.0
    # The prepared coordination call, staged once per submission and reused
    # by every re-attempt (_Stage3SubmitTask).
    task: Any = None


@dataclass(frozen=True)
class _Stage3SubmitTask:
    """One staged source-controller coordination call.

    Everything scheduler-visible about the request is recorded in the worker
    tables before the task is created, so executing the RPC needs no
    connector state beyond the facade and the prepared call arguments.
    """

    pending: _Stage3PendingSubmit
    skip_tokens: int
    fa_skip_bytes: int
    source_units: int
    facade: Any
    call_kwargs: dict[str, Any]
    # Legacy inline-poll mode: the RPC call itself retries a missing producer
    # registration for a short bounded interval instead of reporting it.
    retry_inline: bool


@dataclass(frozen=True)
class _Stage3SubmitOutcome:
    """Result of one coordination RPC, applied on the model-runner thread."""

    task: _Stage3SubmitTask
    error: Optional[BaseException]
    submit_ms: float
    # perf_counter() when the RPC was issued; paces re-attempts.
    attempted_at: float


@dataclass(frozen=True)
class _Stage3RegisteredSend:
    """Worker-local D5 registration retained until terminal send state."""

    uuid: int
    local_block_ids: tuple[int, ...]
    num_tokens: int
    expiration_time: float
