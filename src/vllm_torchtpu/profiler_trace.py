# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for multi-rank TPU trace capture.

Every TPU worker process drives its own PyTorch/XLA profiler session and can
only capture the chips it owns, so a whole-slice trace needs one capture per
worker. Captures cannot go straight into a shared directory: the xplane files
are named after the host, so ranks co-located on a host would overwrite each
other.

These helpers give all ranks a common protocol:

  1. Each rank captures into its own sandbox, ``<root>/dp_rank_<N>/``.
  2. Rank 0 publishes a canonical run timestamp via a marker file; the other
     ranks read it, so everyone agrees on one destination run directory.
  3. On stop, each rank moves its capture into
     ``<root>/plugins/profile/<canonical_ts>/``, prefixing filenames with
     ``rank<N>_`` when more than one worker is capturing.

The result is a single xprof/TensorBoard run containing every core. Used by
both the phased profiler (``runner/utils.py``) and the standard
``torch_profiler_dir`` flow (``worker/tpu_worker.py``).
"""

import datetime
import os
import shutil
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

CANONICAL_TS_FORMAT = "%Y_%m_%d_%H_%M_%S"
CANONICAL_TS_POLL_TIMEOUT_S = 5.0
CANONICAL_TS_POLL_INTERVAL_S = 0.05

# Overrides the id that scopes canonical-ts markers. Only needed when the
# workers of one server are not siblings and TORCH_TPU_DP_MASTER_PORT is
# unset; see `profile_session_id`.
PROFILE_SESSION_ID_ENV = "VLLM_TORCHTPU_PROFILE_SESSION_ID"


def profile_session_id() -> str:
    """Return an id shared by every worker process of this server run.

    Markers are keyed by this id so that a stale marker left behind by an
    earlier run is never mistaken for the current one. ``TORCH_TPU_DP_MASTER_PORT``
    is picked once per server and inherited by every engine and worker, which
    also covers DP>1 with TP>1, where workers of different DP replicas have
    different parents. The parent pid is the fallback for the single-engine
    case, where all workers are children of the same engine core.
    """
    override = os.environ.get(PROFILE_SESSION_ID_ENV)
    if override:
        return override
    dp_master_port = os.environ.get("TORCH_TPU_DP_MASTER_PORT")
    if dp_master_port:
        return f"dp{dp_master_port}"
    return str(os.getppid())


def rank_capture_dir(dst_root: str, worker_rank: int) -> str:
    """Return the per-rank capture sandbox under `dst_root`."""
    return os.path.join(dst_root, f"dp_rank_{worker_rank}")


def inject_rank_into_filename(fname: str, worker_rank: int) -> str:
    """Prefix `rank<N>_` to an xplane or trace filename."""
    return f"rank{worker_rank}_{fname}"


def is_multi_worker_capture(worker_rank: int, world_size: int) -> bool:
    """Whether more than one rank is expected to capture into `dst_root`."""
    dp_size = int(os.getenv("TORCH_TPU_DP_SIZE", "1"))
    return (dp_size > 1 or os.getenv("TPU_MULTIPROCESS_DP") == "1"
            or world_size > 1 or worker_rank > 0)


def canonical_ts_marker_path(dst_root: str, session_key: str) -> str:
    return os.path.join(dst_root, f".canonical_ts_{session_key}")


def resolve_canonical_dst_ts(
    dst_root: str,
    worker_rank: int,
    *,
    session_key: str | None = None,
    poll_timeout_s: float = CANONICAL_TS_POLL_TIMEOUT_S,
    poll_interval_s: float = CANONICAL_TS_POLL_INTERVAL_S,
) -> str:
    """Resolve the destination run timestamp shared by all ranks.

    Rank 0 generates the timestamp and writes a marker file; the other ranks
    poll for it so every worker merges into the same run directory. A rank
    that never sees the marker falls back to its own timestamp, which costs a
    split run directory but never loses a trace.
    """
    if session_key is None:
        session_key = profile_session_id()
    marker = canonical_ts_marker_path(dst_root, session_key)

    if worker_rank == 0:
        canonical_ts = datetime.datetime.now().strftime(CANONICAL_TS_FORMAT)
        marker_tmp = f"{marker}.tmp"
        try:
            with open(marker_tmp, "w") as f:
                f.write(canonical_ts)
            os.replace(marker_tmp, marker)
        except Exception as e:
            logger.warning("Rank 0 failed to write canonical ts marker: %s", e)
        return canonical_ts

    deadline = time.monotonic() + poll_timeout_s
    while time.monotonic() < deadline:
        try:
            with open(marker) as f:
                ts = f.read().strip()
            if ts:
                return ts
        except OSError:
            pass
        time.sleep(poll_interval_s)

    fallback_ts = datetime.datetime.now().strftime(CANONICAL_TS_FORMAT)
    logger.warning(
        "rank %d did not find rank 0's canonical-ts marker at %s "
        "within %.1fs; falling back to own timestamp %s",
        worker_rank,
        marker,
        poll_timeout_s,
        fallback_ts,
    )
    return fallback_ts


def clear_canonical_ts_marker(dst_root: str,
                              session_key: str | None = None) -> None:
    """Drop the marker once the run it describes is merged (rank 0 only)."""
    if session_key is None:
        session_key = profile_session_id()
    try:
        os.remove(canonical_ts_marker_path(dst_root, session_key))
    except OSError:
        pass


def merge_rank_capture(
    capture_dir: str,
    dst_root: str,
    canonical_ts: str,
    worker_rank: int,
    world_size: int = 1,
) -> None:
    """Move this rank's capture into the shared run directory.

    `capture_dir` is the rank's sandbox (as returned by `rank_capture_dir`);
    the profiler writes into `<capture_dir>/plugins/profile/<its own ts>/`.
    Everything found there lands in `<dst_root>/plugins/profile/<canonical_ts>/`,
    rank-prefixed when several workers share the destination. Non-trace files
    the caller put in the sandbox (e.g. batch composition stats) are left
    alone.
    """
    source_profile_path = os.path.join(capture_dir, "plugins", "profile")
    if not os.path.exists(source_profile_path):
        return
    dst_ts_dir = os.path.join(dst_root, "plugins", "profile", canonical_ts)
    multi_worker = is_multi_worker_capture(worker_rank, world_size)

    try:
        os.makedirs(dst_ts_dir, exist_ok=True)
        for ts in os.listdir(source_profile_path):
            src_ts_dir = os.path.join(source_profile_path, ts)
            if not os.path.isdir(src_ts_dir):
                continue
            for fname in os.listdir(src_ts_dir):
                new_fname = (inject_rank_into_filename(fname, worker_rank)
                             if multi_worker else fname)
                shutil.move(
                    os.path.join(src_ts_dir, fname),
                    os.path.join(dst_ts_dir, new_fname),
                )
            try:
                os.rmdir(src_ts_dir)
            except OSError:
                pass
        for cleanup in (
                source_profile_path,
                os.path.dirname(source_profile_path),
        ):
            try:
                os.rmdir(cleanup)
            except OSError:
                pass
        logger.info("Successfully merged profile directories into: %s",
                    dst_ts_dir)
    except Exception as e:
        logger.warning("Failed to merge profile directories: %s", e)
