# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for multi-rank TPU trace capture.

Every TPU worker process drives its own PyTorch/XLA profiler session and can
only capture the chips it owns, so a whole-slice trace needs one capture per
worker. Captures cannot go straight into a shared directory: the xplane files
are named after the host, so ranks co-located on a host would overwrite each
other.

These helpers give all ranks a common protocol:

  1. Each rank captures into its own sandbox, ``<root>/rank_<N>/``.
  2. Rank 0 publishes a canonical run timestamp via a marker file; the other
     ranks read it, so everyone agrees on one destination run directory.
  3. On stop, each rank moves its capture into
     ``<root>/plugins/profile/<canonical_ts>/``, prefixing filenames with
     ``rank<N>_`` when more than one worker is capturing.

The result is a single xprof/TensorBoard run containing every core. Used by
both the phased profiler (``runner/utils.py``) and the standard
``torch_profiler_dir`` flow (``worker/tpu_worker.py``).
"""

import contextlib
import datetime
import os
import shutil
import time

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

CANONICAL_TS_FORMAT = "%Y_%m_%d_%H_%M_%S"
CANONICAL_TS_POLL_TIMEOUT_S = 5.0
CANONICAL_TS_POLL_INTERVAL_S = 0.05


def profile_session_id() -> str:
    """Return an id shared by every worker process of this server run.

    Markers are keyed by this id so that a stale marker left behind by an
    earlier run is never mistaken for the current one. ``TORCH_TPU_DP_MASTER_PORT``
    is picked once per server and inherited by every engine and worker, which
    also covers DP>1 with TP>1, where workers of different DP replicas have
    different parents. The parent pid is the fallback for DP=1, where the port
    is unset and all workers are children of the same engine core.
    """
    dp_master_port = os.environ.get("TORCH_TPU_DP_MASTER_PORT")
    if dp_master_port:
        return f"dp{dp_master_port}"
    return str(os.getppid())


def rank_capture_dir(dst_root: str, worker_rank: int) -> str:
    """Return the per-rank capture sandbox under `dst_root`.

    `worker_rank` is the slice-global rank, so this names TP and PP ranks as
    well as DP replicas.
    """
    return os.path.join(dst_root, f"rank_{worker_rank}")


def canonical_ts_marker_path(dst_root: str, session_key: str) -> str:
    return os.path.join(dst_root, f".canonical_ts_{session_key}")


def resolve_canonical_dst_ts(
    dst_root: str,
    worker_rank: int,
    *,
    session_key: str | None = None,
) -> str:
    """Resolve the destination run timestamp shared by all ranks.

    Rank 0 generates the timestamp and writes a marker file; the other ranks
    poll for it so every worker merges into the same run directory. A rank
    that never sees the marker falls back to its own timestamp, which costs a
    split run directory but never loses a trace.

    The marker is only visible to ranks that share a filesystem with rank 0,
    so on a multi-host slice `dst_root` has to be a shared mount for the
    merge to produce one run directory.
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

    deadline = time.monotonic() + CANONICAL_TS_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            with open(marker) as f:
                ts = f.read().strip()
            if ts:
                return ts
        except OSError:
            pass
        time.sleep(CANONICAL_TS_POLL_INTERVAL_S)

    fallback_ts = datetime.datetime.now().strftime(CANONICAL_TS_FORMAT)
    logger.warning(
        "rank %d did not find rank 0's canonical-ts marker at %s "
        "within %.1fs; falling back to own timestamp %s. Traces are kept, "
        "but this rank lands in its own run directory; check that %s is on "
        "a filesystem rank 0 can also write to.",
        worker_rank,
        marker,
        CANONICAL_TS_POLL_TIMEOUT_S,
        fallback_ts,
        dst_root,
    )
    return fallback_ts


def clear_canonical_ts_marker(dst_root: str, session_key: str | None = None) -> None:
    """Drop the marker once the run it describes is merged (rank 0 only)."""
    if session_key is None:
        session_key = profile_session_id()
    # TODO: Ignore only FileNotFoundError (the marker was never written).
    # Other OSErrors, such as permission errors, should not be hidden.
    with contextlib.suppress(OSError):
        os.remove(canonical_ts_marker_path(dst_root, session_key))


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
    alone, and the sandbox itself is removed only if moving the trace left it
    empty.
    """
    source_profile_path = os.path.join(capture_dir, "plugins", "profile")
    if not os.path.exists(source_profile_path):
        return
    dst_ts_dir = os.path.join(dst_root, "plugins", "profile", canonical_ts)
    # `world_size` is the slice-global world (TP*PP*DP), so this one
    # comparison covers every parallelism mode. Deciding purely from it also
    # keeps the verdict identical on every rank, which is what makes the rank
    # prefix present on all of a run's files or absent from all of them.
    multi_worker = world_size > 1

    try:
        os.makedirs(dst_ts_dir, exist_ok=True)
        for ts in os.listdir(source_profile_path):
            src_ts_dir = os.path.join(source_profile_path, ts)
            if not os.path.isdir(src_ts_dir):
                continue
            for fname in os.listdir(src_ts_dir):
                # Mind the two spellings of the same idea: the capture
                # sandbox is `rank_<N>` (see `rank_capture_dir`), the merged
                # filename is `rank<N>_`.
                new_fname = f"rank{worker_rank}_{fname}" if multi_worker else fname
                shutil.move(
                    os.path.join(src_ts_dir, fname),
                    os.path.join(dst_ts_dir, new_fname),
                )
            # TODO: Check that the directory is empty instead of ignoring every
            # OSError, so real failures such as permission errors surface.
            with contextlib.suppress(OSError):
                os.rmdir(src_ts_dir)
        # rmdir only succeeds on an empty directory, so the sandbox survives
        # when the caller left other artifacts (batch composition stats) in
        # it and is cleaned up when the trace was all it held.
        for cleanup in (
            source_profile_path,
            os.path.dirname(source_profile_path),
            capture_dir,
        ):
            # TODO: Check that the directory is empty instead of ignoring every
            # OSError, so real failures such as permission errors surface.
            with contextlib.suppress(OSError):
                os.rmdir(cleanup)
        logger.info("Successfully merged profile directories into: %s", dst_ts_dir)
    except Exception as e:
        logger.warning("Failed to merge profile directories: %s", e)
