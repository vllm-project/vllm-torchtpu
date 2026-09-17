# SPDX-License-Identifier: Apache-2.0
"""Wait for TPU device nodes to be released before starting another engine."""

import glob
import os
import re
import time

_TPU_NODE = re.compile(r"/dev/(?:accel|vfio/)\d+")

_POLL_INTERVAL_S = 0.5

# libtpu's session teardown runs inside `Py_FinalizeEx` and can take tens of
# seconds. The wait returns as soon as the chips are free, so a generous
# ceiling costs nothing on the common path and only fires when a process is
# genuinely wedged.
_DEFAULT_TIMEOUT_S = 180.0


def _fd_links() -> list[str]:
    """Every open file descriptor of every process this one can see.

    Only numeric entries are scanned. `/proc` also holds `self` and
    `thread-self`, which point back at the caller and would otherwise be
    counted as a second owner of the caller's own chips.
    """
    return glob.glob("/proc/[0-9]*/fd/*")


def tpu_device_owners() -> dict[str, int]:
    """Device nodes held by a process other than this one.

    The caller is excluded because an in-process test that touched the TPU
    runtime holds its chip for the lifetime of pytest, and no amount of
    waiting will change that.
    """
    mine = str(os.getpid())
    owners: dict[str, int] = {}
    for link in _fd_links():
        pid = link.split("/")[2]
        if pid == mine:
            continue
        try:
            target = os.readlink(link)
        except FileNotFoundError:
            # The process exited while the scan was walking its table, which
            # is the outcome this module waits for. Every other error is left
            # to propagate.
            continue
        if _TPU_NODE.fullmatch(target):
            owners[target] = int(pid)
    return owners


def wait_for_tpu_release(timeout: float = _DEFAULT_TIMEOUT_S) -> None:
    """Block until no other process holds a TPU device node.

    An engine that outlasts vLLM's shutdown grace period is sent SIGKILL, and
    the shutdown call returns once the signal is sent rather than once the
    process is gone. Building the next engine straight after therefore finds
    the chip still owned and fails with `Failed to acquire a TPU device node`.
    Polling the real owner list is exact, where a fixed sleep is either too
    short or wasted time.

    Raises:
        RuntimeError: a device is still held when the timeout expires. The
            message names the holding PIDs, which are the processes to look at.
    """
    deadline = time.monotonic() + timeout
    while True:
        owners = tpu_device_owners()
        if not owners:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"TPU still held after {timeout:.0f}s by "
                               f"{owners}; the holding process never released "
                               "its chip")
        time.sleep(_POLL_INTERVAL_S)
