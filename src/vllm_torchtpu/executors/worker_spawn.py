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
"""Spawning multiproc workers without waiting on each other.

Spawning a worker writes the pickled spawn preparation data and process
object into a pipe that the child reads only after it has started Python and
re-imported the main module, several seconds later. Nearly all of it is
`vllm_config`, chiefly the model's HF config and quantization config: about
75 KB for Qwen3.8-Flash-FP8, just over the default 64 KiB pipe. So that write
blocked the parent until the child read it, and workers started one after
another, about 6 s apart for 8 workers. `large_spawn_pipes` enlarges the
spawn pipes so the write returns at once and every worker imports in
parallel.

The fix holds while the payload fits in the pipe: `TPU_WORKER_SPAWN_PIPE_BYTES`
(default 1 MiB), capped at `/proc/sys/fs/pipe-max-size`. A larger payload
blocks again and the starts serialize as before.

TODO: retire this once torch_tpu can bind rank and chip after import. Today it
binds them from RANK / LOCAL_RANK / TPU_VISIBLE_DEVICES at import, and a child
cannot set those itself: multiproc_executor resolves `current_platform` at
import, which imports tpu_platform and so torch_tpu, before `worker_main`
runs. So the parent writes each worker's binding into its own os.environ
right before spawning it (`SPAWN_LOCK`). A bind-after-import API would let
the child bind itself, retiring both the lock and these pipes, and vLLM's
concurrent start would then overlap the imports on its own.
"""

import contextlib
import fcntl
import os
import threading
from collections.abc import Iterator

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Serializes a worker's spawn: its binding env writes, the pipe swap below and
# the process start. MultiprocExecutor may start workers from several threads
# (e.g. on --headless hosts); with large pipes each start returns at once, so
# holding the lock costs nothing.
SPAWN_LOCK = threading.Lock()


def _pipe_max_size() -> int:
    with open("/proc/sys/fs/pipe-max-size") as f:
        return int(f.read())


class _LargePipeOs:
    """`os` for multiprocessing.popen_spawn_posix, with pipes of `size` bytes.

    `Popen._launch` creates two pipes per worker: the sentinel and the one it
    writes the payload into. Both are enlarged; their capacity is accounted
    to the uid for the worker's lifetime (against pipe-user-pages-soft), which
    is why `size` is bounded rather than the system maximum. Everything else
    is `os` itself.
    """

    def __init__(self, size: int) -> None:
        self._size = size

    def __getattr__(self, name: str):
        return getattr(os, name)

    def pipe(self) -> tuple[int, int]:
        r, w = os.pipe()
        try:
            fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, self._size)
        except OSError as e:
            # The default pipe still works; workers just start one by one.
            logger.warning_once(
                "Could not enlarge the worker spawn pipe to %d bytes (%s); "
                "workers will start one after another.",
                self._size,
                str(e),
            )
        return r, w


def spawn_pipe_size() -> int:
    """Bytes requested for each spawn pipe; 0 leaves the pipes alone."""
    requested = envs.TPU_WORKER_SPAWN_PIPE_BYTES
    if requested == 0:
        return 0
    try:
        return min(requested, _pipe_max_size())
    except (OSError, ValueError) as e:
        logger.warning_once(
            "Could not read /proc/sys/fs/pipe-max-size (%s); requesting "
            "%d-byte worker spawn pipes.",
            str(e),
            requested,
        )
        return requested


@contextlib.contextmanager
def large_spawn_pipes() -> Iterator[None]:
    """Enlarge the pipes of spawns started in this block. Hold SPAWN_LOCK."""
    import multiprocessing.popen_spawn_posix as popen_spawn_posix

    size = spawn_pipe_size()
    if size == 0:
        yield
        return
    # popen_spawn_posix reaches os.pipe through its module global `os`, which
    # its typeshed stub does not declare, hence the module's namespace dict.
    namespace = vars(popen_spawn_posix)
    spawn_os = namespace["os"]
    namespace["os"] = _LargePipeOs(size)
    try:
        yield
    finally:
        namespace["os"] = spawn_os
