# SPDX-License-Identifier: Apache-2.0
"""Model-owned, immutable CPU PLE storage shared by local DP processes."""

import contextlib
import fcntl
import hashlib
import json
import mmap
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SharedPLEOptions:
    directory: Path
    key: str
    identity: dict[str, object]
    participants: int


def host_available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("PLE host capacity requires Linux MemAvailable")


def shared_ple_options(
    model_config, prefix: str, tp_rank: int, tp_size: int
) -> SharedPLEOptions:
    """Use the pre-worker descriptor, never the worker's collapsed DP size."""
    launch = model_config._qwen4_exp_ple_host_launch
    return SharedPLEOptions(
        directory=Path(tempfile.gettempdir()) / f"vllm-qwen-ple-{os.getuid()}",
        key=f"{launch['instance']}/{prefix}/tp{tp_rank}",
        identity=dict(launch, prefix=prefix, tp_rank=tp_rank, tp_size=tp_size),
        participants=launch["participants"],
    )


def _process_identity(pid):
    # A proc entry can disappear before open (ENOENT) or during read (ESRCH).
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if fields[0] in ("Z", "X"):
        return None
    return fields[19]


class SharedPLETable:
    """Attach during loading; finish collectively before reading immutable bytes.

    Large storage is an anonymous memfd, never a named weight file. A small
    failed-generation record is retained on error so late workers cannot
    recreate a half-loaded table. Hot worker restart is not supported.
    """

    def __init__(self, *, directory, key, identity, nbytes, participants, timeout=600):
        if nbytes <= 0 or participants < 1 or timeout <= 0 or not key:
            raise ValueError("Invalid PLE shared storage geometry or timeout")
        self.nbytes, self.timeout = nbytes, timeout
        self._expected = dict(
            identity=identity, nbytes=nbytes, participants=participants
        )
        self._member = f"{os.getpid()}:{_process_identity(os.getpid())}"
        self._ready = self._published = self._closed = False
        self._mapping: mmap.mmap | None = None
        self.data_fd = self._control_fd = -1
        self.is_writer = False
        directory = Path(directory)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._path = directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")
        try:
            self._control_fd = os.open(
                self._path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600
            )
            with self._locked():
                if os.fstat(self._control_fd).st_size == 0:
                    self.is_writer = True
                    self.data_fd = os.memfd_create(
                        "qwen-ple", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
                    )
                    os.ftruncate(self.data_fd, nbytes)
                    self._mapping = mmap.mmap(self.data_fd, nbytes)
                    state: dict[str, Any] = dict(
                        **self._expected,
                        status="loading",
                        writer=dict(
                            pid=os.getpid(),
                            start=_process_identity(os.getpid()),
                            fd=self.data_fd,
                            inode=os.fstat(self.data_fd).st_ino,
                        ),
                        members={},
                    )
                else:
                    state = self._load()
                    if any(state[k] != v for k, v in self._expected.items()):
                        raise ValueError("PLE shared table identity mismatch")
                    self._check_error(state)
                    self._check_writer(state)
                    source = state["writer"]
                    self.data_fd = os.open(
                        f"/proc/{source['pid']}/fd/{source['fd']}",
                        os.O_RDONLY | os.O_CLOEXEC,
                    )
                    stat = os.fstat(self.data_fd)
                    if stat.st_ino != source["inode"] or stat.st_size != nbytes:
                        raise RuntimeError("PLE shared writer descriptor changed")
                    self._mapping = mmap.mmap(
                        self.data_fd, nbytes, access=mmap.ACCESS_READ
                    )
                if (
                    self._member in state["members"]
                    or len(state["members"]) >= participants
                ):
                    raise ValueError("Unexpected PLE shared table participant")
                state["members"][self._member] = False
                self._save(state)
        except BaseException:
            self._release()
            raise

    @contextlib.contextmanager
    def _locked(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self._control_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("PLE shared metadata lock timeout") from None
                time.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(self._control_fd, fcntl.LOCK_UN)

    def _load(self):
        size = os.fstat(self._control_fd).st_size
        return json.loads(os.pread(self._control_fd, size, 0))

    def _save(self, state):
        value = json.dumps(state, sort_keys=True).encode()
        offset = 0
        while offset < len(value):
            written = os.pwrite(self._control_fd, value[offset:], offset)
            if written == 0:
                raise OSError("PLE metadata write made no progress")
            offset += written
        os.ftruncate(self._control_fd, len(value))

    @staticmethod
    def _check_error(state):
        if state["status"] == "error":
            raise RuntimeError(state["error"])

    @staticmethod
    def _check_writer(state):
        writer = state["writer"]
        if _process_identity(writer["pid"]) != writer["start"]:
            raise RuntimeError("PLE shared writer exited before publication")

    @contextlib.contextmanager
    def writable_buffer(self):
        """Borrow writer storage for the upstream tensor checkpoint loader."""
        if not self.is_writer or self._published or self._closed:
            raise RuntimeError("PLE table is read-only or already published")
        assert self._mapping is not None
        with memoryview(self._mapping) as view:
            yield view

    def finish(self, metadata=None):
        """Publish writer bytes and wait until every expected reader is ready."""
        if self._ready:
            return
        deadline = time.monotonic() + self.timeout
        try:
            if self.is_writer:
                assert self._mapping is not None
                self._mapping.close()
                self._mapping = None
                fcntl.fcntl(
                    self.data_fd,
                    fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_GROW
                    | fcntl.F_SEAL_SHRINK
                    | fcntl.F_SEAL_WRITE
                    | fcntl.F_SEAL_SEAL,
                )
                # A sealed shared mapping also requires a read-only file
                # description. Keep the published fd number stable for peers
                # that attach while the writer transitions to read-only.
                readonly = os.open(
                    f"/proc/self/fd/{self.data_fd}", os.O_RDONLY | os.O_CLOEXEC
                )
                try:
                    os.dup2(readonly, self.data_fd, inheritable=False)
                finally:
                    os.close(readonly)
                self._mapping = mmap.mmap(
                    self.data_fd, self.nbytes, access=mmap.ACCESS_READ
                )
                self._published = True
                with self._locked():
                    state = self._load()
                    self._check_error(state)
                    state["status"] = "ready"
                    state["metadata"] = metadata
                    self._save(state)
            while True:
                with self._locked():
                    state = self._load()
                    self._check_error(state)
                    if state["status"] != "ready":
                        self._check_writer(state)
                    for member, finished in state["members"].items():
                        pid, start = member.split(":")
                        if not finished and _process_identity(int(pid)) != start:
                            raise RuntimeError(
                                "PLE shared participant exited before ready"
                            )
                    if state["status"] == "ready":
                        if state["metadata"] != metadata:
                            raise ValueError("PLE shared checkpoint metadata mismatch")
                        state["members"][self._member] = True
                        self._save(state)
                        if len(state["members"]) == state["participants"] and all(
                            state["members"].values()
                        ):
                            self._ready = True
                            self._path.unlink(missing_ok=True)
                            return
                if time.monotonic() >= deadline:
                    raise TimeoutError("PLE shared participants did not become ready")
                time.sleep(0.01)
        except BaseException as exc:
            self.abort(str(exc))
            raise

    @contextlib.contextmanager
    def buffer(self):
        """Borrow the immutable mapping; no view may outlive this scope."""
        if not self._ready or self._closed:
            raise RuntimeError("PLE table is not ready")
        assert self._mapping is not None
        with memoryview(self._mapping) as view:
            yield view

    def abort(self, reason):
        if self._control_fd < 0 or self._ready:
            return
        with self._locked():
            state = self._load()
            if state["status"] != "error":
                state.update(status="error", error=reason)
                self._save(state)

    def _release(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self.data_fd >= 0:
            os.close(self.data_fd)
            self.data_fd = -1
        if self._control_fd >= 0:
            os.close(self._control_fd)
            self._control_fd = -1
        self._closed = True

    def close(self):
        if self._closed:
            return
        try:
            if not self._ready:
                self.abort("PLE participant closed before table was ready")
        finally:
            self._release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.abort(str(exc))
        self.close()
