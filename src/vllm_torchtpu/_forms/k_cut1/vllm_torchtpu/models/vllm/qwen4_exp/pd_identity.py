# SPDX-License-Identifier: Apache-2.0
"""Bounded identity of checkpoint tensors actually consumed by a PD model."""

import hashlib
import json
import time
from collections import deque
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import torch

# Every rank digests the whole checkpoint (172.8 GiB for Qwen3.8-Flash), one
# SHA-256 stream per tensor at about 1.5 GiB/s per core, so the digests run on
# a thread pool (hashlib releases the GIL) ahead of the loader.
_DIGEST_THREADS = 8
# Bytes and tensors digested ahead of the loader at most. Checkpoint tensors are
# safetensors mmap views, so the window costs page cache, not process memory.
_READ_AHEAD_BYTES = 4 << 30
_READ_AHEAD_TENSORS = 8192
# Smaller tensors are digested inline: a pool round trip costs more.
_INLINE_DIGEST_BYTES = 256 << 10
_CHUNK_BYTES = 4 * 1024 * 1024


def _sha256(tensor: torch.Tensor) -> str:
    # A uint8 view also supports BF16 and FP8 without conversion or a second
    # tensor-sized allocation.
    raw = memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy())
    digest = hashlib.sha256()
    for offset in range(0, len(raw), _CHUNK_BYTES):
        digest.update(raw[offset : offset + _CHUNK_BYTES])
    return digest.hexdigest()


class LoadedWeightIdentity:
    def __init__(self, threads: int = _DIGEST_THREADS) -> None:
        self._records: dict[str, tuple[str, tuple[int, ...], str] | None] = {}
        self._threads = threads
        # Loader time spent waiting for a digest, for load timing logs.
        self.wait_seconds = 0.0

    def _admit(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._records:
            raise ValueError(f"Duplicate PD checkpoint tensor: {name}")
        if (
            tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or tensor.is_conj()
            or tensor.is_neg()
        ):
            raise ValueError("PD identity requires contiguous CPU checkpoint tensors")
        # Reserved until the digest lands, so a duplicate inside the read-ahead
        # window is caught too.
        self._records[name] = None

    def observe(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yields ``weights`` unchanged, in order, recording each tensor's digest.

        A tensor is handed on only after its digest is complete, so the loader
        can never change bytes that are still being read. Keeps no tensor
        reference after the tensor has been handed on.
        """
        if self._threads <= 1:
            for name, tensor in weights:
                self._admit(name, tensor)
                self._records[name] = (
                    str(tensor.dtype),
                    tuple(tensor.shape),
                    _sha256(tensor),
                )
                yield name, tensor
            return
        source: Iterator[tuple[str, torch.Tensor]] | None = iter(weights)
        window: deque[tuple[str, torch.Tensor, int, Future[str] | str]] = deque()
        window_bytes = 0
        with ThreadPoolExecutor(
            self._threads, thread_name_prefix="pd-identity"
        ) as pool:
            try:
                while True:
                    while source is not None and (
                        not window
                        or (
                            window_bytes < _READ_AHEAD_BYTES
                            and len(window) < _READ_AHEAD_TENSORS
                        )
                    ):
                        item = next(source, None)
                        if item is None:
                            source = None
                            break
                        name, tensor = item
                        self._admit(name, tensor)
                        nbytes = tensor.numel() * tensor.element_size()
                        digest: Future[str] | str = (
                            _sha256(tensor)
                            if nbytes <= _INLINE_DIGEST_BYTES
                            else pool.submit(_sha256, tensor)
                        )
                        window.append((name, tensor, nbytes, digest))
                        window_bytes += nbytes
                    if not window:
                        return
                    name, tensor, nbytes, digest = window.popleft()
                    window_bytes -= nbytes
                    if isinstance(digest, Future):
                        start = time.perf_counter()
                        digest = digest.result()
                        self.wait_seconds += time.perf_counter() - start
                    self._records[name] = (
                        str(tensor.dtype),
                        tuple(tensor.shape),
                        digest,
                    )
                    yield name, tensor
            finally:
                for *_, digest in window:
                    if isinstance(digest, Future):
                        digest.cancel()

    def finish(self, *, name_fragment: str | None = None) -> str:
        if not self._records or None in self._records.values():
            raise ValueError(
                "PD identity requires successfully loaded checkpoint tensors"
            )
        records = {
            name: value
            for name, value in self._records.items()
            if name_fragment is None or name_fragment in name
        }
        payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def canonical_rank_identity(records: list[dict], *, world_size: int) -> dict:
    """Compare rank ownership using engine-relative coordinates, not host paths."""
    if (
        world_size <= 0
        or len(records) != world_size
        or [r["rank"] for r in records] != list(range(world_size))
    ):
        raise ValueError("Qwen PD identity needs every parallel rank in rank order")
    for record in records:
        for field in ("checkpoint", "ple_checkpoint"):
            digest = record[field]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Qwen PD checkpoint identity is missing or incomplete")
        if record["tp_size"] <= 0 or not 0 <= record["tp_rank"] < record["tp_size"]:
            raise ValueError("Qwen PD has an invalid TP ownership map")
    if any(
        (r["tp_size"], r["config"], r["state_flags"], r["ple_checkpoint"])
        != (
            records[0]["tp_size"],
            records[0]["config"],
            records[0]["state_flags"],
            records[0]["ple_checkpoint"],
        )
        for r in records[1:]
    ):
        raise ValueError("Qwen PD peers disagree on model or state configuration")
    return dict(schema="qwen4-loaded-state-v1", ranks=records)
