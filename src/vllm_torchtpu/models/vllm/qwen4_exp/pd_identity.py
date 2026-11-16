# SPDX-License-Identifier: Apache-2.0
"""Bounded identity of checkpoint tensors actually consumed by a PD model."""

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import torch


def _sha256(tensor: torch.Tensor) -> str:
    # A uint8 view also supports BF16 and FP8 without conversion or a
    # second tensor-sized allocation.
    raw = memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy())
    try:
        return hashlib.sha256(raw).hexdigest()
    finally:
        del raw


class LoadedWeightIdentity:
    """Per-tensor ``(dtype, shape, sha256)`` of what this rank loaded.

    Hashing every loaded byte is ~1.7 GB/s on one core, ~46 s of a TP8/EP8
    decode rank's ~76 GB load when done inline in the loading thread. With
    VLLM_TPU_PD_IDENTITY_HASH_THREADS=N > 0 the digests are computed on N
    threads (hashlib releases the GIL, so they scale) while loading goes on;
    at most VLLM_TPU_PD_IDENTITY_HASH_INFLIGHT_MB of tensors (default 4096) are
    held for hashing at once. The records, and so the identity, are the same.
    """

    def __init__(self) -> None:
        self._records: dict[str, tuple[str, tuple[int, ...], str]] = {}
        self._names: set[str] = set()
        threads = int(os.environ.get("VLLM_TPU_PD_IDENTITY_HASH_THREADS", "0") or 0)
        self._pool = ThreadPoolExecutor(threads, "pd-identity") if threads > 0 else None
        self._pending: dict[str, tuple[str, tuple[int, ...], Future]] = {}
        budget = int(os.environ.get("VLLM_TPU_PD_IDENTITY_HASH_INFLIGHT_MB", "4096"))
        self._budget = max(budget, 1) << 20
        self._inflight = 0
        self._cv = threading.Condition()
        # Loader time spent on digests (hashing inline, or waiting for the
        # pool), for the load timing log (model._LoadTimes).
        self.wait_seconds = 0.0

    def _release(self, nbytes: int) -> None:
        with self._cv:
            self._inflight -= nbytes
            self._cv.notify_all()

    def observe_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._names:
            raise ValueError(f"Duplicate PD checkpoint tensor: {name}")
        if (
            tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or tensor.is_conj()
            or tensor.is_neg()
        ):
            raise ValueError("PD identity requires contiguous CPU checkpoint tensors")
        self._names.add(name)
        meta = (str(tensor.dtype), tuple(tensor.shape))
        if self._pool is None:
            start = time.perf_counter()
            self._records[name] = (*meta, _sha256(tensor))
            self.wait_seconds += time.perf_counter() - start
            return
        nbytes = tensor.numel() * tensor.element_size()
        start = time.perf_counter()
        with self._cv:
            # A tensor larger than the budget still goes once nothing else waits.
            while self._inflight and self._inflight + nbytes > self._budget:
                self._cv.wait()
            self._inflight += nbytes
        self.wait_seconds += time.perf_counter() - start

        def work(t=tensor, n=nbytes):
            try:
                return _sha256(t)
            finally:
                self._release(n)

        self._pending[name] = (*meta, self._pool.submit(work))

    def _drain(self) -> None:
        start = time.perf_counter()
        for name, (dtype, shape, fut) in self._pending.items():
            self._records[name] = (dtype, shape, fut.result())
        self._pending.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self.wait_seconds += time.perf_counter() - start

    def observe(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            self.observe_tensor(name, tensor)
            yield name, tensor

    def records(self) -> dict[str, tuple[str, tuple[int, ...], str]]:
        """Per-tensor ``(dtype, shape, sha256)`` records of what this rank loaded."""
        self._drain()
        return dict(self._records)

    def finish(self, *, name_fragment: str | None = None) -> str:
        self._drain()
        return merged_identity([self._records], name_fragment=name_fragment)


def merged_identity(
    per_rank_records: Iterable[dict], *, name_fragment: str | None = None
) -> str:
    """Digest of the union of the tensors a group of ranks loaded.

    With load-time filtering (``--enable-ep-weight-filter``, or a pipeline
    stage that streams only its own layers) each rank reads only its share, so
    per-rank digests differ even though the ranks together consumed the whole
    checkpoint. Merging the per-tensor records first gives the digest one rank
    that loaded every tensor would report, and a tensor two ranks loaded with
    different bytes is still rejected.
    """
    merged = merged_records(per_rank_records, name_fragment=name_fragment)
    payload = json.dumps(merged, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def merged_records(
    per_rank_records: Iterable[dict], *, name_fragment: str | None = None
) -> dict[str, tuple[str, tuple[int, ...], str]]:
    """Union of per-rank ``(dtype, shape, sha256)`` records (see merged_identity)."""
    merged: dict = {}
    loaded = False
    for records in per_rank_records:
        for name, value in records.items():
            loaded = True
            if name_fragment is not None and name_fragment not in name:
                continue
            value = (value[0], tuple(value[1]), value[2])
            previous = merged.setdefault(name, value)
            if previous != value:
                raise ValueError(f"Qwen PD ranks loaded different bytes for {name}")
    # Something must have been loaded; a fragment that matches none of it
    # still digests the empty set, as finish() always did.
    if not loaded:
        raise ValueError("PD identity requires successfully loaded checkpoint tensors")
    return merged


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
