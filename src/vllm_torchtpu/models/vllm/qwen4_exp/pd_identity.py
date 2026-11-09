# SPDX-License-Identifier: Apache-2.0
"""Bounded identity of checkpoint tensors actually consumed by a PD model."""

import hashlib
import json
from collections.abc import Iterable, Iterator

import torch


class LoadedWeightIdentity:
    def __init__(self) -> None:
        self._records: dict[str, tuple[str, tuple[int, ...], str]] = {}

    def observe(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if name in self._records:
                raise ValueError(f"Duplicate PD checkpoint tensor: {name}")
            if (
                tensor.device.type != "cpu"
                or not tensor.is_contiguous()
                or tensor.is_conj()
                or tensor.is_neg()
            ):
                raise ValueError(
                    "PD identity requires contiguous CPU checkpoint tensors"
                )
            # A uint8 view also supports BF16 and FP8 without conversion or a
            # second tensor-sized allocation. Keep no reference after loading.
            raw = memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy())
            digest = hashlib.sha256()
            for offset in range(0, len(raw), 4 * 1024 * 1024):
                digest.update(raw[offset : offset + 4 * 1024 * 1024])
            self._records[name] = (
                str(tensor.dtype),
                tuple(tensor.shape),
                digest.hexdigest(),
            )
            del raw
            yield name, tensor

    def records(self) -> dict[str, tuple[str, tuple[int, ...], str]]:
        """Per-tensor ``(dtype, shape, sha256)`` records of what this rank loaded."""
        return dict(self._records)

    def finish(self, *, name_fragment: str | None = None) -> str:
        return merged_identity([self._records], name_fragment=name_fragment)


def merged_identity(
    per_rank_records: Iterable[dict], *, name_fragment: str | None = None
) -> str:
    """Digest of the union of the tensors a group of ranks loaded.

    With load-time filtering (``--enable-ep-weight-filter``) each rank reads
    only the experts it owns, so per-rank digests differ even though the ranks
    together consumed the whole checkpoint. Merging the per-tensor records
    first gives the digest one rank that loaded every tensor would report, and
    a tensor two ranks loaded with different bytes is still rejected.
    """
    merged: dict = {}
    for records in per_rank_records:
        for name, value in records.items():
            if name_fragment is not None and name_fragment not in name:
                continue
            value = (value[0], tuple(value[1]), value[2])
            previous = merged.setdefault(name, value)
            if previous != value:
                raise ValueError(f"Qwen PD ranks loaded different bytes for {name}")
    if not merged:
        raise ValueError("PD identity requires successfully loaded checkpoint tensors")
    payload = json.dumps(merged, sort_keys=True, separators=(",", ":"))
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
