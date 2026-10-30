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
"""Bounded identity of checkpoint tensors actually consumed by a PD model."""

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch
from torch.overrides import TorchFunctionMode


class LoadedWeightIdentity(TorchFunctionMode):
    def __init__(self, destinations: Iterable[torch.Tensor] = ()) -> None:
        super().__init__()
        self._destinations = {tensor.untyped_storage() for tensor in destinations}
        self._current_name = ""
        self._copy_index = 0
        self._records: dict[str, tuple[str, tuple[int, ...], str]] = {}

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._records:
            raise ValueError(f"Duplicate PD checkpoint tensor: {name}")
        if tensor.device.type != "cpu":
            raise ValueError("PD identity requires CPU checkpoint slices")
        digest = hashlib.sha256()
        limit = 4 * 1024 * 1024

        def update(view: torch.Tensor) -> None:
            if view.is_contiguous() and not view.is_conj() and not view.is_neg():
                raw = memoryview(view.detach().reshape(-1).view(torch.uint8).numpy())
                for offset in range(0, len(raw), limit):
                    digest.update(raw[offset : offset + limit])
            elif view.numel() * view.element_size() <= limit:
                update(view.resolve_conj().resolve_neg().contiguous())
            elif view.shape[0] == 1:
                update(view[0])
            else:
                row_bytes = view[0].numel() * view.element_size()
                rows = max(1, limit // row_bytes)
                for start in range(0, view.shape[0], rows):
                    update(view.narrow(0, start, min(rows, view.shape[0] - start)))

        update(tensor)
        self._records[name] = (
            str(tensor.dtype),
            tuple(tensor.shape),
            digest.hexdigest(),
        )

    @contextmanager
    def observe(self, name: str) -> Iterator[None]:
        self._current_name, self._copy_index = name, 0
        with self:
            yield

    def __torch_function__(
        self,
        func: Callable[..., object],
        types,
        args: tuple[object, ...] = (),
        kwargs=None,
    ):
        result = func(*args, **(kwargs or {}))
        if (
            func is torch.Tensor.copy_
            and len(args) >= 2
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and args[0].untyped_storage() in self._destinations
            and args[1].device.type == "cpu"
        ):
            self.add(f"{self._current_name}/{self._copy_index}", args[1])
            self._copy_index += 1
        return result

    def finish(self, *, name_fragment: str | None = None) -> str:
        if not self._records:
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
        (r["tp_size"], r["config"], r["state_flags"])
        != (
            records[0]["tp_size"],
            records[0]["config"],
            records[0]["state_flags"],
        )
        for r in records[1:]
    ):
        raise ValueError("Qwen PD peers disagree on model or state configuration")
    return dict(schema="qwen4-loaded-state-v1", ranks=records)
