# SPDX-License-Identifier: Apache-2.0
"""Checks that device tensors built from host tensors hold the host bytes.

Enabled by VLLM_TPU_HOST_COPY_CHECK=1, on every rank. While load_model runs,
host-to-device copies made through ``Tensor.to`` and ``Tensor.copy_`` record
the sha256 of the host bytes against the device storage they create or fill
whole. At the end of load_model every parameter and buffer whose storage has
such a record is read back and compared with it: one ``host-copy-check`` line
per tensor (MATCH or DIFF) and one ``host-copy-check-summary`` line per rank.

A DIFF means the device bytes are not the host bytes that were copied. Device
writes made after the copy through ops other than ``copy_`` (``fill_``,
in-place arithmetic, kernels) also show as DIFF, so each line names the
source line of the copy.
"""

import hashlib
import os
import sys
import weakref
from typing import Any

import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_THIS_FILE = os.path.abspath(__file__)


def _sha256(host: torch.Tensor) -> str:
    raw = host.detach().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(raw.numpy())).hexdigest()


def _site() -> str:
    """The first caller outside torch and this file, as file:line."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if os.path.abspath(name) != _THIS_FILE and f"{os.sep}torch{os.sep}" not in name:
            return f"{os.path.basename(name)}:{frame.f_lineno}"
        frame = frame.f_back
    return "?"


class _Recorder:
    def __init__(self) -> None:
        # Storage key -> record. A record is a dict with the host digest, the
        # view it covers (offset, shape, dtype), the copy site, a weak
        # reference to the device tensor and its state.
        self.records: dict[Any, dict[str, Any]] = {}
        self.storage_keys = True
        self.orig_to = torch.Tensor.to
        self.orig_copy = torch.Tensor.copy_
        # Whether torch.Tensor defines the methods itself or inherits them.
        self.own = {n: n in torch.Tensor.__dict__ for n in ("to", "copy_")}

    def key(self, t: torch.Tensor) -> Any:
        if self.storage_keys:
            try:
                return t.untyped_storage()._cdata
            except Exception:  # noqa: BLE001 - backends without storage
                self.storage_keys = False
                logger.info(
                    "host-copy-check: device tensors have no storage; keying by id"
                )
        return id(t)

    def record(self, dest: torch.Tensor, digest: str, site: str) -> None:
        self.records[self.key(dest)] = dict(
            digest=digest,
            view=(dest.storage_offset(), tuple(dest.shape), dest.dtype),
            site=site,
            ref=weakref.ref(dest),
            state="copied",
        )

    def written(self, dest: torch.Tensor, state: str) -> None:
        record = self.records.get(self.key(dest))
        if record is not None:
            record["state"] = state

    def install(self) -> None:
        recorder = self

        def to(self, *args, **kwargs):
            if self.device.type != "cpu":
                return recorder.orig_to(self, *args, **kwargs)
            try:
                device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
            except Exception:  # noqa: BLE001 - leave unparsed calls to torch
                return recorder.orig_to(self, *args, **kwargs)
            if device is None or device.type == "cpu":
                return recorder.orig_to(self, *args, **kwargs)
            digest = _sha256(self) if dtype in (None, self.dtype) else None
            result = recorder.orig_to(self, *args, **kwargs)
            if digest is not None:
                recorder.record(result, digest, _site())
            return result

        def copy_(self, src, non_blocking=False):
            if self.device.type != "cpu" and isinstance(src, torch.Tensor):
                if src.device.type != "cpu":
                    recorder.written(self, "written on device")
                elif (
                    src.dtype == self.dtype
                    and tuple(src.shape) == tuple(self.shape)
                    and self.is_contiguous()
                    and self.storage_offset() == 0
                ):
                    recorder.record(self, _sha256(src), _site())
                else:
                    recorder.written(self, "partly copied")
            return recorder.orig_copy(self, src, non_blocking)

        torch.Tensor.to = to
        torch.Tensor.copy_ = copy_

    def uninstall(self) -> None:
        for name, orig in (("to", self.orig_to), ("copy_", self.orig_copy)):
            if self.own[name]:
                setattr(torch.Tensor, name, orig)
            else:
                delattr(torch.Tensor, name)


_recorder: _Recorder | None = None


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def start() -> None:
    """Starts recording host-to-device copies when the check is enabled."""
    global _recorder
    if not envs.VLLM_TPU_HOST_COPY_CHECK or _recorder is not None:
        return
    _recorder = _Recorder()
    _recorder.install()


def finish(models: dict[str, nn.Module | None]) -> None:
    """Stops recording and compares every recorded parameter and buffer."""
    global _recorder
    recorder, _recorder = _recorder, None
    if recorder is None:
        return
    recorder.uninstall()
    rank = _rank()
    counts = dict(match=0, diff=0, not_host_built=0, written_after=0)
    seen: set[Any] = set()
    for label, model in models.items():
        if model is None:
            continue
        for kind, named in (
            ("param", model.named_parameters(remove_duplicate=False)),
            ("buffer", model.named_buffers(remove_duplicate=False)),
        ):
            for name, tensor in named:
                if tensor.device.type == "cpu":
                    continue
                key = recorder.key(tensor)
                record = recorder.records.get(key)
                view = (tensor.storage_offset(), tuple(tensor.shape), tensor.dtype)
                if record is None:
                    counts["not_host_built"] += 1
                    continue
                dest = record["ref"]()
                # "same": the tensor the copy produced or filled; "alias": a
                # tensor sharing its storage (Parameter.data); "storage": the
                # copied tensor is gone and only the storage key matches.
                how = (
                    "same"
                    if dest is tensor
                    else ("alias" if dest is not None else "storage")
                )
                if not recorder.storage_keys and how != "same":
                    counts["not_host_built"] += 1
                    continue
                if record["state"] != "copied" or record["view"] != view:
                    counts["written_after"] += 1
                    logger.info(
                        "host-copy-check rank=%d %s %s %s SKIP %s (copied at %s)",
                        rank,
                        label,
                        kind,
                        name,
                        record["state"]
                        if record["state"] != "copied"
                        else "other view",
                        record["site"],
                    )
                    continue
                readback = _sha256(tensor.detach().cpu())
                same = readback == record["digest"]
                counts["match" if same else "diff"] += 1
                if key in seen and same:
                    continue
                seen.add(key)
                logger.info(
                    "host-copy-check rank=%d %s %s %s %s %s %s expected=%s "
                    "readback=%s site=%s via=%s",
                    rank,
                    label,
                    kind,
                    name,
                    tensor.dtype,
                    list(tensor.shape),
                    "MATCH" if same else "DIFF",
                    record["digest"],
                    readback,
                    record["site"],
                    how,
                )
    logger.info(
        "host-copy-check-summary rank=%d match=%d diff=%d written_after=%d "
        "not_host_built=%d copies_recorded=%d storage_keys=%s",
        rank,
        counts["match"],
        counts["diff"],
        counts["written_after"],
        counts["not_host_built"],
        len(recorder.records),
        recorder.storage_keys,
    )
