# SPDX-License-Identifier: Apache-2.0
"""CPU-only distributed regression for Engram token/head redistribution."""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# This helper has no vLLM/TPU dependencies; exercise real Gloo collectives on
# CPU-only development hosts without importing the TPU platform package.
_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/vllm_torchtpu/models/vllm/deepseek_v41/common/host_exchange.py"
)
_SPEC = importlib.util.spec_from_file_location("host_exchange", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
exchange_rows = _MODULE.exchange_rows
start_exchange_rows = _MODULE.start_exchange_rows


def _worker(rank: int, size: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=45),
    )
    try:
        for slot in (1, 7, 2):
            # Distinguish source, destination, token, head and raw/scale bytes.
            raw = torch.empty(size, slot, 2, 4, dtype=torch.uint8)
            scales = torch.empty(size, slot, 2, 1, dtype=torch.uint8)
            for destination in range(size):
                for token in range(slot):
                    for head in range(2):
                        raw[destination, token, head] = (
                            rank * 24 + destination * 3 + token + head
                        )
                        scales[destination, token, head] = (
                            255 if token == slot - 1 else 127 + rank + head
                        )
            out, scale_out = exchange_rows(
                raw.flatten(0, 1), scales.flatten(0, 1), dist.group.WORLD, size
            )
            assert out.shape == (slot, size * 2, 4)
            for source in range(size):
                for token in range(slot):
                    for head in range(2):
                        assert torch.all(
                            out[token, source * 2 + head]
                            == (source * 24 + rank * 3 + token + head)
                        )
                        assert scale_out[token, source * 2 + head, 0] == (
                            255 if token == slot - 1 else 127 + source + head
                        )
            # Both layers are in flight before either wait. Distinct payloads
            # and alternating buckets expose stale/reused communication buffers.
            pending = [
                start_exchange_rows(
                    (raw + offset).flatten(0, 1),
                    scales.flatten(0, 1),
                    dist.group.WORLD,
                    size,
                )
                for offset in (0, 64)
            ]
            for offset, transfer in zip((0, 64), pending):
                actual, actual_scales = transfer.wait()
                assert torch.equal(actual, out + offset)
                assert torch.equal(actual_scales, scale_out)
    finally:
        dist.destroy_process_group()


def test_exchange_retains_local_tokens_and_head_order(tmp_path):
    size = 2
    mp.spawn(_worker, args=(size, (tmp_path / "gloo").as_uri()), nprocs=size)


def test_rejects_partial_dp_slot():
    with pytest.raises(ValueError, match="equal DP token slots"):
        exchange_rows(torch.zeros(3, 2, 4), torch.zeros(3, 2, 1), None, 2)
