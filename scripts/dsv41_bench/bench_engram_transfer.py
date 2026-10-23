# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate Engram DMA against changing compiled producers on one TPU device."""

import argparse
import importlib.metadata
import importlib.util
import json
import random
import statistics
import sys
import time
from functools import partial
from pathlib import Path


def stats(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "samples_ms": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import sync

    source = Path(__file__).resolve().parents[2] / (
        "src/vllm_torchtpu/models/vllm/deepseek_v41/common/host_transfer.py"
    )
    spec = importlib.util.spec_from_file_location("engram_dma_probe", source)
    transfer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = transfer
    spec.loader.exec_module(transfer)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    device = torch.device("tpu:0")
    torch.empty(1, device=device).cpu()
    result = {
        "scope": "synthetic completed-transfer latency on one device",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-tpu", "tpu-sync-torch", "libtpu")
        },
        "cases": [],
        "passed": False,
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2))

    save()
    rng = random.Random(123)
    for tokens in (1, 3, 128, 288, 2048):
        # Eight FP8 heads of 256 elements plus one scale byte per 32 values.
        words = tokens * (8 * 256 // 4 + 8 * 8 // 4)
        host = torch.arange(words, dtype=torch.int32)
        ordinary = host.to(device)
        slots = [transfer.PreparedEngramUpload(words, device) for _ in range(2)]
        audit = transfer.PreparedEngramUpload(words, device)
        original = audit.host_words.clone()
        audit.device.copy_(host)
        sync.synchronize(tensors=[audit.device], wait=True)
        audit.transfer.d2h()
        assert torch.equal(audit.device.cpu(), host)
        copy_retargets_buffer = torch.equal(audit.host_words, original)
        assert copy_retargets_buffer or torch.equal(audit.host_words, host)
        del audit
        consume = torch.compile(lambda data: data + 7, backend="tpu", fullgraph=True)
        outputs = []
        for step in range(8):
            slot = slots[step % 2]
            slot.acquire().copy_(host + step)
            slot.start()
            slot.activate()
            output = consume(slot.device)
            sync.synchronize(tensors=[output], wait=False)
            slot.record_consumers([output])
            outputs.append(output)
        for step, output in enumerate(outputs):
            assert torch.equal(output.cpu(), host + step + 7), (tokens, step, "H2D")
        for slot in slots:
            slot.release_consumers()

        download = transfer.EngramHashDownload()
        host_hashes = torch.arange(tokens * 2 * 8, dtype=torch.int32).view(tokens, 2, 8)
        seed = host_hashes.to(device)
        produce = torch.compile(
            lambda data: data * 3 + 11, backend="tpu", fullgraph=True
        )
        for step in range(8):
            current = produce(seed + step)
            actual = download.copy_to_host(current)
            assert torch.equal(actual, (host_hashes + step) * 3 + 11), (
                tokens,
                step,
                "D2H",
            )

        slot = slots[0]
        slot.acquire().copy_(host)
        current = produce(seed)
        sync.synchronize(tensors=[current], wait=True)
        methods = {
            "h2d_copy": partial(ordinary.copy_, host),
            "h2d_prepared": slot.transfer.h2d,
            "h2d_prepared_with_host_copy": partial(slot.copy_from, host),
            "d2h_to_cpu": partial(current.to, "cpu"),
            "d2h_raw_rebind": partial(download.copy_to_host, current),
        }
        for fn in methods.values():
            for _ in range(5):
                fn()
                torch.accelerator.synchronize()
        samples = {name: [] for name in methods}
        for iteration in range(args.iterations):
            current = produce(seed + iteration)
            sync.synchronize(tensors=[current], wait=True)
            methods["d2h_to_cpu"] = partial(current.to, "cpu")
            methods["d2h_raw_rebind"] = partial(download.copy_to_host, current)
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                torch.accelerator.synchronize()
                started = time.perf_counter_ns()
                methods[name]()
                torch.accelerator.synchronize()
                samples[name].append((time.perf_counter_ns() - started) / 1e6)
        case = {
            "tokens": tokens,
            "upload_logical_bytes": host.nbytes,
            "upload_physical_bytes": slots[0].host_buffer.size_bytes,
            "hash_logical_bytes": host_hashes.nbytes,
            "copy_retargets_buffer": copy_retargets_buffer,
            "pipelined_consumers_equal": len(outputs),
            "changing_producers_equal": 8,
            "timings": {name: stats(values) for name, values in samples.items()},
        }
        result["cases"].append(case)
        save()
        print(
            json.dumps(
                {
                    "tokens": tokens,
                    "median_ms": {
                        name: value["median_ms"]
                        for name, value in case["timings"].items()
                    },
                }
            ),
            flush=True,
        )
    result["passed"] = True
    save()


if __name__ == "__main__":
    main()
