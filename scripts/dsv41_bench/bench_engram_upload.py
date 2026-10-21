# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone Engram H2D correctness and completed-transfer wall-time probe.

Run with the repository's torch/torch-tpu/tpu-sync pins. No model is loaded.
The consumer uses the production byte unpack and dequant functions, loaded
from source without importing the vLLM server. This is not a serving benchmark.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import mmap
import random
import statistics
import sys
import tempfile
import time
import traceback
import types
from dataclasses import dataclass
from pathlib import Path


def load_upload(path):
    package = types.ModuleType("engram_upload_fixture")
    package.__path__ = [str(path.parent)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        "engram_upload_fixture.host_transfer", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PreparedEngramUpload


def load_consumer(path, torch):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"dequant_fp8_rows", "_unpack_bytes"}
    ]
    assert len(nodes) == 2
    namespace = {"torch": torch, "_INVALID_SCALE": 255}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_unpack_bytes"], namespace["dequant_fp8_rows"]


def load_staging(path, torch, unpack, dequant):
    """Execute real staging methods without unrelated model/server imports."""
    tree = ast.parse(path.read_text())
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Engram"
    )
    methods = {
        "_init_staging",
        "_staged_tensors",
        "_staged_views",
        "enable_prepared_upload",
        "_reserve_upload",
        "_activate_upload",
        "record_upload_consumers",
        "prepare_embeddings",
        "start_prepare_embeddings",
        "finish_prepare_embeddings",
        "_ready_rows",
    }
    cls = ast.ClassDef(
        name="EngramStagingFixture",
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ],
    )
    staging = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HostEngramStaging"
    )
    module = types.ModuleType("engram_upload_fixture.staging")
    module.__package__ = "engram_upload_fixture"
    module.__dict__.update(
        torch=torch,
        dataclass=dataclass,
        PendingRowExchange=object,
        PreparedEngramUpload=object,
        get_current_vllm_config_or_none=lambda: None,
        logger=logging.getLogger(__name__),
        _INVALID_SCALE=255,
        _unpack_bytes=unpack,
        dequant_fp8_rows=dequant,
    )
    sys.modules[module.__name__] = module
    code = ast.fix_missing_locations(ast.Module(body=[staging, cls], type_ignores=[]))
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module.EngramStagingFixture


def stats(samples):
    values = sorted(samples)
    return {
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, int(len(values) * 0.95))],
        "min_ms": values[0],
        "samples_ms": samples,
    }


def check_staging(path, torch, device, prepared):
    unpack, dequant = load_consumer(path, torch)
    Staging = load_staging(path, torch, unpack, dequant)
    cols, dim, block = 2, 32, 32
    weight = torch.randint(0, 120, (64, dim), dtype=torch.uint8)
    scales = torch.randint(120, 132, (64, 1), dtype=torch.uint8)

    def gather(ids, raw, scale):
        torch.index_select(weight, 0, ids.flatten(), out=raw.view(-1, dim))
        torch.index_select(scales, 0, ids.flatten(), out=scale.view(-1, 1))

    emb = types.SimpleNamespace(
        host_offload=True,
        part_n_hash_cols=cols,
        dp_size=1,
        block_size=block,
        gather_raw=gather,
    )
    staging = Staging()
    staging.embed_tokens = emb
    with torch.device(device):
        staging._init_staging(4, dim)
    if prepared:
        staging.enable_prepared_upload()
    # The standalone dynamic compiler cannot lower the FP8 bitcast in this
    # runtime. Exercise staging eagerly; run_case checks a static compiled consumer.
    consume = staging._ready_rows
    identities = {}
    for tokens in (4, 1, 3, 4, 1, 3):
        ids = torch.randint(64, (tokens, cols))
        staging.prepare_embeddings(ids)
        result = consume(tokens)
        staging.record_upload_consumers([result])
        actual = result.cpu()
        expected = dequant(weight[ids], scales[ids], block)
        assert torch.equal(actual, expected), (tokens, prepared)
        pair = (id(staging._staged_tensors(tokens)), id(staging._active_raw))
        if tokens in identities and not prepared:
            assert identities[tokens] == pair
        identities[tokens] = pair
        raw_words = tokens * cols * dim // 4
        packed = (
            staging._active_upload.host_words
            if prepared
            else staging._host_packed_buf.view(torch.int32)
        )
        padding = packed[raw_words:].view(torch.uint8)
        assert torch.all(padding[tokens * cols : tokens * 4] == 255)
    return {
        "prepared": prepared,
        "buckets": list(identities),
        "iterations": 6,
        "exact_equal": True,
        "cached_tensor_and_view_identity": not prepared,
        "double_buffered": prepared,
        "scale_padding_checked": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-source", type=Path, required=True)
    parser.add_argument("--engram-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("metrics.json"))
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 3, 32, 128, 288, 2048, 8192]
    )
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    if args.cpu_check:
        import torch

        print(json.dumps(check_staging(args.engram_source, torch, "cpu", False)))
        return
    result = {
        "run_id": "20261021T081725Z-44074cc03da2",
        "baseline_sha": "bdb38700dd25959c029c7d77f00cee91ccd3a0bc",
        "scope": "one process/device; synthetic mmap table; completed H2D wall time",
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torch-tpu", "tpu-sync-torch", "libtpu")
        },
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.upload_source, args.engram_source)
        },
        "cases": [],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, default=str))

    save()
    import torch
    import torch_tpu  # noqa: F401
    from tpu_sync.api.torch import raw_transfer

    Upload = load_upload(args.upload_source)
    unpack, dequant = load_consumer(args.engram_source, torch)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    device = torch.device("tpu:0")
    torch.empty(1, device=device).cpu()
    result["device_name"] = torch.tpu.get_device_name(0)
    result["pacing"] = raw_transfer.get_dma_pacing()
    result["torch_threads"] = torch.get_num_threads()
    result["staging_integration"] = check_staging(
        args.engram_source, torch, device, True
    )
    save()

    # Same-host shared-memory representation; no other rank reads this fixture.
    rows, cols, head_dim, block_size = 65536, 8, 256, 32
    blocks = head_dim // block_size
    with tempfile.TemporaryFile(dir="/dev/shm") as file:
        file.truncate(rows * (head_dim + blocks))
        mapping = mmap.mmap(file.fileno(), 0, flags=mmap.MAP_SHARED)
    table = torch.frombuffer(mapping, dtype=torch.uint8)
    table[: rows * head_dim].copy_(
        torch.randint(0, 120, (rows * head_dim,), dtype=torch.uint8)
    )
    table[rows * head_dim :].copy_(
        torch.randint(120, 132, (rows * blocks,), dtype=torch.uint8)
    )
    weight = table[: rows * head_dim].view(rows, head_dim)
    scales = table[rows * head_dim :].view(rows, blocks)

    def run_case(tokens):
        case = {
            "tokens": tokens,
            "cols": cols,
            "head_dim": head_dim,
            "block_size": block_size,
        }
        result["cases"].append(case)
        save()
        raw_words = tokens * cols * head_dim // 4
        scale_words = tokens * ((cols * blocks + 3) // 4)
        words = raw_words + scale_words
        packed = torch.empty(words, dtype=torch.int32)
        raw_out = packed[:raw_words].view(torch.uint8).view(-1, head_dim)
        scale_buffer = torch.empty(tokens * cols, blocks, dtype=torch.uint8)
        indices = torch.randint(rows, (tokens * cols,))

        def gather_pack():
            torch.index_select(weight, 0, indices, out=raw_out)
            torch.index_select(scales, 0, indices, out=scale_buffer)
            packed[raw_words:].copy_(scale_buffer.view(-1).view(torch.int32))

        gather_pack()
        candidate = Upload(words, device)
        baseline = packed.to(device)
        case.update(
            logical_bytes=packed.nbytes,
            physical_bytes=candidate.transfer.physical_size_bytes,
            layout=candidate.layout,
            host_buffer_pjrt_backed=candidate.host_buffer.is_pjrt_backed,
            host_tensor_reports_pinned=candidate.host_bytes.is_pinned(),
        )

        # Test the pointer lifetime claim on the exact installed runtime.
        audit = Upload(words, device)
        previous = audit.host_words.clone()
        audit.device.copy_(packed)
        audit.transfer.d2h()
        case["copy_retargets_buffer"] = bool(
            torch.equal(audit.host_words, previous)
            and not torch.equal(previous, packed)
            and torch.equal(audit.device.cpu(), packed)
        )
        del audit

        def consumer(view):
            by_word = view.view(words // tokens, tokens)
            raw = unpack(by_word[: raw_words // tokens].reshape(-1), raw_words * 4)
            scale = unpack(
                by_word[raw_words // tokens :].reshape(-1), tokens * cols * blocks
            )
            return dequant(
                raw.view(tokens, cols, head_dim),
                scale.view(tokens, cols, blocks),
                block_size,
            )

        compiled = torch.compile(consumer, backend="tpu", fullgraph=True, dynamic=False)
        candidate_view = candidate.device.view(tokens, words // tokens)
        checks = []
        for iteration in range(4):
            indices.random_(rows)
            gather_pack()
            if iteration == 3:
                packed[raw_words:].view(torch.uint8)[::7] = 255
            reference = consumer(packed.view(tokens, words // tokens))
            candidate.copy_from(packed)
            actual = compiled(candidate_view).cpu()
            assert torch.equal(actual, reference), (
                tokens,
                iteration,
                (actual.float() - reference.float()).abs().max().item(),
            )
            assert torch.equal(candidate.device.cpu(), packed)
            checks.append(True)
        case["repeated_compiled_consumer_equal"] = checks

        # Keep consumers asynchronous while uploading the next step to the
        # other slot. Validate every result only after several slot reuses.
        from torch_tpu._internal import sync

        slots = [Upload(words, device), Upload(words, device)]
        outputs = []
        references = []
        for iteration in range(8):
            slot = slots[iteration % 2]
            indices.random_(rows)
            gather_pack()
            references.append(consumer(packed.view(tokens, words // tokens)))
            slot.acquire().copy_(packed)
            slot.start()
            slot.activate()
            output = compiled(slot.device.view(tokens, words // tokens))
            sync.synchronize(tensors=[output], wait=False)
            slot.record_consumers([output])
            outputs.append(output)
        for output, reference in zip(outputs, references, strict=True):
            assert torch.equal(output.cpu(), reference), (tokens, "pipeline")
        case["pipelined_compiled_consumers_equal"] = len(outputs)
        del slots, outputs, references
        save()

        candidate.host_words.copy_(packed)
        methods = {
            "copy_": lambda: baseline.copy_(packed),
            "raw_api": lambda: raw_transfer.transfer_h2d(
                candidate.host_bytes, candidate.device
            ),
            "reprepare_api": lambda: raw_transfer.PreparedTorchRawTransfer(
                candidate.device, candidate.host_buffer, unsafe_skip_buffer_lock=False
            ).h2d(),
            "prepared_api": candidate.transfer.h2d,
            "prepared_upload_with_copy": lambda: candidate.copy_from(packed),
            "host_gather_pack": gather_pack,
        }
        for fn in methods.values():
            for _ in range(5):
                fn()
        samples = {name: [] for name in methods}
        rng = random.Random(20261021)
        for _ in range(args.iterations):
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                torch.accelerator.synchronize()
                start = time.perf_counter_ns()
                methods[name]()
                samples[name].append((time.perf_counter_ns() - start) / 1e6)
        case["timings"] = {name: stats(values) for name, values in samples.items()}
        case["correct"] = True
        save()
        print(
            "CASE",
            json.dumps({k: v for k, v in case.items() if k != "timings"}),
            flush=True,
        )
        print(
            "TIMING",
            json.dumps({k: v["median_ms"] for k, v in case["timings"].items()}),
            flush=True,
        )

        # Save timings before optional profiling; profiler failures must not erase them.
        if args.profile_dir is not None and tokens == max(args.tokens):
            from torch_tpu._internal.profiler import TpuProfilerConfig

            profile_dir = str(args.profile_dir)
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profile_dir, use_gzip=True
                ),
                experimental_config=TpuProfilerConfig(
                    run_dir=profile_dir,
                    host_tracer_level=2,
                    device_tracer_level=1,
                    python_tracer_level=1,
                ),
            ):
                for name in (
                    "copy_",
                    "raw_api",
                    "prepared_api",
                    "prepared_upload_with_copy",
                ):
                    for _ in range(5):
                        with torch.profiler.record_function("engram_upload/" + name):
                            methods[name]()
                            torch.accelerator.synchronize()
            result["profile_dir"] = profile_dir
            save()

    for tokens in args.tokens:
        run_case(tokens)
    result["passed"] = True
    save()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
