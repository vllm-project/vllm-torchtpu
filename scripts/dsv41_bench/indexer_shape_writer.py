# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the real FP8 cache writer with a benchmark-only token guard.

Run on TPU from the repository root (no Torch dependency)::

    python scripts/dsv41_bench/indexer_shape_writer.py --out /tmp/shape-writer \
        --tokens 128 256 288 384 512 --iterations 100 --profile-iterations 20

Baseline freezes the original T288-only guard in a separately loaded real writer.
Candidate uses the actual production writer and its broadened token guard.
Host timings include dispatch and synchronization, NOT device-only latency. Use
saved HLO and optional profiler device tracks to resolve small device differences.
"""

import argparse
import functools
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from indexer_prepare_probe import cache, reference

TOKEN_COUNTS = (128, 256, 288, 384, 512)


def baseline_guard(
    banks: tuple[jax.Array, ...],
    records: jax.Array,
    keys: jax.Array,
    page_size: int,
) -> bool:
    """Freeze the original prepare.py guard before token eligibility expansion."""
    return (
        jax.default_backend() == "tpu"
        and keys.shape == (288, 128)
        and keys.dtype == jnp.bfloat16
        and records.shape == (288,)
        and records.dtype == jnp.int32
        and page_size == 256
        and len(banks) == 2
        and all(
            bank.ndim == 2 and bank.shape[1] == 128 and bank.dtype == jnp.int32
            for bank in banks
        )
    )


def load_baseline() -> ModuleType:
    """Load the real writer with private globals, without patching production."""
    name = f"{cache.__package__}._shape_writer_baseline"
    spec = importlib.util.spec_from_file_location(name, cache.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass resolves its defining module here.
    spec.loader.exec_module(module)
    module.supports_sorted_fp8 = baseline_guard
    return module


def fixture(tokens: int, offset: int, kind: str, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    stride, bank_rows, page_size = 128, 512, 256
    layout = cache.PackedFP8Layout(8 - bool(offset), stride, offset, page_size)
    banks = tuple(
        rng.integers(-(2**31), 2**31, (bank_rows, 128), dtype=np.int32)
        for _ in range(2)
    )
    records = rng.choice(layout.pages * page_size, tokens, replace=False).astype(
        np.int32
    )
    # Ensure both banks are touched, even if a future smaller fixture is added.
    records = records[~np.isin(records, [0, layout.pages * page_size - 1])]
    extra = rng.permutation(layout.pages * page_size).astype(np.int32)
    extra = extra[~np.isin(extra, np.r_[records, 0, layout.pages * page_size - 1])]
    records = np.r_[0, layout.pages * page_size - 1, records, extra][:tokens].astype(
        np.int32
    )
    if kind == "concentrated":
        pool = np.r_[
            np.arange(page_size),
            np.arange((layout.pages - 1) * page_size, layout.pages * page_size),
        ]
        records = rng.choice(pool, tokens, replace=False).astype(np.int32)
    if kind in ("mixed", "invalid"):
        invalid = np.arange(2 if kind == "mixed" else 0, tokens)
        if kind == "mixed":
            invalid = rng.choice(invalid, tokens // 4, replace=False)
        records[invalid] = np.where(
            np.arange(invalid.size) % 2,
            layout.pages * page_size + np.arange(invalid.size),
            -1 - np.arange(invalid.size),
        )
    rng.shuffle(records)
    keys = rng.normal(size=(tokens, 128)).astype(ml_dtypes.bfloat16)
    return banks, records, keys, layout


def expected_cache(banks: tuple, records: np.ndarray, keys: np.ndarray, layout):
    """Independent NumPy address writes, using the existing NumPy FP8 oracle."""
    _, packed = reference(
        keys, records % layout.page_size, np.arange(records.size, dtype=np.int32)
    )
    expected = np.concatenate(banks).copy()
    for index, record in enumerate(records):
        if not 0 <= record < layout.pages * layout.page_size:
            continue
        addresses = (
            (layout.page_offset + record // layout.page_size * layout.page_stride)
            * cache.PAGE_RECORDS
            + np.arange(cache.PAGE_ROWS) * layout.page_size
            + record % layout.page_size
        )
        expected.reshape(-1)[addresses] = packed[:, index]
    return expected


def device_inputs(banks: tuple, records: np.ndarray, keys: np.ndarray) -> tuple:
    # Explicit copies prevent A/B from sharing aliased Pallas owner buffers.
    return jax.block_until_ready(
        (
            tuple(jnp.array(b, copy=True) for b in banks),
            jnp.array(records),
            jnp.array(keys),
        )
    )


def check_inputs(inputs: tuple, banks: tuple) -> None:
    for actual, expected in zip(inputs[0], banks, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), expected)


def run_shape(baseline: ModuleType, args: argparse.Namespace, tokens: int, offset: int):
    out = args.out / f"t{tokens}-offset{offset}"
    out.mkdir(parents=True, exist_ok=True)
    banks, records, keys, layout = fixture(tokens, offset, "mixed", args.seed)
    functions = {
        "baseline": functools.partial(baseline.write_fp8_records, layout=layout),
        "candidate": functools.partial(cache.write_fp8_records, layout=layout),
    }
    inputs = {name: device_inputs(banks, records, keys) for name in functions}
    guards = {
        "baseline": baseline.supports_sorted_fp8(*inputs["baseline"], layout.page_size),
        "candidate": cache.supports_sorted_fp8(*inputs["candidate"], layout.page_size),
    }
    assert guards == {"baseline": tokens == 288, "candidate": True}, guards
    jaxprs = {
        name: str(jax.make_jaxpr(fn)(*inputs[name])) for name, fn in functions.items()
    }
    for name, text in jaxprs.items():
        (out / f"{name}.jaxpr.txt").write_text(text)
        assert ("indexer_sorted_fp8_prepare" in text) == guards[name]
    same_jaxpr = jaxprs["baseline"] == jaxprs["candidate"]
    if tokens == 288:
        assert same_jaxpr, "T288 production and candidate jaxprs must be identical"

    compiled = {}
    module_mapping = {}
    for name, fn in functions.items():
        lowered = jax.jit(fn, donate_argnums=(0,) if args.donate else ()).lower(
            *inputs[name]
        )
        (out / f"{name}.stablehlo.txt").write_text(lowered.as_text())
        compiled[name] = lowered.compile()
        hlo = compiled[name].as_text()
        (out / f"{name}.optimized-hlo.txt").write_text(hlo)
        modules = compiled[name].runtime_executable().hlo_modules()
        module_mapping[name] = {
            "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            "modules": [module.name for module in modules],
            "annotation": f"{name}_writer_t{tokens}_offset{offset}",
            "profile_directory": f"profile/{name}",
            "attribution": "isolated capture executes only this side; not order-based",
        }
        for index, module in enumerate(modules):
            (out / f"{name}.module{index}.hlo.txt").write_text(module.to_string())
    (out / "profile-module-mapping.json").write_text(
        json.dumps(module_mapping, indent=2) + "\n"
    )

    def execute(name: str) -> None:
        updated = jax.block_until_ready(compiled[name](*inputs[name]))
        if args.donate:
            # Never reuse a donated owner or share one between the A/B chains.
            inputs[name] = (updated, *inputs[name][1:])

    validation = []
    for kind in ("mixed", "valid", "concentrated", "invalid"):
        host_banks, slots, values, _ = fixture(tokens, offset, kind, args.seed)
        expected = expected_cache(host_banks, slots, values, layout)
        outputs = {}
        for name, executable in compiled.items():
            check = device_inputs(host_banks, slots, values)
            outputs[name] = np.concatenate(jax.block_until_ready(executable(*check)))
            np.testing.assert_array_equal(outputs[name], expected, err_msg=name + kind)
            if not args.donate:
                check_inputs(check, host_banks)
        np.testing.assert_array_equal(outputs["baseline"], outputs["candidate"])
        validation.append({"case": kind, "three_way_byte_exact": True})

    for _ in range(args.warmup):
        for name in compiled:
            execute(name)
    samples = {name: [] for name in compiled}
    names = tuple(compiled)
    for iteration in range(args.iterations):
        for name in names[:: 1 if iteration % 2 == 0 else -1]:
            start = time.perf_counter_ns()
            execute(name)
            samples[name].append((time.perf_counter_ns() - start) / 1000)

    # Isolated captures disambiguate even T288's identical executable modules.
    # These diagnostic passes do not contribute to the AB/BA timing samples.
    if args.profile_iterations:
        for name in names:
            with jax.profiler.trace(str(out / "profile" / name)):
                for iteration in range(args.profile_iterations):
                    with jax.profiler.TraceAnnotation(
                        module_mapping[name]["annotation"], step_num=iteration
                    ):
                        execute(name)
    if args.donate:
        if args.warmup or args.iterations or args.profile_iterations:
            expected = expected_cache(banks, records, keys, layout)
            for value in inputs.values():
                np.testing.assert_array_equal(np.concatenate(value[0]), expected)
    else:
        for value in inputs.values():
            check_inputs(value, banks)
    result = {
        "tokens": tokens,
        "offset": offset,
        "layout": vars(layout),
        "bank_shapes": [list(b.shape) for b in banks],
        "fused_guard": guards,
        "same_jaxpr": same_jaxpr,
        "validation": validation,
        "inputs_preserved": not args.donate,
        "donation": args.donate,
        "bank_policy": "independent_returned_bank_chains"
        if args.donate
        else "immutable",
        "profile_module_mapping": module_mapping,
        "scope": "actual_write_fp8_records_including_quantize_sort_prepare_and_write",
        "timing": (
            "host_dispatch_plus_device_synchronized_donated"
            if args.donate
            else "host_dispatch_plus_device_synchronized_non_donating"
        ),
        "timing_fixture": "mixed",
        "order": "alternating_AB_BA",
        "samples_us": samples,
        "median_us": {n: float(np.median(s)) for n, s in samples.items() if s},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "samples_us"}), flush=True
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        choices=TOKEN_COUNTS,
        default=list(TOKEN_COUNTS),
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument(
        "--donate",
        action="store_true",
        help="Donate banks and chain returned owners independently for each side",
    )
    args = parser.parse_args()
    if min(args.iterations, args.warmup, args.profile_iterations) < 0:
        parser.error("iteration counts and warmup must be nonnegative")
    if len(set(args.tokens)) != len(args.tokens):
        parser.error("--tokens must not contain duplicates")
    if jax.default_backend() != "tpu":
        parser.error("Requires TPU: CPU fallback cannot measure the real fused writer")
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(cache.__file__),
        Path(cache.__file__).with_name("prepare.py"),
        Path(__file__),
        Path(__file__).with_name("indexer_prepare_probe.py"),
    ]
    metadata = {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "device_kind": jax.devices()[0].device_kind,
        "command": sys.argv,
        "options": {**vars(args), "out": str(args.out)},
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "baseline": "private real cache writer module with frozen original T288 guard",
        "candidate": "actual production cache.write_fp8_records and production guard",
        "donation": args.donate,
        "limitation": (
            "Donated timing uses repeated idempotent writes to independent bank chains."
            if args.donate
            else "Non-donating writer may copy banks; HLO/host/device timings "
            "include those copies and do not model donated production costs."
        ),
        "profile_policy": "separate labeled capture per side; diagnostic not AB/BA",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for path in sources:
        (args.out / path.name).write_bytes(path.read_bytes())
    baseline = load_baseline()
    results = []
    for tokens in args.tokens:
        for offset in (0, 16):
            results.append(run_shape(baseline, args, tokens, offset))
            (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
