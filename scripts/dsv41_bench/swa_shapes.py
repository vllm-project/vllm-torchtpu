# SPDX-License-Identifier: Apache-2.0
"""Compare frozen original SWA dispatch with the actual production candidate.

TPU: python scripts/dsv41_bench/swa_shapes.py --cached-write --out /tmp/swa
CPU: JAX_PLATFORMS=cpu python scripts/dsv41_bench/swa_shapes.py --cpu-reference

Original preparation AND writer are pinned to 6541810d4. Their reciprocal local
imports are redirected to private frozen modules; candidate imports stay intact.
CPU mode checks metadata/pack math with scalar publication, not Pallas or DMA.
"""

import argparse
import ast
import hashlib
import importlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ((96, 16), (192, 32), (288, 32), (288, 48))
CASES = ("regular", "ragged", "empty", "padded")


BASELINE_REVISION = "6541810d49abf20ef6257d8f282c003106b67a7e"
BASELINE_HASHES = {
    "experimental.decode_preparation": (
        "6e6a04385700cad26f29105cce112b386652ca8ac877316ef3d46c0b259663bd"
    ),
    "mla.paged_swa": "daccdf3f0bd5ad3c0df4186d1b5a58354b9a887e4c27c1cafd369c0b5372717e",
}


def load_modules(baseline_dir=None):
    """Stub package initializers only; execute unmodified kernel modules."""
    for name in (
        "vllm_torchtpu",
        "vllm_torchtpu.kernels",
        "vllm_torchtpu.kernels.experimental",
        "vllm_torchtpu.kernels.deepseek_v41",
        "vllm_torchtpu.kernels.deepseek_v41.csa2",
        "vllm_torchtpu.kernels.mla",
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
        sys.modules.setdefault(name, package)
    prefix = "vllm_torchtpu.kernels."
    private = {prefix + key: prefix + key + "_swa_original" for key in BASELINE_HASHES}
    frozen, fingerprints = {}, {}
    for key, expected in BASELINE_HASHES.items():
        name = prefix + key
        relative = Path("src") / (name.replace(".", "/") + ".py")
        source = (
            (Path(baseline_dir) / relative.name).read_text()
            if baseline_dir
            else subprocess.check_output(
                ["git", "show", f"{BASELINE_REVISION}:{relative}"],
                cwd=ROOT,
                text=True,
            )
        )
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Frozen baseline fingerprint mismatch: {key}")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in private:
                node.module = private[node.module]
        module = types.ModuleType(private[name])
        module.__file__ = f"{BASELINE_REVISION}:{relative}"
        module.__package__ = name.rpartition(".")[0]
        sys.modules[private[name]] = module
        exec(compile(tree, module.__file__, "exec"), vars(module))
        frozen[key] = module
        fingerprints[key] = {
            "baseline": digest,
            "candidate": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
    candidate = importlib.import_module(prefix + "experimental.decode_preparation")
    cache = importlib.import_module(prefix + "deepseek_v41.csa2.cache")
    writer = importlib.import_module(prefix + "mla.paged_swa")
    return (
        frozen["experimental.decode_preparation"],
        candidate,
        cache,
        frozen["mla.paged_swa"],
        writer,
        fingerprints,
    )


def fixture(tokens, requests, bank_count, case):
    rng = np.random.default_rng(1000 + tokens + requests)
    lengths = np.full(requests, tokens // requests, np.int32)
    if case == "ragged":
        lengths = (np.arange(requests, dtype=np.int32) * 5 + 1) % (
            2 * (tokens // requests)
        )
    elif case == "empty":
        lengths[:] = 0
    elif case == "padded":
        lengths[::4] = 0
        lengths[1::4] = 1
    assert lengths.sum() <= tokens
    starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    positions = np.zeros(tokens, np.int32)
    for r, n in enumerate(lengths):
        # Both parities and page crossings, with bounded contiguous requests.
        positions[starts[r] : starts[r + 1]] = 126 + r % 5 + np.arange(n)
    total_pages = requests * 8 + 2
    pages_per_bank = total_pages // bank_count
    table = np.arange(1, requests * 8 + 1, dtype=np.int32).reshape(requests, 8)
    if case in ("ragged", "padded"):
        # Null, out-of-range, and the first page in bank two (NOT a null page).
        table[1, :4] = 0
        table[2, :4] = total_pages + 3
        if bank_count == 2:
            # Swap instead of aliasing another request's physical page.
            old = int(table[3, 2])
            table[table == pages_per_bank] = old
            table[3, 2] = pages_per_bank
    banks = tuple(
        rng.normal(size=(pages_per_bank, 64, 512)).astype(ml_dtypes.bfloat16)
        for _ in range(bank_count)
    )
    values = rng.normal(size=(tokens, 512)).astype(ml_dtypes.bfloat16)
    minimum = (np.arange(requests, dtype=np.int32) * 3) % 129
    return banks, table, starts, positions, minimum, values


def reference(banks, table, starts, positions, minimum, values):
    """Scalar independent metadata/address contract plus full owner updates."""
    columns = (positions.size + 128 + 63) // 64 + 1
    metadata = np.zeros((table.shape[0], columns + 2), np.int32)
    bounds = np.zeros((table.shape[0], 2), np.int32)
    records = np.full(positions.size, -1, np.int32)
    owners = [b.copy() for b in banks]
    capacities = [b.shape[0] * 64 for b in banks]
    seen = set()
    for r in range(table.shape[0]):
        a, b = map(int, starts[r : r + 2])
        first = int(positions[min(a, positions.size - 1)])
        left, right = (max(first - 127, 0), first + b - a) if b > a else (0, 0)
        bounds[r] = left, right
        metadata[r, :2] = left // 64 * 64, minimum[r]
        for c in range(columns):
            page = left // 64 + c
            if b > a and page < table.shape[1] and page * 64 < right:
                metadata[r, c + 2] = table[r, page]
        for t in range(a, b):
            p = int(positions[t])
            rec = int(table[r, min(p // 64, table.shape[1] - 1)]) * 64 + p % 64
            records[t] = rec
            if 64 <= rec < sum(capacities):
                assert rec not in seen, f"Duplicate valid record: {rec}"
                seen.add(rec)
                base = 0
                for owner, capacity in zip(owners, capacities, strict=True):
                    if base <= rec < base + capacity:
                        owner.reshape(-1, 512)[rec - base] = values[t]
                    base += capacity
    return (*owners, metadata, bounds, records)


def check(expected, actual):
    assert len(expected) == len(actual)
    for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
        b = np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, (i, a.shape, b.shape)
        # Bitwise BF16 comparison also checks untouched owner rows and halves.
        np.testing.assert_array_equal(
            a.view(np.uint16) if a.dtype == ml_dtypes.bfloat16 else a,
            b.view(np.uint16) if b.dtype == ml_dtypes.bfloat16 else b,
            err_msg=f"output {i}",
        )


def cpu_check(xs, expected, baseline, cache):
    banks, table, starts, positions, minimum, values = xs
    args = tuple(jnp.asarray(x) for x in (table, starts, positions, minimum))
    check(
        expected[-3:],
        cache.prepare_swa_metadata(*args, page_size=64, page_stride=64, row_offset=0),
    )
    matrix = baseline.metadata_matrix(
        args[0], args[1][None], args[2][None], args[3][:, None]
    )
    check(expected[-3:], (*matrix[:2], matrix[2].reshape(-1)))
    owners, base = [], 0
    for bank in banks:
        capacity = bank.shape[0] * 64
        packed = baseline.pack_matrix(
            jnp.asarray(expected[-1])[None],
            jnp.asarray(values),
            base=base,
            capacity=capacity,
            padded=(values.shape[0] + 127) // 128 * 128,
        )
        dst, masks, words = map(np.asarray, packed)
        bits = bank.copy().view(np.uint16).reshape(-1, 512)
        for row, mask, word in zip(dst.reshape(-1), masks, words, strict=True):
            if row < 0:
                continue
            word = word.view(np.uint32)
            if mask[0] & 65535:
                bits[row * 2] = (word & 65535).astype(np.uint16)
            if int(mask[0]) & 0xFFFF0000:
                bits[row * 2 + 1] = (word >> 16).astype(np.uint16)
        owners.append(bits.view(ml_dtypes.bfloat16).reshape(bank.shape))
        base += capacity
    check(expected[: len(banks)], owners)


def run_pair(
    functions,
    xs,
    expected,
    rounds,
    warmup,
    label,
    profile_dir,
    modules,
    out=None,
    profile_samples=16,
):
    """Compile first; maintain independent donated owners; time sync ABBA."""
    executables, states = {}, {}
    for name, fn in functions.items():
        args = jax.tree.map(lambda x: jnp.asarray(x.copy()), xs)
        executable = jax.jit(fn, donate_argnums=(0,)).lower(*args).compile()
        if modules or out:
            text = executable.as_text()
            if out:
                (out / f"{label}_{name}.hlo").write_text(text)
            print(
                json.dumps(
                    {
                        "module": label + "/" + name,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "lines": len(text.splitlines()),
                        "custom_call_lines": [
                            line.strip().split(", backend_config=", 1)[0][:512]
                            for line in text.splitlines()
                            if "custom-call(" in line
                        ],
                    }
                )
            )
        output = jax.block_until_ready(executable(*args))
        check(expected, output)
        states[name] = (tuple(output[: len(xs[0])]), *args[1:])
        executables[name] = executable

    def step(name):
        start = time.perf_counter_ns()
        out = jax.block_until_ready(executables[name](*states[name]))
        elapsed = (time.perf_counter_ns() - start) / 1000
        states[name] = (tuple(out[: len(xs[0])]), *states[name][1:])
        return elapsed

    for _ in range(warmup):
        for name in ("baseline", "candidate"):
            step(name)
    samples = {name: [] for name in functions}
    for _ in range(rounds):
        for name in ("baseline", "candidate", "candidate", "baseline"):
            samples[name].append(step(name))
    result = {
        "case": label,
        "samples": 2 * rounds,
        "timing": "host_sync_ABBA_us",
        **{
            name: {
                "median_us": float(np.median(v)),
                "p10_us": float(np.percentile(v, 10)),
                "p90_us": float(np.percentile(v, 90)),
            }
            for name, v in samples.items()
        },
    }
    result["speedup"] = (
        result["baseline"]["median_us"] / result["candidate"]["median_us"]
    )
    print(json.dumps(result), flush=True)
    if out:
        (out / f"{label}.json").write_text(
            json.dumps({**result, "samples_us": samples}, indent=2) + "\n"
        )
    if profile_dir:
        for name in functions:
            directory = str(Path(profile_dir) / label / name)
            with jax.profiler.trace(directory):
                for _ in range(profile_samples):
                    step(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-reference", action="store_true")
    parser.add_argument("--cached-write", action="store_true")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=[f"{t}/{r}" for t, r in SHAPES],
        default=[f"{t}/{r}" for t, r in SHAPES],
    )
    parser.add_argument("--banks", nargs="+", type=int, choices=(1, 2), default=(1, 2))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-samples", type=int, default=16)
    parser.add_argument(
        "--out", type=Path, help="Compact JSON timings and HLO directory"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Directory with git-show original modules (no git needed)",
    )
    parser.add_argument("--module-summary", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 0:
        parser.error("rounds must be positive and warmup nonnegative")
    if args.profile_samples < 1:
        parser.error("profile-samples must be positive")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    baseline, candidate, cache, original_writer, writer, digest = load_modules(
        args.baseline_dir
    )
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported; standalone benchmark isolation failed")
    if not args.cpu_reference and jax.default_backend() != "tpu":
        parser.error(
            "Device benchmark requires TPU with SparseCore; use --cpu-reference"
        )
    print(
        json.dumps(
            {
                "source_sha256": digest,
                "jax": jax.__version__,
                "devices": list(map(str, jax.devices())),
                "scope": "page64_stride64_offset0_BF16x512_causal128",
                "candidate": "actual production preparation and writer dispatch",
                "baseline": BASELINE_REVISION,
                "profile_samples": args.profile_samples,
                "cpu_reference": args.cpu_reference,
            }
        ),
        flush=True,
    )
    if args.out:
        (args.out / "sources.json").write_text(
            json.dumps(
                {
                    "baseline_revision": BASELINE_REVISION,
                    "sha256": digest,
                    "jax": jax.__version__,
                    "devices": list(map(str, jax.devices())),
                    "profile_samples": args.profile_samples,
                },
                indent=2,
            )
            + "\n"
        )
    for shape in args.shapes:
        tokens, requests = map(int, shape.split("/"))
        for banks in args.banks:
            for case in args.cases:
                label = f"T{tokens}_R{requests}_banks{banks}_{case}"
                xs = fixture(tokens, requests, banks, case)
                expected = reference(*xs)
                if args.cpu_reference:
                    cpu_check(xs, expected, baseline, cache)
                    cpu_check(xs, expected, candidate, cache)
                    if args.out:
                        (args.out / f"{label}.json").write_text(
                            json.dumps({"case": label, "cpu_contract": "pass"}) + "\n"
                        )
                    print(
                        json.dumps({"case": label, "cpu_contract": "pass"}), flush=True
                    )
                    continue
                run_pair(
                    {
                        "baseline": baseline.swa_prepare_publish,
                        "candidate": candidate.swa_prepare_publish,
                    },
                    xs,
                    expected,
                    args.rounds,
                    args.warmup,
                    label + "_full",
                    args.profile_dir,
                    args.module_summary,
                    args.out,
                    args.profile_samples,
                )
                if args.cached_write:

                    def cached(owners, records, values, *, module):
                        pages = owners if len(owners) == 2 else owners[0]
                        updated = module.write_swa_pages(pages, records, values)
                        return updated if isinstance(updated, tuple) else (updated,)

                    run_pair(
                        {
                            "baseline": lambda *a: cached(*a, module=original_writer),
                            "candidate": lambda *a: cached(*a, module=writer),
                        },
                        (xs[0], expected[-1], xs[-1]),
                        expected[:banks],
                        args.rounds,
                        args.warmup,
                        label + "_cached_write",
                        args.profile_dir,
                        args.module_summary,
                        args.out,
                        args.profile_samples,
                    )


if __name__ == "__main__":
    main()
