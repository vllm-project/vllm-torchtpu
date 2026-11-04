# SPDX-License-Identifier: Apache-2.0
"""CSA integrated A/B: frozen old dispatcher versus live production dispatcher.

TPU: --out /tmp/main-records --repeat 100 --profile
CPU fixture/reference smoke only: --out /tmp/main-records-cpu --reference-only
T288/R48 is intentionally baseline=candidate, not a claimed optimization.
Full signed INT32 page limbs use separate page_size=1 correctness checks; timed
page128/page64 fixtures keep all physical record arithmetic in INT32 range.
"""

import argparse
import functools
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
package = types.ModuleType("vllm_torchtpu")
package.__path__ = [str(ROOT / "src/vllm_torchtpu")]
sys.modules.setdefault("vllm_torchtpu", package)
from _main_records_baseline import (  # noqa: E402
    prepare_main_records as baseline_prepare_main_records,
)

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import (  # noqa: E402
    prepare_main_records,
)

SHAPES = ((96, 16), (192, 32), (288, 32), (288, 48))
GEOMETRIES = ((386, 128), (772, 64))
FIXTURES = ("regular", "padding", "ragged", "empty")


def reference(selected, pages, starts, *, page_size: int):
    """Independent per-request NumPy oracle; never silently wrap record IDs."""
    records = np.full(selected.shape, -1, np.int32)
    counts = np.zeros(selected.shape[0], np.int32)
    assert starts[0] == 0 and np.all(np.diff(starts) >= 0)
    assert 0 <= starts[-1] <= selected.shape[0]
    for request, (begin, end) in enumerate(zip(starts[:-1], starts[1:], strict=True)):
        for token in range(int(begin), int(end)):
            row = selected[token]
            count = int(np.count_nonzero(row >= 0))
            assert np.all(row[:count] >= 0) and np.all(row[count:] < 0)
            logical = row[:count].astype(np.int64)
            assert np.all(logical < pages.shape[1] * page_size)
            physical = pages[request, logical // page_size].astype(np.int64)
            values = physical * page_size + logical % page_size
            assert np.all(values >= np.iinfo(np.int32).min)
            assert np.all(values <= np.iinfo(np.int32).max)
            records[token, :count] = values.astype(np.int32)
            counts[token] = count
    return records, counts


def fixture(
    tokens: int,
    requests: int,
    columns: int,
    page_size: int,
    kind: str,
    seed: int,
    *,
    full_limbs: bool = False,
):
    rng = np.random.default_rng(seed)
    lengths = np.full(requests, tokens // requests, np.int32)
    lengths[-1] += tokens - int(lengths.sum())
    if kind == "padding":
        lengths[-3:] = 0
    elif kind == "ragged":
        lengths = rng.multinomial(tokens - 13, np.ones(requests) / requests)
        lengths[1] += lengths[0]
        lengths[0] = 0
        lengths[-2] += lengths[-1]
        lengths[-1] = 0
    elif kind == "empty":
        lengths[:] = 0
    starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    maximum_page = np.iinfo(np.int32).max // page_size
    pages = rng.integers(0, maximum_page + 1, (requests, columns), dtype=np.int32)
    if full_limbs:
        assert page_size == 1
        pages = rng.integers(-(2**31), 2**31, pages.shape, dtype=np.int32)
        special = np.array(
            [
                0,
                -1,
                -(2**31),
                2**31 - 1,
                0x01020304,
                -0x1020304,
                255,
                256,
                65535,
                65536,
                16777215,
                16777217,
            ],
            dtype=np.int32,
        )
    else:
        special = np.array([0, 1, 255, 256, 65535, 65539, maximum_page], np.int32)
    pages[:, : special.size] = special
    selected = rng.integers(0, columns * page_size, (tokens, 512), dtype=np.int32)
    counts = rng.integers(0, 513, tokens)
    counts[:4] = [512, 0, 1, 400]
    # Force reads of every special page, byte/chunk boundaries, and the last page.
    boundary = np.unique(
        np.r_[np.arange(special.size), 127, 128, 255, 256, columns - 1]
    )
    selected[:, : boundary.size] = boundary * page_size + page_size - 1
    counts[::3] = 512
    selected[np.arange(512)[None, :] >= counts[:, None]] = -1
    return selected, pages, starts


def assert_equal(expected, actual) -> None:
    for want, got in zip(expected, actual, strict=True):
        got = np.asarray(got)
        assert got.dtype == np.int32 and got.shape == want.shape
        np.testing.assert_array_equal(got, want)


def pallas_bodies(value) -> list[str]:
    """Inspect nested production JITs, rather than assuming a guard was hit."""
    found = []
    if hasattr(value, "jaxpr") and not hasattr(value, "eqns"):
        return pallas_bodies(value.jaxpr)
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            if equation.primitive.name == "pallas_call":
                found.append(str(equation.params["jaxpr"]))
            else:
                found.extend(pallas_bodies(equation.params))
    elif isinstance(value, dict):
        for child in value.values():
            found.extend(pallas_bodies(child))
    elif isinstance(value, (tuple, list)):
        for child in value:
            found.extend(pallas_bodies(child))
    return found


def compile_pair(inputs, page_size: int, directory: Path, control: bool):
    directory.mkdir(parents=True, exist_ok=True)
    executables, graphs = {}, {}
    expanded = (inputs[0].shape[0], inputs[1].shape[0]) in SHAPES[:3] and (
        inputs[1].shape[1],
        page_size,
    ) in GEOMETRIES
    for side, fn in (
        ("baseline", baseline_prepare_main_records),
        ("candidate", prepare_main_records),
    ):
        call = functools.partial(fn, page_size=page_size)
        graph = jax.make_jaxpr(call)(*inputs)
        graphs[side] = graph
        (directory / f"{side}.jaxpr").write_text(str(graph))
        lowered = jax.jit(call).lower(*inputs)
        (directory / f"{side}.stablehlo").write_text(lowered.as_text())
        executable = lowered.compile()
        hlo = executable.as_text()
        (directory / f"{side}.hlo").write_text(hlo)
        matrix = control or (side == "candidate" and expanded)
        assert ("dsv41_main_records_matrix" in str(graph)) == matrix
        assert ("gather_page_ids" in str(graph)) == (not matrix)
        executables[side] = executable
    if control or not expanded:
        baseline = pallas_bodies(graphs["baseline"])
        proposed = pallas_bodies(graphs["candidate"])
        assert len(baseline) == len(proposed) == 1
        assert baseline == proposed, "Unchanged paths must have identical Pallas bodies"
    else:
        assert "dsv41_main_records_matrix" not in str(graphs["baseline"])
    return executables


def device_summary(directory: Path) -> list[dict]:
    """Read device module events only after the isolated capture has stopped."""
    from jax.profiler import ProfileData

    groups = {}
    for path in directory.rglob("*.xplane.pb"):
        for plane in ProfileData.from_file(str(path)).planes:
            if not plane.name.startswith("/device:TPU:"):
                continue
            for line in plane.lines:
                if line.name != "XLA Modules":
                    continue
                for event in line.events:
                    key = (plane.name, event.name)
                    groups.setdefault(key, []).append(event.duration_ns / 1000)
    rows = [
        dict(
            device=device,
            module=name,
            samples=len(values),
            median_us=float(np.median(values)),
            min_us=float(np.min(values)),
            max_us=float(np.max(values)),
        )
        for (device, name), values in groups.items()
    ]
    print("DEVICE_MODULE_SUMMARY", directory, json.dumps(rows), flush=True)
    if not rows:
        print(
            "WARNING: no TPU XLA Modules events; device latency unobservable",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=100, help="ABBA blocks per shape")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-repeat", type=int, default=20)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.profile_repeat < 1:
        parser.error("repeat/profile-repeat must be positive; warmup nonnegative")
    if not args.reference_only and jax.default_backend() != "tpu":
        parser.error(
            "TPU required for actual production gather and candidate; "
            "use --reference-only for CPU smoke"
        )
    if args.reference_only and args.profile:
        parser.error("--reference-only does not capture profiles")
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [
        Path(__file__),
        Path(__file__).with_name("_main_records_baseline.py"),
        ROOT / "src/vllm_torchtpu/kernels/deepseek_v41/csa2/cache.py",
        ROOT / "src/vllm_torchtpu/kernels/experimental/decode_preparation.py",
        ROOT / "src/vllm_torchtpu/kernels/deepseek_v4/core_attention/sparse_mla.py",
    ]
    report = dict(
        jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        seed=args.seed,
        reference_only=args.reference_only,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        timing="synchronized host dispatch+device, ABBA, no profiler",
        cases={},
    )
    for tokens, requests in SHAPES:
        for columns, page_size in GEOMETRIES:
            name = f"T{tokens}-R{requests}-P{columns}-page{page_size}"
            control = (tokens, requests) == (288, 48)
            row = dict(control=control, checks=[], timing={})
            executables = None
            for size, kinds in ((page_size, FIXTURES), (1, ("regular", "ragged"))):
                for kind in kinds:
                    host = fixture(
                        tokens,
                        requests,
                        columns,
                        size,
                        kind,
                        args.seed,
                        full_limbs=size == 1,
                    )
                    expected = reference(*host, page_size=size)
                    inputs = jax.device_put(host, jax.devices()[0])
                    if args.reference_only:
                        for fn in (baseline_prepare_main_records, prepare_main_records):
                            assert_equal(expected, fn(*inputs, page_size=size))
                    else:
                        if kind == kinds[0]:
                            compiled = compile_pair(
                                inputs, size, args.out / name / f"page{size}", control
                            )
                            if size == page_size:
                                executables = compiled
                        for executable in compiled.values():
                            assert_equal(
                                expected, jax.block_until_ready(executable(*inputs))
                            )
                    row["checks"].append(
                        dict(page_size=size, fixture=kind, full_limbs=size == 1)
                    )
            if executables is not None:
                host = fixture(
                    tokens, requests, columns, page_size, "regular", args.seed
                )
                inputs = jax.block_until_ready(jax.device_put(host, jax.devices()[0]))
                for executable in executables.values():
                    for _ in range(args.warmup):
                        jax.block_until_ready(executable(*inputs))
                samples = {side: [] for side in executables}
                for _ in range(args.repeat):
                    for side in ("baseline", "candidate", "candidate", "baseline"):
                        start = time.perf_counter_ns()
                        jax.block_until_ready(executables[side](*inputs))
                        samples[side].append((time.perf_counter_ns() - start) / 1000)
                row["timing"] = {
                    side: dict(host_us=values, median_us=float(np.median(values)))
                    for side, values in samples.items()
                }
                if args.profile:
                    row["profiles"] = {}
                    for side, executable in executables.items():
                        directory = args.out / name / f"profile-{side}-{time.time_ns()}"
                        with jax.profiler.trace(str(directory)):
                            for _ in range(args.profile_repeat):
                                jax.block_until_ready(executable(*inputs))
                        row["profiles"][side] = device_summary(directory)
            report["cases"][name] = row
            (args.out / "results.json").write_text(json.dumps(report, indent=2))
            print(
                name,
                "CPU reference smoke (no TPU path assertion)"
                if args.reference_only
                else "same-kernel control"
                if control
                else "generic production baseline",
                "checks=",
                len(row["checks"]),
                {s: t["median_us"] for s, t in row["timing"].items()},
                flush=True,
            )
    assert "torch" not in sys.modules, "standalone probe unexpectedly imported Torch"


if __name__ == "__main__":
    main()
