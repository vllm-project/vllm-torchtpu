# SPDX-License-Identifier: Apache-2.0
"""Current production Torch bridge versus private state-packed shape widening.

No historical Torch-native metadata baseline. No production monkeypatches.
"""

import argparse
import ast
import copy
import hashlib
import json
import time
import types
from pathlib import Path

import numpy as np
import state_packed_consumer as fixtures
import state_packed_shapes as metadata

TOKENS = (96, 192, 240, 256, 288)


ORIGINAL_COMMIT = "f8ced378ae65af5a5c16c77695d0ad40795b80bf"
ORIGINAL = {
    metadata.PREP: (
        "state_packed_original_decode.py",
        "45196d72d3b617228b7cb71709d6f692963564b7eb49e7b426eec1723e1840a0",
    ),
    metadata.DISPATCH: (
        "state_packed_original_preprocess.py",
        "313c79a8f20eef7c9730c98dd316fe64c7c5ba90232dc78d4d82c2f03bd1e6be",
    ),
}


def sources():
    frozen = {}
    for path, (companion, digest) in ORIGINAL.items():
        data = Path(__file__).with_name(companion).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, companion
        frozen[path] = data.decode()
    return frozen


def trees(frozen, candidate):
    decode = ast.parse(frozen[metadata.PREP])
    preprocess = ast.parse(frozen[metadata.DISPATCH])
    entry = next(
        n
        for n in preprocess.body
        if isinstance(n, ast.FunctionDef) and n.name == "compress_rope_packed"
    )
    build = next(
        n for n in entry.body if isinstance(n, ast.FunctionDef) and n.name == "build"
    )
    run = next(
        n for n in build.body if isinstance(n, ast.FunctionDef) and n.name == "run"
    )
    gate = run.body[0]
    assert isinstance(gate, ast.If)
    assert ast.unparse(gate.test) == "positions.shape[0] != 288"
    assert ast.unparse(gate.body[0].value.func) == "_compress_native_metadata"
    if candidate:
        run.body.pop(0)
        packed = next(
            n
            for n in decode.body
            if isinstance(n, ast.FunctionDef) and n.name == "compressor_state_packed"
        )
        shapes = next(
            n
            for n in packed.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "shapes" for t in n.targets)
        )
        assert ast.literal_eval(shapes.value) == ((48, 8), (49,), (49,), (288,), (48,))
        shapes.value.elts[3] = ast.parse("(positions.shape[0],)", mode="eval").body
    return ast.fix_missing_locations(decode), ast.fix_missing_locations(preprocess)


def cpu_test(frozen):
    from state_packed_guards import cpu_test as production_cpu_test

    baseline = trees(frozen, False)
    candidate = trees(frozen, True)
    # Restore exactly the two allowed edits, then require whole-module identity.
    restored = copy.deepcopy(candidate)

    def run(tree):
        entry = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "compress_rope_packed"
        )
        build = next(
            n
            for n in entry.body
            if isinstance(n, ast.FunctionDef) and n.name == "build"
        )
        return next(
            n for n in build.body if isinstance(n, ast.FunctionDef) and n.name == "run"
        )

    run(restored[1]).body.insert(0, copy.deepcopy(run(baseline[1]).body[0]))
    packed = next(
        n
        for n in restored[0].body
        if isinstance(n, ast.FunctionDef) and n.name == "compressor_state_packed"
    )
    shapes = next(
        n
        for n in packed.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "shapes" for t in n.targets)
    )
    shapes.value.elts[3] = ast.parse("(288,)", mode="eval").body
    assert all(
        ast.dump(a) == ast.dump(b) for a, b in zip(baseline, restored, strict=True)
    )
    # T288 executes precisely the same run body, with only a widened shape check.
    assert ast.dump(ast.Module(run(baseline[1]).body[1:], [])) == ast.dump(
        ast.Module(run(candidate[1]).body, [])
    )
    return {
        "two_edits_only": True,
        "t288_identical_path_ast": True,
        "fixtures": production_cpu_test(frozen),
    }


def private_ops(mesh, frozen, use_production=False):
    from attention_integration import production_ops

    production = production_ops(mesh)
    original_decode = production.decode_preparation
    original_entry = production.compress_rope_packed
    modules = []
    for label, candidate in (("baseline", False), ("candidate", True)):
        if candidate and use_production:
            modules.append(production)
            continue
        dtree, ptree = trees(frozen, candidate)
        decode = types.ModuleType("state_packed_bridge_" + label + "_decode")
        exec(compile(dtree, str(metadata.PREP), "exec"), decode.__dict__)
        # Bind a private decode module, never mutate production imports/globals.
        ptree.body = [
            n
            for n in ptree.body
            if not (
                isinstance(n, ast.ImportFrom)
                and n.module == "vllm_torchtpu.kernels.experimental"
                and any(a.name == "decode_preparation" for a in n.names)
            )
        ]
        module = types.ModuleType("state_packed_bridge_" + label)
        module.decode_preparation = decode
        exec(compile(ptree, str(metadata.DISPATCH), "exec"), module.__dict__)
        # Separate custom-op names/cache; preserve the production bridge builder.
        original_get = module._get
        module._get = lambda name, build, donate=(), get=original_get, tag=label: get(
            "bench_state_packed_" + tag + "_" + name, build, donate
        )
        modules.append(module)
    assert production.decode_preparation is original_decode
    assert production.compress_rope_packed is original_entry
    assert modules[0].decode_preparation is not modules[1].decode_preparation
    return modules


def execute(args, frozen, result):
    import importlib.metadata

    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import torch
    from attention_consumer import dense_reference
    from torch_tpu import _loader
    from torch_tpu._internal import sync

    # Invoke TorchTPU's torch.backends entry point even when autoload is disabled.
    _loader.load()
    torch.set_num_threads(1)
    # Five static shapes times two isolated ops share the wrapper code object.
    # Admit this bounded benchmark matrix without weakening fullgraph capture.
    torch._dynamo.config.recompile_limit = 16
    torch.empty(1, device="tpu").cpu()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    modules = private_ops(mesh, frozen, args.production)

    def owned_bridge(module):
        # Keep the exact production-op + copy_ skeleton, but export the updated
        # owners as graph outputs. Never resynchronize or reuse a pre-donation
        # Python owner: the backend may have consumed its underlying PJRT buffer.
        op = module.compress_rope_packed(1e-6)
        assert op is not None

        def run(
            table, starts, kv_starts, positions, minimum, x, weight, gamma, rope, a, b
        ):
            latent, words, scales, low, high, valid = op(
                table,
                starts,
                kv_starts,
                positions,
                minimum,
                x,
                weight,
                gamma,
                rope,
                a,
                b,
            )
            a.copy_(low)
            b.copy_(high)
            return latent, words, scales, valid, a, b

        return run

    functions = {
        label: torch.compile(
            owned_bridge(module), backend="tpu", fullgraph=True, dynamic=False
        )
        for label, module in zip(("baseline", "candidate"), modules, strict=True)
    }
    result["ownership"] = (
        "Fresh materialized host uploads per correctness call; no seed copy_ "
        "aliases. Updated owners explicitly returned by compiled graph and "
        "chained for timing/profiling; synchronize returned tensors only. "
        "Production donation (9, 10) and consumer/bank copy_ unchanged."
    )
    result["runtime"] = {
        n: importlib.metadata.version(n)
        for n in ("torch", "torch-tpu", "jax", "jaxlib", "libtpu", "numpy")
    }
    result["devices"] = [str(d) + ":" + d.device_kind for d in jax.devices()]
    dense = dense_reference()
    rng = np.random.default_rng(20261104)

    def host(t):
        return (
            t.detach().cpu().float().numpy()
            if t.dtype == torch.bfloat16
            else t.detach().cpu().numpy()
        )

    def unpack(words):
        planes = np.stack(
            [(words.astype(np.uint32) >> s) & 255 for s in (0, 8, 16, 24)], axis=1
        )
        return (
            planes.astype(np.uint8)
            .view(ml_dtypes.float8_e4m3fn)
            .astype(np.float32)
            .reshape(-1, 512)
        )

    for tokens in TOKENS:
        x = torch.tensor(rng.normal(0, 0.5, (tokens, 5120)), dtype=torch.bfloat16)
        weight = torch.tensor(rng.normal(0, 0.02, (5120, 1024)), dtype=torch.bfloat16)
        gamma = torch.tensor(rng.normal(size=512), dtype=torch.bfloat16)
        angle = rng.uniform(-3, 3, (tokens, 32)).astype(np.float32)
        rope = np.concatenate((np.cos(angle), np.sin(angle)), axis=1)
        compute = tuple(t.to("tpu") for t in (x, weight, gamma, torch.from_numpy(rope)))
        initial = rng.normal(0, 0.1, (fixtures.RECORDS, 1024)).astype(np.float32)
        owners = initial.view(np.int32).reshape(2, fixtures.BANK_RECORDS * 8, 128)
        banks = {}

        def reset(label, banks=banks, owners=owners):
            # A device copy_ of a reusable seed can alias that seed's PJRT
            # storage. Donation then invalidates the seed too. Allocate via
            # independent host uploads instead, and materialize before donation.
            banks[label] = tuple(torch.from_numpy(a.copy()).to("tpu") for a in owners)
            sync.synchronize(tensors=list(banks[label]), wait=True)

        for kind in metadata.CASES:
            arrays = fixtures.fixture(tokens, kind)
            addresses, packed_expected = fixtures.reference_metadata(arrays)
            jarrays = tuple(jnp.asarray(a) for a in arrays)
            original_addresses = modules[
                0
            ].decode_preparation._compressor_state_addresses(*jarrays)
            metadata.check(original_addresses, addresses)
            metadata.check(
                modules[1].decode_preparation.compressor_state_packed(
                    *jarrays, total_records=fixtures.RECORDS
                ),
                packed_expected,
            )
            if tokens == 288:
                metadata.check(
                    modules[0].decode_preparation.compressor_state_packed(
                        *jarrays, total_records=fixtures.RECORDS
                    ),
                    packed_expected,
                )
            print(f"T{tokens}/{kind}: metadata exact; running both arms", flush=True)
            table, starts, kv_starts, pos, minimum = arrays
            operands = (
                tuple(
                    torch.from_numpy(a).to("tpu")
                    for a in (table, starts, kv_starts, pos, minimum)
                )
                + compute
            )
            prev, valid, write, read = addresses
            ring = initial[np.maximum(read, 0)].copy()
            ring[read < 0] = 0
            projection, latent, words, scales = jax.device_get(
                dense(
                    jnp.asarray(host(x), jnp.bfloat16),
                    jnp.asarray(host(weight), jnp.bfloat16),
                    jnp.asarray(host(gamma), jnp.bfloat16),
                    prev,
                    ring,
                    valid,
                    rope[:, :32],
                    rope[:, 32:],
                )
            )
            expected_bank = initial.copy()
            live = np.flatnonzero(write >= 0)
            expected_bank[write[live]] = projection[live]
            outputs = {}
            for label, fn in functions.items():
                print(f"T{tokens}/{kind}/{label}: execute", flush=True)
                reset(label)
                output = fn(*operands, *banks[label])
                banks[label] = output[-2:]
                sync.synchronize(tensors=list(output), wait=True)
                got = tuple(host(t) for t in output)
                outputs[label] = got
                print(f"T{tokens}/{kind}/{label}: complete", flush=True)
            # Persist original inputs and full results before any numerical
            # assertion, so a failure cannot hide which arm already disagrees.
            np.savez_compressed(
                args.out / f"t{tokens}-{kind}-correctness.npz",
                table=table,
                starts=starts,
                kv_starts=kv_starts,
                positions=pos,
                minimum=minimum,
                x=host(x),
                weight=host(weight),
                gamma=host(gamma),
                rope=rope,
                initial=initial,
                prev=prev,
                valid=valid,
                write=write,
                read=read,
                expected_latent=latent.astype(np.float32),
                expected_words=words,
                expected_scales=scales,
                expected_bank=expected_bank,
                **{
                    f"{arm}_{i}": value
                    for arm, values in outputs.items()
                    for i, value in enumerate(values)
                },
            )
            diagnostics = {}
            for label, got in outputs.items():
                delta = np.abs(got[0] - latent.astype(np.float32))
                diagnostics[label] = {
                    "max_latent_error": float(delta.max()),
                    "latent_bad_rows": np.flatnonzero(
                        np.any(
                            ~np.isclose(
                                got[0], latent.astype(np.float32), rtol=0.02, atol=0.02
                            ),
                            axis=1,
                        )
                    ).tolist(),
                    "valid_exact": bool(np.array_equal(got[3], valid != 0)),
                }
            print(f"T{tokens}/{kind}: {diagnostics}", flush=True)
            (args.out / f"t{tokens}-{kind}-diagnostics.json").write_text(
                json.dumps(diagnostics, indent=2)
            )
            metadata.check(outputs["baseline"], outputs["candidate"])
            print(f"T{tokens}/{kind}: full A/B exact; checking oracle", flush=True)
            for label, got in outputs.items():
                print(f"T{tokens}/{kind}/{label}: oracle", flush=True)
                np.testing.assert_allclose(
                    got[0], latent.astype(np.float32), rtol=0.02, atol=0.02
                )
                np.testing.assert_allclose(got[2][:, 0], scales, rtol=0.02, atol=2e-5)
                np.testing.assert_allclose(
                    unpack(got[1]) * got[2],
                    unpack(words) * scales[:, None],
                    rtol=0.15,
                    atol=0.02,
                )
                np.testing.assert_array_equal(got[3], valid != 0)
                actual_bank = np.stack(got[4:]).view(np.float32).reshape(initial.shape)
                np.testing.assert_allclose(
                    actual_bank, expected_bank, rtol=0.02, atol=0.002
                )
                untouched = np.ones(fixtures.RECORDS, bool)
                untouched[write[live]] = False
                np.testing.assert_array_equal(
                    actual_bank[untouched], initial[untouched]
                )
            metadata.check(outputs["baseline"], outputs["candidate"])
            result["correctness"].append(
                {
                    "tokens": tokens,
                    "case": kind,
                    "full_outputs_and_banks_exact": True,
                    "independent_oracle": True,
                    "t288_identical_path": tokens == 288,
                }
            )
        # Use the uniform fixture, not the last adversarial case, for timings.
        timing_arrays = fixtures.fixture(tokens, "uniform")
        table, starts, kv_starts, pos, minimum = timing_arrays
        (_, _, timing_write, timing_read), _ = fixtures.reference_metadata(
            timing_arrays
        )
        assert not np.intersect1d(
            timing_write[timing_write >= 0], timing_read[timing_read >= 0]
        ).size, "Chained timing requires no read-after-write across invocations"
        operands = (
            tuple(
                torch.from_numpy(a).to("tpu")
                for a in (table, starts, kv_starts, pos, minimum)
            )
            + compute
        )

        def run(label, operands=operands, banks=banks):
            output = functions[label](*operands, *banks[label])
            banks[label] = output[-2:]
            sync.synchronize(tensors=list(output), wait=True)

        # Each path owns its own state sequence; subsequent invocations consume
        # only the previous graph's returned owners, not stale donated inputs.
        for label in functions:
            reset(label)
        for _ in range(8):
            for label in functions:
                run(label)
        timings = {label: [] for label in functions}
        for _ in range(args.samples):
            for label in ("baseline", "candidate", "candidate", "baseline"):
                start = time.perf_counter_ns()
                run(label)
                timings[label].append((time.perf_counter_ns() - start) / 1000)
        result["timings"][str(tokens)] = {
            label: {
                "samples_us": values,
                "median_us": float(np.median(values)),
                "p10_us": float(np.percentile(values, 10)),
                "p90_us": float(np.percentile(values, 90)),
            }
            for label, values in timings.items()
        }
        if args.profile:
            # Isolated traces per T/path, sixteen synchronized region invocations.
            for label in functions:
                reset(label)
                with jax.profiler.trace(str(args.out / f"profile-t{tokens}-{label}")):
                    for step in range(16):
                        with jax.profiler.TraceAnnotation(
                            f"compressor_region_{label}", step_num=step
                        ):
                            run(label)
        (args.out / "result.json").write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Compare frozen original against actual production modules",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--samples", type=int, default=100, help="ABBA quartets per T")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    frozen = sources()
    if args.production:
        from state_packed_guards import cpu_test as production_cpu_test

        audit = production_cpu_test(frozen)
    else:
        audit = cpu_test(frozen)
    if args.cpu_test:
        print(json.dumps(audit, indent=2))
        return
    if args.out is None:
        parser.error("--out required for TPU execution")
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": (
            "compressor region: CURRENT production preprocess bridge + unchanged "
            "consumer + bank copy_; NOT E2E"
        ),
        "original_commit": ORIGINAL_COMMIT,
        "candidate_route": "actual production import"
        if args.production
        else "private AST",
        "candidate_source_sha256": {
            str(p.relative_to(metadata.ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (
                metadata.PREP,
                metadata.DISPATCH,
                metadata.COMPRESS,
                metadata.TUNING,
            )
        },
        "baseline_route": (
            "T!=288: current JAX _compress_native_metadata fallback; "
            "T288: current packed route"
        ),
        "geometry": {
            "B": 48,
            "hidden": 5120,
            "ratio": 2,
            "capacity": 16,
            "stride": 16,
            "offset": 0,
            "window": 128,
            "records": fixtures.RECORDS,
        },
        "audit": audit,
        "correctness": [],
        "timings": {},
        "source_sha256": {
            str(p.relative_to(metadata.ROOT)): hashlib.sha256(s.encode()).hexdigest()
            for p, s in frozen.items()
        },
    }
    execute(args, frozen, result)
    result["passed"] = True
    (args.out / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
