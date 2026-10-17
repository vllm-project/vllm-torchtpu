#!/usr/bin/env python3
"""Execute production Index-K TorchTPU writers; no weights or performance claims.

Imports of Torch/JAX are deliberately worker-local. The parent launches eight
real TorchTPU workers and only publishes passed after every receipt is checked.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

WORLD_SIZE = 8
EXACT_ROWS = 19_424_832
SENTINEL = 731


def case_specs():
    return [
        {"n": n, "layout": layout, "after": after, "rows": rows, "banks": banks}
        for n in (288, 2048)
        for layout in ("ordinary", "halfplane")
        for after in (False, True)
        for rows, banks in ((EXACT_ROWS, 1), (2048, 2))
    ]


def fixture(n, bank_records, banks):
    """Unique records: paired siblings, odd-only, even-only, invalid padding."""
    import numpy as np

    ids = np.arange(n, dtype=np.int64) * 2
    ids[:4] = [0, 1, 5, 6]
    if banks == 2:
        ids[n // 2 : -1] += bank_records
    ids[-2] = bank_records * banks - 1
    ids[-1] = -1
    valid = ids[ids >= 0]
    assert len(np.unique(valid)) == len(valid)
    values = ((np.arange(n * 128).reshape(n, 128) % 17) + 1).astype(np.float32)
    return ids, values


def expected_rows(ids, bits, layout, bank_records, banks):
    """Independent integer oracle, bounded to touched and neighboring rows."""
    import numpy as np

    result = []
    for bank in range(banks):
        local = ids - bank * bank_records
        valid = (ids >= 0) & (local >= 0) & (local < bank_records)
        rows = set((local[valid] // 2).tolist()) | {0, 2, bank_records // 2 - 1}
        sample = np.asarray(sorted(rows), dtype=np.int32)
        words = np.full((len(sample), 128), SENTINEL, dtype=np.uint32)
        positions = {int(row): i for i, row in enumerate(sample)}
        for record, value in zip(local[valid], bits[valid]):
            dst = words[positions[int(record // 2)]]
            if layout == "ordinary":
                offset = int(record % 2) * 64
                dst[offset : offset + 64] = value[::2] | (value[1::2] << 16)
            else:
                split = np.concatenate((value[::2], value[1::2]))
                shift = int(record % 2) * 16
                mask = np.uint32(0xFFFF0000 if shift == 0 else 0x0000FFFF)
                dst[:] = (dst & mask) | (split << shift)
        result.append((sample, words.view(np.int32)))
    return result


def run_case(spec):
    import numpy as np
    import torch
    from torch_tpu._internal import compile as tpu_compile
    from torch_tpu._internal import sync

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows

    n, rows, banks = spec["n"], spec["rows"], spec["banks"]
    ids, values = fixture(n, rows * 2, banks)
    host_keys = torch.from_numpy(values).to(torch.bfloat16)
    bits = host_keys.view(torch.int16).numpy().astype(np.uint32) & 65535
    oracle = expected_rows(ids, bits, spec["layout"], rows * 2, banks)
    slots, keys = torch.from_numpy(ids).to("tpu"), host_keys.to("tpu")
    assert slots.dtype == torch.int64, "gate must retain real model long slots"
    geometry = ((rows * 2 * banks // 128, 128, 128), torch.bfloat16, 32768, 0)
    results = []
    for mode in ("eager", "compiled"):
        owners = tuple(
            torch.full((rows, 128), SENTINEL, dtype=torch.int32, device="tpu")
            for _ in range(banks)
        )

        # Model cache binding happens before graph capture; reproduce that lifetime.
        view = PagedCacheRows(
            owners[0] if banks == 1 else owners,
            geometry,
            index_key_layout=spec["layout"],
        )

        def forward(input_slots, input_keys, bound_view=view):
            # Read actual owner data before mutation and consume its result.
            before = bound_view.raw[0, 0].float()
            bound_view.write(
                input_slots, input_keys, after=before if spec["after"] else None
            )
            return before

        run = (
            torch.compile(
                forward, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend()
            )
            if mode == "compiled"
            else forward
        )
        sync.synchronize([*owners, slots, keys], wait=True)
        expected_before = float(SENTINEL)
        for iteration in range(2):
            before = run(slots, keys)
            sync.synchronize([*owners, before], wait=True)
            assert before.cpu().item() == expected_before, "prior owner read reordered"
            for owner, (sample, expected) in zip(owners, oracle):
                index = torch.from_numpy(sample).long().to("tpu")
                actual = owner.index_select(0, index).cpu().numpy()
                np.testing.assert_array_equal(actual, expected)
            expected_before = float(oracle[0][1][0, 0].astype(np.float32))
            results.append({"mode": mode, "iteration": iteration, "status": "passed"})
        del run, forward, view, owners, owner, before, actual, index
        gc.collect()
    torch.compiler.reset()
    return {
        **spec,
        "status": "passed",
        "executions": results,
        "slot_dtype": "int64",
        "owner_scope": "exact_packet04_bank"
        if banks == 1
        else "small_two_bank_semantics",
    }


def worker(out):
    result = {"rank": int(os.environ["RANK"]), "status": "failed", "cases": []}
    try:
        import jax
        import numpy as np
        import torch
        import torch_tpu  # noqa: F401
        from torch import distributed as dist
        from torch_tpu._internal import pallas
        from torch_tpu._internal.distributed import tpu_distributed

        from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import initialize_cache_ops
        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        device_id = int(tpu_distributed.global_device_id())
        # Match TPUModelRunner._create_mesh_for_parallelism exactly: one
        # local device, axis model. Check it is the actual Torch worker chip.
        devices = list(jax.local_devices())[:1]
        assert len(devices) == 1 and int(devices[0].id) == device_id, (
            device_id,
            jax.local_devices(),
        )
        mesh = jax.sharding.Mesh(np.asarray(devices).reshape((1,)), ("model",))
        source = inspect.getsource(pallas)
        result.update(
            device_id=device_id,
            runtime={"torch": torch.__version__, "jax": jax.__version__},
            pallas_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        )
        (out / f"pallas-rank-{result['rank']}.py").write_text(source)
        with set_vllm_model_wrapper_context(mesh=mesh):
            initialize_cache_ops()
            for spec in case_specs():
                result["active_case"] = spec
                (out / f"rank-{result['rank']}.json").write_text(json.dumps(result))
                result["cases"].append(run_case(spec))
        result.pop("active_case", None)
        result["status"] = "passed"
    except BaseException:
        result["error"] = traceback.format_exc()
        print(
            f"writer gate rank={result['rank']} failed:\n{result['error']}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        (out / f"rank-{result['rank']}.json").write_text(json.dumps(result, indent=2))
        if "dist" in locals() and dist.is_initialized():
            dist.destroy_process_group()
    return 0 if result["status"] == "passed" else 1


def validate_results(results):
    """Require every rank, distinct chip, exact case, mode and repeat."""
    expected = {tuple(sorted(case.items())) for case in case_specs()}
    if len(results) != WORLD_SIZE:
        return False
    if {item.get("rank") for item in results} != set(range(WORLD_SIZE)):
        return False
    devices = [item.get("device_id") for item in results]
    if any(device is None for device in devices) or len(set(devices)) != WORLD_SIZE:
        return False
    executions = {
        (mode, iteration) for mode in ("eager", "compiled") for iteration in range(2)
    }
    for item in results:
        cases = item.get("cases", [])
        if item.get("status") != "passed" or len(cases) != len(expected):
            return False
        actual = []
        for case in cases:
            actual.append(
                tuple(sorted((key, case.get(key)) for key in case_specs()[0]))
            )
            calls = case.get("executions", [])
            if case.get("status") != "passed" or len(calls) != len(executions):
                return False
            if any(call.get("status") != "passed" for call in calls):
                return False
            if {
                (call.get("mode"), call.get("iteration")) for call in calls
            } != executions:
                return False
        if set(actual) != expected:
            return False
    return True


def launch(out, timeout):
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment,
    )
    from torch_tpu._internal.utils import hardware

    if hardware.get_tpu_device_count() != WORLD_SIZE:
        raise RuntimeError("writer gate requires exactly eight TPU devices")
    for key in (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    ):
        os.environ.pop(key, None)
    prepare_tpu_environment(world_size=WORLD_SIZE)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--out",
        str(out),
    ]
    with (out / "workers.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = -1
    results = []
    for rank in range(WORLD_SIZE):
        path = out / f"rank-{rank}.json"
        results.append(
            json.loads(path.read_text()) if path.exists() else {"rank": rank}
        )
    passed = code == 0 and validate_results(results)
    if not passed:
        error = next((item.get("error") for item in results if item.get("error")), None)
        if error is None:
            error = (
                f"worker exit={code}; incomplete or invalid writer receipts; "
                f"see {out / 'workers.log'}"
            )
        print(
            f"TorchTPU writer gate failed:\n{error[:12000]}",
            file=sys.stderr,
            flush=True,
        )
    receipt = {
        "status": "passed" if passed else "failed",
        "backend": "torch_tpu",
        "logical_devices": len({item.get("device_id") for item in results})
        if passed
        else 0,
        "fullgraph": True,
        "exact_owner_shape_executed": passed,
        "scope": "production PagedCacheRows immediate and after writers",
        "whole_model_compiled": False,
        "deferred_indexer_covered": False,
        "checks": sum(
            len(case.get("executions", []))
            for item in results
            for case in item.get("cases", [])
        ),
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worker_exit_code": code,
        "workers": results,
    }
    passed = passed and receipt["logical_devices"] == WORLD_SIZE
    receipt["status"] = "passed" if passed else "failed"
    (out / "smoke.json").write_text(json.dumps(receipt, indent=2))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path(os.environ.get("AB_SMOKE_OUT", "."))
    )
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    return worker(args.out) if args.worker else launch(args.out, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
