# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER on real collectives.

`test_dcp2_packed_gather.py` pins the packing against a threaded all-gather on
CPU; this runs `qsa_op._gathered_lists` on one host's 8 ranks (4 chips x 2
cores) over the groups the served decode uses: the main-KV DCP groups of 4
(0-3, 4-7, built like `qsa.main_kv_dcp_group`) and the TP group of 8 that the
sequence-parallel graph gathers over. Eager and under `torch.compile`, flag off
and on:

  1. the gathered score bits and group ids equal the lists taken from every
     rank's inputs (served-like lists and arbitrary bit patterns, rows past the
     tokens, padding rows with no candidate);
  2. the merged selections built on device from both gathers (global and this
     rank's local merge, with the SparseCore top-k) are byte-identical, and
     padding rows select nothing.

With PACKED_GATHER_BENCH=1 the workers also time one QSA layer's list gather
(two all-gathers, the packed one, and a single list's) at 96, 192 and 256 rows.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
import zlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

HERE = Path(__file__).resolve().parent
WORLD_SIZE = 8
SIZE = 4
K = 512
FLAG = "VLLM_TPU_QSA_COMPRESSED_DCP_PACKED_GATHER"
LAUNCH_TIMEOUT_SECONDS = 3600
# (name, group, rows a rank holds, tokens): the decode forwards at 98304:24/48/64,
# one with rows past the tokens, and the SP graph's two token shards.
CHECKS = (
    ("pair_96", "pair", 96, 96),
    ("pair_192", "pair", 192, 192),
    ("pair_256", "pair", 256, 256),
    ("pair_256_cut", "pair", 256, 250),
    ("tp_shards_256", "tp", 128, 256),
)
SELECTION_ROWS = (96, 256)
BENCH_ROWS = (96, 192, 256)
BENCH_ARMS = ("two", "packed", "one_list")


def _worker_command(result_dir: Path, bench: bool) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-dir",
        str(result_dir),
        *(["--bench"] if bench else []),
    ]


def _prepare_worker_env() -> dict[str, str]:
    """The engine's single-host bootstrap env for 8 ranks (what
    TpuPlatform._prepare_singlehost_tpu_env gives vLLM's workers). A bench
    pod's JOB_COMPLETION_INDEX and unresolved TPU_WORKER_HOSTNAMES would make
    torch_tpu treat the host as one of several, so they are dropped. Without
    devices the test skips, or fails under PACKED_GATHER_REQUIRE_TPU=1."""
    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "MASTER_ADDR",
        "MASTER_PORT",
        "JOB_COMPLETION_INDEX",
        "TPU_WORKER_HOSTNAMES",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

        TpuPlatform._prepare_singlehost_tpu_env(WORLD_SIZE)
        env = os.environ.copy()
    except Exception as exc:  # noqa: BLE001 - becomes the skip or failure reason
        reason = f"Eight TPU devices are unavailable: {exc!r}"
        if os.environ.get("PACKED_GATHER_REQUIRE_TPU") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for k in ("JOB_COMPLETION_INDEX", "TPU_WORKER_HOSTNAMES"):
        env.pop(k, None)
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    return env


# --------------------------------------------------------------------- worker
def _compile(fn):
    import torch
    from torch_tpu._internal import compile as tpu_compile

    torch._dynamo.reset()
    return torch.compile(fn, fullgraph=True, dynamic=False, backend=tpu_compile.TpuBackend())


def _set_flag(value: bool) -> None:
    from vllm_torchtpu import envs

    setattr(envs, FLAG, value)


def _group_ranks(kind, rank, world):
    if kind == "tp":
        return list(range(world))
    start = rank // SIZE * SIZE
    return list(range(start, start + SIZE))


def _expected(lists, ranks, tokens):
    import numpy as np

    shards = len(ranks) // SIZE
    vals = np.stack([np.concatenate([lists[ranks[s * SIZE + d]][0] for s in range(shards)])
                     for d in range(SIZE)])[:, :tokens]
    pos = np.stack([np.concatenate([lists[ranks[s * SIZE + d]][1] for s in range(shards)])
                    for d in range(SIZE)])[:, :tokens]
    return vals.view(np.int32), pos


def _bits(x):
    import torch

    return x.cpu().contiguous().view(torch.int32).numpy()


def _log(rank, msg):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _check(rank, world, groups):
    import numpy as np
    import torch
    import test_dcp2_packed_gather as cpu
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    report = {}
    for name, kind, rows, tokens in CHECKS:
        group = groups[kind]
        ranks = _group_ranks(kind, rank, world)
        inputs = {}
        for data in ("served", "bits"):
            lists, _ = cpu.make_inputs(data, world, SIZE, rows,
                                       seed=zlib.crc32(name.encode()) + rows)
            inputs[data] = (
                _expected(lists, ranks, tokens),
                torch.from_numpy(lists[rank][0].copy()).to("tpu"),
                torch.from_numpy(lists[rank][1].copy()).to("tpu"),
            )

        def gather(v, p, group=group, tokens=tokens):
            return qsa_op._gathered_lists(v, p, group, SIZE, tokens)

        got = {}
        for packed in (False, True):
            _set_flag(packed)
            arm = "packed" if packed else "two"
            for how in ("eager", "compiled"):
                try:
                    fn = gather if how == "eager" else _compile(gather)
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    traceback.print_exc()
                    for data in inputs:
                        report[f"{name}/{data}/{arm}/{how}"] = {"error": repr(exc)[:400]}
                    continue
                for data, ((want_vals, want_pos), vals, pos) in inputs.items():
                    key = f"{name}/{data}/{arm}/{how}"
                    try:
                        gv, gp = fn(vals, pos)
                        gv_bits, gp_np = _bits(gv), gp.cpu().numpy()
                        got[key] = (gv_bits, gp_np)
                        report[key] = {
                            "shape": list(gv.shape),
                            "dtypes": [str(gv.dtype), str(gp.dtype)],
                            "score_bits_mismatch": int((gv_bits != want_vals).sum())
                            if gv_bits.shape == want_vals.shape else -1,
                            "id_mismatch": int((gp_np != want_pos).sum())
                            if gp_np.shape == want_pos.shape else -1,
                        }
                    except Exception as exc:  # noqa: BLE001 - reported to the parent
                        traceback.print_exc()
                        report[key] = {"error": repr(exc)[:400]}
                    _log(rank, f"check {key}: {report[key]}")
        for data in inputs:
            for how in ("eager", "compiled"):
                a = got.get(f"{name}/{data}/two/{how}")
                b = got.get(f"{name}/{data}/packed/{how}")
                report[f"{name}/{data}/arms_equal/{how}"] = bool(
                    a is not None and b is not None
                    and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                )
    _set_flag(False)
    return report


def _selection_op():
    import jax
    from torch_tpu._internal import pallas

    def step(vals_pair: jax.Array, pos_pair: jax.Array,
             dcp_rank: jax.Array) -> tuple[jax.Array, jax.Array]:
        from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa

        return (pooled_qsa.compressed_dcp_blocks(vals_pair, pos_pair, K),
                pooled_qsa.compressed_dcp_own_blocks(vals_pair, pos_pair, dcp_rank, K))

    op = pallas.jax_op("qwen4_exp_test::packed_gather_selection", step)

    @op.register_fake
    def fake(vals_pair, pos_pair, dcp_rank):
        t = vals_pair.shape[1]
        return pos_pair.new_empty((t, K)), pos_pair.new_empty((t, K))

    return op


def _selection(rank, world, groups):
    """Gather and merge in one compiled program, flag off and on."""
    import numpy as np
    import torch
    import test_dcp2_packed_gather as cpu
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    op = _selection_op()
    group = groups["pair"]
    report = {}
    for rows in SELECTION_ROWS:
        lists, padding = cpu.make_inputs("served", world, SIZE, rows, seed=7 * rows)
        vals = torch.from_numpy(lists[rank][0].copy()).to("tpu")
        pos = torch.from_numpy(lists[rank][1].copy()).to("tpu")
        dcp_rank = torch.tensor([rank % SIZE], dtype=torch.int32).to("tpu")

        def merged(v, p, r, rows=rows):
            vp, pp = qsa_op._gathered_lists(v, p, group, SIZE, rows)
            return op(vp, pp, r)

        got = {}
        for packed in (False, True):
            _set_flag(packed)
            arm = "packed" if packed else "two"
            try:
                blocks, own = _compile(merged)(vals, pos, dcp_rank)
                got[arm] = (blocks.cpu().numpy(), own.cpu().numpy())
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[f"{rows}/{arm}"] = {"error": repr(exc)[:400]}
        if len(got) == 2:
            (b2, o2), (b1, o1) = got["two"], got["packed"]
            pad = sorted(padding)
            real = sorted(set(range(rows)) - padding)
            report[f"{rows}"] = entry = {
                "blocks_equal": bool(np.array_equal(b1, b2)),
                "own_equal": bool(np.array_equal(o1, o2)),
                "padding_rows_select_nothing": bool((b1[pad] == -1).all() and (o1[pad] == -1).all()),
                "real_rows_selected": int((b1[real] >= 0).sum()),
                "own_selected": int((o1 >= 0).sum()),
            }
            _log(rank, f"selection rows={rows}: {entry}")
    _set_flag(False)
    return report


def _bench(rank, world, groups, reps=12, samples=7):
    """Per-layer device time of the list gather: (wall of 1+reps chained
    layers - wall of 1) / reps. Each layer's lists are an elementwise function
    of the last layer's gathered lists, so nothing is loop invariant and the
    packing has separate producers, as after the select step."""
    import torch
    import test_dcp2_packed_gather as cpu
    from vllm_torchtpu.layers.adapter.custom_ops import qsa_op

    group = groups["pair"]
    out = {}

    def chain(n, rows, arm):
        def run(vals, pos):
            for _ in range(n):
                if arm == "one_list":
                    g = group.all_gather(vals.unsqueeze(0).contiguous(), dim=0)
                    vals = g[1] * 1.0001
                    continue
                vp, pp = qsa_op._gathered_lists(vals, pos, group, SIZE, rows)
                vals = vp[1] * 1.0001
                pos = pp[1] ^ 1
            return vals[:1, :1], pos[:1, :1]

        return run

    for rows in BENCH_ROWS:
        lists, _ = cpu.make_inputs("served", world, SIZE, rows, seed=11 * rows)
        vals = torch.from_numpy(lists[rank][0].copy()).to("tpu")
        pos = torch.from_numpy(lists[rank][1].copy()).to("tpu")
        for arm in BENCH_ARMS:
            _set_flag(arm == "packed")
            try:
                times = {}
                for n in (1, 1 + reps):
                    fn = _compile(chain(n, rows, arm))
                    for _ in range(2):
                        [x.cpu() for x in fn(vals, pos)]
                    walls = []
                    for _ in range(samples):
                        t0 = time.perf_counter()
                        [x.cpu() for x in fn(vals, pos)]
                        walls.append(time.perf_counter() - t0)
                    walls.sort()
                    times[n] = walls[len(walls) // 2]
                out[f"{arm}@{rows}"] = {
                    "per_layer_us": (times[1 + reps] - times[1]) / reps * 1e6,
                    "single_call_us": times[1] * 1e6,
                }
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                out[f"{arm}@{rows}"] = {"error": repr(exc)[:400]}
            _log(rank, f"BENCH rows={rows} {arm}: {out[f'{arm}@{rows}']}")
    _set_flag(False)
    return out


def _run_worker(result_dir: Path, bench: bool) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    sys.path.insert(0, str(HERE))
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result_path = result_dir / f"rank_{rank}.json"
    result: dict[str, object] = {"rank": rank, "world": world}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
            init_model_parallel_group,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        world_group = get_world_group()
        backend = dist.get_backend(world_group.device_group)
        groups = {
            # qsa.main_kv_dcp_group(4): TP ranks (0..3), (4..7).
            "pair": init_model_parallel_group(
                [list(range(i, i + SIZE)) for i in range(0, world, SIZE)],
                world_group.local_rank, backend, group_name="packed_gather_dcp4",
            ),
            "tp": init_model_parallel_group(
                [list(range(world))], world_group.local_rank, backend,
                group_name="packed_gather_tp",
            ),
        }
        result["check"] = _check(rank, world, groups)
        result["selection"] = _selection(rank, world, groups)
        if bench:
            result["bench"] = _bench(rank, world, groups)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if rank == 0:
            print("RESULT rank0", json.dumps(result), flush=True)
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


# --------------------------------------------------------------------- pytest
def _launch(result_dir: Path, bench: bool) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir, bench),
        cwd=Path(__file__).resolve().parents[4],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # libtpu logs can carry non-UTF-8 bytes; they must not fail the test.
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        process.wait(timeout=60)
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def _results(result_dir: Path) -> list[dict]:
    out = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        out.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    return out


def assert_results(results: list[dict]) -> None:
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"
    for r in results:
        rank = r["rank"]
        for key, entry in r["check"].items():
            if "/arms_equal/" in key:
                assert entry is True, (rank, key)
                continue
            assert "error" not in entry, (rank, key, entry)
            tokens = next(t for n, _, _, t in CHECKS if key.startswith(n + "/"))
            assert entry["shape"] == [SIZE, tokens, K], (rank, key, entry)
            assert entry["dtypes"] == ["torch.float32", "torch.int32"], (rank, key, entry)
            assert entry["score_bits_mismatch"] == 0 and entry["id_mismatch"] == 0, (
                rank, key, entry)
        for rows in SELECTION_ROWS:
            entry = r["selection"].get(f"{rows}")
            assert entry is not None, (rank, rows, r["selection"])
            assert entry["blocks_equal"] and entry["own_equal"], (rank, rows, entry)
            assert entry["padding_rows_select_nothing"], (rank, rows, entry)
            assert entry["real_rows_selected"] > 0, (rank, rows, entry)


def test_packed_gather_on_real_collectives(tmp_path):
    result_dir = tmp_path / "packed_gather"
    result_dir.mkdir()
    bench = os.environ.get("PACKED_GATHER_BENCH", "0") == "1"
    completed = _launch(result_dir, bench)
    print(completed.stdout[-20000:])
    results = _results(result_dir)
    print("RESULTS", json.dumps(results))
    assert completed.returncode == 0 or any("error" in r for r in results), (
        completed.stdout[-12000:]
    )
    assert_results(results)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    assert args.worker
    args.result_dir.mkdir(parents=True, exist_ok=True)
    _run_worker(args.result_dir, args.bench)


if __name__ == "__main__":
    _main()
