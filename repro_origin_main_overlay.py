# SPDX-License-Identifier: Apache-2.0
"""Benchmark and capture TPU XPlane traces for vllm-torchtpu PR #908 vs main."""

import os
import sys

# Re-exec once at process start so PYTHONPATH and TPU_MOE_* env vars are active
# before Python startup hooks, sitecustomize, or torch_tpu import vllm_torchtpu.
if "--summarize-jsonl" not in sys.argv and os.environ.get("_PR908_REEXEC") != "1":
    _case = ""
    for _idx, _arg in enumerate(sys.argv):
        if _arg == "--case" and _idx + 1 < len(sys.argv):
            _case = sys.argv[_idx + 1]
            break
    _is_cand = "candidate" in _case
    _repo_src = "/app/vllm_torchtpu_repo/src"
    _env = os.environ.copy()
    _env["_PR908_REEXEC"] = "1"
    _env["TPU_MOE_COUNTING_SORT"] = "1"
    _env["TPU_MOE_SINGLE_SORT"] = "1"
    _env["TPU_MOE_JOINT_ROUTING_128"] = "1" if _is_cand else "0"
    _env["TPU_MOE_SKIP_PADDED_TOKENS"] = "0"
    _env["TPU_MOE_DENSE_EXPERT_THRESHOLD"] = "0"
    _env["USE_MOE_SPARSE_CORE"] = "0"
    _env["ONEHOT_MOE_PERMUTE_THRESHOLD"] = "3840"
    _env["TPU_MOE_OWNER_OUTPUT_MODE"] = "off"
    _env["TPU_MOE_TOPK_WEIGHTS_FLOAT32"] = "0"
    _old_pp = [
        p
        for p in _env.get("PYTHONPATH", "").split(":")
        if p and "/workspace/vllm-torchtpu" not in p and "/root/torchtpu-vllm" not in p
    ]
    _env["PYTHONPATH"] = ":".join([_repo_src, *_old_pp])
    os.execve(sys.executable, [sys.executable, *sys.argv], _env)

import argparse
import gzip
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, required=False, default="")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/pr908_trace_out")
    )
    parser.add_argument("--llo-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--append-jsonl", type=Path, default=None)
    parser.add_argument("--summarize-jsonl", type=Path, default=None)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--profile-steps", type=int, default=5)
    return parser.parse_args()


def format_summary_table(jsonl_path: Path) -> str:
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    header = (
        f"{'Tag':24s} | {'Joint128':8s} | {'ChainedUs':>10s} | "
        f"{'SingleUs':>9s} | {'XlaModUs':>9s} | {'HloSorts':>8s} | {'Pallas128':>9s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row.get('tag', row.get('case'))):24s} | "
            f"{str(row.get('joint_routing_128')):8s} | "
            f"{row.get('chained_per_call_median_us', 0.0):10.3f} | "
            f"{row.get('single_wall_median_us', 0.0):9.3f} | "
            f"{row.get('xla_module_median_us', 0.0):9.3f} | "
            f"{str(row.get('hlo_sort_ops', '?')):>8s} | "
            f"{str(row.get('hlo_joint_pallas_ops', '?')):>9s}"
        )
    return "\n".join(lines)


def _git_sha(path: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _summarize_hlo(hlo_dir: Path) -> dict[str, object]:
    files = sorted(hlo_dir.glob("*after_optimizations.txt"))
    candidates = []
    for f in files:
        text = f.read_text(errors="replace")
        has_routing = (
            "stable_order_and_inverse_128" in text
            or "sort(" in text
            or "gmm" in text.lower()
        )
        if not has_routing:
            continue
        sorts = len(re.findall(r"=\s*[^\n]*\bsort\(", text))
        scatters = len(re.findall(r"=\s*[^\n]*\bscatter\(", text))
        pallas128 = len(re.findall(r"stable_order_and_inverse_128", text))
        candidates.append(
            {
                "file": f.name,
                "lines": text.count("\n"),
                "sorts": sorts,
                "scatters": scatters,
                "pallas128": pallas128,
            }
        )
    if not candidates:
        return {
            "hlo_file": None,
            "hlo_sort_ops": None,
            "hlo_scatter_ops": None,
            "hlo_joint_pallas_ops": None,
            "hlo_modules": [],
        }
    single = min(candidates, key=lambda c: c["lines"])
    return {
        "hlo_file": single["file"],
        "hlo_sort_ops": single["sorts"],
        "hlo_scatter_ops": single["scatters"],
        "hlo_joint_pallas_ops": single["pallas128"],
        "hlo_modules": candidates,
    }


def _extract_xplane_metrics(
    profile_dir: Path, xspace_cls, out_gz_path: Path
) -> dict[str, object]:
    xplane_paths = sorted(profile_dir.rglob("*.xplane.pb"))
    if not xplane_paths:
        raise FileNotFoundError(f"No .xplane.pb found under {profile_dir}")
    target_xplane = xplane_paths[-1]
    raw_bytes = target_xplane.read_bytes()
    out_gz_path.parent.mkdir(parents=True, exist_ok=True)
    out_gz_path.write_bytes(gzip.compress(raw_bytes))

    space = xspace_cls()
    space.ParseFromString(raw_bytes)

    plane_names = [p.name for p in space.planes]
    for plane in space.planes:
        if plane.name != "/device:TPU:0":
            continue
        events_by_line: dict[str, list[tuple[int, int, str]]] = {}
        for line in plane.lines:
            bucket = events_by_line.setdefault(line.name, [])
            for ev in line.events:
                meta = plane.event_metadata[ev.metadata_id]
                label = meta.display_name or meta.name
                start_ps = line.timestamp_ns * 1000 + ev.offset_ps
                bucket.append((start_ps, ev.duration_ps, label))

        modules = sorted(events_by_line.get("XLA Modules", []))
        ops = events_by_line.get("XLA Ops", [])

        calls_by_mod: dict[str, list[tuple[int, int]]] = {}
        for s_ps, d_ps, mod_name in modules:
            calls_by_mod.setdefault(mod_name, []).append((s_ps, d_ps))

        mod_summary = {
            name: [round(d_ps / 1e6, 3) for _, d_ps in calls]
            for name, calls in calls_by_mod.items()
        }

        target_mod_name = None
        target_calls: list[tuple[int, int]] = []
        for mod_name, calls in calls_by_mod.items():
            if len(calls) >= 3 and len(calls) >= len(target_calls):
                target_mod_name = mod_name
                target_calls = calls

        module_samples_us = [d_ps / 1e6 for _, d_ps in target_calls]
        eval_calls = target_calls[1:] if len(target_calls) >= 4 else target_calls
        op_totals_ps: dict[str, int] = {}
        op_counts: dict[str, int] = {}
        for s_ps, d_ps, op_name in ops:
            for m_start, m_dur in eval_calls:
                if m_start <= s_ps and s_ps + d_ps <= m_start + m_dur + 100_000:
                    op_totals_ps[op_name] = op_totals_ps.get(op_name, 0) + d_ps
                    op_counts[op_name] = op_counts.get(op_name, 0) + 1
                    break

        num_eval = max(1, len(eval_calls))
        top_xla_ops = sorted(
            (
                {
                    "op": name,
                    "per_call_us": round(total_ps / num_eval / 1e6, 3),
                    "total_calls": op_counts[name],
                }
                for name, total_ps in op_totals_ps.items()
            ),
            key=lambda x: x["per_call_us"],
            reverse=True,
        )[:20]

        return {
            "xplane_file": str(target_xplane),
            "xplane_gz_file": str(out_gz_path),
            "xplane_bytes": len(raw_bytes),
            "xplane_gz_bytes": out_gz_path.stat().st_size,
            "planes": plane_names,
            "line_names": list(events_by_line.keys()),
            "all_xla_modules_us": mod_summary,
            "xla_module_name": target_mod_name,
            "xla_module_call_count": len(target_calls),
            "xla_module_all_us": [round(v, 3) for v in module_samples_us],
            "xla_module_median_us": (
                round(statistics.median(module_samples_us), 3)
                if module_samples_us
                else 0.0
            ),
            "top_xla_ops": top_xla_ops,
        }
    raise RuntimeError(
        f"Missing /device:TPU:0 plane in {target_xplane}; planes={plane_names}"
    )


def _run_native_tpu_check(fused_moe_gmm_mod, stable_fn, jax_op_fn) -> dict[str, object]:
    """Run native TPU7x parity checks through torch_tpu jax_op."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch

    def _native_kernel(k: jax.Array) -> jax.Array:
        return jnp.stack(stable_fn(k), axis=0)

    kernel_op = jax_op_fn("pallas::pr908_native_kernel_check", _native_kernel)
    kernel_op.register_fake(lambda k: k.new_empty((2, 128)))
    kernel_compiled = torch.compile(
        kernel_op, backend="tpu", fullgraph=True, dynamic=False
    )

    rng = np.random.default_rng(1729)
    int32_info = np.iinfo(np.int32)
    cases = [
        np.zeros(128, dtype=np.int32),
        np.arange(127, -1, -1, dtype=np.int32),
        np.tile(
            np.array([int32_info.min, int32_info.max, -1, 0], dtype=np.int32),
            32,
        ),
        rng.integers(-8, 9, size=128, dtype=np.int32),
        rng.integers(
            int32_info.min, int32_info.max, size=128, dtype=np.int32, endpoint=True
        ),
    ]
    for keys in cases:
        t_keys = torch.from_numpy(keys).to(torch.int32).to("tpu")
        out = kernel_compiled(t_keys).cpu().numpy()
        order_np, inv_np = out[0], out[1]
        expected = np.argsort(keys, kind="stable").astype(np.int32)
        np.testing.assert_array_equal(order_np, expected)
        np.testing.assert_array_equal(inv_np[order_np], np.arange(128, dtype=np.int32))
        np.testing.assert_array_equal(order_np[inv_np], np.arange(128, dtype=np.int32))

    orig_flag = fused_moe_gmm_mod._JOINT_ROUTING_128
    try:
        hidden_np = rng.standard_normal((16, 2048)).astype(np.float32) * 0.25
        ids_np = rng.integers(0, 256, size=(16, 8), dtype=np.int32)
        ids_np[1::2] = ids_np[:8]
        ids_np.ravel()[:2] = [0, 255]
        weights_np = rng.random((16, 8), dtype=np.float32)

        h_tpu = torch.from_numpy(hidden_np).to(torch.bfloat16).to("tpu")
        i_tpu = torch.from_numpy(ids_np).to(torch.int32).to("tpu")
        w_tpu = torch.from_numpy(weights_np).to(torch.bfloat16).to("tpu")

        def _prep_call_off(
            h: jax.Array, i: jax.Array, w: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return fused_moe_gmm_mod.prepare_routed_gmm_inputs(
                h,
                i,
                w,
                local_num_experts=256,
                topk=8,
                use_ep=False,
                use_sparse_core=False,
                skip_padded_tokens=False,
            )

        def _prep_call_on(
            h: jax.Array, i: jax.Array, w: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return fused_moe_gmm_mod.prepare_routed_gmm_inputs(
                h,
                i,
                w,
                local_num_experts=256,
                topk=8,
                use_ep=False,
                use_sparse_core=False,
                skip_padded_tokens=False,
            )

        fused_moe_gmm_mod._JOINT_ROUTING_128 = False
        op_off = jax_op_fn("pallas::pr908_prep_parity_off", _prep_call_off)
        op_off.register_fake(
            lambda h, i, w: (
                h.new_empty((128, 2048)),
                i.new_empty((256,)),
                i.new_empty((128,)),
                w.new_empty((128,)),
                h.new_empty((128,), dtype=torch.bool),
                i.new_empty((128,)),
            )
        )
        outs_off = [
            x.cpu()
            for x in torch.compile(op_off, backend="tpu", fullgraph=True, dynamic=False)(
                h_tpu, i_tpu, w_tpu
            )
        ]

        fused_moe_gmm_mod._JOINT_ROUTING_128 = True
        op_on = jax_op_fn("pallas::pr908_prep_parity_on", _prep_call_on)
        op_on.register_fake(
            lambda h, i, w: (
                h.new_empty((128, 2048)),
                i.new_empty((256,)),
                i.new_empty((128,)),
                w.new_empty((128,)),
                h.new_empty((128,), dtype=torch.bool),
                i.new_empty((128,)),
            )
        )
        outs_on = [
            x.cpu()
            for x in torch.compile(op_on, backend="tpu", fullgraph=True, dynamic=False)(
                h_tpu, i_tpu, w_tpu
            )
        ]

        for got, want in zip(outs_on, outs_off):
            assert got.dtype == want.dtype
            np.testing.assert_array_equal(
                got.view(torch.uint16).numpy()
                if got.dtype == torch.bfloat16
                else got.numpy(),
                want.view(torch.uint16).numpy()
                if want.dtype == torch.bfloat16
                else want.numpy(),
            )
    finally:
        fused_moe_gmm_mod._JOINT_ROUTING_128 = orig_flag

    return {"native_kernel_cases": len(cases), "prepare_bit_exact_tpu": True}


def main() -> None:
    args = parse_args()
    if args.summarize_jsonl is not None:
        print(format_summary_table(args.summarize_jsonl))
        return

    is_candidate = "candidate" in args.case

    import importlib.metadata
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal.pallas import jax_op
    from torch_tpu._internal.profiler import TpuProfilerConfig

    import vllm_torchtpu
    print(
        f"[debug] PYTHONPATH={os.environ.get('PYTHONPATH')!r} "
        f"vllm_torchtpu.__file__={getattr(vllm_torchtpu, '__file__', None)!r}",
        flush=True,
    )
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.stable_order_and_inverse import (
        stable_order_and_inverse_128,
    )
    from vllm_torchtpu.layers.core import fused_moe_gmm
    from vllm_torchtpu.tools.xplane_pb2 import XSpace

    assert "/app/vllm_torchtpu_repo" in str(vllm_torchtpu.__file__), (
        f"Imported wrong vllm_torchtpu: {vllm_torchtpu.__file__}, sys.path={sys.path}"
    )
    assert fused_moe_gmm._JOINT_ROUTING_128 is is_candidate, (
        f"Expected _JOINT_ROUTING_128={is_candidate}, got "
        f"{fused_moe_gmm._JOINT_ROUTING_128}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.empty(1, device="tpu").cpu()

    is_prep = args.case.startswith("prep")
    num_tokens, hidden_size, intermediate_size, num_experts, topk = (
        16,
        2048,
        128,
        256,
        8,
    )
    iters = 32 if is_prep else 8

    rng = np.random.default_rng(1729)
    hidden_cpu = rng.standard_normal((num_tokens, hidden_size)).astype(np.float32) * 0.25
    ids_bank_np = np.empty((iters, num_tokens, topk), dtype=np.int32)
    for b in range(iters):
        ids_b = rng.integers(0, num_experts, size=(num_tokens, topk), dtype=np.int32)
        ids_b[1::2] = ids_b[: num_tokens // 2]
        ids_b.ravel()[:2] = [0, num_experts - 1]
        ids_bank_np[b] = ids_b
    weights_cpu = rng.dirichlet(np.ones(topk), size=num_tokens).astype(np.float32)

    hidden_tpu = torch.from_numpy(hidden_cpu).to(torch.bfloat16).to("tpu")
    ids_bank_tpu = torch.from_numpy(ids_bank_np).to(torch.int32).to("tpu")
    ids_single_tpu = [
        torch.from_numpy(ids_bank_np[i]).to(torch.int32).to("tpu")
        for i in range(args.profile_steps)
    ]
    weights_tpu = torch.from_numpy(weights_cpu).to(torch.bfloat16).to("tpu")

    if is_prep:
        tag_prefix = f"pr908_{args.case}"

        def single_jax_fn(
            h: jax.Array, ids: jax.Array, w: jax.Array
        ) -> jax.Array:
            x, gs, rev, tw, vm, ti_s = fused_moe_gmm.prepare_routed_gmm_inputs(
                h,
                ids,
                w,
                local_num_experts=num_experts,
                topk=topk,
                use_ep=False,
                use_sparse_core=False,
                skip_padded_tokens=False,
            )
            return (
                rev.astype(jnp.int32)
                + ti_s.astype(jnp.int32)
                + vm.astype(jnp.int32)
                + gs[:128].astype(jnp.int32)
                + gs[128:].astype(jnp.int32)
                + x[:, 0].astype(jnp.int32)
                + tw.astype(jnp.int32)
            )

        def chained_jax_fn(
            h: jax.Array, ids_bank: jax.Array, w: jax.Array
        ) -> jax.Array:
            acc = jnp.zeros((128,), dtype=jnp.int32)
            cur_ids = ids_bank[0]
            cur_h = h
            for k in range(iters):
                x, gs, rev, tw, vm, ti_s = fused_moe_gmm.prepare_routed_gmm_inputs(
                    cur_h,
                    cur_ids,
                    w,
                    local_num_experts=num_experts,
                    topk=topk,
                    use_ep=False,
                    use_sparse_core=False,
                    skip_padded_tokens=False,
                )
                acc = (
                    acc
                    + rev
                    + ti_s
                    + vm.astype(jnp.int32)
                    + x[:, 0].astype(jnp.int32)
                    + tw.astype(jnp.int32)
                    + gs[:128]
                    + gs[128:]
                )
                if k + 1 < iters:
                    cur_ids = jnp.where(rev[0] == -99999, ids_bank[0], ids_bank[k + 1])
                    cur_h = (
                        h.astype(jnp.float32) + 1e-7 * x[:num_tokens, :1].astype(jnp.float32)
                    ).astype(h.dtype)
            return acc

        single_op = jax_op(f"pallas::{tag_prefix}_single", single_jax_fn)
        single_op.register_fake(lambda h, ids, w: ids.reshape(-1).new_empty((128,)))
        chained_op = jax_op(f"pallas::{tag_prefix}_chained", chained_jax_fn)
        chained_op.register_fake(
            lambda h, ids_b, w: ids_b[0].reshape(-1).new_empty((128,))
        )

        single_inputs_fn = lambda s: (hidden_tpu, ids_single_tpu[s], weights_tpu)
        chained_inputs = (hidden_tpu, ids_bank_tpu, weights_tpu)
    else:
        tag_prefix = f"pr908_{args.case}"
        w1_cpu = (
            rng.standard_normal((num_experts, hidden_size, intermediate_size * 2)).astype(
                np.float32
            )
            * 0.05
        )
        w2_cpu = (
            rng.standard_normal((num_experts, intermediate_size, hidden_size)).astype(
                np.float32
            )
            * 0.05
        )
        w1_tpu = torch.from_numpy(w1_cpu).to(torch.bfloat16).to("tpu")
        w2_tpu = torch.from_numpy(w2_cpu).to(torch.bfloat16).to("tpu")
        start_tpu = torch.tensor(0, dtype=torch.int32, device="tpu")

        def _call_moe(
            h: jax.Array,
            w1: jax.Array,
            w2: jax.Array,
            w: jax.Array,
            ids: jax.Array,
            st: jax.Array,
        ) -> jax.Array:
            return fused_moe_gmm.fused_moe_func(
                h,
                w1,
                w2,
                None,
                None,
                None,
                None,
                w,
                ids,
                st,
                topk=topk,
                activation="silu",
                use_ep=False,
                use_sparse_core=False,
                onehot_moe_permute_threshold=0,
                rhs_quant_dtype=None,
                skip_padded_tokens=False,
            )

        def single_jax_fn(
            h: jax.Array,
            w1: jax.Array,
            w2: jax.Array,
            w: jax.Array,
            ids: jax.Array,
            st: jax.Array,
        ) -> jax.Array:
            return _call_moe(h, w1, w2, w, ids, st)

        def chained_jax_fn(
            h: jax.Array,
            w1: jax.Array,
            w2: jax.Array,
            w: jax.Array,
            ids_bank: jax.Array,
            st: jax.Array,
        ) -> jax.Array:
            acc = jnp.zeros((num_tokens, hidden_size), dtype=jnp.float32)
            cur_h = h
            for k in range(iters):
                y = _call_moe(cur_h, w1, w2, w, ids_bank[k], st)
                acc = acc + y.astype(jnp.float32)
                if k + 1 < iters:
                    cur_h = (h.astype(jnp.float32) + 1e-6 * y.astype(jnp.float32)).astype(
                        h.dtype
                    )
            return acc

        single_op = jax_op(f"pallas::{tag_prefix}_single", single_jax_fn)
        single_op.register_fake(lambda h, *_: torch.empty_like(h))
        chained_op = jax_op(f"pallas::{tag_prefix}_chained", chained_jax_fn)
        chained_op.register_fake(
            lambda h, *_: torch.empty(h.shape, dtype=torch.float32, device=h.device)
        )

        single_inputs_fn = lambda s: (
            hidden_tpu,
            w1_tpu,
            w2_tpu,
            weights_tpu,
            ids_single_tpu[s],
            start_tpu,
        )
        chained_inputs = (
            hidden_tpu,
            w1_tpu,
            w2_tpu,
            weights_tpu,
            ids_bank_tpu,
            start_tpu,
        )

    single_compiled = torch.compile(
        single_op, backend="tpu", fullgraph=True, dynamic=False
    )
    chained_compiled = torch.compile(
        chained_op, backend="tpu", fullgraph=True, dynamic=False
    )

    for s in range(args.profile_steps):
        out = single_compiled(*single_inputs_fn(s))
        torch.accelerator.synchronize()
        assert torch.isfinite(out.to(torch.float32).cpu()).all()

    out_c = chained_compiled(*chained_inputs)
    torch.accelerator.synchronize()
    assert torch.isfinite(out_c.to(torch.float32).cpu()).all()

    for _ in range(5):
        chained_compiled(*chained_inputs)
    torch.accelerator.synchronize()

    chained_round_us: list[float] = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        chained_compiled(*chained_inputs)
        torch.accelerator.synchronize()
        chained_round_us.append((time.perf_counter() - t0) * 1e6 / float(iters))

    single_round_us: list[float] = []
    s_in = single_inputs_fn(0)
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        for _ in range(25):
            single_compiled(*s_in)
        torch.accelerator.synchronize()
        single_round_us.append((time.perf_counter() - t0) * 1e6 / 25.0)

    # Throwaway first profiler capture so PJRT initializes device plane cleanly.
    warmup_prof_dir = Path("/tmp/xprof_warmup") / (args.tag or args.case)
    warmup_prof_dir.mkdir(parents=True, exist_ok=True)
    wcfg = TpuProfilerConfig(
        run_dir=str(warmup_prof_dir),
        host_tracer_level=1,
        device_tracer_level=1,
        python_tracer_level=0,
        experimental_options={"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"},
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(warmup_prof_dir), use_gzip=True
        ),
        experimental_config=wcfg,
    ):
        single_compiled(*s_in)
        torch.accelerator.synchronize()

    prof_dir = args.out_dir / "profile"
    prof_dir.mkdir(parents=True, exist_ok=True)
    cfg = TpuProfilerConfig(
        run_dir=str(prof_dir),
        host_tracer_level=1,
        device_tracer_level=1,
        python_tracer_level=0,
        experimental_options={"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"},
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.PrivateUse1,
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(prof_dir), use_gzip=True
        ),
        experimental_config=cfg,
    ):
        time.sleep(0.05)
        for s in range(args.profile_steps):
            with torch.profiler.record_function(f"{args.tag or args.case}_step{s}"):
                single_compiled(*single_inputs_fn(s))
                torch.accelerator.synchronize()
        time.sleep(0.05)

    tag_label = args.tag or args.case
    xplane_summary = _extract_xplane_metrics(
        prof_dir,
        XSpace,
        args.out_dir / f"{tag_label}.xplane.pb.gz",
    )
    hlo_summary = _summarize_hlo(args.out_dir / "hlo_dump")

    correctness = (
        _run_native_tpu_check(fused_moe_gmm, stable_order_and_inverse_128, jax_op)
        if args.case == "prep_candidate"
        else {}
    )

    versions: dict[str, str] = {}
    for pkg in ("torch", "torch_tpu", "jax", "jaxlib", "libtpu", "vllm"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue

    report = {
        "tag": tag_label,
        "case": args.case,
        "joint_routing_128": is_candidate,
        "shape": {
            "num_tokens": num_tokens,
            "topk": topk,
            "routed_rows": num_tokens * topk,
            "local_num_experts": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size if not is_prep else None,
            "use_ep": False,
            "use_sparse_core": False,
            "skip_padded_tokens": False,
        },
        "iters_per_round": iters,
        "rounds": args.rounds,
        "chained_per_call_median_us": round(statistics.median(chained_round_us), 3),
        "chained_per_call_rounds_us": [round(v, 3) for v in chained_round_us],
        "single_wall_median_us": round(statistics.median(single_round_us), 3),
        "single_wall_rounds_us": [round(v, 3) for v in single_round_us],
        **hlo_summary,
        **xplane_summary,
        **correctness,
        "runtime": {
            "versions": versions,
            "vllm_torchtpu_file": fused_moe_gmm.__file__,
            "repo_sha": _git_sha(Path("/app/vllm_torchtpu_repo")),
            "env_TPU_MOE_COUNTING_SORT": envs.TPU_MOE_COUNTING_SORT,
            "env_TPU_MOE_SINGLE_SORT": envs.TPU_MOE_SINGLE_SORT,
            "env_TPU_MOE_JOINT_ROUTING_128": envs.TPU_MOE_JOINT_ROUTING_128,
        },
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.append_jsonl is not None:
        args.append_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.append_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
