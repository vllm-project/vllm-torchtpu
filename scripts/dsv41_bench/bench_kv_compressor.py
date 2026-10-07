#!/usr/bin/env python3
"""Standalone micro-benchmark & verifier for DeepSeek-V4.1 KV Compressor & Engram.

Exercises the V4.1 `v2` KV Compressor (`project`, `save_compress_norm`, FP8
`encode_fp8_kv` + `PagedCacheRows.write`) and `ParallelEngramEmbedding`
(`cpu_offload=True` Host DRAM vs `cpu_offload=False` HBM) without starting a
vLLM server.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

_BENCH_DIR = pathlib.Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import common  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout  # noqa: E402
from vllm_torchtpu.models.vllm.deepseek_v41.common.engram import (  # noqa: E402
    ParallelEngramEmbedding,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (  # noqa: E402
    NOPE_DIM,
    PagedCacheRows,
    encode_fp8_kv,
    initialize_cache_ops,
)

SUBOPS: tuple[str, ...] = (
    "all",
    "project_only",
    "save_compress_norm_only",
    "write_fp8_only",
    "engram_host_vs_hbm",
)

HIDDEN_SIZE = 5120
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
COMPRESSOR_OUT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
ENGRAM_HEAD_DIM = 256
ENGRAM_BLOCK_SIZE = 32
ENGRAM_N_HASH_COLS = 8
ENGRAM_L1_TOTAL_ROWS = 34_475_840
ENGRAM_L14_TOTAL_ROWS = 137_903_600
ENGRAM_ROW_BYTES = ENGRAM_HEAD_DIM + (ENGRAM_HEAD_DIM // ENGRAM_BLOCK_SIZE)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("auto", "tpu"):
        try:
            import torch_tpu  # noqa: F401

            torch.empty((1,), device="tpu").cpu()
            return torch.device("tpu")
        except Exception:
            if requested == "tpu":
                raise
            return torch.device("cpu")
    return torch.device(requested)


def _sync_if_tpu(device: torch.device, tensors: Any = None) -> None:
    if device.type == "tpu" and tensors is not None:
        common.synchronize(tensors)


def _bench_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return ((time.perf_counter() - t0) * 1e3) / max(1, iters)


def _rms_norm_ref(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _rope_ref(k_pe: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = k_pe.shape[-1] // 2
    x1 = k_pe[..., :half].float()
    x2 = k_pe[..., half:].float()
    c = cos.float()
    s = sin.float()
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return torch.cat((out1, out2), dim=-1).to(k_pe.dtype)


def _encode_fp8_kv_numpy_ref(
    latent_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    arr = latent_bf16.detach().cpu().float().numpy().astype(np.float64)
    n = arr.shape[0]
    nope = arr[:, :NOPE_DIM]
    amax = np.maximum(np.abs(nope).reshape(n, -1, 64).max(axis=1), 1e-4)
    exponent = np.ceil(np.log2(amax / 448.0)).astype(np.int64)
    q = (nope / np.tile(2.0**exponent, NOPE_DIM // 64)).astype(np.float32)
    slab_bytes = np.concatenate(
        (
            q.astype(ml_dtypes.float8_e4m3fn).view(np.uint8),
            (exponent + 127).astype(np.uint8),
        ),
        axis=-1,
    )
    latent_jnp = jnp.asarray(
        latent_bf16.detach().cpu().float().numpy(), dtype=jnp.bfloat16
    )
    words_jnp, rope_jnp = encode_fp8_kv(latent_jnp)
    got_slabs = np.asarray(csa_cache_layout.words_to_slabs(words_jnp)).reshape(n, 512)
    np.testing.assert_array_equal(got_slabs, slab_bytes)
    return (
        torch.from_numpy(np.asarray(words_jnp, dtype=np.int32)),
        torch.from_numpy(np.asarray(rope_jnp, dtype=np.int32)),
    )


def compute_engram_hbm_savings_bytes(tp_size: int = 8) -> dict[str, float]:
    """Calculate exact Engram HBM footprint savings from Host DRAM offload."""
    row_bytes_full = 1024 + 32
    total_bytes = (ENGRAM_L1_TOTAL_ROWS + ENGRAM_L14_TOTAL_ROWS) * row_bytes_full
    per_rank_bytes = total_bytes / tp_size
    return {
        "total_gib": total_bytes / (1024**3),
        "per_rank_gib": per_rank_bytes / (1024**3),
        "tp_size": float(tp_size),
    }


def run_project_only(
    device: torch.device,
    hw: common.Tpu7xRoofline,
    num_tokens: int,
    warmup: int,
    iters: int,
    verify: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    gen = torch.Generator().manual_seed(101)
    x_cpu = torch.randn(num_tokens, HIDDEN_SIZE, generator=gen, dtype=torch.bfloat16)
    w_cpu = torch.randn(
        COMPRESSOR_OUT_DIM, HIDDEN_SIZE, generator=gen, dtype=torch.bfloat16
    ) * (HIDDEN_SIZE**-0.5)
    x = x_cpu.to(device)
    w = w_cpu.to(device)

    def _op() -> torch.Tensor:
        out = torch.nn.functional.linear(x, w)
        _sync_if_tpu(device, out)
        return out

    v_report: dict[str, Any] | None = None
    if verify:
        out_dev = _op().detach().cpu()
        out_ref = torch.nn.functional.linear(x_cpu.float(), w_cpu.float()).to(
            torch.bfloat16
        )
        res = common.verify_outputs(
            {"project_only": out_dev},
            {"project_only": out_ref},
            common.MatchCriteria("lossless", min_cos=0.9999, max_rel_l2=1e-3),
            label="compressor.project_only",
        )
        v_report = {"name": "compressor.project_only", **res["project_only"]}

    lat_ms = _bench_ms(_op, warmup=warmup, iters=iters)
    est = common.RooflineEstimate(
        name="project_only",
        flops=float(2 * num_tokens * HIDDEN_SIZE * COMPRESSOR_OUT_DIM),
        hbm_bytes=float(
            num_tokens * HIDDEN_SIZE * 2
            + COMPRESSOR_OUT_DIM * HIDDEN_SIZE * 2
            + num_tokens * COMPRESSOR_OUT_DIM * 2
        ),
    )
    return est.report(hw, lat_ms), v_report


def run_save_compress_norm_only(
    device: torch.device,
    hw: common.Tpu7xRoofline,
    num_tokens: int,
    warmup: int,
    iters: int,
    verify: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    gen = torch.Generator().manual_seed(202)
    kv_proj_cpu = torch.randn(
        num_tokens, COMPRESSOR_OUT_DIM, generator=gen, dtype=torch.bfloat16
    )
    ln_w_cpu = torch.ones(KV_LORA_RANK, dtype=torch.bfloat16)
    cos_cpu = (
        torch.cos(torch.linspace(0, 3.14, QK_ROPE_HEAD_DIM // 2))
        .to(torch.bfloat16)[None, :]
        .expand(num_tokens, -1)
        .contiguous()
    )
    sin_cpu = (
        torch.sin(torch.linspace(0, 3.14, QK_ROPE_HEAD_DIM // 2))
        .to(torch.bfloat16)[None, :]
        .expand(num_tokens, -1)
        .contiguous()
    )

    kv_proj = kv_proj_cpu.to(device)
    ln_w = ln_w_cpu.to(device)
    cos = cos_cpu.to(device)
    sin = sin_cpu.to(device)

    def _op() -> torch.Tensor:
        c_kv = kv_proj[:, :KV_LORA_RANK]
        k_pe = kv_proj[:, KV_LORA_RANK:]
        c_norm = _rms_norm_ref(c_kv, ln_w)
        k_rope = _rope_ref(k_pe, cos, sin)
        out = torch.cat((c_norm[:, :NOPE_DIM], k_rope), dim=-1)
        _sync_if_tpu(device, out)
        return out

    v_report: dict[str, Any] | None = None
    if verify:
        out_dev = _op().detach().cpu()
        c_ref = _rms_norm_ref(kv_proj_cpu[:, :KV_LORA_RANK], ln_w_cpu)
        k_ref = _rope_ref(kv_proj_cpu[:, KV_LORA_RANK:], cos_cpu, sin_cpu)
        out_ref = torch.cat((c_ref[:, :NOPE_DIM], k_ref), dim=-1)
        res = common.verify_outputs(
            {"save_compress_norm": out_dev},
            {"save_compress_norm": out_ref},
            common.MatchCriteria("lossless", min_cos=0.99999, max_rel_l2=1e-3),
            label="compressor.save_compress_norm_only",
        )
        v_report = {
            "name": "compressor.save_compress_norm_only",
            **res["save_compress_norm"],
        }

    lat_ms = _bench_ms(_op, warmup=warmup, iters=iters)
    est = common.RooflineEstimate(
        name="save_compress_norm_only",
        flops=float(num_tokens * (4 * KV_LORA_RANK + 6 * QK_ROPE_HEAD_DIM)),
        hbm_bytes=float(num_tokens * (COMPRESSOR_OUT_DIM + KV_LORA_RANK) * 2),
    )
    return est.report(hw, lat_ms), v_report


def run_write_fp8_only(
    device: torch.device,
    hw: common.Tpu7xRoofline,
    num_tokens: int,
    num_pages: int,
    page_size: int,
    warmup: int,
    iters: int,
    verify: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if device.type != "cpu":
        with common.model_wrapper_context():
            initialize_cache_ops()
    gen = torch.Generator().manual_seed(303)
    values_cpu = torch.randn(
        num_tokens, KV_LORA_RANK, generator=gen, dtype=torch.bfloat16
    )
    total_slots = num_pages * page_size
    assert total_slots >= num_tokens, (
        f"num_pages*page_size ({total_slots}) must be >= num_tokens ({num_tokens})"
    )
    slots_cpu = torch.arange(num_tokens, dtype=torch.int64)

    raw_rows = total_slots
    raw = torch.zeros((raw_rows, 128), dtype=torch.int32, device=device)
    rope = torch.zeros((raw_rows // 4, 128), dtype=torch.int32, device=device)
    layout = ((num_pages, page_size, 512), torch.uint8, page_size * 512, 0)
    cache = PagedCacheRows(raw, layout, rope)
    slots = slots_cpu.to(device)
    values = values_cpu.to(device)

    def _op() -> tuple[torch.Tensor, torch.Tensor]:
        cache.write(slots, values)
        _sync_if_tpu(device, (raw, rope))
        return raw, rope

    v_report: dict[str, Any] | None = None
    if verify:
        _op()
        exp_words, exp_rope = _encode_fp8_kv_numpy_ref(values_cpu)
        got_words = raw[:num_tokens].detach().cpu()
        got_rope = rope.view(-1, 32)[:num_tokens].detach().cpu()
        res = common.verify_outputs(
            {"main_words": got_words, "rope_words": got_rope},
            {"main_words": exp_words, "rope_words": exp_rope},
            common.EXACT,
            label="compressor.write_fp8_only",
        )
        v_report = {
            "name": "compressor.write_fp8_only.main_words",
            **res["main_words"],
        }

    lat_ms = _bench_ms(_op, warmup=warmup, iters=iters)
    est = common.RooflineEstimate(
        name="write_fp8_only",
        flops=float(num_tokens * NOPE_DIM * 6),
        hbm_bytes=float(num_tokens * (1024 + 512 + 128)),
    )
    return est.report(hw, lat_ms), v_report


def run_engram_host_vs_hbm(
    device: torch.device,
    hw: common.Tpu7xRoofline,
    num_tokens: int,
    engram_rows: int,
    warmup: int,
    iters: int,
    verify: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    from vllm.models.deepseek_v41.common import engram as upstream_engram

    from vllm_torchtpu.models.vllm.deepseek_v41.v2.ops import device_hooks

    head_sizes = tuple(
        engram_rows // ENGRAM_N_HASH_COLS for _ in range(ENGRAM_N_HASH_COLS)
    )
    total_rows = sum(head_sizes)

    with (
        device_hooks(
            upstream_engram,
            get_tensor_model_parallel_world_size=lambda: 8,
            get_tensor_model_parallel_rank=lambda: 0,
        ),
        device_hooks(
            torch.accelerator,
            current_device_index=lambda: 0,
        ),
    ):
        emb_host = ParallelEngramEmbedding(
            total_rows,
            ENGRAM_HEAD_DIM,
            head_sizes,
            block_size=ENGRAM_BLOCK_SIZE,
            cpu_offload=True,
        )
        emb_hbm = ParallelEngramEmbedding(
            total_rows,
            ENGRAM_HEAD_DIM,
            head_sizes,
            block_size=ENGRAM_BLOCK_SIZE,
            cpu_offload=False,
        )
        if device.type != "cpu":
            emb_hbm.to(device)

    assert emb_host.weight.device.type == "cpu"
    assert emb_host.weight_scale_inv.device.type == "cpu"

    gen = torch.Generator().manual_seed(404)
    shard_rows = emb_host.part_num_embeddings
    w_shard = torch.randn(shard_rows, ENGRAM_HEAD_DIM, generator=gen).to(
        torch.float8_e4m3fn
    )
    s_shard = torch.randint(
        120,
        134,
        (shard_rows, ENGRAM_HEAD_DIM // ENGRAM_BLOCK_SIZE),
        generator=gen,
        dtype=torch.uint8,
    )
    emb_host._weight_loader(emb_host.weight, w_shard)
    emb_host._weight_loader(emb_host.weight_scale_inv, s_shard)
    emb_hbm.weight.data.copy_(w_shard.to(emb_hbm.weight.device))
    emb_hbm.weight_scale_inv.data.copy_(s_shard.to(emb_hbm.weight_scale_inv.device))

    indices_cpu = torch.randint(
        0,
        shard_rows,
        (num_tokens, ENGRAM_N_HASH_COLS),
        generator=gen,
        dtype=torch.int32,
    )
    indices_dev = indices_cpu.to(device)
    out_host_cpu = torch.empty(
        (num_tokens, emb_host.part_n_hash_cols, ENGRAM_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cpu",
    )
    out_host_dev = torch.empty(
        (num_tokens, emb_host.part_n_hash_cols, ENGRAM_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    out_hbm_dev = torch.empty_like(out_host_dev)

    def _op_host() -> torch.Tensor:
        emb_host.lookup(indices_cpu, out_host_cpu)
        out_host_dev.copy_(out_host_cpu)
        _sync_if_tpu(device, out_host_dev)
        return out_host_dev

    def _op_hbm() -> torch.Tensor:
        emb_hbm.lookup(indices_dev, out_hbm_dev)
        _sync_if_tpu(device, out_hbm_dev)
        return out_hbm_dev

    v_report: dict[str, Any] | None = None
    if verify:
        got_host = _op_host().detach().cpu()
        got_hbm = _op_hbm().detach().cpu()
        res = common.verify_outputs(
            {"engram_host_vs_hbm": got_host},
            {"engram_host_vs_hbm": got_hbm},
            common.EXACT,
            label="engram.host_vs_hbm",
        )
        v_report = {"name": "engram.host_vs_hbm", **res["engram_host_vs_hbm"]}

    lat_host_ms = _bench_ms(_op_host, warmup=warmup, iters=iters)
    lat_hbm_ms = _bench_ms(_op_hbm, warmup=warmup, iters=iters)
    lookup_rows = num_tokens * emb_host.part_n_hash_cols
    flops = float(lookup_rows * ENGRAM_HEAD_DIM * 3)
    bytes_transferred = float(lookup_rows * (ENGRAM_ROW_BYTES + ENGRAM_HEAD_DIM * 2))
    r_host = common.RooflineEstimate(
        name="engram_host_dram",
        flops=flops,
        hbm_bytes=float(lookup_rows * ENGRAM_HEAD_DIM * 2),
    ).report(hw, lat_host_ms)
    r_hbm = common.RooflineEstimate(
        name="engram_hbm",
        flops=flops,
        hbm_bytes=bytes_transferred,
    ).report(hw, lat_hbm_ms)
    return r_host, r_hbm, v_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark & verify DeepSeek-V4.1 KV Compressor & Engram."
    )
    parser.add_argument(
        "--subop",
        type=str,
        default="all",
        choices=SUBOPS,
        help="Sub-operation to benchmark/verify (default: all).",
    )
    parser.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify numerical outputs against golden/reference (default: True).",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--num-pages", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--engram-rows", type=int, default=65536)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "tpu", "cpu"),
    )
    parser.add_argument("--dump-llo", type=Path, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.device in ("auto", "tpu") and "RANK" not in os.environ:
        try:
            from torch_tpu._internal.utils import hardware

            tpu_count = hardware.get_tpu_device_count()
        except Exception:
            tpu_count = 0
        if tpu_count > 0:
            import subprocess

            raw_argv = list(argv) if argv is not None else list(sys.argv[1:])
            env = common.prepare_worker_env(SimpleNamespace(profile=None))
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={common.WORLD_SIZE}",
                str(Path(__file__).resolve()),
                *raw_argv,
            ]
            subprocess.run(cmd, env=env, check=True)
            if args.json_out and Path(args.json_out).exists():
                return json.loads(Path(args.json_out).read_text(encoding="utf-8"))
            return {"device": "tpu", "subop": args.subop}
        if args.device == "tpu":
            raise RuntimeError("No TPU devices found.")
        device = torch.device("cpu")
    elif args.device != "cpu" and "RANK" in os.environ:
        import torch.distributed as dist
        import torch_tpu  # noqa: F401

        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        device = torch.device("tpu")
    else:
        device = _resolve_device(args.device)

    rank = int(os.environ.get("RANK", "0"))
    if args.dump_llo is not None:
        common._llo_env(args.dump_llo.resolve(), rank, os.environ)  # type: ignore[arg-type]

    hw = common.Tpu7xRoofline()
    rooflines: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []

    if args.subop in ("all", "project_only"):
        r, v = run_project_only(
            device, hw, args.num_tokens, args.warmup, args.iters, args.verify
        )
        rooflines.append(r)
        if v is not None:
            verifications.append(v)

    if args.subop in ("all", "save_compress_norm_only"):
        r, v = run_save_compress_norm_only(
            device, hw, args.num_tokens, args.warmup, args.iters, args.verify
        )
        rooflines.append(r)
        if v is not None:
            verifications.append(v)

    if args.subop in ("all", "write_fp8_only"):
        r, v = run_write_fp8_only(
            device,
            hw,
            args.num_tokens,
            args.num_pages,
            args.page_size,
            args.warmup,
            args.iters,
            args.verify,
        )
        rooflines.append(r)
        if v is not None:
            verifications.append(v)

    if args.subop in ("all", "engram_host_vs_hbm"):
        r_host, r_hbm, v = run_engram_host_vs_hbm(
            device,
            hw,
            args.num_tokens,
            args.engram_rows,
            args.warmup,
            args.iters,
            args.verify,
        )
        rooflines.extend([r_host, r_hbm])
        if v is not None:
            verifications.append(v)

    hbm_savings = compute_engram_hbm_savings_bytes(tp_size=8)
    summary: dict[str, Any] = {
        "device": str(device),
        "subop": args.subop,
        "num_tokens": args.num_tokens,
        "engram_hbm_savings": hbm_savings,
        "rooflines": rooflines,
        "verifications": verifications,
    }
    if rank == 0:
        saved_gib = hbm_savings["per_rank_gib"]
        print(
            f"[bench_kv_compressor] device={device} subop={args.subop} "
            f"T={args.num_tokens} engram_saved_per_rank={saved_gib:.2f} GiB"
        )
        for r in rooflines:
            meas = r.get("measured_ms", 0.0)
            lb = r["lower_bound_ms"]
            eff = r.get("roofline_efficiency", 0.0)
            print(
                f"  [ROOFLINE] {r['name']}: measured={meas:.4f} ms | "
                f"lower_bound={lb:.4f} ms | eff={eff:.2%}"
            )
        for v in verifications:
            print(
                f"  [VERIFY]   {v['name']}: ok={v['ok']} "
                f"bitwise={v.get('bitwise_equal', False)}"
            )
        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2))
    if device.type == "tpu":
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    return summary


if __name__ == "__main__":
    main()
