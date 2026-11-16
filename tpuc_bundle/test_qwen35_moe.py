# pytype: skip-file
"""Unit test and benchmark for Qwen 3.5 MoE under EP8 (`fused_moe_gmm` vs `fused_moe_tpuc`).

Validates numerical parity and benchmarks latency (median of 10 runs after
warmup) between `vllm_torchtpu.layers.adapter.fused_moe.fused_moe_gmm`
(reference) and `fused_moe_tpuc_layer.fused_moe_tpuc` (bare-metal TPU-C MoE
kernel) across 8 EP ranks on a single TPU v7x host (2x2x1 = 4 chips = 8
TensorCores) using Qwen3.5-397B-A17B-FP8 dimensions:
  - EP_SIZE = 8, E_GLOBAL = 512, E_LOCAL = 64
  - HIDDEN_SIZE = 4096, INTERMEDIATE_SIZE = 1024 (gate+up = 2048)
  - `TPU_MOE_MODEL=qwen35_35b`: Qwen3.5-35B-A3B dims instead (E_GLOBAL = 256,
    E_LOCAL = 32, HIDDEN_SIZE = 2048, INTERMEDIATE_SIZE = 512, top-8)
  - TOPK = 10, activation = "silu"
  - Local tokens per rank: [8, 32, 64, 1024, 2048]
    (All-gathered global tokens: [64, 256, 512, 8192, 16384])
  - Collective flow: All-Gather (dispatch) -> MoE kernel -> Reduce-Scatter
    (combine)

Usage on Cloud TPU VM / CDK:
  source env.sh
  python3 test_qwen35_moe.py
"""

import argparse
import faulthandler
import importlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

try:
  importlib.import_module("symbol_loader")
except ImportError:
  pass

import torch
from torch import distributed as dist
import torch_tpu  # pylint: disable=unused-import
from vllm.distributed import parallel_state
from vllm_torchtpu.layers.adapter.fused_moe import fused_moe_gmm


def _import_fused_moe_tpuc():
  user = os.environ.get("USER") or os.environ.get("LOGNAME") or "hning"
  for mod_name in (
      f"google3.experimental.users.{user}.torchtpu_tpuc_cdk.torchtpu_tpuc_kernels.moe.fused_moe_tpuc_layer",
      "fused_moe_tpuc_layer",
      "plugins.fused_moe_tpuc_layer",
  ):
    try:
      mod = importlib.import_module(mod_name)
      return mod.fused_moe_tpuc
    except ImportError:
      continue
  raise ImportError("Could not import fused_moe_tpuc_layer.fused_moe_tpuc")


fused_moe_tpuc = _import_fused_moe_tpuc()

WORLD_SIZE = 8
# MoE dims: Qwen3.5-397B-A17B by default; `TPU_MOE_MODEL=qwen35_35b` selects
# Qwen3.5-35B-A3B (E=256, H=2048, I=512, top-8).
if os.environ.get("TPU_MOE_MODEL", "qwen35_397b") == "qwen35_35b":
  E_GLOBAL, HIDDEN_SIZE, INTERMEDIATE_SIZE, TOPK = 256, 2048, 512, 8
else:
  E_GLOBAL, HIDDEN_SIZE, INTERMEDIATE_SIZE, TOPK = 512, 4096, 1024, 10
E_LOCAL = E_GLOBAL // WORLD_SIZE  # 64 (397B) / 32 (35B) local experts per rank
LOCAL_TOKEN_COUNTS = (8, 32, 64, 1024, 2048)
# Optional override, e.g. `TPU_MOE_LOCAL_TOKENS=1024` (T_global=8192 on EP8).
if os.environ.get("TPU_MOE_LOCAL_TOKENS"):
  LOCAL_TOKEN_COUNTS = tuple(
      int(t) for t in os.environ["TPU_MOE_LOCAL_TOKENS"].split(",") if t
  )
FP8_MAX = 448.0
WARMUP_RUNS = 3
BENCH_RUNS = 10
LAUNCH_TIMEOUT_SECONDS = 480


def _dump_tpu_error_logs() -> None:
  """Print any TPU driver ERROR logs from /tmp/tpu_logs on failure/timeout."""
  log_root = Path("/tmp/tpu_logs")
  if not log_root.exists():
    return
  for err_file in sorted(log_root.rglob("*.ERROR*")):
    try:
      if err_file.is_file() and not err_file.is_symlink():
        content = err_file.read_text(encoding="utf-8", errors="replace").strip()
        if content:
          print(f"\n=== TPU ERROR LOG: {err_file} ===\n{content}\n", flush=True)
    except OSError:
      pass


def _prepare_tpu_launch_env(world_size: int) -> None:
  """Configure single-host multi-process TPU launch environment variables."""
  tpu = getattr(torch, "tpu", None)
  tpu_dist = getattr(tpu, "distributed", None) if tpu is not None else None
  if tpu_dist is not None and hasattr(tpu_dist, "set_tpu_launch_env"):
    tpu_dist.set_tpu_launch_env(nproc_per_node=world_size)
    return

  from torch_tpu._internal.distributed.launchers import singlehost_wrapper  # pylint: disable=g-import-not-at-top

  singlehost_wrapper.prepare_tpu_environment(world_size=world_size)


def _prepare_worker_env() -> dict[str, str]:
  """Prepare environment variables for spawning 8 TorchTPU worker processes."""
  keys_to_clear = (
      "TORCH_TPU_XPROF_SESSION_ID",
      "TORCH_TPU_SLICEBUILDER_ADDRESSES",
      "TORCH_TPU_TOPOLOGY",
      "TPU_CHIPS_PER_HOST_BOUNDS",
      "TPU_HOST_BOUNDS",
      "TPU_VISIBLE_DEVICES",
  )
  saved = {k: os.environ.get(k) for k in keys_to_clear}
  try:
    for k in keys_to_clear:
      os.environ.pop(k, None)
    _prepare_tpu_launch_env(WORLD_SIZE)
    env = os.environ.copy()
  finally:
    for k, v in saved.items():
      if v is None:
        os.environ.pop(k, None)
      else:
        os.environ[k] = v

  env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
  env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
  env.setdefault("VLLM_USE_AOT_COMPILE", "0")
  env.setdefault("VLLM_NO_USAGE_STATS", "1")
  env.setdefault("VLLM_DO_NOT_TRACK", "1")
  # TPU-C executables require continuations; the container's libtpu needs the
  # runtime ABI check bypassed for executables serialized from google3.
  libtpu_args = env.get("LIBTPU_INIT_ARGS", "")
  for flag in (
      "--tpu_use_continuations=true",
      "--DANGEROUS_tpu_runtime_abi_verification_disabled=true",
  ):
    if flag.split("=", 1)[0] not in libtpu_args:
      libtpu_args = f"{libtpu_args} {flag}".strip()
  env["LIBTPU_INIT_ARGS"] = libtpu_args
  return env


def _build_local_weights(rank: int, device: torch.device):
  """Allocate deterministic FP8 local expert weights and scales for `rank`.

  Matches `vllm_torchtpu` `fused_moe_gmm` weight and scale layout for
  Qwen3.5-397B-A17B-FP8:
    w1: float8_e4m3fn [E_LOCAL, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE]
    w2: float8_e4m3fn [E_LOCAL, INTERMEDIATE_SIZE, HIDDEN_SIZE]
    w1_scale: float32 [E_LOCAL, 1, 1, 2 * INTERMEDIATE_SIZE]
    w2_scale: float32 [E_LOCAL, 1, 1, HIDDEN_SIZE]
  """
  gen = torch.Generator(device="cpu").manual_seed(1000 + rank)
  w1_f32 = (
      torch.randn(
          (E_LOCAL, HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE),
          generator=gen,
          dtype=torch.float32,
      )
      * 0.1
  )
  w2_f32 = (
      torch.randn(
          (E_LOCAL, INTERMEDIATE_SIZE, HIDDEN_SIZE),
          generator=gen,
          dtype=torch.float32,
      )
      * 0.1
  )

  w1_amax = w1_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
  w1_scale_cpu = (w1_amax / FP8_MAX).to(torch.float32)  # [E_LOCAL, 1, 2*I]
  w1_q_cpu = (
      (w1_f32 / w1_scale_cpu)
      .clamp(-FP8_MAX, FP8_MAX)
      .to(torch.float8_e4m3fn)
      .contiguous()
  )
  w1_scale_4d_cpu = w1_scale_cpu.unsqueeze(
      2
  ).contiguous()  # [E_LOCAL, 1, 1, 2*I]

  w2_amax = w2_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
  w2_scale_cpu = (w2_amax / FP8_MAX).to(torch.float32)  # [E_LOCAL, 1, H]
  w2_q_cpu = (
      (w2_f32 / w2_scale_cpu)
      .clamp(-FP8_MAX, FP8_MAX)
      .to(torch.float8_e4m3fn)
      .contiguous()
  )
  w2_scale_4d_cpu = w2_scale_cpu.unsqueeze(2).contiguous()  # [E_LOCAL, 1, 1, H]

  return (
      w1_q_cpu.to(device),
      w2_q_cpu.to(device),
      w1_scale_4d_cpu.to(device),
      w2_scale_4d_cpu.to(device),
  )


def _build_local_inputs(rank: int, num_local_tokens: int, device: torch.device):
  """Create local token activations and router top-k assignments for `rank`."""
  gen = torch.Generator(device="cpu").manual_seed(
      5000 + num_local_tokens * 31 + rank
  )
  hidden_local_cpu = (
      torch.randn(
          (num_local_tokens, HIDDEN_SIZE), generator=gen, dtype=torch.float32
      )
      * 0.5
  ).to(torch.bfloat16)

  router_logits_cpu = torch.randn(
      (num_local_tokens, E_GLOBAL), generator=gen, dtype=torch.float32
  )
  scores_cpu = torch.softmax(router_logits_cpu, dim=-1)
  topk_weights_cpu, topk_ids_cpu = torch.topk(scores_cpu, k=TOPK, dim=-1)
  topk_weights_cpu = (
      topk_weights_cpu / topk_weights_cpu.sum(dim=-1, keepdim=True)
  ).to(torch.bfloat16)
  topk_ids_cpu = topk_ids_cpu.to(torch.int32)

  return (
      hidden_local_cpu.to(device),
      topk_weights_cpu.to(device),
      topk_ids_cpu.to(device),
  )


def _sync_tensor(t) -> None:
  """Materializes `t` (tensor or sequence of tensors) and waits for completion."""
  from torch_tpu._internal import sync as tpu_sync  # pylint: disable=g-import-not-at-top

  if isinstance(t, torch.Tensor):
    tpu_sync.synchronize([t], wait=True)
  else:
    tpu_sync.synchronize(list(t), wait=True)


def _sync_barrier(ep_group, device: torch.device) -> None:
  """Synchronous device + host barrier across all ranks in `ep_group`.

  Note: On `backend="tpu_dist"`, `torch.distributed.barrier()` enqueues a lazy
  TPU `all_reduce` and returns immediately on the host CPU without waiting for
  the TPU collective to finish. Using `ep_group.all_gather` followed by `.cpu()`
  blocks the host CPU thread until all 8 TPU ranks complete all prior device
  work and complete the ICI collective.
  """
  token = torch.ones((1,), dtype=torch.int32, device=device)
  gathered = ep_group.all_gather(token, dim=0)
  _sync_tensor(gathered)
  _ = gathered.cpu()


def run_ep8_fused_moe(
    moe_fn,
    hidden_states_local: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights_local: torch.Tensor,
    topk_ids_local: torch.Tensor,
    experts_start: torch.Tensor,
    ep_group,
) -> torch.Tensor:
  """Run EP8 + SP / Attention DP MoE flow using `moe_fn` (`fused_moe_gmm` or `fused_moe_tpuc`)."""
  ag_hidden_states = ep_group.all_gather(hidden_states_local, dim=0)
  ag_topk_weights = ep_group.all_gather(topk_weights_local, dim=0)
  ag_topk_ids = ep_group.all_gather(topk_ids_local, dim=0)

  out_partial = moe_fn(
      hidden_states=ag_hidden_states,
      w1=w1,
      w2=w2,
      w1_scale=w1_scale,
      w2_scale=w2_scale,
      w1_bias=None,
      w2_bias=None,
      topk_weights=ag_topk_weights,
      topk_ids=ag_topk_ids,
      experts_start=experts_start,
      topk=TOPK,
      activation="silu",
      rhs_quant_dtype=None,
      skip_padded_tokens=False,
  )

  out_local = ep_group.reduce_scatter(out_partial, dim=0)
  return out_local


def _compute_parity_metrics(
    cand: torch.Tensor, ref: torch.Tensor
) -> dict[str, float]:
  cand_f32 = cand.to(torch.float32)
  ref_f32 = ref.to(torch.float32)
  diff = (cand_f32 - ref_f32).abs()
  max_abs_err = diff.max().item()
  mean_abs_err = diff.mean().item()
  rel_l2_err = (
      torch.linalg.vector_norm(cand_f32 - ref_f32)
      / (torch.linalg.vector_norm(ref_f32) + 1e-12)
  ).item()
  cos_sim = torch.nn.functional.cosine_similarity(
      cand_f32.flatten(), ref_f32.flatten(), dim=0
  ).item()
  return {
      "cos_sim": cos_sim,
      "rel_l2_err": rel_l2_err,
      "max_abs_err": max_abs_err,
      "mean_abs_err": mean_abs_err,
  }


def _stage_times() -> list[dict[str, float]] | None:
  """Returns `fused_moe_tpuc_layer.STAGE_TIMES` (TPU_MOE_STAGE_TIMES=1)."""
  if os.environ.get("TPU_MOE_STAGE_TIMES") != "1":
    return None
  mod = sys.modules.get(getattr(fused_moe_tpuc, "__module__", ""))
  return getattr(mod, "STAGE_TIMES", None)


def _stage_medians(records: list[dict[str, float]]) -> str:
  if not records:
    return "<none>"
  return ", ".join(
      f"{k}={statistics.median(r[k] for r in records):.3f}" for k in records[0]
  )


def _benchmark_median_ms(
    step_fn,
    ep_group,
    device: torch.device,
    *,
    warmup: int = WARMUP_RUNS,
    runs: int = BENCH_RUNS,
) -> tuple[float, list[float]]:
  """Runs `warmup` untimed iterations followed by `runs` timed iterations."""
  _sync_barrier(ep_group, device)
  for _ in range(warmup):
    out = step_fn()
    _sync_tensor(out)
  _sync_barrier(ep_group, device)

  times_ms: list[float] = []
  for _ in range(runs):
    _sync_barrier(ep_group, device)
    t0 = time.perf_counter()
    out = step_fn()
    _sync_tensor(out)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    times_ms.append(elapsed_ms)
  _sync_barrier(ep_group, device)
  return statistics.median(times_ms), times_ms


def _run_worker(result_dir: Path) -> None:
  """Worker process executed on each of the 8 TPU ranks."""
  # Dump the Python stack of every thread on SIGSEGV/SIGABRT for diagnosis.
  faulthandler.enable(file=sys.stderr, all_threads=True)
  rank = int(os.environ["RANK"])
  world = int(os.environ["WORLD_SIZE"])
  local_rank = int(os.environ.get("LOCAL_RANK", rank))
  result_path = result_dir / f"rank_{rank}.json"
  result: dict[str, object] = {"rank": rank, "world": world, "tokens": {}}
  initialized = False

  try:
    dist.init_process_group(backend="tpu_dist")
    initialized = True
    device = torch.device("tpu")
    torch.empty((1,), device=device).cpu()

    parallel_state.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="tpu_dist",
    )
    world_group = parallel_state.get_world_group()
    ep_group = parallel_state.init_model_parallel_group(
        [list(range(world))],
        world_group.local_rank,
        dist.get_backend(world_group.device_group),
        group_name="ep8_qwen35_moe_test",
    )

    if rank == 0:
      print(
          f"[Rank 0] Initialized EP{world} group. Allocating Qwen 3.5 FP8"
          f" weights (E_global={E_GLOBAL}, E_local={E_LOCAL}, H={HIDDEN_SIZE},"
          f" I={INTERMEDIATE_SIZE}, topk={TOPK})...",
          flush=True,
      )

    w1, w2, w1_scale, w2_scale = _build_local_weights(rank, device)
    experts_start_val = rank * E_LOCAL
    experts_start = torch.tensor(
        experts_start_val, dtype=torch.int32, device=device
    )
    _sync_tensor([w1, w2, w1_scale, w2_scale, experts_start])

    token_reports = {}
    for num_local_tokens in LOCAL_TOKEN_COUNTS:
      _sync_barrier(ep_group, device)
      num_global_tokens = num_local_tokens * world
      if rank == 0:
        print(
            f"\n--- Testing & Benchmarking T_local={num_local_tokens}"
            f" (T_global={num_global_tokens}) ---",
            flush=True,
        )

      hidden_local, topk_weights_local, topk_ids_local = _build_local_inputs(
          rank, num_local_tokens, device
      )
      _sync_tensor([hidden_local, topk_weights_local, topk_ids_local])

      # 1. Pre-gather inputs for standalone kernel parity & benchmarking
      ag_hidden = ep_group.all_gather(hidden_local, dim=0)
      ag_weights = ep_group.all_gather(topk_weights_local, dim=0)
      ag_ids = ep_group.all_gather(topk_ids_local, dim=0)
      _sync_tensor([ag_hidden, ag_weights, ag_ids])
      _sync_barrier(ep_group, device)

      ref_partial = fused_moe_gmm(
          hidden_states=ag_hidden,
          w1=w1,
          w2=w2,
          w1_scale=w1_scale,
          w2_scale=w2_scale,
          w1_bias=None,
          w2_bias=None,
          topk_weights=ag_weights,
          topk_ids=ag_ids,
          experts_start=experts_start,
          topk=TOPK,
          activation="silu",
          rhs_quant_dtype=None,
          skip_padded_tokens=False,
      )
      _sync_tensor(ref_partial)
      ref_partial_cpu = ref_partial.cpu()
      _sync_barrier(ep_group, device)

      tpuc_partial = fused_moe_tpuc(
          hidden_states=ag_hidden,
          w1=w1,
          w2=w2,
          w1_scale=w1_scale,
          w2_scale=w2_scale,
          w1_bias=None,
          w2_bias=None,
          topk_weights=ag_weights,
          topk_ids=ag_ids,
          experts_start=experts_start,
          topk=TOPK,
          activation="silu",
          rhs_quant_dtype=None,
          skip_padded_tokens=False,
      )
      _sync_tensor(tpuc_partial)
      tpuc_partial_cpu = tpuc_partial.cpu()
      _sync_barrier(ep_group, device)

      # 2. Run full EP8 E2E (all_gather -> moe_fn -> reduce_scatter) parity
      ref_local = run_ep8_fused_moe(
          fused_moe_gmm,
          hidden_states_local=hidden_local,
          w1=w1,
          w2=w2,
          w1_scale=w1_scale,
          w2_scale=w2_scale,
          topk_weights_local=topk_weights_local,
          topk_ids_local=topk_ids_local,
          experts_start=experts_start,
          ep_group=ep_group,
      )
      _sync_tensor(ref_local)
      ref_local_cpu = ref_local.cpu()
      _sync_barrier(ep_group, device)

      tpuc_local = run_ep8_fused_moe(
          fused_moe_tpuc,
          hidden_states_local=hidden_local,
          w1=w1,
          w2=w2,
          w1_scale=w1_scale,
          w2_scale=w2_scale,
          topk_weights_local=topk_weights_local,
          topk_ids_local=topk_ids_local,
          experts_start=experts_start,
          ep_group=ep_group,
      )
      _sync_tensor(tpuc_local)
      tpuc_local_cpu = tpuc_local.cpu()
      _sync_barrier(ep_group, device)

      assert tpuc_local_cpu.shape == (num_local_tokens, HIDDEN_SIZE)
      assert tpuc_partial_cpu.shape == (num_global_tokens, HIDDEN_SIZE)
      assert tpuc_local_cpu.dtype == torch.bfloat16
      assert torch.isfinite(tpuc_local_cpu).all(), (
          f"Non-finite values in tpuc_local on rank {rank} at"
          f" T_local={num_local_tokens}"
      )
      assert torch.isfinite(tpuc_partial_cpu).all(), (
          f"Non-finite values in tpuc_partial on rank {rank} at"
          f" T_local={num_local_tokens}"
      )

      partial_parity = _compute_parity_metrics(
          tpuc_partial_cpu, ref_partial_cpu
      )
      local_parity = _compute_parity_metrics(tpuc_local_cpu, ref_local_cpu)

      # Note: fused_moe_tpuc uses W8A8 (per-token dynamic float8_e4m3fn
      # activation quantization, 3 mantissa bits) whereas fused_moe_gmm uses
      # W8A16 (bfloat16 activations), yielding ~0.9983 cosine similarity and
      # ~0.058 relative L2 difference (sqrt(2 * (1 - 0.9983)) = 0.0583).
      assert partial_parity["cos_sim"] > 0.997, (
          f"Rank {rank} T_local={num_local_tokens} partial cos_sim too low:"
          f" {partial_parity}"
      )
      assert partial_parity["rel_l2_err"] < 0.075, (
          f"Rank {rank} T_local={num_local_tokens} partial rel_l2_err too high:"
          f" {partial_parity}"
      )
      assert local_parity["cos_sim"] > 0.997, (
          f"Rank {rank} T_local={num_local_tokens} local cos_sim too low:"
          f" {local_parity}"
      )
      assert local_parity["rel_l2_err"] < 0.075, (
          f"Rank {rank} T_local={num_local_tokens} local rel_l2_err too high:"
          f" {local_parity}"
      )

      # 3. Benchmark standalone kernel (pre-gathered inputs) and full EP8 layer
      ref_kernel_median_ms, _ = _benchmark_median_ms(
          lambda: fused_moe_gmm(
              hidden_states=ag_hidden,
              w1=w1,
              w2=w2,
              w1_scale=w1_scale,
              w2_scale=w2_scale,
              w1_bias=None,
              w2_bias=None,
              topk_weights=ag_weights,
              topk_ids=ag_ids,
              experts_start=experts_start,
              topk=TOPK,
              activation="silu",
              rhs_quant_dtype=None,
              skip_padded_tokens=False,
          ),
          ep_group,
          device,
      )
      stage_times = _stage_times()
      if stage_times is not None:
        stage_times.clear()
      tpuc_kernel_median_ms, _ = _benchmark_median_ms(
          lambda: fused_moe_tpuc(
              hidden_states=ag_hidden,
              w1=w1,
              w2=w2,
              w1_scale=w1_scale,
              w2_scale=w2_scale,
              w1_bias=None,
              w2_bias=None,
              topk_weights=ag_weights,
              topk_ids=ag_ids,
              experts_start=experts_start,
              topk=TOPK,
              activation="silu",
              rhs_quant_dtype=None,
              skip_padded_tokens=False,
          ),
          ep_group,
          device,
      )

      if stage_times is not None and rank == 0:
        # Drop the warmup calls; report medians of the timed runs (ms).
        print(
            f"  [Rank {rank}] T_local={num_local_tokens} tpuc stage medians"
            f" (ms): {_stage_medians(stage_times[-BENCH_RUNS:])}",
            flush=True,
        )
      ref_e2e_median_ms, _ = _benchmark_median_ms(
          lambda: run_ep8_fused_moe(
              fused_moe_gmm,
              hidden_states_local=hidden_local,
              w1=w1,
              w2=w2,
              w1_scale=w1_scale,
              w2_scale=w2_scale,
              topk_weights_local=topk_weights_local,
              topk_ids_local=topk_ids_local,
              experts_start=experts_start,
              ep_group=ep_group,
          ),
          ep_group,
          device,
      )
      tpuc_e2e_median_ms, _ = _benchmark_median_ms(
          lambda: run_ep8_fused_moe(
              fused_moe_tpuc,
              hidden_states_local=hidden_local,
              w1=w1,
              w2=w2,
              w1_scale=w1_scale,
              w2_scale=w2_scale,
              topk_weights_local=topk_weights_local,
              topk_ids_local=topk_ids_local,
              experts_start=experts_start,
              ep_group=ep_group,
          ),
          ep_group,
          device,
      )

      if rank == 0:
        print(
            f"  [Rank 0] T_local={num_local_tokens:4d}"
            f" (T_global={num_global_tokens:5d}) Parity:"
            f" local_cos_sim={local_parity['cos_sim']:.6f},"
            f" local_rel_l2={local_parity['rel_l2_err']:.6f},"
            f" partial_cos_sim={partial_parity['cos_sim']:.6f},"
            f" partial_rel_l2={partial_parity['rel_l2_err']:.6f}",
            flush=True,
        )
        print(
            f"  [Rank 0] T_local={num_local_tokens:4d} Benchmark (median of"
            f" {BENCH_RUNS} runs after {WARMUP_RUNS} warmup):"
            f" Kernel [gmm={ref_kernel_median_ms:.3f} ms vs"
            f" tpuc={tpuc_kernel_median_ms:.3f} ms] |"
            f" EP8 E2E [gmm={ref_e2e_median_ms:.3f} ms vs"
            f" tpuc={tpuc_e2e_median_ms:.3f} ms]",
            flush=True,
        )

      token_reports[str(num_local_tokens)] = {
          "num_local_tokens": num_local_tokens,
          "num_global_tokens": num_global_tokens,
          "local_parity": local_parity,
          "partial_parity": partial_parity,
          "ref_kernel_median_ms": ref_kernel_median_ms,
          "tpuc_kernel_median_ms": tpuc_kernel_median_ms,
          "ref_e2e_median_ms": ref_e2e_median_ms,
          "tpuc_e2e_median_ms": tpuc_e2e_median_ms,
      }

    result["tokens"] = token_reports
  except BaseException as exc:  # pylint: disable=broad-exception-caught
    traceback.print_exc()
    result["error"] = repr(exc)
  finally:
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if initialized:
      try:
        dist.destroy_process_group()
      except Exception:  # pylint: disable=broad-exception-caught
        traceback.print_exc()

  if "error" in result:
    raise SystemExit(1)


def _launch_workers() -> None:
  """Spawn 8-rank `torch.distributed.run` workers and verify all results."""
  with tempfile.TemporaryDirectory(prefix="qwen35_moe_ep8_") as tmp_dir:
    result_dir = Path(tmp_dir)
    cmd = [
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
    ]
    print(
        "=== Launching 8-Rank EP8 Qwen 3.5 MoE Parity & Benchmark Test"
        f" (T_local={list(LOCAL_TOKEN_COUNTS)}) ===",
        flush=True,
    )
    env = _prepare_worker_env()
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
      assert process.stdout is not None
      for line in process.stdout:
        print(line, end="", flush=True)
      process.wait(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
      _dump_tpu_error_logs()
      os.killpg(process.pid, signal.SIGTERM)
      process.wait(timeout=60)
      raise AssertionError(
          f"EP8 worker job timed out after {LAUNCH_TIMEOUT_SECONDS}s"
      ) from None

    if process.returncode != 0:
      _dump_tpu_error_logs()
      raise AssertionError(
          f"EP8 worker job failed with exit code {process.returncode}"
      )

    results = []
    for rank in range(WORLD_SIZE):
      path = result_dir / f"rank_{rank}.json"
      if not path.exists():
        raise AssertionError(f"Missing result file for rank {rank}: {path}")
      results.append(json.loads(path.read_text(encoding="utf-8")))

    errors = [r for r in results if "error" in r]
    assert not errors, f"Worker failures: {errors}"

    print(
        "\n=== Parity & Benchmark Summary Across All 8 EP Ranks ===", flush=True
    )
    header = (
        f"{'T_local':>7} | {'T_global':>8} | {'Min CosSim':>10} | "
        f"{'Max RelL2':>9} | {'GMM Kernel (ms)':>15} | "
        f"{'TPU-C Kernel (ms)':>17} | {'GMM EP8 E2E (ms)':>16} | "
        f"{'TPU-C EP8 E2E (ms)':>18}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for num_local_tokens in LOCAL_TOKEN_COUNTS:
      key = str(num_local_tokens)
      min_cos = min(
          r["tokens"][key]["local_parity"]["cos_sim"] for r in results
      )
      max_rel_l2 = max(
          r["tokens"][key]["local_parity"]["rel_l2_err"] for r in results
      )
      gmm_k_ms = results[0]["tokens"][key]["ref_kernel_median_ms"]
      tpuc_k_ms = results[0]["tokens"][key]["tpuc_kernel_median_ms"]
      gmm_e2e_ms = results[0]["tokens"][key]["ref_e2e_median_ms"]
      tpuc_e2e_ms = results[0]["tokens"][key]["tpuc_e2e_median_ms"]
      print(
          f"{num_local_tokens:7d} | {num_local_tokens * WORLD_SIZE:8d} | "
          f"{min_cos:10.6f} | {max_rel_l2:9.6f} | "
          f"{gmm_k_ms:15.3f} | {tpuc_k_ms:17.3f} | "
          f"{gmm_e2e_ms:16.3f} | {tpuc_e2e_ms:18.3f}",
          flush=True,
      )
    print(
        "\n🎉 ALL QWEN 3.5 EP8 FUSED_MOE_TPUC PARITY & BENCHMARK TESTS PASSED!",
        flush=True,
    )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("--result-dir", type=Path, default=None)
  args = parser.parse_args()

  if args.worker:
    assert args.result_dir is not None
    _run_worker(args.result_dir)
    return

  _launch_workers()


if __name__ == "__main__":
  main()
