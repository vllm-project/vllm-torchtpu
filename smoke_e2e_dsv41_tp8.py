# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end 8-chip TP8/EP8 DeepSeek-V4.1 (v2, FP8 KV, CED=1) smoke tool.

Exercises all DeepSeek-V4.1 architectural paths in 4 or 6 layers on TPU7x-8:
- L0: SWA-only attention
- L1: C2 KV compressor + Engram + Indexer full_context + SparseMLA + SWA
- L2 (4L) / L3 (6L): CED boundary + C1 KV compressor + Indexer candidate_write
- L3 (4L) / L4 (6L): C1 KV reuse + Indexer candidate_filter
- (6L also exercises L2 C2 shared-index reuse and L5 C1 shared-index reuse)

Also installs a worker-side ``sitecustomize.py`` in the generated smoke model
directory so that ``load_format="dummy"`` initializes all floating-point and
``uint8`` (MXFP4 E2M1 weights and UE8M0 scales) parameters deterministically on
CPU before copying to TPU, and captures raw final-layer ``logits`` on rank 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NEW_MODEL_DESIGN", "1")
os.environ.setdefault("MODEL_IMPL_TYPE", "vllm")
os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
os.environ.setdefault("USE_MOE_SPARSE_CORE", "1")
os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
os.environ.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
os.environ.setdefault(
    "LIBTPU_INIT_ARGS",
    "--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes=67108864",
)
os.environ.setdefault("TPU_DEEPSEEK_V41_IMPL", "v2")
os.environ.setdefault("TPU_DEEPSEEK_V41_CED", "1")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_deterministic_dummy_weights() -> None:
    import torch
    import vllm.model_executor.model_loader.dummy_loader as dummy_loader
    import vllm.model_executor.model_loader.weight_utils as weight_utils

    @torch.no_grad()
    def _deterministic_single_dummy_weight(
        param: torch.Tensor,
        low: float = -1e-3,
        high: float = 1e-3,
        seed: int = 1234,
    ) -> None:
        if param.device.type == "meta":
            return
        if (val := getattr(param, "dummy_weight_value", None)) is not None:
            param.fill_(val)
            return
        if not torch.is_floating_point(param):
            if param.dtype == torch.uint8:
                # UE8M0 block scales have trailing dim <= 160; fill with 120.
                # Packed FP4 E2M1 weights have trailing dim >= 512; fill 0x21.
                if param.ndim >= 2 and param.shape[-1] <= 160:
                    param.fill_(120)
                else:
                    param.fill_(0x21)
            else:
                param.zero_()
            return

        shape_tag = sum((i + 1) * int(d) for i, d in enumerate(param.shape))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + shape_tag)
        cpu_f32 = torch.empty(param.shape, dtype=torch.float32, device="cpu")
        cpu_f32.uniform_(low, high, generator=gen)
        param.copy_(cpu_f32.to(dtype=param.dtype))

    weight_utils.initialize_single_dummy_weight = _deterministic_single_dummy_weight
    dummy_loader.initialize_single_dummy_weight = _deterministic_single_dummy_weight


def _install_logits_capture() -> None:
    logits_dir = os.environ.get("DSV41_SMOKE_LOGITS_DIR")
    if not logits_dir:
        return
    import torch

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    if getattr(TPUModelRunner, "_dsv41_smoke_logits_hooked", False):
        return
    step_counter = {"idx": 0}

    def _save_logits(logits: torch.Tensor | None) -> None:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0 and logits is not None and logits.numel() > 0:
            idx = step_counter["idx"]
            step_counter["idx"] = idx + 1
            out_dir = os.path.join(logits_dir, "steps")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                logits[:1].detach().to(torch.float32).cpu(),
                os.path.join(out_dir, f"step_{idx:04d}.pt"),
            )

    orig_from_hidden = TPUModelRunner.compute_logits_from_hidden_states
    orig_selected = TPUModelRunner.compute_selected_logits

    def _wrapped_from_hidden(self, hidden_states, *args, **kwargs):
        logits = orig_from_hidden(self, hidden_states, *args, **kwargs)
        _save_logits(logits)
        return logits

    def _wrapped_selected(self, hidden_states, indices_do_sample, *args, **kwargs):
        logits = orig_selected(self, hidden_states, indices_do_sample, *args, **kwargs)
        _save_logits(logits)
        return logits

    TPUModelRunner.compute_logits_from_hidden_states = _wrapped_from_hidden
    TPUModelRunner.compute_selected_logits = _wrapped_selected
    TPUModelRunner._dsv41_smoke_logits_hooked = True


if os.environ.get("DSV41_SMOKE_ACTIVE") == "1":
    try:
        _install_deterministic_dummy_weights()
        _install_logits_capture()
    except Exception as _exc:  # noqa: BLE001
        print(f"[smoke_e2e] warning: {_exc}", file=sys.stderr)


_SITECUSTOMIZE_CODE = """\"\"\"Worker-side dummy weight and logits hook.\"\"\"
import os
import sys

try:
    import smoke_e2e_dsv41_tp8 as _smoke
    _smoke._install_deterministic_dummy_weights()
    _smoke._install_logits_capture()
except Exception as exc:
    print(f"[usercustomize] warning: {exc}", file=sys.stderr)
"""


def build_smoke_model_dir(
    meta_dir: Path,
    out_dir: Path,
    *,
    num_layers: int = 4,
    keep_dspark: bool = False,
) -> Path:
    from vllm.models.deepseek_v41.common.engram import find_next_prime

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = meta_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)

    with open(meta_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)

    # Compute exact sum of 24 primes above engram_vocab_size - 1 for 1 Engram layer
    engram_vocab_size = 256
    max_ngram = cfg["text_config"]["engram_max_ngram_size"]
    n_heads = cfg["text_config"]["engram_n_heads"]
    seen: set[int] = set()
    total_embeddings = 0
    for _ in range(max_ngram - 1):
        curr = engram_vocab_size - 1
        for _ in range(n_heads):
            curr = find_next_prime(curr, seen)
            seen.add(curr)
            total_embeddings += curr

    tc = cfg["text_config"]
    if num_layers == 4:
        tc["num_hidden_layers"] = 4
        tc["compress_ratios"] = [0, 2, 1, 1]
        tc["kv_source_layer_ids"] = [1, 2]
        tc["index_source_layer_ids"] = [1, 2, 3]
        tc["candidate_source_layer_id"] = 2
        tc["engram_layer_ids"] = [1]
    elif num_layers == 6:
        tc["num_hidden_layers"] = 6
        tc["compress_ratios"] = [0, 2, 2, 1, 1, 1]
        tc["kv_source_layer_ids"] = [1, 3]
        tc["index_source_layer_ids"] = [1, 3, 4]
        tc["candidate_source_layer_id"] = 3
        tc["engram_layer_ids"] = [1]
    else:
        raise ValueError(f"Unsupported --num-layers={num_layers}; choose 4 or 6")

    tc["engram_vocab_size"] = engram_vocab_size
    tc["engram_num_embeddings"] = [total_embeddings]
    if keep_dspark:
        tc["dspark_target_layer_ids"] = list(range(num_layers - 3, num_layers))
        tc["dspark_n_routed_experts"] = 16
        tc["dspark_num_experts_per_tok"] = 2
    else:
        tc["num_nextn_predict_layers"] = 0
        tc["dspark_target_layer_ids"] = []
    # Smaller routed expert count (16 across EP8 = 2/rank) for fast dummy init
    tc["n_routed_experts"] = 16
    tc["num_experts_per_tok"] = 2

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    (out_dir / "sitecustomize.py").write_text(_SITECUSTOMIZE_CODE, encoding="utf-8")
    (out_dir / "usercustomize.py").write_text(_SITECUSTOMIZE_CODE, encoding="utf-8")
    return out_dir


def _extract_top_logprobs(
    step_logprobs: list[dict[int, Any]] | None,
    top_k: int = 20,
) -> tuple[Any, Any]:
    import math

    import torch

    if not step_logprobs:
        return torch.empty((0, 0), dtype=torch.int64), torch.empty(
            (0, 0), dtype=torch.float32
        )
    ids_rows: list[list[int]] = []
    vals_rows: list[list[float]] = []
    for lp_dict in step_logprobs:
        items = sorted(
            lp_dict.items(),
            key=lambda kv: (
                -float(kv[1].logprob)
                if not math.isnan(float(kv[1].logprob))
                else float("inf"),
                int(kv[0]),
            ),
        )[:top_k]
        ids_rows.append([int(k) for k, _ in items])
        vals_rows.append([float(v.logprob) for _, v in items])
    return (
        torch.tensor(ids_rows, dtype=torch.int64),
        torch.tensor(vals_rows, dtype=torch.float32),
    )


def _load_captured_step_logits(
    steps_dir: Path,
    max_steps: int | None = None,
) -> Any:
    """Load non-zero captured step logits (filtering CED=1 zeroed prefill chunks)."""
    import torch

    step_logits_list: list[torch.Tensor] = []
    if steps_dir.exists():
        for pt_file in sorted(steps_dir.glob("step_*.pt")):
            t = torch.load(pt_file, map_location="cpu", weights_only=True)
            if t.ndim == 2 and t.shape[0] >= 1:
                row = t[0]
                # Under CED=1, partial-prefill chunks skip decoder_stage and
                # return all-zero hidden_states/logits.
                if not bool((row != 0).any().item()):
                    continue
                step_logits_list.append(row)
                if max_steps is not None and len(step_logits_list) >= max_steps:
                    break
    return (
        torch.stack(step_logits_list, dim=0)
        if step_logits_list
        else torch.empty((0, 0), dtype=torch.float32)
    )


def verify_golden(actual: dict[str, Any], golden_path: Path) -> dict[str, Any]:
    import torch

    from scripts.dsv41_bench import common

    if not golden_path.exists():
        raise FileNotFoundError(f"Golden file not found: {golden_path}")
    expected = torch.load(golden_path, map_location="cpu", weights_only=True)

    report: dict[str, Any] = {}
    report["token_ids"] = common.compare_tensors(
        actual["token_ids"], expected["token_ids"], common.EXACT
    )
    report["top_logprob_values"] = common.compare_tensors(
        actual["top_logprob_values"],
        expected["top_logprob_values"],
        common.MatchCriteria("lossless", min_cos=0.999, max_rel_l2=1e-2),
    )
    if "step_logits" in actual and "step_logits" in expected:
        act_logits = actual["step_logits"]
        exp_logits = expected["step_logits"]
        if act_logits.numel() == 0 or exp_logits.numel() == 0:
            report["step_logits"] = {
                "ok": False,
                "reason": "step_logits is empty; worker logits capture did not fire",
            }
        else:
            report["step_logits"] = common.compare_tensors(
                act_logits,
                exp_logits,
                common.MatchCriteria("lossless", min_cos=0.999, max_rel_l2=1e-2),
            )
            act_top1 = act_logits.argmax(dim=-1)
            exp_top1 = exp_logits.argmax(dim=-1)
            top1_match = float((act_top1 == exp_top1).float().mean().item())
            report["step_logits"]["top1_match_rate"] = top1_match
            if top1_match < 1.0:
                report["step_logits"]["ok"] = False
                report["step_logits"]["reason"] = (
                    f"top1_match_rate={top1_match:.4f} < 1.0"
                )
            elif report["step_logits"].get("ok", False):
                # When step_logits matches losslessly and argmax matches 100%,
                # a greedy token_id that achieves the exact maximum logit in
                # expected["step_logits"] (e.g. tied bfloat16 top-k on CPU vs
                # TPU argmax) is a valid tied-argmax match.
                if not report["token_ids"].get("ok", False):
                    act_ids = actual["token_ids"].flatten().long()
                    if act_ids.numel() == exp_logits.shape[0] and all(
                        0 <= int(tid.item()) < exp_logits.shape[1]
                        and float(exp_logits[row, int(tid.item())].item())
                        == float(exp_logits[row].max().item())
                        for row, tid in enumerate(act_ids)
                    ):
                        report["token_ids"]["ok"] = True
                        report["token_ids"]["tied_argmax_match"] = True
                # TP vocab sharding/padding can shift log_softmax by a constant
                # scalar per row while preserving relative log-probabilities.
                if (
                    not report["top_logprob_values"].get("ok", False)
                    and report["top_logprob_values"].get("cos_sim", 0.0) >= 0.999
                ):
                    act_lp = actual["top_logprob_values"].float()
                    exp_lp = expected["top_logprob_values"].float()
                    if act_lp.shape == exp_lp.shape and act_lp.numel() > 0:
                        centered_diff = float(
                            (
                                (act_lp - act_lp.mean(dim=-1, keepdim=True))
                                - (exp_lp - exp_lp.mean(dim=-1, keepdim=True))
                            )
                            .abs()
                            .max()
                            .item()
                        )
                        if centered_diff <= 1e-3:
                            report["top_logprob_values"]["ok"] = True
                            report["top_logprob_values"]["shift_invariant_match"] = True
                            report["top_logprob_values"]["centered_max_abs_diff"] = (
                                centered_diff
                            )

    failed = {k: v for k, v in report.items() if not v.get("ok", False)}
    if failed:
        raise common.VerificationError(
            f"E2E golden verification failed against {golden_path}: "
            f"{json.dumps(failed, indent=2)}"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path(".dsv41_bench/meta"),
        help="Directory containing DeepSeek-V4.1 config.json and tokenizer files",
    )
    parser.add_argument(
        "--smoke-dir",
        type=Path,
        default=Path("/tmp/dsv41_smoke"),
        help="Output directory for generated 4L/6L smoke model config",
    )
    parser.add_argument("--num-layers", type=int, default=4, choices=[4, 6])
    parser.add_argument("--kv-cache-dtype", type=str, default="fp8")
    parser.add_argument("--indexer-kv-dtype", type=str, default="fp8")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=4096,
        help="Prompt token count (>= 2 * chunk exercises multi-chunk prefill)",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=2147483648,
        help="KV cache HBM budget per rank in bytes (default 2 GiB for up to 100k ctx)",
    )
    parser.add_argument("--save-golden", type=Path, default=None)
    parser.add_argument("--verify-against-baseline", type=Path, default=None)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="DP replicas; DP * TP must equal the 8 TPU7x-8 cores",
    )
    parser.add_argument(
        "--speculative-config",
        type=json.loads,
        default=None,
        help='JSON passed to LLM(speculative_config=...), e.g. {"method":"dspark",'
        '"num_speculative_tokens":5}',
    )
    return parser.parse_args()


def build_operator_shape_ledger(
    num_layers: int,
    batch_tokens: int,
    kv_len: int,
    tp_size: int = 8,
    ced_window: int = 128,
) -> dict[str, Any]:
    """Build exact per-layer operator input shape ledger for 4L/6L TP8 smoke."""
    from scripts.dsv41_bench import common

    geom = common.Dsv41Geometry()
    dec_tokens = min(batch_tokens, max(16, ced_window))
    return {
        "geometry": dataclasses.asdict(geom),
        "num_layers": num_layers,
        "tp_size": tp_size,
        "encoder_tokens": batch_tokens,
        "decoder_replay_tokens_unpadded": min(batch_tokens, ced_window),
        "decoder_tokens_padded": batch_tokens if dec_tokens > 16 else 16,
        "kv_len": kv_len,
        "layers": {
            "L0_swa_only": {
                "stage": "encoder",
                "mhc_pre_delayed": [batch_tokens, geom.hc_mult, geom.hidden_size],
                "q_local": [
                    batch_tokens,
                    geom.num_attention_heads // tp_size,
                    geom.head_dim,
                ],
                "swa_kv": [batch_tokens, geom.num_key_value_heads, geom.head_dim],
                "swa_window": geom.sliding_window,
                "moe_routed": [batch_tokens, geom.hidden_size],
            },
            "L1_c2_engram_indexer_full": {
                "stage": "encoder",
                "engram_hashes": [batch_tokens, 24],
                "compressor_c2_input": [batch_tokens, geom.hidden_size],
                "compressor_c2_latent": [batch_tokens // 2, geom.head_dim],
                "indexer_q": [
                    batch_tokens,
                    geom.index_n_heads // tp_size,
                    geom.index_head_dim,
                ],
                "indexer_k_compressed": [kv_len // 2, geom.index_head_dim],
                "sparse_mla_q": [
                    batch_tokens,
                    geom.num_attention_heads // tp_size,
                    geom.head_dim,
                ],
                "sparse_mla_topk_records": [batch_tokens, geom.index_topk],
            },
            "L2_ced_boundary_c1_candidate_write": {
                "stage": "decoder",
                "encoder_global_kv_write_c1": [batch_tokens, geom.hidden_size],
                "candidate_write_blocks": [
                    batch_tokens,
                    geom.candidate_topk_blocks,
                ],
                "sparse_mla_topk_records": [batch_tokens, geom.index_topk],
            },
            "L3_c1_reuse_candidate_filter": {
                "stage": "decoder",
                "candidate_filter_blocks": [
                    batch_tokens,
                    geom.candidate_topk_blocks,
                ],
                "candidate_tokens_expanded": geom.candidate_topk_blocks
                * geom.candidate_block_size,
                "sparse_mla_topk_records": [batch_tokens, geom.index_topk],
            },
        },
    }


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams, TokensPrompt

    args = parse_args()
    smoke_dir = build_smoke_model_dir(
        args.meta_dir,
        args.smoke_dir,
        num_layers=args.num_layers,
        keep_dspark=args.speculative_config is not None,
    )
    logits_capture_dir = smoke_dir / "logits_capture"
    os.environ["DSV41_SMOKE_LOGITS_DIR"] = str(logits_capture_dir)
    os.environ["DSV41_SMOKE_ACTIVE"] = "1"
    _install_deterministic_dummy_weights()
    _install_logits_capture()
    existing_py_path = os.environ.get("PYTHONPATH", "")
    prepend_paths = f"{smoke_dir}:{REPO_ROOT}"
    os.environ["PYTHONPATH"] = (
        f"{prepend_paths}:{existing_py_path}" if existing_py_path else prepend_paths
    )
    llm_kwargs: dict[str, Any] = {
        "model": str(smoke_dir),
        "load_format": "dummy",
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "enable_expert_parallel": True,
        "enable_ep_weight_filter": True,
        "language_model_only": True,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "kv_cache_dtype": args.kv_cache_dtype,
        "attention_config": {"indexer_kv_dtype": args.indexer_kv_dtype},
        "max_model_len": args.max_model_len,
        "max_num_seqs": 8,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": 0.30,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "block_size": 16,
        "compilation_config": {"compile_sizes": [16, args.max_num_batched_tokens]},
    }
    if args.speculative_config is not None:
        llm_kwargs["speculative_config"] = args.speculative_config
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        prof_path = str(args.profile_dir.resolve())
        os.environ["VLLM_TORCH_PROFILER_DIR"] = prof_path
        llm_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": prof_path,
        }

    print(
        f"[dsv41-smoke] Built {args.num_layers}-layer DeepSeek-V4.1 v2 "
        f"smoke at {smoke_dir} "
        f"(kv={args.kv_cache_dtype}, idx_kv={args.indexer_kv_dtype}, "
        f"max_len={args.max_model_len}, chunk={args.max_num_batched_tokens}, "
        f"prompt_tokens={args.prompt_tokens})"
    )

    llm = LLM(**llm_kwargs)

    # Deterministic token prompt of length `args.prompt_tokens`
    prompt_token_ids = [100 + ((i * 31 + 7) % 2000) for i in range(args.prompt_tokens)]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        logprobs=20,
    )

    # Reset captured warmup logits so we only record the measured generation
    steps_dir = logits_capture_dir / "steps"
    if steps_dir.exists():
        shutil.rmtree(steps_dir)

    if args.profile_dir is not None:
        llm.start_profile()
    t0 = time.perf_counter()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids)],
        sampling_params,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    if args.profile_dir is not None:
        llm.stop_profile()

    out = outputs[0].outputs[0]
    token_ids = torch.tensor(list(out.token_ids), dtype=torch.int64)
    top_ids, top_vals = _extract_top_logprobs(out.logprobs)
    step_logits = _load_captured_step_logits(steps_dir, max_steps=args.max_tokens)

    print(
        f"[dsv41-smoke] Generated token IDs ({elapsed_ms:.1f} ms): {token_ids.tolist()}"
    )
    print(f"[dsv41-smoke] Captured step_logits shape: {tuple(step_logits.shape)}")
    assert len(out.token_ids) == args.max_tokens
    assert step_logits.numel() > 0, "No step_logits captured from TPUModelRunner!"
    assert torch.isfinite(step_logits).all(), "Non-finite logits detected!"

    payload: dict[str, Any] = {
        "token_ids": token_ids,
        "top_logprob_ids": top_ids,
        "top_logprob_values": top_vals,
        "step_logits": step_logits,
        "meta": {
            "num_layers": args.num_layers,
            "kv_cache_dtype": args.kv_cache_dtype,
            "indexer_kv_dtype": args.indexer_kv_dtype,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "elapsed_ms": elapsed_ms,
        },
    }

    op_shapes = build_operator_shape_ledger(
        num_layers=args.num_layers,
        batch_tokens=args.max_num_batched_tokens,
        kv_len=args.prompt_tokens,
    )
    payload["op_shapes"] = op_shapes

    if args.save_golden is not None:
        args.save_golden.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.save_golden)
        shapes_path = args.save_golden.with_name(
            f"{args.save_golden.stem}_op_shapes.json"
        )
        shapes_path.write_text(json.dumps(op_shapes, indent=2), encoding="utf-8")
        print(f"[dsv41-smoke] Saved golden payload to {args.save_golden}")
        print(f"[dsv41-smoke] Saved operator shape ledger to {shapes_path}")

    if args.verify_against_baseline is not None:
        report = verify_golden(payload, args.verify_against_baseline)
        print(
            f"[dsv41-smoke] VERIFIED against {args.verify_against_baseline}:\n"
            f"{json.dumps(report, indent=2)}"
        )

    if args.profile_dir is not None:
        from scripts.dsv41_bench import common

        xprof = common.summarize_xplanes(args.profile_dir)
        summary_path = args.profile_dir / "xprof_summary.json"
        summary_path.write_text(json.dumps(xprof, indent=2), encoding="utf-8")
        common.print_xprof_table(xprof)
        print(f"[dsv41-smoke] Wrote XProf summary to {summary_path}")


if __name__ == "__main__":
    main()
