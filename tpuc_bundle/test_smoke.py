"""Smoke and numerical verification tests for Deepsea TPU-C MoE kernels.

Usage in Google3:
  blaze test
  //experimental/users/$USER/torchtpu_tpuc_cdk/torchtpu_tpuc_tests:test_smoke

Usage on Cloud TPU VM / CDK:
  source env.sh
  python3 test_smoke.py
"""

import importlib
import os

try:
  importlib.import_module("symbol_loader")
except ImportError:
  pass


def _user_pkg_prefix() -> str:
  if __package__ and ".torchtpu_tpuc_tests" in __package__:
    return __package__.rsplit(".torchtpu_tpuc_tests", 1)[0]
  path = os.path.abspath(__file__).replace("\\", "/")
  if "/experimental/users/" in path:
    user = path.split("/experimental/users/", 1)[1].split("/", 1)[0]
    return f"google3.experimental.users.{user}.torchtpu_tpuc_cdk"
  user = os.environ.get("USER") or os.environ.get("LOGNAME") or "hning"
  return f"google3.experimental.users.{user}.torchtpu_tpuc_cdk"


def _import_deepsea_torch_bridge():
  prefix = _user_pkg_prefix()
  for mod_name in (
      f"{prefix}.torchtpu_tpuc_interop.deepsea_torch_bridge",
      "deepsea_torch_bridge",
  ):
    try:
      return importlib.import_module(mod_name)
    except ImportError:
      continue
  raise ImportError("Could not import deepsea_torch_bridge")


def _import_moe_dropin():
  prefix = _user_pkg_prefix()
  for mod_name in (
      f"{prefix}.torchtpu_tpuc_kernels.moe.moe_dropin",
      "moe_dropin",
  ):
    try:
      return importlib.import_module(mod_name)
    except ImportError:
      continue
  raise ImportError("Could not import moe_dropin")


def test_1_registration():
  print("\n=== Test 1: Custom Op Registration ===")
  try:
    importlib.import_module("symbol_loader")
  except ImportError:
    pass

  import torch  # pylint: disable=g-import-not-at-top

  try:
    import torch_tpu  # pylint: disable=g-import-not-at-top,unused-import
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Note: torch_tpu import warning: {e}")

  _import_deepsea_torch_bridge()

  assert hasattr(
      torch.ops.deepsea_moe, "expert_ffn_out"
  ), "expert_ffn_out missing!"
  assert hasattr(torch.ops.deepsea_moe, "expert_ffn"), "expert_ffn missing!"
  assert hasattr(
      torch.ops.deepsea_moe, "expert_ffn_mxuonly"
  ), "expert_ffn_mxuonly missing!"
  assert hasattr(torch.ops.deepsea_moe, "describe"), "describe missing!"
  assert hasattr(
      torch.ops.deepsea_moe, "await_dispatch"
  ), "await_dispatch missing!"

  moe_ops = [op for op in dir(torch.ops.deepsea_moe) if not op.startswith("_")]

  print("Found torch.ops.deepsea_moe ops:", moe_ops)
  print("✓ Test 1 Passed: Ops registered successfully.")


def test_2_moe_kernel():
  print("\n=== Test 2: MoE Expert FFN Kernel Execution on TPU ===")
  import torch  # pylint: disable=g-import-not-at-top

  moe_dropin = _import_moe_dropin()

  device = torch.device("tpu:0")
  print(f"Allocating dummy tensors on {device}...")

  # Test configuration: E=64 local experts, tokens=16, top_k=8, H=4096, I=1024
  e_count = 64
  h_dim = 4096
  i_dim = 1024
  tokens = 16
  top_k = 8

  # Weights: FP8 (float8_e4m3fn), Scales: Float32
  w13 = torch.zeros(
      (e_count, h_dim, 2 * i_dim), dtype=torch.float8_e4m3fn, device=device
  )
  w2 = torch.zeros(
      (e_count, i_dim, h_dim), dtype=torch.float8_e4m3fn, device=device
  )
  s13 = torch.ones((e_count, 2 * i_dim), dtype=torch.float32, device=device)
  s2 = torch.ones((e_count, h_dim), dtype=torch.float32, device=device)

  w = moe_dropin.prepare_weights(w13, w2, s13, s2)
  print("Weights prepared successfully.")

  # Inputs
  hidden_states = torch.randn(
      (tokens, h_dim), dtype=torch.bfloat16, device=device
  )
  topk_ids = torch.randint(
      0, e_count, (tokens, top_k), dtype=torch.int32, device=device
  )
  topk_weights = (
      torch.ones((tokens, top_k), dtype=torch.float32, device=device) / top_k
  )

  print("Dispatching fused_moe_ep1 (eager mode)...")
  out = moe_dropin.fused_moe_ep1(
      hidden_states, topk_ids, topk_weights, w, compiled=False
  )
  out_cpu = out.cpu()
  print(f"Output shape: {out_cpu.shape}, dtype: {out_cpu.dtype}")
  assert out_cpu.shape == (
      tokens,
      h_dim,
  ), f"Expected {(tokens, h_dim)}, got {out_cpu.shape}"
  assert (
      out_cpu.dtype == torch.bfloat16
  ), f"Expected bfloat16, got {out_cpu.dtype}"
  print("✓ Test 2 Passed: MoE FFN kernel executed on device successfully.")


def _pytorch_moe_reference(
    hidden_states, topk_ids, topk_weights, w13, w2, s13, s2
):
  """PyTorch reference implementation matching the FP8 W8A8 + bf16 SwiGLU MoE math.

  For each routed (token t, slot k) -> expert e = topk_ids[t, k], weight rw =
  topk_weights[t, k]:
    1. Per-row FP8 E4M3 quantization of hidden_states[t] -> (x_q, xs)
    2. Gate/Up projection: acc1 = (x_q @ w13[e]) * xs * s13[e]
       (gate=cols[:I], up=cols[I:])
    3. SwiGLU intermediate in bf16: h = (silu(gate) * up).to(bfloat16)
    4. Down projection: acc2 = (h @ w2[e].to(bfloat16)) * s2[e]
    5. Weighted expert output rounded to bf16: out_row = (acc2 *
    rw).to(bfloat16)
    6. Token output: sum_{k} out_row(t, k).to(float32) -> bfloat16
  """
  import torch  # pylint: disable=g-import-not-at-top
  import torch.nn.functional as F  # pylint: disable=g-import-not-at-top

  moe_dropin = _import_moe_dropin()

  # Run reference arithmetic on CPU in float32/bfloat16 for exact verification
  x_cpu = hidden_states.detach().cpu()
  ids_cpu = topk_ids.detach().cpu().to(torch.int64)
  wts_cpu = topk_weights.detach().cpu().to(torch.float32)
  w13_cpu = w13.detach().cpu().to(torch.float32)
  w2_cpu = w2.detach().cpu().to(torch.bfloat16).to(torch.float32)
  s13_cpu = s13.detach().cpu().to(torch.float32)
  s2_cpu = s2.detach().cpu().to(torch.float32)

  tokens, h_dim = x_cpu.shape
  i_dim = w2_cpu.shape[1]

  # Step 1: Per-row FP8 E4M3 quantization matching moe_dropin.quantize_rows
  xq_cpu, xs_cpu = moe_dropin.quantize_rows(x_cpu)
  xq_f32 = xq_cpu.to(torch.float32)

  out_acc = torch.zeros((tokens, h_dim), dtype=torch.float32)
  unique_experts = torch.unique(ids_cpu).tolist()

  for e in unique_experts:
    mask = ids_cpu == e  # [tokens, top_k]
    t_idx, k_idx = torch.where(mask)
    if t_idx.numel() == 0:
      continue

    x_e = xq_f32[t_idx]  # [N_e, H]
    xs_e = xs_cpu[t_idx, None]  # [N_e, 1]
    rw_e = wts_cpu[t_idx, k_idx, None]  # [N_e, 1]

    # Step 2: Gate / Up matmul + scale
    acc1 = (x_e @ w13_cpu[e]) * (xs_e * s13_cpu[e][None, :])  # [N_e, 2*I]
    gate = acc1[:, :i_dim]
    up = acc1[:, i_dim:]

    # Step 3: SwiGLU -> bfloat16 intermediate
    mid = (F.silu(gate) * up).to(torch.bfloat16).to(torch.float32)

    # Step 4: Down matmul + scale + routing weight -> bfloat16 per-row output
    acc2 = (mid @ w2_cpu[e]) * (s2_cpu[e][None, :] * rw_e)
    row_out_bf16 = acc2.to(torch.bfloat16).to(torch.float32)

    # Step 6: Accumulate into token output
    out_acc.index_add_(0, t_idx, row_out_bf16)

  return out_acc.to(hidden_states.dtype)


def test_3_moe_reference_comparison():
  print(
      "\n=== Test 3: MoE PyTorch Reference vs TPU-C Kernel Numerical"
      " Comparison ==="
  )
  import torch  # pylint: disable=g-import-not-at-top

  moe_dropin = _import_moe_dropin()

  torch.manual_seed(42)
  device = torch.device("tpu:0")

  e_count = 64
  h_dim = 4096
  i_dim = 1024
  tokens = 16
  top_k = 8

  print(
      "Generating non-trivial random FP8 weights and inputs"
      f" (E={e_count}, T={tokens}, K={top_k}, H={h_dim}, I={i_dim})..."
  )
  # Create random weights on CPU, quantize to FP8 E4M3, and move to TPU
  w13_f32 = torch.randn((e_count, h_dim, 2 * i_dim), dtype=torch.float32) * 0.5
  w2_f32 = torch.randn((e_count, i_dim, h_dim), dtype=torch.float32) * 0.5
  w13 = (
      w13_f32.clamp(-moe_dropin.FP8_MAX, moe_dropin.FP8_MAX)
      .to(torch.float8_e4m3fn)
      .to(device)
  )
  w2 = (
      w2_f32.clamp(-moe_dropin.FP8_MAX, moe_dropin.FP8_MAX)
      .to(torch.float8_e4m3fn)
      .to(device)
  )

  # Realistic per-channel weight scales
  s13 = (
      torch.rand((e_count, 2 * i_dim), dtype=torch.float32) * 0.02 + 0.005
  ).to(device)
  s2 = (torch.rand((e_count, h_dim), dtype=torch.float32) * 0.02 + 0.005).to(
      device
  )

  w = moe_dropin.prepare_weights(w13, w2, s13, s2)

  hidden_states = (
      (torch.randn((tokens, h_dim), dtype=torch.float32) * 0.5)
      .to(torch.bfloat16)
      .to(device)
  )
  router_logits = torch.randn(
      (tokens, e_count), dtype=torch.float32, device=device
  )
  topk_vals, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)
  topk_ids_i32 = topk_ids.to(torch.int32)
  topk_weights = torch.softmax(topk_vals, dim=-1).to(torch.float32)

  print("Running TPU-C MoE kernel (eager & compiled glue)...")
  out_eager = (
      moe_dropin.fused_moe_ep1(
          hidden_states, topk_ids_i32, topk_weights, w, compiled=False
      )
      .cpu()
      .to(torch.float32)
  )
  out_compiled = (
      moe_dropin.fused_moe_ep1(
          hidden_states, topk_ids_i32, topk_weights, w, compiled=True
      )
      .cpu()
      .to(torch.float32)
  )

  print("Running PyTorch MoE reference implementation...")
  out_ref = _pytorch_moe_reference(
      hidden_states, topk_ids_i32, topk_weights, w13, w2, s13, s2
  ).to(torch.float32)

  # Compare metrics: Cosine similarity, Max absolute diff, Relative L2 error
  for name, out_k in [("eager", out_eager), ("compiled", out_compiled)]:
    diff = (out_k - out_ref).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    rel_l2_err = (
        torch.norm(out_k - out_ref) / (torch.norm(out_ref) + 1e-12)
    ).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_k.flatten(), out_ref.flatten(), dim=0
    ).item()
    print(
        f"  [{name}] cos_sim={cos_sim:.6f}, rel_l2_err={rel_l2_err:.6f}, "
        f"max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}",
        flush=True,
    )
    assert cos_sim > 0.999, f"[{name}] Cosine similarity too low: {cos_sim}"
    assert (
        rel_l2_err < 0.02
    ), f"[{name}] Relative L2 error too high: {rel_l2_err}"

  print(
      "✓ Test 3 Passed: TPU-C MoE kernel matches PyTorch MoE reference"
      " implementation!"
  )


def test_4_fused_moe_tpuc_ep8_shapes():
  print("\n=== Test 4: fused_moe_tpuc EP8 Qwen 3.5 Shapes & Parity ===")
  import torch  # pylint: disable=g-import-not-at-top

  prefix = _user_pkg_prefix()
  fused_moe_tpuc = None
  for mod_name in (
      f"{prefix}.torchtpu_tpuc_kernels.moe.fused_moe_tpuc_layer",
      "fused_moe_tpuc_layer",
  ):
    try:
      mod = importlib.import_module(mod_name)
      fused_moe_tpuc = mod.fused_moe_tpuc
      break
    except ImportError:
      continue
  assert fused_moe_tpuc is not None, "Could not import fused_moe_tpuc"

  moe_dropin = _import_moe_dropin()
  device = torch.device("tpu:0")
  e_global = 512
  e_local = 64
  h_dim = 4096
  i_dim = 1024
  top_k = 10
  rank = 5
  experts_start_val = rank * e_local
  experts_start = torch.tensor(
      experts_start_val, dtype=torch.int32, device=device
  )

  gen = torch.Generator(device="cpu")
  gen.manual_seed(1000 + rank)
  w13_f32 = (
      torch.randn(
          (e_local, h_dim, 2 * i_dim), generator=gen, dtype=torch.float32
      )
      * 0.5
  )
  w2_f32 = (
      torch.randn((e_local, i_dim, h_dim), generator=gen, dtype=torch.float32)
      * 0.5
  )
  w1 = (
      w13_f32.clamp(-moe_dropin.FP8_MAX, moe_dropin.FP8_MAX)
      .to(torch.float8_e4m3fn)
      .to(device)
  )
  w2 = (
      w2_f32.clamp(-moe_dropin.FP8_MAX, moe_dropin.FP8_MAX)
      .to(torch.float8_e4m3fn)
      .to(device)
  )
  w1_scale = (
      torch.rand((e_local, 1, 1, 2 * i_dim), generator=gen, dtype=torch.float32)
      * 0.02
      + 0.005
  ).to(device)
  w2_scale = (
      torch.rand((e_local, 1, 1, h_dim), generator=gen, dtype=torch.float32)
      * 0.02
      + 0.005
  ).to(device)

  for t_local in (8, 32, 64, 1024, 2048):
    t_global = t_local * 8
    igen = torch.Generator(device="cpu")
    igen.manual_seed(2000 + t_local * 17)
    hidden = (
        (
            torch.randn((t_global, h_dim), generator=igen, dtype=torch.float32)
            * 0.5
        )
        .to(torch.bfloat16)
        .to(device)
    )
    logits = torch.randn(
        (t_global, e_global), generator=igen, dtype=torch.float32
    )
    topk_vals, topk_ids_i64 = torch.topk(logits, k=top_k, dim=-1)
    topk_weights = (
        torch.softmax(topk_vals, dim=-1).to(torch.bfloat16).to(device)
    )
    topk_ids = topk_ids_i64.to(torch.int32).to(device)

    out_tpuc = fused_moe_tpuc(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_bias=None,
        w2_bias=None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        experts_start=experts_start,
        topk=top_k,
        activation="silu",
        rhs_quant_dtype=None,
        skip_padded_tokens=False,
    )
    out_cpu = out_tpuc.cpu().to(torch.float32)
    assert torch.isfinite(
        out_cpu
    ).all(), f"Non-finite output at T_local={t_local}"

    # Verify numerical parity against PyTorch reference on local expert shard
    if t_local <= 64:
      local_ids = topk_ids - experts_start_val
      valid_mask = (local_ids >= 0) & (local_ids < e_local)
      safe_ids = torch.where(valid_mask, local_ids, torch.zeros_like(local_ids))
      masked_wts = torch.where(
          valid_mask,
          topk_weights.to(torch.float32),
          torch.zeros_like(topk_weights, dtype=torch.float32),
      )
      out_ref = _pytorch_moe_reference(
          hidden,
          safe_ids,
          masked_wts,
          w1,
          w2,
          w1_scale.reshape(e_local, 2 * i_dim),
          w2_scale.reshape(e_local, h_dim),
      ).to(torch.float32)
      cos_sim = torch.nn.functional.cosine_similarity(
          out_cpu.flatten(), out_ref.flatten(), dim=0
      ).item()
      rel_l2_err = (
          torch.norm(out_cpu - out_ref) / (torch.norm(out_ref) + 1e-12)
      ).item()
      print(
          f"  [parity T_local={t_local:4d}] cos_sim={cos_sim:.6f},"
          f" rel_l2_err={rel_l2_err:.6f}",
          flush=True,
      )
      assert cos_sim > 0.999, f"T_local={t_local} cos_sim too low: {cos_sim}"
      assert (
          rel_l2_err < 0.02
      ), f"T_local={t_local} rel_l2_err too high: {rel_l2_err}"

    print(
        f"  ✓ T_local={t_local:4d} (T_global={t_global:5d}):"
        f" shape={tuple(out_cpu.shape)}, norm={torch.norm(out_cpu).item():.4f}",
        flush=True,
    )
  print("✓ Test 4 Passed: All EP8 Qwen 3.5 token counts succeeded!")


def main():
  test_1_registration()
  try:
    import torch  # pylint: disable=g-import-not-at-top

    # Check if TPU device is accessible in PyTorch
    tpu_available = False
    try:
      _ = torch.empty(1, device="tpu")
      tpu_available = True
    except Exception:  # pylint: disable=broad-exception-caught
      pass

    if tpu_available:
      test_2_moe_kernel()
      test_3_moe_reference_comparison()
      test_4_fused_moe_tpuc_ep8_shapes()
      print("\n🎉 ALL TPU-C MOE SMOKE & UNIT TESTS PASSED ON DEVICE!")
    else:
      print(
          "\n[NOTE] No TPU hardware device active in current session. Verified"
          " op registration only."
      )
  except Exception as e:
    print(f"\n[FAIL] Test encountered error: {e}")
    raise


if __name__ == "__main__":
  try:
    from absl import app  # pylint: disable=g-import-not-at-top

    app.run(lambda _: main())
  except ImportError:
    main()
