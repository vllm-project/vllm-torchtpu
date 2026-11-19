#!/usr/bin/env python3
"""Source mutants of the VPS per-row fallback / VPB code (vp_sampler.vp_verify_fb and helpers); each must make
tests/decfb2/test_vps_fallback.py fail. Run: python tests/decfb2/q38_decfb2_vpsfb_mutrun.py"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
S = "vllm_torchtpu/layers/adapter/sample"
V = f"{S}/vp_sampler.py"
FILES = [V, f"{S}/rejection_sampler.py", f"{S}/top_k_top_p.py", "vllm_torchtpu/runner/tpu_runner.py",
         "vllm_torchtpu/envs.py", "vllm_torchtpu/runner/vps_wiring.py", "vllm_torchtpu/compilation/tpu_compiler.py"]
M = [
    ("sentinel_kth", "return vals[:, :kc], ids[:, :kc], vals[:, kc:]",
     "return vals[:, :kc], ids[:, :kc], vals[:, kc - 1:kc]"),
    ("complete_strict", "complete = (scaled_c > m).sum(dim=-1) >= k", "complete = (scaled_c > m).sum(dim=-1) > k"),
    ("complete_ge_m", "complete = (scaled_c > m).sum(dim=-1) >= k", "complete = (scaled_c >= m).sum(dim=-1) >= k"),
    ("sentinel_amin", "m = scaled_s.amax(dim=-1, keepdim=True)", "m = scaled_s.amin(dim=-1, keepdim=True)"),
    ("range_no_upper", "in_range = (k >= 1) & (k <= kc)", "in_range = (k >= 1)"),
    ("greedy_flagged", "live = active & ~greedy", "live = active"),
    ("padding_flagged", "active_b = torch.arange(num_bonus, device=device) < num_active",
     "active_b = torch.arange(num_bonus, device=device) >= 0"),
    ("undeclared_flagged", "group_indices.to(torch.int64) < draft_lengths.to(torch.int64)[seg_c]",
     "group_indices.to(torch.int64) >= 0"),
    ("tie_rows_first", "prio = torch.where(first, 0, torch.where(flag, 1, 2))",
     "prio = torch.where(first, 1, torch.where(flag, 0, 2))"),
    ("no_bonus_scatter", "use_b = used & is_b\n", "use_b = used & is_b & False\n"),
    ("no_target_override", "row_override = (flag_t[:num_target], acc_t[:num_target], rec_t[:num_target])",
     "row_override = None"),
    ("override_keeps_accept", "        accepted = torch.where(use_row, accepted_full, accepted)\n", ""),
    ("override_keeps_recover", "        recovered_token_ids = torch.where(use_row, recovered_full, recovered_token_ids)\n",
     ""),
    ("vpbx_swapped_uniforms", "u_b, u_t = bonus_u[b_idx], recover_u[t_idx]",
     "u_b, u_t = recover_u[t_idx], bonus_u[b_idx]"),
    ("vpb_wrong_fb_rows", "u_b, u_t = fb_u, fb_u", "u_b, u_t = fb_u, fb_u.flip(0)"),
    ("full_target_bonus_scaling", "masked_logits = apply_top_k_top_p(_scaled_target(rows, temperatures), top_k, top_p)",
     "masked_logits = apply_top_k_top_p(_scaled_bonus(rows, temperatures), top_k, top_p)"),
    ("full_target_no_topp", "masked_logits = apply_top_k_top_p(_scaled_target(rows, temperatures), top_k, top_p)",
     "masked_logits = apply_top_k_top_p(_scaled_target(rows, temperatures), top_k, torch.ones_like(top_p))"),
    ("full_bonus_no_topp", "masked_logits = apply_top_k_top_p(_scaled_bonus(rows, temperatures), top_k, top_p)",
     "masked_logits = apply_top_k_top_p(_scaled_bonus(rows, temperatures), top_k, torch.ones_like(top_p))"),
    ("wrong_logits_rows", "lrow = torch.where(is_b, bonus_logits_indices[b_idx], target_logits_indices[t_idx])",
     "lrow = torch.where(is_b, target_logits_indices[t_idx], bonus_logits_indices[b_idx])"),
    ("synthetic_rates_group0", "rates_f = synthetic_conditional_rates[group_indices.to(torch.int64)[t_idx]]",
     "rates_f = synthetic_conditional_rates[group_indices.to(torch.int64)[t_idx] * 0]"),
    ("stats_uncovered", "stats = torch.stack([num_flagged, num_covered, num_flagged - num_covered])",
     "stats = torch.stack([num_flagged, num_covered, num_flagged])"),
    ("stats_covered_is_F", "num_covered = used.sum().to(torch.int32)",
     "num_covered = torch.tensor(fallback_rows, dtype=torch.int32)"),
]

test = os.path.join(HERE, "test_vps_fallback.py")
killed = 0
for name, a, b in M:
    src = open(os.path.join(SRC, V)).read()
    if src.count(a) != 1:
        print("BAD-ANCHOR", name, src.count(a))
        continue
    with tempfile.TemporaryDirectory() as d:
        for rel in FILES:
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
            shutil.copy(os.path.join(SRC, rel), os.path.join(d, rel))
        open(os.path.join(d, V), "w").write(src.replace(a, b))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--noconftest", test],
                           env=dict(os.environ, Q38MM_SRC_ROOT=d, PYTHONDONTWRITEBYTECODE="1"),
                           capture_output=True, text=True)
        k = r.returncode != 0
        killed += k
        print("KILLED  " if k else "SURVIVED", name, (r.stdout.strip().splitlines() or ["?"])[-1][:90])
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", test],
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
print("control:", "PASS" if r.returncode == 0 else "FAIL", r.stdout.strip().splitlines()[-1])
print(f"VPSFB MUTANTS {killed}/{len(M)} KILLED")
