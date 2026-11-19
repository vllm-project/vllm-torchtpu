#!/usr/bin/env python3
"""Source mutants of the PD DEC-FB-2 VPS fallback wiring (tpu_runner.py, vps_wiring.py, tpu_compiler.py); each must
make tests/decfb2/test_vps_fb_wiring.py fail. Run: python tests/decfb2/q38_decfb2_vpsfb_wiring_mutrun.py"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "src")
R = "vllm_torchtpu/runner/tpu_runner.py"
W = "vllm_torchtpu/runner/vps_wiring.py"
T = "vllm_torchtpu/compilation/tpu_compiler.py"
S = "vllm_torchtpu/layers/adapter/sample"
EV = "vllm_torchtpu/envs.py"
FILES = [R, W, T, EV, f"{S}/vp_sampler.py", f"{S}/rejection_sampler.py", f"{S}/top_k_top_p.py",
         "vllm_torchtpu/spec_decode/eagle3.py", "vllm_torchtpu/spec_decode/mtp_fused_loop.py"]
M = [
    ("R_fb_never", R, "        if self._vps_fb_path:\n            out = self._vps_verify_fb(",
     "        if False:\n            out = self._vps_verify_fb("),
    ("R_chunk_drops_F", R, "logits, md, cur_start_idx, cur_end_idx, sampling_generator, vps_fallback\n",
     "logits, md, cur_start_idx, cur_end_idx, sampling_generator, 0\n"),
    ("R_cand_full_width", R, "            vocab = vh.tp_size * vp_sampler.VP_KC\n", "            vocab = vh.full_vocab\n"),
    ("R_fbu_missing", R, "        if self._vps_cand_draws and fallback_rows > 0:\n            fb_u = torch.rand(",
     "        if False:\n            fb_u = torch.rand("),
    ("R_num_active_zero", R, '"vps_num_active", torch.tensor([num_active], dtype=torch.int32)',
     '"vps_num_active", torch.tensor([0], dtype=torch.int32)'),
    ("R_stats_not_accumulated", R,
     "self._vps_fb_stats = stats if self._vps_fb_stats is None else self._vps_fb_stats + stats",
     "self._vps_fb_stats = stats"),
    ("R_check_under_vpb", R, "        if self._vps_cand_draws:\n            # Per-candidate draws are not the stock draws",
     "        if False:\n            # Per-candidate draws are not the stock draws"),
    ("R_host_kc64", R, "            vp_sampler.VP_KC,\n            self.speculative_config.num_speculative_tokens + 1,",
     "            64,\n            self.speculative_config.num_speculative_tokens + 1,"),
    ("R_host_rows_per_req", R, "            self.speculative_config.num_speculative_tokens + 1,\n        )",
     "            self.speculative_config.num_speculative_tokens,\n        )"),
    ("R_host_tiegap_cap", R, "need, self._vps_tiegap, max(self._vps_fb_max, self._vps_tiegap)",
     "need, self._vps_tiegap, self._vps_fb_max"),
    ("R_exec_always_vps", R, "                vps_chunk = fb_rows is not None\n", "                vps_chunk = True\n"),
    ("R_exec_F_zero", R, "                vps_fb = fb_rows or 0\n", "                vps_fb = 0\n"),
    ("R_state_drops_F", R, "            vps_fallback=vps_fb_list if any(vps_fb_list) else [],\n",
     "            vps_fallback=[],\n"),
    ("R_sample_drops_F", R, "                            vps_fallback=vps_fb,\n", ""),
    ("R_precompile_F0_only", R, "        for rows in vps_wiring.fallback_variants(self._vps_fb_max, self._vps_tiegap):",
     "        for rows in [0]:"),
    ("R_precompile_no_fbu", R, "            if self._vps_cand_draws and rows > 0:\n                fb_u = torch.zeros(",
     "            if False:\n                fb_u = torch.zeros("),
    ("R_lockstep_dropped", R, "                dp_lockstep=self._dp_lockstep_enabled(),\n", ""),
    ("W_lockstep_ignored", W, "    if dp_lockstep:\n        return None,", "    if False:\n        return None,"),
    ("W_width_kc", W, "    if width % block or width // block < kc + 1:\n", "    if width % block or width // block < kc:\n"),
    ("W_range_counts_greedy", W, "live = temperatures.to(torch.float32) != 0", "live = temperatures.to(torch.float32) >= 0"),
    ("W_bucket_over_max", W, "        if need <= b <= max_rows:\n", "        if need <= b:\n"),
    ("W_variants_no_zero", W, "    out = {fallback_slots(0, tiegap, max(max_rows, tiegap))}\n", "    out = set()\n"),
    ("W_stats_default_0", EV, 'env_nonnegative_int("VLLM_TPU_VPS_STATS_EVERY", 1000)',
     'env_nonnegative_int("VLLM_TPU_VPS_STATS_EVERY", 0)'),
    ("W_draws_unhashed", W,
     "HASHED_ENVS = (VPS_VERIFY_ENV, FALLBACK_ROWS_ENV, TIEGAP_SLOTS_ENV, DRAWS_ENV)",
     "HASHED_ENVS = (VPS_VERIFY_ENV, FALLBACK_ROWS_ENV, TIEGAP_SLOTS_ENV)"),
    ("W_vpb_alias_missing", W, '    if v in ("candidate", "vpb"):\n', '    if v in ("candidate",):\n'),
    ("W_fallback_path_ignores_draws", W, "    return fallback_max() > 0 or tiegap_slots() > 0 or cand_draws()",
     "    return fallback_max() > 0 or tiegap_slots() > 0"),
    ("T_draws_unkeyed", T, '    "VLLM_TPU_VPS_CHECK",\n    "VLLM_TPU_VPS_CHECK_FAULT",\n',
     '    "VLLM_TPU_VPS_CHECK",\n    "VLLM_TPU_VPS_DRAWS",\n    "VLLM_TPU_VPS_CHECK_FAULT",\n'),
    ("T_stats_hashed", T, '    "VLLM_TPU_VPS_STATS_EVERY",\n    # Startup', '    # Startup'),
]

test = os.path.join(HERE, "test_vps_fb_wiring.py")
killed = 0
for name, f, a, b in M:
    src = open(os.path.join(SRC, f)).read()
    if src.count(a) != 1:
        print("BAD-ANCHOR", name, src.count(a))
        continue
    with tempfile.TemporaryDirectory() as d:
        for rel in FILES:
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
            shutil.copy(os.path.join(SRC, rel), os.path.join(d, rel))
        open(os.path.join(d, f), "w").write(src.replace(a, b))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--noconftest", test],
                           env=dict(os.environ, Q38MM_SRC_ROOT=d, PYTHONDONTWRITEBYTECODE="1"),
                           capture_output=True, text=True)
        k = r.returncode != 0
        killed += k
        print("KILLED  " if k else "SURVIVED", name, (r.stdout.strip().splitlines() or ["?"])[-1][:90])
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", test],
                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
print("control:", "PASS" if r.returncode == 0 else "FAIL", r.stdout.strip().splitlines()[-1])
print(f"VPSFB WIRING MUTANTS {killed}/{len(M)} KILLED")
