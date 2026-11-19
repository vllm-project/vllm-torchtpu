#!/usr/bin/env python3
"""Source mutants of layers/adapter/sample/vp_sampler.py; each must make test_vp_sampler.py fail."""
import os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "..", "..", "src")
FILES = ["vllm_torchtpu/layers/adapter/sample/vp_sampler.py", "vllm_torchtpu/layers/adapter/sample/rejection_sampler.py",
         "vllm_torchtpu/layers/adapter/sample/top_k_top_p.py", "vllm_torchtpu/runner/tpu_runner.py"]
VPF = FILES[0]
M = [("block_order_unsorted", "bid = torch.sort(order, dim=-1).values", "bid = order"),
     ("inner_topk_unstable_order", "pos = torch.sort(cand, dim=-1, descending=True, stable=True).indices[:, :kc]",
      "pos = torch.sort(cand, dim=-1, descending=True, stable=True).indices[:, :kc].flip(-1)"),
     ("ids_div_mod_swapped", "torch.div(pos, block, rounding_mode=\"floor\")) * block + pos % block",
      "pos % block) * block + torch.div(pos, block, rounding_mode=\"floor\")"),
     ("no_shard_offset", "+ pos % block + shard_offset", "+ pos % block"),
     ("tie_max_id", "return torch.where(score == best, ids, big).amin(dim=-1)", "return torch.where(score == best, ids, -big).amax(dim=-1)"),
     ("cutoff_strict", "masked = torch.where(cand >= cutoff, cand,", "masked = torch.where(cand > cutoff, cand,"),
     ("bonus_no_greedy_branch", "final_logits = torch.where(is_greedy, cv, noisy_logits)", "final_logits = noisy_logits"),
     ("bonus_no_temperature", "masked_logits = _candidate_mask(cv / safe_temperatures, top_k, top_p)", "masked_logits = _candidate_mask(cv, top_k, top_p)"),
     ("recovery_keeps_draft", "draft_token_mask = draft_hit & valid_draft_token_ids.unsqueeze(-1)", "draft_token_mask = draft_hit & False"),
     ("recover_u_off_by_one", "recover_u_c = torch.gather(recover_u, 1, target_ci.to(torch.int64))",
      "recover_u_c = torch.gather(recover_u, 1, (target_ci.to(torch.int64) + 1) % recover_u.shape[1])"),
     ("bonus_u_wrong_rows", "bonus_u_c = torch.gather(bonus_u, 1, bonus_ci.to(torch.int64))", "bonus_u_c = torch.gather(bonus_u.flip(0), 1, bonus_ci.to(torch.int64))"),
     ("no_mass_check", "recovered_token_ids = torch.where(is_greedy | ~has_recovery_mass, target_logits_argmax, recovered_token_ids)",
      "recovered_token_ids = torch.where(is_greedy, target_logits_argmax, recovered_token_ids)"),
     ("accept_ignores_valid", "    accepted = accepted & valid_draft_token_ids\n", "    accepted = accepted\n"),
     ("greedy_accept_by_prob", "accepted = torch.where(is_greedy, draft_token_ids == target_logits_argmax, accept_u < target_token_probs)",
      "accepted = accept_u < target_token_probs"),
     ("offsets_off_by_one", "row_offsets = torch.where(valid, (num_valid - 1).clamp(min=0), 0)", "row_offsets = torch.where(valid, num_valid, 0)"),
     ("target_no_f32", "scaled_logits = cv.to(torch.float32) / (temperatures.to(torch.float32) + TEMPERATURE_EPS)",
      "scaled_logits = (cv / (temperatures + TEMPERATURE_EPS)).to(torch.float32)"),
     ("assembly_drift", "    all_accepted = first_rejection_idx_per_segment == large_value\n",
      "    all_accepted = first_rejection_idx_per_segment >= large_value\n")]
src = open(os.path.join(SRC, VPF)).read(); test = os.path.join(HERE, "test_vp_sampler.py"); k = 0
for name, a, b in M:
    if src.count(a) != 1: print("BAD-ANCHOR", name, src.count(a)); continue
    with tempfile.TemporaryDirectory() as d:
        for rel in FILES:
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True); shutil.copy(os.path.join(SRC, rel), os.path.join(d, rel))
        open(os.path.join(d, VPF), "w").write(src.replace(a, b))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--noconftest", test],
                           env=dict(os.environ, Q38MM_SRC_ROOT=d), capture_output=True, text=True)
        kk = r.returncode != 0; k += kk
        print(("KILLED  " if kk else "SURVIVED"), name, (r.stdout.strip().splitlines() or ["?"])[-1][:90])
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", test], capture_output=True, text=True)
print("control:", "PASS" if r.returncode == 0 else "FAIL", r.stdout.strip().splitlines()[-1])
print(f"VPS MUTANTS {k}/{len(M)} KILLED")
