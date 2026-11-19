#!/usr/bin/env python3
"""Source mutants of the VPS wiring; each must make test_vps_wiring.py fail."""
import os, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(HERE, "..", "..", "src")
R = "vllm_torchtpu/runner/tpu_runner.py"; W = "vllm_torchtpu/runner/vps_wiring.py"; E = "vllm_torchtpu/spec_decode/eagle3.py"
L = "vllm_torchtpu/spec_decode/mtp_fused_loop.py"; T = "vllm_torchtpu/compilation/tpu_compiler.py"
S = "vllm_torchtpu/layers/adapter/sample"
V = "vllm_torchtpu/envs.py"
FILES = [R, W, E, L, T, V, f"{S}/vp_sampler.py", f"{S}/rejection_sampler.py", f"{S}/top_k_top_p.py"]
M = [
 ("R_draws_local_width", R, "        vocab = vh.full_vocab\n", "        vocab = local_logits.shape[-1]\n"),
 ("R_draw_order", R, '''        bonus_u = torch.rand(
            (padded_num_reqs, vocab),
            dtype=torch.float32,
            device=local_logits.device,
            generator=sampling_generator,
        )
        accept_u = torch.rand(
            md.draft_token_ids.shape,
            dtype=torch.float32,
            device=local_logits.device,
            generator=sampling_generator,
        )''', '''        accept_u = torch.rand(
            md.draft_token_ids.shape,
            dtype=torch.float32,
            device=local_logits.device,
            generator=sampling_generator,
        )
        bonus_u = torch.rand(
            (padded_num_reqs, vocab),
            dtype=torch.float32,
            device=local_logits.device,
            generator=sampling_generator,
        )'''),
 ("R_check_self_compare", R, "        mismatches = (compared != ref).sum().to(torch.int64)\n        # [clean mismatched positions, faulted checks that were caught]\n        delta = torch.stack(\n            [\n                mismatches * (0 if faulted else 1),\n                (mismatches > 0).to(torch.int64) * (1 if faulted else 0),\n            ]\n        )\n        self._vps_mismatches",
  "        mismatches = (compared != compared).sum().to(torch.int64)\n        # [clean mismatched positions, faulted checks that were caught]\n        delta = torch.stack(\n            [\n                mismatches * (0 if faulted else 1),\n                (mismatches > 0).to(torch.int64) * (1 if faulted else 0),\n            ]\n        )\n        self._vps_mismatches"),
 ("R_no_fault", R, "        faulted = self._vps_checks_done < self._vps_check_faults\n", "        faulted = False\n"),
 ("R_check_skipped", R, "        if self._vps_checks_left > 0:\n            self._check_vps(", "        if False:\n            self._check_vps("),
 ("R_no_dispatch", R, "        if vps:\n            # The chunk's logits are the rank's vocab shard (VLLM_TPU_VPS_VERIFY).\n", "        if False:\n            # The chunk's logits are the rank's vocab shard (VLLM_TPU_VPS_VERIFY).\n"),
 ("R_step_ignores_top_tokens", R, "            and not top_tokens\n            and vps_wiring.step_eligible(", "            and vps_wiring.step_eligible("),
 ("R_chunk_ignores_md", R, "                and spec_decode_metadata is not None\n                and self._sampler_topk_candidates(start_index, end_index, None) > 0", "                and self._sampler_topk_candidates(start_index, end_index, None) > 0"),
 ("R_chunk_ignores_cands", R, "                and self._sampler_topk_candidates(start_index, end_index, None) > 0\n            )\n            vps_fb = 0\n", "            )\n            vps_fb = 0\n"),  # PD: anchor follows the fallback's vps_fb line
 ("R_dispatch_sel_full", R, "                    if vps_chunk:\n                        logits = self.compute_selected_local_logits(", "                    if False:\n                        logits = self.compute_selected_local_logits("),
 ("R_dispatch_h_full", R, "                elif vps_chunk:\n                    logits = self.compute_local_logits_from_hidden_states(", "                elif False:\n                    logits = self.compute_local_logits_from_hidden_states("),
 ("W_no_softcap_check", W, 'if getattr(processor, "soft_cap", None) is not None or getattr(processor, "scale", 1.0) != 1.0:', 'if getattr(processor, "scale", 1.0) != 1.0:'),
 ("W_no_padding_check", W, '    if int(getattr(si, "num_org_vocab_padding", 0)) != 0:\n', '    if False:\n'),
 ("W_no_width_check", W, '    if width % block or width // block < kc + 1:\n', '    if False:\n'),
 ("W_no_tp_check", W, '    if tp < 2:\n', '    if False:\n'),
 ("W_no_grammar_check", W, '        if params is not None and params.structured_outputs is not None:\n            return False', '        if False:\n            return False'),
 ("W_no_logprobs_check", W, 'if input_batch.all_greedy or input_batch.max_num_logprobs is not None:', 'if input_batch.all_greedy:'),
 ("W_local_gathers", W, "return vh.processor._get_logits(hidden, vh.head, None, skip_gather=True)", "return vh.processor._get_logits(hidden, vh.head, None, skip_gather=False)"),
 ("W_draft_no_optin", W, '    if not getattr(type(draft_model), "plain_compute_logits", False):\n        return False\n    processor', '    processor'),
 ("W_draft_ignores_d2t", W, 'if draft_model is None or getattr(draft_model, "draft_id_to_target_id", None) is not None:', 'if draft_model is None:'),
#("E_propose_no_dvp", E, "            if self._dvp:\n                return vps_wiring.draft_top_tokens(self.draft_model, hidden)\n", ""),  # PD port: draft sites not ported
#("E_carry_no_dvp", E, "        if d2t is None and self._dvp:\n", "        if False:\n"),  # PD port: draft sites not ported
#("L_loop_no_dvp", L, "            if self.dvp:\n                input_ids = self.mtp.logits_processor.get_top_tokens(", "            if False:\n                input_ids = self.mtp.logits_processor.get_top_tokens("),  # PD port: draft sites not ported
 ("T_draws_ignored", T, '    "VLLM_TPU_VPS_CHECK",\n    "VLLM_TPU_VPS_CHECK_FAULT",\n', '    "VLLM_TPU_VPS_CHECK",\n    "VLLM_TPU_VPS_DRAWS",\n    "VLLM_TPU_VPS_CHECK_FAULT",\n'),  # PD stack: registered flags
 ("T_check_hashed", T, '    "VLLM_TPU_VPS_CHECK",\n    "VLLM_TPU_VPS_CHECK_FAULT",\n', '    "VLLM_TPU_VPS_CHECK_FAULT",\n'),
 ("V_verify_default_on", V, '"VLLM_TPU_VPS_VERIFY": env_bool("VLLM_TPU_VPS_VERIFY"),', '"VLLM_TPU_VPS_VERIFY": env_bool("VLLM_TPU_VPS_VERIFY", True),'),
 ("V_draws_default_candidate", V, 'env_str("VLLM_TPU_VPS_DRAWS", "full")', 'env_str("VLLM_TPU_VPS_DRAWS", "candidate")'),
]
test = os.path.join(HERE, "test_vps_wiring.py"); k = 0
for name, f, a, b in M:
    src = open(os.path.join(SRC, f)).read()
    if src.count(a) != 1: print("BAD-ANCHOR", name, src.count(a)); continue
    with tempfile.TemporaryDirectory() as d:
        for rel in FILES:
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True); shutil.copy(os.path.join(SRC, rel), os.path.join(d, rel))
        open(os.path.join(d, f), "w").write(src.replace(a, b))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--noconftest", test],
                           env=dict(os.environ, Q38MM_SRC_ROOT=d), capture_output=True, text=True)
        kk = r.returncode != 0; k += kk
        print(("KILLED  " if kk else "SURVIVED"), name, (r.stdout.strip().splitlines() or ["?"])[-1][:80])
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", test], capture_output=True, text=True)
print("control:", "PASS" if r.returncode == 0 else "FAIL", r.stdout.strip().splitlines()[-1])
print(f"WIRING MUTANTS {k}/{len(M)} KILLED")
