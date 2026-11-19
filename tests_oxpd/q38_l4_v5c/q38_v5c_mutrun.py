#!/usr/bin/env python3
"""q38 L4 v5 composition mutants (CDK CPU pod only). V5COMP-BUILD. Usage: q38_v5c_mutrun.py <head tree> [LO HI]

Each mutant edits one composed file in a fresh copy of the tree (src, tests, q38_l4vfy) and runs the composition test
that must catch it; pytest must exit 1 (tests failed) to count as killed. Exit 0 = SURVIVED; any other exit (a
collection or import error) = BAD (neither). A clean copy first runs every target and must pass (control).
Every anchor must occur at least `nth` times in its file (else BAD).
Lines: MUTRUN_ORIGIN, CONTROL <target> rc=..., MUT <name> KILLED|SURVIVED|BAD rc=..., and last
Q38_V5C_MUT PASS|FAIL killed K/N of N range LO-HI of TOTAL control_ok=<0|1>.
"""

import os
import shutil
import subprocess
import sys
import tempfile

RUNNER = "src/vllm_torchtpu/runner/tpu_runner.py"
L6AB = "src/vllm_torchtpu/core/l6_ab.py"
DONE = "src/vllm_torchtpu/spec_decode/draft_one.py"
R_T = "tests/q38_l4_v5c/test_v5c_runner.py"
D_T = "tests/q38_l4_v5c/test_v5c_drafter.py"
U_T = "tests/q38_l4_v5c/test_v5c_union.py"
F_T = "tests/q38_l4_v5c/test_v5c_flags.py"
O_T = "tests/q38_l4_v5c/test_v5c_flagoff.py"
MTP = "src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py"
SAMP2 = "src/vllm_torchtpu/runner/l4_samp2.py"
V_T = "tests/q38_l4_samp2v/test_l4_samp2.py"
E_T = "tests/q38_l4_v5c/test_v5c_emb_agree.py"
K_T = "tests/q38_l4_v5c/test_v5c_cap.py"
W_T = "tests/q38_l4_v5c/test_v5c_keylite_vrng.py"
L_T = "tests/q38_lmhs0/test_lmh_hidden_major.py"
SFX = "src/vllm_torchtpu/runner/seedfix.py"
S_T = "tests/q38_l4_seedfix/test_seedfix_vrng.py"
# v6 (q38-l4-v6c2, V6COMP-BUILD)
EAGLE = "src/vllm_torchtpu/spec_decode/eagle3.py"
K1F = "src/vllm_torchtpu/spec_decode/k1fuse.py"
H3A = "src/vllm_torchtpu/megakernel_runtime/qwen38_draft_attn.py"
K6_T = "tests/q38_l4_v5c/test_v6c_k1.py"
K1O_T = "tests/q38_k1fuse/test_k1fuse_flagoff.py"
H6_T = "tests/kernels/qwen4_exp/dcp2/test_v6c_h3_compose.py"
AW_T = "tests/q38_attnwire/test_attnwire_wiring.py"
MGR = "src/vllm_torchtpu/runner/speculative_decoding_manager.py"
DASM = "src/vllm_torchtpu/spec_decode/draft_asm.py"
FX_T = "tests/q38_l4_foldfix/test_foldfix_realpath.py"
KF_T = "tests/q38_l4_v5c/test_v6c_foldfix_k1.py"
# targets run as ATTNWIRE runs them: VLLM_TARGET_DEVICE=cpu, L1's q38_cpu_device plugin (Q38_V6C_PLUGINS), -s
H3ENV = ("tests/kernels/", "tests/q38_attnwire/", "tests/megakernel")
NOCONF = {R_T, U_T, O_T}  # VFY-ONE's stub-harness files run with --noconftest (node ids: the file part decides)

SNAP_P = ('        if self._vps_fb_path and _l6_ab.on("sampsnap"):\n'
          "            # q38 L4 v5 composition (V5COMP-BUILD): `_sample_spec_verify_vps`'s SAMP-SNAP parameter uploads.\n")
SNAP_N = ('        if _l6_ab.on("sampsnap"):\n'
          "            # q38 L4 v5 composition (V5COMP-BUILD): `_vps_verify_fb`'s SAMP-SNAP num_active upload.\n")
DRAIN = ("        if pend.pending:\n"
         "            # q38 L4 T-1(i): earlier windows whose non-blocking copy has landed.\n"
         "            for chunks, counts in pend.ready():\n"
         "                self._log_vps_fallback(chunks, counts)\n")
STATS = '            if _l6_ab.on("statsasync"):\n'

# (name, file, old, new, nth occurrence (1-based), target test)
MUTANTS = [
    # 786f7bac2, the composition fix: each VFY copy without its lever branch
    ("C01_vfy_verify_params_without_sampsnap", RUNNER, SNAP_P, SNAP_P.replace('_l6_ab.on("sampsnap")', "False"), 1, R_T),
    ("C02_vfy_verify_num_active_without_sampsnap", RUNNER, SNAP_N, SNAP_N.replace('_l6_ab.on("sampsnap")', "False"), 2, R_T),
    ("C03_vfy_chain_num_active_without_sampsnap", RUNNER, SNAP_N, SNAP_N.replace('_l6_ab.on("sampsnap")', "False"), 1, R_T),
    ("C04_vfy_tick_without_statsasync", RUNNER, STATS, "            if False:\n", 2, R_T),
    ("C05_vfy_tick_does_not_drain_landed_copies", RUNNER, DRAIN, "", 2, R_T),
    # one piece's flag (or hook) silently ignored on the composed tree
    ("C06_stock_chain_ignores_sampsnap", RUNNER,
     '        if _l6_ab.on("sampsnap"):\n            # q38 L4 T-2 stage 1 SAMP-SNAP: one upload per num_active value.\n',
     '        if False:\n            # q38 L4 T-2 stage 1 SAMP-SNAP: one upload per num_active value.\n', 1, R_T),
    ("C07_stock_chain_ignores_statsasync", RUNNER, STATS, "            if False:\n", 1, R_T),
    ("C08_dhd_flag_ignored", DONE, "        self.one_conf = bool(envs.VLLM_TPU_DRAFT_ASM_ONE) and self.one_ab not in _OFF\n",
     "        self.one_conf = False\n", 1, D_T),
    ("C09_dhd_ignores_the_verify_seed", DONE, '        if st.mode == "fold" and hasattr(p, "_vfy_seed_take"):\n',
     '        if False and hasattr(p, "_vfy_seed_take"):\n', 1, D_T),
    # a LEVERS entry dropped / two levers on one env var
    ("C10_levers_draftasm_dropped", L6AB, '    "draftasm": "VLLM_TPU_DRAFT_ASM_AB",\n', "", 1, U_T),
    ("C11_levers_statsasync_dropped", L6AB, '    "statsasync": "VLLM_TPU_L4_VPS_STATS_ASYNC",\n', "", 1, U_T),
    ("C12_levers_draftasmone_dropped_flags", L6AB, '    "draftasmone": "VLLM_TPU_DRAFT_ASM_ONE_AB",\n', "", 1, F_T),
    ("C13_levers_two_pieces_share_an_env_var", L6AB, '    "vfyone": "VLLM_TPU_VPS_VFY_ONE_AB",\n',
     '    "vfyone": "VLLM_TPU_DRAFT_ASM_AB",\n', 1, U_T),
    # two pieces' agreement bits colliding (DRAFT-ASM entry 0: stage A and DH-D on one bit)
    ("C14_agreement_bits_collide", DONE, "BIT_FOLD, BIT_CL, BIT_ONE = 1, 2, 4\n", "BIT_FOLD, BIT_CL, BIT_ONE = 1, 2, 2\n",
     1, D_T),
    ("C15_agreement_fold_bit_collides_with_stage_a", DONE, "BIT_FOLD, BIT_CL, BIT_ONE = 1, 2, 4\n",
     "BIT_FOLD, BIT_CL, BIT_ONE = 1, 1, 4\n", 1, D_T),
    # flag-off identity: an unguarded hook
    ("C16_vfy_warmup_hook_unguarded", RUNNER,
     '                if self.__dict__.get("_vfy") is not None:\n'
     "                    # q38 L4 VFY-ONE: its verify programs beside the D83 chain's.\n"
     "                    self._vfy_precompile()\n",
     "                self._vfy_precompile()\n", 1, O_T),
    ("C17_sampsnap_snapshot_read_unguarded", RUNNER, "        snap = self._samp_snap if _l6_ab.on(\"sampsnap\") else None\n",
     "        snap = self._samp_snap\n", 1, O_T),
    # AGREE x DRAFT-2 (c): the load-time agreement through DRAFT-ASM's removed private gather
    ("C18_draftemb_gathers_through_the_removed_draft_asm_gather", MTP,
     "    from vllm_torchtpu.runner.l4_agree import gather as _gather\n",
     "    from vllm_torchtpu.spec_decode.draft_asm import _gather\n", 1, E_T),
    # DRAFT-CAP's flag silently ignored on the composed tree
    ("C19_draftcap_flag_ignored", DONE, "        self.one_cap = bool(envs.VLLM_TPU_DRAFT_ASM_CAP) and self.one_conf\n",
     "        self.one_cap = False\n", 1, K_T),
    # VRNG-VPS on the agreement clock (L4-E's two mutants)
    ("C20_vrng_counter_outside_the_vector", SAMP2,
     "        return [int(self.wish()), int(bool(self.local_disabled)), self.fingerprint(), self.counter_word()]\n",
     "        return [int(self.wish()), int(bool(self.local_disabled)), self.fingerprint(), 0]\n", 1,
     V_T + "::test_vrng_one_rank_counter_offset_is_caught"),
    ("C21_vrng_begin_still_gathers", SAMP2,
     "        self.serving_calls += 1\n        use = False\n",
     "        self.serving_calls += 1\n        from vllm_torchtpu.runner.l4_agree import gather as _g\n"
     "        self.apply(_g(self.local_vector()))\n        use = False\n", 1,
     V_T + "::test_vrng_eight_ranks_equal_counters_and_tokens"),
    # composition fix 3: the VFY warm check given VRNG's raw call (None draws) instead of its meta-draw key call
    ("C22_vfy_warm_check_without_vrng_kcall", RUNNER, "        if not vfy.warm_check(kcall, lite, mode == \"check\"):\n",
     "        if not vfy.warm_check(call, lite, mode == \"check\"):\n", 1, W_T),
    # LMH-STREAM stage 0's flag silently ignored on the composed tree
    ("C23_lmh_flag_ignored", RUNNER,
     "            envs.VLLM_TPU_LMH_HIDDEN_MAJOR_DRAFT or envs.VLLM_TPU_LMH_HIDDEN_MAJOR_VERIFY\n", "            False\n", 1,
     L_T + "::test_runner_arms_after_the_drafter_on_the_last_stage_only"),
    # SEEDFIX's flag silently ignored on the composed tree (its gate reads off whatever the env says)
    ("C24_seedfix_flag_ignored", SFX, "    return bool(envs.VLLM_TPU_SEEDED_SAMPLING)\n", "    return False\n", 1, S_T),
    # v6: K1FUSE's flag ignored, in the controller (K1 never serves under DRAFT-CAP) and at eagle3's build
    ("C25_k1_controller_ignores_its_flag", K1F,
     "        self.k1_conf = bool(envs.VLLM_TPU_K1FUSE) and self.one_conf and self.k1_ab not in _OFF\n",
     "        self.k1_conf = False\n", 1, K6_T + "::test_k1c_k1_cap_runs_at_r_d_and_is_bitwise_the_cap_chain[8-0]"),
    ("C26_eagle3_builds_draft_one_whatever_k1", EAGLE, "                if envs.VLLM_TPU_K1FUSE:\n",
     "                if False:\n", 1, K6_T + "::test_k1e_eagle3_builds_draft_k1_only_under_its_flag"),
    # v6: K1 allowed while VRNG-VPS is configured (L4-E: must crash or diverge)
    ("C27_k1_allowed_with_vrng", K1F, '            elif getattr(runner, "_vrng", None) is not None:\n',
     "            elif False:\n", 1, K6_T + "::test_k1v_k1_refuses_for_the_run_when_vrng_is_configured[0-33]"),
    # v6: K1 ignores DRAFT-CAP (its draft at the capacity again; L4-E's mutant)
    ("C28_k1_ignores_one_cap", K1F, "        if self.one_cap:\n            # q38 L4 v6 composition (V6COMP-BUILD), K1 x DRAFT-CAP",
     "        if False:\n            # q38 L4 v6 composition (V6COMP-BUILD), K1 x DRAFT-CAP", 1,
     K6_T + "::test_k1c_k1_cap_runs_at_r_d_and_is_bitwise_the_cap_chain[20-0]"),
    # v6: K1's window bit collides with DraftOne's DH-D bit in the draftasm slot of the l4_agree gather
    ("C29_k1_bit_collides_with_dhd", K1F, "BIT_K1, BIT_K1CONF, BIT_K1CLOSE, BIT_K1BAD = 8, 16, 32, 64\n",
     "BIT_K1, BIT_K1CONF, BIT_K1CLOSE, BIT_K1BAD = 4, 16, 32, 64\n", 1,
     K6_T + "::test_k1a_k1_bits_ride_draftasm_entry_0_without_collision"),
    # v6: the k1fuse LEVERS entry dropped
    ("C30_levers_k1fuse_dropped", L6AB, '    "k1fuse": "VLLM_TPU_K1FUSE_AB",\n', "", 1, F_T),
    # v6: VRNG's branch disabled, so a VRNG-configured step reaches the K1 hook
    ("C31_vfy_verify_vrng_branch_disabled", RUNNER,
     "        if rng is not None:\n            res = _vfy_one_rng_compiled(*rcall)\n",
     "        if rng is not None and False:\n            res = _vfy_one_rng_compiled(*rcall)\n", 1,
     K6_T + "::test_k1o_the_k1_hook_sits_only_in_the_non_vrng_arm"),
    # v6: H3's flag ignored (capture_model never attaches the kind)
    ("C32_h3_flag_ignored", RUNNER, "            if envs.VLLM_TPU_MK_DRAFT_ATTN:\n", "            if False:\n", 1,
     AW_T + "::test_runner_hook_attaches_only_under_the_flag[1]"),
    # v6: H3 at the wrong B under DRAFT-CAP (the kind built at its largest loop bucket, not the pass's rows)
    ("C33_h3_wrong_b_under_draft_cap", H3A,
     "            return list(self.callable_for(int(block.shape[0]))(block, *args))\n",
     "            return list(self.callable_for(max(self.buckets))(block, *args))\n", 1,
     H6_T + "::test_hc_kind_inside_draft_cap_programs_at_r_d[2-2-16]"),
    # FOLDFIX on v6c2 (2fde9f5e8; FOLDFIX-BUILD's F01-F09 from ee3113e5a, its targets in tests/q38_l4_foldfix)
    ("C34_F01_manager_old_assert_restored", MGR,
     "                    result = self._fold(result, next_tokens_per_chunk, scheduler_output)\n",
     "                    assert isinstance(result, torch.Tensor | list)\n", 1,
     FX_T + "::test_fold_check_to_fold_through_the_manager[asm]"),
    ("C35_F02_consumer_ignores_the_fold_result", RUNNER,
     "            tail, drafts = draft_asm.assemble_tail(\n                self, drafts, next_tokens_per_chunk, state\n"
     "            )\n",
     "            tail, drafts = None, (drafts.drafts() if hasattr(drafts, \"drafts\") else drafts)\n", 1,
     FX_T + "::test_fold_check_to_fold_through_the_manager[asm]"),
    ("C36_F03_manager_passes_the_fold_to_structured_staging", MGR,
     "        if getattr(scheduler_output, \"has_structured_output_requests\", False):\n", "        if False:\n", 1,
     FX_T + "::test_m_forced_fold_result_on_a_structured_step_hands_over_eager_drafts"),
    ("C37_F04_manager_passes_the_fold_to_the_prefill_bootstrap", MGR,
     "        if next_tokens_per_chunk is None:\n            return result.drafts()\n",
     "        if False:\n            return result.drafts()\n", 1,
     FX_T + "::test_m_forced_fold_result_without_verify_output_hands_over_eager_drafts"),
    ("C38_F05_draftasm_folds_structured_steps", DASM,
     "        if getattr(scheduler_output, \"has_structured_output_requests\", False):\n            return \"structured_output\"\n",
     "        if False:\n            return \"structured_output\"\n", 1,
     FX_T + "::test_reach5_a_structured_output_step_never_folds"),
    ("C39_F06_draftasm_folds_under_the_kv_fence", DASM,
     "        elif getattr(runner, \"_fence_spec_kv_under_token_copy\", False):\n            reason = \"kv_fence\"\n",
     "        elif False:\n            reason = \"kv_fence\"\n", 1,
     FX_T + "::test_reach3_a_fenced_kv_producer_never_folds"),
    ("C40_F07_draftasm_folds_sync_steps", DASM,
     "        if next_tokens_per_chunk is None or not return_device:\n            return \"not_async_verify\"\n",
     "        if next_tokens_per_chunk is None:\n            return \"not_async_verify\"\n", 1,
     FX_T + "::test_reach11_a_sync_step_never_folds"),
    ("C41_F08_draftasm_folds_without_verify_output", DASM,
     "        if next_tokens_per_chunk is None or not return_device:\n            return \"not_async_verify\"\n",
     "        if not return_device:\n            return \"not_async_verify\"\n", 1,
     FX_T + "::test_reach7_a_call_without_verify_output_never_folds"),
    ("C42_F09_draftasm_folds_under_the_tp_pin", DASM,
     "        elif self._needs_tp_pin():\n            reason = \"tp_pin\"\n",
     "        elif False:\n            reason = \"tp_pin\"\n", 1,
     FX_T + "::test_reach4_a_replicated_drafter_under_tp_never_folds"),
    # v6 (L4-E): the manager's old assert restored must FAIL K1's take through the real path (K1Result)
    ("C43_manager_old_assert_vs_k1_take", MGR,
     "                    result = self._fold(result, next_tokens_per_chunk, scheduler_output)\n",
     "                    assert isinstance(result, torch.Tensor | list)\n", 1,
     KF_T + "::test_k1_real_path_through_the_manager[cap_nocheck]"),
]


def _replace_nth(s, old, new, nth):
    i = -1
    for _ in range(nth):
        i = s.find(old, i + 1)
        if i < 0:
            return None
    return s[:i] + new + s[i + len(old):]


def _copy(tree, dst):
    for d in ("src", "tests", "q38_l4vfy"):
        if os.path.isdir(os.path.join(tree, d)):
            shutil.copytree(os.path.join(tree, d), os.path.join(dst, d),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("pyproject.toml",):
        if os.path.exists(os.path.join(tree, f)):
            shutil.copy(os.path.join(tree, f), dst)


def _run(tree, target, base_src):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH="src", PYTHONDONTWRITEBYTECODE="1",
               Q38MM_SRC_ROOT=os.path.join(tree, "src"), Q38_V5C_BASE_SRC=base_src)
    for k in [k for k in env if k.startswith(("VLLM_TPU_L4_", "VLLM_TPU_L6_", "VLLM_TPU_DRAFT_ASM", "VLLM_TPU_VPS_VFY"))]:
        del env[k]
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"]
    if target.startswith(H3ENV):  # v6: ATTNWIRE's environment (its node runs: VLLM_TARGET_DEVICE=cpu + q38_cpu_device)
        plugins = os.environ.get("Q38_V6C_PLUGINS", "")
        env.update(VLLM_TARGET_DEVICE="cpu", TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONPATH=f"{plugins}:src:.")
        env.pop("XLA_FLAGS", None)
        cmd += ["-s", "-p", "q38_cpu_device"]
    else:
        cmd += ["--noconftest"] if target.split("::")[0] in NOCONF else ["-p", "no:logging"]
    p = subprocess.run(cmd + [target], cwd=tree, env=env, capture_output=True, text=True, timeout=3600)
    tail = [ln for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR", "E "))
            or "ImportError" in ln][:4]
    return p.returncode, tail


def main():
    tree = os.path.abspath(sys.argv[1])
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(MUTANTS)
    base_src = os.environ.get("Q38_V5C_BASE_SRC", "")
    sel = MUTANTS[lo - 1:hi]
    work = tempfile.mkdtemp(prefix="v5cmut_")
    ctl = os.path.join(work, "control")
    _copy(tree, ctl)
    got = subprocess.run([sys.executable, "-c", "import os, vllm_torchtpu; print(os.path.realpath(vllm_torchtpu.__file__))"],
                         cwd=ctl, env=dict(os.environ, PYTHONPATH="src"), capture_output=True, text=True).stdout.strip()
    print(f"MUTRUN_ORIGIN {got} {'INSIDE' if got.startswith(os.path.realpath(ctl)) else 'OUTSIDE'}")
    control_ok = got.startswith(os.path.realpath(ctl))
    for target in sorted({m[5] for m in sel}):
        rc, tail = _run(ctl, target, base_src)
        print(f"CONTROL {target} rc={rc} {tail}")
        control_ok &= rc == 0
    killed, rows = 0, []
    for name, rel, old, new, nth, target in sel:
        dst = os.path.join(work, name)
        _copy(tree, dst)
        path = os.path.join(dst, rel)
        src = open(path).read()
        bad = _replace_nth(src, old, new, nth)
        if bad is None or bad == src:
            print(f"MUT {name} BAD anchor not found (occurrence {nth}) in {rel}")
            rows.append((name, "BAD"))
            continue
        with open(path, "w") as f:
            f.write(bad)
        rc, tail = _run(dst, target, base_src)
        verdict = "KILLED" if rc == 1 else "SURVIVED" if rc == 0 else "BAD"
        killed += verdict == "KILLED"
        rows.append((name, verdict))
        print(f"MUT {name} {verdict} rc={rc} target={target} {tail}")
        shutil.rmtree(dst, ignore_errors=True)
    n = len(sel)
    ok = killed == n and control_ok
    print(f"Q38_V5C_MUT {'PASS' if ok else 'FAIL'} killed {killed}/{n} of {n} range {lo}-{hi} of {len(MUTANTS)} "
          f"control_ok={int(control_ok)}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
