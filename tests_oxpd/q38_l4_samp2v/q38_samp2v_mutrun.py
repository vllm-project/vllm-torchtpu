# SPDX-License-Identifier: Apache-2.0
"""q38 L4 SAMP2 x VFY-ONE (composed v5 tree) mutation runner (CDK CPU pod only): break a lever deliberately; its target tests must FAIL (a failed
test, not an import or collection error). The pattern of tests/q38_l4_vpshost/q38_vpshost_mutrun.py.

Usage: q38_samp2_mutrun.py <tree> [LO HI]   (LO..HI = a 1-based inclusive slice, for parallel CPU jobs)

Copies src/ and tests/ (and pyproject.toml) to a scratch dir, probes that the copy is the tree imported, runs the
control (every target of the slice passes: 0 failed, 0 skipped, 0 errors), then for each mutant applies one exact text
replacement (the old text must occur exactly once), runs that mutant's targets with -x and requires a non-zero exit
whose summary says "failed", with no collection error and no ERROR line. The original tree is never edited.
Last line: Q38_SAMP2_MUT PASS|FAIL killed N/N of N [range LO-HI of M] control_ok=<bool>
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
RANGE = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None

R = "src/vllm_torchtpu/runner/tpu_runner.py"
VP = "src/vllm_torchtpu/layers/adapter/sample/vp_sampler.py"
VR = "src/vllm_torchtpu/layers/adapter/sample/vps_rng.py"
CTL = "src/vllm_torchtpu/runner/l4_samp2.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
EV = "src/vllm_torchtpu/envs.py"

T = "tests/q38_l4_samp2v/test_l4_samp2.py"
TC2 = "tests/q38_l4_samp2v/test_samp2_vfy.py"
c = lambda name: f"{TC2}::{name}"  # noqa: E731
REGF = "tests/compilation/test_tpu_compile_key_registration.py"
t = lambda name: f"{T}::{name}"  # noqa: E731
REG_HASH = REGF + "::test_graph_flag_moves_the_compile_hash"
REG_HOST = REGF + "::test_host_only_flag_keeps_the_compile_hash"
REG_ALL = REGF + "::test_all_host_only_flags_together_keep_the_compile_hash"
FLAGS = t("test_flags_registered_and_keyed")
S2_CUT = t("test_s2_topk_mask_cutoff_bitwise")
S2_APPLY = t("test_s2_apply_top_k_top_p_bitwise")
S2_FLAG = t("test_s2_candidate_mask_flag_on_off")
S2_TOK = t("test_s2_vps_verify_tokens_bitwise")
S2_IMP = t("test_s2_flag_read_at_import")
V_REF = t("test_vrng_equals_full_draw_reference")
V_DIST = t("test_vrng_uniforms_distribution")
V_INDEP = t("test_vrng_uniforms_independence")
V_OUT = t("test_vrng_accept_reject_outcomes")
V_PATH = t("test_vrng_path_tokens_and_generator")
V_SEED = t("test_vrng_seeded_reproducibility")
V_WARM = t("test_vrng_warmup_calls_and_precompile_keep_the_serving_counter")
V_COLD = t("test_vrng_cold_key_takes_the_base_path")
V_SEEDED = t("test_vrng_seeded_request_runs_the_base_path")
V_WRAP = t("test_vrng_counter_wrap")
V_ALT = t("test_vrng_alt_windows_and_modes")
V_RULES = t("test_vrng_agreement_rules")
V_RANKS = t("test_vrng_eight_ranks_equal_counters_and_tokens")
V_OFFSET = t("test_vrng_one_rank_counter_offset_is_caught")
V_CENSUS = t("test_vrng_aot_census")
V_AL = t("test_vrng_al_canary")
V_CHECK = t("test_vrng_check_mode_uses_the_counter_draws")
V_SITES = t("test_vrng_runner_sites")

C_ONE = c("test_vfy_mode_runs_one_composed_call")
C_IMPL = c("test_vfy_rng_impl_equals_vfy_impl_on_counter_draws")
C_CHAIN = c("test_chain_rng_impl_with_read_offsets_equals_base")
C_OFF = c("test_vrng_window_off_is_bitwise_flag_off")
C_COLD = c("test_cold_base_key_runs_the_chain_vrng_program")
C_NORO = c("test_no_read_offsets_takes_the_base_draws")
C_CHECK = c("test_composed_check")
C_WARM = c("test_warmup_registers_the_composed_keys")
C_SERVE = c("test_served_step_after_warmup_runs_the_composed_program")
C_CENSUS = c("test_vfy_step_census")

MUTANTS = [
    # ---- VFY x VRNG composition (T1-NODEFER-VFY route (a)) ----
    ("C_vfy_path_ignores_vrng", R, "        use_vrng = vrng is not None and vrng.begin(\n            self,\n            self._vfy_vrng_key(",
     "        use_vrng = False and vrng.begin(\n            self,\n            self._vfy_vrng_key(", [C_ONE]),
    ("C_composed_counter_not_kept", R, "            rng.advanced(nxt)\n            rng.add_al(al)\n            res = res[:4] + res[6:]\n",
     "            rng.add_al(al)\n            res = res[:4] + res[6:]\n", [C_ONE]),
    ("C_seed_outputs_misaligned", R, "            res = res[:4] + res[6:]\n            vfy.note(\"vfy_vrng\")\n",
     "            res = res[:4] + res[4:]\n            vfy.note(\"vfy_vrng\")\n", [C_ONE]),
    ("C_check_compares_the_chain_with_itself", R, "                self._vfy_check(res[:4] + res[6:], out, pre, spec)\n",
     "                self._vfy_check((out, self._vps_fb_stats) + tuple(res[2:4]) + tuple(res[6:]), out, pre, spec)\n",
     [C_CHECK]),
    ("C_check_on_the_advanced_counter", R, "                res = _vfy_one_rng_compiled(*rcall)\n                self._vfy_check(",
     "                res = _vfy_one_rng_compiled(*self._vfy_rng_call(call, rng.device_counter()))\n"
     "                self._vfy_check(", [C_CHECK]),
    ("C_warmup_does_not_register", R, "                                vrng.warm_key((\"vfy\",) + _vfy_one.key(*rc[1:]))\n",
     "                                pass\n", [C_WARM]),
    ("C_cold_key_runs_base_chain", R,
     "            return self._vfy_chain(local_cold, args, num_active, fallback_rows, fb_u, ro, rng)\n",
     "            return self._vfy_chain(local_cold, args, num_active, fallback_rows, fb_u, ro)\n", [C_COLD]),
    ("C_meta_key_wrong_width", R, "        return (*call[:8], meta(b, width), meta(*draft.shape), meta(t, width), *call[11:21], fb_u, *call[22:])\n",
     "        return (*call[:8], meta(b, width + 1), meta(*draft.shape), meta(t, width), *call[11:21], fb_u, *call[22:])\n",
     [C_SERVE]),
    ("C_rng_call_keeps_fb_u", R, "        return (*call[:8], counter, *call[11:20], *call[22:])\n",
     "        return (*call[:8], counter, *call[11:20], *call[21:])\n", [C_ONE]),
    ("C_composed_impl_fb_path_off", R, "        num_active,\n        fallback_rows,\n        True,\n        vh.full_vocab,\n        ro_groups,\n",
     "        num_active,\n        fallback_rows,\n        False,\n        vh.full_vocab,\n        ro_groups,\n", [C_IMPL]),
    ("C_chain_rng_ro_operands_dropped", R, "        *_read_offset_scatter_operands(out, ro_num_reqs, ro_groups, ro_dtype),\n        al,\n",
     "        *_read_offset_scatter_operands(out, ro_num_reqs, ro_groups[:1], ro_dtype),\n        al,\n", [C_CHAIN]),
    ("C_no_ro_key_warmable", R, '            return ("vfy", "no_ro")\n', '            return ("anything",)\n', [C_NORO]),
    # ---- S2 (bitwise exact top-k cutoff) ----
    ("S2_kth_index_off_by_one", VP, "(kk - 1).clamp(min=0, max=w - 1)", "kk.clamp(min=0, max=w - 1)", [S2_CUT]),
    ("S2_no_minus_inf_clamp", VP, "    v = torch.clamp(v, min=-_F32_MAX)\n", "", [S2_CUT]),
    ("S2_no_k_nonpositive_clause", VP, '    cutoff = torch.where(kk <= 0, torch.full_like(v, float("inf")), v)\n',
     "    cutoff = v\n", [S2_CUT]),
    ("S2_nan_not_mapped", VP,
     '    x = torch.where(torch.isnan(logits), torch.full_like(logits, float("-inf")), logits)\n', "    x = logits\n",
     [S2_CUT]),
    ("S2_smallest_instead_of_largest", VP, "vals = torch.topk(x, w, dim=-1, largest=True, sorted=True).values",
     "vals = torch.topk(x, w, dim=-1, largest=False, sorted=True).values", [S2_CUT]),
    ("S2_top_p_dropped", VP, "    return torch.where(should_apply_topp, topp_masked, masked_logits)\n",
     "    return masked_logits\n", [S2_APPLY]),
    ("S2_flag_ignored", VP, "    if _TOPK_CUTOFF:\n        kept = _apply_top_k_top_p_cutoff(",
     "    if False:\n        kept = _apply_top_k_top_p_cutoff(", [S2_FLAG]),
    ("S2_import_reads_wrong_env", VP, 'os.environ.get("VLLM_TPU_VPS_TOPK_CUTOFF", "")',
     'os.environ.get("VLLM_TPU_VPS_TOPK_CUT", "")', [S2_IMP]),
    ("S2_not_hashed", TC, '    "VLLM_TPU_VPS_INGRAPH_RNG_AB",\n',
     '    "VLLM_TPU_VPS_INGRAPH_RNG_AB",\n    "VLLM_TPU_VPS_TOPK_CUTOFF",\n', [FLAGS, REG_HASH]),
    ("S2_env_not_registered", EV, '    "VLLM_TPU_VPS_TOPK_CUTOFF": env_bool("VLLM_TPU_VPS_TOPK_CUTOFF", default=False),\n',
     "", [FLAGS]),
    # ---- VRNG: class R distribution (wrong range, correlated streams, wrong count) ----
    ("V_range_0_2", VR, "((h >> 8) & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)",
     "((h >> 8) & 0xFFFFFF).to(torch.float32) * (1.0 / 8388608.0)", [V_DIST]),
    ("V_range_0_half", VR, "((h >> 8) & 0xFFFFFF).to(torch.float32) * (1.0 / 16777216.0)",
     "((h >> 9) & 0x7FFFFF).to(torch.float32) * (1.0 / 16777216.0)", [V_OUT]),
    ("V_streams_shifted_by_counter", VR, "    h = _fmix32(_fmix32(idx ^ k1) + k2)\n",
     "    h = _fmix32(idx + counter)\n", [V_INDEP]),
    ("V_counter_not_advanced", R, "    return out, stats, al, counter + 1\n", "    return out, stats, al, counter + 0\n",
     [V_SEED]),
    ("V_recover_reuses_bonus_region", VP, "    recover_u_c = at(counter, (nb * vocab + na) + rt + target_ci.to(torch.int32))\n",
     "    recover_u_c = at(counter, rt + target_ci.to(torch.int32))\n", [V_REF]),
    ("V_accept_region_overlaps_by_one", VP, "accept_u = at(counter, nb * vocab + torch.arange(",
     "accept_u = at(counter, nb * vocab - 1 + torch.arange(", [V_REF]),
    ("V_fallback_rows_wrong_stride", VP, "    return at(counter, base + rows.to(torch.int32).unsqueeze(-1) * vocab + col)\n",
     "    return at(counter, base + rows.to(torch.int32).unsqueeze(-1) * (vocab - 1) + col)\n", [V_REF]),
    ("V_fallback_target_rows_from_bonus_region", VP,
     "            u_t = _counter_rows(draws, num_bonus * draws[1] + na, t_idx)\n",
     "            u_t = _counter_rows(draws, 0, t_idx)\n", [V_REF]),
    ("V_eager_draw_left_on_vrng_path", R,
     "            bonus_u = accept_u = recover_u = None\n        else:\n            bonus_u = torch.rand(\n"
     "                (padded_num_reqs, vocab),\n                dtype=torch.float32,\n                device=local_logits.device,",
     "            bonus_u = accept_u = recover_u = None\n            torch.rand(1, generator=sampling_generator)\n"
     "        else:\n            bonus_u = torch.rand(\n"
     "                (padded_num_reqs, vocab),\n                dtype=torch.float32,\n                device=local_logits.device,",
     [V_CENSUS]),
    ("C_eager_draw_left_on_the_vfy_path", R,
     "            bonus_u = accept_u = recover_u = None\n        else:\n            bonus_u = torch.rand(\n"
     "                (padded_num_reqs, vocab),\n                dtype=torch.float32,\n                device=device,",
     "            bonus_u = accept_u = recover_u = None\n            torch.rand(1, generator=sampling_generator)\n"
     "        else:\n            bonus_u = torch.rand(\n"
     "                (padded_num_reqs, vocab),\n                dtype=torch.float32,\n                device=device,",
     [C_CENSUS]),
    # ---- VRNG: keys and registration ----
    ("V_not_hashed", TC, '    "VLLM_TPU_VPS_INGRAPH_RNG_AB",\n',
     '    "VLLM_TPU_VPS_INGRAPH_RNG_AB",\n    "VLLM_TPU_VPS_INGRAPH_RNG",\n', [FLAGS, REG_HASH]),
    ("V_ab_hashed_by_mistake", TC,
     '    # q38 L4 VRNG-VPS path switch: routes between programs that are all warmed (host only).\n'
     '    "VLLM_TPU_VPS_INGRAPH_RNG_AB",\n', "", [FLAGS, REG_HOST, REG_ALL]),
    ("V_env_not_registered", EV, '    "VLLM_TPU_VPS_INGRAPH_RNG": env_bool("VLLM_TPU_VPS_INGRAPH_RNG", default=False),\n',
     "", [FLAGS]),
    ("V_runner_ignores_the_flag", R, "                if envs.VLLM_TPU_VPS_INGRAPH_RNG:\n",
     "                if False:\n", [V_SITES]),
    # ---- VRNG: agreement, cross-rank identity (L4-E #1) ----
    ("V_one_rank_counter_offset", CTL, "        self.counter0 = vps_rng.vps_initial_counter(self.seed)\n",
     "        self.counter0 = vps_rng.vps_initial_counter(self.seed) + _tp_rank()\n", [V_RANKS]),
    ("V_counter_not_in_agreement", CTL,
     "        return [int(self.wish()), int(bool(self.local_disabled)), self.fingerprint(), self.counter_word()]\n",
     "        return [int(self.wish()), int(bool(self.local_disabled)), self.fingerprint(), 0]\n", [V_OFFSET]),
    ("V_unequal_counters_ignored", CTL, "        if len(set(ctrs)) != 1:\n", "        if False:\n", [V_OFFSET]),
    ("V_any_wish_suffices", CTL, "        agreed = not self.failed and all(wants)\n",
     "        agreed = not self.failed and any(wants)\n", [V_RULES]),
    ("V_local_disable_ignored", CTL, "        if any(disabled):\n", "        if False:\n", [V_RULES]),
    ("V_unequal_fingerprints_ignored", CTL, "        if len(set(fps)) != 1:\n", "        if False:\n", [V_RULES]),
    ("V_alt_parity_flipped", CTL, "        return k % 2 == 0\n", "        return k % 2 == 1\n", [V_ALT]),
    ("V_agreement_only_at_first_call", CTL, "        return n == 1 or (n - 1) % self.period == 0\n",
     "        return n == 1\n", [V_ALT]),
    ("V_l6_note_missing", CTL, '                l6.note(NOTE, "calls")\n', "                pass\n", [V_ALT]),
    # ---- VRNG: base-path routes (cold, seeded L4-E #2, wrap L4-E #3, warm-up) ----
    ("V_cold_key_runs_vrng", CTL, "        elif key not in self.warm:\n", "        elif False:\n", [V_COLD]),
    ("V_seeded_chunk_runs_vrng", CTL, "        elif seeded_chunk(getattr(runner, \"input_batch\", None), start, end):\n",
     "        elif False:\n", [V_SEEDED]),
    ("V_seeded_window_off_by_one", CTL, "    return any(start <= int(i) < end for i in gens)\n",
     "    return any(start <= int(i) <= end for i in gens)\n", [V_SEEDED]),
    ("V_wrap_guard_removed", CTL, "        elif self.vrng_calls >= WRAP:\n", "        elif False:\n", [V_WRAP]),
    ("V_warmup_calls_count", CTL, "        if not getattr(runner, \"warmup_done\", False):\n            return False\n", "",
     [V_WARM]),
    ("V_seed_ignored", CTL, "        self.seed = int(seed or 0)\n", "        self.seed = 0\n", [V_SEED]),
    ("V_counter_not_kept", CTL, "        self.counter = nxt\n", "        pass\n", [V_PATH]),
    # ---- VRNG: AL canary and check mode ----
    ("V_al_counts_the_bonus", R, "            torch.where(spec, (emitted - 1).clamp(min=0), 0).sum(),\n",
     "            torch.where(spec, emitted, 0).sum(),\n", [V_AL]),
    ("V_al_path_counter_wrong", CTL, '        p = self.path["vrng" if use else "base"]\n', '        p = self.path["vrng"]\n',
     [V_AL]),
    ("V_check_without_the_counter_draws", R, "                args[:5] + full + args[8:],\n", "                args,\n",
     [V_CHECK]),
    ("V_device_counter_mismatch_unseen", R, "            if int(counter.item()) != vrng.counter_word():\n",
     "            if False:\n", [V_CHECK]),
]
CONTROL = [T, REGF]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")
PROBE_MODS = ("vllm_torchtpu.runner.tpu_runner", "vllm_torchtpu.layers.adapter.sample.vp_sampler",
              "vllm_torchtpu.layers.adapter.sample.vps_rng", "vllm_torchtpu.runner.l4_samp2",
              "vllm_torchtpu.compilation.tpu_compiler")


def _env(work):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    for k in list(env):
        if k.startswith(("VLLM_TPU_L4_", "VLLM_TPU_L6_", "VLLM_TPU_VPS_")) or k == "Q38_SAMP2_EXPECT_ROOT":
            env.pop(k)
    return env


def run(work, targets, stop_first=True):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=_env(work), capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="samp2mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    inside = True
    for mod in PROBE_MODS:
        probe = subprocess.run(
            [sys.executable, "-c", f"import importlib, os; m = importlib.import_module('{mod}'); "
             "print(os.path.realpath(m.__file__))"],
            cwd=work, env=_env(work), capture_output=True, text=True, timeout=900)
        where = (probe.stdout.strip().splitlines() or ["none"])[-1]
        ok_mod = where.startswith(os.path.realpath(os.path.join(work, "src")) + os.sep)
        inside = inside and ok_mod
        print(f"MUTRUN_ORIGIN {mod} {where} {'INSIDE' if ok_mod else 'FALLBACK'}")
    ok = True
    if RANGE is not None and not (1 <= RANGE[0] <= RANGE[1] <= len(MUTANTS)):
        print(f"MUTRUN bad range {RANGE} of {len(MUTANTS)}")
        ok = False
    selected = MUTANTS if RANGE is None else MUTANTS[RANGE[0] - 1:RANGE[1]]
    # control: the slice's own targets (each once) all pass on the unmutated copy
    targets = []
    for _, _, _, _, tg in selected:
        for x in tg:
            if x not in targets:
                targets.append(x)
    rc, tail, out = run(work, targets or CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ctl_ok = (rc == 0 and m is not None and int(m.group(1)) > 0
              and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ctl_ok} targets={len(targets)} {tail}")
    if not ctl_ok:
        print("\n".join(out.splitlines()[-40:]))
    ok = ok and ctl_ok
    killed = total = 0
    for name, rel, old, new, tg in selected:
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        if text.count(old) != 1:
            print(f"MUT {name} BAD (anchor occurs {text.count(old)} times)")
            ok = False
            continue
        with open(path, "w") as f:
            f.write(text.replace(old, new))
        try:
            rc, tail, out = run(work, tg)
        finally:
            with open(path, "w") as f:
                f.write(text)
        total += 1
        # killed = the targets ran and an assertion failed: pytest's own summary counts a failure and no error (a
        # setup / collection error is not a kill). v2 m3 j-e8d627a6 used `"ERROR " not in out`, which also matched
        # text the failing test printed (two real failures were scored SURVIVED); the pytest error lines are matched
        # at line start instead, and every survivor prints its evidence.
        pytest_error = bool(re.search(r"^(ERROR tests/|=+ ERRORS =+)", out, re.M)) or re.search(r"\d+ errors?\b", tail)
        dead = rc != 0 and " failed" in tail and "errors during collection" not in out and not pytest_error
        killed += dead
        hits = [ln for ln in out.splitlines() if ln.startswith(("E ", "FAILED", "ERROR"))]
        why = " | " + " / ".join(h.strip()[:160] for h in hits[:3])
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}{why}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and inside and len(selected) > 0 and killed == total == len(selected)
    scope = "" if RANGE is None else f" range {RANGE[0]}-{RANGE[1]} of {len(MUTANTS)}"
    print(f"Q38_SAMP2_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {len(selected)}{scope} "
          f"control_ok={ctl_ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
