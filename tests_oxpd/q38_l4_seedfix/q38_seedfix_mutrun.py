# SPDX-License-Identifier: Apache-2.0
"""q38 L4 SEEDFIX mutation runner (CDK CPU pod only): break the fix deliberately; its target tests must FAIL (a failed
test, not an import or collection error). The pattern of tests/q38_l4_samp2v/q38_samp2v_mutrun.py (SAMP2-BUILD). Composition tree: the primary mutants plus two composition mutants.

Usage: q38_seedfix_mutrun.py <tree> [LO HI]   (LO..HI = a 1-based inclusive slice, for parallel CPU jobs)

Copies src/ and tests/ (and pyproject.toml) to a scratch dir, probes that the copy is the tree imported, runs the
control (every target of the slice passes: 0 failed, 0 skipped, 0 errors), then for each mutant applies its exact text
replacements (each old text must occur exactly once), runs that mutant's targets with -x and requires a non-zero exit
whose summary says "failed", with no collection error and no ERROR line. The original tree is never edited.
Last line: Q38_SEEDFIX_MUT PASS|FAIL killed N/N of N [range LO-HI of M] control_ok=<bool>
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
SF = "src/vllm_torchtpu/runner/seedfix.py"
PL = "src/vllm_torchtpu/platforms/tpu_platform.py"
CTL = "src/vllm_torchtpu/runner/l4_samp2.py"
T = "tests/q38_l4_seedfix/test_seedfix.py"
t = lambda name: f"{T}::{name}"  # noqa: E731
T2 = "tests/q38_l4_seedfix/test_seedfix_vrng.py"
t2 = lambda name: f"{T2}::{name}"  # noqa: E731

REPRO = lambda r: t(f"test_seeded_reproducible_across_compositions[{r}]")  # noqa: E731
SLOTS = lambda r: t(f"test_seeded_uniforms_in_their_slots[{r}]")  # noqa: E731
UNSEEDED = lambda kind, r: t(f"test_unseeded_bitwise[{kind}-{r}]")  # noqa: E731
FIRST = lambda s: t(f"test_first_token_reproducible[{s}]")  # noqa: E731
PIECE = "        plan.pieces[i] = draw_pieces(gens[i], plan.vocab, plan.k, plan.cw, device)\n"

MUTANTS = [
    # ---- the brief's six ----
    ("seeded_row_draws_from_shared_generator", SF,
     [(PIECE, "        plan.pieces[i] = draw_pieces(runner._get_sampling_generator(), plan.vocab, plan.k, plan.cw, device)\n")],
     [REPRO("vps_fb")]),
    ("per_row_draw_shape_depends_on_batch", SF,
     [(PIECE, "        torch.rand((num_reqs,), generator=gens[i], device=device)\n" + PIECE)],
     [REPRO("vps_fb")]),
    ("generators_advanced_in_rank_dependent_order", SF,
     [(PIECE, "        from vllm.distributed import get_tensor_model_parallel_rank as _r\n"
              "        if _r() % 2:\n            torch.rand((1,), generator=gens[i], device=device)\n" + PIECE)],
     [t("test_eight_ranks_agree")]),
    ("unseeded_rows_moved_onto_request_generators", SF,
     [("    for i in sorted(gens):\n        if i >= num_reqs:\n", "    for i in range(num_reqs):\n        if i >= num_reqs:\n"),
      (PIECE, "        plan.pieces[i] = draw_pieces(gens.get(i, gens[min(gens)]), plan.vocab, plan.k, plan.cw, device)\n")],
     [UNSEEDED("mixed", "vps_fb")]),
    ("first_token_nonspec_site_unfixed", R,
     [("                        if self._seed_plan is not None:\n                            _seedfix.overlay_rows(\n"
       "                                self, self._seed_plan, u, cur_start_idx, cur_end_idx\n                            )\n",
       "")],
     [FIRST("plain")]),
    ("first_token_spec_nonverify_site_unfixed", R,
     [("                            if self._seed_plan is not None:\n                                _seedfix.overlay_rows(\n"
       "                                    self, self._seed_plan, u, cur_start_idx, cur_end_idx\n"
       "                                )\n", "")],
     [FIRST("spec_plain")]),
    ("fallback_row_draws_left_on_shared_generator", R,
     [("                _seedfix.overlay_fb(self, seed_plan, fb_u, fallback_rows)\n", "                pass\n")],
     [SLOTS("vps_cand")]),
    # ---- L4-E: warm-up ----
    ("overlay_not_warmed", SF,
     [("    if not enabled():\n        return\n    warm_pieces(runner)\n    vocab, k, cw =",
       "    return\n    warm_pieces(runner)\n    vocab, k, cw =")],
     [t("test_census_no_new_eager_key_after_warmup")]),
    ("piece_draws_not_warmed", SF,
     [('    if "pieces" in done:\n        return\n', '    if True:\n        return\n')],
     [t("test_census_no_new_eager_key_after_warmup")]),
    ("runner_warmup_call_removed", R,
     [("                    _seedfix.warm_bucket(self, num_tokens, num_reqs, seed_fb_rows)\n"
       "                    # padded_num_reqs-shaped inputs.\n", "                    # padded_num_reqs-shaped inputs.\n")],
     [t("test_runner_warmup_wiring")]),
    # ---- flag off = base ----
    ("admission_ignores_the_flag", PL,
     [("                if not envs.VLLM_TPU_SEEDED_SAMPLING:\n", "                if False:\n")],
     [t("test_admission_on_off")]),
    ("flag_off_still_overlays", SF,
     [("    global _logged_first\n    if not enabled():\n        return None\n", "    global _logged_first\n")],
     [t("test_flag_off_is_base")]),
    ("rewind_not_skipped_with_flag_on", R,
     [("                if generator is not None and not _seedfix.enabled():\n", "                if generator is not None:\n")],
     [t("test_partial_rows[True]")]),
    ("partial_rows_draw", SF,
     [("        if rs.num_computed_tokens + n_sched < rs.num_tokens:\n            partial.append(i)\n        else:\n"
       "            rows.append(i)\n", "        rows.append(i)\n")],
     [t("test_partial_rows[True]")]),
    # ---- every site ----
    ("vps_site_overlay_removed", R,
     [("                _seedfix.overlay_verify(\n                    self, seed_plan, bonus_u, accept_u, recover_u, cur_start_idx, "
       "cur_end_idx, md,\n                    self._vps_cand_draws,\n                )\n                seed_plan.chunk = (cur_start_idx, "
       "cur_end_idx, md, local_logits.dtype)\n", "                pass\n")],
     [SLOTS("vps_fb")]),
    ("stock_bonus_overlay_removed", R,
     [("                    _seedfix.overlay_rows(self, seed_plan, bonus_u, cur_start_idx, cur_end_idx)\n",
       "                    pass\n")],
     [SLOTS("stock")]),
    ("stock_target_overlay_removed", R,
     [("            _seedfix.overlay_targets(\n                self, seed_plan, accept_u, recover_u, cur_start_idx, cur_end_idx, md\n"
       "            )\n", "            pass\n")],
     [SLOTS("stock")]),
    ("fused_site_overlay_removed", R,
     [("            _seedfix.overlay_verify(\n                self, seed_plan, bonus_u, accept_u, recover_u, cur_start_idx, "
       "cur_end_idx, md\n            )\n", "            pass\n")],
     [SLOTS("fused")]),
    # ---- slot arithmetic ----
    ("target_slot_off_by_one", SF,
     [("        t0 = int(np.sum(dl[:b]))\n        p = plan.pieces[i]\n", "        t0 = int(np.sum(dl[:b + 1]))\n        p = plan.pieces[i]\n")],
     [SLOTS("vps_fb")]),
    ("bonus_row_written_to_the_neighbour", SF,
     [("        u.index_put_((dev_index(runner, i - start, 1, u.device),), src.view(1, -1))\n",
       "        u.index_put_((dev_index(runner, min(i - start + 1, u.shape[0] - 1), 1, u.device),), src.view(1, -1))\n")],
     [UNSEEDED("mixed", "stock")]),
    ("fallback_slots_target_rule_from_bonus_rule", SF,
     [("    range_t = ((temps != 0) & ~in_range).tolist()\n", "    range_t = (~(temps <= SAMPLING_EPS) & ~in_range).tolist()\n")],
     [t("test_fallback_slots_match_the_graph")]),
    ("candidate_rows_take_the_full_piece", SF,
     [("        src = plan.pieces[i][3 if cand else 0]\n", "        src = plan.pieces[i][0]\n")],
     [SLOTS("vps_cand")]),
    # ---- composition (VRNG-VPS x VFY-ONE) ----
    ("vfy_site_overlay_removed", R,
     [("                _seedfix.overlay_verify(\n                    self, seed_plan, bonus_u, accept_u, recover_u, cur_start_idx, "
       "cur_end_idx, md,\n                    self._vps_cand_draws,\n                )\n                seed_plan.chunk = (cur_start_idx, "
       "cur_end_idx, md, dtype)\n", "                pass\n")],
     [t2("test_seeded_chunk_base_path_uniforms")]),
    ("vrng_takes_seeded_chunks", CTL,
     [("    gens = getattr(input_batch, \"generators\", None)\n    if not gens:\n        return False\n",
       "    return False\n")],
     [t2("test_seeded_chunk_base_path_uniforms")]),
]
CONTROL = [T, T2]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")
PROBE_MODS = ("vllm_torchtpu.runner.tpu_runner", "vllm_torchtpu.runner.seedfix", "vllm_torchtpu.platforms.tpu_platform",
              "vllm_torchtpu.runner.l4_samp2", "vllm_torchtpu.runner.vfy_one")


def _env(work):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    for k in list(env):
        if k.startswith(("VLLM_TPU_L4_", "VLLM_TPU_L6_", "VLLM_TPU_VPS_")) or k in ("Q38_SEEDFIX_EXPECT_ROOT",
                                                                                  "VLLM_TPU_SEEDED_SAMPLING"):
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
    work = tempfile.mkdtemp(prefix="seedfixmut_")
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
    targets = []
    for _, _, _, tg in selected:
        for x in tg:
            if x not in targets:
                targets.append(x)
    rc, tail, out = run(work, targets or CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ctl_ok = (rc == 0 and m is not None and int(m.group(1)) == len(targets)
              and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ctl_ok} targets={len(targets)} {tail}")
    if not ctl_ok:
        print("\n".join(out.splitlines()[-40:]))
    ok = ok and ctl_ok
    killed = total = 0
    for name, rel, reps, tg in selected:
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        bad = [old for old, _ in reps if text.count(old) != 1]
        if bad:
            print(f"MUT {name} BAD (an anchor occurs {[text.count(o) for o in bad]} times)")
            ok = False
            continue
        new_text = text
        for old, new in reps:
            new_text = new_text.replace(old, new)
        with open(path, "w") as f:
            f.write(new_text)
        try:
            rc, tail, out = run(work, tg)
        finally:
            with open(path, "w") as f:
                f.write(text)
        total += 1
        pytest_error = bool(re.search(r"^(ERROR tests/|=+ ERRORS =+)", out, re.M)) or re.search(r"\d+ errors?\b", tail)
        dead = rc != 0 and " failed" in tail and "errors during collection" not in out and not pytest_error
        killed += dead
        hits = [ln for ln in out.splitlines() if ln.startswith(("E ", "FAILED", "ERROR"))]
        why = " | " + " / ".join(h.strip()[:160] for h in hits[:3])
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}{why}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and inside and len(selected) > 0 and killed == total == len(selected)
    scope = "" if RANGE is None else f" range {RANGE[0]}-{RANGE[1]} of {len(MUTANTS)}"
    print(f"Q38_SEEDFIX_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {len(selected)}{scope} "
          f"control_ok={ctl_ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
