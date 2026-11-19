# SPDX-License-Identifier: Apache-2.0
"""q38 L4 VPSHOST mutation runner (CDK CPU pod only): break a lever deliberately; its target tests must FAIL (a
failed test, not an import or collection error).

Usage: q38_vpshost_mutrun.py <tree> [LO HI]   (LO..HI = a 1-based inclusive slice, for parallel CPU jobs)

Copies src/ and tests/ (and pyproject.toml) to a scratch dir, probes that the copy is the tree imported, runs the
control (every target passes: 0 failed, 0 skipped, 0 errors), then for each mutant applies one exact text
replacement (the old text must occur exactly once), runs that mutant's targets with -x and requires a non-zero exit
whose summary says "failed", with no collection error and no ERROR line. The original tree is never edited.
Last line: Q38_VPSHOST_MUT PASS|FAIL killed N/N of N [range LO-HI of M] control_ok=<bool>
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
AB = "src/vllm_torchtpu/core/l6_ab.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
EV = "src/vllm_torchtpu/envs.py"
L4V = "src/vllm_torchtpu/runner/l4_vpshost.py"

T = "tests/q38_l4_vpshost/test_l4_vpshost.py"
REGF = "tests/compilation/test_tpu_compile_key_registration.py"
T_CEN = T + "::test_sampfb_bitwise_and_census"
T_NOFB = T + "::test_sampfb_non_fallback_path_keeps_the_scan"
T_WIN = T + "::test_sampfb_windows_and_engagement"
T_BLK = T + "::test_sampfb_block_sits_between_the_layout_plan_and_the_logits"
T_REG = T + "::test_lever_registered_host_only"
REG_ALL = REGF + "::test_all_host_only_flags_together_keep_the_compile_hash"
T_TOK = T + "::test_sampfb_sampled_tokens_bitwise"
S_CEN = T + "::test_sampsnap_bitwise_and_census"
S_PAD = T + "::test_sampsnap_same_ids_different_padding"
S_PAR = T + "::test_sampsnap_same_ids_changed_params"
S_DT = T + "::test_sampsnap_dtype_in_key"
S_NEWB = T + "::test_sampsnap_new_input_batch_object"
S_FBK = T + "::test_sampsnap_fallback_settings_in_key"
S_CHK = T + "::test_sampsnap_check_mode_catches_a_stale_snapshot"
S_GATE = T + "::test_sampsnap_check_mode_follows_l6_check_allowed"
S_WIN = T + "::test_sampsnap_windows_and_engagement"
A_CEN = T + "::test_statsasync_same_lines_zero_blocking_reads"
A_WAIT = T + "::test_statsasync_line_waits_for_its_copy"
A_WIN = T + "::test_statsasync_windows_flush_before_a_blocking_line"

MUTANTS = [
    # ---- T-2 stage 0 FB-ORDER (sampfb) ----
    ("S0_skips_off_the_fallback_path", R,
     '            vps_skip_topk = self._vps_fb_path and _l6_ab.on("sampfb")\n',
     '            vps_skip_topk = _l6_ab.on("sampfb")\n', [T_NOFB]),
    ("S0_lever_ignored", R,
     '            vps_skip_topk = self._vps_fb_path and _l6_ab.on("sampfb")\n',
     '            vps_skip_topk = self._vps_fb_path and False\n', [T_CEN]),
    ("S0_lever_not_in_LEVERS", AB,
     '    "sampfb": "VLLM_TPU_L4_SAMP_FB_ORDER",\n', '', [T_CEN]),
    ("S0_skip_also_drops_F", R,
     "            if vps_step and spec_decode_metadata is not None and self._vps_fb_path:\n"
     "                # PD DEC-FB-2: the fallback decides eligibility itself",
     "            if vps_step and spec_decode_metadata is not None and self._vps_fb_path and not vps_skip_topk:\n"
     "                # PD DEC-FB-2: the fallback decides eligibility itself", [T_CEN]),
    ("S0_off_path_ignores_vps_step", R,
     "                vps_step\n                and not vps_skip_topk\n",
     "                not vps_skip_topk\n", [T_CEN]),
    ("S0_note_removed", R,
     '                _l6_ab.note("sampfb", "skip")\n', '                pass\n', [T_WIN]),
    ("S0_note_on_non_vps_steps", R,
     "            if vps_skip_topk and vps_step and spec_decode_metadata is not None:\n",
     "            if vps_skip_topk:\n", [T_WIN]),
    ("S0_compile_hashed_by_mistake", TC,
     '    "VLLM_TPU_L4_SAMP_FB_ORDER",\n', '', [T_REG, REG_ALL]),
    ("S0_env_not_registered", EV,
     '    "VLLM_TPU_L4_SAMP_FB_ORDER": env_str("VLLM_TPU_L4_SAMP_FB_ORDER", ""),\n', '', [T_REG]),
    # SAMPFB-TOKEN: a skip that perturbs what the chunk samples must change the tokens and be caught
    ("TOK_skip_also_skips_the_fallback", R,
     "            if vps_step and spec_decode_metadata is not None and self._vps_fb_path:\n"
     "                # PD DEC-FB-2: the fallback decides eligibility itself",
     "            if vps_step and spec_decode_metadata is not None and self._vps_fb_path and not vps_skip_topk:\n"
     "                # PD DEC-FB-2: the fallback decides eligibility itself", [T_TOK]),
    ("TOK_skip_zeroes_the_fallback_rows", R, "                vps_fb = fb_rows or 0\n",
     "                vps_fb = 0 if vps_skip_topk else (fb_rows or 0)\n", [T_TOK]),
    # ---- T-2 stage 1 SAMP-SNAP (sampsnap) ----
    ("SNAP_padding_not_in_key", L4V, "        key = (padded, dtype, device)\n", "        key = (dtype, device)\n", [S_PAD]),
    ("SNAP_dtype_not_in_key", L4V, "        key = (padded, dtype, device)\n", "        key = (padded, device)\n", [S_DT]),
    ("SNAP_not_invalidated_on_add", L4V,
     "        def snap_add_request(*args, **kwargs):\n            snap.invalidate()\n",
     "        def snap_add_request(*args, **kwargs):\n            pass\n", [S_PAR]),
    ("SNAP_new_batch_not_watched", L4V, "        if batch is self._batch:\n            return\n",
     "        if batch is self._batch or self._batch is not None:\n            return\n", [S_NEWB]),
    ("SNAP_fallback_settings_not_in_key", L4V,
     "        return (runner._vps_fb_max, runner._vps_tiegap)\n", "        return 0\n", [S_FBK]),
    ("SNAP_request_ids_not_in_key", L4V, "        key = (tuple(batch.req_ids[s:e]), s, e)\n",
     "        key = (s, e)\n", [S_CEN]),
    ("SNAP_num_active_not_keyed", L4V, "        got = self._nact.get(n)\n",
     "        got = next(iter(self._nact.values()), None)\n", [S_CEN]),
    ("SNAP_lever_ignored_at_params", R, '        if self._vps_fb_path and _l6_ab.on("sampsnap"):\n',
     "        if self._vps_fb_path and False:\n", [S_CEN]),
    ("SNAP_lever_ignored_at_F", R, '        snap = self._samp_snap if _l6_ab.on("sampsnap") else None\n',
     "        snap = None\n", [S_CEN]),
    ("SNAP_lever_off_still_uses_the_snapshot", R,
     '        snap = self._samp_snap if _l6_ab.on("sampsnap") else None\n',
     "        snap = self._samp_snap\n", [S_CEN]),
    ("SNAP_lever_ignored_at_num_active", R,
     '        if _l6_ab.on("sampsnap"):\n            # q38 L4 T-2 stage 1 SAMP-SNAP: one upload per num_active value.\n',
     "        if False:\n            # q38 L4 T-2 stage 1 SAMP-SNAP: one upload per num_active value.\n", [S_CEN]),
    ("SNAP_lever_not_in_LEVERS", AB, '    "sampsnap": "VLLM_TPU_L4_SAMP_SNAP",\n', '', [S_CEN]),
    ("SNAP_check_blind", L4V, "        if _fingerprint(batch, s, e) == ent.fp:\n            return ent\n",
     "        if True:\n            return ent\n", [S_CHK]),
    ("SNAP_check_ignores_l6_check_allowed", L4V,
     "            if self.hits % self.check_every == 0 and _check_allowed():\n",
     "            if self.hits % self.check_every == 0:\n", [S_GATE]),
    ("SNAP_hit_not_counted", L4V, '            _l6_ab.note("sampsnap", "hit")\n', "            pass\n", [S_WIN]),
    ("SNAP_compile_hashed_by_mistake", TC, '    "VLLM_TPU_L4_SAMP_SNAP",\n', '', [T_REG, REG_ALL]),
    ("SNAP_check_env_hashed_by_mistake", TC, '    "VLLM_TPU_L4_SAMP_SNAP_CHECK",\n', '', [REG_ALL]),
    ("SNAP_env_not_registered", EV, '    "VLLM_TPU_L4_SAMP_SNAP": env_str("VLLM_TPU_L4_SAMP_SNAP", ""),\n', '', [T_REG]),
    # ---- T-1(i) STATS-ASYNC (statsasync) ----
    ("T1_blocking_tolist_restored", R, "                pend.issue(self._vps_fb_chunks, self._vps_fb_stats)\n",
     "                self._log_vps_fallback(self._vps_fb_chunks, self._vps_fb_stats.tolist())\n", [A_CEN]),
    ("T1_lever_ignored", R, '            if _l6_ab.on("statsasync"):\n', "            if False:\n", [A_CEN]),
    ("T1_lever_not_in_LEVERS", AB, '    "statsasync": "VLLM_TPU_L4_VPS_STATS_ASYNC",\n', '', [A_CEN]),
    ("T1_line_off_by_a_window", R,
     "            for chunks, counts in pend.ready():\n                self._log_vps_fallback(chunks, counts)\n",
     "            for chunks, counts in pend.ready():\n                self._log_vps_fallback(self._vps_fb_chunks, counts)\n",
     [A_WAIT]),
    ("T1_counts_read_from_the_device_at_drain", L4V, '        host = dev.to("cpu", non_blocking=True)\n',
     "        host = dev\n", [A_CEN]),
    ("T1_drain_ignores_the_event", L4V, "            elif not ev.query():\n                break\n",
     "            elif False:\n                break\n", [A_WAIT]),
    ("T1_window_lost", L4V, "            out.append((tag, host.tolist()))\n", "            pass\n", [A_CEN]),
    ("T1_no_flush_before_a_blocking_line", R,
     "                for chunks, counts in pend.ready(wait=True):\n                    self._log_vps_fallback(chunks, counts)\n",
     "                pass\n", [A_WIN]),
    ("T1_drain_skipped", R, "        if pend.pending:\n", "        if False:\n", [A_CEN]),
    ("T1_issue_not_counted", R, '                _l6_ab.note("statsasync", "issued")\n', "                pass\n", [A_CEN]),
    ("T1_compile_hashed_by_mistake", TC, '    "VLLM_TPU_L4_VPS_STATS_ASYNC",\n', '', [T_REG, REG_ALL]),
    ("T1_env_not_registered", EV,
     '    "VLLM_TPU_L4_VPS_STATS_ASYNC": env_str("VLLM_TPU_L4_VPS_STATS_ASYNC", ""),\n', '', [T_REG]),
]
CONTROL = [T, REGF]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")
PROBE_MODS = ("vllm_torchtpu.runner.tpu_runner", "vllm_torchtpu.core.l6_ab", "vllm_torchtpu.compilation.tpu_compiler",
              "vllm_torchtpu.runner.l4_vpshost")


def _env(work):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    for k in list(env):
        if k.startswith(("VLLM_TPU_L4_", "VLLM_TPU_L6_")) or k == "Q38_VPSHOST_EXPECT_ROOT":
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
    work = tempfile.mkdtemp(prefix="vpshostmut_")
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
    rc, tail, out = run(work, CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ok = (rc == 0 and m is not None and int(m.group(1)) > 0
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} {tail}")
    if not ok:
        print("\n".join(out.splitlines()[-40:]))
    killed = total = 0
    if RANGE is not None and not (1 <= RANGE[0] <= RANGE[1] <= len(MUTANTS)):
        print(f"MUTRUN bad range {RANGE} of {len(MUTANTS)}")
        ok = False
    selected = MUTANTS if RANGE is None else MUTANTS[RANGE[0] - 1:RANGE[1]]
    for name, rel, old, new, targets in selected:
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
            rc, tail, out = run(work, targets)
        finally:
            with open(path, "w") as f:
                f.write(text)
        total += 1
        dead = rc != 0 and " failed" in tail and "errors during collection" not in out and "ERROR " not in out
        killed += dead
        why = ""
        if dead:
            hits = [ln for ln in out.splitlines() if ln.startswith(("E ", "FAILED"))]
            why = " | " + " / ".join(h.strip()[:160] for h in hits[:2])
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}{why}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and inside and len(selected) > 0 and killed == total == len(selected)
    scope = "" if RANGE is None else f" range {RANGE[0]}-{RANGE[1]} of {len(MUTANTS)}"
    print(f"Q38_VPSHOST_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {len(selected)}{scope} "
          f"control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
