# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-CAP mutation runner (CDK CPU pod only; DRAFT1-B's runner engine): break the change deliberately; its
target tests must FAIL (a failed test, not an import or collection error).

Copies src/ and tests/ (and pyproject.toml) to a scratch dir, checks the copy is what imports (the lead's origin rule),
runs the control (unmutated: every target file passes, 0 failed, 0 skipped, 0 errors), then for each mutant applies one
exact text replacement (the old text must occur exactly once), runs that mutant's targets with -x and requires a
non-zero exit whose summary says "failed", with no collection error and no ERROR line. The original tree is never
edited. Exit 0 only if the control passes and every selected mutant is killed that way.
  q38_draftcap_mutrun.py <root> [LO HI]     (1-based inclusive slice of MUTANTS)
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
RANGE = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None
DC = "src/vllm_torchtpu/spec_decode/draft_cap.py"
D1 = "src/vllm_torchtpu/spec_decode/draft_one.py"
ENVS = "src/vllm_torchtpu/envs.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"

T = "tests/q38_draftcap/test_draftcap.py"
F = "tests/q38_draftcap/test_draftcap_flagoff.py"
REGF = "tests/compilation/test_tpu_compile_key_registration.py"
A12 = T + "::test_cap_is_bitwise_capacity_dhd_and_base_over_8_steps[12-False]"
A16 = T + "::test_cap_is_bitwise_capacity_dhd_and_base_over_8_steps[16-False]"
A20 = T + "::test_cap_is_bitwise_capacity_dhd_and_base_over_8_steps[20-False]"
A40 = T + "::test_rd_is_the_capacity_when_the_loop_bucket_exceeds_it[33]"
B0 = T + "::test_each_pass_reads_the_rd_prefix_of_the_capacity_pass[rs0-64]"
C12 = T + "::test_census_rd_shapes_two_draft_programs[12]"
D_PROG = T + "::test_one_rd_program_per_loop_bucket_compiling_only_its_token_buckets"
E_COLD = T + "::test_cold_rd_key_falls_to_stage_a"
F_BUFS = T + "::test_verify_metadata_and_tables_untouched_and_buffers_persist"
G_NEEDS = T + "::test_flag_needs_dhd"
G_FLAG = T + "::test_flag_default_off_registered_and_hashed"
H_SLOTS = T + "::test_more_tables_than_slots_keeps_dhd_off"
H_SHARE = T + "::test_groups_sharing_one_table_share_one_buffer"
K_MARK = T + "::test_engagement_line_once_per_rd_in_serving_only"
FO_D1 = F + "::test_draft_one_flag_off_is_29ea1e98d"
FO_FLAG = F + "::test_flag_default_off_registered_and_keyed"
REG_HASH = REGF + "::test_graph_flag_moves_the_compile_hash[VLLM_TPU_DRAFT_ASM_CAP-1-False]"

BODY = "    return draft_one.dh_one_body(loop, md0=prefix_md(md0, r_d), md1=prefix_md(md1, r_d), bufs=bufs, **kw)\n"
COPY = "            buf.copy_(src[: buf.shape[0]])\n"
CAPROWS = "    return min(p, capacity)\n"
PREFIX = "    return (md[0][:r_d], md[1][: r_d + 1], md[2])\n"
ROWS = "    rows = seed[1] if seed[1].shape[0] == r_d else seed[3]  # [p] gather indices; [cap] rows when p > cap\n"
REPL = "                m, seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2], block_tables=tb\n"

MUTANTS = [
    # --- R_d below the live count
    ("M01_rd_p_minus_one", DC, CAPROWS, "    return min(p, capacity) - 1\n", [A16]),
    ("M02_rd_16_at_R20", DC, CAPROWS, "    return min(16, capacity)\n", [A20]),
    ("M03_rd_not_capped_at_capacity", DC, CAPROWS, "    return p\n", [A40]),
    # --- slot leakage: tables from the wrong offset, a wrong qsl tail, stale rows
    ("M04_table_offset_by_one", DC, COPY, "            buf.copy_(src[1 : buf.shape[0] + 1])\n", [A12]),
    ("M05_table_from_the_end", DC, COPY, "            buf.copy_(src[src.shape[0] - buf.shape[0]:])\n", [A12]),
    ("M06_qsl_tail_zero", DC, PREFIX,
     "    return (md[0][:r_d], torch.nn.functional.pad(md[1][:r_d], (0, 1)), md[2])\n", [A12]),
    ("M07_qsl_tail_from_slot_rd_minus_1", DC, PREFIX,
     "    return (md[0][:r_d], torch.cat([md[1][:r_d], md[1][r_d - 1 : r_d]]), md[2])\n", [A16]),
    ("M08_stale_seq_lens_row_past_R", DC, BODY,
     "    return draft_one.dh_one_body(loop, md0=(torch.cat([md0[0][: r_d - 1], bufs[0][r_d - 1 :].clone()]), "
     "*prefix_md(md0, r_d)[1:]), md1=prefix_md(md1, r_d), bufs=bufs, **kw)\n", [B0]),
    ("M09_table_tail_never_written", DC, COPY, "            buf[:-1].copy_(src[: buf.shape[0] - 1])\n", [B0]),
    ("M10_tables_copied_once", DC, "    copy_table_prefixes(tables, table_bufs)\n",
     "    if not getattr(loop, '_q38_once', False):\n        copy_table_prefixes(tables, table_bufs)\n"
     "        loop._q38_once = True\n", [A12]),
    # --- draft_rows still at the capacity
    ("M11_draft_rows_at_capacity", DC, ROWS, "    rows = seed[3]\n", [C12, B0]),
    # --- md0 / md1 slicing
    ("M12_md1_not_sliced", DC, BODY, BODY.replace("md1=prefix_md(md1, r_d)", "md1=md1"), [A12]),
    ("M13_md1_sliced_from_md0", DC, BODY, BODY.replace("md1=prefix_md(md1, r_d)", "md1=prefix_md(md0, r_d)"), [A12]),
    ("M14_md0_not_sliced", DC, BODY, BODY.replace("md0=prefix_md(md0, r_d)", "md0=md0"), [A12]),
    # --- the forward context's metadata
    ("M15_tables_not_installed", DC, REPL, REPL.replace(", block_tables=tb", ""), [A12]),
    ("M16_one_buffer_for_every_group", DC,
     "            tb = next(b for t, b in zip(tables, tbufs) if t is m.block_tables)\n", "            tb = tbufs[0]\n",
     [A12]),
    ("M17_table_copy_skipped", DC, "    copy_table_prefixes(tables, table_bufs)\n", "    pass\n", [A12]),
    ("M18_seq_lens_buffer_not_installed", DC, REPL, REPL.replace("seq_lens=bufs[0], ", ""), [A12]),
    # --- configuration, the hook, compile keys
    ("M19_cap_without_dhd", D1, "        self.one_cap = bool(envs.VLLM_TPU_DRAFT_ASM_CAP) and self.one_conf\n",
     "        self.one_cap = bool(envs.VLLM_TPU_DRAFT_ASM_CAP)\n", [G_NEEDS]),
    ("M20_cap_branch_skipped", D1, "        if self.one_cap:\n", "        if False:\n", [C12]),
    ("M21_flag_ignored_by_compile_hash", TC, '    "VLLM_TPU_DRAFT_ASM_ONE_AB",\n',
     '    "VLLM_TPU_DRAFT_ASM_ONE_AB",\n    "VLLM_TPU_DRAFT_ASM_CAP",\n', [G_FLAG, REG_HASH]),
    ("M22_flag_default_on", ENVS, '    "VLLM_TPU_DRAFT_ASM_CAP": env_bool("VLLM_TPU_DRAFT_ASM_CAP"),\n',
     '    "VLLM_TPU_DRAFT_ASM_CAP": env_bool("VLLM_TPU_DRAFT_ASM_CAP", True),\n', [FO_FLAG]),
    ("M23_unguarded_draft_one_edit", D1, "        bufs = self._one_buffers(md0)\n",
     "        bufs = tuple(self._one_buffers(md0))\n", [FO_D1]),
    # --- programs, keys, buffers, tables
    ("M24_cold_key_runs", DC, '    if st.mode != "warm" and draft_one._key(*args) not in ctl.warm["one"]:\n',
     "    if False:\n", [E_COLD]),
    ("M25_no_bucket_filter", DC, "        bucket_filter.assign(module, lambda n, serves=serves: n in serves)\n",
     "        pass\n", [D_PROG]),
    ("M26_shared_code_object", DC, '    code = base.__code__.replace(co_name=f"forward_cap_p{p}")\n',
     "    code = base.__code__\n", [D_PROG]),
    ("M27_table_slot_guard_off", DC, "    if len(tables) > TABLE_SLOTS:\n", "    if False:\n", [H_SLOTS]),
    ("M28_table_dedup_off", DC, "        if all(t is not u for u in tables):\n", "        if True:\n", [H_SHARE]),
    ("M29_buffers_not_persistent", DC, "    got = ctl.one_bufs.get(key)\n", "    got = None\n", [F_BUFS]),
    ("M30_body_writes_its_source", DC, COPY, COPY + "            src[:1].fill_(0)\n", [F_BUFS]),
    ("M31_marker_in_warmup", DC, '    if st.mode == "fold" and r_d not in logged:\n', "    if r_d not in logged:\n",
     [K_MARK]),
]

CONTROL = [T, F, REGF]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")
CLEAR = ("VLLM_TPU_DRAFT_ASM_FOLD", "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "VLLM_TPU_DRAFT_ASM_AB",
         "VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "VLLM_TPU_DRAFT_ASM_ONE", "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
         "VLLM_TPU_DRAFT_ASM_ONE_AB", "VLLM_TPU_DRAFT_ASM_CAP")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    for name in CLEAR:
        env.pop(name, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="draftcap_mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib, os; m = importlib.import_module("
         "'vllm_torchtpu.spec_decode.draft_cap'); print(os.path.realpath(m.__file__))"],
        cwd=work, env=dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu"),
        capture_output=True, text=True, timeout=600)
    where = (probe.stdout.strip().splitlines() or ["none"])[-1]
    inside = where.startswith(os.path.realpath(os.path.join(work, "src")) + os.sep)
    print(f"MUTRUN_ORIGIN {where} {'INSIDE' if inside else 'FALLBACK'}")
    ok = True
    if RANGE is not None and not (1 <= RANGE[0] <= RANGE[1] <= len(MUTANTS)):
        print(f"MUTRUN bad range {RANGE} of {len(MUTANTS)}")
        ok = False
    selected = MUTANTS if RANGE is None else MUTANTS[RANGE[0] - 1: RANGE[1]]
    control = sorted({t.split("::")[0] for *_, ts in selected for t in ts}) if RANGE is not None else CONTROL
    rc, tail, out = run(work, control, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    cok = (rc == 0 and m is not None and int(m.group(1)) > 0
           and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={cok} files={len(control)} {tail}")
    if not cok:
        print("\n".join(ln for ln in out.splitlines() if ln.startswith(("FAILED", "ERROR", "E ")))[:4000])
    ok = ok and cok
    killed = total = 0
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
        if rc == 4 or "not found:" in out:
            print(f"MUT {name} BAD (a target node id is not collected: rc={rc})")
            ok = False
            continue
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
    print(f"Q38_DRAFTCAP_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {len(selected)}{scope} "
          f"control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
