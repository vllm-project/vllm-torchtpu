#!/usr/bin/env python3
"""q38 L6 lever admitinc mutant runner (CPU pod only). Usage: python3 q38_l6_admitinc_mutrun.py <head tree> [survey]

Modelled on q38_l6_engalloc_mutrun.py. For each mutant: copy the tree, apply one exact textual replacement (it
must match exactly once), run the mutant's target test id with pytest and require that it FAILED (a failing test,
not an error or a collection failure). A control run on the unmutated tree must pass every target first. 'survey'
reports every mutant without requiring all to be killed (dev runs).
Last line: Q38_L6_ADMITINC_MUT PASS killed N/N of N | Q38_L6_ADMITINC_MUT FAIL ... | Q38_L6_ADMITINC_MUT SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

AM = "src/vllm_torchtpu/layers/core/attention_metadata.py"
T = "tests/q38_l6/test_l6_admit_inc.py"
RS = T + "::test_admit_pack_equals_full_pack_over_random_streams[1-0]"
DET = T + "::test_count_changes_go_incremental_and_geometry_changes_full"
OFF = T + "::test_lever_off_count_changes_are_full_packs"
BW = T + "::test_batched_row_writes_equal_vllms_loop[mixed-True]"
BWH = T + "::test_batched_row_writes_equal_vllms_loop[hybrid-True]"
E2E = T + "::test_end_to_end_identity_through_the_steady_world[True]"
E2EN = T + "::test_end_to_end_identity_through_the_steady_world[False]"
CHK = T + "::test_check_catches_a_stale_pack_and_keeps_the_full_one"
CHKB = T + "::test_check_catches_a_wrong_batched_write_and_keeps_vllms_rows"
CHKS = T + "::test_check_steps_bound_the_check_turns"
REG = T + "::test_lever_is_registered_and_compile_ignored"
PIN = T + "::test_a_different_blocktable_refuses_the_batched_writes"
PROBE = T + "::test_flag_decides_whether_vllm_is_patched_at_import"
CK = "tests/compilation/test_tpu_compile_key_registration.py::test_all_host_only_flags_together_keep_the_compile_hash"

MUTANTS = [
    # (i) ADMIT-INC: the count-change pack
    ("M1 no zeroing of the vacated rows", AM,
     "            dst[offset + n1 * width : offset + n0 * width] = 0\n", "            pass\n", RS),
    ("M2 dirty marks ignored", AM,
     "        if dirty:\n            for row in dirty:\n                if lo <= row < lo + keep:\n",
     "        if False:\n            for row in dirty:\n                if lo <= row < lo + keep:\n", RS),
    ("M2e dirty marks ignored (end to end, STEADY world)", AM,
     "        if dirty:\n            for row in dirty:\n                if lo <= row < lo + keep:\n",
     "        if False:\n            for row in dirty:\n                if lo <= row < lo + keep:\n", E2E),
    ("M3 a bucket change accepted (layout dropped from the key)", AM,
     "    return (ids, start, num_reqs_padded, tuple(g[2] for g in table_geom), ptr)\n",
     "    return (ids, start, ptr)\n", DET),
    ("M4 the padded count dropped from the key", AM,
     "    return (ids, start, num_reqs_padded, tuple(g[2] for g in table_geom), ptr)\n",
     "    return (ids, start, tuple(g[2] for g in table_geom), ptr)\n", DET),
    ("M5 entering rows rely on dirty marks", AM,
     "            )[:] = source[lo + n0 : lo + n1, :width]\n", "            )[:] = 0\n", DET),
    ("M6 clear() (every row dirty) accepted", AM,
     "    if any(bt.__dict__.get(\"_q38_dirty_all\") for _, bt, _ in entries):\n        return None\n"
     "    lo = ctx.start_index\n",
     "    lo = ctx.start_index\n", DET),
    ("M7 a dummy run accepted", AM,
     "        ctx.position_ids_override is not None\n        or state is None\n        or state[0] is None\n"
     "        or _admit_key(state[0]) != _admit_key(geom)\n",
     "        state is None\n        or state[0] is None\n"
     "        or _admit_key(state[0]) != _admit_key(geom)\n", DET),
    ("M8 the state keeps the old count", AM,
     "    runner._bt_inc_state = (geom, staged_dev)\n    if envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE:\n"
     "        runner._block_table_stage_prev = (table_geom, None, staged_dev, stage_key)\n    st = _ADMIT_STATS\n",
     "    runner._bt_inc_state = (state[0], staged_dev)\n    if envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE:\n"
     "        runner._block_table_stage_prev = (table_geom, None, staged_dev, stage_key)\n    st = _ADMIT_STATS\n", RS),
    ("M9 the gate dropped (lever state ignored)", AM,
     "    return _l6_ab.on(_ADMIT_LEVER) and not _ADMIT_OFF[0]\n", "    return not _ADMIT_OFF[0]\n", OFF),
    ("M10 the CHECK bound dropped", AM,
     "    if n <= 0 or not _l6_ab.check_allowed():\n", "    if n <= 0:\n", CHKS),
    ("M11 the pack CHECK never reports", AM,
     "        if not torch.equal(ref, flat):\n            bad = int((ref != flat).sum())\n",
     "        if False:\n            bad = int((ref != flat).sum())\n", CHK),
    ("M12 the pack CHECK reports but keeps the stale bytes", AM,
     "            flat.copy_(ref)\n", "", CHK),
    ("M13 registration: lever missing from l6_ab.LEVERS", "src/vllm_torchtpu/core/l6_ab.py",
     "    \"admitinc\": \"VLLM_TPU_L6_ADMIT_INC\",\n", "", REG),
    ("M14 registration: flag missing from envs", "src/vllm_torchtpu/envs.py",
     "    \"VLLM_TPU_L6_ADMIT_INC\": env_str(\"VLLM_TPU_L6_ADMIT_INC\", \"\"),\n", "", REG),
    ("M15 registration: flags missing from the compile-key ignore list", "src/vllm_torchtpu/compilation/tpu_compiler.py",
     "    \"VLLM_TPU_L6_ADMIT_INC\",\n    \"VLLM_TPU_L6_ADMIT_INC_CHECK\",\n", "", CK),
    ("M16 the import-time install dropped", AM,
     "if _l6_ab.configured(_ADMIT_LEVER):\n    _admit_batch_install()\n", "", PROBE),
    # (ii) batched row writes
    ("B1 add_row writes no blocks", AM,
     "        bt.num_blocks_per_row[row_idx] = n\n        bt.block_table.np[row_idx, 0:n] = ids\n",
     "        bt.num_blocks_per_row[row_idx] = n\n", BW),
    ("B2 add_row keeps a stale count for an empty list", AM,
     "        if not ids:\n            bt.num_blocks_per_row[row_idx] = 0\n            continue\n",
     "        if not ids:\n            continue\n", BW),
    ("B3 append_row writes from column 0", AM, "        start = nbr[row_idx]\n", "        start = 0\n", BW),
    ("B4 clear_row does not zero the row", AM,
     "            bt.block_table.np[row_idx, :n] = 0\n", "            pass\n", BW),
    ("B5 move_row keeps the source row", AM, "        a[src, :n] = 0\n", "        pass\n", BW),
    ("B6 swap_row swaps the counts only", AM,
     "        bt.block_table.np[src_tgt] = bt.block_table.np[tgt_src]\n", "", BW),
    ("B7 batched writes leave no dirty marks", AM, "    dirty.add(int(row))\n", "    pass\n", BW),
    ("B7e batched writes leave no dirty marks (end to end, no STEADY)", AM, "    dirty.add(int(row))\n",
     "    pass\n", E2EN),
    ("B8 clear_row does not bump the version", AM,
     "        _BLOCK_TABLE_VERSION[0] += 1\n    inc = _STAGE_INC\n    for bt in tables:\n        if inc:\n"
     "            _mark_row(bt, row_idx)\n",
     "        pass\n    inc = _STAGE_INC\n    for bt in tables:\n        if inc:\n"
     "            _mark_row(bt, row_idx)\n", BW),
    ("B9 an all-empty append_row bumps the version", AM,
     "    bumped = False\n    inc = _STAGE_INC\n",
     "    bumped = False\n    inc = _STAGE_INC\n    _BLOCK_TABLE_VERSION[0] += 1\n", BW),
    ("B10 hybrid-block groups take the batched write", AM,
     "        if bt.use_hybrid_blocks:\n            bt.add_row(ids, row_idx)\n            continue\n", "", BWH),
    ("B11 the batched CHECK never reports", AM,
     "        if not (np.array_equal(wa, ga) and np.array_equal(wn, gn)):\n", "        if False:\n", CHKB),
    ("B12 the BlockTable source pin ignored", AM, "    if got != _ADMIT_BT_SRC_SHA1:\n", "    if False:\n", PIN),
]


def run(tree, test_id, env):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging", test_id]
    p = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=1800)
    out = p.stdout + p.stderr
    summ = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error|errors|skipped).* in [0-9.]+s", ln)]
    return p.returncode, (summ[-1] if summ else "NO SUMMARY"), out


def main():
    tree = os.path.abspath(sys.argv[1])
    survey = len(sys.argv) > 2 and sys.argv[2] == "survey"
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH="src")
    env.pop("Q38_L6_EXPECT_TREE", None)
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_"):
            env.pop(k)
    targets = sorted({m[4] for m in MUTANTS})
    ok = True
    for t in targets:
        rc, s, out = run(tree, t, env)
        good = rc == 0 and re.search(r"\b[1-9][0-9]* passed\b", s) and "failed" not in s and "error" not in s
        print(f"CONTROL {'ok' if good else 'BAD'} {t}: {s}", flush=True)
        if not good:
            print(out[-3000:])
            ok = False
    if not ok and not survey:
        print("Q38_L6_ADMITINC_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in MUTANTS:
        d = tempfile.mkdtemp(prefix="l6admut.")
        shutil.copytree(os.path.join(tree, "src"), os.path.join(d, "src"))
        shutil.copytree(os.path.join(tree, "tests"), os.path.join(d, "tests"))
        f = os.path.join(d, path)
        s = open(f).read()
        n = s.count(old)
        if n != 1:
            print(f"MUT {name}: anchor matched {n} times -> INVALID", flush=True)
            shutil.rmtree(d, ignore_errors=True)
            continue
        open(f, "w").write(s.replace(old, new))
        rc, summ, out = run(d, t, env)
        dead = rc != 0 and re.search(r"\b[1-9][0-9]* failed\b", summ) and "error" not in summ
        killed += bool(dead)
        if dead:
            names.append(name.split()[0])
        print(f"MUT {name}: {'KILLED' if dead else 'SURVIVED'} rc={rc} {summ}", flush=True)
        if not dead:
            print(out[-1500:])
        shutil.rmtree(d, ignore_errors=True)
    tot = len(MUTANTS)
    if survey:
        print(f"Q38_L6_ADMITINC_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_ADMITINC_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_ADMITINC_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
