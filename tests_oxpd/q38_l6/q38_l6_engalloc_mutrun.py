#!/usr/bin/env python3
"""q38 L6 lever engalloc mutant runner (CPU pod only). Usage: python3 q38_l6_engalloc_mutrun.py <head tree> <set> [survey]

Modelled on L6-HOST's q38_l6_mutrun.py (q38-l6-trims b5ec67de2 / 4a0cdb6d5). For each mutant: copy the tree,
apply one exact textual replacement (it must match exactly once), run the mutant's target test id with pytest and
require that it FAILED (a failing test, not an error or a collection failure). A control run on the unmutated tree
must pass every target first. 'survey' reports every mutant without requiring all to be killed (dev runs).
Last line: Q38_L6_ENGALLOC_MUT PASS killed N/N of N | Q38_L6_ENGALLOC_MUT FAIL ... | Q38_L6_ENGALLOC_MUT SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

E = "src/vllm_torchtpu/core/l6_engine_alloc.py"
T = "tests/q38_l6/test_l6_engine_alloc.py"
EQ = T + "::test_fast_path_matches_stock_over_random_streams[served-fine-twins]"
EQN = T + "::test_fast_path_matches_stock_over_random_streams[fine-no-twins]"
# Mamba 384 / full attention 512: the two groups' next-full-block bounds bind at different cache lengths
# (with 128-token Mamba / compressed groups every full-attention boundary is also theirs, which masks E5 / E6).
EQS = T + "::test_fast_path_matches_stock_over_random_streams[skew-twins]"
REG = "tests/compilation/test_tpu_compile_key_registration.py"

MUTANTS = {
    "engalloc": [
        ("E1 full-attention span one block loose", E,
         "            if nb * bs < need:\n                need = nb * bs\n",
         "            if nb * bs + bs < need:\n                need = nb * bs + bs\n", EQ),
        ("E2 Mamba span ignores the speculative blocks", E,
         "            t = (nb - m.num_speculative_blocks) * bs\n",
         "            t = nb * bs\n", EQ),
        ("E3 Mamba retire threshold dropped", E,
         "            if nb > r and (r + 1) * bs < p:\n",
         "            if False and nb > r and (r + 1) * bs < p:\n", EQ),
        ("E4 Mamba last-state free threshold dropped", E,
         "            if live and (last + 1) * bs < p:\n",
         "            if False and live and (last + 1) * bs < p:\n", T + "::test_mamba_last_state_bound_is_tight"),
        ("E5 full-attention next-full-block bound off by one", E,
         "                    fh = self._hi(eagle, bs, (nc + 1) * bs - 1)",
         "                    fh = self._hi(eagle, bs, (nc + 1) * bs)", T + "::test_full_attention_full_block_bound_is_tight"),
        ("E6 Mamba next-full-block bound off by one", E,
         "                    mh = self._hi(eagle, bs, (nc + 1) * bs - 1)",
         "                    mh = self._hi(eagle, bs, (nc + 1) * bs)", T + "::test_mamba_full_block_bound_is_tight"),
        ("E7 inserts not replayed", E,
         "        for key, block, nt in c.ins:\n            inner(key, block, nt)\n",
         "        for key, block, nt in c.ins:\n            pass\n", EQ),
        ("E8 coordinator calls do not drop the certificate", E,
         "                    self._drop(k[\"request_id\"] if \"request_id\" in k else a[0])\n",
         "                    pass\n", T + "::test_coordinator_calls_drop_the_certificate"),
        ("E9 certificate active without an observed quiet call", E,
         "        if new is not None:\n            self._set(rid, new)\n",
         "        if new is not None:\n            new.active = True\n            self._set(rid, new)\n", EQ),
        ("E10 check never reports a mismatch", E,
         "            if not (quiet and same and ins_ok):\n",
         "            if False:\n", T + "::test_broken_certificate_is_caught_by_check"),
        ("E11 check turn never comes", E,
         "            self.k = 0\n            return True\n",
         "            self.k = 0\n            return False\n", T + "::test_check_mode_runs_stock_and_finds_no_mismatch"),
        ("E12 lever state ignored", E,
         "        if self.off or not _ab.on(LEVER):\n            return self._alloc(*args)\n",
         "        if self.off:\n            return self._alloc(*args)\n", T + "::test_lever_off_never_takes_the_fast_path"),
        ("E13 lever flag dropped from the compile-key ignore list", "src/vllm_torchtpu/compilation/tpu_compiler.py",
         "    \"VLLM_TPU_L6_ENGINE_ALLOC_FAST\",\n", "",
         REG + "::test_all_host_only_flags_together_keep_the_compile_hash"),
        ("E14 engine-core hook skips begin_step", E,
         "        _ab.begin_step(\"engine_core\")\n        return orig(*args, **kwargs)\n",
         "        return orig(*args, **kwargs)\n", T + "::test_engine_core_hook_flushes_counts_and_begins_the_step"),
        ("E15 engine-core hook does not flush counts", E,
         "        flush()\n        _ab.begin_step(\"engine_core\")\n",
         "        _ab.begin_step(\"engine_core\")\n", T + "::test_engine_core_hook_flushes_counts_and_begins_the_step"),
        ("E16 unmodelled manager accepted", E,
         "            return f\"group {m.kv_cache_group_id}: {t.__name__} is not modelled\"\n",
         "            continue\n", T + "::test_unsupported_manager_is_refused"),
        ("E17 lever missing from l6_ab.LEVERS", "src/vllm_torchtpu/core/l6_ab.py",
         "    \"engalloc\": \"VLLM_TPU_L6_ENGINE_ALLOC_FAST\",\n", "",
         T + "::test_lever_is_registered_and_compile_ignored"),
        ("E18 hash removal does not drop dependent certificates", E,
         "        for rid in list(self.deps.pop(block_id, ())):\n",
         "        for rid in []:\n", T + "::test_hash_removal_on_a_replayed_block_drops_its_certificates"),
        ("E19 reset_prefix_cache keeps the certificates", E,
         "            self.certs.clear()\n            self.deps.clear()\n            r = reset(*a, **k)\n",
         "            r = reset(*a, **k)\n", T + "::test_reset_prefix_cache_drops_all_certificates"),
        ("E21 direct coordinator.cache_blocks does not drop the certificate", E,
         "                    self._drop(request.request_id)\n",
         "                    pass\n", T + "::test_coordinator_calls_drop_the_certificate"),
        # Each bound one too tight (the fast path refuses a quiet call): the boundary tests require it taken.
        ("E22 full-attention next-full-block bound one too tight", E,
         "                    fh = self._hi(eagle, bs, (nc + 1) * bs - 1)",
         "                    fh = self._hi(eagle, bs, (nc + 1) * bs - 2)", T + "::test_full_attention_full_block_bound_is_tight"),
        ("E23 Mamba next-full-block bound one too tight", E,
         "                    mh = self._hi(eagle, bs, (nc + 1) * bs - 1)",
         "                    mh = self._hi(eagle, bs, (nc + 1) * bs - 2)", T + "::test_mamba_full_block_bound_is_tight"),
        ("E24 Mamba last-state bound one too tight", E,
         "            if live and (last + 1) * bs < p:\n                p = (last + 1) * bs\n",
         "            if live and (last + 1) * bs - 1 < p:\n                p = (last + 1) * bs - 1\n",
         T + "::test_mamba_last_state_bound_is_tight"),
        # Each next-full-block bound one too loose on the random stream whose groups' boundaries do not coincide:
        # identity itself must break (the safety side of the bound).
        ("E25 full-attention bound one too loose (skew stream)", E,
         "                    fh = self._hi(eagle, bs, (nc + 1) * bs - 1)",
         "                    fh = self._hi(eagle, bs, (nc + 1) * bs)", EQS),
        ("E26 Mamba bound one too loose (skew stream)", E,
         "                    mh = self._hi(eagle, bs, (nc + 1) * bs - 1)",
         "                    mh = self._hi(eagle, bs, (nc + 1) * bs)", EQS),
        ("E20 inserts not replayed (no twins)", E,
         "        for key, block, nt in c.ins:\n            inner(key, block, nt)\n",
         "        for key, block, nt in c.ins[:1]:\n            inner(key, block, nt)\n", EQN),
        # L6-HOST composition (q38-l6-next): VLLM_TPU_L6_CHECK_STEPS bounds the check turns at both call sites.
        ("E27 allocate_slots check turns ignore VLLM_TPU_L6_CHECK_STEPS", E,
         "\n                if c.active:\n                    if not (self.check and self._check_turn() and _ab.check_allowed()):",
         "\n                if c.active:\n                    if not (self.check and self._check_turn()):",
         T + "::test_check_steps_bound_the_check_turns"),
        ("E28 cache_blocks check turns ignore VLLM_TPU_L6_CHECK_STEPS", E,
         "\n            if c.active:\n                if not (self.check and self._check_turn() and _ab.check_allowed()):",
         "\n            if c.active:\n                if not (self.check and self._check_turn()):",
         T + "::test_check_steps_bound_the_check_turns"),
    ],
}


# The boundary set (L6-HOST's request): E4 / E5 / E6 and their companions alone, for a short CPU check.
MUTANTS["bound"] = [m for m in MUTANTS["engalloc"] if m[0].split()[0] in ("E4", "E5", "E6", "E22", "E23", "E24",
                                                                          "E25", "E26")]


def run(tree, test_id, env):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging", test_id]
    p = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=1500)
    out = p.stdout + p.stderr
    summ = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed|error|errors|skipped).* in [0-9.]+s", ln)]
    return p.returncode, (summ[-1] if summ else "NO SUMMARY"), out


def main():
    tree, which = os.path.abspath(sys.argv[1]), sys.argv[2]
    survey = len(sys.argv) > 3 and sys.argv[3] == "survey"
    muts = MUTANTS[which]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH="src")
    env.pop("Q38_L6_EXPECT_TREE", None)
    for k in list(env):
        if k.startswith("VLLM_TPU_L6_"):
            env.pop(k)
    targets = sorted({m[4] for m in muts})
    ok = True
    for t in targets:
        rc, s, out = run(tree, t, env)
        good = rc == 0 and re.search(r"\b[1-9][0-9]* passed\b", s) and "failed" not in s and "error" not in s
        print(f"CONTROL {'ok' if good else 'BAD'} {t}: {s}")
        if not good:
            print(out[-3000:])
            ok = False
    if not ok and not survey:
        print("Q38_L6_ENGALLOC_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6eamut.")
        shutil.copytree(os.path.join(tree, "src"), os.path.join(d, "src"))
        shutil.copytree(os.path.join(tree, "tests"), os.path.join(d, "tests"))
        f = os.path.join(d, path)
        s = open(f).read()
        n = s.count(old)
        if n != 1:
            print(f"MUT {name}: anchor matched {n} times -> INVALID")
            shutil.rmtree(d, ignore_errors=True)
            continue
        open(f, "w").write(s.replace(old, new))
        rc, summ, out = run(d, t, env)
        dead = rc != 0 and re.search(r"\b[1-9][0-9]* failed\b", summ) and "error" not in summ
        killed += bool(dead)
        if dead:
            names.append(name.split()[0])
        print(f"MUT {name}: {'KILLED' if dead else 'SURVIVED'} rc={rc} {summ}")
        if not dead:
            print(out[-1500:])
        shutil.rmtree(d, ignore_errors=True)
    tot = len(muts)
    if survey:
        print(f"Q38_L6_ENGALLOC_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_ENGALLOC_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_ENGALLOC_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
