#!/usr/bin/env python3
"""q38 L6-MK mutant runner (CDK CPU pod only). Usage: python3 q38_l6_mk_mutrun.py <head tree> <set> [survey]

Modelled on q38_l6_engalloc_mutrun.py. For each mutant: copy the tree, apply one exact textual
replacement (it must match exactly once), run the mutant's target test id and require that it FAILED.
A control run on the unmutated tree must pass every target first. 'survey' reports without requiring
every mutant killed. Last line: Q38_L6_MK_MUT PASS killed N/N of N | FAIL ... | SURVEY ...
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

MK = "src/vllm_torchtpu/runner/l6_mk.py"
DRV = "src/vllm_torchtpu/megakernel_runtime/qwen38_driver.py"
FR = "src/vllm_torchtpu/core/l6_fuse_rpc.py"
AB = "src/vllm_torchtpu/core/l6_ab.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
T = "tests/q38_l6/test_l6_mk.py"
TF = "tests/q38_l6/test_l6_fuse_rpc.py"
REG = "tests/compilation/test_tpu_compile_key_registration.py::test_all_host_only_flags_together_keep_the_compile_hash"
CTX_EQ = T + "::test_mkctx_matches_stock_over_random_streams[0]"
THIN_EQ = T + "::test_mkthin_matches_stock_over_random_streams[0]"
FUSE_EQ = TF + "::test_fused_matches_stock_over_random_streams[0-kv-aggregator]"
KV_EQ = TF + "::test_kvonly_matches_stock_and_skips_the_other_ranks_waits[2]"

MUTANTS = {
    "mk": [
        # ---- mkctx
        ("C1 whole contexts enter the segment tag", MK,
         'lambda: set_model_tag("model_mk_whole")', 'lambda: set_model_tag("model_mk_segments")', CTX_EQ),
        ("C2 whole contexts enter the segment config", MK,
         '    cfg = getattr(driver, "whole_cfg", None)\n    return cfg or', '    cfg = None\n    return cfg or', CTX_EQ),
        ("C3 check_real dropped in the check branch", DRV,
         "            elif self.check_left > 0 and self.shadow is not None and (not self.check_rows or rows in self.check_rows):\n"
         "                check_real = bool(getattr(runner, \"warmup_done\", False)) and _real_traffic(kw, runner, n_req)\n",
         "            elif self.check_left > 0 and self.shadow is not None and (not self.check_rows or rows in self.check_rows):\n"
         "                check_real = False\n", CTX_EQ),
        ("C4 lever state ignored", MK,
         '    return _l6_ab.on(MKCTX) and not driver.__dict__.get("_l6_mkctx_off", False)',
         '    return not driver.__dict__.get("_l6_mkctx_off", False)', T + "::test_mkctx_lever_off_never_builds_the_cache"),
        ("C5 CHECK never reports a mismatch", MK,
         "    if bad:\n        driver._l6_mkctx_off = True", "    if False:\n        driver._l6_mkctx_off = True",
         T + "::test_mkctx_check_catches_a_broken_eligibility_memo"),
        ("C6 CHECK window ignored", MK,
         "    _TURN[lever] = 0\n    return _l6_ab.check_allowed()", "    _TURN[lever] = 0\n    return True",
         T + "::test_mkctx_check_steps_bound_the_check"),
        ("C7 whole-call engagement not counted", MK,
         '    _count("mkctx.whole")', "    pass", CTX_EQ),
        ("C8 eligibility memo kept when the lever is off", DRV,
         "            elif elig if (mkc and elig is not None) else (elig := self._whole_eligible(runner, n_req, rows)):",
         "            elif elig if (elig is not None) else (elig := self._whole_eligible(runner, n_req, rows)):",
         T + "::test_mkctx_off_keeps_the_stock_eligibility_reads"),
        ("C9 mkctx missing from l6_ab.LEVERS", AB, '    "mkctx": "VLLM_TPU_L6_MK_CTX",\n', "",
         T + "::test_levers_registered_and_compile_ignored"),
        ("C10 mkctx flag dropped from the compile-key ignore list", TC, '    "VLLM_TPU_L6_MK_CTX",\n', "", REG),
        # ---- mkthin
        ("T1 runner arguments not permuted", MK,
         "        r = fr(tuple(a) if plan.identity else tuple(a[i] for i in plan.arg_idx))",
         "        r = fr(tuple(a[:len(plan.arg_idx)]))", THIN_EQ),
        ("T2 singleton unwrap ignored", MK,
         "        return _thin_build(plan, a, r[0] if plan.unwrap and isinstance(r, (tuple, list)) else r)",
         "        return _thin_build(plan, a, r)", THIN_EQ),
        ("T3 getitem index dropped", MK, '            outs.append(("r", e.args[1]))', '            outs.append(("r", 0))',
         THIN_EQ),
        ("T4 passthrough output read from the result", MK,
         '            outs.append(("a", pos[e]))', '            outs.append(("x", None))', THIN_EQ),
        ("T5 multi-partition graph accepted", MK,
         "    if len(calls) != 1 or output is None:", "    if not calls or output is None:", T + "::test_mkthin_plans"),
        ("T6 module hooks ignored", MK,
         "            or _hooked(plan.gm, plan.wrapper)\n", "", T + "::test_mkthin_module_hooks_take_the_stock_path"),
        ("T7 recompile_fn accepted", MK,
         '    if d.get("recompile_fn") is not None:\n        return "recompile_fn',
         '    if False:\n        return "recompile_fn', T + "::test_mkthin_refuses_recompile_fn_and_kwargs"),
        ("T8 lever state ignored", MK,
         "        if kwargs or not _l6_ab.on(MKTHIN):", "        if kwargs:",
         T + "::test_mkthin_ab_windows_match_stock"),
        ("T9 CHECK never compares outputs", MK,
         "    if same:\n        _l6_ab.note(MKTHIN", "    if True:\n        _l6_ab.note(MKTHIN",
         T + "::test_mkthin_check_catches_a_broken_plan"),
        ("T10 CHECK never compares arguments", MK,
         "    if type(got) is not tuple or len(got) != len(want) or any(g is not w for g, w in zip(got, want)):",
         "    if False:", T + "::test_mkthin_check_catches_wrong_arguments"),
        ("T11 torch_tpu source guard removed", MK,
         "    if bad:\n        logger.error(\"q38 L6 MK-THIN-TT: NOT installed",
         "    if False:\n        logger.error(\"q38 L6 MK-THIN-TT: NOT installed",
         T + "::test_mkthin_refuses_unverified_sources"),
        ("T12 thin engagement not counted", MK, '        _count("mkthin.thin")\n', "        pass\n", THIN_EQ),
        ("T13 mkthin missing from l6_ab.LEVERS", AB, '    "mkthin": "VLLM_TPU_L6_MK_THIN_TT",\n', "",
         T + "::test_levers_registered_and_compile_ignored"),
        ("T14 mkthin flag dropped from the compile-key ignore list", TC, '    "VLLM_TPU_L6_MK_THIN_TT",\n', "", REG),
        # ---- fuserpc
        ("F1 structured steps fused", FR,
         "            and not scheduler_output.has_structured_output_requests\n", "",
         TF + "::test_grammar_and_deferred_steps_stay_split"),
        ("F2 empty steps fused", FR,
         "            and scheduler_output.total_num_scheduled_tokens > 0\n            and not scheduler_output.pending",
         "            and not scheduler_output.pending", FUSE_EQ),
        ("F3 lever state ignored", FR,
         "            _l6_ab.on(LEVER)\n            and not self.is_pooling_model",
         "            True\n            and not self.is_pooling_model", FUSE_EQ),
        ("F4 worker skips execute_model", FR,
         "        out = self.execute_model(scheduler_output)\n", "        out = None\n", FUSE_EQ),
        ("F5 worker swallows the execute_model error", FR,
         "    if exc is not None and sampled is None:\n        raise exc", "    if False:\n        raise exc",
         TF + "::test_execute_model_failure_raises_the_same_error[kv-aggregator]"),
        ("F6 fused step also sends sample_tokens", FR,
         "        elif fused:  # q38 L6 FUSE-RPC", "        elif False:  # q38 L6 FUSE-RPC", FUSE_EQ),
        ("F7 fused RPC drops the KV output aggregator", FR,
         "        kv_output_aggregator=self.kv_output_aggregator if kv_aggregator is None else kv_aggregator,\n",
         "        kv_output_aggregator=None,\n", FUSE_EQ),
        ("F8 vLLM source guard removed", FR,
         "    if bad:\n        logger.error(\"q38 L6 FUSE-RPC: NOT installed: vLLM sources",
         "    if False:\n        logger.error(\"q38 L6 FUSE-RPC: NOT installed: vLLM sources",
         TF + "::test_install_binds_the_copy_and_refuses_unverified_sources"),
        ("F9 pipeline stages > 1 accepted", FR, "    if pp > 1:\n", "    if False:\n",
         TF + "::test_install_binds_the_copy_and_refuses_unverified_sources"),
        ("F10 copy diverges from vLLM's method", FR,
         "            if len(batch_queue) < self.batch_queue_size and (",
         "            if len(batch_queue) <= self.batch_queue_size and (",
         TF + "::test_copy_is_vllms_method_plus_the_fused_branch"),
        ("F11 fuserpc missing from l6_ab.LEVERS", AB, '    "fuserpc": "VLLM_TPU_L6_FUSE_RPC",\n', "", FUSE_EQ),
        ("F12 fuserpc flag dropped from the compile-key ignore list", TC, '    "VLLM_TPU_L6_FUSE_RPC",\n', "", REG),
        # ---- e2kvonly
        ("K1 one rank drops its KV output", FR,
         "    o.kv_connector_output = mro.kv_connector_output\n", "    o.kv_connector_output = None\n", KV_EQ),
        ("K2 the output rank replies kv-only too", FR,
         'getattr(self, "global_rank", kvonly_rank) != kvonly_rank:', "True:", KV_EQ),
        ("K3 kv-only reply still waits (full reply)", FR,
         "        return kv_only(sampled)\n", "        return sampled\n", KV_EQ),
        ("K4 lever state ignored", FR,
         "    if real is None or not _l6_ab.on(KVONLY) or", "    if real is None or",
         TF + "::test_fused_without_kvonly_keeps_full_replies"),
        ("K5 CHECK never reports a mismatch", FR,
         "        if same:\n            _l6_ab.note(KVONLY, \"check_ok\")", "        if True:\n            _l6_ab.note(KVONLY, \"check_ok\")",
         TF + "::test_kvonly_check_catches_a_dropped_kv_output"),
        ("K6 CHECK compares neither the aggregated output nor the pending counts", FR,
         "        same = (want is None) == (got is None) and (want is None or _out_key(want) == _out_key(got))\n"
         "        same = same and all(",
         "        same = True\n        same = same or all(", TF + "::test_kvonly_check_catches_a_dropped_kv_output"),
        ("K7 e2kvonly missing from l6_ab.LEVERS", AB, '    "e2kvonly": "VLLM_TPU_L6_OUT_KVONLY",\n', "",
         T + "::test_levers_registered_and_compile_ignored"),
        ("K8 e2kvonly flag dropped from the compile-key ignore list", TC, '    "VLLM_TPU_L6_OUT_KVONLY",\n', "", REG),
    ],
}


def run(tree, test_id, env):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-p", "no:logging",
           "-o", "faulthandler_timeout=240", test_id]
    try:
        p = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:  # a hung run is reported (with its partial output), not fatal
        part = (e.stdout or b"") + (e.stderr or b"")
        part = part.decode(errors="replace") if isinstance(part, bytes) else part
        return 124, "TIMEOUT after 900s", part
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
        print("Q38_L6_MK_MUT FAIL control")
        return 1
    killed, names = 0, []
    for name, path, old, new, t in muts:
        d = tempfile.mkdtemp(prefix="l6mkmut.")
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
        print(f"Q38_L6_MK_MUT SURVEY killed {killed}/{tot}: {' '.join(names)}")
        return 0
    if killed == tot:
        print(f"Q38_L6_MK_MUT PASS killed {killed}/{tot} of {tot}")
        return 0
    print(f"Q38_L6_MK_MUT FAIL killed {killed}/{tot} of {tot}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
