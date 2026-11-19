# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT1-B mutation runner (CDK CPU pod only; DRAFT-ASM's runner engine): break the change deliberately;
its target tests must FAIL (a failed test, not an import or collection error).

Copies src/ and tests/ (and pyproject.toml) to a scratch dir, checks the copy is what imports (the lead's
origin rule), runs the control (unmutated: every target passes, 0 failed, 0 skipped, 0 errors), then for
each mutant applies one exact text replacement (the old text must occur exactly once; ``None`` = replace
the whole file with the named reference file), runs that mutant's targets with -x and requires a non-zero
exit whose summary says "failed", with no collection error and no ERROR line. The original tree is never
edited. Exit 0 only if the control passes and every selected mutant is killed that way.
  q38_draft1b_mutrun.py <root> [LO HI]     (1-based inclusive slice of MUTANTS)
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
RANGE = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None
D1 = "src/vllm_torchtpu/spec_decode/draft_one.py"
E3 = "src/vllm_torchtpu/spec_decode/eagle3.py"
ENVS = "src/vllm_torchtpu/envs.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"
REF_E3 = "tests/q38_draft1b/eagle3_ref_9b8d39a74.py.txt"
MTPF = "src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py"

U = "tests/q38_draft1b/test_draft1b_units.py"
S = "tests/q38_draft1b/test_draft1b_stagea.py"
MR = "tests/q38_draft1b/test_draft1b_multirank.py"
F = "tests/q38_draft1b/test_draft1b_flagoff.py"
REGF = "tests/compilation/test_tpu_compile_key_registration.py"
U_ROWS_PAD = U + "::test_carry_rows_is_the_fresh_and_plain_carry[dtype0-40-48]"
U_ROWS_FULL = U + "::test_carry_rows_is_the_fresh_and_plain_carry[dtype0-64-64]"
U_SEED = U + "::test_superset_seed_is_draft_asm_seed_plus_position_carry[False-12]"
U_TOK_DVP = U + "::test_propose_tokens_is_the_carry_token_rule[True]"
U_BODY = U + "::test_carry_loop_body_is_carry_then_loop_then_source[16-64]"
U_AGREE = U + "::test_agreement_bits_pack_and_need_every_rank"
U_CONF = U + "::test_stage_configuration_and_windows"
S_BIT = S + "::test_stage_a_is_bitwise_draft_asm_and_base_over_8_steps[12-False]"
S_BIT64 = S + "::test_stage_a_is_bitwise_draft_asm_and_base_over_8_steps[64-False]"
S_CEN = S + "::test_census_three_draft_programs[8]"
S_COLDCL = S + "::test_cold_carry_loop_key_falls_to_draft_asm_when_kept_warm"
S_PCAP_DHA = S + "::test_loop_bucket_above_capacity_keeps_draft_asm_or_base_with_no_eager_op[True]"
S_PCAP = S + "::test_loop_bucket_above_capacity_keeps_draft_asm_or_base_with_no_eager_op[False]"
S_WARMED = S + "::test_windowed_stage_keeps_draft_asm_warm"
S_WIN = S + "::test_window_switching_per_step_never_compiles_and_stays_bitwise"
S_PRE = S + "::test_superset_seed_precompile_covers_every_decode_call_site"
S_CHK = S + "::test_check_mode_compares_then_arms_stage_a[False]"
S_CHK_DHA = S + "::test_check_mode_compares_then_arms_stage_a[True]"
S_CHK_SRC = S + "::test_check_mode_catches_a_perturbed_stage_a_output[source]"
S_CHK_LOOP = S + "::test_check_mode_catches_a_perturbed_stage_a_output[loop_token]"
S_MARK = S + "::test_engagement_markers"
S_INIT = S + "::test_init_builds_draft_one_only_with_a_stage_flag"
S_VFY = S + "::test_vfy_hooks_name_the_superset_seed"
S_BUILD = S + "::test_a_carry_loop_build_failure_stops"
S_SUBS = S + "::test_host_sub_marks_name_the_folded_programs"
D = "tests/q38_draft1b/test_draft1b_dhd.py"
D_BIT = D + "::test_dhd_is_bitwise_base_over_8_steps[12-False]"
D_PAGES = D + "::test_each_pass_reads_the_base_paths_metadata[12]"
D_UNTOUCHED = D + "::test_dhd_leaves_the_verify_metadata_untouched"
D_PROG = D + "::test_one_program_per_loop_bucket_compiling_only_its_token_buckets"
D_COLD = D + "::test_cold_dhd_key_falls_to_stage_a"
D_DEBUG = D + "::test_debug_checks_keep_dhd_off"
D_UNSERVED = D + "::test_unserved_token_bucket_is_never_called"
D_CHK_SRC = D + "::test_check_mode_catches_a_perturbed_dhd_output[source]"
D_CHK_LOOP = D + "::test_check_mode_catches_a_perturbed_dhd_output[loop_token]"
D_BODY = D + "::test_body_switches_buffers_in_place_and_never_writes_md0_md1"
D_SWITCH = D + "::test_int32_switch_is_set_only_inside_dhd_calls"
D_GLUE = D + "::test_mtp_glue_int32_metadata_has_the_int64_values"
D_AGREE = D + "::test_one_ranks_dhd_wish_off_turns_dhd_off_everywhere"
F_MTP = F + "::test_mtp_meta_i32_off_is_9b8d39a74"
D_BODY_EQ = D + "::test_draft_body_is_the_base_paths_drafts_source_and_rejected[12]"
D_NOHOST = D + "::test_draft_body_traces_with_no_host_read"
MR_WISH = MR + "::test_stage_wish_changes_only_at_boundaries_on_every_rank"
MR_L6W = MR + "::test_l6_lockstep_write_failure_on_the_stage_lever"
F_E3 = F + "::test_eagle3_stage_flags_off_is_9b8d39a74"
F_FLAGS = F + "::test_flags_default_off_registered_and_keyed"
REG_HASH = REGF + "::test_graph_flag_moves_the_compile_hash[VLLM_TPU_DRAFT_ASM_CARRY_LOOP-1-False]"
REG_PARSE = REGF + "::test_flag_is_registered_and_parses[VLLM_TPU_DRAFT_ASM_CARRY_LOOP-1-False]"
REG_HOST = REGF + "::test_host_only_flag_keeps_the_compile_hash[VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB-ab:1+3-]"

MUTANTS = [
    # --- stage A's prologue: the branch-free carry
    ("A01_carry_rows_no_zero_pad", D1,
     "    return torch.where(keep, got, torch.zeros_like(got))\n", "    return got\n", [U_ROWS_PAD]),
    ("A02_carry_rows_clamp_off_by_one", D1,
     "    got = torch.index_select(x, 0, idx.clamp(max=n - 1))\n",
     "    got = torch.index_select(x, 0, idx.clamp(max=n - 2))\n", [U_ROWS_FULL]),
    ("A03_token0_from_the_hidden_carry", D1,
     "    token0 = propose_tokens(loop.mtp, carry_rows(last_hidden, p), loop.dvp)\n    tokens = _MtpFusedLoop.forward(\n"
     "        loop, token0, positions, carry_rows(hidden_states, p), seq_lens,\n",
     "    token0 = propose_tokens(loop.mtp, carry_rows(hidden_states, p), loop.dvp)\n    tokens = _MtpFusedLoop.forward(\n"
     "        loop, token0, positions, carry_rows(hidden_states, p), seq_lens,\n", [U_BODY, S_BIT]),
    ("A04_source_without_token0", D1,
     "        intermediate_tensors, inputs_embeds,\n    )\n"
     "    return (token0, *tokens, fold_source(token0, tokens, tail_bonus, num_reqs, max_reqs))\n",
     "        intermediate_tensors, inputs_embeds,\n    )\n"
     "    return (token0, *tokens, fold_source(tokens[0], tokens[1:], tail_bonus, num_reqs, max_reqs))\n",
     [U_BODY, S_BIT]),
    ("A05_dvp_token_rule_ignored", D1,
     "    if dvp:\n        return mtp.logits_processor.get_top_tokens(mtp.lm_head, hidden).to(torch.int32)\n",
     "    if False:\n        return mtp.logits_processor.get_top_tokens(mtp.lm_head, hidden).to(torch.int32)\n",
     [U_TOK_DVP]),
    # --- the superset seed
    ("B01_step1_positions_not_plus_one", D1,
     "    return (*out, positions_carry, positions_carry + 1)\n",
     "    return (*out, positions_carry, positions_carry)\n", [U_SEED, S_BIT]),
    ("B02_position_carry_wrong_indices", D1,
     "    indices = out[1].clamp_min(0)\n", "    indices = out[3].clamp_min(0)\n", [U_SEED]),
    ("B03_vfy_take_names_draft_asm_seed", D1,
     "            out = p._vfy_seed_take(chunk, nt, _seed_cl_impl, args[1:])\n",
     "            out = p._vfy_seed_take(chunk, nt, draft_asm._async_seed_inputs_asm_impl, args[1:])\n", [S_VFY]),
    ("B04_vfy_spec_drops_positions", D1,
     "        return (*base[:4], seq_lens, positions, base[4], base[5], self.max_reqs), base\n",
     "        return (*base[:4], seq_lens, seq_lens, base[4], base[5], self.max_reqs), base\n", [S_VFY]),
    ("B05_superset_precompile_key_dropped", D1,
     '                    draft_asm.synchronize_tensors(_seed_cl_compiled(*args))\n'
     '                    self.warm["seed"].add(key)\n',
     '                    draft_asm.synchronize_tensors(_seed_cl_compiled(*args))\n', [S_PRE]),
    # --- the fold step's wiring
    ("C01_stage_a_result_not_used", D1,
     '                        st.used = "cl"\n', "                        pass\n", [S_BIT]),
    ("C02_loop_appends_the_source", D1,
     "            draft_tokens_per_chunk[0].extend(out[1:-1])\n",
     "            draft_tokens_per_chunk[0].extend(out[1:])\n", [S_BIT]),
    ("C03_loop_md_keeps_step0_seq_lens", D1,
     "                    m, seq_lens=seq_lens1, query_start_loc=qsl, request_distribution=rd\n",
     "                    m, query_start_loc=qsl, request_distribution=rd\n", [S_BIT]),
    ("C04_loop_md_keeps_step0_query_start_loc", D1,
     "                    m, seq_lens=seq_lens1, query_start_loc=qsl, request_distribution=rd\n",
     "                    m, seq_lens=seq_lens1, request_distribution=rd\n", [S_BIT]),
    ("C05_loop_md_keeps_step0_request_distribution", D1,
     "                    m, seq_lens=seq_lens1, query_start_loc=qsl, request_distribution=rd\n",
     "                    m, seq_lens=seq_lens1, query_start_loc=qsl\n", [S_BIT]),
    ("C06_rejected_count_from_an_earlier_step", D1,
     "            num_rejected=st.seed[SEED_REJ],\n",
     '            num_rejected=self.__dict__.setdefault("_m_rej", st.seed[SEED_REJ]),\n', [S_BIT]),
    ("C07_seq_lens_not_copied_off_fold", D1,
     '        seq_lens1 = args[3] if st.mode == "fold" else args[3].clone()\n',
     "        seq_lens1 = args[3]\n", [S_CHK, S_CHK_DHA]),
    ("C08_never_warmed_carry_loop_key_called", D1,
     '        if st.mode != "warm" and _key(*args) not in self.warm["carry_loop"]:\n            return None\n',
     "", [S_COLDCL]),
    ("C09_loop_bucket_above_capacity_allowed", D1,
     "        if p_rows > last_hidden.shape[0] or p_rows > hidden.shape[0]:\n            return None\n",
     "", [S_PCAP_DHA]),
    ("C10_views_skipped_off_fold", D1,
     '        if st.mode != "fold":\n            st.raw = (last_hidden, hidden)\n            return False\n',
     '        if st.mode != "fold":\n            st.raw = (last_hidden, hidden)\n            return True\n',
     [S_PCAP]),
    # --- agreement and windows (lead safety rule)
    ("D01_stage_agreed_on_any_rank", D1,
     "        cl = self.agreed and self.cl_conf and all(w & BIT_CL for w in wants)\n",
     "        cl = self.agreed and self.cl_conf and any(w & BIT_CL for w in wants)\n", [U_AGREE, MR_WISH]),
    ("D02_stage_wish_ignores_its_window", D1,
     "        if self.cl_conf and self._stage_window(CL_LEVER, self.cl_ab):\n",
     "        if self.cl_conf:\n", [MR_L6W, U_CONF]),
    ("D03_fold_decision_sees_stage_bits", D1,
     "        super().apply([[int(r[0]) & BIT_FOLD, *r[1:]] for r in rows])\n",
     "        super().apply(rows)\n", [U_AGREE]),
    ("D04_stage_used_without_agreement", D1,
     '        return st.mode == "fold" and (self.cl_agreed if stage == "cl" else self.one_agreed)\n',
     '        return st.mode == "fold"\n', [MR_WISH, S_WIN]),
    ("D05_period_ignores_stage_windows", D1,
     "                for lever, ab in ((draft_asm.LEVER, self.ab), (CL_LEVER, self.cl_ab), (ONE_LEVER, self.one_ab))\n",
     "                for lever, ab in ((draft_asm.LEVER, self.ab),)\n", [S_WIN]),
    ("D06_draft_asm_never_kept_warm", D1,
     "        self.dha_needed = not (self.cl_conf and self.cl_ab in _ON)\n",
     "        self.dha_needed = False\n", [S_WARMED, U_CONF]),
    ("D07_build_failure_swallowed", D1,
     "            except Exception as exc:\n                # Fail-stop: every rank runs the stage A program",
     "            except Exception as exc:\n                return None\n                # Fail-stop: every rank runs the stage A program",
     [S_BUILD]),
    # --- CHECK
    ("E01_check_tail_skips_stage_a", D1,
     '        for name, out in (("DRAFT-ASM", st.loop if self.dha_needed else None), ("stage A", st.cl),\n',
     '        for name, out in (("DRAFT-ASM", st.loop if self.dha_needed else None), ("stage A", None),\n',
     [S_CHK_SRC]),
    ("E02_check_skips_stage_a_drafts", D1,
     "        if self.cl_conf:\n            variants.append((\"stage A\"",
     "        if False:\n            variants.append((\"stage A\"", [S_CHK_LOOP]),
    # --- markers, init, flags, flag-off identity, whole revert
    ("F01_engagement_text", D1,
     '            logger.info("q38 DRAFT1-B %s: on (TP rank %d)", "DH-D one program" if st.used == "one"\n',
     '            logger.info("q38 DRAFT1-B %s on (TP rank %d)", "DH-D one program" if st.used == "one"\n',
     [S_MARK]),
    ("F02_sub_mark_name", D1,
     '            host_phases.sub("dl_fold")\n', '            host_phases.sub("dl_call")\n', [S_SUBS]),
    ("F03_init_one_alone_builds_draft_asm", E3,
     "            if envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP or envs.VLLM_TPU_DRAFT_ASM_ONE:\n",
     "            if envs.VLLM_TPU_DRAFT_ASM_CARRY_LOOP:\n", [S_INIT]),
    ("F04_flag_default_on", ENVS,
     '    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP": env_bool("VLLM_TPU_DRAFT_ASM_CARRY_LOOP"),\n',
     '    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP": env_bool("VLLM_TPU_DRAFT_ASM_CARRY_LOOP", default=True),\n',
     [F_FLAGS, REG_PARSE]),
    ("F05_flag_in_compile_ignore_list", TC,
     '    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",\n',
     '    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",\n    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP",\n', [F_FLAGS, REG_HASH]),
    ("F06_window_hashed", TC,
     '    "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",\n', "", [F_FLAGS, REG_HOST]),
    ("F07_unguarded_eagle3_change", E3,
     "                    hidden = self._fit_draft_rows_output(hidden, rows)\n",
     "                    hidden = self._fit_draft_rows_output(hidden, rows)\n                hidden = hidden + 0\n",
     [F_E3]),
    ("F08_revert_eagle3", E3, None, REF_E3, [S_INIT, S_BIT]),
    # --- DH-D: the one program's body (the KLOOP interface)
    ("G01_step0_seq_lens_not_switched", D1, "    slb.copy_(md0[0])\n", "", [D_BODY, D_PAGES]),
    ("G02_loop_query_start_loc_not_switched", D1, "    qb.copy_(md1[1])\n", "", [D_BODY, D_PAGES]),
    ("G03_loop_request_distribution_not_switched", D1, "    rb.copy_(md1[2])\n", "", [D_BODY, D_PAGES]),
    ("G04_loop_advances_md1_not_the_buffer", D1,
     "        loop, token0, positions_step1, carry_rows(hidden, p), slb, intermediate_tensors, inputs_embeds\n",
     "        loop, token0, positions_step1, carry_rows(hidden, p), md1[0], intermediate_tensors, inputs_embeds\n",
     [D_BODY]),
    ("G05_step0_without_draft_rows", D1,
     "        spec_step_idx=0, draft_rows=draft_rows,\n", "        spec_step_idx=0, draft_rows=None,\n", [D_BIT]),
    ("G06_dhd_token0_from_the_hidden", D1,
     "    token0 = propose_tokens(loop.mtp, carry_rows(last_hidden, p), loop.dvp)\n    slb.copy_(md1[0])\n",
     "    token0 = propose_tokens(loop.mtp, carry_rows(hidden, p), loop.dvp)\n    slb.copy_(md1[0])\n", [D_BIT]),
    # --- DH-D: the controller
    ("H01_never_warmed_one_key_called", D1,
     '        if st.mode != "warm" and _key(*args) not in self.warm["one"]:\n            return None, "one_cold"\n',
     "", [D_COLD]),
    ("H02_unserved_token_bucket_called", D1,
     '        if input_ids.shape[0] not in self._one_serves(p_rows):\n            return None, "one_bucket"\n',
     "", [D_UNSERVED]),
    ("H03_dhd_result_not_used", D1, '        st.used = "one"\n', "", [D_BIT]),
    ("H04_debug_checks_ignored", D1,
     '            return None, "one_debug_checks"  # those checks read draft step 0\'s real outputs\n',
     "            pass\n", [D_DEBUG]),
    ("H05_buffers_alias_the_verify_metadata", D1,
     "        bufs = self.one_bufs.get(key)\n", "        bufs = tuple(md0)\n", [D_UNTOUCHED]),
    ("H06_shared_program_class", D1,
     '    return type(f"DraftOneProgramP{p}", (_DraftOneProgram,), {"forward": fwd})\n',
     "    return _DraftOneProgram\n", [D_PROG]),
    ("H07_bucket_filter_not_assigned", D1,
     "            bucket_filter.assign(module, lambda n, serves=serves: n in serves)\n", "", [D_PROG]),
    ("H08_dhd_agreed_on_the_stage_a_bit", D1,
     "        one = self.agreed and self.one_conf and all(w & BIT_ONE for w in wants)\n",
     "        one = self.agreed and self.one_conf and all(w & BIT_CL for w in wants)\n", [D_AGREE]),
    ("H09_check_tail_skips_dhd", D1,
     '                          ("DH-D", st.one)):\n', '                          ("DH-D", None)):\n', [D_CHK_SRC]),
    ("H10_check_skips_dhd_drafts", D1,
     '            variants.append(("DH-D", None if st.one is None else list(st.one[:-1])))\n', "            pass\n",
     [D_CHK_LOOP]),
    # --- X-D14-DRAFT-3 (a): the int32 trace-time switch
    ("I01_switch_never_set", D1, "    model.meta_i32 = True\n", "", [D_SWITCH]),
    ("I02_switch_not_restored", D1, "        model.meta_i32 = old\n", "        pass\n", [D_SWITCH]),
    # --- the frozen K-step interface (draft_body)
    ("K01_draft_body_reads_the_host", D1,
     "    return torch.stack(out[:-1], dim=1), out[-1], seed[SEED_REJ]\n",
     "    return torch.stack(out[:-1], dim=1), out[-1], seed[SEED_REJ] + 0 * int(seed[SEED_REJ].sum())\n",
     [D_NOHOST]),
    ("K02_draft_body_step0_metadata_from_the_seed", D1,
     "        md0=(seq_lens, query_start_loc, request_distribution),\n",
     "        md0=(seed[SEED_SEQ1], query_start_loc, request_distribution),\n", [D_BODY_EQ]),
    ("I03_glue_dtype_swapped", MTPF,
     "            torch.int32 if self.meta_i32 else torch.int64\n        )\n",
     "            torch.int64 if self.meta_i32 else torch.int32\n        )\n", [D_GLUE, F_MTP]),
]

CONTROL = [U, S, MR, F, D, REGF]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")
CLEAR = ("VLLM_TPU_DRAFT_ASM_FOLD", "VLLM_TPU_DRAFT_ASM_CHECK_STEPS", "VLLM_TPU_DRAFT_ASM_AB",
         "VLLM_TPU_DRAFT_ASM_CARRY_LOOP", "VLLM_TPU_DRAFT_ASM_ONE", "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB",
         "VLLM_TPU_DRAFT_ASM_ONE_AB")


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
    work = tempfile.mkdtemp(prefix="draft1b_mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib, os; m = importlib.import_module("
         "'vllm_torchtpu.spec_decode.draft_one'); print(os.path.realpath(m.__file__))"],
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
        if old is None:
            with open(os.path.join(work, new)) as f:
                mutated = f.read()
        elif text.count(old) != 1:
            print(f"MUT {name} BAD (anchor occurs {text.count(old)} times)")
            ok = False
            continue
        else:
            mutated = text.replace(old, new)
        with open(path, "w") as f:
            f.write(mutated)
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
    print(f"Q38_DRAFT1B_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {len(selected)}{scope} "
          f"control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
