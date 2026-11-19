# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-D14-DRAFT-1 mutation runner (CDK CPU pod only): break the change
deliberately; its target tests must FAIL (a failed test, not an import or
collection error).

Copies src/ and tests/ (and pyproject.toml) to a scratch dir, runs the control
(unmutated: every target passes, 0 failed, 0 skipped, 0 errors), then for each
mutant applies one exact text replacement (the old text must occur exactly
once), runs that mutant's targets with -x and requires a non-zero exit whose
summary says "failed", with no collection error and no ERROR line. The
original tree is never edited. Exit 0 only if the control passes and every
mutant is killed that way.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
# Optional 1-based inclusive mutant range (a P4 slice): q38_draft1_mutrun.py <root> LO HI.
RANGE = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None
PQ = "src/vllm_torchtpu/kernels/qwen4_exp/pooled_qsa.py"
KQ = "src/vllm_torchtpu/kernels/qwen4_exp/qsa.py"
OP = "src/vllm_torchtpu/layers/adapter/custom_ops/qsa_op.py"
MQ = "src/vllm_torchtpu/models/vllm/qwen4_exp/qsa.py"
ENVS = "src/vllm_torchtpu/envs.py"
TC = "src/vllm_torchtpu/compilation/tpu_compiler.py"

K = "tests/kernels/qwen4_exp/dcp2/test_dcp2_draft_no_index_state.py"
K_DCP = K + "::test_eight_step_draft_bitwise_dcp"
K_REP = K + "::test_eight_step_draft_bitwise_replicated"
K_CEN_D = K + "::test_census_index_state_kernels_6_to_0[dcp]"
K_CEN_R = K + "::test_census_index_state_kernels_6_to_0[replicated]"
T = "tests/models/vllm/test_qwen4_exp_draft_no_index_state.py"
T_OFF = T + "::test_flag_off_builds_and_calls_the_unchanged_shared_ops"
T_ON = T + "::test_flag_on_draft_takes_noidx_ops_without_the_index_projection"
T_UNB = T + "::test_flag_on_without_a_bound_selection_keeps_the_drafts_own_indexer"
T_TGT = T + "::test_flag_on_leaves_target_layers_unchanged"
T_NEED = T + "::test_flag_needs_shared_selection"
T_DCP = T + "::test_dcp_draft_ops_take_index_state_for_the_shared_ops_only"
T_BLD = T + "::test_shared_op_builders_register_the_noidx_steps"
T_DBLD = T + "::test_draft_dcp_op_runs_the_noidx_step"
T_REF = T + "::test_draft_dcp_rows_op_refuses_no_index_state"
T_MRK = T + "::test_engagement_marker_once_per_process_only_when_engaged"
T_PRD = T + "::test_producer_refuses_the_flag_at_load"
MTP = "src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py"
REGF = "tests/compilation/test_tpu_compile_key_registration.py"
REG = REGF + "::test_graph_flag_moves_the_compile_hash[VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE-1-False]"
REGP = REGF + "::test_flag_is_registered_and_parses[VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE-1-False]"


def noidx_tail(next_def):
    """The ``index_state=False`` keyword closing the noidx step defined before ``next_def``."""
    return f"index_state=False,\n    )\n\n\ndef {next_def}("


MUTANTS = [
    # (a) the draft's index state is still written with the flag on (each noidx step).
    ("A1_noidx_shared_writes_index_state", PQ, noidx_tail("pooled_qsa_shared_rows_noidx_step"),
     noidx_tail("pooled_qsa_shared_rows_noidx_step").replace("False", "True"), [K_CEN_R]),
    ("A2_noidx_shared_rows_writes_index_state", PQ, noidx_tail("pooled_qsa_shared_dcp_noidx_step"),
     noidx_tail("pooled_qsa_shared_dcp_noidx_step").replace("False", "True"), [K_CEN_R]),
    ("A3_noidx_shared_dcp_writes_index_state", PQ,
     noidx_tail("pooled_qsa_shared_rows_dcp_noidx_step"),
     noidx_tail("pooled_qsa_shared_rows_dcp_noidx_step").replace("False", "True"), [K_CEN_D]),
    ("A4_noidx_shared_rows_dcp_writes_index_state", PQ, noidx_tail("pooled_qsa_select_step"),
     noidx_tail("pooled_qsa_select_step").replace("False", "True"), [K_CEN_D]),
    ("A5_index_projection_still_computed", MQ,
     "            hidden if no_index else self.indexer.index_qk_proj(hidden),\n",
     "            (self.indexer.index_qk_proj(hidden), hidden)[1] if no_index"
     " else self.indexer.index_qk_proj(hidden),\n", [T_ON]),
    # (b) the selection is read from the draft's own indexer, not the shared one.
    ("B1_kernel_selects_from_own_index_dcp", PQ,
     "        q, k, v, gate = produce_main(\n"
     "            qgate, key, value, nq, nk, positions, trig, norm_eps, rotary_dim\n"
     "        )\n        iq = raw = None\n",
     "        q, k, v, gate, iq, raw = produce(\n"
     "            qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps, rotary_dim\n"
     "        )\n        shared = None\n", [K_DCP]),
    ("B2_kernel_selects_from_own_index_replicated", PQ,
     "        q, k, v, gate = produce_main(\n"
     "            qgate, key, value, nq, nk, positions, trig, norm_eps, rotary_dim\n"
     "        )\n        iq = raw = None\n",
     "        q, k, v, gate, iq, raw = produce(\n"
     "            qgate, key, value, index_qk, nq, nk, niq, positions, trig, norm_eps, rotary_dim\n"
     "        )\n        shared = None\n", [K_REP]),
    ("B3_layer_routes_noidx_to_own_indexer", MQ,
     "        use_shared = shared is not None and shared_ops is not None and not pick_rows\n",
     "        use_shared = (shared is not None and shared_ops is not None and not pick_rows\n"
     "                      and not getattr(self, \"_shared_no_index_state\", False))\n", [T_ON]),
    # (c) the flag-off path takes the new branch.
    ("C1_layer_ignores_the_flag", MQ,
     "            vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE\n"
     "            and vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_SHARED_SELECTION\n",
     "            vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_SHARED_SELECTION\n", [T_OFF]),
    ("C2_flag_off_builder_registers_noidx", OP, "    if index_state:\n        name = f\"qwen4_exp",
     "    if False:\n        name = f\"qwen4_exp", [T_BLD]),
    ("C3_flag_default_on", ENVS, '"VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE": env_bool("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE"),',
     '"VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE": env_bool("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", default=True),',
     [T_OFF]),
    ("C4_flag_off_passes_hidden", MQ,
     '        no_index = use_shared and getattr(self, "_shared_no_index_state", False)\n',
     "        no_index = use_shared\n", [T_OFF]),
    ("C5_cache_writes_default_no_index", PQ,
     "    compressed_dcp: tuple[jax.Array, int] | None = None,\n    index_state: bool = True,\n",
     "    compressed_dcp: tuple[jax.Array, int] | None = None,\n    index_state: bool = False,\n",
     [K_CEN_D]),
    ("C6_pooled_qsa_default_no_index", PQ,
     "    request_shard: tuple[jax.Array, int] | None = None,\n    index_state: bool = True,\n",
     "    request_shard: tuple[jax.Array, int] | None = None,\n    index_state: bool = False,\n",
     [K_CEN_R]),
    ("C7_flag_compile_ignored", TC, '    "VLLM_TPU_HOST_PHASES",\n',
     '    "VLLM_TPU_HOST_PHASES",\n    "VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE",\n', [REG]),
    ("C8_flag_default_on_registration", ENVS,
     '"VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE": env_bool("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE"),',
     '"VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE": env_bool("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", default=True),',
     [REGP]),
    # (d) the bitwise / routing tests bite on the rest of the change.
    ("D1_noidx_skips_main_kv_write", PQ,
     "    if not index_state:\n"
     "        pool = write_main_cache(pool, rows_packed, page, logical, main_valid, **slot_kw)\n",
     "    if not index_state:\n", [K_REP]),
    ("D2_produce_main_wrong_value", KQ, "    return q, key, value, qgate[..., head_dim:]\n",
     "    return q, key, value * 2, qgate[..., head_dim:]\n", [K_REP]),
    ("D3_target_layers_take_noidx", MQ, '            and "mtp" in prefix\n        )\n', "        )\n",
     [T_TGT]),
    ("D4_noidx_without_shared_selection", MQ,
     "            vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE\n"
     "            and vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_SHARED_SELECTION\n",
     "            vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE\n", [T_NEED]),
    ("D5_dcp_rows_op_gets_no_index", MQ, "                            **(no_index if shared else {}),\n",
     "                            **no_index,\n", [T_DCP]),
    ("D6_dcp_noidx_ops_swapped", OP, "        shared_op, shared_rows_op = _dcp_draft_noidx_ops()\n",
     "        shared_rows_op, shared_op = _dcp_draft_noidx_ops()\n", [T_DBLD]),
    ("D7_rows_guard_dropped", OP,
     '    assert index_state or shared, "index_state=False needs the shared selection"\n', "",
     [T_REF]),
    ("D8_unbound_draft_gets_placeholder", MQ,
     '        no_index = use_shared and getattr(self, "_shared_no_index_state", False)\n',
     '        no_index = getattr(self, "_shared_no_index_state", False)\n', [T_UNB]),
    # (e) the engagement marker (STACK-D v3 entry rule).
    ("E1_marker_dropped", MQ, "                _log_draft_no_index_state(self.tp_rank)\n",
     "                pass\n", [T_MRK]),
    ("E2_marker_unconditional", MQ,
     '            if no_index and getattr(self, "_selection_source", None) is not None:\n',
     "            if True:\n", [T_MRK]),
    ("E3_marker_every_bind", MQ, "    if _Q38_DRAFT1_LOGGED:\n        return\n", "", [T_MRK]),
    # (f) the producer guard (L2 review, audit item 7).
    ("F1_producer_guard_dropped", MTP,
     "        if vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE and self.kv_producer:\n",
     "        if False:\n", [T_PRD]),
    ("F2_kv_both_not_a_producer", MTP,
     '    return getattr(transfer, "kv_role", None) in ("kv_producer", "kv_both")\n',
     '    return getattr(transfer, "kv_role", None) in ("kv_producer",)\n', [T_PRD]),
]

CONTROL = [K, T, REGF]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    for name in ("VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION"):
        env.pop(name, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:logging",
           "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True,
                       timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    work = tempfile.mkdtemp(prefix="draft1mut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    # The lead's origin rule: the mutated copy must be the one imported.
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib, os; m = importlib.import_module("
         "'vllm_torchtpu.kernels.qwen4_exp.pooled_qsa'); print(os.path.realpath(m.__file__))"],
        cwd=work, env=dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu"),
        capture_output=True, text=True, timeout=600)
    where = (probe.stdout.strip().splitlines() or ["none"])[-1]
    inside = where.startswith(os.path.realpath(os.path.join(work, "src")) + os.sep)
    print(f"MUTRUN_ORIGIN {where} {'INSIDE' if inside else 'FALLBACK'}")
    rc, tail, _ = run(work, CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ok = (rc == 0 and m is not None and int(m.group(1)) > 0
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} {tail}")
    killed = total = 0
    selected = MUTANTS if RANGE is None else MUTANTS[RANGE[0] - 1 : RANGE[1]]
    if RANGE is not None and not (1 <= RANGE[0] <= RANGE[1] <= len(MUTANTS)):
        print(f"MUTRUN bad range {RANGE} of {len(MUTANTS)}")
        ok = False
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
        dead = (rc != 0 and " failed" in tail and "errors during collection" not in out
                and "ERROR " not in out)
        killed += dead
        why = ""
        if dead:
            hits = [ln for ln in out.splitlines() if ln.startswith(("E ", "FAILED"))]
            why = " | " + " / ".join(h.strip()[:160] for h in hits[:2])
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}{why}")
    shutil.rmtree(work, ignore_errors=True)
    verdict = ok and inside and len(selected) > 0 and killed == total == len(selected)
    scope = "" if RANGE is None else f" range {RANGE[0]}-{RANGE[1]} of {len(MUTANTS)}"
    print(f"Q38_DRAFT1_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} "
          f"of {len(selected)}{scope} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
