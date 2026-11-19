# SPDX-License-Identifier: Apache-2.0
"""q38 L4 X-D14-DRAFT-2 (c) mutation runner: break the code deliberately; the targeted tests must
FAIL (a failed test, not a collection / import error).

Usage: python3 tests/q38_draftemb/q38_draftemb_mutrun.py <tree> [LO HI]
Copies src/ and tests/ of <tree> to a scratch dir, checks that the copy is what the tests import
(MUTRUN_ORIGIN ... INSIDE), runs the control (every target passes; 0 failed / skipped / errors),
then for each mutant in LO..HI (1-based, inclusive; all by default) applies one exact text
replacement (the old text must occur exactly once), runs that mutant's targets and requires a
non-zero exit whose summary says "failed" with no collection error. <tree> is never edited.
Last line: Q38_DRAFTEMB_MUT PASS|FAIL killed K/N of N [range LO-HI of TOTAL] control_ok=...
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
MTP = "src/vllm_torchtpu/models/vllm/qwen4_exp/mtp.py"
ENVS = "src/vllm_torchtpu/envs.py"
UTILS = "src/vllm_torchtpu/spec_decode/utils.py"
ASM = "src/vllm_torchtpu/spec_decode/draft_asm.py"
COMP = "src/vllm_torchtpu/compilation/tpu_compiler.py"
INIT = "src/vllm_torchtpu/__init__.py"
T = "tests/q38_draftemb/test_draft_embed_repl.py"
LOOK = T + "::test_replicated_lookup_is_the_tp_summed_stock_lookup"
REPLAY = T + "::test_replay_stock_vs_replicated_is_bitwise"
CEN = T + "::test_census_draft_step_embedding_all_reduces"
ON = T + "::test_unanimous_on_binds_the_gathered_table"
OFF = T + "::test_unanimous_off_agrees_and_assembles_nothing"
DIV = T + "::test_divergent_flag_stops_every_rank"
GFAIL = T + "::test_failed_agreement_gather_stops"
UNBOUND = T + "::test_flag_on_without_a_bound_table_refuses_to_trace"
QUANT = T + "::test_quantized_embedding_is_refused"
TP1 = T + "::test_tp1_embedding_is_left_alone"
HOOK = T + "::test_bind_shared_selection_runs_the_replicated_bind_first"
BUILD = T + "::test_flag_default_and_build_time_read"
REG = ("tests/compilation/test_tpu_compile_key_registration.py::"
       "test_graph_flag_moves_the_compile_hash[VLLM_TPU_DRAFT_EMBED_REPL-1-False]")
TT = "tests/q38_draftemb/test_target_embed_repl.py"
TBIT = TT + "::test_target_lookup_is_bitwise_including_signed_zero"
TCEN = TT + "::test_census_target_embedding_all_reduce"
TBIND = TT + "::test_bind_shares_the_draft_table_only_with_the_target_flag"
TREF = TT + "::test_target_flag_without_the_draft_flag_is_refused"
TDIV = TT + "::test_divergent_target_flag_stops_every_rank"
TBUILD = TT + "::test_target_flag_default_and_build_time_read"
TREG = ("tests/compilation/test_tpu_compile_key_registration.py::"
        "test_graph_flag_moves_the_compile_hash[VLLM_TPU_TARGET_EMBED_REPL-1-False]")

MUTANTS = [
    # --- the replicated path itself
    ("E01_flag_ignored_in_embed", MTP,
     '        if getattr(self, "embed_repl", False) and int(getattr(emb, "tp_size", 1)) > 1:\n',
     '        if False and int(getattr(emb, "tp_size", 1)) > 1:\n', [CEN, REPLAY]),
    ("E02_lookup_without_range_mask", MTP,
     "    return rows.masked_fill(~valid.unsqueeze(-1), 0)\n", "    return rows\n", [LOOK, REPLAY]),
    ("E03_lookup_upper_bound_dropped", MTP,
     "    valid = (ids >= 0) & (ids < table.shape[0])\n", "    valid = ids >= 0\n", [LOOK]),
    ("E04_lookup_lower_bound_dropped", MTP,
     "    valid = (ids >= 0) & (ids < table.shape[0])\n", "    valid = ids < table.shape[0]\n", [LOOK]),
    ("E05_lookup_in_float32", MTP,
     "    rows = torch.nn.functional.embedding(ids * valid, table)\n",
     "    rows = torch.nn.functional.embedding(ids * valid, table.float())\n", [LOOK, REPLAY]),
    ("E06_h8_takes_precedence", MTP,
     '        if getattr(self, "embed_repl", False) and int(getattr(emb, "tp_size", 1)) > 1:\n',
     '        if getattr(self, "embed_repl", False) and not self.embed_h8 and int(getattr(emb, "tp_size", 1)) > 1:\n',
     [CEN, REPLAY]),
    ("E07_unbound_table_falls_back_to_stock", MTP,
     "            if table is None:\n                raise RuntimeError(\n",
     "            if table is None:\n                return emb(input_ids)\n                raise RuntimeError(\n", [UNBOUND]),
    # --- the bind
    ("E08_bind_this_ranks_rows_only", MTP,
     "        full, org_vocab, dim, _ = gather_sharded_weight(emb)\n",
     "        full = torch.zeros(emb.org_vocab_size, emb.embedding_dim, dtype=emb.weight.dtype)\n"
     "        full[emb.shard_indices.org_vocab_start_index:emb.shard_indices.org_vocab_end_index] = "
     "emb.weight.data[:emb.shard_indices.org_vocab_end_index - emb.shard_indices.org_vocab_start_index]\n"
     "        org_vocab, dim = emb.org_vocab_size, emb.embedding_dim\n", [ON, REPLAY]),
    ("E09_gather_helper_drops_the_rows", UTILS,
     "    full[si.org_vocab_start_index : si.org_vocab_end_index] = my_rows\n",
     "    full[si.org_vocab_start_index : si.org_vocab_end_index] = my_rows * 0\n", [ON, LOOK]),
    ("E10_quantized_accepted", MTP,
     '            hasattr(emb, "weight_scale")\n            or weight is None\n'
     "            or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)\n",
     "            weight is None\n", [QUANT]),
    ("E11_tp1_binds", MTP,
     '        if emb is None or int(getattr(emb, "tp_size", 1) or 1) <= 1:\n',
     "        if emb is None:\n", [TP1]),
    ("E12_hook_not_called", MTP,
     "        self.bind_replicated_embedding()\n        if vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE",
     "        if vllm_torchtpu_envs.VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE", [HOOK]),
    # --- the all-rank agreement (fail-stop)
    ("E13_agreement_skipped", MTP,
     "        votes = agree_draft_embed_repl(on, tgt)\n", "        votes = [(int(on), int(tgt))]\n", [DIV, OFF]),
    ("E14_off_ranks_skip_agreement", MTP,
     '        tgt = bool(getattr(model, "embed_repl_tgt", False))\n',
     '        tgt = bool(getattr(model, "embed_repl_tgt", False))\n        if not on and not tgt:\n            return False\n',
     [DIV, OFF]),
    ("E15_divergence_not_raised", MTP,
     "    if len(set(votes)) != 1:\n", "    if False:\n", [DIV]),
    ("E16_failed_gather_swallowed", ASM,
     "        torch.distributed.all_gather(out, t, group=g.cpu_group)\n    except Exception as exc:\n        raise RuntimeError(\n",
     "        torch.distributed.all_gather(out, t, group=g.cpu_group)\n    except Exception as exc:\n        return [list(vec)]\n        raise RuntimeError(\n",
     [GFAIL]),
    # --- flag registration
    ("E17_flag_compile_ignored", COMP,
     '    "VLLM_TPU_HOST_PHASES",\n', '    "VLLM_TPU_HOST_PHASES",\n    "VLLM_TPU_DRAFT_EMBED_REPL",\n', [REG]),
    ("E18_flag_default_on", ENVS,
     '    "VLLM_TPU_DRAFT_EMBED_REPL": env_bool("VLLM_TPU_DRAFT_EMBED_REPL", default=False),\n',
     '    "VLLM_TPU_DRAFT_EMBED_REPL": env_bool("VLLM_TPU_DRAFT_EMBED_REPL", default=True),\n', [BUILD]),
    ("E19_build_time_read_dropped", MTP,
     "        self.embed_repl = bool(vllm_torchtpu_envs.VLLM_TPU_DRAFT_EMBED_REPL)\n",
     "        self.embed_repl = False\n", [BUILD, CEN]),
    # --- whole-lever revert of the embed branch: the census must read 3 all-reduces again
    ("E20_revert_embed_branch", MTP,
     '            return replicated_lookup(table, input_ids)\n',
     '            return emb(input_ids)\n', [CEN, REPLAY]),
    # --- EMB-REP-TGT (VLLM_TPU_TARGET_EMBED_REPL)
    ("T01_patched_forward_ignores_table", INIT,
     "        if table is not None:\n            # q38 L4 EMB-REP-TGT",
     "        if False:\n            # q38 L4 EMB-REP-TGT", [TBIT, TCEN]),
    ("T02_table_not_bound_on_target", MTP,
     '            emb.register_buffer("_q38_repl_table", full, persistent=False)\n', "            pass\n", [TBIND]),
    ("T03_target_bound_with_flag_off", MTP,
     "        if tgt:\n            # The same tensor", "        if True:\n            # The same tensor", [TBIND]),
    ("T04_target_without_draft_accepted", MTP,
     "        if tgt and not on:\n            raise ValueError(", "        if False:\n            raise ValueError(", [TREF]),
    ("T05_agreement_ignores_target_vote", MTP,
     "    votes = [tuple(int(x) for x in v) for v in _gather([int(bool(on)), int(bool(target))])]\n",
     "    votes = [tuple(int(x) for x in v)[:1] for v in _gather([int(bool(on)), int(bool(target))])]\n", [TDIV]),
    ("T06_target_gets_a_copy", MTP,
     '            emb.register_buffer("_q38_repl_table", full, persistent=False)\n',
     '            emb.register_buffer("_q38_repl_table", full.clone(), persistent=False)\n', [TBIND]),
    ("T07_target_flag_compile_ignored", COMP,
     '    "VLLM_TPU_HOST_PHASES",\n', '    "VLLM_TPU_HOST_PHASES",\n    "VLLM_TPU_TARGET_EMBED_REPL",\n', [TREG]),
    ("T08_target_flag_default_on", ENVS,
     '    "VLLM_TPU_TARGET_EMBED_REPL": env_bool("VLLM_TPU_TARGET_EMBED_REPL", default=False),\n',
     '    "VLLM_TPU_TARGET_EMBED_REPL": env_bool("VLLM_TPU_TARGET_EMBED_REPL", default=True),\n', [TBUILD]),
    ("T09_target_lookup_unmasked", INIT,
     "            return replicated_lookup(table, input_)\n",
     "            return torch.nn.functional.embedding(input_.long().clamp(0, table.shape[0] - 1), table)\n", [TBIT]),
    ("T10_target_build_time_read_dropped", MTP,
     "        self.embed_repl_tgt = bool(vllm_torchtpu_envs.VLLM_TPU_TARGET_EMBED_REPL)\n",
     "        self.embed_repl_tgt = False\n", [TBUILD]),
]

CONTROL = [T, TT, "tests/compilation/test_tpu_compile_key_registration.py"]
SUMMARY = re.compile(r"^=*\s*(\d+ (passed|failed|error|errors|skipped)(, )?)+.* in [0-9.]+s")


def run(work, targets, stop_first=True):
    env = dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu")
    for k in ("VLLM_TPU_DRAFT_EMBED_REPL", "VLLM_TPU_TARGET_EMBED_REPL", "VLLM_TPU_DRAFT_EMBED_H8",
              "VLLM_TPU_QSA_DRAFT_SHARED_SELECTION",
              "VLLM_TPU_QSA_DRAFT_NO_INDEX_STATE"):
        env.pop(k, None)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:logging", "-rfE"]
    if stop_first:
        cmd.append("-x")
    p = subprocess.run(cmd + targets, cwd=work, env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if SUMMARY.search(ln)]
    return p.returncode, (lines[-1] if lines else ""), p.stdout


def main():
    lo = int(sys.argv[2]) if len(sys.argv) > 3 else 1
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(MUTANTS)
    assert 1 <= lo <= hi <= len(MUTANTS), (lo, hi, len(MUTANTS))
    work = tempfile.mkdtemp(prefix="draftembmut_")
    for d in ("src", "tests"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d),
                        ignore=shutil.ignore_patterns("__pycache__"))
    if os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    probe = subprocess.run(
        [sys.executable, "-c", "import os, vllm_torchtpu.models.vllm.qwen4_exp.mtp as m; "
         "print(os.path.realpath(m.__file__))"],
        cwd=work, env=dict(os.environ, PYTHONPATH=os.path.join(work, "src"), JAX_PLATFORMS="cpu"),
        capture_output=True, text=True, timeout=900)
    got = (probe.stdout.strip().splitlines() or [""])[-1]
    inside = got.startswith(os.path.realpath(os.path.join(work, "src")) + os.sep)
    print(f"MUTRUN_ORIGIN mtp {got or 'none'} {'INSIDE' if inside else 'FALLBACK'}")
    rc, tail, out = run(work, CONTROL, stop_first=False)
    m = re.search(r"(\d+) passed", tail)
    ok = (inside and rc == 0 and m is not None and int(m.group(1)) > 0
          and "failed" not in tail and "skipped" not in tail and "error" not in tail)
    print(f"CONTROL rc={rc} ok={ok} {tail}")
    if not ok:
        print(out[-4000:])
    killed = total = 0
    for idx, (name, rel, old, new, targets) in enumerate(MUTANTS, start=1):
        if not lo <= idx <= hi:
            continue
        path = os.path.join(work, rel)
        with open(path) as f:
            text = f.read()
        if text.count(old) != 1:
            print(f"MUT {idx:02d} {name} BAD (anchor occurs {text.count(old)} times)")
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
        print(f"MUT {idx:02d} {name} {'KILLED' if dead else 'SURVIVED'} rc={rc} {tail}")
    shutil.rmtree(work, ignore_errors=True)
    n = hi - lo + 1
    verdict = ok and killed == total == n
    rng = f" range {lo}-{hi} of {len(MUTANTS)}" if (lo, hi) != (1, len(MUTANTS)) else ""
    print(f"Q38_DRAFTEMB_MUT {'PASS' if verdict else 'FAIL'} killed {killed}/{total} of {n}{rng} control_ok={ok}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
