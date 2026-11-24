"""q38 DEC-FB-2 POSTFWD G2 CPU verdict: the bitwise test on the clean tree must PASS, and each mutant of
subtract_substitute_extract (applied to a private copy of src/) must make it FAIL. Run from the tree root:
    python3 -u q38_postfwd/run_g2.py
Prints one line per step and a final Q38_POSTFWD_G2_SUMMARY line; exit 0 only if every step behaves as required."""
import os, shutil, subprocess, sys, tempfile

TEST = "tests/runner/test_postfwd_fused.py"
TARGET = "src/vllm_torchtpu/runner/tpu_runner_async_output.py"
MUTANTS = {  # name: (exact original text inside subtract_substitute_extract, replacement)
    "m1_mask_all": ("    mask = pre_next_tokens_indices > -1\n", "    mask = pre_next_tokens_indices >= -1\n"),
    "m2_no_scatter": ("    input_ids = input_ids.scatter(0, cur_input_indices, update_values)\n",
                      "    input_ids = input_ids.clone()\n"),
    "m3_src_plus1": ("    new_token_values = next_tokens.reshape(-1)[safe_idx].to(input_ids.dtype)\n",
                     "    new_token_values = next_tokens.reshape(-1)[safe_idx].to(input_ids.dtype) + 1\n"),
    "m4_no_subtract": ("    seq_lens, positions = subtract_num_rejected_tokens(\n        seq_lens,\n        positions,\n        num_rejected,\n",
                       "    seq_lens, positions = subtract_num_rejected_tokens(\n        seq_lens,\n        positions,\n        torch.zeros_like(num_rejected),\n"),
    "m5_clamp1": ("    safe_idx = torch.clamp_min(pre_next_tokens_indices, 0)\n    new_token_values = next_tokens.reshape(-1)",
                  "    safe_idx = torch.clamp_min(pre_next_tokens_indices, 1)\n    new_token_values = next_tokens.reshape(-1)"),
    "m6_draft_presub": ("    draft_token_ids = extract_draft_token_ids(\n        input_ids, logits_indices, target_logits_indices\n    )\n    return seq_lens, positions, input_ids, draft_token_ids\n",
                        "    draft_token_ids = extract_draft_token_ids(\n        original_ids, logits_indices, target_logits_indices\n    )\n    return seq_lens, positions, input_ids, draft_token_ids\n"),
}
EXTRA = {"m6_draft_presub": ("    mask = pre_next_tokens_indices > -1\n", "    original_ids = input_ids\n    mask = pre_next_tokens_indices > -1\n")}


def pytest(pythonpath):
    env = dict(os.environ, PYTHONPATH=pythonpath, PYTHONUNBUFFERED="1")
    p = subprocess.run([sys.executable, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider", TEST],
                       env=env, capture_output=True, text=True)
    tail = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:] or ["<no output>"]
    return p.returncode, tail[0], p.stdout + p.stderr


ok = True
rc, tail, out = pytest("src")
clean_pass = rc == 0
print(f"CLEAN rc={rc} -> {'PASS' if clean_pass else 'FAIL'} :: {tail}", flush=True)
if not clean_pass:
    print(out[-6000:], flush=True)
ok &= clean_pass
src_text = open(TARGET).read()
fn_start = src_text.index("def subtract_substitute_extract(")
fn_end = src_text.index("\ndef ", fn_start + 10)
killed = 0
for name, (old, new) in MUTANTS.items():
    fn = src_text[fn_start:fn_end]
    edits = [(old, new)] + ([EXTRA[name]] if name in EXTRA else [])
    for o, n in edits:
        if fn.count(o) != 1:
            print(f"MUTANT {name}: ABORT, anchor found {fn.count(o)} times in subtract_substitute_extract", flush=True)
            ok = False
            break
        fn = fn.replace(o, n)
    else:
        d = tempfile.mkdtemp(prefix=f"q38_pf_{name}_")
        shutil.copytree("src", os.path.join(d, "src"))
        with open(os.path.join(d, TARGET), "w") as f:
            f.write(src_text[:fn_start] + fn + src_text[fn_end:])
        rc, tail, _ = pytest(os.path.join(d, "src"))
        k = rc != 0
        killed += k
        ok &= k
        print(f"MUTANT {name} rc={rc} -> {'KILLED' if k else 'SURVIVED'} :: {tail}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
print(f"Q38_POSTFWD_G2_SUMMARY clean={'PASS' if clean_pass else 'FAIL'} mutants_killed={killed}/{len(MUTANTS)} "
      f"VERDICT={'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
