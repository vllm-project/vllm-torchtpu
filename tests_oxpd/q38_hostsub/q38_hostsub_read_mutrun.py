# SPDX-License-Identifier: Apache-2.0
"""Source mutants of q38_hostsub_read.py: each must make its selftest FAIL (rc != 0,
'HOSTSUB_SELFTEST FAIL', no traceback). Usage: python3 -I q38_hostsub_read_mutrun.py <reader>"""
import os
import shutil
import subprocess
import sys
import tempfile

MUTANTS = [
    ("R1_no_warmup_skip", "            if d[\"win\"][r] <= skip:\n                continue\n", "            if False:\n                continue\n"),
    ("R2_rank_max_is_min", "f\"max_mean={max(meds.values()):.0f}", "f\"max_mean={min(meds.values()):.0f}"),
    ("R3_pf_maps_to_other", '"pf_": "post_forward", ', '"pf_": "other", '),
    ("R4_no_data_reads_ok", '            out.append(f"HOSTSUBSUM arm={arm} ranks=0 windows=0 subs=0 verdict=NO_DATA")\n            ok = False\n',
     '            out.append(f"HOSTSUBSUM arm={arm} ranks=0 windows=0 subs=0 verdict=NO_DATA")\n'),
    ("R5_remainder_sign", 'remainder={pmm - s:.0f}', 'remainder={s - pmm:.0f}'),
    ("R6_mean_field_is_p90", "                p50, _p90, mean, n = v.split(\"/\")\n", "                p50, mean, _x, n = v.split(\"/\")\n"),
    ("R7_arm_end_ignored", "        if ARM_END_RX.search(ln):\n            arm = None if marked else default\n            continue\n", ""),
    # q38 L6-INSTR rows and the GC census summary.
    ("R8_dc_row_dropped", '"dc_": "dp_carries",', ""),
    ("R9_pi_seed_sync_row_dropped", '"pi_seed_sync": "prepare_inputs", ', ""),
    ("R10_ss_row_dropped", '\n    "ss_": "sample_setup",\n', "\n"),
    ("R11_gc_sum_not_per_step", "(int(m.group(4)) / n, ", "(int(m.group(4)), "),
    ("R12_gc_warmup_not_skipped", '            if d["gwin"][r] > skip:\n', "            if True:\n"),
    # q38 L4-E DRAFT1B-BUILD rows (11-14).
    ("R13_dl_row_dropped", '"dl_": "dp_loop", ', ""),
    ("R14_df_row_dropped", '"df_": "dp_forward0",\n', "\n"),
    # q38 L6 P-METER rows and the premium reader.
    ("R15_step_misaligned_running_count", "                base = seq0  # the window's own first seq, never a per-rank running count\n",
     '                base = (d["pmlines"][r] - 1) * w + 1\n'),
    ("R16_missing_rank_not_dropped", "if sum(s in pm[r] for r in ranks) == len(ranks)]", "if sum(s in pm[r] for r in ranks) >= 1]"),
    ("R17_premium_vs_mean", "            prem = vals[top] - med\n", "            prem = vals[top] - st.mean(vals.values())\n"),
    ("R18_mpr_row_dropped", '    "mpr_": "dp_entry",\n', ""),
    ("R19_abs_across_hosts", '    one_host = len(set(d["pmhost"].values())) == 1\n', "    one_host = True\n"),
    ("R20_wait_sign", "(pm[r][s][3] - pm[r][s][2])", "(pm[r][s][2] - pm[r][s][3])"),
    ("R21_vr_row_dropped", '    "vr_": "vfy_rejection",\n', ""),
]


def main():
    src = os.path.abspath(sys.argv[1])
    text = open(src).read()
    base = subprocess.run([sys.executable, "-I", src, "--selftest"], capture_output=True, text=True, timeout=120)
    ok = base.returncode == 0 and "HOSTSUB_SELFTEST PASS (32/32 checks)" in base.stdout
    print(f"BASELINE rc={base.returncode} ok={ok}")
    killed = 0
    for name, old, new in MUTANTS:
        if text.count(old) != 1:
            print(f"MUT {name} BAD anchor x{text.count(old)}")
            ok = False
            continue
        d = tempfile.mkdtemp(prefix="hsrm_")
        p = os.path.join(d, "q38_hostsub_read.py")
        open(p, "w").write(text.replace(old, new))
        r = subprocess.run([sys.executable, "-I", p, "--selftest"], capture_output=True, text=True, timeout=120)
        tb = "Traceback" in r.stdout + r.stderr
        dead = r.returncode != 0 and "HOSTSUB_SELFTEST FAIL" in r.stdout and not tb
        killed += dead
        print(f"MUT {name} {'KILLED' if dead else 'SURVIVED'} rc={r.returncode} tb={tb}")
        shutil.rmtree(d, ignore_errors=True)
    v = ok and killed == len(MUTANTS)
    print(f"Q38_HOSTSUB_READ_MUT {'PASS' if v else 'FAIL'} killed {killed}/{len(MUTANTS)}")
    return 0 if v else 1


if __name__ == "__main__":
    sys.exit(main())
