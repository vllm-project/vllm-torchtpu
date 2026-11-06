# SPDX-License-Identifier: Apache-2.0
"""One-line result of a measured arm, from the client summary, the device step
(pjrt_step.py) and the host's server log.

  python3 arm_result.py --arm NAME --chips 32 --summary S.summary.json \
      --device-step device-step.json --log server.log.gz

Prints ``RESULT {json}``: device step, steady tok/s per chip, mean request
TPOT, whether every request finished exact and progressed, the lowest true
free device memory over this host's workers (runtime reservation excluded) at
the latest logged decode step and after the first-request pool zero, and the
exchange capacity line of the ragged all-to-all.
"""

import argparse
import gzip
import json
import re

HBM = re.compile(r"SYNTHETIC_DECODE_HBM (\S+) .*?true_free_gib=([-0-9.]+)")
CAP = re.compile(r"Ragged A2A MoE capacity (x[^|]*)\| decode step: ([^\"\n]*)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--chips", type=int, required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--device-step", default="")
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    out = {"arm": a.arm, "chips": a.chips}
    try:
        with open(a.summary) as f:
            s = json.load(f)
        st = s.get("steady") or {}
        out.update(
            global_seqs=s.get("global_seqs"),
            req_per_engine=s.get("global_seqs", 0) // max(s.get("engines", 1), 1),
            tok_s_chip=round(st["tokens_per_second"] / a.chips, 2)
            if st.get("tokens_per_second")
            else None,
            tpot_ms=round(st["mean_request_tpot_ms"], 2)
            if st.get("mean_request_tpot_ms")
            else None,
            exact=bool(s.get("all_exact_output") and st.get("all_requests_progressed")),
            sampling=s.get("sampling"),
        )
        if s.get("ids_sha256"):
            out["ids_sha256"] = s["ids_sha256"]
    except (OSError, ValueError, KeyError) as e:
        out["summary_error"] = repr(e)
    if a.device_step:
        try:
            with open(a.device_step) as f:
                d = json.load(f)
            out["step_ms"] = d.get("step_ms_median")
        except (OSError, ValueError) as e:
            out["device_step_error"] = repr(e)
    if a.log:
        free = {}
        cap = None
        opener = gzip.open if a.log.endswith(".gz") else open
        with opener(a.log, "rt", errors="ignore") as f:
            for line in f:
                if "tpu_pjrt_client" in line:
                    continue
                m = HBM.search(line)
                if m:
                    free.setdefault(m.group(1), []).append(float(m.group(2)))
                elif cap is None:
                    m = CAP.search(line)
                    if m:
                        cap = f"{m.group(1).strip()} | {m.group(2).strip()}"
        steady = [t for t in free if t.startswith("load_call=")]
        if steady:
            last = max(steady, key=lambda t: int(t.split("=")[1]))
            out["true_free_gib_min"] = min(free[last])
            out["free_at"] = last
        if "pool_ready" in free:
            out["true_free_gib_after_zero_min"] = min(free["pool_ready"])
        out["capacity"] = cap
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
