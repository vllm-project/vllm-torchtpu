#!/usr/bin/env python3
"""Reproduce the DSv4.1-Flash prefill result (handoff v3, section 6.1).

4 x (TP=2, EP=2) vllm servers on one tpu7x 2x2x1 node, 20-layer CED prefill,
RR proxy on :9000, `vllm bench serve` sweep of 6 workloads.
Reference: 197,315.9 input tok/s/node at 128K c=8 (commit cd4eedb).

Equivalent to the handoff's run_e2e_all3_fixes_and_4chip_sweep.py, minus the
workstation paths and the multi-cluster race; the in-pod driver is byte-identical.

Usage:
  scripts/prefill/run_prefill.py --ctx <kube-context> [--thread prefill] [--suffix a] \
      [--ubench] [--keep-pod] [--vtt repos/vtt-prefill] [--vllm repos/vllm-prefill]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dsv41 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
SUMMARY_IN_POD = "/dev/shm/summary_v85_4xtp2_prefill_clean_flags_b16k_4chip_node.json"


def manifest(pod: str, thread: str) -> dict:
    # Same as the handoff prefill pod: hostNetwork, privileged, 4 TPUs, /dev/shm emptyDir(Memory).
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {
            "name": pod, "namespace": "default",
            "labels": {"owner": L.USER, "dsv41-thread": thread, "dsv41-role": "prefill"},
            "annotations": {"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"},
        },
        "spec": {
            "hostNetwork": True, "dnsPolicy": "ClusterFirstWithHostNet", "restartPolicy": "Never",
            "nodeSelector": {"cloud.google.com/gke-tpu-accelerator": "tpu7x",
                             "cloud.google.com/gke-tpu-topology": "2x2x1"},
            "tolerations": [{"effect": "NoSchedule", "operator": "Exists"}],
            "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
            "containers": [{
                "name": "tpu-worker", "image": L.IMAGE, "command": ["bash", "-c", "sleep 28800"],
                "securityContext": {"privileged": True},
                "resources": {"requests": {"google.com/tpu": "4"}, "limits": {"google.com/tpu": "4"}},
                "volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
            }],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--thread", default="prefill")
    ap.add_argument("--suffix", default="a")
    ap.add_argument("--vtt", default=str(L.DEFAULT_VTT))
    ap.add_argument("--vllm", default=str(L.REPOS / "vllm-prefill"))
    ap.add_argument("--ubench", action="store_true", help="also run the 3-fix microbenchmarks first")
    ap.add_argument("--keep-pod", action="store_true", help="don't delete the pod at the end")
    ap.add_argument("--allow-foreign", action="store_true", help="skip the 'cluster has other users' check")
    a = ap.parse_args()
    L.install_signal_handlers()

    L.preflight(a.ctx)
    pod = L.pod_name(a.thread, f"pf-{a.suffix}")
    k = L.Kube(a.ctx, pod)
    if not a.allow_foreign:
        foreign = k.foreign_running_pods()
        if foreign:
            L.log(f"WARNING: other users' running pods on {a.ctx}: {foreign[:5]} (pass --allow-foreign to proceed)")
            sys.exit(2)

    stack, prov = L.build_stack(Path(a.vtt), Path(a.vllm), L.SUPPORT / "sitecustomize_bench/sitecustomize.py")
    out = L.run_dir(a.thread, f"prefill_{a.suffix}")
    (out / "provenance.json").write_text(json.dumps({"ctx": a.ctx, "pod": pod, **prov}, indent=2))
    claim = L.Claim(a.ctx, pod, a.thread, {"role": "prefill", "run_dir": str(out)})
    L.log(f"run dir: {out}")
    def pod_phase() -> str:
        try:
            return k.out(["get", "pod", pod, "-o", "jsonpath={.status.phase}"]).strip() or "Missing"
        except Exception:  # noqa: BLE001
            return "Missing"

    def boot_and_stage() -> None:
        for attempt in range(1, 4):
            try:
                k.delete()
                k.apply(manifest(pod, a.thread))
                node = k.wait_ready()
                L.log(f"pod {pod} Ready on {node}")
                k.exec_sh(r"""
set -e
mount -o remount,size=1050G,mpol=interleave:0-1 /dev/shm || mount -o remount,size=1050G /dev/shm
echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
if [ ! -L /workspace ]; then mkdir -p /dev/shm/workspace; cp -a /workspace/. /dev/shm/workspace/ 2>/dev/null || true; rm -rf /workspace; ln -sfn /dev/shm/workspace /workspace; fi
mkdir -p /dev/shm/perf_results
""")
                L.log("staging stack via GCS ...")
                L.stage_stack(k, stack)
                k.exec_sh(L.STAGE_BASE_SH)
                k.exec_sh('pgrep -f /workspace/mock_metadata.py >/dev/null || (LD_PRELOAD="" nohup python3 /workspace/mock_metadata.py >/tmp/mock_metadata.log 2>&1 &)\n')

                tok = L.get_token()
                with concurrent.futures.ThreadPoolExecutor(2) as ex:
                    f1 = ex.submit(k.exec_sh, L.INSTALL_SH)
                    f2 = ex.submit(k.exec_sh, L.download_ckpt_sh("prefill", tok), False, 3600)
                    f1.result(); f2.result()
                L.log("install + checkpoint (24 shards) done")

                if a.ubench:
                    k.cp_in(HERE / "ubench_all3_fixes_and_moe_tune.py", "/tmp/ubench_all3.py")
                    k.exec_sh("rm -f /tmp/libtpu_lockfile; cd /tmp && python3 -u /tmp/ubench_all3.py 2>&1 | tee /dev/shm/ubench.log", timeout=1800)
                    k.cp_out("/dev/shm/ubench_all3_fixes_and_moe_tune.json", out / "ubench_all3_fixes.json")

                k.cp_in(HERE / "in_pod_driver_clean_flags.py", "/dev/shm/in_pod_driver.py")
                sha = prov["vllm_torchtpu"]["head"][:7] + (f"+dirty.{prov['vllm_torchtpu']['dirty']}" if prov["vllm_torchtpu"]["dirty"] else "")
                k.exec_sh("rm -f /dev/shm/BENCH_ALL_DONE")
                L.start_detached(k, f"cd /dev/shm && VLLM_BENCH_POD={pod} VLLM_BENCH_NODE={node} VLLM_GIT_COMMIT='{sha}' "
                                    "exec python3 -u /dev/shm/in_pod_driver.py > /dev/shm/driver.log 2>&1", "in_pod_driver.py")
                L.log("in-pod driver started (chip0 compile ~15-25 min, then chips 1-3, warmup, 6-workload sweep)")
                return
            except Exception as e:  # noqa: BLE001
                ph = pod_phase()
                if attempt < 3 and ph != "Running":
                    L.log(f"stage attempt {attempt} interrupted (pod phase={ph}, err={e}); recreating pod ...")
                    continue
                raise

    try:
        boot_and_stage()

        seen, t0 = 0, time.time()
        while time.time() - t0 < 3 * 3600:
            time.sleep(20)
            ph = pod_phase()
            if ph != "Running":
                L.log(f"pod {pod} phase={ph} (spot preemption); recreating + restarting prefill driver ...")
                seen = 0
                boot_and_stage()
                continue
            txt = k.exec_sh("cat /dev/shm/driver.log 2>/dev/null || true", capture=True)
            if len(txt) > seen:
                sys.stdout.write(txt[seen:]); sys.stdout.flush(); seen = len(txt)
            (out / "driver.log").write_text(txt)
            k.cp_out(SUMMARY_IN_POD, out / "summary.json")
            if k.exec_sh("test -f /dev/shm/BENCH_ALL_DONE && echo YES || true", capture=True).strip() == "YES":
                break
            if "Traceback (most recent call last)" in txt:
                for c in range(4):
                    k.cp_out(f"/dev/shm/vllm_chip{c}.log", out / f"vllm_chip{c}.log")
                raise RuntimeError("driver failed; see " + str(out))
        k.cp_out(SUMMARY_IN_POD, out / "summary.json")
        for c in range(4):
            k.cp_out(f"/dev/shm/vllm_chip{c}.log", out / f"vllm_chip{c}.log")
        k.exec_sh("cd /dev/shm && tar -cf /tmp/perf_results.tar perf_results")
        k.cp_out("/tmp/perf_results.tar", out / "perf_results.tar")
        L.log(f"DONE -> {out}/summary.json ; compare with scripts/compare_results.py prefill {out}/summary.json")
    finally:
        if not a.keep_pod:
            k.delete()
            L.log(f"deleted pod {pod}")
        claim.release()


if __name__ == "__main__":
    main()
