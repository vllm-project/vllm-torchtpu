#!/usr/bin/env python3
"""Reproduce the DSv4.1-Flash decode result (handoff v3, section 6.2).

DP=8, TP=1, EP=8 on one tpu7x 2x2x1 node, DSpark K=5, 100K context via
DecodeBenchConnector synthetic KV (cache_hit_ratio 0.99936), 2+2 / USA gates.
Reference (Oct 12, w384 tree): c=64 step 37.0 ms -> 6,108.5 tok/s/node @tau=3.5;
c=256 step 65.1 ms -> 14,223.4 tok/s/node @tau=3.5. Output text is known-incorrect.

Equivalent to the handoff's multi_cluster_launch_and_harvest.py for ONE pod; the
in-pod driver (run_dp8_scale_driver.py) is byte-identical to paste 6755399435850903.

Usage:
  scripts/decode/run_decode.py --ctx <kube-context> --mode dp8_c64|dp8_c256|dp8_c512|dp8_all|tp8_c64 \
      [--thread decode] [--suffix a] [--keep-pod] [--vtt repos/vtt-decode] [--vllm repos/vllm-decode]
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dsv41 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = [
    "accuracy_dp8_usa_states_verified.json", "accuracy_tp8_usa_states_verified.json", "gate_meta.json",
    "result_dp8_hbm0p90_64x1024_verified.json", "result_dp8_4x_c256.json", "result_dp8_8x_c512.json",
    "result_tp8_hbm0p90_64x1024.json",
]
EXPECTED = {
    "dp8_c64": ["result_dp8_hbm0p90_64x1024_verified.json"],
    "dp8_c256": ["result_dp8_4x_c256.json"],
    "dp8_c512": ["result_dp8_8x_c512.json"],
    "dp8_all": ["result_dp8_hbm0p90_64x1024_verified.json", "result_dp8_4x_c256.json", "result_dp8_8x_c512.json"],
    "tp8_c64": ["result_tp8_hbm0p90_64x1024.json"],
}


def manifest(pod: str, thread: str) -> dict:
    # Handoff 4.1: hostPath /dev/shm (NOT a 64Gi emptyDir -> eviction), 200Gi memory request.
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": pod, "namespace": "default",
                     "labels": {"owner": L.USER, "dsv41-thread": thread, "dsv41-role": "decode"},
                     "annotations": {"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}},
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"cloud.google.com/gke-tpu-accelerator": "tpu7x",
                             "cloud.google.com/gke-tpu-topology": "2x2x1"},
            "tolerations": [{"operator": "Exists"}],
            "volumes": [{"name": "dshm", "hostPath": {"path": "/dev/shm"}},
                        {"name": "host-disk", "hostPath": {"path": "/mnt/stateful_partition"}}],
            "containers": [{
                "name": "tpu-worker", "image": L.IMAGE, "securityContext": {"privileged": True},
                "command": ["bash", "-lc", "sleep infinity"],
                "env": [{"name": "TPU_WORKER_ID", "value": "0"},
                        {"name": "TPU_LOG_DIR", "value": "/tmp/tpu_logs/tpu-worker"},
                        {"name": "TPU_PROCESS_PORT", "value": "8471"},
                        {"name": "TPU_PROCESS_ADDRESSES", "value": "localhost:8471"},
                        {"name": "TPU_WORKER_HOSTNAMES", "value": "localhost"}],
                "resources": {"limits": {"google.com/tpu": "4"},
                              "requests": {"google.com/tpu": "4", "memory": "200Gi"}},
                "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"},
                                 {"name": "host-disk", "mountPath": "/host_disk"}],
            }],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--mode", required=True, choices=list(EXPECTED))
    ap.add_argument("--thread", default="decode")
    ap.add_argument("--suffix", default="a")
    ap.add_argument("--vtt", default=str(L.DEFAULT_VTT))
    ap.add_argument("--vllm", default=str(L.REPOS / "vllm-decode"))
    ap.add_argument("--keep-pod", action="store_true")
    ap.add_argument("--allow-foreign", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true", help="serve with --no-enable-prefix-caching (as the handoff reference rows)")
    ap.add_argument("--libtpu", default="0.0.48", help="libtpu version pinned in the pod")
    ap.add_argument("--extra-serve-args", default="", help="extra `vllm serve` args appended before --port (in_pod_runner DSV41_EXTRA_SERVE_ARGS)")
    ap.add_argument("--extra-env", default="", help="space-separated K=V exported before `vllm serve` (DSV41_EXTRA_ENV); note sitecustomize force-sets some keys")
    ap.add_argument("--dp-rr", type=int, default=1, help="1 (default): pin warmup/bench requests to DP ranks via X-data-parallel-rank (DSV41_DP_RR); 0: original driver behaviour")
    ap.add_argument("--buckets", default="", help="override compile_sizes, e.g. '[48, 60, 72, 96, 384]' (DSV41_BUCKETS)")
    ap.add_argument("--attach", metavar="RUN_DIR", help="monitor/harvest an already-running pod into RUN_DIR")
    a = ap.parse_args()
    L.install_signal_handlers()

    L.preflight(a.ctx)
    pod = L.pod_name(a.thread, f"dc-{a.mode.replace('_', '')}-{a.suffix}")
    k = L.Kube(a.ctx, pod)
    if not a.allow_foreign:
        foreign = k.foreign_running_pods()
        if foreign:
            L.log(f"WARNING: other users' running pods on {a.ctx}: {foreign[:5]} (pass --allow-foreign to proceed)")
            sys.exit(2)

    stack, prov = L.build_stack(Path(a.vtt), Path(a.vllm), L.SUPPORT / "decode/sitecustomize.py")
    if a.attach:
        out = Path(a.attach)
        assert out.is_dir(), f"--attach expects an existing run dir, got {out}"
    else:
        out = L.run_dir(a.thread, f"decode_{a.mode}_{a.suffix}")
        (out / "provenance.json").write_text(json.dumps({"ctx": a.ctx, "pod": pod, "mode": a.mode, "no_prefix_caching": a.no_prefix_caching, "libtpu": a.libtpu,
                                                         "extra_serve_args": a.extra_serve_args, "extra_env": a.extra_env,
                                                         "dp_rr": a.dp_rr, "buckets": a.buckets, **prov}, indent=2))
    claim = L.Claim(a.ctx, pod, a.thread, {"role": "decode", "mode": a.mode, "run_dir": str(out)})
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
                L.log("staging stack via GCS ...")
                L.stage_stack(k, stack)
                k.exec_sh(L.STAGE_BASE_SH)
                k.cp_in(HERE / "run_dp8_scale_driver.py", "/tmp/run_dp8_scale_driver.py")
                k.cp_in(HERE / "in_pod_runner.sh", "/tmp/in_pod_runner.sh")
                tok = L.get_token()
                dl = L.download_ckpt_sh("all", tok)
                # Write the downloader + env (token stays inside the pod, never in logs/run dir).
                k.exec_sh(dl.replace("python3 -u /tmp/dsv41_dl.py\n", "") +
                          f"cat > /tmp/dsv41_env.sh <<'EOF'\nexport GCS_TOKEN='{tok}' DSV41_GCS_BUCKET='{L.GCS_BUCKET}' "
                          f"DSV41_GCS_PREFIX='{L.GCS_PREFIX}' DSV41_SHARDS=all DSV41_NO_PREFIX_CACHING={int(a.no_prefix_caching)} DSV41_LIBTPU={a.libtpu}\n"
                          f"export DSV41_EXTRA_SERVE_ARGS={shlex.quote(a.extra_serve_args)} DSV41_EXTRA_ENV={shlex.quote(a.extra_env)} "
                          f"DSV41_DP_RR={int(a.dp_rr)} DSV41_BUCKETS={shlex.quote(a.buckets)}\nEOF\n")
                k.exec_sh("rm -f /dev/shm/result_*.json /dev/shm/accuracy_*.json /dev/shm/gate_meta.json")
                L.start_detached(k, f"source /tmp/dsv41_env.sh; cd /tmp && exec bash /tmp/in_pod_runner.sh {a.mode} "
                                    "> /dev/shm/dsv41_runner.log 2>&1", "in_pod_runner.sh")
                L.log("runner started (download ~2 min + boot ~15-25 min cold + gates + benchmark)")
                return
            except Exception as e:  # noqa: BLE001
                ph = pod_phase()
                if attempt < 3 and ph != "Running":
                    L.log(f"stage attempt {attempt} interrupted (pod phase={ph}, err={e}); recreating pod ...")
                    continue
                raise

    try:
        if not a.attach:
            boot_and_stage()
        else:
            L.log(f"attached to running pod {pod}")

        need = set(EXPECTED[a.mode])
        seen, t0 = 0, time.time()
        while time.time() - t0 < 4 * 3600:
            time.sleep(20)
            ph = pod_phase()
            if ph != "Running":
                L.log(f"pod {pod} phase={ph} (spot preemption); recreating + restarting decode runner ...")
                seen = 0
                boot_and_stage()
                continue
            txt = k.exec_sh("cat /dev/shm/dsv41_runner.log 2>/dev/null || true", capture=True)
            if len(txt) > seen:
                sys.stdout.write(txt[seen:]); sys.stdout.flush(); seen = len(txt)
            (out / "runner.log").write_text(txt)
            ls = k.exec_sh("ls -1 /dev/shm 2>/dev/null || true", capture=True).split()
            for fn in RESULTS:
                if fn in ls and not (out / fn).exists():
                    if k.cp_out(f"/dev/shm/{fn}", out / fn):
                        L.log(f"harvested {fn}")
            if need <= {p.name for p in out.iterdir()}:
                break
            alive = k.exec_sh("pgrep -f 'run_dp8_scale_driver.py|in_pod_runner.sh' >/dev/null && echo Y || echo N",
                              capture=True).strip()
            if alive == "N":
                k.cp_out("/dev/shm/vllm_dp8_scale.log", out / "vllm_server.log")
                missing = need - {p.name for p in out.iterdir()}
                raise RuntimeError(f"runner exited without {sorted(missing)}; see {out}/runner.log")
        k.cp_out("/dev/shm/vllm_dp8_scale.log", out / "vllm_server.log")
        L.log(f"DONE -> {out} ; compare with scripts/compare_results.py decode {out}")
    finally:
        if not a.keep_pod:
            k.delete()
            L.log(f"deleted pod {pod}")
        claim.release()


if __name__ == "__main__":
    main()
