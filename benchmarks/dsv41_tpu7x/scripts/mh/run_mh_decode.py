#!/usr/bin/env python3
"""Multi-host DP16/EP16 (2x2x2) or DP32/EP32 (2x2x4) decode-only server for DeepSeek-V4.1-Flash.

Creates ONE Indexed Job (parallelism = #hosts) on a multi-host tpu7x node pool, stages the same
stack/checkpoint as run_pd.py on every host, then launches vllm-torchtpu's "mp" multi-host DP
engine: host 0 = head (API servers on :9400), hosts 1.. = --headless --data-parallel-start-rank.

  python3 run_mh_decode.py --ctx <kube ctx> --topology 2x2x2 --hosts 2 --suffix mh1 [--reuse]
"""
from __future__ import annotations

import argparse, json, subprocess, sys, time, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import dsv41 as L  # noqa: E402

PD = HERE.parent / "pd"
THREAD = "mh"
PORT = 9400


def job_manifest(name: str, hosts: int, topology: str) -> dict:
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": "default",
                     "labels": {"owner": L.USER, "dsv41-thread": THREAD}},
        "spec": {
            "parallelism": hosts, "completions": hosts, "completionMode": "Indexed",
            "backoffLimit": 0, "ttlSecondsAfterFinished": 600,
            "template": {
                "metadata": {"labels": {"owner": L.USER, "dsv41-thread": THREAD, "dsv41-job": name},
                             "annotations": {"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}},
                "spec": {
                    "restartPolicy": "Never",
                    "hostNetwork": True,
                    "dnsPolicy": "ClusterFirstWithHostNet",
                    "subdomain": name,
                    "nodeSelector": {"cloud.google.com/gke-tpu-accelerator": "tpu7x",
                                     "cloud.google.com/gke-tpu-topology": topology},
                    "tolerations": [{"operator": "Exists"}],
                    "volumes": [{"name": "dshm", "hostPath": {"path": "/dev/shm"}},
                                {"name": "host-disk", "hostPath": {"path": "/mnt/stateful_partition"}}],
                    "containers": [{
                        "name": "tpu-worker", "image": L.IMAGE, "securityContext": {"privileged": True},
                        "command": ["bash", "-lc", "sleep infinity"],
                        "env": [{"name": "TPU_LOG_DIR", "value": "/tmp/tpu_logs/tpu-worker"},
                                {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}],
                        "resources": {"limits": {"google.com/tpu": "4"},
                                      "requests": {"google.com/tpu": "4", "memory": "200Gi"}},
                        "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"},
                                         {"name": "host-disk", "mountPath": "/host_disk"}],
                    }],
                },
            },
        },
    }


def kget(ctx: str, args: list[str]) -> str:
    return subprocess.check_output(["kubectl", f"--context={ctx}", *args], text=True, timeout=120)


def job_pods(ctx: str, name: str) -> dict[int, str]:
    out = kget(ctx, ["get", "pods", "-l", f"dsv41-job={name}", "-o",
                     "jsonpath={range .items[*]}{.metadata.name} {.metadata.annotations.batch\\.kubernetes\\.io/job-completion-index} {.status.phase}\n{end}"])
    pods = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            pods[int(parts[1])] = parts[0]
    return pods


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--topology", default="2x2x2")
    ap.add_argument("--hosts", type=int, default=2)
    ap.add_argument("--suffix", default="mh1")
    ap.add_argument("--tag", default="dsv41_mh_decode")
    ap.add_argument("--vtt", default=str(L.DEFAULT_VTT))
    ap.add_argument("--vllm", default=str(L.REPOS / "vllm-decode"))
    ap.add_argument("--dspark-tokens", type=int, default=5)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--mnbt", type=int, default=384)
    ap.add_argument("--compile-sizes", default="[16, 40, 48, 64, 96, 192, 384]")
    ap.add_argument("--extra-serve-args", default="")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL", help="extra env for the servers (repeatable)")
    ap.add_argument("--reuse", action="store_true", help="job/pods already exist: re-stage + relaunch only")
    ap.add_argument("--no-setup", action="store_true", help="skip pip/ckpt setup (already done)")
    ap.add_argument("--pending-timeout", type=int, default=7200)
    ap.add_argument("--boot-timeout", type=int, default=6000)
    a = ap.parse_args()
    L.install_signal_handlers()
    L.preflight(a.ctx)

    name = f"{L.USER}-dsv41-mh-{a.suffix}"
    extra = {f"workspace/whls_pd/{w.name}": w for w in sorted((PD / "whls").glob("*.whl"))}
    stack, prov = L.build_stack(Path(a.vtt), Path(a.vllm), PD / "sitecustomize_pd.py", extra_files=extra)
    out = L.run_dir(THREAD, f"{a.tag}_{a.suffix}")
    run_id = f"{out.name.split('_')[0]}-{uuid.uuid4().hex[:6]}"
    (out / "provenance.json").write_text(json.dumps({"ctx": a.ctx, "job": name, "run_id": run_id, "args": vars(a), **prov}, indent=2))
    L.log(f"run dir: {out}  job: {name} topology={a.topology} hosts={a.hosts}")

    if not a.reuse:
        try:
            kget(a.ctx, ["delete", "job", name, "--ignore-not-found", "--wait=true"])
        except subprocess.CalledProcessError:
            pass
        m = job_manifest(name, a.hosts, a.topology)
        subprocess.run(["kubectl", f"--context={a.ctx}", "apply", "-f", "-"], input=json.dumps(m).encode(), check=True)

    # wait for all pods Running
    t0 = time.time()
    while True:
        pods = job_pods(a.ctx, name)
        phases = {}
        for i, p in pods.items():
            phases[i] = kget(a.ctx, ["get", "pod", p, "-o", "jsonpath={.status.phase}"]).strip()
        if len(pods) == a.hosts and all(ph == "Running" for ph in phases.values()):
            break
        if any(ph in ("Failed", "Succeeded") for ph in phases.values()):
            raise RuntimeError(f"a pod ended: {phases}")
        if time.time() - t0 > a.pending_timeout:
            raise TimeoutError(f"pods not Running after {a.pending_timeout}s: {phases}")
        L.log(f"waiting for {a.hosts} pods Running: {phases}")
        time.sleep(30)
    kube = {i: L.Kube(a.ctx, pods[i]) for i in sorted(pods)}
    ips = {i: kget(a.ctx, ["get", "pod", pods[i], "-o", "jsonpath={.status.podIP}"]).strip() for i in kube}
    nodes = {i: kget(a.ctx, ["get", "pod", pods[i], "-o", "jsonpath={.spec.nodeName}"]).strip() for i in kube}
    L.log(f"pods: {pods} ips: {ips} nodes: {nodes}")
    (out / "summary.json").write_text(json.dumps({"job": name, "pods": pods, "ips": ips, "nodes": nodes, "run_id": run_id}, indent=2))

    tok = L.get_token()
    dl = L.download_ckpt_sh("all", tok, "").replace("python3 -u /tmp/dsv41_dl.py\n", "")
    dl += (f"cat > /tmp/dsv41_env.sh <<'EOF'\nexport GCS_TOKEN='{tok}' DSV41_GCS_BUCKET='{L.GCS_BUCKET}' DSV41_GCS_PREFIX='{L.GCS_PREFIX}' "
           f"DSV41_SHARDS=all DSV41_HF_REPO=''\nEOF\n")
    L.upload_stack(stack)

    def stage(i: int) -> None:
        k = kube[i]
        k.exec_sh("for p in $(pgrep -f 'launch_dsv41_mh_decode|vllm serve|VLLM::|EngineCore'); do [ \"$p\" != \"$$\" ] && kill -9 $p 2>/dev/null || true; done; sleep 2", timeout=120)
        if not a.no_setup:
            L.stage_stack(k, stack)
            k.exec_sh(L.STAGE_BASE_SH)
            k.cp_in(PD / "in_pod_setup.sh", "/tmp/in_pod_setup.sh")
            k.exec_sh(dl)
            L.log(f"[host{i}] installing + downloading checkpoint ...")
            k.exec_sh("bash /tmp/in_pod_setup.sh > /dev/shm/mh_setup.log 2>&1; tail -2 /dev/shm/mh_setup.log", timeout=3600, retries=1)
            tail = k.exec_sh("tail -2 /dev/shm/mh_setup.log", capture=True)
            if "PD_SETUP_OK" not in tail:
                raise RuntimeError(f"[host{i}] setup failed: {tail}")
        k.cp_in(HERE / "launch_dsv41_mh_decode.sh", "/tmp/launch_dsv41_mh_decode.sh")
        env = {
            "HOST_INDEX": str(i), "NUM_HOSTS": str(a.hosts), "HEAD_IP": ips[0], "DP_LOCAL": "8", "POD_IP": ips[i],
            "PORT": str(PORT), "DSPARK_TOKENS": str(a.dspark_tokens), "MAX_NUM_SEQS": str(a.max_num_seqs),
            "MNBT": str(a.mnbt), "COMPILE_SIZES": a.compile_sizes, "DSV41_EXTRA_SERVE_ARGS": a.extra_serve_args,
            "RUN_ID": run_id,
            **dict(kv.split("=", 1) for kv in a.env),
        }
        env_sh = "".join(f"export {kk}='{vv}'\n" for kk, vv in env.items())
        k.exec_sh(f"cat > /tmp/mh_env.sh <<'EOF'\n{env_sh}EOF\nrm -f /dev/shm/mh_decode.log\n")

    for i in kube:  # sequential: the checkpoint download saturates GCS egress per host anyway
        stage(i)
    # launch head first (binds the DP rendezvous), then the headless hosts
    for i in sorted(kube):
        L.start_detached(kube[i], "exec bash /tmp/launch_dsv41_mh_decode.sh", "launch_dsv41_mh_decode.sh|vllm serve")
        time.sleep(5)
    L.log("servers launched on all hosts; multi-host boot (compile + rendezvous) can take 30-45 min")

    seen = {i: 0 for i in kube}
    t0 = time.time()
    pat = r"\[MH-LAUNCH\]|Memory statistics|KV cache size|DeepSeek-V4.1-Flash KV cache|Application startup complete|POST_WARMUP_HBM|Error|Traceback|rendezvous|slicebuilder|SLICEBUILDER|multihost|TORCH_TPU_TOPOLOGY"
    while True:
        if time.time() - t0 > a.boot_timeout:
            raise TimeoutError("boot timeout")
        time.sleep(30)
        for i, k in kube.items():
            try:
                lines = k.exec_sh(f"grep -E {json.dumps(pat)} /dev/shm/mh_decode.log 2>/dev/null | cut -c1-300 || true", capture=True, timeout=120).splitlines()
            except subprocess.CalledProcessError:
                continue
            for ln in lines[seen[i]:]:
                print(f"  [host{i}] {ln}", flush=True)
            seen[i] = len(lines)
            if any(("Engine core initialization failed" in ln or "RuntimeBufferAllocationFailure" in ln) for ln in lines):
                raise RuntimeError(f"[host{i}] server failed during boot")
        try:
            code = kube[0].exec_sh(f"curl -s -m 5 -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}/health || true", capture=True, timeout=60).strip()
        except subprocess.CalledProcessError:
            code = ""
        if code == "200":
            L.log(f"HEALTHY after {time.time() - t0:.0f}s: head {pods[0]} ({ips[0]}:{PORT}) serving DP={8 * a.hosts}/EP={8 * a.hosts}")
            break
    (out / "summary.json").write_text(json.dumps({"job": name, "pods": pods, "ips": ips, "nodes": nodes, "run_id": run_id, "healthy": True}, indent=2))


if __name__ == "__main__":
    main()
