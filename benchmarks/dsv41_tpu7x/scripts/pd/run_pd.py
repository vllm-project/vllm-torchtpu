#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash 1P1D disaggregated run (PD_DESIGN.md section 5, Phase 1).

Two pods on one cluster (each a whole tpu7x 2x2x1 node):
  prefill pod: vllm serve DP=8 TP=1 EP=8, VLLM_DSV41_CED=1 PURE_20L=0, TPURaidenConnector
               kv_producer; toy_proxy_server.py on :8000; the gates run here.
  decode pod : vllm serve DP=8 TP=1 EP=8, kv_consumer, no DSpark,
               --no-enable-prefix-caching --no-async-scheduling.
KV moves over Raiden Stage 3 (DCN, host-staged) between the pod IPs. Both pods run the
same stack (vtt-pd + vllm-decode), so their KV layouts match by construction (G3).

Flow: create both pods -> stage the stack via GCS on both -> install + download all 48
shards on both -> launch decode, then prefill -> wait for both /health -> proxy ->
gates (scripts/pd/pd_gates.py) -> check_dsv4_stage3_log.py on the prefill log -> harvest
to runs/pd/<ts>_<tag>/ -> delete both pods (unless --keep-pods).

Usage:
  scripts/pd/run_pd.py --ctx <kube-context> [--suffix a] [--keep-pods] [--prefill-ced 1]
      [--reuse-pods]   # pods already exist (from --keep-pods): re-stage + relaunch servers
      [--gates-only]   # servers already healthy on kept pods: proxy + gates + harvest only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dsv41 as L  # noqa: E402

HERE = Path(__file__).resolve().parent
THREAD = "pd"
PORTS = {"prefill": 8400, "decode": 9400}
PROXY_PORT = 8000
FAIL_PAT = ("Engine core initialization failed", "RuntimeBufferAllocationFailure",
            "EngineCore failed", "[PD-SERVE] server exited")
PROGRESS_PAT = (r"\[PD-LAUNCH\]|Memory statistics|TPU KV cache size|DeepSeek-V4.1-Flash KV cache|"
                r"Raiden DSv4 pool admission|Raiden DSv4 pool tag=|layout fingerprint|"
                r"Application startup complete|POST_WARMUP_HBM|Error|Traceback|ManifestError|"
                r"deleted checkpoint")


def manifest(pod: str, role: str) -> dict:
    # Same as scripts/decode/run_decode.py: hostPath /dev/shm (a 64Gi emptyDir gets the pod
    # evicted), 200Gi memory request. hostNetwork: Raiden's transport server maps each
    # data NIC to its NUMA node via /sys/class/net/<nic>/device/numa_node; a pod veth has
    # no device, and the lookup aborts every worker (raw_hash_map::at) at KV admission.
    # The peers reach each other on the node IPs.
    return {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": pod, "namespace": "default",
                     "labels": {"owner": L.USER, "dsv41-thread": THREAD, "dsv41-role": role},
                     "annotations": {"cluster-autoscaler.kubernetes.io/safe-to-evict": "false"}},
        "spec": {
            "restartPolicy": "Never",
            "hostNetwork": True,
            "dnsPolicy": "ClusterFirstWithHostNet",
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


def pod_ip(k: L.Kube) -> str:
    return k.out(["get", "pod", k.pod, "-o", "jsonpath={.status.podIP}"]).strip()


def pod_phase(k: L.Kube) -> str:
    try:
        return k.out(["get", "pod", k.pod, "-o", "jsonpath={.status.phase}"]).strip() or "Missing"
    except subprocess.CalledProcessError:
        return "Missing"


def grep_log(k: L.Kube, path: str, pat: str) -> str:
    return k.exec_sh(f"grep -E {json.dumps(pat)} {path} 2>/dev/null | cut -c1-400 || true",
                     capture=True, timeout=120)


def healthy(k: L.Kube, port: int, path: str = "/health") -> bool:
    out = k.exec_sh(f"curl -fsS -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}{path} "
                    "2>/dev/null || true", capture=True, timeout=60).strip()
    return out == "200"


def parallel(fns: dict) -> dict:
    res, errs = {}, {}

    def run(name, fn):
        try:
            res[name] = fn()
        except BaseException as e:  # noqa: BLE001
            errs[name] = e
    ths = [threading.Thread(target=run, args=(n, f)) for n, f in fns.items()]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise RuntimeError(f"parallel step failed: {errs}")
    return res


def harvest(kp: L.Kube, kd: L.Kube, out: Path) -> None:
    for k, files in ((kp, ["pd_prefill.log", "pd_proxy.log", "pd_gates.log", "pd_gates.json",
                           "pd_stage3_check.txt", "pd_setup.log"]),
                     (kd, ["pd_decode.log", "pd_setup.log"])):
        role = "prefill" if k is kp else "decode"
        for fn in files:
            try:
                k.exec_sh(f"cd /dev/shm && [ -f {fn} ] && gzip -kf {fn} || true", timeout=300)
                local = out / f"{role}__{fn}.gz"
                if k.cp_out(f"/dev/shm/{fn}.gz", local):
                    subprocess.run(["gunzip", "-f", str(local)], check=False)
            except Exception as e:  # noqa: BLE001
                L.log(f"harvest {role}/{fn} failed: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--suffix", default="a")
    ap.add_argument("--tag", default="1p1d")
    ap.add_argument("--vtt", default=str(L.DEFAULT_VTT))
    ap.add_argument("--vllm", default=str(L.REPOS / "vllm-decode"))
    ap.add_argument("--prefill-ced", type=int, default=1)
    ap.add_argument("--prefill-mnbt", type=int, default=8192)
    ap.add_argument("--decode-mnbt", type=int, default=384)
    ap.add_argument("--enable-dspark", type=int, default=1,
                    help="enable DSpark K=5 + async-scheduling on decode (1=on, 0=off)")
    ap.add_argument("--gpu-mem", default="0.85")
    ap.add_argument("--gates-args", default="", help="extra args for pd_gates.py")
    ap.add_argument("--keep-pods", action="store_true")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL",
                    help="extra env for both servers (repeatable)")
    ap.add_argument("--decode-env", action="append", default=[], metavar="KEY=VAL",
                    help="extra env for the decode server only (repeatable)")
    ap.add_argument("--prefill-env", action="append", default=[], metavar="KEY=VAL",
                    help="extra env for the prefill server only (repeatable)")
    ap.add_argument("--proxy-dp-size", type=int, default=8,
                    help="proxy pins decode requests over this many DP ranks (0 = server LB)")
    ap.add_argument("--no-extra-whls", action="store_true", help="skip scripts/pd/whls/*.whl")
    ap.add_argument("--reuse-pods", action="store_true")
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="recreate/relaunch only the roles whose pod or server is gone")
    ap.add_argument("--allow-foreign", action="store_true")
    ap.add_argument("--boot-timeout", type=int, default=5400)
    ap.add_argument("--pending-timeout", type=int, default=7200,
                    help="how long a pod may wait for a free node")
    ap.add_argument("--hf-repo", default="",
                    help="optional HuggingFace repo id (e.g. deepseek-ai/DeepSeek-V4.1-Flash-1017) to pull instead of GCS default")
    a = ap.parse_args()
    L.install_signal_handlers()
    L.preflight(a.ctx)

    pods = {"prefill": L.pod_name(THREAD, f"pf-{a.suffix}"), "decode": L.pod_name(THREAD, f"dc-{a.suffix}")}
    kp, kd = L.Kube(a.ctx, pods["prefill"]), L.Kube(a.ctx, pods["decode"])
    kube = {"prefill": kp, "decode": kd}
    if not a.allow_foreign:
        foreign = kp.foreign_running_pods()
        if foreign:
            L.log(f"WARNING: other users' running pods on {a.ctx}: {foreign[:5]} (pass --allow-foreign)")
            sys.exit(2)

    # Extra wheels override the handoff set (e.g. the tpu-sync-torch the vtt tree pins:
    # 1d02e4a's pyproject wants 0.0.1.dev20260923040928, the handoff ships ...0917).
    extra = {f"workspace/whls_pd/{w.name}": w for w in sorted((HERE / "whls").glob("*.whl"))
             if not a.no_extra_whls}
    stack, prov = L.build_stack(Path(a.vtt), Path(a.vllm), HERE / "sitecustomize_pd.py", extra_files=extra)
    prov["extra_whls"] = sorted(extra)
    out = L.run_dir(THREAD, f"{a.tag}_{a.suffix}")
    run_id = f"{out.name.split('_')[0]}-{uuid.uuid4().hex[:6]}"
    (out / "provenance.json").write_text(json.dumps(
        {"ctx": a.ctx, "pods": pods, "run_id": run_id, "args": vars(a), **prov}, indent=2))
    claims = [L.Claim(a.ctx, p, THREAD, {"role": r, "run_dir": str(out)}) for r, p in pods.items()]
    L.log(f"run dir: {out}  pods: {pods}")
    summary: dict = {"run_dir": str(out), "pods": pods, "run_id": run_id}
    t_start = time.time()
    try:
        # Which roles get a fresh pod / a fresh stage + server launch.
        recreate = set() if (a.reuse_pods or a.gates_only) else set(pods)
        relaunch = set() if a.gates_only else set(pods)
        if a.resume:
            # Keep a role whose pod runs and whose server supervisor is alive (e.g. the
            # prefill pod still booting after the decode pod was OOM-killed or preempted).
            recreate, relaunch = set(), set()
            for r in pods:
                try:
                    phase = kube[r].out(["get", "pod", pods[r], "-o", "jsonpath={.status.phase}"]).strip()
                except subprocess.CalledProcessError:
                    phase = "Missing"
                alive = phase == "Running" and kube[r].exec_sh(
                    "pgrep -f in_pod_serve.sh >/dev/null && echo Y || echo N", capture=True).strip() == "Y"
                L.log(f"[{r}] resume: phase={phase} server_alive={alive}")
                if phase != "Running":
                    recreate.add(r)
                if not alive:
                    relaunch.add(r)
                else:
                    surviving_rid = kube[r].exec_sh(
                        "source /tmp/pd_env.sh 2>/dev/null && echo -n \"$RUN_ID\" || true", capture=True).strip()
                    if surviving_rid:
                        run_id = surviving_rid
                        summary["run_id"] = run_id
                        L.log(f"[{r}] resume: reusing surviving RUN_ID={run_id}")
            (out / "provenance.json").write_text(json.dumps(
                {"ctx": a.ctx, "pods": pods, "run_id": run_id, "args": vars(a), **prov}, indent=2))
        if recreate:
            for r in recreate:
                kube[r].delete()
                kube[r].apply(manifest(pods[r], r))
            nodes = parallel({r: (lambda r=r: kube[r].wait_ready(timeout_s=a.pending_timeout)) for r in recreate})
            L.log(f"pods Ready: {nodes}")
            summary["nodes"] = nodes
        summary["relaunched"] = sorted(relaunch)
        ips = {r: pod_ip(kube[r]) for r in pods}
        L.log(f"pod IPs: {ips}")
        summary["ips"] = ips

        if relaunch:
            tok = L.get_token()
            dl = L.download_ckpt_sh("all", tok, a.hf_repo).replace("python3 -u /tmp/dsv41_dl.py\n", "")
            # The token stays inside the pods (never in logs or the run dir).
            dl += (f"cat > /tmp/dsv41_env.sh <<'EOF'\nexport GCS_TOKEN='{tok}' "
                   f"DSV41_GCS_BUCKET='{L.GCS_BUCKET}' DSV41_GCS_PREFIX='{L.GCS_PREFIX}' "
                   f"DSV41_SHARDS=all DSV41_HF_REPO='{a.hf_repo}'\nEOF\n")
            L.upload_stack(stack)  # once, before the two pods fetch it in parallel

            def recreate_role(r: str) -> None:
                k = kube[r]
                L.log(f"[{r}] recreating pod {pods[r]} after spot preemption / pod failure ...")
                k.delete()
                k.apply(manifest(pods[r], r))
                node = k.wait_ready(timeout_s=a.pending_timeout)
                ips[r] = pod_ip(k)
                summary.setdefault("nodes", {})[r] = node
                summary.setdefault("ips", {})[r] = ips[r]
                L.log(f"[{r}] replacement pod Ready on {node} (IP {ips[r]})")

            def stage_once(r: str) -> None:
                k = kube[r]
                # Kill a previous run's processes by PID (this shell is `bash -s`, which
                # the pattern cannot match).
                k.exec_sh("for p in $(pgrep -f 'in_pod_serve.sh|vllm serve|VLLM::|EngineCore|"
                          "toy_proxy_server|pd_gates.py'); do [ \"$p\" != \"$$\" ] && "
                          "kill -9 $p 2>/dev/null || true; done; sleep 3", timeout=120)
                shm = k.exec_sh("free -g | head -2; df -h /dev/shm | tail -1; "
                                "du -sh /dev/shm/* 2>/dev/null | sort -h | tail -6", capture=True)
                L.log(f"[{r}] node memory / shm before staging:\n{shm}")
                L.stage_stack(k, stack)
                k.exec_sh(L.STAGE_BASE_SH)
                for fn in ("in_pod_setup.sh", "in_pod_serve.sh", "pd_gates.py"):
                    k.cp_in(HERE / fn, f"/tmp/{fn}")
                env = {
                    "ADVERTISE_HOST": ips[r], "RUN_ID": run_id,
                    "CED_PREFILL": str(a.prefill_ced),
                    "MAX_NUM_BATCHED_TOKENS": str(a.prefill_mnbt if r == "prefill" else a.decode_mnbt),
                    "ENABLE_DSPARK": "0" if r == "prefill" else str(a.enable_dspark),
                    "GPU_MEMORY_UTILIZATION": a.gpu_mem,
                    **dict(kv.split("=", 1) for kv in a.env),
                    **dict(kv.split("=", 1) for kv in (a.decode_env if r == "decode" else a.prefill_env)),
                }
                env_sh = "".join(f"export {kk}='{vv}'\n" for kk, vv in env.items())
                k.exec_sh(dl + f"cat > /tmp/pd_env.sh <<'EOF'\n{env_sh}EOF\n"
                          "rm -f /dev/shm/pd_*.log /dev/shm/pd_*.json /dev/shm/pd_*.gz /dev/shm/pd_stage3_check.txt\n")
                L.log(f"[{r}] installing + downloading checkpoint ...")
                k.exec_sh("bash /tmp/in_pod_setup.sh > /dev/shm/pd_setup.log 2>&1; tail -3 /dev/shm/pd_setup.log",
                          timeout=3600, retries=1)
                tail = k.exec_sh("tail -2 /dev/shm/pd_setup.log", capture=True)
                if "PD_SETUP_OK" not in tail:
                    raise RuntimeError(f"[{r}] setup failed: {tail}")
                L.log(f"[{r}] setup done: {tail.strip()}")

            def stage(r: str) -> None:
                for attempt in range(1, 4):
                    try:
                        stage_once(r)
                        return
                    except Exception as e:  # noqa: BLE001
                        phase = pod_phase(kube[r])
                        if attempt < 3 and phase != "Running":
                            L.log(f"[{r}] stage attempt {attempt} interrupted (pod phase={phase}, err={e}); recovering ...")
                            recreate_role(r)
                            continue
                        raise

            parallel({r: (lambda r=r: stage(r)) for r in relaunch})
            # Decode first, then prefill (the producer's first send needs a live consumer
            # only at request time, but this keeps the decode boot ahead).
            for r in [x for x in ("decode", "prefill") if x in relaunch]:
                L.start_detached(kube[r], f"exec bash /tmp/in_pod_serve.sh {r}", "in_pod_serve.sh")
            L.log("servers launched; cold boot is ~25 min")

            seen = {r: 0 for r in pods}
            t0 = time.time()
            ready = {r: False for r in pods}
            while not all(ready.values()):
                if time.time() - t0 > a.boot_timeout:
                    raise TimeoutError(f"boot timeout; ready={ready}")
                time.sleep(30)
                for r in pods:
                    k = kube[r]
                    phase = pod_phase(k)
                    if phase != "Running":
                        L.log(f"[{r}] pod phase={phase} during boot (spot preemption); recreating + re-staging {r} ...")
                        ready[r] = False
                        seen[r] = 0
                        recreate_role(r)
                        stage(r)
                        L.start_detached(k, f"exec bash /tmp/in_pod_serve.sh {r}", "in_pod_serve.sh")
                        continue
                    if ready[r]:
                        continue
                    try:
                        lines = grep_log(k, f"/dev/shm/pd_{r}.log", PROGRESS_PAT).splitlines()
                        for ln in lines[seen[r]:]:
                            print(f"  [{r}] {ln}", flush=True)
                        seen[r] = len(lines)
                        fails = grep_log(k, f"/dev/shm/pd_{r}.log", "|".join(
                            p.replace("[", r"\[").replace("]", r"\]") for p in FAIL_PAT))
                        if fails.strip():
                            tail = k.exec_sh(f"tail -n 80 /dev/shm/pd_{r}.log", capture=True)
                            (out / f"{r}_boot_failure_tail.txt").write_text(tail)
                            raise RuntimeError(f"[{r}] server failed during boot:\n{fails[-2000:]}\n--- tail ---\n{tail[-6000:]}")
                        is_4x_tp2 = any(x == "PREFILL_MODE=4x_tp2" for x in a.env)
                        role_healthy = (
                            all(healthy(k, PORTS["prefill"] + idx) for idx in range(4))
                            if (r == "prefill" and is_4x_tp2)
                            else healthy(k, PORTS[r])
                        )
                        if role_healthy:
                            ready[r] = True
                            L.log(f"[{r}] HEALTHY after {time.time() - t0:.0f}s")
                    except subprocess.CalledProcessError as e:
                        phase = pod_phase(k)
                        if phase != "Running":
                            L.log(f"[{r}] pod phase={phase} mid-poll (spot preemption: {e}); recreating + re-staging {r} ...")
                            ready[r] = False
                            seen[r] = 0
                            recreate_role(r)
                            stage(r)
                            L.start_detached(k, f"exec bash /tmp/in_pod_serve.sh {r}", "in_pod_serve.sh")
                            continue
                        raise
            summary["boot_s"] = round(time.time() - t0, 1)

        # Proxy on the prefill pod.
        is_4x_tp2 = any(x == "PREFILL_MODE=4x_tp2" for x in a.env)
        pf_proxy_flags = (
            "--prefiller-hosts 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 "
            "--prefiller-ports 8400 8401 8402 8403 --prefiller-dp-size 0"
            if is_4x_tp2
            else f"--prefiller-hosts 127.0.0.1 --prefiller-ports {PORTS['prefill']} --prefiller-dp-size {a.proxy_dp_size}"
        )
        # If the decode role was recreated (new pod IP) a surviving proxy on the prefill pod still
        # points at the old decoder -> every decode call fails with ConnectError. Restart it.
        if "decode" in relaunch and healthy(kp, PROXY_PORT, "/healthcheck"):
            L.log("decode was relaunched; restarting proxy so it targets the new decoder IP")
            kp.exec_sh("for p in $(pgrep -f toy_proxy_server); do kill -9 $p 2>/dev/null || true; done; sleep 2", timeout=60)
        if not healthy(kp, PROXY_PORT, "/healthcheck"):
            L.start_detached(kp, (
                "cd /workspace/vllm-torchtpu && exec python3 examples/disagg/toy_proxy_server.py "
                f"--host 0.0.0.0 --port {PROXY_PORT} {pf_proxy_flags} "
                f"--decoder-hosts {ips['decode']} --decoder-ports {PORTS['decode']} --decoder-dp-size {a.proxy_dp_size} "
                "> /dev/shm/pd_proxy.log 2>&1"),
                "toy_proxy_server.py")
            for _ in range(30):
                if healthy(kp, PROXY_PORT, "/healthcheck"):
                    break
                time.sleep(2)
            else:
                raise RuntimeError("proxy not healthy: " + kp.exec_sh("tail -30 /dev/shm/pd_proxy.log", capture=True))
        L.log("proxy healthy; running gates")
        if a.gates_only:
            kp.cp_in(HERE / "pd_gates.py", "/tmp/pd_gates.py")
        L.start_detached(kp, (f"exec python3 -u /tmp/pd_gates.py --decode http://{ips['decode']}:{PORTS['decode']} "
                              f"{a.gates_args} > /dev/shm/pd_gates.log 2>&1"), "pd_gates.py")
        seen_g = 0
        t0 = time.time()
        while True:
            time.sleep(20)
            txt = kp.exec_sh("cat /dev/shm/pd_gates.log 2>/dev/null || true", capture=True)
            if len(txt) > seen_g:
                sys.stdout.write(txt[seen_g:])
                sys.stdout.flush()
                seen_g = len(txt)
            alive = kp.exec_sh("pgrep -f pd_gates.py >/dev/null && echo Y || echo N", capture=True).strip()
            if alive == "N":
                break
            if time.time() - t0 > 7200:
                raise TimeoutError("gates did not finish in 2h")
        summary["gates_s"] = round(time.time() - t0, 1)

        chk = kp.exec_sh(
            "python3 /workspace/vllm-torchtpu/scripts/vllm/integration/check_dsv4_stage3_log.py "
            "/dev/shm/pd_prefill.log --min-sends 5 > /dev/shm/pd_stage3_check.txt 2>&1; "
            "echo rc=$? >> /dev/shm/pd_stage3_check.txt; cat /dev/shm/pd_stage3_check.txt", capture=True)
        print(chk, flush=True)
        summary["stage3_check"] = chk.strip().splitlines()
        summary["decode_stage3_events"] = grep_log(
            kd, "/dev/shm/pd_decode.log", "raiden_stage3|Stage-3").count("\n")
    finally:
        try:
            harvest(kp, kd, out)
        except Exception as e:  # noqa: BLE001
            L.log(f"harvest failed: {e}")
        gates = out / "prefill__pd_gates.json"
        if gates.exists():
            g = json.loads(gates.read_text())
            summary["gates"] = {
                "2p2_pd_passed": g.get("gate_2p2", {}).get("pd_passed"),
                "2p2_local_passed": g.get("gate_2p2", {}).get("local_passed"),
                "usa_pd_states": g.get("gate_usa", {}).get("pd_states"),
                "usa_local_states": g.get("gate_usa", {}).get("local_states"),
                "usa_pd_equals_local": g.get("gate_usa", {}).get("pd_equals_local"),
                "token_equality": [(c["prompt_tokens"], c["equal"], c["first_divergence"])
                                   for c in g.get("gate_token_equality", {}).get("cases", [])],
                "hard_gates_passed": g.get("hard_gates_passed"),
                "bench_pd": {k: v for k, v in g.get("bench_pd", {}).items() if k != "per_request"},
                "bench_local": {k: v for k, v in g.get("bench_local", {}).items() if k != "per_request"},
                "bench_pd_c64": {k: v for k, v in g.get("bench_pd_c64", {}).items() if k != "per_request"},
            }
        summary["wall_s"] = round(time.time() - t_start, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        L.log(f"summary -> {out / 'summary.json'}")
        if not a.keep_pods:
            for k in kube.values():
                k.delete()
            L.log(f"deleted pods {list(pods.values())}")
        for c in claims:
            c.release()


if __name__ == "__main__":
    main()
