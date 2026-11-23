# SPDX-License-Identifier: Apache-2.0
"""TIER-X2 sidecars (offload/tier_x2_sidecar.py) and the prefill's registry wait, on CPU with fake binaries.

The fakes stand in for tpu_sync's global_registry_server and kv_cache_host_store_node_main: the registry listens on
its --port; a node records its argv and listens on its --raiden_controller_port after a moment (a node named
"mute" never listens, one named "bad" exits with rc 3).
"""

import os
import signal
import socket
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

FAKE_REGISTRY = r'''
import socket, sys, time
port = int([a for a in sys.argv if a.startswith("--port=")][0].split("=", 1)[1])
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", port)); s.listen(8)
print("fake registry on", port, flush=True)
while True:
    time.sleep(1)
'''

FAKE_NODE = r'''
import os, socket, sys, time
args = dict((a[2:].split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
open(os.path.join(os.environ["FAKE_ARGS_DIR"], args["job_name"] + ".argv"), "w").write("\n".join(sys.argv[1:]))
if "bad" in args["job_name"]:
    sys.exit(3)
time.sleep(0.3)
if "mute" not in args["job_name"]:
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((args["store_server_ip"], int(args["raiden_controller_port"]))); s.listen(8)
if "stats_period_s" in args:
    print("Q38X2NODE_STATS free=3 total=4 job=" + args["job_name"], flush=True)
while True:
    time.sleep(1)
'''


def _free_base(n):
    for base in range(23000, 29000, 37):
        ok = True
        for p in range(base, base + n + 1):
            with socket.socket() as s:
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    ok = False
                    break
        if ok:
            return base
    raise RuntimeError("no free port run")


def _fake_root(tmp_path):
    root = tmp_path / "tpu_sync"
    for rel, body in (
        (("kv_cache", "global_registry", "global_registry_server"), FAKE_REGISTRY),
        (("store_node", "kv_cache_host_store_node_main"), FAKE_NODE),
    ):
        p = root.joinpath(*rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"#!{sys.executable}\n" + body)
        os.chmod(p, 0o644)  # as zipfile extracts a wheel: no execute bit
    return str(root)


def _env(tmp_path, nodes="main.r1:1000;main.r2:2000", registry=True, **extra):
    base = _free_base(8)
    (tmp_path / "args").mkdir(exist_ok=True)
    (tmp_path / "out").mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "FAKE_ARGS_DIR": str(tmp_path / "args"),
        "CDK_OUTPUT_DIR": str(tmp_path / "out"),
        "Q38X2_LOCK_DIR": str(tmp_path),
        "Q38X2_BIN_DIR": str(tmp_path / "bin"),
        "Q38X2_NODE_PORT_BASE": str(base + 1),
        "THE_HOST": "127.0.0.1",
    }
    if registry:
        env["Q38X2_REGISTRY_PORT"] = str(base)
    if nodes:
        env.update(
            Q38X2_NODES=nodes,
            Q38X2_NODE_REGISTRY=f"127.0.0.1:{base}",
            Q38X2_NODE_IP="${THE_HOST}",
        )
    env.update(extra)
    return env


def _lines(tmp_path):
    p = tmp_path / "out" / "q38x2" / "sidecar.log"
    return p.read_text().splitlines() if p.exists() else []


def _kill_started(tmp_path):
    for line in _lines(tmp_path):
        for word in line.split():
            if word.startswith("pid=") and ("_START" in line):
                try:
                    os.kill(int(word[4:]), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass


def test_parse_nodes():
    from vllm_torchtpu.offload.tier_x2_sidecar import parse_nodes

    assert parse_nodes("main.r1:10;ple:3") == [("main.r1", 10), ("ple", 3)]
    for bad in ("", "main.r1", "main.r1:", ":5", "main.r1:0", "a:1;a:2", "a:x"):
        with pytest.raises(ValueError):
            parse_nodes(bad)


def test_node_commands_name_the_prefill_shards_pool_groups(monkeypatch):
    from vllm_torchtpu.offload.raiden_store import TPURaidenStoreOffloadingSpec
    from vllm_torchtpu.offload.tier_classes import SERVED_SHARDS
    from vllm_torchtpu.offload.tier_x2_sidecar import node_commands

    env = {
        "JS_NAME": "jobx",
        "POD_IP": "10.9.8.7",
        "Q38X2_NODES": ";".join(f"{s}:{(i + 1) * 1000}" for i, s in enumerate(SERVED_SHARDS)),
        "Q38X2_NODE_REGISTRY": "${JS_NAME}-prefill-0-0.${JS_NAME}:31600",
        "Q38X2_NODE_IP": "${POD_IP}",
        "Q38X2_NODE_GROUP_PREFIX": "q38x2",
    }
    cmds = node_commands(env, "/bin/node")
    assert [c[0] for c in cmds] == list(SERVED_SHARDS)
    assert [c[1] for c in cmds] == list(range(31700, 31709))
    for i, (shard, port, argv) in enumerate(cmds):
        want = TPURaidenStoreOffloadingSpec.tier_x2_pool_group(
            NS(kv_pool_group="q38x2"), NS(name=shard)
        )
        assert f"--kv_pool_group={want}" in argv
        assert "--global_registry_address=jobx-prefill-0-0.jobx:31600" in argv
        assert "--store_server_ip=10.9.8.7" in argv
        assert f"--raiden_controller_port={port}" in argv
        assert f"--dram_budget_bytes={(i + 1) * 1000}" in argv
        assert "--evict_tier=1" in argv and argv[0] == "/bin/node"
    del env["POD_IP"]
    with pytest.raises(ValueError, match="unset environment variable"):
        node_commands(env, "/bin/node")


def test_node_commands_carry_the_lru_flags_only_when_asked():
    """W1f: the node's own evict sweep, stats line and log threshold, only on request (a W1e node exits on an
    unknown flag)."""
    from vllm_torchtpu.offload import tier_x2_sidecar as sc

    base = {"Q38X2_NODES": "main.r1:1000", "Q38X2_NODE_REGISTRY": "r:1", "Q38X2_NODE_IP": "10.0.0.2"}
    lru = ("--enable_evict_sweep", "--evict_low_watermark", "--evict_high_watermark", "--evict_sweep_period_s",
           "--stats_period_s", "--log_stderr_threshold")
    ((_, _, argv),) = sc.node_commands(base, "/bin/node")
    assert not [a for a in argv if any(k in a for k in lru)]
    ((_, _, argv),) = sc.node_commands({**base, "Q38X2_NODE_EVICT_SWEEP": "0"}, "/bin/node")
    assert not [a for a in argv if any(k in a for k in lru)]
    ((_, _, argv),) = sc.node_commands({**base, "Q38X2_NODE_EVICT_SWEEP": "1"}, "/bin/node")
    assert argv[-4:] == [
        "--enable_evict_sweep", "--evict_low_watermark=0.05", "--evict_high_watermark=0.1", "--evict_sweep_period_s=1",
    ]
    ((_, _, argv),) = sc.node_commands(
        {
            **base, "Q38X2_NODE_EVICT_SWEEP": "1", "Q38X2_NODE_EVICT_LOW": "0.2", "Q38X2_NODE_EVICT_HIGH": "0.4",
            "Q38X2_NODE_EVICT_PERIOD_S": "0.5", "Q38X2_NODE_STATS_S": "30", "Q38X2_NODE_LOG_THRESHOLD": "0",
        },
        "/bin/node",
    )
    assert argv[-6:] == [
        "--enable_evict_sweep", "--evict_low_watermark=0.2", "--evict_high_watermark=0.4",
        "--evict_sweep_period_s=0.5", "--stats_period_s=30", "--log_stderr_threshold=0",
    ]
    for bad in (
        {"Q38X2_NODE_EVICT_LOW": "0.3", "Q38X2_NODE_EVICT_HIGH": "0.2"},
        {"Q38X2_NODE_EVICT_PERIOD_S": "0"},
        {"Q38X2_NODE_STATS_S": "-1"},
        {"Q38X2_NODE_LOG_THRESHOLD": "5"},
    ):
        with pytest.raises(ValueError):
            sc.node_commands({**base, "Q38X2_NODE_EVICT_SWEEP": "1", **bad}, "/bin/node")


def test_watch_gives_each_node_its_own_output_file(tmp_path):
    """Each node's output (its Q38X2NODE_STATS lines) lands in node_<shard>.log, so the read can tell the nodes apart."""
    from vllm_torchtpu.offload import tier_x2_sidecar as sc

    root = _fake_root(tmp_path)
    env = _env(tmp_path, nodes="main.r1:1000;ple:2000", registry=False, Q38X2_NODE_EVICT_SWEEP="1", Q38X2_NODE_STATS_S="1")
    try:
        sc.watch(root, env, poll_s=0.1, stop_after_s=3)
        for shard, other in (("main.r1", "ple"), ("ple", "main.r1")):
            text = (tmp_path / "out" / "q38x2" / f"node_{shard}.log").read_text()
            assert f"Q38X2NODE_STATS free=3 total=4 job=q38x2-node-{shard}" in text
            assert f"job=q38x2-node-{other}" not in text
        argv = (tmp_path / "args" / "q38x2-node-main.r1.argv").read_text().splitlines()
        assert "--enable_evict_sweep" in argv and "--stats_period_s=1" in argv
    finally:
        _kill_started(tmp_path)


def test_inert_without_the_environment(tmp_path):
    from vllm_torchtpu.offload import tier_x2_sidecar as sc

    assert sc.maybe_start({"Q38X2_LOCK_DIR": str(tmp_path)}) is False
    assert not (tmp_path / "q38x2_sidecar.lock").exists()
    # The package import runs the hook (inert here: the test env names no sidecar).
    import os

    import vllm_torchtpu

    # On the Q38_OXPD_FORM-gated package the default form's body is _bl_init.py
    # (vllm_torchtpu/__init__.py is the gate that executes it).
    body = os.path.join(os.path.dirname(vllm_torchtpu.__file__), "_bl_init.py")
    if not os.path.exists(body):
        body = vllm_torchtpu.__file__
    assert "_tier_x2_sidecar.maybe_start()" in open(body).read()


def test_nothing_starts_before_the_cars_wheel_is_installed(tmp_path, monkeypatch):
    from vllm_torchtpu.offload import tier_x2_sidecar as sc

    root = _fake_root(tmp_path)
    monkeypatch.setattr(sc, "_tpu_sync_dir", lambda env=None: root)
    monkeypatch.setattr(sc, "_tpu_sync_version", lambda: "0.0.1.dev20261012084241")
    env = _env(tmp_path, Q38X2_TPU_SYNC_VERSION="0.0.1.dev20261112161350")
    assert sc.maybe_start(env) is False
    assert not (tmp_path / "q38x2_sidecar.lock").exists()
    assert any("Q38X2_SIDECAR waiting" in l for l in _lines(tmp_path))


def test_watch_starts_the_registry_and_nodes_and_reports_them(tmp_path):
    from vllm_torchtpu.offload import tier_x2_sidecar as sc

    root = _fake_root(tmp_path)
    env = _env(tmp_path, nodes="main.r1:1000;mute:5;bad:7;ple:2000")
    try:
        sc.watch(root, env, poll_s=0.1, stop_after_s=6)
        lines = _lines(tmp_path)
        text = "\n".join(lines)
        assert sum("Q38X2_REGISTRY_UP" in l for l in lines) == 1
        ups = [l for l in lines if l.startswith("Q38X2_NODE_UP")]
        assert sorted(l.split()[1] for l in ups) == ["shard=main.r1", "shard=ple"]
        assert all(" boot_s=" in l and " rss_gib=" in l for l in ups)
        assert "Q38X2_NODE_EXIT shard=bad" in text and "rc=3" in text and "was_up=False" in text
        assert "shard=mute" not in "\n".join(ups)
        assert "Q38X2_NODES_ALL_UP" not in text  # mute never came up
        assert any(l.startswith("Q38X2_MEM nodes_rss_gib=") and "cgroup_max_gib=" in l for l in lines)
        argv = (tmp_path / "args" / "q38x2-node-main.r1.argv").read_text().splitlines()
        assert "--kv_pool_group=q38x2-main.r1" in argv and "--store_server_ip=127.0.0.1" in argv
        assert "--dram_budget_bytes=1000" in argv
        assert os.access(tmp_path / "bin" / "kv_cache_host_store_node_main", os.X_OK)
    finally:
        _kill_started(tmp_path)


def test_one_watcher_per_pod(tmp_path, monkeypatch):
    from vllm_torchtpu.offload import tier_x2_sidecar as sc

    root = _fake_root(tmp_path)
    monkeypatch.setattr(sc, "_tpu_sync_dir", lambda env=None: root)
    monkeypatch.setattr(sc, "_tpu_sync_version", lambda: "0.0.1.dev20261112161350")
    env = _env(tmp_path, Q38X2_TPU_SYNC_VERSION="0.0.1.dev20261112161350")
    watcher = None
    try:
        assert sc.maybe_start(env) is True
        assert sc.maybe_start(env) is False  # the lock: one watcher a pod
        line = next(l for l in _lines(tmp_path) if "Q38X2_SIDECAR watcher pid" in l)
        watcher = int(line.split("watcher pid ")[1].split()[0])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if any(l.startswith("Q38X2_NODES_ALL_UP up=2 of 2") for l in _lines(tmp_path)):
                break
            time.sleep(0.2)
        lines = _lines(tmp_path)
        assert any(l.startswith("Q38X2_NODES_ALL_UP up=2 of 2") for l in lines), lines
        assert sum(l.startswith("Q38X2_REGISTRY_UP") for l in lines) == 1
        # The watcher and the binaries write to their own file, not the importing process's pipes.
        procs = (tmp_path / "out" / "q38x2" / "sidecar_procs.log").read_text()
        assert "fake registry on" in procs and "Q38X2_NODE_UP" in procs
    finally:
        if watcher:
            try:
                os.kill(watcher, signal.SIGKILL)
            except ProcessLookupError:
                pass
        _kill_started(tmp_path)


def test_the_registry_wait(monkeypatch):
    from vllm_torchtpu.offload.raiden_store import x2_wait_registry

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert x2_wait_registry(f"127.0.0.1:{port}", timeout_s=5) < 5
    with pytest.raises(ValueError, match="did not answer within 1 s"):
        x2_wait_registry(f"127.0.0.1:{port}", timeout_s=1)


def test_the_spec_waits_for_the_registry_before_the_shard_stores(monkeypatch):
    from tests.offload.test_q38_tier_x2 import REGISTRY, _x2_env
    from tests.offload.test_tier_classes import _served, _spec
    from vllm_torchtpu.offload import raiden_store

    cfg, _, stages, kvc = _served(monkeypatch)
    _x2_env(monkeypatch)
    spec = _spec(monkeypatch, cfg, stages, kvc, **REGISTRY)
    calls = []
    monkeypatch.setattr(raiden_store, "x2_wait_registry", lambda a: calls.append(a) or 0.0)
    monkeypatch.setattr(raiden_store, "RaidenOffloadingManager", MagicMock())
    monkeypatch.setattr(
        "vllm_torchtpu.offload.tier_classes.TieredOffloadingManager",
        lambda plan, managers, check=False, fp4_slots=None: NS(plan=plan, managers=managers),
    )
    spec.build_tier_manager()
    assert calls == [REGISTRY["global_registry_address"]]
    spec.build_tier_manager(stores=[object()] * 9, store_types=object())
    assert calls == [REGISTRY["global_registry_address"]]  # injected stores (tests): no wait


def test_the_importing_process_pipes_are_not_held(tmp_path):
    """The importer returns and its pipes close although the watcher and the binaries keep running."""
    import subprocess

    import vllm_torchtpu.offload.tier_x2_sidecar as sc

    root = _fake_root(tmp_path)
    env = _env(tmp_path, Q38X2_TPU_SYNC_DIR=root)
    code = "import importlib.util as u, sys; s = u.spec_from_file_location('sc', sys.argv[1]); m = u.module_from_spec(s); s.loader.exec_module(m); print(m.maybe_start())"
    try:
        out = subprocess.run([sys.executable, "-c", code, sc.__file__], env=env, capture_output=True, text=True, timeout=30)
        assert out.stdout.strip() == "True", out
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not any(l.startswith("Q38X2_NODES_ALL_UP") for l in _lines(tmp_path)):
            time.sleep(0.2)
        assert any(l.startswith("Q38X2_NODES_ALL_UP up=2 of 2") for l in _lines(tmp_path)), _lines(tmp_path)
    finally:
        for line in _lines(tmp_path):
            if "Q38X2_SIDECAR watcher pid" in line:
                try:
                    os.kill(int(line.split("watcher pid ")[1].split()[0]), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
        _kill_started(tmp_path)
