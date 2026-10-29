# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The TPU release wait must poll real ownership, not a fixed duration."""

import os

import pytest
import tpu_release


def _fake_proc(monkeypatch, links: dict[str, str]) -> None:
    """Present `links` as the open descriptors of every visible process."""
    monkeypatch.setattr(tpu_release, "_fd_links", lambda: list(links))
    monkeypatch.setattr(tpu_release.os, "readlink", lambda link: links[link])


def test_no_owner_when_nothing_holds_a_device(monkeypatch):
    _fake_proc(monkeypatch, {"/proc/123/fd/4": "/tmp/scratch"})
    assert tpu_release.tpu_device_owners() == {}


def test_reports_another_process_holding_a_chip(monkeypatch):
    _fake_proc(monkeypatch, {"/proc/123/fd/4": "/dev/vfio/0"})
    assert tpu_release.tpu_device_owners() == {"/dev/vfio/0": 123}


def test_ignores_a_chip_held_by_the_calling_process(monkeypatch):
    """pytest owns its own chip for good once it touches the TPU runtime."""
    _fake_proc(monkeypatch, {f"/proc/{os.getpid()}/fd/4": "/dev/vfio/0"})
    assert tpu_release.tpu_device_owners() == {}


def test_scan_skips_the_self_symlinks():
    """`/proc/self` resolves to the caller, and its name is not a number.

    A scan that includes it reports the caller as a second owner of its own
    chip, and anything parsing that name as a pid fails on it.
    """
    assert all(link.split("/")[2].isdigit() for link in tpu_release._fd_links())


def test_a_process_exiting_mid_scan_is_not_an_owner(monkeypatch):
    """The holder exiting is the outcome being waited for, not an error."""

    def vanished(_link):
        raise FileNotFoundError

    monkeypatch.setattr(tpu_release, "_fd_links", lambda: ["/proc/123/fd/4"])
    monkeypatch.setattr(tpu_release.os, "readlink", vanished)
    assert tpu_release.tpu_device_owners() == {}


def test_returns_once_the_holder_exits(monkeypatch):
    """The wait ends on the poll after the owner disappears, not on a timer."""
    remaining = [{"/dev/vfio/0": 123}] * 3 + [{}]
    monkeypatch.setattr(tpu_release, "tpu_device_owners", lambda: remaining.pop(0))
    monkeypatch.setattr(tpu_release.time, "sleep", lambda _: None)

    tpu_release.wait_for_tpu_release()

    assert not remaining


def test_raises_and_names_the_holder_when_the_chip_never_frees(monkeypatch):
    monkeypatch.setattr(tpu_release, "tpu_device_owners", lambda: {"/dev/vfio/0": 4242})

    with pytest.raises(RuntimeError, match="4242"):
        tpu_release.wait_for_tpu_release(timeout=0.0)


@pytest.mark.parametrize(
    "log_text",
    [
        "RuntimeError: PjRtClient is not initialized",
        (
            "RuntimeError: failed to retrieve the PjRt client: "
            "TPU initialization failed: GRPC_ERROR error message: "
            "START_SESSION failed: deadline exceeded"
        ),
        "libtpu timed out at PRE_START_SESSION_BARRIER",
        "TPU initialization failed: Failed to start the gRPC server.",
        (
            "TPU initialization failed: Failed to connect to "
            "unix:///var/run/tpu-plugin/vbar.sock"
        ),
        "TPU initialization failed: Failed to acquire a TPU device node",
    ],
)
def test_recognizes_transient_tpu_init_race_messages(log_text: str):
    assert tpu_release.is_tpu_init_race(log_text)


@pytest.mark.parametrize(
    "log_text",
    [
        "",
        "TPU initialization failed: Invalid argument: Unknown flag --xla_bad",
        "AssertionError: output mismatch",
        "xla::ifrt::PjRtClient::~PjRtClient()",
    ],
)
def test_rejects_deterministic_errors_as_non_retryable(log_text: str):
    assert not tpu_release.is_tpu_init_race(log_text)


def test_cp_mla_launch_retries_libtpu_050_start_session_race_and_clears_stale_json(
    monkeypatch, tmp_path
):
    import json
    import subprocess

    from layers.adapter import test_cp_mla_merge_multichip as cp_mla

    calls = 0

    def fake_launch_once(result_dir, _extra):
        nonlocal calls
        calls += 1
        if calls == 1:
            (result_dir / "rank_3.json").write_text(
                json.dumps(
                    {
                        "rank": 3,
                        "error": (
                            "RuntimeError('failed to retrieve the PjRt client: "
                            "TPU initialization failed: GRPC_ERROR error message: "
                            "START_SESSION failed')"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(["worker"], 1, "worker failed", "")
        assert not list(result_dir.glob("rank_*.json"))
        (result_dir / "rank_0.json").write_text(
            json.dumps({"rank": 0, "ok": True}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(["worker"], 0, "ok", "")

    monkeypatch.setattr(cp_mla, "_launch_once", fake_launch_once)
    monkeypatch.setattr(cp_mla.time, "sleep", lambda _: None)
    monkeypatch.setattr(tpu_release, "wait_for_tpu_release", lambda *a, **kw: None)

    completed = cp_mla._launch(tmp_path, [])
    assert completed.returncode == 0
    assert calls == 2
    assert not (tmp_path / "rank_3.json").exists()


def test_latent_proj_does_not_retry_non_init_failure(monkeypatch, tmp_path):
    import json
    import subprocess

    from layers.adapter import test_latent_proj_intra_chip_multichip as latent

    calls = 0

    def fake_once(result_dir):
        nonlocal calls
        calls += 1
        (result_dir / "rank_0.json").write_text(
            json.dumps({"rank": 0, "error": "AssertionError('max diff 0.25')"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(["worker"], 1, "AssertionError", "")

    monkeypatch.setattr(latent, "_run_worker_group_once", fake_once)
    monkeypatch.setattr(latent.time, "sleep", lambda _: None)
    monkeypatch.setattr(tpu_release, "wait_for_tpu_release", lambda *a, **kw: None)

    completed = latent._run_worker_group(tmp_path)
    assert completed.returncode == 1
    assert calls == 1
