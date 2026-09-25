# SPDX-License-Identifier: Apache-2.0
"""The TPU release wait must poll real ownership, not a fixed duration."""

import os

import pytest
import tpu_release


def _fake_proc(monkeypatch, links: dict[str, str]) -> None:
    """Present `links` as the open descriptors of every visible process."""
    monkeypatch.setattr(tpu_release, "_fd_links", lambda: list(links))
    monkeypatch.setattr(tpu_release.os, "readlink", lambda link: links[link])


@pytest.mark.cpu_test
def test_no_owner_when_nothing_holds_a_device(monkeypatch):
    _fake_proc(monkeypatch, {"/proc/123/fd/4": "/tmp/scratch"})
    assert tpu_release.tpu_device_owners() == {}


@pytest.mark.cpu_test
def test_reports_another_process_holding_a_chip(monkeypatch):
    _fake_proc(monkeypatch, {"/proc/123/fd/4": "/dev/vfio/0"})
    assert tpu_release.tpu_device_owners() == {"/dev/vfio/0": 123}


@pytest.mark.cpu_test
def test_ignores_a_chip_held_by_the_calling_process(monkeypatch):
    """pytest owns its own chip for good once it touches the TPU runtime."""
    _fake_proc(monkeypatch, {f"/proc/{os.getpid()}/fd/4": "/dev/vfio/0"})
    assert tpu_release.tpu_device_owners() == {}


@pytest.mark.cpu_test
def test_scan_skips_the_self_symlinks():
    """`/proc/self` resolves to the caller, and its name is not a number.

    A scan that includes it reports the caller as a second owner of its own
    chip, and anything parsing that name as a pid fails on it.
    """
    assert all(link.split("/")[2].isdigit() for link in tpu_release._fd_links())


@pytest.mark.cpu_test
def test_a_process_exiting_mid_scan_is_not_an_owner(monkeypatch):
    """The holder exiting is the outcome being waited for, not an error."""

    def vanished(_link):
        raise FileNotFoundError

    monkeypatch.setattr(tpu_release, "_fd_links", lambda: ["/proc/123/fd/4"])
    monkeypatch.setattr(tpu_release.os, "readlink", vanished)
    assert tpu_release.tpu_device_owners() == {}


@pytest.mark.cpu_test
def test_returns_once_the_holder_exits(monkeypatch):
    """The wait ends on the poll after the owner disappears, not on a timer."""
    remaining = [{"/dev/vfio/0": 123}] * 3 + [{}]
    monkeypatch.setattr(tpu_release, "tpu_device_owners", lambda: remaining.pop(0))
    monkeypatch.setattr(tpu_release.time, "sleep", lambda _: None)

    tpu_release.wait_for_tpu_release()

    assert not remaining


@pytest.mark.cpu_test
def test_raises_and_names_the_holder_when_the_chip_never_frees(monkeypatch):
    monkeypatch.setattr(tpu_release, "tpu_device_owners", lambda: {"/dev/vfio/0": 4242})

    with pytest.raises(RuntimeError, match="4242"):
        tpu_release.wait_for_tpu_release(timeout=0.0)
