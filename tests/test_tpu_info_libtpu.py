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
"""libtpu identification when the pip package may be absent."""

import io
import logging
import os
import stat
import types
from importlib import metadata as importlib_metadata
from unittest import mock

import pytest

from vllm_torchtpu import envs, tpu_info
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.worker import tpu_worker


def _versions(libtpu):
    def version(name):
        if name == "libtpu" and libtpu is None:
            raise importlib_metadata.PackageNotFoundError(name)
        return libtpu if name == "libtpu" else "1"

    return version


def test_get_libtpu_version_uses_package_metadata(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.50"))
    monkeypatch.delenv("TPU_LIBRARY_PATH", raising=False)

    assert tpu_info.get_libtpu_version() == "0.0.50"
    assert tpu_info.get_libtpu_cache_key() == "0.0.50"


def test_get_libtpu_version_without_package(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))

    assert tpu_info.get_libtpu_version() is None


def test_libtpu_cache_key_fingerprints_tpu_library_path(monkeypatch, tmp_path):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))
    first = tmp_path / "a" / "libtpu.so"
    second = tmp_path / "b" / "libtpu.so"
    for path, payload in ((first, b"build-a" * 4096), (second, b"build-b" * 4096)):
        path.parent.mkdir()
        path.write_bytes(payload)

    monkeypatch.setenv("TPU_LIBRARY_PATH", str(first))
    key_first = tpu_info.get_libtpu_cache_key()
    assert key_first.startswith("unpackaged-")
    assert tpu_info.get_libtpu_cache_key() == key_first

    monkeypatch.setenv("TPU_LIBRARY_PATH", str(second))
    assert tpu_info.get_libtpu_cache_key() != key_first


def test_libtpu_cache_key_without_package_or_library(monkeypatch, tmp_path):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(tmp_path / "missing.so"))

    assert tpu_info.get_libtpu_cache_key() == "unpackaged-unknown"


def test_libtpu_cache_key_prefers_tpu_library_path_over_package(monkeypatch, tmp_path):
    # JAX loads TPU_LIBRARY_PATH in preference to the pip package, so the key
    # must follow the library file even when libtpu==0.0.50 is installed.
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.50"))
    library = tmp_path / "libtpu.so"
    library.write_bytes(b"build-a" * 4096)
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(library))

    key = tpu_info.get_libtpu_cache_key()
    assert key.startswith("unpackaged-")
    assert key != "0.0.50"

    library.write_bytes(b"build-b" * 8192)
    assert tpu_info.get_libtpu_cache_key() not in (key, "0.0.50")


def test_libtpu_cache_key_uses_package_version_for_its_own_library(
    monkeypatch, tmp_path
):
    # `import libtpu` exports TPU_LIBRARY_PATH as the package's own library;
    # like JAX, that path is not custom, so the package version identifies it.
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.50"))
    library = tmp_path / "libtpu.so"
    library.write_bytes(b"build-a" * 4096)
    monkeypatch.setattr(tpu_info, "_pip_libtpu_library_path", lambda: str(library))
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(library))

    assert tpu_info.get_tpu_library_path() is None
    assert tpu_info.get_libtpu_cache_key() == "0.0.50"


def test_validate_libtpu_version_skips_without_package(monkeypatch, caplog):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions(None))
    monkeypatch.delenv("TPU_LIBRARY_PATH", raising=False)

    with caplog.at_level(logging.WARNING):
        tpu_runner._validate_libtpu_version()

    assert "libtpu package metadata not found" in caplog.text
    assert "skipping the libtpu version check" in caplog.text


def test_validate_libtpu_version_skips_package_check_with_tpu_library_path(
    monkeypatch, tmp_path, caplog
):
    # An installed package older than the minimum would raise, but JAX loads
    # the TPU_LIBRARY_PATH library instead, so the package is not checked.
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.34"))
    library = tmp_path / "libtpu.so"
    library.write_bytes(b"build-a" * 4096)
    monkeypatch.setenv("TPU_LIBRARY_PATH", str(library))

    with caplog.at_level(logging.WARNING):
        tpu_runner._validate_libtpu_version()

    assert f"TPU_LIBRARY_PATH={library}" in caplog.text
    assert "skipping the libtpu package version check" in caplog.text


def test_validate_libtpu_version_still_rejects_old_package(monkeypatch):
    monkeypatch.setattr(tpu_info.importlib_metadata, "version", _versions("0.0.34"))
    monkeypatch.delenv("TPU_LIBRARY_PATH", raising=False)

    with pytest.raises(RuntimeError, match="libtpu < 0.0.35"):
        tpu_runner._validate_libtpu_version()


def test_get_node_worker_id_prefers_tpu_worker_id_then_lws_worker_index(monkeypatch):
    monkeypatch.setattr(
        tpu_info,
        "get_tpu_metadata",
        lambda key="": pytest.fail("GCE metadata should not be queried"),
    )
    monkeypatch.setenv("TPU_WORKER_ID", "2")
    monkeypatch.setenv("LWS_WORKER_INDEX", "1")
    assert tpu_info.get_node_worker_id() == 2

    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    assert tpu_info.get_node_worker_id() == 1


def test_get_node_worker_id_falls_back_to_gce_metadata_and_zero(monkeypatch):
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    monkeypatch.delenv("LWS_WORKER_INDEX", raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "3")
    assert tpu_info.get_node_worker_id() == 3

    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": None)
    assert tpu_info.get_node_worker_id() == 0


def _patch_vfio_fs(
    monkeypatch,
    *,
    vfio_entries,
    iommu_devices_fn,
    open_sysfs_fn,
    stat_fn=None,
    access_fn=None,
    exists_fn=None,
    readlink_fn=None,
):
    real_stat = os.stat
    real_access = os.access
    real_listdir = os.listdir
    real_exists = os.path.exists
    real_readlink = os.readlink
    real_open = open

    def _stat(path, *args, **kwargs):
        s = os.fspath(path)
        if isinstance(s, str) and s.startswith("/dev/vfio/"):
            if stat_fn is not None:
                return stat_fn(s)
            return types.SimpleNamespace(st_mode=stat.S_IFCHR | 0o660)
        return real_stat(path, *args, **kwargs)

    def _access(path, mode, *args, **kwargs):
        s = os.fspath(path)
        if isinstance(s, str) and s.startswith("/dev/vfio/"):
            return access_fn(s, mode) if access_fn is not None else True
        return real_access(path, mode, *args, **kwargs)

    def _listdir(path="."):
        s = os.fspath(path)
        if s == "/dev/vfio":
            return list(vfio_entries)
        if isinstance(s, str) and s.startswith("/sys/kernel/iommu_groups/"):
            return iommu_devices_fn(s)
        return real_listdir(path)

    def _exists(path):
        s = os.fspath(path)
        if isinstance(s, str) and s.startswith("/sys/kernel/iommu_groups/"):
            return exists_fn(s) if exists_fn is not None else False
        return real_exists(path)

    def _readlink(path, *args, **kwargs):
        s = os.fspath(path)
        if (
            readlink_fn is not None
            and isinstance(s, str)
            and s.startswith("/sys/kernel/iommu_groups/")
        ):
            return readlink_fn(s)
        return real_readlink(path, *args, **kwargs)

    def _open(file, *args, **kwargs):
        s = os.fspath(file) if isinstance(file, (str, os.PathLike)) else ""
        if isinstance(s, str) and s.startswith("/sys/kernel/iommu_groups/"):
            return open_sysfs_fn(s)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(tpu_info.os, "stat", _stat)
    monkeypatch.setattr(tpu_info.os, "access", _access)
    monkeypatch.setattr(tpu_info.os, "listdir", _listdir)
    monkeypatch.setattr(tpu_info.os.path, "exists", _exists)
    monkeypatch.setattr(tpu_info.os, "readlink", _readlink)
    monkeypatch.setattr("builtins.open", _open)


def test_get_num_chips_ignores_dev_accel_directory(monkeypatch):
    real_isdir = os.path.isdir
    monkeypatch.setattr(
        tpu_info.glob, "glob", lambda pattern: ["/dev/accel", "/dev/accel0"]
    )
    monkeypatch.setattr(
        tpu_info.os.path,
        "isdir",
        lambda path: True
        if path == "/dev/accel"
        else (False if str(path).startswith("/dev/accel") else real_isdir(path)),
    )
    assert tpu_info.get_num_chips() == 1

    # When only /dev/accel directory exists (e.g. NVIDIA DRM accel), fall back to VFIO.
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: ["/dev/accel"])
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=["0", "1"],
        iommu_devices_fn=lambda _: ["0000:00:04.0"],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x1ae0\n" if s.endswith("/vendor") else "0x0076\n"
        ),
    )
    assert tpu_info.get_num_chips() == 2


def test_get_num_chips_filters_accessible_vfio_tpu_device_ids_without_cap(monkeypatch):
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: [])
    # Groups 0..7: accessible TPU7x VFs (0x1ae0:0x0076); 8..13: PFs/mgmt
    # (0x0075/0x0077/0x0078); 14: Google NVMe (0x001f); 15: BlueField (0x15b3);
    # 16: unreadable placeholder; 17: regular file.
    pci_ids_by_group = {
        **{g: ("0x1ae0", "0x0076") for g in range(8)},
        **{g: ("0x1ae0", "0x0075") for g in range(8, 12)},
        12: ("0x1ae0", "0x0077"),
        13: ("0x1ae0", "0x0078"),
        14: ("0x1ae0", "0x001f"),
        15: ("0x15b3", "0xa2dc"),
        16: ("0x1ae0", "0x0076"),
        17: ("0x1ae0", "0x0076"),
    }

    def _open_mixed(path: str):
        group = int(path.split("/sys/kernel/iommu_groups/")[1].split("/")[0])
        vendor, device = pci_ids_by_group[group]
        return io.StringIO(f"{vendor if path.endswith('/vendor') else device}\n")

    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=[str(i) for i in range(18)] + ["vfio"],
        iommu_devices_fn=lambda _: ["0000:00:04.0"],
        open_sysfs_fn=_open_mixed,
        stat_fn=lambda s: types.SimpleNamespace(
            st_mode=(stat.S_IFREG if s == "/dev/vfio/17" else stat.S_IFCHR) | 0o660
        ),
        access_fn=lambda s, _: s != "/dev/vfio/16",
    )
    assert tpu_info.get_num_chips() == 8

    # Genuine >8 accessible TPU VF devices are not capped at 8.
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=[str(i) for i in range(12)],
        iommu_devices_fn=lambda _: ["0000:00:04.0"],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x1ae0\n" if s.endswith("/vendor") else "0x0076\n"
        ),
    )
    assert tpu_info.get_num_chips() == 12

    # Bare-metal SR-IOV hosts expose 3 0x0076 VFs per PF (virtfn0..1 compute,
    # virtfn2 ICI management) plus the PF (with virtfn0). Only virtfn0..1 count.
    sriov_bdfs = {
        0: "0000:10:00.0",  # PF (has virtfn0)
        1: "0000:10:00.1",  # virtfn0 -> compute
        2: "0000:10:00.2",  # virtfn1 -> compute
        3: "0000:10:00.3",  # virtfn2 -> ICI management
    }
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=[str(i) for i in range(4)],
        iommu_devices_fn=lambda s: [
            sriov_bdfs[int(s.split("/iommu_groups/")[1].split("/")[0])]
        ],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x1ae0\n" if s.endswith("/vendor") else "0x0076\n"
        ),
        exists_fn=lambda s: (
            s.endswith("0000:10:00.0/virtfn0")
            or (s.endswith("/physfn") and not s.endswith("0000:10:00.0/physfn"))
        ),
        readlink_fn=lambda s: (
            "../0000:10:00.1" if s.endswith("/virtfn0") else "../0000:10:00.2"
        ),
    )
    assert tpu_info.get_num_chips() == 2
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    assert tpu_info.get_tpu_type() == "tpu7x"

    # Fail closed if /sys/kernel/iommu_groups/<group>/devices is unreadable.
    def _raise_missing_sysfs(_: str):
        raise OSError("missing sysfs")

    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=["0", "1"],
        iommu_devices_fn=_raise_missing_sysfs,
        open_sysfs_fn=lambda s: io.StringIO("0x1ae0\n"),
    )
    assert tpu_info.get_num_chips() == 0


def test_reconcile_tpu_visible_device_paths_narrows_idempotently(monkeypatch):
    bdf_by_group = {
        "15": ["0000:89:00.1"],
        "3": ["0000:08:00.0"],
        "9": ["0000:89:00.0"],
        "7": ["0000:08:00.1"],
    }
    real_listdir = os.listdir
    monkeypatch.setattr(
        tpu_worker.os,
        "listdir",
        lambda path=".": (
            bdf_by_group[str(path).split("/sys/kernel/iommu_groups/")[1].split("/")[0]]
            if str(path).startswith("/sys/kernel/iommu_groups/")
            else real_listdir(path)
        ),
    )
    raw_paths = "/dev/vfio/15,/dev/vfio/3,/dev/vfio/9,/dev/vfio/7"
    expected_pci_order = ["/dev/vfio/3", "/dev/vfio/7", "/dev/vfio/9", "/dev/vfio/15"]

    monkeypatch.delenv("VLLM_TPU_DEVICE_ORDER", raising=False)
    for rank, expected in enumerate(expected_pci_order):
        monkeypatch.setenv("TPU_VISIBLE_DEVICE_PATHS", raw_paths)
        tpu_worker._reconcile_tpu_visible_device_paths(rank, local_world_size=4)
        assert tpu_worker.os.environ["TPU_VISIBLE_DEVICE_PATHS"] == expected
        # Calling a second time in the same worker is idempotent and keeps the
        # narrowed path rather than treating it as < local_world_size.
        tpu_worker._reconcile_tpu_visible_device_paths(rank, local_world_size=4)
        assert tpu_worker.os.environ["TPU_VISIBLE_DEVICE_PATHS"] == expected

    monkeypatch.setenv("VLLM_TPU_DEVICE_ORDER", "0,2,1,3")
    monkeypatch.setenv("TPU_VISIBLE_DEVICE_PATHS", raw_paths)
    tpu_worker._reconcile_tpu_visible_device_paths(1, local_world_size=4)
    assert tpu_worker.os.environ["TPU_VISIBLE_DEVICE_PATHS"] == "/dev/vfio/9"

    # Exclude bare-metal SR-IOV ICI management VFs (virtfn2) from device paths.
    monkeypatch.delenv("VLLM_TPU_DEVICE_ORDER", raising=False)
    sriov_bdfs = {
        "10": "0000:10:00.1",  # virtfn0 -> compute
        "11": "0000:10:00.2",  # virtfn1 -> compute
        "12": "0000:10:00.3",  # virtfn2 -> ICI management (excluded)
        "20": "0000:20:00.1",  # virtfn0 -> compute
    }
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=list(sriov_bdfs),
        iommu_devices_fn=lambda s: [
            sriov_bdfs[s.split("/iommu_groups/")[1].split("/")[0]]
        ],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x1ae0\n" if s.endswith("/vendor") else "0x0076\n"
        ),
        exists_fn=lambda s: s.endswith("/physfn"),
        readlink_fn=lambda s: (
            "../0000:10:00.1"
            if "0000:10:" in s and s.endswith("/virtfn0")
            else (
                "../0000:10:00.2"
                if "0000:10:" in s and s.endswith("/virtfn1")
                else "../0000:20:00.1"
            )
        ),
    )
    monkeypatch.setenv(
        "TPU_VISIBLE_DEVICE_PATHS",
        "/dev/vfio/10,/dev/vfio/11,/dev/vfio/12,/dev/vfio/20",
    )
    tpu_worker._reconcile_tpu_visible_device_paths(2, local_world_size=3)
    assert tpu_worker.os.environ["TPU_VISIBLE_DEVICE_PATHS"] == "/dev/vfio/20"


def test_reconcile_tpu_visible_device_paths_clears_insufficient_and_keeps_single(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_TPU_DEVICE_ORDER", raising=False)
    monkeypatch.setenv(
        "TPU_VISIBLE_DEVICE_PATHS", "/dev/vfio/0,/dev/vfio/1,/dev/vfio/2,/dev/vfio/3"
    )
    with mock.patch.object(tpu_worker.logger, "warning_once") as warn_mock:
        tpu_worker._reconcile_tpu_visible_device_paths(0, local_world_size=8)
        warn_mock.assert_called_once()
    assert "TPU_VISIBLE_DEVICE_PATHS" not in tpu_worker.os.environ

    monkeypatch.setenv("TPU_VISIBLE_DEVICE_PATHS", "/dev/vfio/5")
    tpu_worker._reconcile_tpu_visible_device_paths(0, local_world_size=1)
    assert tpu_worker.os.environ["TPU_VISIBLE_DEVICE_PATHS"] == "/dev/vfio/5"


@pytest.mark.parametrize("bad_value", ["not-an-int", "-1"])
def test_lws_group_size_env_rejects_invalid_and_negative_values(monkeypatch, bad_value):
    monkeypatch.setenv("LWS_GROUP_SIZE", bad_value)
    with pytest.raises(ValueError, match="LWS_GROUP_SIZE"):
        _ = envs.LWS_GROUP_SIZE
