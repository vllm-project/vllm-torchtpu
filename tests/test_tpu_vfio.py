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
"""Unit tests for VFIO TPU detection, tpu_info helpers, and env_override fallback."""

import io
import logging
import os
import stat
import types

import jax.numpy as jnp
import pytest

from vllm_torchtpu import env_override, tpu_info, tpu_vfio
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs as batched_rpa_configs,
)
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    wrapper as rpa_batched_wrapper,
)


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

    monkeypatch.setattr(tpu_vfio.os, "stat", _stat)
    monkeypatch.setattr(tpu_vfio.os, "access", _access)
    monkeypatch.setattr(tpu_vfio.os, "listdir", _listdir)
    monkeypatch.setattr(tpu_vfio.os.path, "exists", _exists)
    monkeypatch.setattr(tpu_vfio.os, "readlink", _readlink)
    monkeypatch.setattr("builtins.open", _open)


def test_get_num_chips_ignores_dev_accel_directory(monkeypatch):
    real_isdir = os.path.isdir
    monkeypatch.setattr(
        tpu_info.glob, "glob", lambda pattern: ["/dev/accel", "/dev/accel0"]
    )
    monkeypatch.setattr(
        tpu_info.os.path,
        "isdir",
        lambda path: (
            True
            if path == "/dev/accel"
            else (False if str(path).startswith("/dev/accel") else real_isdir(path))
        ),
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


def test_get_num_chips_filters_mixed_vfio_devices_without_cap(monkeypatch):
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: [])
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


def test_get_num_chips_excludes_sriov_pf_and_mgmt_vf(monkeypatch):
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: [])
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

    # Only bare-metal-verified 0x0076 carries a per-PF cap; GCE/GKE guest VMs
    # (e.g. v6e / 0x006f without physfn in sysfs) count all compute VFs.
    assert tpu_vfio.TPU_VFIO_MAX_APP_VFS_PER_PF == {"0x0076": 2}
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=["0", "1", "2", "3"],
        iommu_devices_fn=lambda _: ["0000:00:04.0"],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x1ae0\n" if s.endswith("/vendor") else "0x006f\n"
        ),
    )
    assert tpu_info.get_num_chips() == 4
    assert tpu_info.get_tpu_type() == "v6e"


def test_get_num_chips_unreadable_sysfs_fallback_vs_inaccessible_vfio(monkeypatch):
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: [])

    def _raise_missing_sysfs(_: str):
        raise OSError("missing sysfs")

    # Accessible /dev/vfio char devices with hidden/unreadable sysfs fall back
    # to the legacy count of accessible numeric entries.
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=["0", "1", "vfio"],
        iommu_devices_fn=_raise_missing_sysfs,
        open_sysfs_fn=lambda s: io.StringIO("0x1ae0\n"),
    )
    assert tpu_info.get_num_chips() == 2

    # When the process cannot read/write ANY /dev/vfio/<g> group, return 0
    # (distinct from unreadable sysfs) while detect_vfio_tpu_type() still reads
    # sysfs without requiring /dev/vfio R/W access.
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=["0", "1"],
        iommu_devices_fn=lambda _: ["0000:00:04.0"],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x1ae0\n" if s.endswith("/vendor") else "0x0076\n"
        ),
        access_fn=lambda _s, _m: False,
    )
    assert tpu_info.get_num_chips() == 0
    assert tpu_vfio.detect_vfio_tpu_type() == "tpu7x"


def test_get_num_chips_unknown_google_pci_id_warns_and_falls_back(monkeypatch, caplog):
    monkeypatch.setattr(tpu_info.glob, "glob", lambda pattern: [])
    # Unrecognized Google PCI IDs (including 0x0056 / plc, which has no public
    # name in JAX) are not misclassified and fall back to legacy VFIO count.
    for device_id in ("0x0099", "0x0056"):
        caplog.clear()
        _patch_vfio_fs(
            monkeypatch,
            vfio_entries=["0", "1", "2", "3"],
            iommu_devices_fn=lambda _: ["0000:00:04.0"],
            open_sysfs_fn=lambda s, dev_id=device_id: io.StringIO(
                "0x1ae0\n" if s.endswith("/vendor") else f"{dev_id}\n"
            ),
        )
        assert tpu_vfio.detect_vfio_tpu_type() is None
        with caplog.at_level(logging.WARNING):
            assert tpu_info.get_num_chips() == 4
        assert device_id in caplog.text

    # Readable non-Google sysfs (e.g. NVIDIA GPU) still returns 0 with no fallback.
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=["0", "1"],
        iommu_devices_fn=lambda _: ["0000:00:04.0"],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x10de\n" if s.endswith("/vendor") else "0x2330\n"
        ),
    )
    assert tpu_info.get_num_chips() == 0


def test_env_override_sc_offload_resolves_vfio_tpu_type(monkeypatch):
    monkeypatch.delenv("SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES", raising=False)
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    monkeypatch.setattr(
        env_override,
        "_fetch_tpu_type_from_metadata",
        lambda: pytest.fail(
            "GCE metadata should not be queried when VFIO detects type"
        ),
    )
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=["0"],
        iommu_devices_fn=lambda _: ["0000:00:04.0"],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x1ae0\n" if s.endswith("/vendor") else "0x0076\n"
        ),
        access_fn=lambda _s, _m: False,
    )
    assert (
        env_override._resolve_sc_offload_threshold_bytes()
        == env_override._TPU_VMEM_BYTES_PER_TENSOR_CORE["tpu7x"]
    )


@pytest.mark.parametrize(
    ("accelerator_type", "expected_cores"),
    [
        ("v5e-8", 1),
        ("v5litepod-8", 1),
        ("v6e-8", 1),
        ("tpu8t", 1),
        ("v5p-8", 2),
        ("tpu7x", 2),
    ],
)
def test_get_num_cores_per_chip_matches_jax_chip_version(
    monkeypatch, accelerator_type: str, expected_cores: int
):
    tpu_info.get_chip_version.cache_clear()
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", accelerator_type)
    try:
        assert tpu_info.get_num_cores_per_chip() == expected_cores
    finally:
        tpu_info.get_chip_version.cache_clear()


def test_get_node_name_prefers_env_then_metadata(monkeypatch):
    monkeypatch.setenv("TPU_NAME", "my-tpu-node")
    monkeypatch.setattr(
        tpu_info,
        "get_tpu_metadata",
        lambda key="": pytest.fail("GCE metadata should not be queried"),
    )
    assert tpu_info.get_node_name() == "my-tpu-node"

    monkeypatch.delenv("TPU_NAME", raising=False)
    monkeypatch.setattr(tpu_info, "get_tpu_metadata", lambda key="": "gce-node-42")
    assert tpu_info.get_node_name() == "gce-node-42"


def test_seq_along_lane_kv_cache_shape_for_tpu8t_vfio(monkeypatch):
    tpu_info.get_chip_version.cache_clear()
    monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
    _patch_vfio_fs(
        monkeypatch,
        vfio_entries=["0"],
        iommu_devices_fn=lambda _: ["0000:00:04.0"],
        open_sysfs_fn=lambda s: io.StringIO(
            "0x1ae0\n" if s.endswith("/vendor") else "0x007c\n"
        ),
        access_fn=lambda _s, _m: False,
    )
    try:
        chip_version = tpu_info.get_chip_version()
        assert chip_version.value == "8t"

        # tpu8t has 16 sublanes: head_dim=80 aligns to 96 in bf16 (packing=2 -> 32),
        # and head_dim=96 aligns to 128 in fp8 (packing=4 -> 64).
        bf16_shape = rpa_batched_wrapper.get_kv_cache_shape(
            total_num_pages=4,
            page_size=16,
            actual_num_kv_heads=2,
            actual_head_dim=80,
            kv_dtype=jnp.bfloat16,
            kv_layout=batched_rpa_configs.KVLayout.SEQ_ALONG_LANE,
            chip_version=chip_version,
        )
        assert bf16_shape == (4, 4, 48, 2, 16)

        fp8_shape = rpa_batched_wrapper.get_kv_cache_shape(
            total_num_pages=4,
            page_size=16,
            actual_num_kv_heads=2,
            actual_head_dim=96,
            kv_dtype=jnp.float8_e4m3fn,
            kv_layout=batched_rpa_configs.KVLayout.SEQ_ALONG_LANE,
            chip_version=chip_version,
        )
        assert fp8_shape == (4, 4, 32, 4, 16)
    finally:
        tpu_info.get_chip_version.cache_clear()
