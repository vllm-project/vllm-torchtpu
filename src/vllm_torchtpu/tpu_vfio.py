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
"""Pure-Python VFIO / sysfs TPU device and generation detection.

Kept free of ``torch``, ``jax``, and ``requests`` imports so both
``env_override.py`` (before ``libtpu`` loads) and ``tpu_info.py`` can share the
same hardware detection without claiming a TPU device.
"""

import os
import stat
from enum import StrEnum

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

TPU_PCI_VENDOR_ID = "0x1ae0"

# Google TPU PCI device IDs mapped to canonical accelerator prefixes (matching
# public generations in jax/_src/hardware_utils.py; omits 0x0056 / plc which
# has no public name and falls through to the unknown-Google legacy fallback).
TPU_PCI_DEVICE_ID_TO_TYPE: dict[str, str] = {
    "0x0027": "v3",
    "0x005e": "v4",
    "0x0062": "v5p",
    "0x0063": "v5litepod",
    "0x006f": "v6e",
    "0x0076": "tpu7x",
    "0x007c": "tpu8t",
    "0x0083": "tpu8i",
}

# Bare-metal-verified SR-IOV application VF counts per physical function
# (virtfn0 .. virtfn{k-1}) when host sysfs exposes ``physfn`` alongside ICI
# management VFs. Only 0x0076 (TPU7x) is bare-metal verified here; GCE/GKE
# guest VMs pass through VFs without ``physfn`` in guest sysfs, so
# ``_is_application_pci_function`` accepts them without a cap.
TPU_VFIO_MAX_APP_VFS_PER_PF: dict[str, int] = {
    "0x0076": 2,
}


class _VfioGroupKind(StrEnum):
    NOT_ACCESSIBLE = "not_accessible"
    SYSFS_UNREADABLE = "sysfs_unreadable"
    TPU_COMPUTE = "tpu_compute"
    TPU_EXCLUDED = "tpu_excluded"
    UNKNOWN_GOOGLE = "unknown_google"
    NON_GOOGLE = "non_google"


def _is_application_pci_function(dev_dir: str, bdf: str, device_id: str) -> bool:
    """Exclude SR-IOV physical functions and verified bare-metal mgmt VFs."""
    if os.path.exists(os.path.join(dev_dir, "virtfn0")):
        return False
    physfn_dir = os.path.join(dev_dir, "physfn")
    max_app_vfs = TPU_VFIO_MAX_APP_VFS_PER_PF.get(device_id)
    if max_app_vfs is None or not os.path.exists(physfn_dir):
        return True
    for vf_idx in range(max_app_vfs):
        try:
            target = os.readlink(os.path.join(physfn_dir, f"virtfn{vf_idx}"))
        except OSError:
            continue
        if os.path.basename(target) == bdf:
            return True
    return False


def _classify_sysfs_iommu_group(group: int) -> tuple[_VfioGroupKind, str | None]:
    """Inspect ``/sys/kernel/iommu_groups/<group>/devices`` without opening VFIO."""
    sysfs_root = f"/sys/kernel/iommu_groups/{group}/devices"
    try:
        devices = os.listdir(sysfs_root)
    except OSError:
        return _VfioGroupKind.SYSFS_UNREADABLE, None

    saw_readable = False
    saw_excluded_tpu = False
    unknown_google_id: str | None = None
    for bdf in devices:
        dev_dir = os.path.join(sysfs_root, bdf)
        try:
            with open(os.path.join(dev_dir, "vendor"), encoding="ascii") as vf:
                vendor = vf.read().strip().lower()
            saw_readable = True
            if vendor != TPU_PCI_VENDOR_ID:
                continue
            with open(os.path.join(dev_dir, "device"), encoding="ascii") as df:
                device_id = df.read().strip().lower()
        except (OSError, UnicodeDecodeError):
            continue

        if device_id in TPU_PCI_DEVICE_ID_TO_TYPE:
            if _is_application_pci_function(dev_dir, bdf, device_id):
                return _VfioGroupKind.TPU_COMPUTE, device_id
            saw_excluded_tpu = True
        elif unknown_google_id is None:
            unknown_google_id = device_id

    if saw_excluded_tpu:
        return _VfioGroupKind.TPU_EXCLUDED, None
    if unknown_google_id is not None:
        return _VfioGroupKind.UNKNOWN_GOOGLE, unknown_google_id
    if saw_readable:
        return _VfioGroupKind.NON_GOOGLE, None
    return _VfioGroupKind.SYSFS_UNREADABLE, None


def _is_accessible_vfio_char_device(group: int) -> bool:
    vfio_path = f"/dev/vfio/{group}"
    try:
        if not stat.S_ISCHR(os.stat(vfio_path).st_mode):
            return False
    except OSError:
        return False
    return os.access(vfio_path, os.R_OK | os.W_OK)


def _classify_vfio_group(group: int) -> tuple[_VfioGroupKind, str | None]:
    if not _is_accessible_vfio_char_device(group):
        return _VfioGroupKind.NOT_ACCESSIBLE, None
    return _classify_sysfs_iommu_group(group)


def detect_vfio_tpu_type() -> str | None:
    """Infer TPU accelerator prefix from IOMMU sysfs without VFIO R/W access."""
    try:
        vfio_entries = os.listdir("/dev/vfio")
        groups = sorted(int(entry) for entry in vfio_entries if entry.isdigit())
    except OSError:
        try:
            iommu_entries = os.listdir("/sys/kernel/iommu_groups")
            groups = sorted(int(entry) for entry in iommu_entries if entry.isdigit())
        except OSError:
            return None

    for group in groups:
        kind, device_id = _classify_sysfs_iommu_group(group)
        if kind is _VfioGroupKind.TPU_COMPUTE and device_id is not None:
            return TPU_PCI_DEVICE_ID_TO_TYPE.get(device_id)
    return None


def count_vfio_tpu_chips() -> int:
    """Count accessible local TPU compute VFIO groups.

    Returns 0 when no numeric ``/dev/vfio/<group>`` is readable/writable by the
    current process. Falls back to the accessible numeric VFIO count when IOMMU
    sysfs metadata is hidden/unreadable or exposes only unrecognized Google PCI
    device IDs.
    """
    try:
        vfio_entries = os.listdir("/dev/vfio")
    except FileNotFoundError as e:
        logger.warning("Failed to detect number of TPUs: %s", e)
        return 0

    numeric_entries = [int(entry) for entry in vfio_entries if entry.isdigit()]
    if not numeric_entries:
        return 0

    accessible_count = 0
    sysfs_readable_count = 0
    tpu_compute_count = 0
    tpu_excluded_count = 0
    unknown_google_ids: set[str] = set()

    for group in numeric_entries:
        kind, device_id = _classify_vfio_group(group)
        if kind is _VfioGroupKind.NOT_ACCESSIBLE:
            continue
        accessible_count += 1
        if kind is _VfioGroupKind.SYSFS_UNREADABLE:
            continue
        sysfs_readable_count += 1
        if kind is _VfioGroupKind.TPU_COMPUTE:
            tpu_compute_count += 1
        elif kind is _VfioGroupKind.TPU_EXCLUDED:
            tpu_excluded_count += 1
        elif kind is _VfioGroupKind.UNKNOWN_GOOGLE and device_id is not None:
            unknown_google_ids.add(device_id)

    if accessible_count == 0:
        return 0
    if sysfs_readable_count == 0:
        return accessible_count
    if tpu_compute_count > 0:
        return tpu_compute_count
    if unknown_google_ids and tpu_excluded_count == 0:
        logger.warning_once(
            "tpu_vfio | Found accessible Google PCI device(s) %s with no "
            "recognized TPU compute device ID; falling back to legacy VFIO "
            "count %d.",
            ", ".join(sorted(unknown_google_ids)),
            accessible_count,
        )
        return accessible_count
    return 0
