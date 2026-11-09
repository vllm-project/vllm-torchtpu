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
import functools
import glob
import hashlib
import importlib.util
import os
from importlib import metadata as importlib_metadata

import requests
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

GCE_TPU_ACCELERATOR_ENDPOINT = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"
)
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}


def get_tpu_metadata(key: str = "") -> str:
    try:
        accelerator_type_request = requests.get(
            os.path.join(GCE_TPU_ACCELERATOR_ENDPOINT, key),
            headers=GCE_TPU_HEADERS,
        )
        if (
            accelerator_type_request.status_code == 200
            and accelerator_type_request.text
        ):
            return accelerator_type_request.text
        else:
            logger.error(
                "Unable to poll TPU GCE Metadata. Got status code: %s and content: %s",
                accelerator_type_request.status_code,
                accelerator_type_request.text,
            )
    except requests.RequestException as e:
        logger.error("Unable to poll the TPU GCE Metadata: %s", e)
    return None


def get_tpu_type() -> str:
    tpu_type = envs.TPU_ACCELERATOR_TYPE
    if tpu_type is None:
        tpu_type = get_tpu_metadata(key="accelerator-type")
    return tpu_type


@functools.cache
def get_chip_version(accelerator_type: str | None = None) -> pltpu.ChipVersion:
    """Resolves the TPU chip version without initializing a TPU client.

    `pltpu.get_tpu_info()` would initialize PJRT and claim the chips, so
    host-side callers (the scheduler, KV offloading connectors) resolve the
    chip from `TPU_ACCELERATOR_TYPE` instead. Falls back to TPU7x with a
    warning when the accelerator type is unrecognized.
    """
    accelerator_type = accelerator_type or get_tpu_type() or ""
    # ChipVersion's values are bare chip names ("7x", "v6e"); accelerator types
    # add a chip-count suffix and sometimes a "tpu" prefix ("tpu7x", "v6e-8").
    name = accelerator_type.strip().lower().split("-")[0].removeprefix("tpu")
    # v5litepod and v7x are the only two that still differ after stripping.
    name = {"v5litepod": "v5e", "v7x": "7x"}.get(name, name)
    try:
        return pltpu.ChipVersion(name)
    except ValueError:
        logger.warning(
            "Unrecognized TPU accelerator type %r; defaulting to TPU7x "
            "static geometry.",
            accelerator_type,
        )
        return pltpu.ChipVersion.TPU_7X


def get_libtpu_version() -> str | None:
    """Version of the pip ``libtpu`` package, or None if it is not installed.

    Some environments load ``libtpu.so`` directly through ``TPU_LIBRARY_PATH``
    (for example container images that ship a prebuilt library) and have no
    ``libtpu`` package metadata.
    """
    try:
        return importlib_metadata.version("libtpu")
    except importlib_metadata.PackageNotFoundError:
        return None


def _pip_libtpu_library_path() -> str | None:
    """``libtpu.so`` shipped by the pip ``libtpu`` package, if installed.

    Located with ``find_spec`` rather than an import: importing ``libtpu`` sets
    ``TPU_LIBRARY_PATH`` to this path when it is unset.
    """
    spec = importlib.util.find_spec("libtpu")
    if spec is None or not spec.origin:
        return None
    return os.path.join(os.path.dirname(spec.origin), "libtpu.so")


def get_tpu_library_path() -> str | None:
    """Custom ``libtpu.so`` from ``TPU_LIBRARY_PATH``, if one is in use.

    JAX loads ``TPU_LIBRARY_PATH`` in preference to the pip ``libtpu`` package
    (``jax/_src/cloud_tpu_init.py``). Like JAX, a path that is just the pip
    package's own library (which ``import libtpu`` exports) is not custom.
    Returns None when the variable is unset, names a missing file, or names the
    pip package's library.
    """
    path = envs.TPU_LIBRARY_PATH or ""
    if not path or not os.path.isfile(path):
        return None
    package_library = _pip_libtpu_library_path()
    if (
        package_library is not None
        and os.path.isfile(package_library)
        and os.path.samefile(path, package_library)
    ):
        return None
    return path


@functools.cache
def _libtpu_file_fingerprint(path: str, size: int, mtime_ns: int) -> str:
    # Hash the size and the first/last MiB: cheap for a ~700 MB library, and
    # it distinguishes builds even when images reset mtimes. This assumes the
    # ELF header and the .note.gnu.build-id note, which differ between builds,
    # lie within the first MiB of libtpu.so; a build that differed only in the
    # middle of the file would not change the fingerprint.
    digest = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        digest.update(f.read(1 << 20))
        if size > (1 << 20):
            f.seek(max(size - (1 << 20), 1 << 20))
            digest.update(f.read(1 << 20))
    return digest.hexdigest()[:16]


def get_libtpu_cache_key() -> str:
    """Identify the libtpu build in use, for compilation and layout cache keys.

    Uses a fingerprint of a custom library at ``TPU_LIBRARY_PATH`` (JAX loads
    it instead of the pip package; see ``get_tpu_library_path``), otherwise
    the ``libtpu`` package version, otherwise ``"unpackaged-unknown"``.
    """
    path = get_tpu_library_path()
    if path is not None:
        st = os.stat(path)
        return "unpackaged-" + _libtpu_file_fingerprint(
            path, st.st_size, st.st_mtime_ns
        )
    libtpu_version = get_libtpu_version()
    if libtpu_version is not None:
        return libtpu_version
    return "unpackaged-unknown"


def get_node_name() -> str:
    tpu_name = envs.TPU_NAME
    if not tpu_name:
        tpu_name = get_tpu_metadata(key="instance-id")
    return tpu_name


def get_node_worker_id() -> int:
    """For multi-host TPU VM, this returns the worker id for the current node."""
    worker_id = envs.TPU_WORKER_ID
    if worker_id is None:
        worker_id = get_tpu_metadata(key="agent-worker-number")
    if worker_id is None:
        return 0
    return int(worker_id)


def get_num_cores_per_chip() -> int:
    tpu_type = get_tpu_type()
    if tpu_type.startswith(("v5litepod", "v6e")):
        return 1
    return 2


def get_num_chips() -> int:
    accel_files = glob.glob("/dev/accel*")
    if accel_files:
        return len(accel_files)
    try:
        vfio_entries = os.listdir("/dev/vfio")
        numeric_entries = [int(entry) for entry in vfio_entries if entry.isdigit()]
        return len(numeric_entries)
    except FileNotFoundError as e:
        logger.warning("Failed to detect number of TPUs: %s", e)
        return 0
