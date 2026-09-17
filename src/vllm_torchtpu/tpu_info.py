import functools
import glob
import os

import requests
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

GCE_TPU_ACCELERATOR_ENDPOINT = (
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/")
GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}


def get_tpu_metadata(key: str = "") -> str | None:
    try:
        clean_key = key.lstrip("/")
        url = os.path.join(GCE_TPU_ACCELERATOR_ENDPOINT, clean_key)
        accelerator_type_request = requests.get(
            url,
            headers=GCE_TPU_HEADERS,
            timeout=5,
        )
        if accelerator_type_request.status_code == 200:
            text = accelerator_type_request.text.strip()
            return text if text else None
        elif accelerator_type_request.status_code == 404:
            logger.debug(
                "TPU GCE Metadata attribute %r not found (status code 404).",
                key)
        else:
            logger.error(
                "Unable to poll TPU GCE Metadata for %r. Got "
                "status code: %s and content: %s",
                key,
                accelerator_type_request.status_code,
                accelerator_type_request.text,
            )
    except requests.RequestException as e:
        logger.error("Unable to poll the TPU GCE Metadata: %s", e)
    return None


def get_tpu_type() -> str | None:
    tpu_type = envs.TPU_ACCELERATOR_TYPE
    if tpu_type is None:
        tpu_type = get_tpu_metadata(key="accelerator-type")
    if tpu_type:
        tpu_type = tpu_type.strip()
    return tpu_type or None


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
            "static geometry.", accelerator_type)
        return pltpu.ChipVersion.TPU_7X


def get_node_name() -> str | None:
    tpu_name = envs.TPU_NAME
    if not tpu_name:
        tpu_name = get_tpu_metadata(key="instance-id")
    if tpu_name:
        tpu_name = tpu_name.strip()
    return tpu_name or None


def get_node_worker_id() -> int:
    """For multi-host TPU VM, this returns the worker id for the current node."""
    worker_id = envs.TPU_WORKER_ID
    if worker_id is None:
        worker_id = get_tpu_metadata(key="agent-worker-number")
    if worker_id is None:
        return 0
    try:
        return int(worker_id)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid TPU worker ID %r; defaulting to worker ID 0.",
            worker_id,
        )
        return 0


def get_num_cores_per_chip() -> int:
    tpu_type = get_tpu_type()
    if tpu_type and tpu_type.startswith(("v5litepod", "v6e")):
        return 1
    return 2


def get_num_chips() -> int:
    accel_files = glob.glob("/dev/accel*")
    if accel_files:
        return len(accel_files)
    try:
        vfio_entries = os.listdir("/dev/vfio")
        numeric_entries = [
            int(entry) for entry in vfio_entries if entry.isdigit()
        ]
        return len(numeric_entries)
    except OSError as e:
        logger.warning("Failed to detect number of TPUs: %s", e)
        return 0
