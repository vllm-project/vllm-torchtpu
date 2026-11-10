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
"""TPU support for vLLM's single-host multi-port DP launch.

``vllm serve --data-parallel-multi-port-external-lb`` starts a DPSupervisor
that spawns one API server per local DP rank (ports ``--port + r``). Each
child is an external-LB rank: ``data_parallel_size = N``,
``data_parallel_rank = r``, ``data_parallel_size_local = 1`` and
``--device-ids [r*tp*pp, (r+1)*tp*pp)``. Its engine core is started with
``local_dp_rank = 0``.

On TPU that shape is ambiguous. ``dp_size_local < dp_size`` is how this
package recognises DP spread across hosts (``prepare_mp_multihost_dp_env``),
and every engine reporting local DP rank 0 pins every engine's workers to the
first chips. When the N ranks share one host and expert parallelism couples
them into one N*TP torch_tpu world, they also need to agree on the slice
bootstrap: one DP master port and one slicebuilder address list.

vLLM builds no VllmConfig in the supervisor process, so no platform hook runs
there, and a child's arguments do not say that its siblings share its host.
Each child therefore derives the shared values itself, deterministically from
arguments that are identical in every child, once the server opts in with
``--additional-config '{"tpu_multiport_dp": {"enabled": true}}'``: the
``--data-parallel-size`` ranks of the multi-port server share this host. The
supervisor copies its arguments, this option included, into every child. Off
(the default) changes nothing.

* ``prepare_multiport_dp_env`` (from ``TpuPlatform.check_and_update_config``,
  in each child API server): the vLLM DP group ports, the torch_tpu DP master
  port and the slicebuilder list for the whole N*TP slice; the cross-host
  rendezvous is not run.
* ``resolve_device_ids`` (wraps vLLM's ``EngineArgs._resolve_device_ids``,
  installed by the package's general plugin, which vLLM loads in
  ``EngineArgs.__post_init__``): the supervisor's ``--device-ids``
  count TPU cores, not the chips ``TPU_VISIBLE_CHIPS`` lists. Chip binding on
  this path is owned by the TPU worker binding, so vLLM's device-control
  variable is not consulted, which is what expert parallelism already does in
  ``check_and_update_config``.
* ``engine_core_kwargs`` (from ``_run_engine_core_with_tpu_patches``, in each
  engine core): ``local_dp_rank = dp_rank % N`` and
  ``data_parallel_size_local = N``, so workers bind chips by their real slot
  on the host.
"""

import dataclasses
import functools
import json
import os
from collections.abc import Mapping
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# The --additional-config section. Launch topology only: each engine compiles
# the same programs as under internal-LB DP, so runner/utils.py keeps it out of
# the compile-cache key.
SECTION = "tpu_multiport_dp"

# vLLM's base Platform value; TpuPlatform.check_and_update_config switches to
# it for expert parallelism.
DEVICE_CONTROL_PLACEHOLDER = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"


@dataclasses.dataclass(frozen=True)
class MultiportDpOptions:
    """``additional_config["tpu_multiport_dp"]``."""

    # The --data-parallel-size ranks of a --data-parallel-multi-port-external-lb
    # server share this host.
    enabled: bool = False


def options(additional_config: Any) -> MultiportDpOptions:
    """The parsed section; unknown keys and non-bool values raise ValueError."""
    if isinstance(additional_config, str):
        additional_config = json.loads(additional_config) if additional_config else {}
    raw = (
        additional_config.get(SECTION)
        if isinstance(additional_config, Mapping)
        else None
    )
    if raw is None:
        return MultiportDpOptions()
    if not isinstance(raw, Mapping):
        raise ValueError(f"additional_config[{SECTION!r}] must be a JSON object")
    known = {f.name for f in dataclasses.fields(MultiportDpOptions)}
    unknown = sorted(str(k) for k in raw if k not in known)
    if unknown:
        raise ValueError(
            f"Unknown key(s) {unknown} in additional_config[{SECTION!r}]; "
            f"known keys: {sorted(known)}"
        )
    if "enabled" in raw and type(raw["enabled"]) is not bool:
        raise ValueError(f"additional_config[{SECTION!r}]['enabled'] must be a bool")
    return MultiportDpOptions(**dict(raw))


def local_dp_size(vllm_config) -> int:
    """The DP ranks that share this host under the option; 0 means off."""
    if not options(getattr(vllm_config, "additional_config", None)).enabled:
        return 0
    return int(vllm_config.parallel_config.data_parallel_size)


def _validate(parallel_config) -> None:
    if not parallel_config.data_parallel_external_lb:
        raise ValueError(
            f"additional_config[{SECTION!r}] is enabled, but this server is not "
            "a rank of --data-parallel-multi-port-external-lb "
            "(data_parallel_external_lb is False). Turn it off for internal or "
            "hybrid DP load balancing."
        )
    if parallel_config.nnodes > 1:
        raise NotImplementedError(
            f"additional_config[{SECTION!r}] does not support --nnodes > 1: "
            "multi-port DP on TPU supports only all DP ranks on one host."
        )


def prepare_multiport_dp_env(vllm_config) -> bool:
    """Set the shared slice bootstrap for one multi-port DP rank.

    Returns False (and does nothing) unless the option is enabled. Must run
    before ``check_and_update_config`` defaults TORCH_TPU_DP_MASTER_PORT, and
    in place of ``prepare_mp_multihost_dp_env``. Idempotent per process.
    """
    local_size = local_dp_size(vllm_config)
    if local_size <= 0:
        return False
    parallel_config = vllm_config.parallel_config
    _validate(parallel_config)

    # Every child picks its own random DP-init ports in ParallelConfig; the
    # engines' DP group would never form. Same fix and same ports as
    # prepare_mp_multihost_dp_env.
    rpc_port = parallel_config.data_parallel_rpc_port
    parallel_config._data_parallel_master_port_list = [
        rpc_port + 100 + i for i in range(5)
    ]

    # torch_tpu's slice rendezvous (tpu_worker.py exports it as MASTER_PORT).
    # The port the multi-host DP TCPStore rendezvous would use: free here,
    # because this path never runs that rendezvous.
    from vllm_torchtpu.distributed.tpu_mp_multihost import _rendezvous_port

    os.environ.setdefault("TORCH_TPU_DP_MASTER_PORT", str(_rendezvous_port(rpc_port)))

    # One address per device of the N*TP slice, the same in every child.
    # Same base-port convention as the multi-host bootstrap.
    world_size = parallel_config.world_size_across_dp
    current = os.environ.get("TORCH_TPU_SLICEBUILDER_ADDRESSES")
    if not current or len(current.split(",")) != world_size:
        base_port = envs.TORCH_TPU_BASE_PORT
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(
            f"localhost:{base_port + i}" for i in range(world_size)
        )

    logger.info(
        "TPU multi-port DP: rank %d of %d on this host; "
        "TORCH_TPU_DP_MASTER_PORT=%s TORCH_TPU_SLICEBUILDER_ADDRESSES=%s "
        "dp_master_ports=%s",
        parallel_config.data_parallel_rank,
        local_size,
        os.environ["TORCH_TPU_DP_MASTER_PORT"],
        os.environ["TORCH_TPU_SLICEBUILDER_ADDRESSES"],
        parallel_config._data_parallel_master_port_list,
    )
    return True


def engine_core_kwargs(kwargs: dict) -> dict:
    """Give a multi-port engine core its real local DP rank and local size.

    ``kwargs`` are EngineCoreProc.run_engine_core's keyword arguments. Returns
    them unchanged unless the option is enabled and the engine is an
    external-LB DP rank.
    """
    vllm_config = kwargs.get("vllm_config")
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None or not parallel_config.data_parallel_external_lb:
        return kwargs
    local_size = local_dp_size(vllm_config)
    if local_size <= 1:
        return kwargs
    dp_rank = kwargs.get("dp_rank", 0)
    kwargs = dict(kwargs)
    kwargs["local_dp_rank"] = dp_rank % local_size
    parallel_config.data_parallel_size_local = local_size
    logger.info(
        "TPU multi-port DP: engine core dp_rank=%d -> local_dp_rank=%d, "
        "data_parallel_size_local=%d",
        dp_rank,
        kwargs["local_dp_rank"],
        local_size,
    )
    return kwargs


def resolve_device_ids(original, engine_args) -> Any:
    """``EngineArgs._resolve_device_ids`` for a multi-port child: with the
    option on, the platform stops consulting TPU_VISIBLE_CHIPS (as expert
    parallelism does later in ``check_and_update_config``), so the
    supervisor's core-indexed ``--device-ids`` pass vLLM's range check."""
    if options(getattr(engine_args, "additional_config", None)).enabled:
        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

        TpuPlatform.device_control_env_var = DEVICE_CONTROL_PLACEHOLDER
    return original(engine_args)


def install_device_ids_patch() -> None:
    """Wrap vLLM's EngineArgs._resolve_device_ids once per process."""
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs._resolve_device_ids, "_tpu_multiport_dp_patch", False):
        return
    original = EngineArgs._resolve_device_ids

    @functools.wraps(original)
    def _resolve_device_ids(self):
        return resolve_device_ids(original, self)

    _resolve_device_ids._tpu_multiport_dp_patch = True  # type: ignore[attr-defined]
    EngineArgs._resolve_device_ids = _resolve_device_ids
