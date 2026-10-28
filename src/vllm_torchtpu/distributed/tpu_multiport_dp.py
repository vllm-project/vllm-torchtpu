# SPDX-License-Identifier: Apache-2.0
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
there. Each child therefore derives the shared values itself, deterministically
from arguments that are identical in every child. This is opt-in:
``TPU_MULTIPORT_DP_LOCAL_SIZE=<N>`` declares that the N ranks of a multi-port
server share this host. Unset (0) changes nothing.

* ``prepare_multiport_dp_env`` (from ``TpuPlatform.check_and_update_config``,
  in each child API server): the vLLM DP group ports, the torch_tpu DP master
  port and the slicebuilder list for the whole N*TP slice; the cross-host
  rendezvous is not run.
* ``device_control_env_var`` (from ``register_tpu_platform``, in every
  process): the supervisor's ``--device-ids`` count TPU cores, not the chips
  ``TPU_VISIBLE_CHIPS`` lists. Chip binding on this path is owned by the TPU
  worker binding, so vLLM's device-control variable is not consulted, which
  is what expert parallelism already does in ``check_and_update_config``.
* ``engine_core_kwargs`` (from ``_run_engine_core_with_tpu_patches``, in each
  engine core): ``local_dp_rank = dp_rank % N`` and
  ``data_parallel_size_local = N``, so workers bind chips by their real slot
  on the host.
"""

import os

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# vLLM's base Platform value; TpuPlatform.check_and_update_config switches to
# it for expert parallelism.
DEVICE_CONTROL_PLACEHOLDER = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"


def local_dp_size() -> int:
    """N from TPU_MULTIPORT_DP_LOCAL_SIZE; 0 means the feature is off."""
    return envs.TPU_MULTIPORT_DP_LOCAL_SIZE


def device_control_env_var(default: str) -> str:
    """The platform's device-control variable under the opt-in, else default."""
    return DEVICE_CONTROL_PLACEHOLDER if local_dp_size() > 0 else default


def _validate(parallel_config, local_size: int) -> None:
    dp_size = parallel_config.data_parallel_size
    if not parallel_config.data_parallel_external_lb:
        raise ValueError(
            f"TPU_MULTIPORT_DP_LOCAL_SIZE={local_size} is set, but this server "
            "is not a rank of --data-parallel-multi-port-external-lb "
            "(data_parallel_external_lb is False). Unset it for internal or "
            "hybrid DP load balancing."
        )
    if local_size != dp_size:
        raise NotImplementedError(
            f"TPU_MULTIPORT_DP_LOCAL_SIZE={local_size} must equal "
            f"--data-parallel-size ({dp_size}): multi-port DP on TPU supports "
            "only all DP ranks on one host."
        )
    if parallel_config.nnodes > 1:
        raise NotImplementedError(
            "TPU_MULTIPORT_DP_LOCAL_SIZE does not support --nnodes > 1."
        )


def prepare_multiport_dp_env(parallel_config) -> bool:
    """Set the shared slice bootstrap for one multi-port DP rank.

    Returns False (and does nothing) unless TPU_MULTIPORT_DP_LOCAL_SIZE is
    set. Must run before ``check_and_update_config`` defaults
    TORCH_TPU_DP_MASTER_PORT, and in place of ``prepare_mp_multihost_dp_env``.
    Idempotent per process.
    """
    local_size = local_dp_size()
    if local_size <= 0:
        return False
    _validate(parallel_config, local_size)

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
    them unchanged unless TPU_MULTIPORT_DP_LOCAL_SIZE is set and the engine is
    an external-LB DP rank.
    """
    local_size = local_dp_size()
    if local_size <= 1:
        return kwargs
    vllm_config = kwargs.get("vllm_config")
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if parallel_config is None or not parallel_config.data_parallel_external_lb:
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
