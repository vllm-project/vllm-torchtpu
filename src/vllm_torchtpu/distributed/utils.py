from vllm.utils.network_utils import get_ip

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# For multi-host usage only, to collect IP and port for all nodes.
_NODES_KV_IP_PORT = dict()


def set_node_kv_ip_port(ip_port: tuple[int, str, int]):
    global _NODES_KV_IP_PORT
    node_id, ip, port = ip_port
    _NODES_KV_IP_PORT[node_id] = (ip, port)


def get_kv_ips() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        # Node-sorted: the consumer indexes the advertised lists by node id.
        return [ip for _, (ip, _) in sorted(_NODES_KV_IP_PORT.items())]
    else:
        return get_host_ip()


def get_kv_ports() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        return [port for _, (_, port) in sorted(_NODES_KV_IP_PORT.items())]
    else:
        return get_kv_transfer_port()


def get_host_ip() -> str:
    """Use `VLLM_HOST_IP` if set, otherwise use default network interface IP."""
    return get_ip()


def get_kv_transfer_port() -> str:
    return envs.TPU_KV_TRANSFER_PORT


def get_side_channel_port() -> str:
    return envs.TPU_SIDE_CHANNEL_PORT


def get_node_id() -> int:
    # TODO(xiang): Is it possible to get this from a pre-defiend env?
    return envs.TPU_NODE_ID


def get_transfer_channel_number() -> int:
    """Parallel TCP stream count for the coord-mode KV wire transfer.

    0 (default) means auto: coord path uses tp_size channels, one per
    producer rank shard, bound on consecutive ports base..base+tp_size-1.
    A non-zero override is clamped to tp_size; ranks are round-robined
    onto channels by ``rank % n_channels``."""
    return envs.TPU_KV_TRANSFER_CHANNEL_NUMBER


def get_kv_coord_executor_max_workers() -> int:
    """Request-level pull executor size.

    0 means auto. This executor owns one task per active pull request; each
    task fans out channel work to the bounded channel executor.
    """
    return envs.TPU_KV_COORD_EXECUTOR_MAX_WORKERS


def get_kv_channel_executor_max_workers() -> int:
    """Channel-level pull executor size.

    0 means auto. This bounds simultaneous channel recv/unpack tasks across
    all active pull requests.
    """
    return envs.TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS


def get_p2p_wait_pull_timeout() -> int:
    """KV-cache transfer timeout in seconds."""
    return envs.TPU_P2P_WAIT_PULL_TIMEOUT


def get_stage3_status_probe_s() -> float:
    """Interval, in seconds, at which a Stage-3 producer worker probes the
    source store's request-block registry for its registrations that have
    not reached a native terminal. A consumer's release-only cancellation
    retires only the registry row, so without the probe the registration
    (and vLLM's delayed block free waiting on it) sits until
    p2p_wait_pull_timeout. 0 disables the probe."""
    return max(envs.TPU_RAIDEN_STAGE3_STATUS_PROBE_S, 0.0)


def get_stage3_deferred_submit_enabled() -> bool:
    """Per-step deferred re-attempt for Stage-3 loads whose producer
    request-block registrations have not arrived yet (default on; the
    registration is out-of-band and rides a later producer scheduler step,
    so a consumer submit can legitimately race ahead of it), instead of the
    legacy
    bounded inline poll that blocks the worker step while waiting.

    Inline-load mode keeps the legacy poll: the worker blocks in-step for
    completion there, so nothing would re-drive a parked submit.
    """
    if get_raiden_inline_load():
        return False
    return envs.TPU_RAIDEN_STAGE3_DEFERRED_SUBMIT


def get_stage3_registration_wait_s() -> float:
    """Total budget for deferred Stage-3 submit re-attempts, bounded by the
    overall pull timeout. The producer's registration rides a later producer
    scheduler step, so under load this must cover multiple step times."""
    val = envs.TPU_RAIDEN_STAGE3_REGISTRATION_WAIT_S
    if val <= 0:
        val = 30.0
    return min(val, float(get_p2p_wait_pull_timeout()))


def get_stage3_test_registration_delay_s() -> float:
    """TEST-ONLY fault injection: delay the producer's Stage-3 request-block
    registration by this many seconds to widen the out-of-band
    registration race (0 = disabled)."""
    return max(envs.TPU_RAIDEN_TEST_REGISTRATION_DELAY_S, 0.0)


def get_kv_stage_wait_timeout_secs() -> float:
    """Per-entry deadline for the async-D2H ``future.wait()`` in the stage
    waiter. ``TransferFuture.wait()`` is unbounded; this caps how long a
    single hung DMA can hold up subsequent uuids on the same rank. On
    timeout the entry is marked stage_failed and STAGE_DONE is signaled so
    the rest of the pipeline keeps moving.
    """
    val = envs.TPU_KV_STAGE_WAIT_TIMEOUT_SECS
    return val if val > 0 else 30.0


def get_kv_stage_waiter_pool_size() -> int:
    """Worker count for the stage waiter pool. 0 = auto (4).

    Sized above 1 so a single hung ``future.wait()`` does not serialize
    every subsequently-enqueued uuid behind it.
    """
    return envs.TPU_KV_STAGE_WAITER_POOL_SIZE


def get_kv_shm_pool_gb() -> float:
    """Total GB budget for the per-host shared-memory KV staging pool.

    The slot count is derived at runtime: ``num_slots = max(1, floor(
    budget_bytes / per_slot_bytes))``. Each slot holds one request's
    worth of KV data across all TP ranks (~1GB for typical 70B-class
    models at max_model_len=8192, TP=8)."""
    return envs.TPU_KV_SHM_POOL_GB


def _get_kv_transfer_namespace() -> str:
    namespace = envs.TPU_KV_TRANSFER_NAMESPACE
    if not namespace:
        return ""
    safe_chars = []
    for char in namespace:
        if char.isalnum() or char in ("_", "-"):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def get_ipc_socket_path(node_id: int, dp_rank: int = 0) -> str:
    """Per-host intra-process IPC endpoint used between rank-0 coordinator
    and the other TP ranks."""
    # One socket per host id and DP namespace; same-host P/D disaggregation
    # may also set TPU_KV_TRANSFER_NAMESPACE for independent engines.
    prefix = envs.TPU_IPC_SOCKET_DIR
    namespace = _get_kv_transfer_namespace()
    suffix = f"_{namespace}" if namespace else ""
    return f"ipc://{prefix}/tpu_conn{suffix}_node{node_id}_{dp_rank}.sock"


def get_shm_name(node_id: int, dp_rank: int = 0) -> str:
    """Name for the per-host SharedMemory block.

    Scoped by host id, DP rank, and optional engine namespace so same-host
    prefill/decode processes do not attach to each other's staging pool.
    """
    namespace = _get_kv_transfer_namespace()
    suffix = f"_{namespace}" if namespace else ""
    return f"tpu_conn_kv{suffix}_node{node_id}_{dp_rank}"


def get_kv_warmup_enabled() -> bool:
    """Whether to pre-compile KV gather/scatter XLA executables at startup.

    Without warmup, the first PULL on the consumer side can time out because
    the producer is blocked compiling index_select / index_put_ for shapes
    it has never seen before."""
    return envs.TPU_KV_WARMUP_ENABLED


def get_kv_latency_log_interval() -> float:
    """How often (seconds) to log aggregated KV transfer latency stats.
    Set to 0 to disable periodic summaries (per-request lines still print)."""
    return envs.TPU_KV_LATENCY_LOG_INTERVAL


def get_kv_pin_shm() -> bool:
    """Whether to mlock(2) the KV shm pool at startup.

    Off by default. Requires RLIMIT_MEMLOCK >= TPU_KV_SHM_POOL_GB or
    CAP_IPC_LOCK; failure is logged and the pool stays pageable. The
    primary motivation is to make `transfer_h2d_batch` from shm closer
    to the D2H direction's latency by keeping pages resident in RAM."""
    return envs.TPU_KV_PIN_SHM


def get_use_raiden_connector() -> bool:
    """Whether TPUConnector should use the opt-in Raiden backend.

    The vLLM config flag ``kv_connector_extra_config.use_raiden_connector``
    takes precedence when present. This environment variable is retained for
    manual launch scripts and defaults to disabled.
    """
    return envs.TPU_USE_RAIDEN_CONNECTOR


def get_raiden_transfer_num_slots() -> int:
    """Override for Raiden-owned per-rank host staging slots.

    0 means auto-size from TPU_KV_SHM_POOL_GB, split across TP ranks.
    """
    return envs.TPU_RAIDEN_TRANSFER_NUM_SLOTS


def get_raiden_inline_load() -> bool:
    """Load remote KV before the first forward instead of a no-forward step."""
    return envs.TPU_RAIDEN_INLINE_LOAD


_RAIDEN_TELEMETRY_MODULE = None
_RAIDEN_TELEMETRY_CONFIGURED = False
_RAIDEN_TELEMETRY_IMPORT_ATTEMPTED = False


def get_raiden_telemetry_module():
    """Gets and caches the TPU Raiden C++ extension module."""
    global _RAIDEN_TELEMETRY_MODULE, _RAIDEN_TELEMETRY_IMPORT_ATTEMPTED
    if not _RAIDEN_TELEMETRY_IMPORT_ATTEMPTED:
        # Try to load raiden module only once and ignore other attempts
        _RAIDEN_TELEMETRY_IMPORT_ATTEMPTED = True
        try:
            from tpu_sync.api.torch import kv_cache_manager as kcm
            _RAIDEN_TELEMETRY_MODULE = kcm._torch_impl()
        except Exception as e:
            logger.warning("Failed to import TPU Raiden telemetry module: %s",
                           e)
            return None
    return _RAIDEN_TELEMETRY_MODULE


def configure_raiden_telemetry(
        backends: list[str] | set[str] | None = None) -> None:
    """Configures TPU Raiden C++ telemetry backends if available."""
    global _RAIDEN_TELEMETRY_CONFIGURED
    try:
        telemetry = get_raiden_telemetry_module()
        if telemetry is not None and hasattr(telemetry, "configure_telemetry"):
            telemetry.configure_telemetry(backends)
            _RAIDEN_TELEMETRY_CONFIGURED = True
            logger.info("Configured TPU Raiden C++ telemetry backends")
    except Exception as e:
        logger.warning("Failed to configure TPU Raiden C++ telemetry: %s", e)
