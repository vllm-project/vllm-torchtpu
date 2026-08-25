import os

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
        num_nodes = len(_NODES_KV_IP_PORT)
        ips = []
        for node_id in range(num_nodes):
            ips.append(_NODES_KV_IP_PORT[node_id][0])
        return ips
    else:
        return get_host_ip()


def get_kv_ports() -> str:
    if envs.TPU_MULTIHOST_BACKEND == "ray":
        num_nodes = len(_NODES_KV_IP_PORT)
        ports = []
        for node_id in range(num_nodes):
            ports.append(_NODES_KV_IP_PORT[node_id][1])
        return ports
    else:
        return get_kv_transfer_port()


def get_host_ip() -> str:
    """Use `VLLM_HOST_IP` if set, otherwise use default network interface IP."""
    return get_ip()


def get_kv_transfer_port() -> str:
    port = os.getenv("TPU_KV_TRANSFER_PORT", "9100")
    return port


def get_side_channel_port() -> str:
    port = os.getenv("TPU_SIDE_CHANNEL_PORT", "9600")
    return port


def get_node_id() -> int:
    # TODO(xiang): Is it possible to get this from a pre-defiend env?
    id = os.getenv("TPU_NODE_ID", 0)
    return int(id)


def get_transfer_channel_number() -> int:
    """Parallel TCP stream count for the coord-mode KV wire transfer.

    0 (default) means auto: coord path uses tp_size channels, one per
    producer rank shard, bound on consecutive ports base..base+tp_size-1.
    A non-zero override is clamped to tp_size; ranks are round-robined
    onto channels by ``rank % n_channels``."""
    n = os.getenv("TPU_KV_TRANSFER_CHANNEL_NUMBER", "0")
    return int(n)


def _get_nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, value, default)
        return default
    if parsed < 0:
        logger.warning("Invalid %s=%d; using %d", name, parsed, default)
        return default
    return parsed


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    logger.warning("Invalid %s=%r; using %s", name, value, default)
    return default


def get_kv_coord_executor_max_workers() -> int:
    """Request-level pull executor size.

    0 means auto. This executor owns one task per active pull request; each
    task fans out channel work to the bounded channel executor.
    """
    return _get_nonnegative_int_env("TPU_KV_COORD_EXECUTOR_MAX_WORKERS", 0)


def get_kv_channel_executor_max_workers() -> int:
    """Channel-level pull executor size.

    0 means auto. This bounds simultaneous channel recv/unpack tasks across
    all active pull requests.
    """
    return _get_nonnegative_int_env("TPU_KV_CHANNEL_EXECUTOR_MAX_WORKERS", 0)


def get_p2p_wait_pull_timeout() -> int:
    """KV-cache transfer timeout in seconds."""
    timeout_str = os.getenv("TPU_P2P_WAIT_PULL_TIMEOUT", "120")
    return int(timeout_str)


def get_kv_stage_wait_timeout_secs() -> float:
    """Per-entry deadline for the async-D2H ``future.wait()`` in the stage
    waiter. ``TransferFuture.wait()`` is unbounded; this caps how long a
    single hung DMA can hold up subsequent uuids on the same rank. On
    timeout the entry is marked stage_failed and STAGE_DONE is signaled so
    the rest of the pipeline keeps moving.
    """
    val_str = os.getenv("TPU_KV_STAGE_WAIT_TIMEOUT_SECS", "30")
    try:
        val = float(val_str)
    except ValueError:
        return 30.0
    return val if val > 0 else 30.0


def get_kv_stage_waiter_pool_size() -> int:
    """Worker count for the stage waiter pool. 0 = auto (4).

    Sized above 1 so a single hung ``future.wait()`` does not serialize
    every subsequently-enqueued uuid behind it.
    """
    return _get_nonnegative_int_env("TPU_KV_STAGE_WAITER_POOL_SIZE", 0)


def get_kv_shm_pool_gb() -> float:
    """Total GB budget for the per-host shared-memory KV staging pool.

    The slot count is derived at runtime: ``num_slots = max(1, floor(
    budget_bytes / per_slot_bytes))``. Each slot holds one request's
    worth of KV data across all TP ranks (~1GB for typical 70B-class
    models at max_model_len=8192, TP=8)."""
    gb_str = os.getenv("TPU_KV_SHM_POOL_GB", "128")
    return float(gb_str)


def _get_kv_transfer_namespace() -> str:
    namespace = os.getenv("TPU_KV_TRANSFER_NAMESPACE", "").strip()
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
    prefix = os.getenv("TPU_IPC_SOCKET_DIR", "/tmp")
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
    enable_str = os.getenv("TPU_KV_WARMUP_ENABLED", "true").lower()
    return enable_str in ("true", "1", "yes")


def get_kv_latency_log_interval() -> float:
    """How often (seconds) to log aggregated KV transfer latency stats.
    Set to 0 to disable periodic summaries (per-request lines still print)."""
    val_str = os.getenv("TPU_KV_LATENCY_LOG_INTERVAL", "30")
    try:
        return float(val_str)
    except ValueError:
        return 30.0


def get_kv_pin_shm() -> bool:
    """Whether to mlock(2) the KV shm pool at startup.

    Off by default. Requires RLIMIT_MEMLOCK >= TPU_KV_SHM_POOL_GB or
    CAP_IPC_LOCK; failure is logged and the pool stays pageable. The
    primary motivation is to make `transfer_h2d_batch` from shm closer
    to the D2H direction's latency by keeping pages resident in RAM."""
    enable_str = os.getenv("TPU_KV_PIN_SHM", "false").lower()
    return enable_str in ("true", "1", "yes")


def get_use_raiden_connector() -> bool:
    """Whether TPUConnector should use the opt-in Raiden backend.

    The vLLM config flag ``kv_connector_extra_config.use_raiden_connector``
    takes precedence when present. This environment variable is retained for
    manual launch scripts and defaults to disabled.
    """
    return _get_bool_env("TPU_USE_RAIDEN_CONNECTOR", False)


def get_raiden_transfer_num_slots() -> int:
    """Override for Raiden-owned per-rank host staging slots.

    0 means auto-size from TPU_KV_SHM_POOL_GB, split across TP ranks.
    """
    return _get_nonnegative_int_env("TPU_RAIDEN_TRANSFER_NUM_SLOTS", 0)


def get_raiden_inline_load() -> bool:
    """Load remote KV before the first forward instead of a no-forward step."""
    return _get_bool_env("TPU_RAIDEN_INLINE_LOAD", False)


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
