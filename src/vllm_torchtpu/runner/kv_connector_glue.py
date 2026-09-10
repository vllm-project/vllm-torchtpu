# SPDX-License-Identifier: Apache-2.0
import copy
from contextlib import contextmanager
from typing import Any, Iterator

from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          kv_transfer_state)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput

from vllm_torchtpu.distributed import utils as dist_utils


@contextmanager
def _suspend_kv_transfer_group() -> Iterator[None]:
    """Hide the process-global connector during worker-local warmup."""
    connector = kv_transfer_state._KV_CONNECTOR_AGENT
    kv_transfer_state._KV_CONNECTOR_AGENT = None
    try:
        yield
    finally:
        kv_transfer_state._KV_CONNECTOR_AGENT = connector


_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP = (
    "invalid_block_group_index" in getattr(KVConnectorOutput,
                                           "__dataclass_fields__", {}))


def _build_kv_connector_output(
    *,
    finished_sending: set[str] | None,
    finished_recving: set[str] | None,
    kv_connector_worker_meta: Any | None,
    invalid_block_ids: set[int],
    invalid_block_group_index: int | None,
    kv_connector_stats: Any | None = None,
) -> KVConnectorOutput:
    """Build a connector output across the vLLM 0.23 API boundary.

    vLLM 0.23 supports invalid block IDs only for a single KV cache group and
    does not expose ``invalid_block_group_index``. Newer vLLM versions support
    group-scoped recovery for hybrid caches. Do not pass the newer keyword to
    0.23, but fail closed if a hybrid load actually fails: silently dropping
    the group would make overlapping block IDs ambiguous to its scheduler.
    """
    kwargs: dict[str, Any] = {
        "finished_sending": finished_sending,
        "finished_recving": finished_recving,
        "kv_connector_worker_meta": kv_connector_worker_meta,
        "invalid_block_ids": invalid_block_ids,
        "kv_connector_stats": kv_connector_stats,
    }
    if _KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP:
        kwargs["invalid_block_group_index"] = invalid_block_group_index
    elif invalid_block_group_index is not None:
        raise RuntimeError(
            "Installed vLLM does not support cache-group-scoped KV load "
            "failure recovery; upgrade vLLM before recovering a failed "
            f"hybrid KV load (group={invalid_block_group_index})")
    return KVConnectorOutput(**kwargs)


class KvConnectorMixin:
    """Bridge to the vLLM KV connector API."""

    def maybe_setup_kv_connector(self,
                                 scheduler_output,
                                 wait_for_completion: bool = False,
                                 report_completion: bool = True) -> None:
        if not has_kv_transfer_group():
            return
        kv_connector = get_kv_transfer_group()
        assert scheduler_output.kv_connector_metadata is not None
        # Upstream parity (gpu_model_runner calls this before every forward;
        # base-class default is a no-op): lets connectors with async saves
        # fence in-flight stores whose source blocks the scheduler is about
        # to reuse (OffloadingConnector jobs_to_flush) before the forward
        # overwrites them.
        kv_connector.handle_preemptions(scheduler_output.kv_connector_metadata)
        kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata)
        # forward_context is unused by TPUConnector; pass None.
        if wait_for_completion:
            kv_connector.start_load_kv(None,
                                       wait_for_completion=True,
                                       report_completion=report_completion)
        else:
            kv_connector.start_load_kv(None)

    def maybe_wait_for_kv_save(self) -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    def get_finished_kv_transfers(self, scheduler_output):
        if not has_kv_transfer_group():
            return None, None, None, set(), None, None
        kv_connector = get_kv_transfer_group()
        finished_sending, finished_recving = kv_connector.get_finished(
            scheduler_output.finished_req_ids)
        invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
        invalid_block_group_index = (
            kv_connector.get_block_ids_with_load_errors_group_index()
            if invalid_block_ids else None)
        # vLLM >=0.21 job model: store completions (and load completions) are
        # reported to the scheduler via the worker meta's `completed_jobs`,
        # NOT via finished_sending. Without plumbing this, the
        # OffloadingConnector's complete_store is never called, host-pool
        # blocks stay not-ready (ref_cnt=-1), and prefix-reuse lookups defer
        # forever -> the engine busy-spins / hangs. Mirrors
        # KVConnectorModelRunnerMixin._get_kv_connector_output.
        worker_meta = kv_connector.build_connector_worker_meta()
        kv_connector_stats = kv_connector.get_kv_connector_stats()
        # Mirror KVConnectorModelRunnerMixin._get_kv_connector_output:
        # metadata is bound per-step and must be cleared after use.
        kv_connector.clear_connector_metadata()
        return (finished_sending, finished_recving, worker_meta,
                invalid_block_ids, invalid_block_group_index,
                kv_connector_stats)

    def kv_connector_no_forward(self, scheduler_output,
                                vllm_config) -> ModelRunnerOutput:
        # Only Raiden inline mode blocks; otherwise the no-forward step returns
        # immediately and the scheduler re-polls get_finished each step until
        # the loads land.
        self.maybe_setup_kv_connector(
            scheduler_output,
            wait_for_completion=dist_utils.get_raiden_inline_load())
        (finished_sending, finished_recving, worker_meta, invalid_block_ids,
         invalid_block_group_index,
         kv_connector_stats) = self.get_finished_kv_transfers(scheduler_output)
        kv_connector_output = _build_kv_connector_output(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            kv_connector_worker_meta=worker_meta,
            invalid_block_ids=invalid_block_ids,
            invalid_block_group_index=invalid_block_group_index,
            kv_connector_stats=kv_connector_stats,
        )
        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output
