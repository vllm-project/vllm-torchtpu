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
"""Lets the last pipeline stage take its next step before the sampled tokens land.

With async scheduling off, the last stage of a pipeline engine waits for each
step's sampled tokens in ``sample_tokens`` and the worker replies from its main
thread, so the next step's host work (the next RPC, ``_update_states``, the
ring receive, the connector hooks, input preparation) runs with the device
empty. Under ``VLLM_TPU_PP_LAST_ASYNC_OUTPUT`` the last stage instead:

* returns the ``AsyncTPUModelRunnerOutput`` without waiting on it;
* replies from vLLM's output thread (``WorkerProc.async_output_busy_loop``).
  It is one thread draining one FIFO queue, and every reply of the worker
  goes through it once it runs, so the engine reads the replies in call order.
  The thread waits on the sampled-token copy before it sends a step's output,
  so the engine still sees step N's output only after forward N and its
  sampler have run (the producer connector's pull gate is unchanged);
* takes the previous step's sampled tokens from the scheduler's
  ``scheduled_cached_reqs.new_token_ids``, which vLLM sends to every pipeline
  rank when async scheduling is off, as vLLM's runner does on the other ranks.

The last stage then runs its per-step host work while its device computes, as
stages before it already do.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner import pp_last_async_drafter as _pparf  # q38 PPARF (default off)

logger = init_logger(__name__)


def decide(
    vllm_config: Any,
    is_last_stage: bool,
    is_pooling_model: bool,
    prefill_only_drafter: bool = False,
) -> tuple[bool, str]:
    """(on, reason): whether this worker's runner returns unwaited outputs."""
    if not envs.VLLM_TPU_PP_LAST_ASYNC_OUTPUT:
        return False, "flag off"
    pc = vllm_config.parallel_config
    if int(pc.pipeline_parallel_size) <= 1:
        return False, "no pipeline"
    if not is_last_stage:
        return False, "not the last stage"
    if vllm_config.scheduler_config.async_scheduling:
        return False, "async scheduling already replies from the output thread"
    # q38 PPARF (VLLM_TPU_PP_LAST_ASYNC_PREFILL_DRAFTER): a prefill-only MTP
    # producer (the runner's _mtp_prefill_only) never verifies and seeds on
    # device, so with the flag on it is admitted; a verifying drafter is not.
    if vllm_config.speculative_config is not None and not (
        _pparf.enabled() and prefill_only_drafter
    ):
        # Sync pipeline spec decode drafts from host tokens after the wait.
        return False, "speculative decoding"
    if is_pooling_model:
        return False, "pooling model"
    stage_ranks = int(pc.tensor_parallel_size) * int(
        getattr(pc, "prefill_context_parallel_size", 1) or 1
    )
    if stage_ranks > 1 and vllm_config.kv_transfer_config is None:
        # The executor drops the reply of every last-stage rank but its output
        # rank, which would drop an unwaited copy still in flight.
        return False, "last stage has ranks whose replies are dropped"
    return True, "on"


def install_worker_output_thread() -> None:
    """Gives a worker the output thread once its runner returns an unwaited output.

    vLLM starts ``WorkerProc``'s output thread only under async scheduling.
    The first ``AsyncModelRunnerOutput`` a worker hands to ``handle_output``
    under sync scheduling starts it, and from then on vLLM's own
    ``handle_output`` queues every reply of that worker to it. Replies before
    that one were sent synchronously, so the order holds across the switch.
    Inert for a runner that never returns such an output. Idempotent; runs in
    the worker process.
    """
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.outputs import AsyncModelRunnerOutput

    if getattr(WorkerProc, "_tpu_pp_last_async_output_patch", False):
        return
    orig_handle_output = WorkerProc.handle_output

    def handle_output(self: Any, output: Any) -> None:
        if not self.use_async_scheduling and isinstance(
            output, AsyncModelRunnerOutput
        ):
            self.async_output_queue = queue.Queue()
            self.async_output_copy_thread = threading.Thread(
                target=self.async_output_busy_loop,
                daemon=True,
                name="WorkerAsyncOutputCopy",
            )
            self.async_output_copy_thread.start()
            self.use_async_scheduling = True
            logger.info(
                "PP last stage async output: replies go through the worker's "
                "output thread"
            )
        orig_handle_output(self, output)

    WorkerProc.handle_output = handle_output
    WorkerProc._tpu_pp_last_async_output_patch = True


def apply_scheduled_token_ids(requests: dict, input_batch: Any, req_data: Any) -> int:
    """Appends the sampled tokens the scheduler sends back to the cached requests.

    Same rule as vLLM's runner on pipeline ranks other than the last
    (gpu_model_runner._update_states): of a request's scheduled tokens,
    the ones past ``req_state.num_tokens`` are tokens sampled since it last
    ran. They go to ``output_token_ids`` and, for a request in the persistent
    batch, to ``token_ids_cpu`` and ``num_tokens_no_spec``: the same writes
    the synchronous ``sample_tokens`` makes once the tokens reach the host.
    A request outside the batch (resumed) is rebuilt from its
    ``output_token_ids``. Returns the number of tokens applied.
    """
    if not req_data.new_token_ids:
        return 0
    applied = 0
    for i, req_id in enumerate(req_data.req_ids):
        req_state = requests.get(req_id)
        if req_state is None:
            continue
        new_token_ids = req_data.new_token_ids[i]
        num_new = (
            req_data.num_computed_tokens[i] + len(new_token_ids) - req_state.num_tokens
        )
        if num_new <= 0:
            continue
        tokens = list(new_token_ids[-num_new:])
        req_index = input_batch.req_id_to_index.get(req_id)
        if req_index is not None:
            start = int(input_batch.num_tokens_no_spec[req_index])
            input_batch.token_ids_cpu[req_index, start : start + num_new] = tokens
            input_batch.num_tokens_no_spec[req_index] = start + num_new
        req_state.output_token_ids.extend(tokens)
        applied += num_new
    return applied
