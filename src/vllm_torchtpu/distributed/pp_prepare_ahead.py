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
"""Prepare the next ring input before waiting for output queue capacity.

The original end_forward already owns a newly produced output while waiting
behind ``depth`` queued outputs. Enqueue that same extra output immediately;
wait before the next forward instead. Live outgoing tensors remain bounded
by depth + 1. One more input can be posted/prepared, so receive credits need
a separate, stricter configuration check.
"""

import functools

_installed = False


def install_worker():
    global _installed
    if _installed:
        return
    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    execute = TPUWorker.execute_model

    @functools.wraps(execute)
    def execute_model(self, scheduler_output):
        # InputBatch.add/condense/remove can mutate the contiguous CPU
        # sampling buffers uploaded by refresh_metadata(copy_slice). Their
        # lifetime alone does not protect their contents from another batch.
        # The ordinary _prepare_inputs fence is too late for _update_states.
        # Include zero-token calls: removing old requests also mutates masks.
        self.model_runner._wait_input_staging_fence()
        return execute(self, scheduler_output)

    TPUWorker.execute_model = execute_model
    _installed = True


class PrepareAhead:
    def __init__(self, ring):
        self.ring = ring

    def wait(self):
        ring = self.ring
        with ring._cv:
            ring._cv.wait_for(
                lambda: (
                    len(ring._sends) <= ring.depth
                    or ring._error is not None
                    or ring._closing
                )
            )
            ring._raise_send_failure_locked()
            if ring._closing:
                raise RuntimeError("PP prepare-ahead forward started during close")

    def enqueue(self, transfer, outgoing, fence):
        ring = self.ring
        with ring._cv:
            ring._raise_send_failure_locked()
            assert outgoing, "Prepare-ahead only owns stages with outgoing tensors"
            assert len(ring._sends) <= ring.depth, (
                "Forward started without a prepare-ahead output credit",
                transfer,
                len(ring._sends),
                ring.depth,
            )
            ring._sends.append((transfer, outgoing, fence))
            ring._cv.notify_all()


def create(ring):
    if ring.rank == ring.last:
        return None
    install_worker()
    return PrepareAhead(ring)
