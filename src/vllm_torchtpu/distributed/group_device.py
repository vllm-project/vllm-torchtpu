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
"""The device of vLLM's parallel groups on TPU workers."""

import functools

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def patch_group_coordinator_device() -> None:
    """Name a parallel group's device as its worker addresses it.

    vLLM names the device of an out-of-tree platform `<device_name>:<local_rank>`
    (`GroupCoordinator.__init__`, vllm/distributed/parallel_state.py), for
    workers that each see every device of their host. A TPU worker sees its
    own device alone, so the local rank is an index it cannot address, and a
    tensor moved to the group's device (the DP batch sync without async
    scheduling) is refused.
    """
    import torch
    from vllm.distributed.parallel_state import GroupCoordinator

    if getattr(GroupCoordinator, "_tpu_group_device_patch", False):
        return
    _orig = GroupCoordinator.__init__

    @functools.wraps(_orig)
    def _wrapped(self, *args, **kwargs):
        _orig(self, *args, **kwargs)
        self.device = torch.device(self.device.type)
        if self.device_communicator is not None:
            self.device_communicator.device = self.device

    GroupCoordinator.__init__ = _wrapped
    GroupCoordinator._tpu_group_device_patch = True
    logger.info("Applied TPU patch: parallel groups on the worker's own device.")
