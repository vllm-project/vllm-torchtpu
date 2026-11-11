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
"""SparseCore/TensorCore engine choice for QSA (local SC-vs-TC experiment).

One process-wide switch, read from the environment. ``VLLM_TPU_QSA_TC_ONLY=1``
runs QSA as if the chip had no SparseCore:
  * the indexer top-k uses ``BisectTopK`` on the TensorCore;
  * index-key preparation stays on the TensorCore;
  * prefill attention always takes the TensorCore block-sparse kernel.

The TensorCore selector replaces the *default* backend, so the default indexer
schedule (short-context enumeration, query-sized score capacity, decode page
grouping) is the same in both modes; only the selection engine changes. Unset,
nothing changes.
"""

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    TopKBackend,
)


def qsa_tc_only() -> bool:
    """Whether QSA must stay off the SparseCore."""
    return bool(envs.environment_variables["VLLM_TPU_QSA_TC_ONLY"]())


def default_qsa_topk_backend() -> TopKBackend:
    """The selector the QSA indexer uses when its caller injects none."""
    if not qsa_tc_only():
        from vllm_torchtpu.kernels.qwen4_exp.resident_topk import QwenSparseCoreTopK

        return QwenSparseCoreTopK()
    # Imported lazily: the stock path must not depend on the model-owned copy.
    from vllm_torchtpu.models.qwen38_flash.kernels.streamindex_topk import BisectTopK

    return BisectTopK()
