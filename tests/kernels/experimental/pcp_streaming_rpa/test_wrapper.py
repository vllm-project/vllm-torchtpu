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

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.wrapper import \
    compute_pcp_rank_major_slot_ids_from_metadata


def test_metadata_slot_ids_support_smaller_interleave_than_page_size():
    slot_ids = compute_pcp_rank_major_slot_ids_from_metadata(
        kv_lens=jnp.asarray([8], dtype=jnp.int32),
        page_indices=jnp.asarray([10], dtype=jnp.int32),
        cu_q_lens=jnp.asarray([0, 8], dtype=jnp.int32),
        distribution=jnp.asarray([0, 0, 1], dtype=jnp.int32),
        local_padded_tokens=4,
        local_kv_cache_num_blocks=16,
        page_size=4,
        pcp_size=2,
        interleave_size=2,
    )

    actual = np.asarray(jax.device_get(slot_ids))
    np.testing.assert_array_equal(
        actual,
        np.array([
            40,
            41,
            42,
            43,
            40,
            41,
            42,
            43,
        ], dtype=np.int32),
    )
