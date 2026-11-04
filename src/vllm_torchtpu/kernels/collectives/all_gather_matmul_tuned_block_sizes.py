# Copyright 2025 Google LLC
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
"""All-gather matmul kernel's tuned block sizes."""

import jax

# key:
#    - tpu_version
#    - m
#    - n
#    - k
#    - dtype
#    - tp_size
# value:
#    - bn
#    - bk
TUNED_BLOCK_SIZES = {
    # keep-sorted start
    (6, 1024, 51200, 5120, "bfloat16", 8): (6400, 2560),
    (6, 1024, 57344, 8192, "bfloat16", 8): (7168, 8192),
    (6, 2048, 51200, 5120, "bfloat16", 8): (1280, 5120),
    (6, 2048, 57344, 8192, "bfloat16", 8): (1024, 8192),
    (6, 4096, 51200, 5120, "bfloat16", 8): (3200, 5120),
    (6, 8192, 51200, 5120, "bfloat16", 8): (1280, 5120),
    # keep-sorted end
}


def get_tpu_version() -> int:
    """Returns the numeric version of the TPU, or -1 if not on TPU."""
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        return -1
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
    assert kind[:-1] == "TPU v", kind
    return int(kind[-1])


def get_key(
    m,
    n,
    k,
    dtype,
    tp_size,
):
    """Returns the key for the given parameters."""
    return (
        get_tpu_version(),
        m,
        n,
        k,
        dtype,
        tp_size,
    )


def get_tuned_block_sizes(m, n, k, dtype_name, tp_size):
    """Returns the tuned block sizes for the given parameters."""
    key = get_key(m, n, k, dtype_name, tp_size)
    return TUNED_BLOCK_SIZES.get(key, (None, None))
