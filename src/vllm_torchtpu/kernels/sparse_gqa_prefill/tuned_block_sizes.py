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
"""TPU7x Q24/KV2/D256, ratio4, page1664 prefill tuning."""

import jax.numpy as jnp

# Validated number of query rows per outer dispatch chunk.
PREFILL_DISPATCH_CHUNK_QUERIES = 8192

# Both native dtypes use a 144 MiB compact selected-KV buffer.
SC_CHUNK_SIZE = {jnp.dtype(jnp.bfloat16): 32, jnp.dtype(jnp.float8_e4m3fn): 64}

# TPU7x: 16 words per reduction batch minimizes measured exact-union latency.
# At 8192 queries / 256K context, scratch stays below 65 MiB.
UNION_WORD_BATCH_SIZE = 16
