# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-FP8 whole-backbone decode megakernel, stage A (campaign glm52-decode-mk, WHITEBOARD.v1):
one gridless TensorCore pallas_call per step ('glm52_decode_mk', kernel.py) + the barrier-free cache writer
('glm52_decode_mk_writer', writer.py). Geometry in config.py, operand plan in layout.py, host / XLA glue and the
instruments candidate in wrapper.py, the batched DSA core in dsa.py, in-kernel collectives in collectives.py."""
