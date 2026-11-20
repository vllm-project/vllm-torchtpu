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
"""Shared fixtures of the Qwen3.5 model tests."""

# The real GDN layer fixtures, for the fused gated-norm op's tests.
from tests.layers.adapter.gdn_fixtures import (  # noqa: F401
    _empty_shared_gdn_ops,
    make_gdn_attention,
)
