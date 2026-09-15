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

import gc

import pytest
from jax._src.api import clear_backends


@pytest.fixture(scope="module", autouse=True)
def release_jax_backend():
    """Release TPU devices once this module's tests are done.

    This directory sorts first under tests/kernels/, so a retained JAX client
    here would still hold every device when the pcp kernel tests later spawn
    their per-device worker processes. Autouse because jtu.JaxTestCase methods
    cannot request a fixture by argument, and module-scoped because a
    function-scoped finalizer runs before pytest drops the TestCase instance —
    the mesh and jitted callable on `self` would still pin the client. The
    same-named fixture under experimental/pcp_streaming_rpa is function-scoped
    and opt-in, which is correct there: its tests are plain functions whose
    locals are already gone when the finalizer runs.
    """
    yield
    clear_backends()
    gc.collect()
