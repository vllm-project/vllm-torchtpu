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
"""An MK draft program failure in the spec-decode warmup is fatal unless
VLLM_TPU_MK_ALLOW_FALLBACK=1.

exos j-5ac1be1f: qwen38_mk_draft_t1 failed to compile (partitionId), the warmup logged
"failed; continuing", and the server then failed every request.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel.errors import MegakernelConfigError, MegakernelSetupError
from vllm_torchtpu.megakernel_runtime import qwen38_driver as drv
from vllm_torchtpu.runner import tpu_runner as tr
from vllm_torchtpu.spec_decode import eagle3


@pytest.fixture
def fallback(monkeypatch):
    def set_(on):
        monkeypatch.setattr(envs, "VLLM_TPU_MK_ALLOW_FALLBACK", on, raising=False)

    set_(False)
    return set_


def _boom(exc):
    def fn(*_a, **_k):
        raise exc

    return fn


def test_driver_reexports_the_shared_error():
    assert drv.MegakernelSetupError is MegakernelSetupError


def test_mk_program_wraps_compile_failures(fallback):
    err = NotImplementedError("partitionId instruction is not supported")
    with pytest.raises(MegakernelSetupError, match="fused draft loop failed") as info:
        eagle3._mk_program(_boom(err), "fused draft loop")()
    assert info.value.__cause__ is err
    for exc in (MegakernelConfigError("budget"), MegakernelSetupError("x")):
        with pytest.raises(type(exc)) as info:
            eagle3._mk_program(_boom(exc), "step-0 program")()
        assert info.value is exc
    assert eagle3._mk_program(lambda a, b=0: a + b, "x")(1, b=2) == 3


def test_mk_program_fallback_keeps_original(fallback):
    fallback(True)
    fn = _boom(NotImplementedError("x"))
    assert eagle3._mk_program(fn, "fused draft loop") is fn


def _runner(exc):
    r = tr.TPUModelRunner.__new__(tr.TPUModelRunner)
    group = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(max_num_blocks_per_req=lambda cfg, pos: 2)
    )
    r.__dict__.update(
        speculative_config=SimpleNamespace(num_speculative_tokens=1),
        kv_cache_config=SimpleNamespace(num_blocks=64, kv_cache_groups=[group]),
        max_model_len=256, vllm_config=None, cleaned=[],
    )
    r.execute_model = _boom(exc)
    r._warmup_spec_decode_cleanup = r.cleaned.append
    return r


@pytest.mark.parametrize("exc", [MegakernelSetupError("mk"), MegakernelConfigError("vmem")])
def test_warmup_reraises_mk_failure(fallback, exc):
    r = _runner(exc)
    for quiet in (True, False):
        with pytest.raises(type(exc)):
            r._warmup_one_spec_request(8, 0, quiet=quiet)
    assert len(r.cleaned) == 2  # the request is still torn down


def test_warmup_mk_failure_continues_with_fallback(fallback):
    fallback(True)
    assert _runner(MegakernelSetupError("mk"))._warmup_one_spec_request(8, 0) is False


def test_warmup_other_failures_still_continue(fallback):
    assert _runner(RuntimeError("symint"))._warmup_one_spec_request(8, 0) is False
