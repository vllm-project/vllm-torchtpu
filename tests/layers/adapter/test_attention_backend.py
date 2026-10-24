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
"""The TPU attention backend registry and the platform's selection of it."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu.layers.adapter import attention_backend as ab
from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

pytestmark = pytest.mark.cpu_test

_TEST_BACKEND = "tpu_test_backend"
_TEST_PATH = (
    f"{PallasAttentionBackend.__module__}.{PallasAttentionBackend.__qualname__}"
)


@pytest.fixture
def registered():
    """Register _TEST_BACKEND; afterwards remove it from the registry and from
    AttentionBackendEnum. Enum has no public way to remove a member, and
    AttentionBackendEnum.register keeps it for the process, so the cleanup
    edits Enum's private member tables; otherwise the member leaks into later
    tests."""
    ab.register_attention_backend(_TEST_BACKEND, vllm_backend=_TEST_PATH)
    member = AttentionBackendEnum[_TEST_BACKEND.upper()]
    yield member
    ab._REGISTERED_BACKENDS.remove(_TEST_BACKEND)
    AttentionBackendEnum._member_map_.pop(member.name)
    AttentionBackendEnum._value2member_map_.pop(member.value)
    AttentionBackendEnum._member_names_.remove(member.name)
    delattr(AttentionBackendEnum, member.name)


def _vllm_config(backend, *, executor="uni"):
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend),
        parallel_config=SimpleNamespace(distributed_executor_backend=executor),
    )


def test_register_adds_an_enum_member(registered):
    assert registered.get_path() == _TEST_PATH
    assert ab.is_registered_attention_backend(registered)


def test_register_again_with_the_same_path_is_a_no_op(registered):
    ab.register_attention_backend(_TEST_BACKEND, vllm_backend=_TEST_PATH)

    assert AttentionBackendEnum[_TEST_BACKEND.upper()] is registered


def test_register_an_existing_name_at_another_path_raises(registered):
    with pytest.raises(ValueError, match="already has an attention backend"):
        ab.register_attention_backend(_TEST_BACKEND, vllm_backend="some.Other")


@pytest.mark.parametrize(
    "backend",
    [None, AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.CUSTOM],
)
def test_builtin_choices_are_not_registered(backend):
    assert not ab.is_registered_attention_backend(backend)


def test_platform_resolves_a_registered_name_to_its_class(registered):
    path = TpuPlatform.get_attn_backend_cls(
        selected_backend=registered,
        attn_selector_config=SimpleNamespace(use_mla=False),
    )

    assert path == _TEST_PATH


def test_platform_rejects_a_registered_name_for_mla(registered):
    with pytest.raises(ValueError, match="does not support MLA"):
        TpuPlatform.get_attn_backend_cls(
            selected_backend=registered,
            attn_selector_config=SimpleNamespace(use_mla=True),
        )


def test_platform_still_overrides_an_unsupported_builtin():
    path = TpuPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum.TRITON_ATTN,
        attn_selector_config=SimpleNamespace(use_mla=False),
    )

    assert path == AttentionBackendEnum.FLASH_ATTN.get_path()


@pytest.mark.parametrize(
    ("env", "executor", "in_ray_actor", "reason"),
    [
        ({"TPU_MULTIHOST_BACKEND": "ray"}, "uni", False, "TPU_MULTIHOST_BACKEND=ray"),
        ({}, "uni", True, "Ray actor"),
        ({"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}, "uni", False, "spawn"),
        (
            {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            },
            "mp",
            False,
            "spawn",
        ),
    ],
)
def test_check_rejects_processes_that_cannot_rebuild_the_name(
    registered, monkeypatch, env, executor, in_ray_actor, reason
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ab, "is_in_ray_actor", lambda: in_ray_actor)

    with pytest.raises(ValueError, match=reason):
        ab.check_attention_backend_config(_vllm_config(registered, executor=executor))


@pytest.mark.parametrize(
    "env",
    [
        {"VLLM_WORKER_MULTIPROC_METHOD": "fork"},
        # No process starts, so nothing unpickles the config.
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        },
    ],
)
def test_check_accepts_forked_or_in_process_engines(registered, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("TPU_MULTIHOST_BACKEND", raising=False)
    monkeypatch.setattr(ab, "is_in_ray_actor", lambda: False)

    ab.check_attention_backend_config(_vllm_config(registered))


def test_check_ignores_the_default_backend(monkeypatch):
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    ab.check_attention_backend_config(_vllm_config(None, executor=MagicMock()))
