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
"""TPU_RPA_DECODE_VMEM_MIB / TPU_RPA_DECODE_BATCH: batched attention kwargs
for decode-sized programs."""

import functools

import jax.numpy as jnp
import pytest

from vllm_torchtpu.layers.core import attention_interface

VMEM_ENV = "TPU_RPA_DECODE_VMEM_MIB"
BATCH_ENV = "TPU_RPA_DECODE_BATCH"


@pytest.fixture(autouse=True)
def _clear_envs(monkeypatch):
    monkeypatch.delenv(VMEM_ENV, raising=False)
    monkeypatch.delenv(BATCH_ENV, raising=False)


@pytest.mark.parametrize("num_q_rows", [1, 8, 64, 128, 8192])
def test_unset_keeps_the_kernel_defaults(monkeypatch, num_q_rows):
    # TPU_RPA_DECODE_BATCH alone does nothing.
    monkeypatch.setenv(BATCH_ENV, "2")
    assert attention_interface.batched_rpa_decode_overrides(num_q_rows, 8) == {}


@pytest.mark.parametrize("num_q_rows", [1, 32, 64])
def test_decode_programs_take_the_vmem_limit(monkeypatch, num_q_rows):
    monkeypatch.setenv(VMEM_ENV, "24")
    assert attention_interface.batched_rpa_decode_overrides(num_q_rows, 8) == {
        "vmem_limit_bytes": 24 << 20,
        "decode_batch_size": 8,
    }


@pytest.mark.parametrize("num_q_rows", [65, 128, 8192])
def test_larger_programs_keep_the_kernel_defaults(monkeypatch, num_q_rows):
    monkeypatch.setenv(VMEM_ENV, "24")
    assert attention_interface.batched_rpa_decode_overrides(num_q_rows, 8) == {}


@pytest.mark.parametrize(
    ("batch_env", "num_seqs", "expected"),
    [(None, 4, 4), (None, 16, 8), ("2", 16, 2), ("8", 4, 8)],
)
def test_decode_batch(monkeypatch, batch_env, num_seqs, expected):
    monkeypatch.setenv(VMEM_ENV, "24")
    if batch_env is not None:
        monkeypatch.setenv(BATCH_ENV, batch_env)
    overrides = attention_interface.batched_rpa_decode_overrides(1, num_seqs)
    assert overrides["decode_batch_size"] == expected


def test_decode_batch_above_eight_is_rejected(monkeypatch):
    monkeypatch.setenv(VMEM_ENV, "24")
    monkeypatch.setenv(BATCH_ENV, "16")
    with pytest.raises(ValueError, match="at most 8"):
        attention_interface.batched_rpa_decode_overrides(1, 8)


def _kernel_kwargs(monkeypatch, num_q_rows, wrap=None, batched=True):
    """kwargs `sharded_ragged_paged_attention` passes to the kernel."""
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append(kwargs)
        return args[0], args[3]

    if batched:
        monkeypatch.setattr(
            attention_interface, "ragged_paged_attention_batched", fake_kernel
        )
    rpa_func = fake_kernel if wrap is None else wrap(fake_kernel)
    num_seqs, head_dim = 8, 128
    q = jnp.zeros((num_q_rows, 4, head_dim), jnp.bfloat16)
    kv = jnp.zeros((num_q_rows, 1, head_dim), jnp.bfloat16)
    attention_interface.sharded_ragged_paged_attention(
        None,
        q,
        kv,
        kv,
        jnp.zeros((4, 16, 1, 2, head_dim), jnp.bfloat16),
        jnp.zeros((num_seqs,), jnp.int32),
        jnp.zeros((num_seqs * 4,), jnp.int32),
        jnp.zeros((num_seqs + 1,), jnp.int32),
        jnp.zeros((3,), jnp.int32),
        None,
        1.0,
        rpa_func=rpa_func,
        shard=False,
    )
    (kwargs,) = calls
    return kwargs


def test_kernel_call_is_unchanged_when_unset(monkeypatch):
    kwargs = _kernel_kwargs(monkeypatch, 8)
    assert "vmem_limit_bytes" not in kwargs
    assert "decode_batch_size" not in kwargs


def test_kernel_call_takes_the_overrides_for_decode_programs(monkeypatch):
    monkeypatch.setenv(VMEM_ENV, "24")
    monkeypatch.setenv(BATCH_ENV, "2")
    kwargs = _kernel_kwargs(monkeypatch, 8)
    assert kwargs["vmem_limit_bytes"] == 24 << 20
    assert kwargs["decode_batch_size"] == 2
    # Speculative decoding passes the batched kernel as a partial.
    wrap = functools.partial(functools.partial, decode_query_size=2)
    kwargs = _kernel_kwargs(monkeypatch, 8, wrap=wrap)
    assert kwargs["decode_batch_size"] == 2


def test_kernel_call_is_unchanged_for_larger_programs(monkeypatch):
    monkeypatch.setenv(VMEM_ENV, "24")
    kwargs = _kernel_kwargs(monkeypatch, 128)
    assert "vmem_limit_bytes" not in kwargs


def test_default_kernel_is_unchanged(monkeypatch):
    monkeypatch.setenv(VMEM_ENV, "24")
    kwargs = _kernel_kwargs(monkeypatch, 8, batched=False)
    assert "vmem_limit_bytes" not in kwargs
