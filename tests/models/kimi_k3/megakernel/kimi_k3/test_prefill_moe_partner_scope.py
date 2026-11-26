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

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_moe import fused

_HIDDEN, _LAYERS, _LOCAL_EXPERTS = 7168, 2, 56


class _TraceRecorder:
    def __init__(self):
        self.events: list[tuple[str, frozenset[str]]] = []
        self._scopes: list[frozenset[str]] = []

    def current_scope(self) -> frozenset[str]:
        return self._scopes[-1] if self._scopes else frozenset()

    @contextlib.contextmanager
    def scope(self, names: frozenset[str]):
        self._scopes.append(names)
        try:
            yield
        finally:
            self._scopes.pop()

    def record(self, event: str):
        self.events.append((event, self.current_scope()))


def _install_recorder(monkeypatch) -> _TraceRecorder:
    recorder = _TraceRecorder()
    run_scoped, signal, wait = pl.run_scoped, pl.semaphore_signal, pl.semaphore_wait
    remote_copy = pltpu.make_async_remote_copy

    def recording_run_scoped(body, *types, **kw_types):
        names = set(kw_types)
        for value in types:
            if isinstance(value, dict):
                names |= set(value)

        def scoped_body(*args, **kwargs):
            with recorder.scope(frozenset(names)):
                return body(*args, **kwargs)

        return run_scoped(scoped_body, *types, **kw_types)

    def recording_signal(*args, **kwargs):
        if kwargs.get("device_id") is not None:
            recorder.record("remote_signal")
        return signal(*args, **kwargs)

    def recording_wait(*args, **kwargs):
        recorder.record("wait")
        return wait(*args, **kwargs)

    def recording_remote_copy(*args, **kwargs):
        recorder.record("remote_copy")
        return remote_copy(*args, **kwargs)

    monkeypatch.setattr(pl, "run_scoped", recording_run_scoped)
    monkeypatch.setattr(pl, "semaphore_signal", recording_signal)
    monkeypatch.setattr(pl, "semaphore_wait", recording_wait)
    monkeypatch.setattr(pltpu, "make_async_remote_copy", recording_remote_copy)
    return recorder


def _trace_prologue(tokens: int, *, dense_fp8: bool) -> None:
    rows = tokens // 32
    dense_dtype = jnp.float8_e4m3fn if dense_fp8 else jnp.bfloat16

    def kernel(
        tables,
        control,
        hidden,
        router,
        bias,
        latent_down,
        shared_gu,
        shared_down,
        *rest,
    ):
        scales, outputs = rest[:-5], rest[-5:]
        latent_out, ids_v, probs_v, counts_v, shared = outputs
        fused._prologue(  # noqa: SLF001 -- partner-scope ordering is only observable in the traced prologue
            tables,
            control,
            hidden,
            router,
            bias,
            latent_down,
            shared_gu,
            shared_down,
            shared,
            latent_out,
            ids_v,
            probs_v,
            counts_v,
            None,
            counts_ready=lambda: None,
            metadata_ready=lambda: None,
            dense_scales=scales if dense_fp8 else None,
        )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((tokens, 2, 1792), jnp.bfloat16),
            jax.ShapeDtypeStruct((tokens // 8, 128), jnp.int32),
            jax.ShapeDtypeStruct((tokens // 8, 128), jnp.float32),
            jax.ShapeDtypeStruct((_LOCAL_EXPERTS,), jnp.int32),
            jax.ShapeDtypeStruct((rows, _HIDDEN), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(hbm,) * (9 if dense_fp8 else 6),
            out_specs=(hbm, vmem, vmem, vmem, vmem),
        ),
    )
    operands = (
        jax.ShapeDtypeStruct((3, 3, 32), jnp.int32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
        jax.ShapeDtypeStruct((rows, _HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((_LAYERS, _HIDDEN, 896), jnp.bfloat16),
        jax.ShapeDtypeStruct((_LAYERS, 1, 896), jnp.float32),
        jax.ShapeDtypeStruct((_LAYERS, _HIDDEN, 3584), dense_dtype),
        jax.ShapeDtypeStruct((_LAYERS, _HIDDEN, 6144), dense_dtype),
        jax.ShapeDtypeStruct((_LAYERS, 3072, _HIDDEN), dense_dtype),
    )
    if dense_fp8:
        operands += tuple(
            jax.ShapeDtypeStruct((_LAYERS, 1, cols), jnp.float32)
            for cols in (3584, 6144, _HIDDEN)
        )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("k3_sp",))
    replicated = jax.sharding.PartitionSpec()
    traced = jax.shard_map(
        call,
        mesh=mesh,
        in_specs=(replicated,) * len(operands),
        out_specs=(replicated,) * 5,
        check_vma=False,
    )
    jax.make_jaxpr(traced)(*operands)


@pytest.mark.parametrize(
    ("tokens", "dense_fp8"),
    [(2048, False), (4096, False), (8192, False), (4096, True)],
)
def test_should_handshake_chip_partner_before_remote_writes_into_shared_down_scope(
    monkeypatch, tokens, dense_fp8
):
    monkeypatch.setattr(
        config, "active", lambda: config.KernelConfig(prefill_expert_parallel=16)
    )
    recorder = _install_recorder(monkeypatch)

    _trace_prologue(tokens, dense_fp8=dense_fp8)

    down_events = [event for event, scope in recorder.events if "peer_y" in scope]
    assert "remote_copy" in down_events, (
        "the shared-down scope must reduce with the partner"
    )
    first_copy = down_events.index("remote_copy")
    assert down_events[:2] == ["remote_signal", "wait"], down_events[: first_copy + 1]
