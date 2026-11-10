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
"""Wiring of the path gdn_paired_heads: the switch reaches the verify-window
kernel, which then forms every dot over two value heads of one key head."""

import re

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.envs as envs
from vllm_torchtpu import decode_switches
from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import config, window_tile, wrapper
from vllm_torchtpu.layers.core import gdn_attention

pytestmark = pytest.mark.cpu_test

N_KQ, N_V, D, KERNEL_SIZE = 4, 32, 128, 4
DIM = N_KQ * D * 2 + N_V * D
PAGE = (2, 64, 4, 128)  # fp8 SEQ_ALONG_LANE pool page
SPLIT = 17  # 2176-token manager blocks of 128-token pages


@pytest.mark.parametrize(
    ("value", "expected"), [(None, False), ("", False), ("0", False), ("1", True)]
)
def test_switch_is_off_unless_set(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TPU_DECODE_FAST_PATHS", raising=False)
    else:
        monkeypatch.setenv("TPU_DECODE_FAST_PATHS", value)
    assert envs.environment_variables["TPU_DECODE_FAST_PATHS"]() is expected
    assert decode_switches.get().path("gdn_paired_heads") is expected


@pytest.mark.parametrize("paired", [False, True])
def test_pooled_gdn_hands_the_switch_to_the_kernel(monkeypatch, paired):
    seen = {}

    def fused_conv1d_gdn(*args, **kwargs):
        seen.update(kwargs)
        return kwargs["state_source"], None

    monkeypatch.setattr(
        gdn_attention, "_build_v3_pool_state_plan", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        gdn_attention.gdn_v3_wrapper, "fused_conv1d_gdn", fused_conv1d_gdn
    )
    monkeypatch.setenv("TPU_DECODE_FAST_PATHS", "1")
    monkeypatch.setenv(
        "TPU_DECODE_FAST_PATHS_OFF", "" if paired else "gdn_paired_heads"
    )
    gdn_attention.run_jax_gdn_attention_pooled_local(
        *([None] * 3),
        jnp.zeros((1,)),
        jnp.zeros((DIM, 1, KERNEL_SIZE)),
        *([None] * 7),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KERNEL_SIZE,
        pool_block_tokens=SPLIT * 128,
    )
    assert seen["paired_heads"] is paired


def _cfg(n_kq: int, n_v: int, paired: bool) -> config.GDNConfig:
    f32 = jnp.dtype(jnp.float32)
    return config.GDNConfig(
        mode=config.GDNMode.BATCHED,
        dtypes=config.Dtypes(
            act_in=f32,
            act_out=jnp.dtype(jnp.bfloat16),
            compute=f32,
            recurrent_state=f32,
            conv_state=f32,
        ),
        batch_size=8,
        dim_size=(2 * n_kq + n_v) * D,
        kernel_size=KERNEL_SIZE,
        tile_size=2,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=D,
        v_head_dim=D,
        window_size=4,
        paired_heads=paired,
    )


@pytest.mark.parametrize(
    ("n_v", "paired", "groups"),
    [
        # Key head 1 of 4: value heads 8-15 in pairs, or one per dot.
        (32, True, [(8, 9), (10, 11), (12, 13), (14, 15)]),
        (32, False, [(head,) for head in range(8, 16)]),
        (8, True, [(2, 3)]),
        # An odd count per key head (a TP split of the value heads): one head
        # per dot.
        (12, True, [(3,), (4,), (5,)]),
        (4, True, [(1,)]),
    ],
)
def test_value_heads_pair_within_their_key_head(n_v, paired, groups):
    assert window_tile._head_groups(_cfg(4, n_v, paired), 1) == groups


@pytest.mark.parametrize("paired", [False, True])
def test_kernel_name_marks_paired_windows(paired):
    cfg = _cfg(4, 32, paired)
    assert cfg.get_kernel_name().endswith("_pairs") is paired


def _verify_jaxpr(paired: bool) -> str:
    """The jaxpr of the served two-slot verify call (one TP4 core, 4
    sequences of 4-token windows, batched segment only)."""
    seqs, tokens = 4, 16
    sds = jax.ShapeDtypeStruct
    bf16, i32 = jnp.bfloat16, jnp.int32
    pool = sds(((1 + seqs * 3 + 1) * SPLIT,) + PAGE, jnp.float8_e4m3fn)
    args = dict(
        qkv=sds((tokens, DIM), bf16),
        b=sds((tokens, N_V), bf16),
        a=sds((tokens, N_V), bf16),
        conv_weight=sds((DIM, 1, KERNEL_SIZE), bf16),
        a_log=sds((N_V,), jnp.float32),
        dt_bias=sds((N_V,), jnp.float32),
        query_start_loc=sds((seqs + 1,), i32),
        state_indices=sds((seqs,), i32),
        distribution=sds((3,), i32),
        seq_lens=sds((seqs,), i32),
        read_offsets=sds((seqs,), i32),
        ckpt_indices=sds((seqs, 4), i32),
    )

    def call(pool, args):
        plan = gdn_attention._build_v3_pool_state_plan(
            pool,
            conv_dim=DIM,
            n_v=N_V,
            d_k=D,
            d_v=D,
            kernel_size=KERNEL_SIZE,
            pool_block_tokens=SPLIT * 128,
            qk_pair_layout=True,
            recurrent_state_dtype=jnp.bfloat16,
        )
        return wrapper.fused_conv1d_gdn(
            conv_state=None,
            recurrent_state=None,
            conv_bias=None,
            state_source=pool,
            state_plan=plan,
            n_kq=N_KQ,
            n_v=N_V,
            d_k=D,
            d_v=D,
            kernel_size=KERNEL_SIZE,
            num_spec_tokens=3,
            batched_only=True,
            two_slot_ckpts=True,
            packed_tokens=True,
            paired_heads=paired,
            **args,
        )

    return str(jax.make_jaxpr(call)(pool, args))


def test_switch_pairs_every_dot_of_the_window_kernel(monkeypatch):
    # Traced only: JAX's static chip description stands in for a device.
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(pool_adapters, "is_seq_along_lane_pool", lambda: True)
    off, on = _verify_jaxpr(False), _verify_jaxpr(True)
    names = [set(re.findall(r"name=(fused_conv1d_gdn_\w+)", t)) for t in (off, on)]
    assert names == [
        {"fused_conv1d_gdn_batched_w4_2slot"},
        {"fused_conv1d_gdn_batched_w4_2slot_pairs"},
    ]
    # Window positions, pending-token replays and boundary recomputes alike:
    # one dot per two value heads.
    assert off.count("dot_general") == 2 * on.count("dot_general") > 0
