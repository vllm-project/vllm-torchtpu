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
"""The GDN window log's plan regions and startup refusals (VLLM_TPU_GDN_WINDOW_LOG)."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.gdn.v3 import config as gdn_v3_config
from vllm_torchtpu.layers.core.gdn_attention import _build_v3_pool_state_plan
from vllm_torchtpu.models.vllm.qwen4_exp import config as qwen_config

# Served Qwen3.8-Flash decode rank at TP8: 2 key heads and 6 value heads of
# 128, conv dim 1280, kernel 4, FP32 state and conv, 512-row FP8 pages of
# 1 KiB rows.
SERVED = dict(
    conv_dim=1280,
    n_v=6,
    d_k=128,
    d_v=128,
    kernel_size=4,
    pool_block_tokens=512,
    qk_pair_layout=False,
    recurrent_state_dtype=jnp.float32,
    conv_state_dtype=jnp.float32,
)


def _pool(rows=512):
    return jnp.zeros((2, rows, 1, 4, 256), dtype=jnp.float8_e4m3fn)


def test_flag_off_plan_has_no_log():
    plan = _build_v3_pool_state_plan(_pool(), **SERVED)
    assert plan.log is None and plan.log_conv is None


def test_served_geometry_matches_the_kernel_contract():
    plan = _build_v3_pool_state_plan(_pool(), **SERVED, log_positions=3, n_kq=2)
    assert (plan.conv.row0, plan.conv.nrows) == (384, 16)
    assert plan.log == gdn_v3_config.StateRegion(
        kb0=0, nblocks=1, row0=400, nrows=14,
        view_dtype=jnp.float32, lane_split=2, rows_used=27,
    )
    assert plan.log_conv == gdn_v3_config.StateRegion(
        kb0=0, nblocks=1, row0=414, nrows=15,
        view_dtype=jnp.float32, lane_split=1, rows_used=15,
    )
    gdn_v3_config.validate_log_plan(
        plan,
        log_positions=3,
        num_kq_heads=2,
        num_v_heads=6,
        kq_head_dim=128,
        v_head_dim=128,
        dim_size=1280,
        block_rows=512,
        raw_lanes=256,
        raw_row_bytes=1024,
    )


def test_log_that_overruns_the_page_is_refused():
    # 12 draft positions: 54 log rows + 60 conv-input rows after row 400.
    with pytest.raises(ValueError, match=r"needs rows \[400, 514\)"):
        _build_v3_pool_state_plan(_pool(), **SERVED, log_positions=12, n_kq=2)


def test_qk_pair_conv_layout_is_refused():
    # TP4 geometry: a rank's Q rows span two 256-lane rows, so the pair layout
    # permutes the conv rows.
    geometry = dict(
        SERVED, conv_dim=2560, n_v=12, qk_pair_layout=True, pool_block_tokens=1024
    )
    with pytest.raises(ValueError, match="QK pair"):
        _build_v3_pool_state_plan(_pool(1024), **geometry, log_positions=3, n_kq=4)


def _cfg(
    *, spec=3, role="kv_consumer", tp=8, megakernel=None, dp=1, pcp=1, offload=False
):
    transfer = SimpleNamespace(
        kv_connector=qwen_config.PD_CONNECTOR, kv_role=role
    ) if role else None
    if offload:
        # The PD connector and the host tier as one TPUMultiConnector's children.
        children = [{"kv_connector": qwen_config.OFFLOAD_CONNECTOR, "kv_role": "kv_both"}]
        if transfer is not None:
            children.insert(
                0, {"kv_connector": qwen_config.PD_CONNECTOR, "kv_role": role}
            )
        transfer = SimpleNamespace(
            kv_connector=qwen_config.MULTI_CONNECTOR,
            kv_connector_extra_config={"connectors": children},
        )
    extra = {} if megakernel is None else {"megakernel": megakernel}
    return SimpleNamespace(
        speculative_config=(
            None if spec is None else SimpleNamespace(num_speculative_tokens=spec)
        ),
        kv_transfer_config=transfer,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
        ),
        additional_config=extra,
    )


@pytest.fixture
def staged_env(monkeypatch):
    monkeypatch.setenv("TPU_RAIDEN_POOL_TAGS_PER_LAYER", "1")
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)


def test_staged_mtp_decode_is_admitted(staged_env):
    qwen_config.validate_gdn_window_log(_cfg())


def test_engine_without_pd_connector_is_admitted(staged_env):
    # The offline decode bench: the decode engine prefills locally.
    qwen_config.validate_gdn_window_log(_cfg(role=None))


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        (dict(spec=None), "speculative config"),
        (dict(role="kv_producer"), "staged decode"),
        (dict(tp=1), "staged decode"),
        (dict(megakernel={"enabled": True}), "megakernel"),
        (dict(dp=2), "data parallel size 2"),
        (dict(role=None, dp=2), "data parallel size 2"),
        (dict(pcp=2), "prefill context parallel size 2"),
        (dict(offload=True), "host offload"),
        (dict(role=None, offload=True), "host offload"),
    ],
)
def test_refusals(staged_env, kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        qwen_config.validate_gdn_window_log(_cfg(**kwargs))


def test_megakernel_env_is_refused(staged_env, monkeypatch):
    monkeypatch.setenv("VLLM_TPU_MEGAKERNEL", "1")
    with pytest.raises(ValueError, match="megakernel"):
        qwen_config.validate_gdn_window_log(_cfg())


def test_unstaged_consumer_is_refused(monkeypatch):
    monkeypatch.setenv("TPU_RAIDEN_POOL_TAGS_PER_LAYER", "0")
    monkeypatch.delenv("VLLM_TPU_MEGAKERNEL", raising=False)
    with pytest.raises(ValueError, match="staged decode"):
        qwen_config.validate_gdn_window_log(_cfg())


@pytest.mark.parametrize("log, factored, refused", [("0", "1", True), ("1", "1", False), ("0", "0", False)])
def test_factored_needs_the_log(monkeypatch, log, factored, refused):
    monkeypatch.setenv("VLLM_TPU_GDN_WINDOW_LOG", log)
    monkeypatch.setenv("VLLM_TPU_GDN_WINDOW_FACTORED", factored)
    if refused:
        with pytest.raises(ValueError, match="VLLM_TPU_GDN_WINDOW_FACTORED refused"):
            qwen_config.validate_gdn_window_factored()
    else:
        qwen_config.validate_gdn_window_factored()


def test_pd_repack_writes_below_the_log():
    """A PD landing writes a GDN block's state and conv rows and nothing from
    the log's first row on: pd_staging's repack, on the served decode geometry
    (gdn_geometry at TP8 over 1 KiB rows), plain and routed (compact) staging.
    A recycled destination page keeps its previous owner's log, which the new
    owner's read offset 0 never applies."""
    import numpy as np
    import torch

    from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging as st

    text = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=text, dtype=torch.bfloat16, max_model_len=262144
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto", mamba_ssm_cache_dtype="float32"
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )
    row_bytes = 1024
    src = st.gdn_geometry(cfg, 1, row_bytes)
    dst = st.gdn_geometry(cfg, 8, row_bytes)
    plan = _build_v3_pool_state_plan(_pool(), **SERVED, log_positions=3, n_kq=2)
    # The decode rank's state and conv rows are the plan's.
    assert dst.ssm_tokens * row_bytes == SERVED["n_v"] * 128 * 128 * 4
    assert (dst.ssm_tokens, dst.conv_tokens) == (plan.conv.row0, plan.conv.nrows)

    block, dst_page, src_page = 512, 3, 1
    rank = st.RankSlice(tp_size=8, tp_rank=5, num_kv_heads=2, main_dcp=4)
    written = []

    def write(pool, rows, page, logical, valid):
        written.append(np.asarray(logical)[np.asarray(valid)])
        page = jnp.where(valid, page, pool.shape[0])
        return pool.at[page, logical % pool.shape[1]].set(rows, mode="drop")

    staging = np.random.default_rng(0).integers(
        0, 255, (16, block, 1, 4, 256), dtype=np.uint8
    )
    empty = jnp.zeros((0,), jnp.int32)
    for compact in (None, (block, 1)):
        fn = st.make_repack(
            src_block=block, dst_block=block, src_gdn=src, dst_gdn=dst,
            kv_head=rank.kv_head, v_head_start=rank.tp_rank * dst.v_heads,
            channels=st.conv_channels(src, dst, rank), write=write,
            compact_gdn=compact,
        )
        gdn_src = jnp.asarray([[src_page]] if compact else [src_page], jnp.int32)
        pool = np.full((5, block, 1, 4, 256), 0xA5, np.uint8)
        written.clear()
        out = np.asarray(
            fn(jnp.asarray(pool), jnp.asarray(staging), empty, empty, empty, empty,
               gdn_src, jnp.asarray([dst_page], jnp.int32))
        )
        rows = np.concatenate(written) - dst_page * block
        assert rows.min() == 0 and rows.max() < plan.log.row0, (compact, rows.max())
        assert rows.max() == dst.ssm_tokens + dst.conv_tokens - 1
        # Every row from the log's first row on is the page's previous bytes.
        np.testing.assert_array_equal(out[dst_page, plan.log.row0 :], 0xA5)
        assert (out[dst_page, : plan.log.row0] != 0xA5).any()
