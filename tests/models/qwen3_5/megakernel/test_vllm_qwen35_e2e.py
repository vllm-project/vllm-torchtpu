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
"""End-to-end CPU tests for the Qwen3.5 template program, metadata prep, and
speculative slots."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.megakernel as mk
import vllm_torchtpu.megakernel_runtime.driver as driver_mod
import vllm_torchtpu.megakernel_runtime.guards as guards_mod
from tests.models.qwen3_5.megakernel.test_weight_collector import (
    _jax_to_torch,
    make_synthetic_qwen35_backbone,
)
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.pool.config import with_cache
from vllm_torchtpu.megakernel.runtime import (
    DEFAULT_BUCKETS,
    DEFAULT_VERIFY_BUCKETS,
    BatchShape,
    JaxStepMetaOps,
    MKABCalibrator,
    check_post_mesh_eligibility,
    check_qsl_uniform,
    explicit_profile_settings,
    facts_from_pool_layout,
    ir_required_buckets,
    pack_step_meta,
    parse_mk_buckets,
    parse_mk_verify_buckets,
    parse_runtime_config,
    resolve_driver_profile,
    resolve_step_positions,
    select_bucket,
)
from vllm_torchtpu.megakernel_runtime.config import (  # noqa: E402
    collect_config,
    resolve_plugin_ref,
)
from vllm_torchtpu.megakernel_runtime.hooks import StepContext  # noqa: E402
from vllm_torchtpu.megakernel_runtime.plugin import (  # noqa: E402
    Bucket,
    LayoutDecline,  # noqa: E402
    RunnerView,
)
from vllm_torchtpu.megakernel_runtime.step_meta_ops import (
    TorchStepMetaOps,  # noqa: E402
)
from vllm_torchtpu.megakernel_runtime.weight_collector import (  # noqa: E402
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.models.qwen3_5 import (  # noqa: E402
    megakernel as qwen_plugin,
)
from vllm_torchtpu.models.qwen3_5.megakernel import (  # noqa: E402
    adapter as qwen_adapter,
)
from vllm_torchtpu.models.qwen3_5.megakernel import model as qwen35
from vllm_torchtpu.models.qwen3_5.megakernel import pool as qwen35_pool
from vllm_torchtpu.models.qwen3_5.megakernel.adapter import (  # noqa: E402
    Qwen35Program,
    prepare_step_inputs_eager,
)
from vllm_torchtpu.models.qwen3_5.megakernel.tests.tiny import (
    allocate_aliased_pools as _allocate_aliased_pools,
)
from vllm_torchtpu.models.qwen3_5.megakernel.tests.tiny import (
    cpu_mesh as _cpu_mesh,
)
from vllm_torchtpu.models.qwen3_5.megakernel.tests.tiny import (
    tiny_exec as _tiny_exec,
)
from vllm_torchtpu.models.qwen3_5.megakernel.tests.tiny import (
    tiny_geom as _tiny_geom,
)

_REAL_FRAMEWORK_SLOT_FACTS = Qwen35Program._framework_slot_facts


def _qwen_driver(
    *, ab: bool = False, stats_period_s: float = 60.0, **kw: Any
) -> driver_mod.ProgramDriver:
    """A `ProgramDriver` over a directly built (loaded) `Qwen35Program`."""
    prog = Qwen35Program(**kw)
    view = RunnerView(None, tp_rank=prog.tp_rank)
    return driver_mod.ProgramDriver(
        prog,
        view=view,
        tp_rank=prog.tp_rank,
        stats_period_s=stats_period_s,
        ab=ab,
        backbone=prog.backbone,
    )


def _decided(driver: driver_mod.ProgramDriver) -> tuple[int, int, int]:
    """`(num_reqs, T, bucket)` of the step `can_run` routed to the megakernel."""
    assert driver._pending is not None
    d = driver._pending[0]
    return (d.num_reqs, d.tokens_per_seq, d.bucket_t)


@pytest.fixture(autouse=True)
def _framework_agrees_on_pool_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic backbones carry no framework GDN layers or cache config: let
    the framework side of the bind-time slot check report the megakernel's
    own `PoolLayout` facts (tests of the check itself override this)."""

    def _facts(self: Any, runner: Any, linear_attn: Any, pool: Any) -> Any:
        return facts_from_pool_layout(self.spec.cache_backend.layout, self.base_exec.R)

    monkeypatch.setattr(Qwen35Program, "_framework_slot_facts", _facts)


@pytest.fixture(autouse=True)
def _ensure_driver_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    try:
        driver_mod.logger._resolve()
    except ModuleNotFoundError:
        monkeypatch.setattr(
            driver_mod.logger, "_logger", logging.getLogger(driver_mod.logger._name)
        )


def _extract_group_mamba_tensors(
    m_md: Any, fallback_1d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
    s_raw = getattr(m_md, "mamba_state_indices", None)
    c_raw = getattr(m_md, "mamba_ckpt_indices", None)
    roff = getattr(m_md, "mamba_slot_read_offsets", None)
    if s_raw is not None:
        s_1d = s_raw[:, 0] if s_raw.ndim == 2 else s_raw.reshape(-1)
    elif c_raw is not None:
        s_1d = c_raw[:, 0] if c_raw.ndim == 2 else c_raw.reshape(-1)
    else:
        s_1d = fallback_1d
    use_2d = bool(c_raw is not None and c_raw.ndim == 2)
    c_2d = (
        c_raw
        if use_2d
        else (c_raw.reshape(-1) if c_raw is not None else s_1d).unsqueeze(1)
    )
    r_1d = roff.reshape(-1) if roff is not None else fallback_1d
    return s_1d, c_2d, r_1d, roff is not None, use_2d


def prepare_step_metadata(
    attn_md: Any,
    mamba_mds: Any,
    positions: torch.Tensor | None,
    num_reqs: int,
    bucket_t: int,
    max_pages_per_seq: int,
    tokens_per_seq: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Test helper building padded `(pos_b, pt_b, ckpt_b, roff_b)` tensors."""
    if len(mamba_mds) != 3:
        raise ValueError(f"expected 3 GDN group metadata objects, got {len(mamba_mds)}")
    T, B = int(tokens_per_seq), int(bucket_t)
    if B % T != 0:
        raise ValueError(f"bucket_t={B} must be divisible by tokens_per_seq={T}")
    bt, dev, active_rows = (
        attn_md.block_tables,
        attn_md.block_tables.device,
        num_reqs * T,
    )
    fb_1d = getattr(attn_md, "seq_lens", None)
    if fb_1d is None:
        fb_1d = torch.zeros((max(1, num_reqs),), dtype=torch.int32, device=dev)
    pos_in = resolve_step_positions(
        positions,
        fb_1d,
        num_reqs=num_reqs,
        tokens_per_seq=T,
        like=bt,
        ops=TorchStepMetaOps(),
    )
    s0, c0, r0, hr0, uc0 = _extract_group_mamba_tensors(mamba_mds[0], fb_1d)
    s1, c1, r1, hr1, uc1 = _extract_group_mamba_tensors(mamba_mds[1], fb_1d)
    s2, c2, r2, hr2, uc2 = _extract_group_mamba_tensors(mamba_mds[2], fb_1d)
    if bt.ndim == 1:
        n_rows_bt, m_pg = int(s0.shape[0]), int(max_pages_per_seq)
        if n_rows_bt > 0 and bt.numel() % n_rows_bt == 0:
            bt = bt.reshape(n_rows_bt, bt.numel() // n_rows_bt)
        elif m_pg > 0 and bt.numel() % m_pg == 0:
            bt = bt.reshape(-1, m_pg)
        else:
            raise ValueError(
                f"1-D block_tables of length {bt.numel()} is not divisible by "
                f"s0.shape[0]={n_rows_bt} or max_pages_per_seq={max_pages_per_seq}"
            )
    dummy_ids = torch.ones((active_rows,), dtype=torch.int32, device=dev)
    _, pos_b, pt_b, ckpt_b, roff_b = prepare_step_inputs_eager(
        dummy_ids,
        pos_in,
        bt[:num_reqs],
        s0[:num_reqs],
        c0[:num_reqs],
        r0,
        s1[:num_reqs],
        c1[:num_reqs],
        r1,
        s2[:num_reqs],
        c2[:num_reqs],
        r2,
        B,
        T,
        int(max_pages_per_seq),
        bool(hr0 or hr1 or hr2),
        bool(uc0 or uc1 or uc2),
    )
    return pos_b, pt_b, ckpt_b, roff_b


def _torch_to_jax(t: torch.Tensor, sharding: NamedSharding) -> jax.Array:
    if t.dtype == torch.float8_e4m3fn:
        arr = jnp.asarray(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    elif t.dtype == torch.bfloat16:
        arr = jnp.asarray(t.view(torch.uint16).numpy()).view(jnp.bfloat16)
    else:
        arr = jnp.asarray(t.numpy())
    return jax.device_put(arr, sharding)


def _make_backbone_from_logical_weights(
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    w_logical: dict[str, Any],
    tp_rank: int = 0,
) -> SimpleNamespace:
    """Build a synthetic PyTorch backbone whose weights match `w_logical`."""
    gdn_lay = qwen35.GDNModule.lin_in_layout(g, ex)
    qkvz_dim = gdn_lay.QKV + gdn_lay.Z
    layers: list[Any] = []
    for l_idx in range(g.L):
        is_att = (l_idx % g.att_every) == (g.att_every - 1)
        ltype = "full_attention" if is_att else "linear_attention"
        if is_att:
            a_idx = l_idx // g.att_every
            qkv_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["att_in"][tp_rank, a_idx]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["att_in_s"][tp_rank, a_idx]),
                    requires_grad=False,
                ),
            )
            setattr(qkv_proj, WEIGHT_FLIPPED_ATTR, True)
            o_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["att_out"][tp_rank, a_idx]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["att_out_s"][tp_rank, a_idx]),
                    requires_grad=False,
                ),
            )
            setattr(o_proj, WEIGHT_FLIPPED_ATTR, True)
            self_attn = SimpleNamespace(
                qkv_proj=qkv_proj,
                o_proj=o_proj,
                q_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["qn"][tp_rank, a_idx]),
                        requires_grad=False,
                    )
                ),
                k_norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["kn"][tp_rank, a_idx]),
                        requires_grad=False,
                    )
                ),
                attn=SimpleNamespace(
                    layer_name=f"model.layers.{l_idx}.self_attn.attn", kv_cache=[]
                ),
            )
            linear_attn = None
        else:
            o_gdn = l_idx - (l_idx + 1) // g.att_every
            w_lin_full = _jax_to_torch(w_logical["lin_in"][tp_rank, o_gdn])
            s_lin_full = _jax_to_torch(w_logical["lin_in_s"][tp_rank, o_gdn])
            w_qkvz = w_lin_full[:, :qkvz_dim].contiguous()
            s_qkvz = s_lin_full[:qkvz_dim].contiguous()
            w_b = w_lin_full[:, gdn_lay.BA : gdn_lay.BA + gdn_lay.vh]
            w_a = w_lin_full[:, gdn_lay.A_OFF : gdn_lay.A_OFF + gdn_lay.vh]
            s_b = s_lin_full[gdn_lay.BA : gdn_lay.BA + gdn_lay.vh]
            s_a = s_lin_full[gdn_lay.A_OFF : gdn_lay.A_OFF + gdn_lay.vh]
            if "_ba_bf16" in w_logical:
                w_ba = _jax_to_torch(w_logical["_ba_bf16"][tp_rank, o_gdn]).contiguous()
            else:
                w_ba_kn = torch.cat(
                    [w_b.float() * s_b.unsqueeze(0), w_a.float() * s_a.unsqueeze(0)],
                    dim=1,
                ).to(torch.bfloat16)
                w_ba = w_ba_kn.transpose(0, 1).contiguous()

            in_proj_qkvz = SimpleNamespace(
                weight=torch.nn.Parameter(w_qkvz, requires_grad=False),
                weight_scale=torch.nn.Parameter(s_qkvz, requires_grad=False),
            )
            setattr(in_proj_qkvz, WEIGHT_FLIPPED_ATTR, True)
            in_proj_ba = SimpleNamespace(
                weight=torch.nn.Parameter(w_ba, requires_grad=False),
            )
            out_proj = SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["lin_out"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                weight_scale=torch.nn.Parameter(
                    _jax_to_torch(w_logical["lin_out_s"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
            )
            setattr(out_proj, WEIGHT_FLIPPED_ATTR, True)
            conv_4_qkv = _jax_to_torch(w_logical["conv"][tp_rank, o_gdn]).reshape(
                4, gdn_lay.QKV
            )
            conv_w = conv_4_qkv.transpose(0, 1).unsqueeze(1).contiguous()
            linear_attn = SimpleNamespace(
                prefix=f"model.layers.{l_idx}.linear_attn",
                in_proj_qkvz=in_proj_qkvz,
                in_proj_ba=in_proj_ba,
                out_proj=out_proj,
                conv1d=SimpleNamespace(
                    weight=torch.nn.Parameter(conv_w, requires_grad=False)
                ),
                A_log=torch.nn.Parameter(
                    _jax_to_torch(w_logical["alog"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                dt_bias=torch.nn.Parameter(
                    _jax_to_torch(w_logical["dtb"][tp_rank, o_gdn]),
                    requires_grad=False,
                ),
                norm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["gnorm"][tp_rank, o_gdn]),
                        requires_grad=False,
                    )
                ),
                kv_cache=[],
            )
            self_attn = None

        experts = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["e13"][tp_rank, l_idx]), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["e2"][tp_rank, l_idx]), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["e13_s"][tp_rank, l_idx]), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["e2_s"][tp_rank, l_idx]), requires_grad=False
            ),
        )
        gu_proj = SimpleNamespace(
            weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh13"][tp_rank, l_idx]), requires_grad=False
            ),
            weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh13_s"][tp_rank, l_idx]), requires_grad=False
            ),
        )
        setattr(gu_proj, WEIGHT_FLIPPED_ATTR, True)
        down_proj = SimpleNamespace(
            weight=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh2"][tp_rank, l_idx]), requires_grad=False
            ),
            weight_scale=torch.nn.Parameter(
                _jax_to_torch(w_logical["sh2_s"][tp_rank, l_idx]), requires_grad=False
            ),
        )
        setattr(down_proj, WEIGHT_FLIPPED_ATTR, True)
        mlp = SimpleNamespace(
            experts=experts,
            gate=SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["router"][l_idx])
                    .transpose(0, 1)
                    .contiguous(),
                    requires_grad=False,
                )
            ),
            shared_expert=SimpleNamespace(gate_up_proj=gu_proj, down_proj=down_proj),
            shared_expert_gate=SimpleNamespace(
                weight=torch.nn.Parameter(
                    _jax_to_torch(w_logical["sh_gate"][tp_rank, l_idx]),
                    requires_grad=False,
                )
            ),
        )
        layers.append(
            SimpleNamespace(
                layer_type=ltype,
                input_layernorm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["in_norm"][l_idx]), requires_grad=False
                    )
                ),
                post_attention_layernorm=SimpleNamespace(
                    weight=torch.nn.Parameter(
                        _jax_to_torch(w_logical["post_norm"][l_idx]),
                        requires_grad=False,
                    )
                ),
                linear_attn=linear_attn,
                self_attn=self_attn,
                mlp=mlp,
            )
        )
    embed = torch.nn.Embedding(64, g.H, dtype=torch.bfloat16)
    norm = SimpleNamespace(
        weight=torch.nn.Parameter(
            _jax_to_torch(w_logical["final_norm"]), requires_grad=False
        )
    )
    return SimpleNamespace(embed_tokens=embed, layers=layers, norm=norm)


def test_bucket_parsing_and_selection() -> None:
    """Verify decode/verify bucket parsing, sorting, config parsing, and selection."""
    assert parse_mk_buckets("16, 4, 8, 4") == (4, 8, 16)
    assert parse_mk_buckets("16:4:8") == (4, 8, 16)
    assert parse_mk_buckets("0") == ()
    assert parse_mk_buckets("off") == ()
    with pytest.raises(ValueError, match="invalid decode buckets"):
        parse_mk_buckets("128")

    assert parse_mk_verify_buckets("16, 4, 8") == (4, 8, 16)
    assert parse_mk_verify_buckets("16:4:8") == (4, 8, 16)
    assert parse_mk_verify_buckets("none") == ()
    with pytest.raises(ValueError, match="invalid verify_buckets"):
        parse_mk_verify_buckets("6")

    # Single VLLM_TPU_MEGAKERNEL string forms + additional_config["megakernel"]
    assert not parse_runtime_config("")["enabled"]
    assert not parse_runtime_config("0")["enabled"]
    cfg_on = parse_runtime_config("1")
    assert cfg_on["enabled"] and cfg_on["mode"] == ""
    cfg_tp = parse_runtime_config("tp")
    assert cfg_tp["enabled"] and cfg_tp["mode"] == "tp"
    cfg_kv = parse_runtime_config(
        "mode=tp,buckets=8:16:32,verify_buckets=4:8,ab=1,stats_period_s=12.5,compile_hlo_check=1,fused_pool_io=0,fused_pool_io_max_b=16,fused_pool_io_min_b_hi=64,pool_gather=0"
    )
    assert cfg_kv["enabled"]
    assert cfg_kv["mode"] == "tp"
    assert cfg_kv["buckets"] == (8, 16, 32)
    assert cfg_kv["verify_buckets"] == (4, 8)
    assert cfg_kv["ab"] is True
    assert cfg_kv["stats_period_s"] == pytest.approx(12.5)
    assert cfg_kv["compile_hlo_check"] is True
    assert cfg_kv["fused_pool_io"] is False
    assert cfg_kv["fused_pool_io_max_b"] == 16
    assert cfg_kv["fused_pool_io_min_b_hi"] == 64
    assert cfg_kv["pool_gather"] is False

    cfg_json = parse_runtime_config(
        '{"mode": "ep", "buckets": [4, 8], "verify_buckets": [4], '
        '"ab": false, "fused_pool_io": "auto", "pool_gather": true}'
    )
    assert cfg_json["enabled"]
    assert cfg_json["mode"] == "ep"
    assert cfg_json["buckets"] == (4, 8)
    assert cfg_json["verify_buckets"] == (4,)
    assert cfg_json["fused_pool_io"] == "auto"
    assert cfg_json["pool_gather"] is True

    # No named profile: the driver runs the model's default profile; the
    # profile-owned keys keep their split pool I/O defaults.
    cfg_def = parse_runtime_config("1")
    assert cfg_def["profile"] is None
    assert cfg_def["fused_pool_io"] is False
    assert cfg_def["pool_gather"] is False
    # An empty value means "off", not "auto"
    cfg_empty = parse_runtime_config("fused_pool_io=,pool_gather=")
    assert cfg_empty["fused_pool_io"] is False
    assert cfg_empty["pool_gather"] is False
    # Named profile selection via VLLM_TPU_MEGAKERNEL; explicit keys win
    cfg_c30 = parse_runtime_config("profile=qwen35-candidate30")
    assert cfg_c30["enabled"] and cfg_c30["profile"] == "qwen35-candidate30"
    assert cfg_c30["pool_gather"] == "auto"
    assert cfg_c30["fused_pool_io"] is False
    assert cfg_c30["buckets"] == cfg_def["buckets"]
    cfg_c30_ov = parse_runtime_config(
        "profile=qwen35-candidate30,pool_gather=0,fused_pool_io=auto"
    )
    assert cfg_c30_ov["pool_gather"] is False
    assert cfg_c30_ov["fused_pool_io"] == "auto"
    with pytest.raises(ValueError, match="unknown megakernel profile"):
        parse_runtime_config("profile=does-not-exist")

    fake_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={
                "megakernel": {"mode": "tp", "buckets": [16, 32], "ab": True}
            }
        )
    )
    qwen_hf = SimpleNamespace(architectures=["Qwen3_5MoeForCausalLM"])

    def _resolved(runner: Any) -> tuple[Any, dict[str, Any]]:
        cfg = collect_config("", runner.vllm_config.additional_config)
        _, opts = resolve_plugin_ref(cfg, qwen_hf)
        return cfg, qwen_adapter.parse_options(opts)

    rt_cfg, cfg_runner = _resolved(fake_runner)
    assert rt_cfg.enabled and rt_cfg.ab is True
    assert cfg_runner["mode"] == "tp"
    assert cfg_runner["buckets"] == (16, 32)
    assert cfg_runner["profile"] is None

    # Named profile selection via `additional_config`
    fake_runner_c30 = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"megakernel": {"profile": "qwen35-candidate30"}}
        )
    )
    rt_c30, cfg_runner_c30 = _resolved(fake_runner_c30)
    assert rt_c30.enabled
    assert cfg_runner_c30["profile"] == "qwen35-candidate30"
    assert cfg_runner_c30["pool_gather"] == "auto"

    import vllm_torchtpu.megakernel_runtime as mk_rt

    assert mk_rt._is_megakernel_requested(fake_runner)
    fake_runner_disabled = SimpleNamespace(
        vllm_config=SimpleNamespace(
            additional_config={"megakernel": {"enabled": False, "mode": "tp"}}
        )
    )
    assert not mk_rt._is_megakernel_requested(fake_runner_disabled)
    assert mk_rt.maybe_create(fake_runner_disabled) is None

    buckets = (2, 4, 8)
    assert select_bucket(1, buckets) == 2
    assert select_bucket(4, buckets) == 4
    assert select_bucket(5, buckets) == 8
    with pytest.raises(ValueError, match="exceeds max bucket"):
        select_bucket(9, buckets)


def test_driver_passes_only_explicit_profile_settings() -> None:
    """Profile-owned keys the user didn't set reach the driver as None, so the
    effective profile (named, else the model default) decides them."""
    owned = (
        "buckets",
        "verify_buckets",
        "fused_pool_io",
        "fused_pool_io_max_b",
        "fused_pool_io_min_b_hi",
        "pool_gather",
    )
    for raw in ("1", "profile=qwen35-candidate30"):
        got = explicit_profile_settings(parse_runtime_config(raw))
        assert got == dict.fromkeys(owned), raw
    got = explicit_profile_settings(
        parse_runtime_config(
            "profile=qwen35-candidate30,buckets=8:16,pool_gather=0,"
            "fused_pool_io_max_b=16"
        )
    )
    assert got == {
        **dict.fromkeys(owned),
        "buckets": (8, 16),
        "pool_gather": False,
        "fused_pool_io_max_b": 16,
    }


def test_resolve_driver_profile_and_fused_ir_rule() -> None:
    """Profile resolution (overrides, model check) and the eager fused IR rule."""
    # No named profile: the model's default profile (Qwen3.5: qwen35-baseline,
    # other models: the library `default`).
    d = resolve_driver_profile(None, model_name="qwen3_5")
    assert d.name == "qwen35-baseline" and d.model_default
    assert d.pool_gather is False and d.fused_pool_io is False
    generic = resolve_driver_profile(None, model_name="x")
    assert generic.name == "default"
    c30 = resolve_driver_profile("qwen35-candidate30", model_name="qwen3_5")
    assert c30.pool_gather is None and c30.numerics == "approx"
    c30_ov = resolve_driver_profile(
        "qwen35-candidate30",
        model_name="qwen3_5",
        fused_pool_io="auto",
        pool_gather="0",
        fused_pool_io_max_b=16,
    )
    assert c30_ov.fused_pool_io == "auto" and c30_ov.pool_gather is False
    assert c30_ov.fused_pool_io_max_b == 16
    assert c30_ov.tiles is c30.tiles
    # Real overrides re-label the profile; values equal to the profile's don't.
    assert (c30_ov.name, c30_ov.numerics) == ("qwen35-candidate30+overrides", "approx")
    d_fused = resolve_driver_profile(None, model_name="qwen3_5", fused_pool_io="auto")
    assert (d_fused.name, d_fused.numerics) == ("qwen35-baseline+overrides", "approx")
    assert resolve_driver_profile(None, model_name="qwen3_5", fused_pool_io="0") is d
    assert resolve_driver_profile(None, model_name="x", fused_pool_io="0") is generic
    fused = resolve_driver_profile("qwen35-candidate30-fused", model_name="qwen3_5")
    assert (fused.name, fused.fused_pool_io) == ("qwen35-candidate30-fused", "auto")
    assert (
        resolve_driver_profile(None, model_name="x", fused_pool_io="").fused_pool_io
        is False
    )
    assert (
        resolve_driver_profile(None, model_name="x", pool_gather="auto").pool_gather
        is None
    )
    with pytest.raises(ValueError, match="targets model"):
        resolve_driver_profile("qwen35-candidate30", model_name="other_model")
    with pytest.raises(ValueError, match="unknown megakernel profile"):
        resolve_driver_profile("nope", model_name="qwen3_5")

    # Eager warmup must still IR-check every fused bucket (donation is required).
    fused_io = {(1, 8): False, (1, 16): True, (4, 8): False}
    warmed = [(1, 8), (1, 16), (4, 8)]
    assert ir_required_buckets(warmed, fused_io, is_eager=False) == warmed
    assert ir_required_buckets(warmed, fused_io, is_eager=True) == [(1, 16)]


def test_ab_calibrator() -> None:
    """Verify `MKABCalibrator` alternation and latency summary formatting."""
    off = MKABCalibrator(enabled=False, period_s=0.0)
    assert off.choose("mk_t1_b2", True) == (True, "mk_t1_b2", "mk")
    assert off.choose("stock_mixed", False) == (False, "stock_mixed", "stock")
    assert off.record(1.5, padded_tokens=2) is None

    ab = MKABCalibrator(enabled=True, period_s=0.0)
    assert ab.choose("mk_t1_b2", True) == (True, "mk_t1_b2_ab_mk", "mk")
    res1 = ab.record(1.2, padded_tokens=2)
    assert res1 is not None

    assert ab.choose("mk_t1_b2", True) == (False, "mk_t1_b2_ab_stock", "stock")
    res2 = ab.record(2.2, padded_tokens=2)
    assert res2 is not None
    _, _, cum_json = res2
    parsed = json.loads(cum_json)
    entry = parsed["mk_t1_b2_p2"]
    assert entry["mk"][0] == 1
    assert entry["stock"][0] == 1
    assert entry["d_mean_ms"] == pytest.approx(-1.0)


def test_prepare_step_metadata_decode_and_verify() -> None:
    """Verify `prepare_step_metadata` for `T=1` decode, `T=4` verify, and `roff`."""
    attn_md = SimpleNamespace(
        block_tables=torch.tensor([[16, 17], [18, 19], [0, 0]], dtype=torch.int32),
        seq_lens=torch.tensor([10, 25, 0], dtype=torch.int32),
    )
    # 2 active sequences: base slots are [1, 5] and 4-slot checkpoint windows are
    # [[1, 2, 3, 4], [5, 6, 7, 8]]. In TPUModelRunner, mamba_slot_read_offsets is
    # indexed by the group-0 base slot id (1 and 5), with roff[1]=2 and roff[5]=1.
    roff_slot_indexed = torch.zeros((16,), dtype=torch.int32)
    roff_slot_indexed[1] = 2
    roff_slot_indexed[5] = 1
    mamba_mds = [
        SimpleNamespace(
            mamba_state_indices=torch.tensor([1, 5, 0], dtype=torch.int32),
            mamba_ckpt_indices=torch.tensor(
                [[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]], dtype=torch.int32
            ),
            mamba_slot_read_offsets=roff_slot_indexed,
        )
        for _ in range(3)
    ]

    # 1. T=1 decode with speculative read offsets:
    #    ckpt_1 has shape (3, 4, 2) where col 0 is read_slot c[s, roff[s]] ->
    #    [3, 6, 0, 0] and col 1 is write_slot s (base slot) -> [1, 5, 0, 0].
    pos_1, pt_1, ckpt_1, roff_1 = prepare_step_metadata(
        attn_md=attn_md,
        mamba_mds=mamba_mds,
        positions=torch.tensor([9, 24], dtype=torch.int32),
        num_reqs=2,
        bucket_t=4,
        max_pages_per_seq=2,
        tokens_per_seq=1,
    )
    assert pos_1.shape == (4,)
    assert pos_1.tolist() == [9, 24, 0, 0]
    assert pt_1.shape == (4, 2)
    assert pt_1[:2].tolist() == [[16, 17], [18, 19]]
    assert ckpt_1.shape == (3, 4, 2)
    assert ckpt_1[0, :, 0].tolist() == [3, 6, 0, 0]
    assert ckpt_1[0, :, 1].tolist() == [1, 5, 0, 0]
    assert roff_1.shape == (3, 4)
    assert roff_1[0].tolist() == [0, 0, 0, 0]

    # 2. T=4 MTP verify with speculative read offsets: preserves 4-col ckpt & roff
    pos_4, pt_4, ckpt_4, roff_4 = prepare_step_metadata(
        attn_md=attn_md,
        mamba_mds=mamba_mds,
        positions=torch.tensor([9, 10, 11, 12, 24, 25, 26, 27], dtype=torch.int32),
        num_reqs=2,
        bucket_t=8,
        max_pages_per_seq=2,
        tokens_per_seq=4,
    )
    assert pos_4.shape == (8,)
    assert pos_4.tolist() == [9, 10, 11, 12, 24, 25, 26, 27]
    assert pt_4.shape == (2, 2)
    assert ckpt_4.shape == (3, 2, 4)
    assert ckpt_4[0].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert roff_4.shape == (3, 2)
    assert roff_4[0].tolist() == [2, 1]

    # 3. `roff` is always slot-indexed even when `roff.numel() <= s_raw.shape[0]`.
    s_padded_32 = torch.zeros((32,), dtype=torch.int32)
    s_padded_32[0] = 1
    s_padded_32[1] = 5
    c_padded_32 = torch.zeros((32, 4), dtype=torch.int32)
    c_padded_32[0] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    c_padded_32[1] = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    roff_small_8 = torch.zeros((8,), dtype=torch.int32)
    roff_small_8[1] = 2
    roff_small_8[5] = 1
    _, _, _, ckpt_small, roff_small = prepare_step_inputs_eager(
        torch.ones((8,), dtype=torch.int32),
        torch.zeros((8,), dtype=torch.int32),
        attn_md.block_tables,
        s_padded_32,
        c_padded_32,
        roff_small_8,
        s_padded_32,
        c_padded_32,
        roff_small_8,
        s_padded_32,
        c_padded_32,
        roff_small_8,
        bucket_t=8,
        tokens_per_seq=4,
        max_pages_per_seq=2,
        has_roff=True,
        use_ckpt_2d=True,
    )
    assert ckpt_small[0].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert roff_small[0].tolist() == [2, 1]

    # 4. `positions=None` (T=1 and T=4), short `positions` (`numel() < num_reqs * T`),
    #    and 2-D M-RoPE `positions` (`(3, 8)`) synthesize / slice positions properly.
    pos_none_t1, _, _, _ = prepare_step_metadata(
        attn_md=attn_md,
        mamba_mds=mamba_mds,
        positions=None,
        num_reqs=2,
        bucket_t=4,
        max_pages_per_seq=2,
        tokens_per_seq=1,
    )
    assert pos_none_t1.tolist() == [9, 24, 0, 0]

    pos_none_t4, _, _, _ = prepare_step_metadata(
        attn_md=attn_md,
        mamba_mds=mamba_mds,
        positions=None,
        num_reqs=2,
        bucket_t=8,
        max_pages_per_seq=2,
        tokens_per_seq=4,
    )
    assert pos_none_t4.tolist() == [7, 8, 9, 10, 22, 23, 24, 25]

    pos_short_t4, _, _, _ = prepare_step_metadata(
        attn_md=attn_md,
        mamba_mds=mamba_mds,
        positions=torch.tensor([1, 2], dtype=torch.int32),
        num_reqs=2,
        bucket_t=8,
        max_pages_per_seq=2,
        tokens_per_seq=4,
    )
    assert pos_short_t4.tolist() == [7, 8, 9, 10, 22, 23, 24, 25]

    mrope_row = torch.tensor([9, 10, 11, 12, 24, 25, 26, 27], dtype=torch.int32)
    pos_mrope_t4, _, _, _ = prepare_step_metadata(
        attn_md=attn_md,
        mamba_mds=mamba_mds,
        positions=torch.stack([mrope_row, mrope_row, mrope_row], dim=0),
        num_reqs=2,
        bucket_t=8,
        max_pages_per_seq=2,
        tokens_per_seq=4,
    )
    assert pos_mrope_t4.tolist() == [9, 10, 11, 12, 24, 25, 26, 27]


def test_block_tables_1d_reshape_most_model_len() -> None:
    """Verify 1-D `block_tables` reshapes by `s0.shape[0]` (`target_num_reqs`) first."""
    target_num_reqs = 16
    target_num_blocks = 2
    max_pages_per_seq = 4
    bt_2d = torch.arange(
        100,
        100 + target_num_reqs * target_num_blocks,
        dtype=torch.int32,
    ).reshape(target_num_reqs, target_num_blocks)
    bt_1d = bt_2d.reshape(-1)
    assert bt_1d.numel() == 32
    assert bt_1d.numel() % max_pages_per_seq == 0

    s_padded = torch.zeros((target_num_reqs,), dtype=torch.int32)
    s_padded[0] = 1
    s_padded[1] = 2
    zero_roff = torch.zeros((1,), dtype=torch.int32)

    _, _, pt_b, _, _ = prepare_step_inputs_eager(
        torch.tensor([11, 22], dtype=torch.int32),
        torch.tensor([5, 9], dtype=torch.int32),
        bt_1d,
        s_padded,
        s_padded,
        zero_roff,
        s_padded,
        s_padded,
        zero_roff,
        s_padded,
        s_padded,
        zero_roff,
        bucket_t=4,
        tokens_per_seq=1,
        max_pages_per_seq=max_pages_per_seq,
        has_roff=False,
        use_ckpt_2d=False,
    )
    assert pt_b.shape == (4, max_pages_per_seq)
    assert pt_b[0].tolist() == [100, 101, 0, 0]
    assert pt_b[1].tolist() == [102, 103, 0, 0]


def _ref_prepare_step_inputs_eager(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    block_tables: torch.Tensor,
    s0: torch.Tensor,
    c0: torch.Tensor,
    r0: torch.Tensor,
    s1: torch.Tensor,
    c1: torch.Tensor,
    r1: torch.Tensor,
    s2: torch.Tensor,
    c2: torch.Tensor,
    r2: torch.Tensor,
    bucket_t: int,
    tokens_per_seq: int,
    max_pages_per_seq: int,
    has_roff: bool,
    use_ckpt_2d: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Reference pre-B6 `prepare_step_inputs_eager` implementation for parity tests."""

    def _sp_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
        x1 = x.reshape(-1)
        n = int(x1.shape[0])
        if n >= target_len:
            return x1[:target_len]
        return torch.nn.functional.pad(x1, (0, target_len - n), value=0)

    def _sp_2d(x: torch.Tensor, target_rows: int, target_cols: int) -> torch.Tensor:
        r, c = int(x.shape[0]), int(x.shape[1])
        out = x[: min(r, target_rows), : min(c, target_cols)]
        pad_r = max(0, target_rows - r)
        pad_c = max(0, target_cols - c)
        if pad_r > 0 or pad_c > 0:
            out = torch.nn.functional.pad(out, (0, pad_c, 0, pad_r), value=0)
        return out

    T = tokens_per_seq
    B = bucket_t
    nseq_bkt = B // T

    bt = block_tables
    if bt.ndim == 1:
        n_rows_bt = int(s0.shape[0])
        if n_rows_bt > 0 and bt.numel() % n_rows_bt == 0:
            bt = bt.reshape(n_rows_bt, bt.numel() // n_rows_bt)
        elif max_pages_per_seq > 0 and bt.numel() % max_pages_per_seq == 0:
            bt = bt.reshape(-1, max_pages_per_seq)
        else:
            raise ValueError("invalid 1-D block_tables")
    pt_b = _sp_2d(bt, nseq_bkt, max_pages_per_seq).to(torch.int32)

    ckpt_rows: list[torch.Tensor] = []
    roff_rows: list[torch.Tensor] = []
    for s_raw, c_raw, r_raw in ((s0, c0, r0), (s1, c1, r1), (s2, c2, r2)):
        s_b = (
            _sp_1d(s_raw[:, 0], nseq_bkt).to(torch.int32)
            if s_raw.ndim == 2
            else _sp_1d(s_raw, nseq_bkt).to(torch.int32)
        )
        if has_roff:
            s_idx = s_b.to(torch.long).clamp(min=0, max=r_raw.numel() - 1)
            roff_g = r_raw.reshape(-1)[s_idx].to(torch.int32)
        else:
            roff_g = torch.zeros_like(s_b)

        if T > 1:
            if use_ckpt_2d and c_raw.ndim == 2 and c_raw.shape[1] >= T:
                ckpt_g = _sp_2d(c_raw, nseq_bkt, T).to(torch.int32)
            else:
                src_1d = c_raw[:, 0] if c_raw.ndim == 2 else c_raw.reshape(-1)
                ckpt_1d = _sp_1d(src_1d, nseq_bkt).to(torch.int32)
                ckpt_g = ckpt_1d.unsqueeze(1).expand(nseq_bkt, T).contiguous()
                roff_g = torch.where(roff_g < 0, roff_g, torch.zeros_like(roff_g))
        else:
            if use_ckpt_2d and has_roff and c_raw.ndim == 2 and c_raw.shape[1] > 1:
                c_b = _sp_2d(c_raw, nseq_bkt, int(c_raw.shape[1])).to(torch.int32)
                r_idx = roff_g.to(torch.long).clamp(min=0, max=c_raw.shape[1] - 1)
                read_col = c_b.gather(1, r_idx.unsqueeze(1))
            else:
                read_col = s_b.unsqueeze(1)
            write_col = s_b.unsqueeze(1)
            ckpt_g = torch.cat([read_col, write_col], dim=1)
            roff_g = torch.where(roff_g < 0, roff_g, torch.zeros_like(roff_g))

        active_g = (s_b > 0) & (ckpt_g[:, 0] > 0)
        ckpt_g = torch.where(active_g.unsqueeze(1), ckpt_g, torch.zeros_like(ckpt_g))
        roff_g = torch.where(active_g, roff_g, torch.zeros_like(roff_g))
        ckpt_rows.append(ckpt_g)
        roff_rows.append(roff_g)

    ckpt_b = torch.stack(ckpt_rows, dim=0)
    roff_b = torch.stack(roff_rows, dim=0)
    active_seq = ckpt_b[0, :, 0] > 0
    active_tok = active_seq.unsqueeze(1).expand(nseq_bkt, T).reshape(B)
    zero_b = torch.zeros((B,), dtype=torch.int32, device=input_ids.device)
    ids_b = torch.where(active_tok, _sp_1d(input_ids, B).to(torch.int32), zero_b)
    pos_1d = positions[0] if positions.ndim == 2 else positions
    pos_b = torch.where(active_tok, _sp_1d(pos_1d, B).to(torch.int32), zero_b)
    return ids_b, pos_b, pt_b, ckpt_b, roff_b


def test_step_meta_bitwise_parity_base_vs_head_and_jax() -> None:
    """Verify bitwise parity of `prepare_step_inputs_eager` vs `pack_step_meta`."""
    torch_ops = TorchStepMetaOps()
    jax_ops = JaxStepMetaOps()
    max_pages_per_seq = 4
    num_req_rows = 6

    s0 = torch.tensor([1, 0, 5, 8, 0, 12], dtype=torch.int32)
    s1 = torch.tensor([2, 0, 6, 9, 0, 13], dtype=torch.int32)
    s2 = torch.tensor([3, 0, 7, 10, 0, 14], dtype=torch.int32)
    c0_2d = torch.tensor(
        [
            [1, 2, 3, 4],
            [0, 0, 0, 0],
            [5, 6, 7, 8],
            [0, 9, 10, 11],
            [15, 16, 17, 18],
            [12, 13, 14, 15],
        ],
        dtype=torch.int32,
    )
    c1_2d = c0_2d + 20
    c1_2d[1] = 0
    c1_2d[3, 0] = 0
    c2_2d = c0_2d + 40
    c2_2d[1] = 0
    c2_2d[3, 0] = 0

    roff_pos = torch.zeros((32,), dtype=torch.int32)
    roff_pos[1] = 2
    roff_pos[5] = 1
    roff_pos[8] = 3
    roff_pos[12] = 0

    roff_neg = roff_pos.clone()
    roff_neg[1] = -1
    roff_neg[5] = 2
    roff_neg[8] = -2
    roff_neg[12] = -1

    bt_2d = torch.arange(
        10, 10 + num_req_rows * max_pages_per_seq, dtype=torch.int32
    ).reshape(num_req_rows, max_pages_per_seq)
    bt_1d_req_blocks = torch.arange(50, 50 + num_req_rows * 2, dtype=torch.int32)
    bt_1d_max_pages = torch.arange(80, 80 + 5 * max_pages_per_seq, dtype=torch.int32)
    s0_j, s1_j, s2_j = (
        jnp.asarray(s0.numpy()),
        jnp.asarray(s1.numpy()),
        jnp.asarray(s2.numpy()),
    )
    c0_2d_j, c1_2d_j, c2_2d_j = (
        jnp.asarray(c0_2d.numpy()),
        jnp.asarray(c1_2d.numpy()),
        jnp.asarray(c2_2d.numpy()),
    )
    r_zero = torch.zeros((1,), dtype=torch.int32)
    r_zero_j = jnp.zeros((1,), dtype=jnp.int32)
    roff_pos_j = jnp.asarray(roff_pos.numpy())
    roff_neg_j = jnp.asarray(roff_neg.numpy())
    bt_pairs = (
        (bt_2d, jnp.asarray(bt_2d.numpy())),
        (bt_1d_req_blocks, jnp.asarray(bt_1d_req_blocks.numpy())),
        (bt_1d_max_pages, jnp.asarray(bt_1d_max_pages.numpy())),
    )

    for T in (1, 4):
        for bucket_t in (4, 8, 16):
            raw_tok_len = 12
            ids_in = torch.arange(100, 100 + raw_tok_len, dtype=torch.int32)
            ids_j = jnp.asarray(ids_in.numpy())
            pos_1d = torch.arange(200, 200 + raw_tok_len, dtype=torch.int32)
            pos_mrope = torch.stack([pos_1d, pos_1d, pos_1d], dim=0)
            for pos_in, pos_j in (
                (pos_1d, jnp.asarray(pos_1d.numpy())),
                (pos_mrope, jnp.asarray(pos_mrope.numpy())),
            ):
                for bt_in, bt_j in bt_pairs:
                    for has_roff, r_in, r_j in (
                        (False, r_zero, r_zero_j),
                        (True, roff_pos, roff_pos_j),
                        (True, roff_neg, roff_neg_j),
                    ):
                        for use_ckpt_2d, c_set, c_set_j in (
                            (False, (s0, s1, s2), (s0_j, s1_j, s2_j)),
                            (True, (c0_2d, c1_2d, c2_2d), (c0_2d_j, c1_2d_j, c2_2d_j)),
                        ):
                            c0_in, c1_in, c2_in = c_set
                            c0_j, c1_j, c2_j = c_set_j
                            if T > 1 and not use_ckpt_2d:
                                with pytest.raises(
                                    ValueError,
                                    match="requires 2-D ckpt_indices with >= T=",
                                ):
                                    prepare_step_inputs_eager(
                                        ids_in,
                                        pos_in,
                                        bt_in,
                                        s0,
                                        c0_in,
                                        r_in,
                                        s1,
                                        c1_in,
                                        r_in,
                                        s2,
                                        c2_in,
                                        r_in,
                                        bucket_t,
                                        T,
                                        max_pages_per_seq,
                                        has_roff,
                                        use_ckpt_2d,
                                    )
                                continue
                            ref_out = _ref_prepare_step_inputs_eager(
                                ids_in,
                                pos_in,
                                bt_in,
                                s0,
                                c0_in,
                                r_in,
                                s1,
                                c1_in,
                                r_in,
                                s2,
                                c2_in,
                                r_in,
                                bucket_t,
                                T,
                                max_pages_per_seq,
                                has_roff,
                                use_ckpt_2d,
                            )
                            head_out = prepare_step_inputs_eager(
                                ids_in,
                                pos_in,
                                bt_in,
                                s0,
                                c0_in,
                                r_in,
                                s1,
                                c1_in,
                                r_in,
                                s2,
                                c2_in,
                                r_in,
                                bucket_t,
                                T,
                                max_pages_per_seq,
                                has_roff,
                                use_ckpt_2d,
                            )
                            meta_torch = pack_step_meta(
                                ids_in,
                                pos_in,
                                bt_in,
                                (
                                    (s0, c0_in, r_in),
                                    (s1, c1_in, r_in),
                                    (s2, c2_in, r_in),
                                ),
                                bucket_t=bucket_t,
                                tokens_per_seq=T,
                                max_pages_per_seq=max_pages_per_seq,
                                has_roff=has_roff,
                                use_ckpt_2d=use_ckpt_2d,
                                ops=torch_ops,
                            ).as_tuple()
                            for r_t, h_t, m_t in zip(
                                ref_out, head_out, meta_torch, strict=True
                            ):
                                assert torch.equal(r_t, h_t)
                                assert torch.equal(r_t, m_t)
                            if bucket_t == 8:
                                meta_jax = pack_step_meta(
                                    ids_j,
                                    pos_j,
                                    bt_j,
                                    (
                                        (s0_j, c0_j, r_j),
                                        (s1_j, c1_j, r_j),
                                        (s2_j, c2_j, r_j),
                                    ),
                                    bucket_t=bucket_t,
                                    tokens_per_seq=T,
                                    max_pages_per_seq=max_pages_per_seq,
                                    has_roff=has_roff,
                                    use_ckpt_2d=use_ckpt_2d,
                                    ops=jax_ops,
                                ).as_tuple()
                                for r_t, m_j in zip(ref_out, meta_jax, strict=True):
                                    assert np.array_equal(r_t.numpy(), np.asarray(m_j))

    # Verify Dynamo traces prepare_step_inputs_eager with fullgraph=True, dynamic=False
    compiled_fn = torch.compile(
        prepare_step_inputs_eager, backend="eager", fullgraph=True, dynamic=False
    )
    comp_out = compiled_fn(
        torch.arange(10, 18, dtype=torch.int32),
        torch.arange(20, 28, dtype=torch.int32),
        bt_2d,
        s0,
        c0_2d,
        roff_neg,
        s1,
        c1_2d,
        roff_neg,
        s2,
        c2_2d,
        roff_neg,
        8,
        4,
        max_pages_per_seq,
        True,
        True,
    )
    eager_out = _ref_prepare_step_inputs_eager(
        torch.arange(10, 18, dtype=torch.int32),
        torch.arange(20, 28, dtype=torch.int32),
        bt_2d,
        s0,
        c0_2d,
        roff_neg,
        s1,
        c1_2d,
        roff_neg,
        s2,
        c2_2d,
        roff_neg,
        8,
        4,
        max_pages_per_seq,
        True,
        True,
    )
    for c_t, e_t in zip(comp_out, eager_out, strict=True):
        assert torch.equal(c_t, e_t)

    # Extra edge cases: int64 inputs, PAD_SLOT_ID = -1 in s/c, slot beyond
    # r_raw.numel()-1, and 2-D ckpt narrower than T (shape=(6, 2) at T=4).
    s_edge = torch.tensor([1, -1, 999, 8, -1, 12], dtype=torch.int64)
    c_narrow = torch.tensor(
        [[1, 2], [-1, -1], [5, 6], [-1, 9], [15, 16], [12, 13]], dtype=torch.int64
    )
    r_edge = roff_neg.to(torch.int64)
    ids_64 = torch.arange(10, 22, dtype=torch.int64)
    pos_64 = torch.arange(30, 42, dtype=torch.int64)
    bt_64 = bt_2d.to(torch.int64)
    for T_edge, bkt_edge in ((1, 4), (4, 8)):
        if T_edge > 1:
            with pytest.raises(
                ValueError, match="requires 2-D ckpt_indices with >= T="
            ):
                prepare_step_inputs_eager(
                    ids_64,
                    pos_64,
                    bt_64,
                    s_edge,
                    c_narrow,
                    r_edge,
                    s_edge,
                    c_narrow,
                    r_edge,
                    s_edge,
                    c_narrow,
                    r_edge,
                    bkt_edge,
                    T_edge,
                    max_pages_per_seq,
                    True,
                    True,
                )
            continue
        ref_edge = _ref_prepare_step_inputs_eager(
            ids_64,
            pos_64,
            bt_64,
            s_edge,
            c_narrow,
            r_edge,
            s_edge,
            c_narrow,
            r_edge,
            s_edge,
            c_narrow,
            r_edge,
            bkt_edge,
            T_edge,
            max_pages_per_seq,
            True,
            True,
        )
        head_edge = prepare_step_inputs_eager(
            ids_64,
            pos_64,
            bt_64,
            s_edge,
            c_narrow,
            r_edge,
            s_edge,
            c_narrow,
            r_edge,
            s_edge,
            c_narrow,
            r_edge,
            bkt_edge,
            T_edge,
            max_pages_per_seq,
            True,
            True,
        )
        jax_edge = pack_step_meta(
            jnp.asarray(ids_64.numpy()),
            jnp.asarray(pos_64.numpy()),
            jnp.asarray(bt_64.numpy()),
            tuple(
                (
                    jnp.asarray(s_edge.numpy()),
                    jnp.asarray(c_narrow.numpy()),
                    jnp.asarray(r_edge.numpy()),
                )
                for _ in range(3)
            ),
            bucket_t=bkt_edge,
            tokens_per_seq=T_edge,
            max_pages_per_seq=max_pages_per_seq,
            has_roff=True,
            use_ckpt_2d=True,
            ops=jax_ops,
        ).as_tuple()
        for r_t, h_t, j_t in zip(ref_edge, head_edge, jax_edge, strict=True):
            assert r_t.dtype == torch.int32 and h_t.dtype == torch.int32
            assert torch.equal(r_t, h_t)
            assert np.array_equal(r_t.numpy(), np.asarray(j_t))


def test_bind_caches_rejects_compact_mamba_and_non_aliasing_la_cache() -> None:
    """Verify `bind` rejects compact-mamba or non-aliasing `la.kv_cache`."""
    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)
    raw_pools_t = _allocate_aliased_pools(backbone, g, 96)

    driver = _qwen_driver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2,),
        verify_buckets=(),
    )

    runner_compact = SimpleNamespace(model=model, _unified_kv_layout=False)
    with pytest.raises(ValueError, match="_unified_kv_layout must be True"):
        driver.bind(runner_compact)

    backbone.layers[0].linear_attn.kv_cache = []
    runner_ok = SimpleNamespace(model=model, _unified_kv_layout=True)
    with pytest.raises(ValueError, match="must alias pool"):
        driver.bind(runner_ok)

    separate_pool = torch.zeros_like(raw_pools_t[0])
    backbone.layers[0].linear_attn.kv_cache = [separate_pool]
    with pytest.raises(ValueError, match="must alias pool"):
        driver.bind(runner_ok)

    backbone.layers[0].linear_attn.kv_cache = [raw_pools_t[0]]
    driver.bind(runner_ok)
    assert driver._bound

    # `bind_and_warmup` falls back gracefully to stock if `bind_caches` fails
    # (e.g. non-zero `storage_offset` from a sliced pool), disabling `_megakernel`
    # instead of aborting startup.
    sliced_pool = raw_pools_t[0][1:]
    assert sliced_pool.storage_offset() > 0
    backbone.layers[3].self_attn.attn.kv_cache = [sliced_pool]
    runner_fallback = SimpleNamespace(
        model=model, _unified_kv_layout=True, _megakernel=driver
    )
    driver.bind_and_warmup(runner_fallback)
    assert not driver._bound
    declined = runner_fallback._megakernel
    assert isinstance(declined, driver_mod.DeclinedDriver)
    assert declined.decline.code == "cache_layout_incompatible"

    # Restore valid pool and verify `bind_and_warmup` does NOT swallow `ValueError`
    # raised inside `warmup()` as a cache-layout error.
    backbone.layers[3].self_attn.attn.kv_cache = [raw_pools_t[0]]
    orig_warmup = driver.warmup

    def _raise_val_err(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("synthetic warmup ValueError")

    driver.warmup = _raise_val_err  # type: ignore[method-assign]
    try:
        with pytest.raises(ValueError, match="synthetic warmup ValueError"):
            driver.bind_and_warmup(runner_ok)
    finally:
        driver.warmup = orig_warmup  # type: ignore[method-assign]

    # Verify `_verify_torchtpu_compile_debug` fails closed on `stablehlo_forward_text`
    driver.bind(runner_ok)
    with pytest.raises(RuntimeError, match="empty stablehlo_forward_text"):
        driver.program.check_compiled_ir(
            SimpleNamespace(stablehlo_forward_text=[]),
            Bucket(1, 2),
        )
    assert driver.program._torchtpu_ir_error is not None
    assert not driver.program._torchtpu_ir_checked_buckets

    driver.program._torchtpu_ir_error = None
    inlined_only_ir = (
        "module @jit_step {\n"
        "  func.func public @main(\n"
        "    %arg0: tensor<96x2x64x4x128xf8E4M3FN>\n"
        "  ) -> (tensor<16x256xbf16>, tensor<96x2x64x4x128xf8E4M3FN>) {\n"
        "    %0:2 = stablehlo.custom_call @tpu_custom_call(%arg0) : "
        "(tensor<96x2x64x4x128xf8E4M3FN>) -> "
        "(tensor<16x256xbf16>, tensor<96x2x64x4x128xf8E4M3FN>)\n"
        "    return %0#0, %0#1 : tensor<16x256xbf16>, "
        "tensor<96x2x64x4x128xf8E4M3FN>\n"
        "  }\n"
        "  func.func private @sub("
        "%p0: tensor<96x2x64x4x128xf8E4M3FN> {tf.aliasing_output = 1 : i32}"
        ") -> tensor<96x2x64x4x128xf8E4M3FN> {\n"
        "    return %p0 : tensor<96x2x64x4x128xf8E4M3FN>\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(RuntimeError, match="expected >= 1 donated pool arguments"):
        driver.program.check_compiled_ir(
            SimpleNamespace(stablehlo_forward_text=[inlined_only_ir]),
            Bucket(1, 2),
        )
    assert not driver.program._torchtpu_ir_checked_buckets

    valid_donor_ir = inlined_only_ir.replace(
        "%arg0: tensor<96x2x64x4x128xf8E4M3FN>",
        "%arg0: tensor<96x2x64x4x128xf8E4M3FN> {jax.buffer_donor = true}",
    )
    donor_with_copy_ir = valid_donor_ir.replace(
        "    return %0#0, %0#1",
        "    %cp = stablehlo.copy %0#1 : tensor<96x2x64x4x128xf8E4M3FN>\n"
        "    return %0#0, %cp",
    )
    with pytest.raises(RuntimeError, match="full-pool copy"):
        driver.program.check_compiled_ir(
            SimpleNamespace(stablehlo_forward_text=[donor_with_copy_ir]),
            Bucket(1, 2),
        )

    driver.program._torchtpu_ir_error = None
    driver.program.check_compiled_ir(
        SimpleNamespace(stablehlo_forward_text=[valid_donor_ir]),
        Bucket(1, 2),
    )
    assert driver.program._torchtpu_ir_checked_buckets == {(1, 2)}

    # Regression for BLOCKER 2: a `pool_gather=True` profile builds T=1 (split)
    # and T=4 (GPR) buckets cleanly in `bind_caches`.
    driver_gpr = _qwen_driver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        profile=mk.Profile(name="test-gpr", numerics="approx", pool_gather=True),
        weights=driver.program.weights,
        buckets=(2,),
        verify_buckets=(4,),
    )
    driver_gpr.bind(runner_ok)
    assert driver_gpr.program._bucket_gpr == {(1, 2): False, (4, 4): True}
    assert driver_gpr.program._bucket_fused_io == {(1, 2): False, (4, 4): False}


def test_compile_hlo_check_config_errors_are_not_layout_errors() -> None:
    """`compile_hlo_check` misconfiguration fails loudly; layout errors disable MK."""
    from vllm_torchtpu.megakernel import MegakernelConfigError
    from vllm_torchtpu.megakernel.runtime import check_compile_hlo_mesh

    class _MockDevice:
        id = 0

    mock_mesh = SimpleNamespace(devices=np.array([_MockDevice()], dtype=object))
    check_compile_hlo_mesh(mock_mesh, False)
    check_compile_hlo_mesh(_cpu_mesh(1), True)
    with pytest.raises(MegakernelConfigError, match="compile_hlo_check=1 requires"):
        check_compile_hlo_mesh(mock_mesh, True)

    def _driver_raising(exc: Exception) -> driver_mod.ProgramDriver:
        program = SimpleNamespace(display_name="test")
        drv = driver_mod.ProgramDriver(program, view=RunnerView(None))
        drv._bound = True

        def _bind(runner: Any) -> None:
            raise exc

        drv.bind = _bind  # type: ignore[method-assign]
        return drv

    runner = SimpleNamespace(_megakernel="mk")
    drv_cfg = _driver_raising(MegakernelConfigError("bad config"))
    with pytest.raises(MegakernelConfigError, match="bad config"):
        drv_cfg.bind_and_warmup(runner)
    assert runner._megakernel == "mk"
    # A plain ValueError (e.g. an internal bug) is not a layout error either.
    drv_val = _driver_raising(ValueError("some internal error"))
    with pytest.raises(ValueError, match="some internal error"):
        drv_val.bind_and_warmup(runner)
    assert runner._megakernel == "mk"
    # Only MegakernelLayoutError disables the megakernel (stock fallback).
    from vllm_torchtpu.megakernel import MegakernelLayoutError

    drv_layout = _driver_raising(MegakernelLayoutError("incompatible KV layout"))
    drv_layout.bind_and_warmup(runner)
    assert drv_layout._bound is False
    assert isinstance(runner._megakernel, driver_mod.DeclinedDriver)
    assert runner._megakernel.decline.code == "cache_layout_incompatible"


def test_bind_caches_fails_closed_on_pool_slot_layout_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A framework slot layout that differs from the megakernel's `PoolLayout`
    declines at bind, before any bucket compiles, with a named reason."""
    import dataclasses

    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)
    _allocate_aliased_pools(backbone, g, 96)
    driver = _qwen_driver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2,),
        verify_buckets=(),
    )
    mk = facts_from_pool_layout(driver.program.spec.cache_backend.layout, 1)
    built: list[Any] = []
    monkeypatch.setattr(
        driver.program, "build", lambda _b: built.append(1) or (lambda *_a: None)
    )
    cases = {
        "conv_state_dtype_unsupported": dataclasses.replace(mk, conv_dtype="float16"),
        "pool_layout_mismatch": dataclasses.replace(
            mk, slot_tokens=mk.slot_tokens + 512, ssm_dtype="float32"
        ),
    }
    for code, fw in cases.items():
        monkeypatch.setattr(
            driver.program, "_framework_slot_facts", lambda *_a, _f=fw: _f
        )
        runner = SimpleNamespace(model=model, _unified_kv_layout=True, _megakernel=0)
        with pytest.raises(LayoutDecline) as exc_info:
            driver.bind(runner)
        assert exc_info.value.result.code == code
        assert not driver._bound and not built
        driver.bind_and_warmup(runner)
        assert isinstance(runner._megakernel, driver_mod.DeclinedDriver)
        assert runner._megakernel.decline.code == code
        assert "megakernel=" in runner._megakernel.decline.message

    # Without the framework's layout objects (no cache_config.block_size), the
    # real framework-side derivation can't run: that declines too.
    monkeypatch.setattr(
        driver.program,
        "_framework_slot_facts",
        _REAL_FRAMEWORK_SLOT_FACTS.__get__(driver.program),
    )
    runner = SimpleNamespace(model=model, _unified_kv_layout=True)
    with pytest.raises(LayoutDecline) as exc_info:
        driver.bind(runner)
    assert exc_info.value.result.code == "pool_layout_unknown"

    monkeypatch.setattr(driver.program, "_framework_slot_facts", lambda *_a: mk)
    driver.bind(runner)
    assert driver._bound and built == [1]
    assert driver.pool_layout_facts == mk


@pytest.mark.parametrize(
    "fused, gather", [("auto", None), (True, True), (False, True), (False, False)]
)
def test_bind_caches_adopts_an_fp32_conv_framework_pool(
    monkeypatch: pytest.MonkeyPatch, fused: Any, gather: Any
) -> None:
    """A framework that stores the pooled conv state as fp32 is served: the
    megakernel switches its layout to an f32 conv window, and the fused pool
    I/O / in-kernel gather options (explicit ones included) plan every bucket
    exactly as they do for a bf16 conv window."""
    import dataclasses

    import jax.numpy as jnp

    from vllm_torchtpu.megakernel import runtime as _rt

    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)
    _allocate_aliased_pools(backbone, g, 96)
    drv = _qwen_driver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2,),
        verify_buckets=(4,),
        fused_pool_io=fused,
        pool_gather=gather,
    )
    driver = drv.program

    def _planned() -> dict[tuple[int, int], tuple[bool, bool]]:
        return {
            (T, b): _rt.bucket_exec_spec(
                driver.spec,
                driver.base_exec,
                driver.profile,
                bucket_t=b,
                tokens_per_seq=T,
                fused_pool_io=driver.fused_pool_io,
                fused_pool_io_max_b=driver.fused_pool_io_max_b,
                fused_pool_io_min_b_hi=driver.fused_pool_io_min_b_hi,
            )[1:]
            for T, b in driver.bucket_policy.iter_buckets()
        }

    planned_bf16 = _planned()
    options = (driver.fused_pool_io, driver.profile.pool_gather)
    bf16 = facts_from_pool_layout(driver.spec.cache_backend.layout, 1)
    assert bf16.conv_dtype == "bfloat16"
    fw = dataclasses.replace(bf16, conv_dtype="float32", conv_bytes=2 * bf16.conv_bytes)
    monkeypatch.setattr(driver, "_framework_slot_facts", lambda *_a: fw)
    monkeypatch.setattr(driver, "build", lambda _b: (lambda *_a: None))
    drv.bind(SimpleNamespace(model=model, _unified_kv_layout=True))
    assert drv._bound and drv.pool_layout_facts == fw
    lay = driver.spec.cache_backend.layout
    assert lay == dataclasses.replace(
        qwen35.qwen35_spec(g).cache_backend.layout, conv_dtype=jnp.float32
    )
    assert driver.spec.cache_backend.supports_fused_io
    gdn = [m for m in driver.spec.all_modules() if hasattr(m, "with_cache_layout")]
    assert gdn and all(m.pool_layout == lay for m in gdn)  # modules rebound too
    assert (driver.fused_pool_io, driver.profile.pool_gather) == options
    planned = _planned()
    assert (4, 4) in planned and planned == planned_bf16, (planned, planned_bf16)
    if fused is True:
        assert planned[(4, 4)][0], planned  # fused pool I/O on f32 conv
    elif gather is True:
        assert planned[(4, 4)] == (False, True), planned  # GPR on f32 conv
    elif fused is False and gather is False:
        assert not any(f or r for f, r in planned.values()), planned


def test_bind_caches_budget_overflow_is_config_error() -> None:
    """VMEM budget overflow in `bind_caches` propagates; it is not a layout error."""
    from vllm_torchtpu.megakernel import MegakernelConfigError

    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    ex_base = replace(ex_base, tiles=replace(ex_base.resolved_tiles, vmem_mib=0))
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)
    _allocate_aliased_pools(backbone, g, 96)
    driver = _qwen_driver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2,),
        verify_buckets=(),
    )
    runner = SimpleNamespace(model=model, _unified_kv_layout=True, _megakernel=driver)
    with pytest.raises(MegakernelConfigError, match="VMEM"):
        driver.bind_and_warmup(runner)
    assert runner._megakernel is driver


def test_driver_validates_profile_before_packing_and_uses_profile_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile + compile-HLO checks run before packing; profile buckets apply."""
    from vllm_torchtpu.megakernel import MegakernelConfigError, Profile
    from vllm_torchtpu.megakernel import profile as profile_mod

    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    model = SimpleNamespace(model=make_synthetic_qwen35_backbone(g, ex_base))

    def _no_pack(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("weights packed before the configuration was validated")

    monkeypatch.setattr(Qwen35Program, "_prepare_and_pack_weights", _no_pack)
    kw = dict(model=model, tp_rank=0, mode="ep", geom=g, base_exec=ex_base)
    with pytest.raises(ValueError, match="unknown megakernel profile"):
        Qwen35Program(mesh=_cpu_mesh(1), profile="nope", **kw)

    class _MockDevice:
        id = 0

    mock_mesh = SimpleNamespace(devices=np.array([_MockDevice()], dtype=object))
    with pytest.raises(MegakernelConfigError, match="compile_hlo_check=1 requires"):
        Qwen35Program(mesh=mock_mesh, compile_hlo_check=True, **kw)

    prof = Profile(name="b1-test-buckets", buckets=(2,), verify_buckets=(4,))
    monkeypatch.setitem(profile_mod._REGISTRY, prof.name, prof)
    drv = Qwen35Program(
        mesh=_cpu_mesh(1), profile=prof.name, weights={"w_flat": []}, **kw
    )
    assert (drv.decode_buckets, drv.verify_buckets) == ((2,), (4,))
    drv2 = Qwen35Program(
        mesh=_cpu_mesh(1),
        profile=prof.name,
        weights={"w_flat": []},
        buckets=(8,),
        verify_buckets=(),
        **kw,
    )
    assert (drv2.decode_buckets, drv2.verify_buckets) == ((8,), ())
    drv3 = Qwen35Program(mesh=_cpu_mesh(1), weights={"w_flat": []}, **kw)
    assert drv3.decode_buckets == DEFAULT_BUCKETS
    assert drv3.verify_buckets == DEFAULT_VERIFY_BUCKETS


def test_maybe_create_disables_on_unsupported_tpu_chip(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unsupported TPU chip (`SPEC.chips`) turns the megakernel off at
    startup, before `create` or any collective."""
    import logging
    import sys

    from tests.megakernel.frozen_utils import dict_to_namespace

    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=8, B=4, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    backbone.layers = list(backbone.layers) * 15
    model = SimpleNamespace(model=backbone)
    real_hf_cfg = dict_to_namespace(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": qwen35.DEFAULT_HF_CONFIG,
        }
    )
    model_cfg = SimpleNamespace(hf_config=real_hf_cfg)

    class _UnsupportedTPUDevice:
        id = 0
        device_kind = "TPU v5 lite"

    v5_mesh = SimpleNamespace(
        devices=np.array([_UnsupportedTPUDevice() for _ in range(8)], dtype=object)
    )
    runner = SimpleNamespace(
        model=model,
        model_config=model_cfg,
        mesh=v5_mesh,
        _unified_kv_layout=True,
        supports_mm_inputs=False,
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _pcp_enabled=False,
        _dcp_enabled=False,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        _dp_lockstep_enabled=lambda: False,
        vllm_config=SimpleNamespace(
            model_config=model_cfg, additional_config={"megakernel": True}
        ),
    )
    fake_vllm_dist = SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0)
    monkeypatch.setitem(sys.modules, "vllm.distributed", fake_vllm_dist)
    monkeypatch.setattr(RunnerView, "device_kind", property(lambda _v: "TPU v5 lite"))
    monkeypatch.setattr(
        driver_mod, "_build_tp_rank_mesh", lambda _r, fallback_mesh=None: fallback_mesh
    )
    try:
        driver_mod.logger._resolve()
    except ModuleNotFoundError:
        monkeypatch.setattr(
            driver_mod.logger, "_logger", logging.getLogger(driver_mod.logger._name)
        )
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.WARNING):
        drv = driver_mod.ProgramDriver.maybe_create(runner)
    # The runtime's spec check declines identically on every rank: no driver.
    assert drv is None
    assert "supports chips ('TPU7x',), not 'TPU v5 lite'" in caplog.text

    # No template serves an unresolvable arch: maybe_create returns None.
    unresolvable_runner = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=[])),
        vllm_config=SimpleNamespace(additional_config={"megakernel": True}),
    )
    assert driver_mod.ProgramDriver.maybe_create(unresolvable_runner) is None
    with pytest.raises(ValueError, match="Cannot locate decoder backbone"):
        Qwen35Program(
            model=SimpleNamespace(model=SimpleNamespace(layers=[])),
            mesh=_cpu_mesh(1),
        )


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{str(k): _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj


@pytest.mark.parametrize("bad", ["mismatch", "unavailable"])
def test_maybe_create_declines_without_usable_pallas_mesh(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, bad: str
) -> None:
    """With torch_tpu but no usable Pallas mesh, `maybe_create` returns a
    `DeclinedDriver` whose decline code shows up in the step-mix stats."""
    import logging
    import sys

    from vllm_torchtpu import megakernel_runtime as mkr

    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=8, B=4, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    backbone.layers = list(backbone.layers) * 15
    hf_cfg = _to_namespace(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": qwen35.DEFAULT_HF_CONFIG,
        }
    )
    model_cfg = SimpleNamespace(hf_config=hf_cfg)
    runner = SimpleNamespace(
        model=SimpleNamespace(model=backbone),
        model_config=model_cfg,
        mesh=_cpu_mesh(8),
        _unified_kv_layout=True,
        supports_mm_inputs=False,
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _pcp_enabled=False,
        _dcp_enabled=False,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="bfloat16"),
        _dp_lockstep_enabled=lambda: False,
        vllm_config=SimpleNamespace(
            model_config=model_cfg,
            additional_config={"megakernel": {"stats_period_s": 0}},
        ),
    )
    fake_vllm_dist = SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0)
    monkeypatch.setitem(sys.modules, "vllm.distributed", fake_vllm_dist)

    def get_pallas_mesh(**_kw: Any) -> Any:
        if bad == "unavailable":
            raise RuntimeError("no Pallas runtime")
        return _cpu_mesh(4)

    monkeypatch.setattr(driver_mod, "_get_pallas_mesh_fn", lambda: get_pallas_mesh)
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    code = f"pallas_mesh_{bad}"
    with caplog.at_level(logging.INFO):
        drv = driver_mod.ProgramDriver.maybe_create(runner)
        assert isinstance(drv, driver_mod.DeclinedDriver)
        assert drv.decline.code == code and not drv.decline.eligible
        assert "skipping" in caplog.text
        drv.bind_and_warmup(runner)
        ctx = StepContext(torch.zeros(2, dtype=torch.int32), torch.zeros(2))
        with mkr.suppressed(runner):
            assert drv.try_forward(runner, ctx) == (False, None)
        assert drv.step_mix_tracker.counts == {}
        caplog.clear()
        assert drv.try_forward(runner, ctx) == (False, None)
    # The periodic step-mix line names the decline category.
    assert f"'stock_{code}': 1" in caplog.text
    assert "0/1 MK" in caplog.text


@pytest.mark.parametrize("why", ["dp_size", "dp_env", "ep_wider_than_tp"])
def test_maybe_create_declines_data_parallel_before_any_collective(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, why: str
) -> None:
    """DP engines (or an EP group wider than TP) are a named startup decline,
    decided before the mesh, the TP agreement or weight packing."""
    import logging
    import sys

    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=8, B=4, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    backbone.layers = list(backbone.layers) * 15
    hf_cfg = _to_namespace(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": qwen35.DEFAULT_HF_CONFIG,
        }
    )
    model_cfg = SimpleNamespace(hf_config=hf_cfg)
    runner = SimpleNamespace(
        model=SimpleNamespace(model=backbone),
        model_config=model_cfg,
        mesh=_cpu_mesh(8),
        _unified_kv_layout=True,
        supports_mm_inputs=False,
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _pcp_enabled=False,
        _dcp_enabled=False,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            data_parallel_size=2 if why == "dp_size" else 1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="bfloat16"),
        _dp_lockstep_enabled=lambda: False,
        vllm_config=SimpleNamespace(
            model_config=model_cfg, additional_config={"megakernel": True}
        ),
    )
    if why == "dp_env":
        monkeypatch.setenv("TORCH_TPU_DP_SIZE", "4")
    else:
        monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    monkeypatch.setattr(
        guards_mod, "_ep_group_size", lambda: 16 if why == "ep_wider_than_tp" else 8
    )

    def no_collective(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("must decline before any collective / mesh build")

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        SimpleNamespace(get_tensor_model_parallel_rank=no_collective),
    )
    monkeypatch.setattr(driver_mod, "_build_tp_rank_mesh", no_collective)
    monkeypatch.setattr(driver_mod, "agree_on_decline", no_collective)
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    with caplog.at_level(logging.INFO):
        assert driver_mod.ProgramDriver.maybe_create(runner) is None
    if why == "ep_wider_than_tp":
        assert "parallelism 'ep_wider_than_tp' undeclared" in caplog.text
    else:
        assert "data_parallel_size" in caplog.text


def test_maybe_create_uses_model_default_profile_and_disables_empty_buckets(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`maybe_create` resolves the model default profile's `buckets` and
    `pool_gather`, and disables gracefully when effective bucket lists are empty."""
    import logging
    import sys

    from tests.megakernel.frozen_utils import dict_to_namespace
    from vllm_torchtpu.megakernel import Profile, register_profile
    from vllm_torchtpu.megakernel import profile as profile_mod

    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=8, B=4, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    backbone.layers = list(backbone.layers) * 15
    model = SimpleNamespace(model=backbone)
    mesh8 = _cpu_mesh(8)
    fake_vllm_dist = SimpleNamespace(get_tensor_model_parallel_rank=lambda: 0)
    monkeypatch.setitem(sys.modules, "vllm.distributed", fake_vllm_dist)
    try:
        driver_mod.logger._resolve()
    except ModuleNotFoundError:
        monkeypatch.setattr(
            driver_mod.logger, "_logger", logging.getLogger(driver_mod.logger._name)
        )
    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)
    monkeypatch.setattr(driver_mod, "_build_tp_rank_mesh", lambda *_a, **_k: mesh8)
    # The template serves its own default profile unless told otherwise; this
    # test covers the library model-default fallback.
    assert qwen_adapter.DEFAULT_PROFILE == "qwen35-candidate30-fused"
    monkeypatch.setattr(qwen_adapter, "DEFAULT_PROFILE", None)
    monkeypatch.setattr(
        Qwen35Program,
        "_prepare_and_pack_weights",
        lambda *_a, **_k: {"w_flat": []},
    )
    monkeypatch.setattr(profile_mod, "_REGISTRY", dict(profile_mod._REGISTRY))
    monkeypatch.setattr(
        profile_mod, "_MODEL_DEFAULTS", dict(profile_mod._MODEL_DEFAULTS)
    )

    real_hf_cfg = dict_to_namespace(
        {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": qwen35.DEFAULT_HF_CONFIG,
        }
    )
    model_cfg = SimpleNamespace(hf_config=real_hf_cfg)

    # 1. Model default profile setting `buckets=(4, 16)` and `pool_gather=None`:
    #    `maybe_create` inherits both without user overrides and validates real config.
    custom_default = Profile(
        name="qwen35-test-default",
        model="qwen3_5",
        buckets=(4, 16),
        verify_buckets=(8,),
        pool_gather=None,
        model_default=True,
    )
    register_profile(custom_default, replace=True)
    runner = SimpleNamespace(
        model=model,
        model_config=model_cfg,
        mesh=mesh8,
        _unified_kv_layout=True,
        speculative_config=None,
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="bfloat16"),
        vllm_config=SimpleNamespace(
            model_config=model_cfg,
            additional_config={"megakernel": {"enabled": True}},
        ),
    )
    drv = driver_mod.maybe_create(runner)
    assert isinstance(drv, driver_mod.ProgramDriver)
    prog = drv.program
    assert prog.model_def is qwen35.Qwen35
    assert prog.profile.name == "qwen35-test-default"
    assert prog.profile.pool_gather is None
    assert prog.decode_buckets == (4, 16)
    assert prog.verify_buckets == ()

    # 2. Explicit `buckets=0` on a non-speculative runner disables gracefully at INFO.
    runner_off = SimpleNamespace(
        model=model,
        model_config=model_cfg,
        mesh=mesh8,
        _unified_kv_layout=True,
        speculative_config=None,
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="bfloat16"),
        vllm_config=SimpleNamespace(
            model_config=model_cfg,
            additional_config={"megakernel": {"enabled": True, "buckets": []}},
        ),
    )
    caplog.clear()
    with caplog.at_level(logging.INFO):
        off = driver_mod.ProgramDriver.maybe_create(runner_off)
    assert isinstance(off, driver_mod.DeclinedDriver)
    assert off.decline.code == "empty_buckets"
    assert "both bucket lists empty" in caplog.text

    # 3. Model default profile with `buckets=()` on a non-speculative runner
    #    disables gracefully before packing weights.
    empty_default = Profile(
        name="qwen35-test-empty",
        model="qwen3_5",
        buckets=(),
        verify_buckets=(8,),
        model_default=True,
    )
    register_profile(empty_default, replace=True)
    empty = driver_mod.ProgramDriver.maybe_create(runner)
    assert isinstance(empty, driver_mod.DeclinedDriver)
    assert empty.decline.code == "empty_buckets"

    # 4. Supported TP sizes: `SPEC.tp_sizes == (8,)` turns tp=4 off at startup
    #    (no driver, no collective); declaring (4, 8) accepts tp=4.
    register_profile(custom_default, replace=True)
    mesh4 = _cpu_mesh(4)
    runner.mesh = mesh4
    monkeypatch.setattr(driver_mod, "_build_tp_rank_mesh", lambda *_a, **_k: mesh4)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        assert driver_mod.ProgramDriver.maybe_create(runner) is None
    assert "supports TP sizes (8,), not 4" in caplog.text

    spec_tp4 = replace(qwen_plugin.SPEC, tp_sizes=(4, 8))
    monkeypatch.setattr(qwen_plugin, "SPEC", spec_tp4)
    drv_tp4 = driver_mod.ProgramDriver.maybe_create(runner)
    assert drv_tp4.program.decode_buckets == (4, 16)

    # 5. Table-driven check that every startup code in `check_startup_eligibility`
    #    and `check_post_mesh_eligibility` maps through `_VLLM_STARTUP_MESSAGES`.
    from vllm_torchtpu.megakernel.runtime import check_startup_eligibility

    base_st = dict(
        num_layers=60,
        expected_num_layers=60,
        unified_cache_layout=True,
        tp_size=8,
        supported_tp_sizes=(8,),
    )
    for ov, exp_code, expected_substr in (
        ({"enabled": False}, "disabled", ""),
        ({"num_layers": 12}, "layer_count_mismatch", ""),
        (
            {"unified_cache_layout": False},
            "unaligned_cache_layout",
            "_unified_kv_layout=True",
        ),
        ({"multimodal_inputs": True}, "multimodal_inputs", "--language-model-only"),
        ({"lora_enabled": True}, "lora_enabled", "LoRA adapters"),
        ({"pipeline_parallel": True}, "pipeline_parallel", "single-stage pipeline"),
        ({"context_parallel": True}, "context_parallel", "context parallelism"),
        ({"dp_lockstep": True}, "dp_lockstep", "data-parallel lockstep"),
        ({"aux_hidden_states": True}, "aux_hidden_states", "aux_hidden_state_layers"),
        (
            {"tp_size": 4},
            "unsupported_tp_size",
            "Megakernel requires 8 TPU devices in mesh, got 4",
        ),
        (
            {"supported_tp_sizes": ()},
            "unsupported_tp_size",
            "does not declare any supported TPU mesh sizes",
        ),
    ):
        dec = check_startup_eligibility(**{**base_st, **ov})
        assert not dec.eligible and dec.code == exp_code
        msgs = {
            **driver_mod._VLLM_STARTUP_MESSAGES,
            "unaligned_cache_layout": qwen_adapter.UNALIGNED_LAYOUT_MESSAGE,
        }
        msg = msgs.get(dec.code, dec.message)
        assert expected_substr in msg

    base_post = dict(
        tp_size=8,
        mesh_size=8,
        device_kind="TPU v7x",
        bucket_policy=prog.bucket_policy,
    )
    for ov, exp_code, expected_substr in (
        (
            {"mesh_size": 4},
            "mesh_size_mismatch",
            "TP-rank mesh has 4 devices, expected 8",
        ),
        ({"device_kind": "TPU v5e"}, "unsupported_tpu_chip", "Unsupported TPU"),
    ):
        dec_post = check_post_mesh_eligibility(**{**base_post, **ov})
        assert not dec_post.eligible and dec_post.code == exp_code
        msg_post = driver_mod._VLLM_STARTUP_MESSAGES.get(
            dec_post.code, dec_post.message
        )
        assert expected_substr in msg_post


def test_can_run_real_tpu_runner_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `can_run` against real `TPUModelRunner` and `AttentionMetadata` fields."""
    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)
    _allocate_aliased_pools(backbone, g, 96)

    driver = _qwen_driver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2, 4),
        verify_buckets=(4, 8),
    )

    qsl_np = np.zeros(130, dtype=np.int32)
    req_dist_cpu = torch.zeros(3, dtype=torch.int32)
    comb_dist_cpu = torch.zeros(6, dtype=torch.int32)
    runner = SimpleNamespace(
        model=model,
        mesh=mesh,
        max_num_blocks_per_req=2,
        num_tokens_paddings=[2, 4, 8],
        _unified_kv_layout=True,
        supports_mm_inputs=False,
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _pcp_enabled=False,
        _dcp_enabled=False,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        _dp_lockstep_enabled=lambda: False,
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=2),
        _request_distribution_cpu=req_dist_cpu,
        _combined_request_distribution_cpu=comb_dist_cpu,
        _cached_comb_key=None,
        query_start_loc_np=qsl_np,
        mamba_slot_read_offsets=None,
        mrope_positions=None,
    )
    driver.bind(runner)

    max_reqs_padded = 128
    attn_md = SimpleNamespace(
        block_tables=torch.zeros((max_reqs_padded * 2,), dtype=torch.int32),
        seq_lens=torch.ones((max_reqs_padded,), dtype=torch.int32),
        query_start_loc=None,
        request_distribution=None,
    )
    mamba_mds = [
        SimpleNamespace(
            mamba_state_indices=torch.zeros((max_reqs_padded,), dtype=torch.int32),
            mamba_ckpt_indices=None,
            mamba_slot_read_offsets=None,
            mamba_request_distribution=None,
        )
        for _ in range(3)
    ]
    md_dict = {driver.program.attn_layer_name: attn_md}
    for name, m_md in zip(driver.program.mamba_layer_names, mamba_mds, strict=True):
        md_dict[name] = m_md

    monkeypatch.setattr(
        driver_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=md_dict),
    )

    ids_2 = torch.zeros((2,), dtype=torch.int32)
    pos_2 = torch.zeros((2,), dtype=torch.int32)
    ids_4 = torch.zeros((4,), dtype=torch.int32)
    pos_4 = torch.zeros((4,), dtype=torch.int32)
    ids_8 = torch.zeros((8,), dtype=torch.int32)
    pos_8 = torch.zeros((8,), dtype=torch.int32)

    # 1. Pure T=1 decode (2 reqs, 1 token each) -> True (mk_t1_b2)
    runner._attn_metadata_builder_ctx.num_reqs = 2
    req_dist_cpu.copy_(torch.tensor([2, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 1, 2]
    assert driver.can_run(ids_2, pos_2, runner=runner)
    assert _decided(driver) == (2, 1, 2)

    # 2. Single-request prefill (1 req, 8 tokens or 4 tokens in non-spec mode) -> False
    runner._attn_metadata_builder_ctx.num_reqs = 1
    req_dist_cpu.copy_(torch.tensor([0, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 8]
    assert not driver.can_run(ids_8, pos_8, runner=runner)
    qsl_np[:2] = [0, 4]
    assert not driver.can_run(ids_4, pos_4, runner=runner)

    # 3. Mixed decode + prefill (2 reqs, [1, 7] tokens) -> False, followed by pure
    #    decode at the same concurrency where `_request_distribution_cpu` remains
    #    stale (`[1, 2, 2]`) on a decode-metadata cache hit -> must still run MK!
    runner._attn_metadata_builder_ctx.num_reqs = 2
    req_dist_cpu.copy_(torch.tensor([1, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 1, 8]
    assert not driver.can_run(ids_8, pos_8, runner=runner)
    # Subsequent pure-decode step hits decode metadata cache: `req_dist_cpu` stays
    # `[1, 2, 2]`, while `qsl_np[:3]` is updated to `[0, 1, 2]`.
    qsl_np[:3] = [0, 1, 2]
    assert driver.can_run(ids_2, pos_2, runner=runner)
    assert _decided(driver) == (2, 1, 2)

    # 4. Spec decode enabled (`mamba_slot_read_offsets` present, 4-col ckpt):
    #    4a. Uniform T=4 verify (1 req, 4 tokens) -> True (mk_t4_b4)
    runner.mamba_slot_read_offsets = torch.zeros((64,), dtype=torch.int32)
    for m_md in mamba_mds:
        m_md.mamba_ckpt_indices = torch.zeros((max_reqs_padded, 4), dtype=torch.int32)
        m_md.mamba_slot_read_offsets = runner.mamba_slot_read_offsets
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 4]
    assert driver.can_run(ids_4, pos_4, runner=runner)
    assert _decided(driver) == (1, 4, 4)

    #    4b. T=1 plain decode under spec decode (`ckpt.shape == (128, 4)`, 2 active
    #        reqs with 1 token each): must dispatch as (num_reqs=2, T=1, bkt=2), NOT
    #        T=4 and NOT num_reqs=128!
    runner._attn_metadata_builder_ctx.num_reqs = 2
    comb_dist_cpu.copy_(torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 1, 2]
    assert driver.can_run(ids_2, pos_2, runner=runner)
    assert _decided(driver) == (2, 1, 2)

    #    4c. Variable draft lengths under spec decode ([4, 2] tokens, both windowed
    #        so comb_dist_cpu[3] == 2 == num_reqs) -> False (stock_mixed)
    runner._attn_metadata_builder_ctx.num_reqs = 2
    comb_dist_cpu.copy_(torch.tensor([0, 0, 2, 2, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 4, 6]
    assert not driver.can_run(ids_8, pos_8, runner=runner)

    #    4d. Regression for BLOCKER 1: 3-element `mamba_request_distribution` /
    #        `request_distribution` tensors on `mamba_mds[0]` / `attn_md` must never
    #        trigger `RuntimeError: Boolean value of Tensor with more than one value
    #        is ambiguous` (neither on the runner fast-path nor on the fallback path).
    dist_tensor_3 = torch.tensor([1, 1, 1], dtype=torch.int32)
    mamba_mds[0].mamba_request_distribution = dist_tensor_3
    attn_md.request_distribution = dist_tensor_3
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 4]
    assert driver.can_run(ids_4, pos_4, runner=runner)
    # Also exercise the fallback path (`bctx is None`, `query_start_loc` on `attn_md`)
    orig_bctx = runner._attn_metadata_builder_ctx
    runner._attn_metadata_builder_ctx = None
    attn_md.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    assert driver.can_run(ids_4, pos_4, runner=runner)

    # Exercise the fallback path with a non-CPU (`device="meta"`) 3-element
    # `mamba_request_distribution` tensor and no runner `qsl`/distribution:
    # must neither raise `Boolean value of Tensor...` nor sync, and fail closed.
    meta_dist_3 = torch.empty((3,), dtype=torch.int32, device="meta")
    mamba_mds[0].mamba_request_distribution = meta_dist_3
    attn_md.request_distribution = meta_dist_3
    attn_md.query_start_loc = None
    assert driver._inspect_step_detail(
        attn_md, mamba_mds[0], runner=None
    ) == BatchShape(False, 0, 0, "stock_no_qsl")
    no_qsl_runner = SimpleNamespace(
        _attn_metadata_builder_ctx=None,
        mamba_slot_read_offsets=None,
        _cached_comb_key=None,
        _combined_request_distribution_cpu=None,
        query_start_loc_np=None,
        query_start_loc_cpu=None,
    )
    assert driver._inspect_step_detail(
        attn_md, mamba_mds[0], runner=no_qsl_runner
    ) == BatchShape(False, 0, 0, "stock_no_qsl")
    assert not driver.can_run(ids_4, pos_4, runner=None)

    # With CPU `query_start_loc` and `num_seqs=1` on `attn_md`, the non-CPU `meta`
    # distribution tensor is ignored without syncing and `fallback_num_seqs` succeeds.
    attn_md.num_seqs = 1
    attn_md.query_start_loc = torch.tensor([0, 4], dtype=torch.int32)
    assert driver._inspect_step_detail(
        attn_md, mamba_mds[0], runner=None
    ) == BatchShape(True, 1, 4, "ok")
    del attn_md.num_seqs

    runner._attn_metadata_builder_ctx = orig_bctx
    runner._attn_metadata_builder_ctx.num_reqs = 2
    comb_dist_cpu.copy_(torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32))
    attn_md.query_start_loc = None
    mamba_mds[0].mamba_request_distribution = None
    attn_md.request_distribution = None

    # 5. Missing host `query_start_loc` must fail closed (`stock_no_qsl`), never
    #    falling back to `mamba_state_indices.shape[0]` (128) or `ckpt.shape[1]` (4).
    runner.query_start_loc_np = None
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    runner.query_start_loc_np = qsl_np
    qsl_np[:3] = [0, 1, 2]
    assert driver.can_run(ids_2, pos_2, runner=runner)

    # 6. Dummy runs (`bctx.position_ids_override is not None`) gate off MK
    runner._attn_metadata_builder_ctx.position_ids_override = pos_2
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    hit_dummy, out_dummy = driver.try_forward(runner, StepContext(ids_2, pos_2))
    assert not hit_dummy and out_dummy is None
    runner._attn_metadata_builder_ctx.position_ids_override = None
    assert driver.can_run(ids_2, pos_2, runner=runner)

    # 7. mrope 2D positions `(3, tok_pad)`: rejected when rows differ, accepted when ==
    pos_mrope_diff = torch.tensor([[5, 9], [5, 10], [5, 9]], dtype=torch.int32)
    assert not driver.can_run(ids_2, pos_mrope_diff, runner=runner)
    pos_mrope_eq = torch.tensor([[5, 9], [5, 9], [5, 9]], dtype=torch.int32)
    assert driver.can_run(ids_2, pos_mrope_eq, runner=runner)

    # 8. Static runner configurations (unified layout, MM, LoRA, PP, PCP/DCP,
    #    DP lockstep) are startup declines (`check_spec`, the template's
    #    `create`); aux hidden-state capture also gates every step.
    spec = replace(qwen_plugin.SPEC, chips=(), tp_sizes=(), requires={})
    view = RunnerView(runner, tp_rank=0)
    assert driver_mod.check_spec(runner, spec, view=view) is None
    for attr, val, code in (
        ("supports_mm_inputs", True, "multimodal_inputs"),
        ("lora_config", object(), "lora_enabled"),
        ("_pp_is_last", False, "parallelism_unsupported:pp"),
        ("_pcp_enabled", True, "parallelism_unsupported:pcp"),
        ("_dp_lockstep_enabled", lambda: True, "parallelism_unsupported:dp_lockstep"),
    ):
        old = getattr(runner, attr)
        setattr(runner, attr, val)
        res = driver_mod.check_spec(runner, spec, view=view)
        assert res is not None and res.code == code, (attr, res)
        setattr(runner, attr, old)

    backbone.aux_hidden_state_layers = (1,)
    assert not driver.can_run(ids_2, pos_2, runner=runner)
    backbone.aux_hidden_state_layers = ()

    # 9. `megakernel_runtime.maybe_create` registry and lazy fast-path
    import vllm_torchtpu.megakernel_runtime as mk_rt

    non_qwen_runner = SimpleNamespace(
        model=SimpleNamespace(),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["LlamaForCausalLM"], model_type="llama"
            )
        ),
    )
    assert driver_mod.maybe_create(non_qwen_runner) is None
    assert mk_rt.maybe_create(non_qwen_runner) is None


def test_megakernel_driver_e2e_decode_verify_and_spec_accept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the Qwen3.5 program's warmup, `T=1`, `T=4`, and speculative accept."""
    mesh = _cpu_mesh(1)
    tp = NamedSharding(mesh, P("tp"))
    unsh = NamedSharding(mesh, P())
    g = _tiny_geom(L=4, att_every=4)
    num_pages = 160
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=num_pages)

    # Generate independent logical weights and packed MK reference weights via pure-JAX
    # `qwen35` helpers, then construct the PyTorch backbone from `w_logical`.
    torch.manual_seed(42)
    hf = dict(mk.random_hf_weights(jax.random.key(42), g))
    jax_ops = mk.JaxArrayOps()
    b_bf16 = (
        jax.random.normal(jax.random.key(43), (g.NGDN, g.H, g.VH), dtype=jnp.bfloat16)
        * 0.05
    )
    a_bf16 = (
        jax.random.normal(jax.random.key(44), (g.NGDN, g.H, g.VH), dtype=jnp.bfloat16)
        * 0.05
    )
    hf["lin_b"], hf["lin_b_s"] = jax.vmap(jax_ops.quantize_fp8)(b_bf16)
    hf["lin_a"], hf["lin_a_s"] = jax.vmap(jax_ops.quantize_fp8)(a_bf16)
    w_logical = mk.load_logical_weights(qwen35.Qwen35, ex_base, hf, g=g)
    w_logical["_ba_bf16"] = jnp.swapaxes(
        jnp.concatenate([b_bf16, a_bf16], axis=-1), -1, -2
    )[None, ...]
    pos_init, hid_l_init, _, st_l = mk.random_logical_inputs(
        jax.random.key(77), g, ex_base
    )
    st_l = dict(
        st_l,
        conv=st_l["conv"].astype(jnp.bfloat16).astype(jnp.float32),
    )
    _, _, w_mk_ref, _ = mk.pack_inputs(
        pos_init,
        hid_l_init,
        w_logical,
        st_l,
        g,
        ex_base,
        mesh=mesh,
        build_default_pool=True,
    )

    backbone = _make_backbone_from_logical_weights(g, ex_base, w_logical, tp_rank=0)
    model = SimpleNamespace(model=backbone)
    raw_pools_t = _allocate_aliased_pools(backbone, g, num_pages)

    qsl_np = np.zeros(16, dtype=np.int32)
    req_dist_cpu = torch.zeros(3, dtype=torch.int32)
    comb_dist_cpu = torch.zeros(6, dtype=torch.int32)
    roff_slot_buf = torch.zeros((32,), dtype=torch.int32)
    runner = SimpleNamespace(
        model=model,
        mesh=mesh,
        max_num_blocks_per_req=2,
        num_reqs_max_model_len=4,
        num_tokens_paddings=[2, 4, 8],
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        _mamba_ckpt_window=4,
        _unified_kv_layout=True,
        lora_config=None,
        _pp_is_first=True,
        _pp_is_last=True,
        _dp_lockstep_enabled=lambda: False,
        _in_dummy_run=False,
        _attn_metadata_builder_ctx=SimpleNamespace(num_reqs=2),
        _request_distribution_cpu=req_dist_cpu,
        _combined_request_distribution_cpu=comb_dist_cpu,
        _cached_comb_key=None,
        query_start_loc_np=qsl_np,
        mamba_slot_read_offsets=roff_slot_buf,
    )

    driver = _qwen_driver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2,),
        verify_buckets=(4,),
    )
    orig_data_ptrs = [p.data_ptr() for p in raw_pools_t]
    orig_storage_ptrs = [p.untyped_storage().data_ptr() for p in raw_pools_t]

    # 1. `bind_and_warmup` binds pools, compiles (1, 2) and (4, 4), runs sentinel check,
    #    and restores pools
    driver.bind_and_warmup(runner)
    assert driver.program._hlo_checked_buckets == {(1, 2), (4, 4)}
    assert driver.program._bucket_fused_io == {(1, 2): False, (4, 4): False}
    # Qwen3.5 default profile: split pool I/O everywhere (GPR is opt-in; it never
    # applies at T=1 and "auto" only selects it for WIDE verify buckets, B > 16).
    assert driver.program.profile.name == "qwen35-baseline"
    assert driver.program._bucket_gpr == {(1, 2): False, (4, 4): False}
    # Each bucket step has its own code object (separate Dynamo cache entries)
    step_t1, step_t4 = driver._steps[(1, 2)], driver._steps[(4, 4)]
    assert step_t1.__code__ is not step_t4.__code__
    assert (step_t1.__code__.co_name, step_t4.__code__.co_name) == (
        "step_t1_b2",
        "step_t4_b4",
    )
    for idx, p in enumerate(raw_pools_t):
        assert p.data_ptr() == orig_data_ptrs[idx]
        assert p.untyped_storage().data_ptr() == orig_storage_ptrs[idx]
        assert torch.count_nonzero(p.view(torch.uint8)).item() == 0

    # Pack initial logical KV and GDN states into Raiden pools via `qwen35_pool`
    # using disjoint GDN slot IDs per sub-layer `sub in {0, 1, 2}`:
    #   sub=0: seq 0 -> 1..4,  seq 1 -> 5..8
    #   sub=1: seq 0 -> 9..12, seq 1 -> 13..16
    #   sub=2: seq 0 -> 17..20, seq 1 -> 21..24
    pt_init = jnp.asarray([[140, 141], [142, 143]], dtype=jnp.int32)
    ckpt_init = jnp.asarray([[[1], [5]], [[9], [13]], [[17], [21]]], dtype=jnp.int32)
    roff_init = jnp.zeros((3, 2), dtype=jnp.int32)
    init_pools_j = qwen35_pool.states_to_pool(
        g, ex_base, st_l, pt_init, ckpt_init, roff_init
    )
    for p_t, p_j in zip(raw_pools_t, init_pools_j, strict=True):
        p_t.view(torch.uint8).copy_(
            torch.from_numpy(np.asarray(p_j).view(np.uint8).copy())
        )

    # Set up serving forward context with 1-D flattened block_tables and padded mamba
    # indices (matching TPUModelRunner's exact tensor shapes) with disjoint slots
    # across the 3 linear-attention layers sharing pool[0].
    attn_md = SimpleNamespace(
        block_tables=torch.tensor([140, 141, 142, 143, 0, 0, 0, 0], dtype=torch.int32),
        seq_lens=torch.tensor([6, 10, 0, 0], dtype=torch.int32),
    )
    mamba_mds = [
        SimpleNamespace(
            mamba_state_indices=torch.tensor(
                [1 + 8 * sub, 5 + 8 * sub, 0, 0], dtype=torch.int32
            ),
            mamba_ckpt_indices=torch.tensor(
                [
                    [1 + 8 * sub, 2 + 8 * sub, 3 + 8 * sub, 4 + 8 * sub],
                    [5 + 8 * sub, 6 + 8 * sub, 7 + 8 * sub, 8 + 8 * sub],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int32,
            ),
            mamba_slot_read_offsets=roff_slot_buf,
        )
        for sub in range(3)
    ]
    md_dict = {driver.program.attn_layer_name: attn_md}
    for name, m_md in zip(driver.program.mamba_layer_names, mamba_mds, strict=True):
        md_dict[name] = m_md

    monkeypatch.setattr(
        driver_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=md_dict),
    )

    # -------------------------------------------------------------------------
    # Step A: T=1 decode (B=2, NS=2, T=1) vs both `qwen35.reference_step` and
    #         split 3-stage `qwen35.make_pooled_step`.
    # -------------------------------------------------------------------------
    runner._attn_metadata_builder_ctx.num_reqs = 2
    comb_dist_cpu.copy_(torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32))
    qsl_np[:3] = [0, 1, 2]
    roff_slot_buf.zero_()

    input_ids_t1 = torch.tensor([5, 12], dtype=torch.int32)
    positions_t1 = torch.tensor([5, 9], dtype=torch.int32)
    ref_pools_j = tuple(_torch_to_jax(p.clone(), tp) for p in raw_pools_t)

    hit_t1, out_t1 = driver.try_forward(runner, StepContext(input_ids_t1, positions_t1))
    assert hit_t1 and out_t1 is not None
    assert out_t1.shape == (2, g.H)
    for idx, p in enumerate(raw_pools_t):
        assert p.data_ptr() == orig_data_ptrs[idx]

    # Independent pure-JAX `qwen35.reference_step` check
    hid_t1_bf16 = driver.program.embed_tokens(input_ids_t1).to(torch.bfloat16)
    hid_l_t1 = _torch_to_jax(hid_t1_bf16, unsh)
    pos_j_t1 = jnp.asarray([5, 9], dtype=jnp.int32)
    ref_out_t1, _ = qwen35.reference_step(
        g, ex_base, pos_j_t1, hid_l_t1, w_logical, st_l, paged_kv=True, pool_io=True
    )
    np.testing.assert_allclose(
        out_t1.float().numpy(),
        np.asarray(ref_out_t1, dtype=np.float32),
        atol=8e-2,
        rtol=8e-2,
    )

    # Independent split 3-stage `qwen35.make_pooled_step` oracle
    ex_t1 = with_cache(
        replace(driver.program.base_exec, B=2, T=1),
        fused_pool_io=False,
        pool_gather=False,
    )
    step_ref_t1, _ = qwen35.make_pooled_step(
        mesh, g, ex_t1, donate_pools=False, fused_pool_io=False
    )
    hid_in_t1_pad = torch.nn.functional.pad(hid_t1_bf16, (0, 0, 0, ex_t1.M - 2))
    ckpt_ref_t1 = jnp.asarray(
        [[[1, 1], [5, 5]], [[9, 9], [13, 13]], [[17, 17], [21, 21]]],
        dtype=jnp.int32,
    )
    roff_ref_t1 = jnp.zeros((3, 2), dtype=jnp.int32)
    hid_ref_t1, ref_pools_j = step_ref_t1(
        pos_j_t1,
        _torch_to_jax(hid_in_t1_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_init,
        ckpt_ref_t1,
        roff_ref_t1,
    )
    np.testing.assert_array_equal(
        out_t1.view(torch.uint8).numpy(),
        np.asarray(hid_ref_t1[:2]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )

    # -------------------------------------------------------------------------
    # Step B: T=4 MTP verify (B=4, NS=1, T=4) writing contiguous slots per sub
    # -------------------------------------------------------------------------
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 4]
    roff_slot_buf.zero_()
    attn_md.block_tables = torch.tensor([140, 141, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    attn_md.seq_lens = torch.tensor([9, 0, 0, 0], dtype=torch.int32)
    for sub, m_md in enumerate(mamba_mds):
        b0 = 1 + 8 * sub
        m_md.mamba_state_indices = torch.tensor([b0, 0, 0, 0], dtype=torch.int32)
        m_md.mamba_ckpt_indices = torch.tensor(
            [[b0, b0 + 1, b0 + 2, b0 + 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=torch.int32,
        )

    input_ids_t4 = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    positions_t4 = torch.tensor([6, 7, 8, 9], dtype=torch.int32)
    assert driver.can_run(input_ids_t4, positions_t4, runner=runner)
    out_t4 = driver(input_ids_t4, positions_t4, num_tokens=4, runner=runner)
    assert out_t4.shape == (4, g.H)

    ex_t4 = with_cache(
        replace(driver.program.base_exec, B=4, T=4),
        fused_pool_io=False,
        pool_gather=False,
    )
    step_ref_t4, _ = qwen35.make_pooled_step(
        mesh, g, ex_t4, donate_pools=False, fused_pool_io=False
    )
    hid_t4_bf16 = driver.program.embed_tokens(input_ids_t4).to(torch.bfloat16)
    hid_in_t4_pad = torch.nn.functional.pad(hid_t4_bf16, (0, 0, 0, ex_t4.M - 4))
    pos_j_t4 = jnp.asarray([6, 7, 8, 9], dtype=jnp.int32)
    pt_ref_t4 = jnp.asarray([[140, 141]], dtype=jnp.int32)
    ckpt_ref_t4 = jnp.asarray(
        [[[1, 2, 3, 4]], [[9, 10, 11, 12]], [[17, 18, 19, 20]]], dtype=jnp.int32
    )
    roff_ref_t4 = jnp.zeros((3, 1), dtype=jnp.int32)
    hid_ref_t4, ref_pools_j = step_ref_t4(
        pos_j_t4,
        _torch_to_jax(hid_in_t4_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_ref_t4,
        ckpt_ref_t4,
        roff_ref_t4,
    )
    np.testing.assert_array_equal(
        out_t4.view(torch.uint8).numpy(),
        np.asarray(hid_ref_t4[:4]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )

    # -------------------------------------------------------------------------
    # Step C: T=1 plain decode after T=4 verify with `roff = 2` (accepting 2
    #         draft tokens so the step reads `ckpt[sub, 0, 2]` and writes
    #         the updated GDN state back to base slot `s[sub, 0]`).
    # -------------------------------------------------------------------------
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 1]
    roff_slot_buf.zero_()
    roff_slot_buf[1] = 2  # sub=0 base slot 1 -> read offset 2 (slot 3)
    roff_slot_buf[9] = 2  # sub=1 base slot 9 -> read offset 2 (slot 11)
    roff_slot_buf[17] = 2  # sub=2 base slot 17 -> read offset 2 (slot 19)

    # Snapshot slot 3 (pages 15..19) and slot 1 (pages 5..9) before Step C
    slot3_before = [p[15:19].view(torch.uint8).clone() for p in raw_pools_t]
    slot1_before = [p[5:9].view(torch.uint8).clone() for p in raw_pools_t]

    input_ids_t1_after_t4 = torch.tensor([19, 0], dtype=torch.int32)
    positions_t1_after_t4 = torch.tensor([8, 0], dtype=torch.int32)
    assert driver.can_run(input_ids_t1_after_t4, positions_t1_after_t4, runner=runner)
    assert _decided(driver) == (1, 1, 2)
    out_t1_after_t4 = driver(
        input_ids_t1_after_t4, positions_t1_after_t4, num_tokens=2, runner=runner
    )

    # Slot 1 (base slot, pages 5..9) is updated; slot 3 (pages 15..19) is untouched
    for idx, p in enumerate(raw_pools_t):
        assert not torch.equal(p[5:9].view(torch.uint8), slot1_before[idx])
        assert torch.equal(p[15:19].view(torch.uint8), slot3_before[idx])

    # Independent `make_pooled_step` oracle (read=3/11/19, write=1/9/17 for seq 0)
    hid_c_bf16 = driver.program.embed_tokens(input_ids_t1_after_t4).to(torch.bfloat16)
    hid_in_c_pad = torch.nn.functional.pad(hid_c_bf16, (0, 0, 0, ex_t1.M - 2))
    pos_j_c = jnp.asarray([8, 0], dtype=jnp.int32)
    pt_ref_c = jnp.asarray([[140, 141], [0, 0]], dtype=jnp.int32)
    ckpt_ref_c = jnp.asarray(
        [[[3, 1], [0, 0]], [[11, 9], [0, 0]], [[19, 17], [0, 0]]], dtype=jnp.int32
    )
    roff_ref_c = jnp.zeros((3, 2), dtype=jnp.int32)
    hid_ref_c, ref_pools_j = step_ref_t1(
        pos_j_c,
        _torch_to_jax(hid_in_c_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_ref_c,
        ckpt_ref_c,
        roff_ref_c,
    )
    np.testing.assert_array_equal(
        out_t1_after_t4[:1].view(torch.uint8).numpy(),
        np.asarray(hid_ref_c[:1]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )

    # -------------------------------------------------------------------------
    # Step D: Subsequent T=4 MTP verify with reset `roff = 0` (reads base slots
    #         written by Step C) followed by Step E with `roff = 2`.
    # -------------------------------------------------------------------------
    runner._attn_metadata_builder_ctx.num_reqs = 1
    comb_dist_cpu.copy_(torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.int32))
    qsl_np[:2] = [0, 4]
    roff_slot_buf.zero_()  # reset to 0 after the 1-token step

    input_ids_t4_d = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    positions_t4_d = torch.tensor([9, 10, 11, 12], dtype=torch.int32)
    assert driver.can_run(input_ids_t4_d, positions_t4_d, runner=runner)
    out_t4_d = driver(input_ids_t4_d, positions_t4_d, num_tokens=4, runner=runner)

    hid_d_bf16 = driver.program.embed_tokens(input_ids_t4_d).to(torch.bfloat16)
    hid_in_d_pad = torch.nn.functional.pad(hid_d_bf16, (0, 0, 0, ex_t4.M - 4))
    pos_j_d = jnp.asarray([9, 10, 11, 12], dtype=jnp.int32)
    hid_ref_d, ref_pools_j = step_ref_t4(
        pos_j_d,
        _torch_to_jax(hid_in_d_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_ref_t4,
        ckpt_ref_t4,
        roff_ref_t4,
    )
    np.testing.assert_array_equal(
        out_t4_d.view(torch.uint8).numpy(),
        np.asarray(hid_ref_d[:4]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )

    # Step E: Speculative acceptance on next T=4 step via `roff_slot_buf = 2`
    roff_slot_buf[1] = 2
    roff_slot_buf[9] = 2
    roff_slot_buf[17] = 2
    input_ids_t4_acc = torch.tensor([31, 32, 33, 34], dtype=torch.int32)
    positions_t4_acc = torch.tensor([11, 12, 13, 14], dtype=torch.int32)
    assert driver.can_run(input_ids_t4_acc, positions_t4_acc, runner=runner)
    out_t4_acc = driver(input_ids_t4_acc, positions_t4_acc, num_tokens=4, runner=runner)

    hid_e_bf16 = driver.program.embed_tokens(input_ids_t4_acc).to(torch.bfloat16)
    hid_in_e_pad = torch.nn.functional.pad(hid_e_bf16, (0, 0, 0, ex_t4.M - 4))
    pos_j_e = jnp.asarray([11, 12, 13, 14], dtype=jnp.int32)
    roff_ref_e = jnp.asarray([[2], [2], [2]], dtype=jnp.int32)
    hid_ref_e, ref_pools_j = step_ref_t4(
        pos_j_e,
        _torch_to_jax(hid_in_e_pad, tp),
        w_mk_ref,
        ref_pools_j,
        pt_ref_t4,
        ckpt_ref_t4,
        roff_ref_e,
    )
    np.testing.assert_array_equal(
        out_t4_acc.view(torch.uint8).numpy(),
        np.asarray(hid_ref_e[:4]).view(np.uint8),
    )
    for p_t, p_ref in zip(raw_pools_t, ref_pools_j, strict=True):
        np.testing.assert_array_equal(
            p_t.view(torch.uint8).numpy(),
            np.asarray(p_ref).view(np.uint8),
        )


def test_megakernel_runtime_package_has_no_model_names_and_no_legacy_exports():
    """`megakernel_runtime/*.py` has zero model literals and no legacy re-exports."""
    import ast
    from pathlib import Path

    import vllm_torchtpu.megakernel_runtime as mkr
    import vllm_torchtpu.megakernel_runtime.driver as drv_mod
    import vllm_torchtpu.megakernel_runtime.exported_bridge as eb_mod

    pkg_dir = Path(mkr.__file__).parent
    py_files = sorted(pkg_dir.glob("*.py"))
    assert len(py_files) >= 4
    forbidden_substrings = ("qwen", "llama", "deepseek", "mistral", "gemma")

    for py_path in py_files:
        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(py_path))
        # The shorthand table and the removed-export replacements name the
        # built-in templates by string.
        table = {"plugin.py": "SHORTHANDS", "__init__.py": "_MOVED"}.get(py_path.name)
        allowed: set[int] = set()
        if table is not None:
            node = next(
                n
                for n in tree.body
                if isinstance(n, (ast.Assign, ast.AnnAssign))
                and table
                in [
                    getattr(t, "id", "")
                    for t in (getattr(n, "targets", None) or [n.target])
                ]
            )
            allowed = set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        low = "\n".join(
            ln for i, ln in enumerate(src.lower().splitlines(), 1) if i not in allowed
        )
        for sub in forbidden_substrings:
            assert sub not in low, f"{py_path.name} contains model name {sub!r}"
        for node in ast.walk(tree):
            if getattr(node, "lineno", 0) in allowed:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s_low = node.value.lower()
                for sub in forbidden_substrings:
                    assert sub not in s_low, (
                        f"{py_path.name}:{getattr(node, 'lineno', 0)} string literal "
                        f"{node.value!r} contains model name {sub!r}"
                    )
            elif isinstance(node, ast.Name):
                n_low = node.id.lower()
                for sub in forbidden_substrings:
                    assert sub not in n_low, (
                        f"{py_path.name}:{getattr(node, 'lineno', 0)} identifier "
                        f"{node.id!r} contains model name {sub!r}"
                    )

    removed_driver_attrs = (
        "Qwen35Megakernel",
        "_MODEL_DRIVER_REGISTRY",
        "register_megakernel_driver",
        "resolve_model_def",
        "_DEFAULT_BUCKETS",
        "_DEFAULT_VERIFY_BUCKETS",
        "verify_pooled_step_hlo",
        "verify_torchtpu_compiled_step_ir",
        "prepare_step_metadata",
    )
    for attr in removed_driver_attrs:
        assert not hasattr(drv_mod, attr), f"driver.py still defines {attr}"
        assert not hasattr(mkr, attr), f"megakernel_runtime still exports {attr}"

    for cls_attr in ("_step_mix", "_torchtpu_ir_checked", "_ir_required_buckets"):
        for cls in (driver_mod.ProgramDriver, Qwen35Program):
            assert not hasattr(cls, cls_attr), f"{cls.__name__} defines {cls_attr}"
    for moved, hint in (
        ("MegakernelDriver", "ProgramDriver"),
        ("prepare_step_inputs_eager", "adapter.prepare_step_inputs_eager"),
    ):
        assert moved not in mkr.__all__ and not hasattr(drv_mod, moved)
        with pytest.raises(AttributeError, match=hint):
            getattr(mkr, moved)

    for attr in (
        "parse_megakernel_config",
        "parse_mk_buckets",
        "parse_mk_verify_buckets",
        "explicit_profile_settings",
        "resolve_driver_profile",
        "select_bucket",
        "check_compile_hlo_mesh",
        "MKABCalibrator",
    ):
        assert attr not in mkr.__all__, f"megakernel_runtime.__all__ still has {attr}"
        assert not hasattr(mkr, attr), f"megakernel_runtime still re-exports {attr}"
        assert not hasattr(drv_mod, attr), f"driver.py still defines {attr}"
        assert not hasattr(eb_mod, attr), f"exported_bridge.py still defines {attr}"

    for attr in (
        "_build_global_avals",
        "verify_pooled_step_hlo",
        "verify_torchtpu_compiled_step_ir",
    ):
        assert not hasattr(eb_mod, attr), f"exported_bridge.py still defines {attr}"


def _base_driver_inspect_step_detail_ref(
    attn_md: Any, m0: Any, runner: Any = None
) -> BatchShape:
    """Direct port of base (`ab8db9c5a`) `MegakernelDriver._inspect_step_detail`."""
    num_reqs: int | None = None
    if runner is not None:
        bctx = getattr(runner, "_attn_metadata_builder_ctx", None)
        if bctx is not None and getattr(bctx, "num_reqs", None) is not None:
            num_reqs = int(bctx.num_reqs)

        has_runner_roff = getattr(runner, "mamba_slot_read_offsets", None) is not None
        comb_key = getattr(runner, "_cached_comb_key", None)
        comb_cpu: Any = getattr(runner, "_combined_request_distribution_cpu", None)
        if has_runner_roff and isinstance(comb_key, tuple) and len(comb_key) == 3:
            _, num_windowed, dist_reqs = (
                int(comb_key[0]),
                int(comb_key[1]),
                int(comb_key[2]),
            )
            if num_reqs is None:
                num_reqs = dist_reqs
            if dist_reqs <= 0 or num_reqs <= 0:
                return BatchShape(False, 0, 0, "stock_n0")
            if num_windowed != dist_reqs:
                return BatchShape(
                    False,
                    num_reqs,
                    0,
                    "stock_prefill" if num_windowed == 0 else "stock_mixed",
                )
        elif (
            has_runner_roff
            and comb_cpu is not None
            and isinstance(comb_cpu, torch.Tensor)
            and comb_cpu.numel() >= 6
        ):
            num_windowed = int(comb_cpu[3].item())
            dist_reqs = int(comb_cpu[5].item())
            if num_reqs is None:
                num_reqs = dist_reqs
            if dist_reqs <= 0 or num_reqs <= 0:
                return BatchShape(False, 0, 0, "stock_n0")
            if num_windowed != dist_reqs:
                return BatchShape(
                    False,
                    num_reqs,
                    0,
                    "stock_prefill" if num_windowed == 0 else "stock_mixed",
                )

        qsl_host = getattr(runner, "query_start_loc_np", None)
        if qsl_host is None:
            qsl_host = getattr(runner, "query_start_loc_cpu", None)
        if qsl_host is not None and num_reqs is not None:
            bs = check_qsl_uniform(qsl_host, num_reqs)
            if bs.eligible and not has_runner_roff and bs.tokens_per_seq != 1:
                return BatchShape(False, bs.num_reqs, 0, "stock_prefill")
            return bs

    dist: Any = getattr(m0, "mamba_request_distribution", None)
    if dist is None:
        dist = getattr(attn_md, "mamba_request_distribution", None)
    if dist is None:
        dist = getattr(attn_md, "request_distribution", None)
    if (
        dist is not None
        and isinstance(dist, torch.Tensor)
        and dist.device.type == "cpu"
        and dist.numel() >= 3
    ):
        d_end = int(dist[0].item())
        total_reqs = int(dist[2].item())
        if total_reqs <= 0:
            return BatchShape(False, 0, 0, "stock_n0")
        if d_end != total_reqs:
            return BatchShape(
                False,
                total_reqs,
                0,
                "stock_prefill" if d_end == 0 else "stock_mixed",
            )
        if num_reqs is None:
            num_reqs = total_reqs

    num_seqs_attr = getattr(attn_md, "__dict__", {}).get("num_seqs")
    if num_reqs is None and num_seqs_attr is not None:
        if isinstance(num_seqs_attr, torch.Tensor):
            if num_seqs_attr.device.type == "cpu":
                num_reqs = int(num_seqs_attr.item())
        else:
            num_reqs = int(num_seqs_attr)

    qsl_md = getattr(attn_md, "__dict__", {}).get("query_start_loc_cpu")
    if qsl_md is None:
        qsl_dev: Any = getattr(attn_md, "query_start_loc", None)
        if (
            qsl_dev is not None
            and isinstance(qsl_dev, torch.Tensor)
            and qsl_dev.device.type == "cpu"
        ):
            qsl_md = qsl_dev
    if qsl_md is not None and num_reqs is not None:
        return check_qsl_uniform(qsl_md, num_reqs)

    return BatchShape(False, 0, 0, "stock_no_qsl")


def test_driver_inspect_step_detail_differential_and_unprepared_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Differential test of `ProgramDriver._inspect_step_detail` vs base."""
    qsl_t1 = np.array([0, 1, 2], dtype=np.int32)
    qsl_t4 = np.array([0, 4, 8], dtype=np.int32)
    qsl_mix = np.array([0, 1, 5], dtype=np.int32)
    meta_dist = torch.empty((3,), dtype=torch.int32, device="meta")
    cpu_dist_ok = torch.tensor([2, 2, 2], dtype=torch.int32)
    cpu_dist_mix = torch.tensor([1, 2, 2], dtype=torch.int32)

    for nr in (None, 0, 2):
        for has_roff in (False, True):
            for comb_key in (None, (0, 2, 2), (0, 1, 2), (0, 0, 0)):
                for qsl_h in (None, qsl_t1, qsl_t4, qsl_mix):
                    for m_dist in (None, meta_dist, cpu_dist_ok, cpu_dist_mix):
                        for ns in (
                            None,
                            0,
                            2,
                            torch.empty((1,), dtype=torch.int32, device="meta"),
                        ):
                            for q_md in (
                                None,
                                torch.tensor([0, 1, 2], dtype=torch.int32),
                                torch.tensor([0, 4, 8], dtype=torch.int32),
                                torch.empty((3,), dtype=torch.int32, device="meta"),
                            ):
                                runner = SimpleNamespace(
                                    _attn_metadata_builder_ctx=(
                                        SimpleNamespace(num_reqs=nr)
                                        if nr is not None
                                        else None
                                    ),
                                    mamba_slot_read_offsets=(
                                        torch.zeros((8,), dtype=torch.int32)
                                        if has_roff
                                        else None
                                    ),
                                    _cached_comb_key=comb_key,
                                    _combined_request_distribution_cpu=None,
                                    query_start_loc_np=qsl_h,
                                    query_start_loc_cpu=None,
                                )
                                attn_md = SimpleNamespace(query_start_loc=q_md)
                                if ns is not None:
                                    attn_md.num_seqs = ns
                                m0 = SimpleNamespace(mamba_request_distribution=m_dist)
                                got = driver_mod.ProgramDriver._inspect_step_detail(
                                    attn_md, m0, runner=runner
                                )
                                exp = _base_driver_inspect_step_detail_ref(
                                    attn_md, m0, runner=runner
                                )
                                assert got == exp

    # Also verify direct `driver(...)` without prior `can_run()` when
    # `_inspect_step_detail` returns an ineligible shape (`tokens_per_seq=0`)
    # falls back to `tokens_per_seq = 1` instead of raising `KeyError: (0, ...)`.
    mesh = _cpu_mesh(1)
    g = _tiny_geom(L=4, att_every=4)
    ex_base = _tiny_exec(R=1, B=2, T=1, ctx=256, num_pages=96)
    backbone = make_synthetic_qwen35_backbone(g, ex_base)
    model = SimpleNamespace(model=backbone)
    _allocate_aliased_pools(backbone, g, 96)
    driver = _qwen_driver(
        model=model,
        mesh=mesh,
        tp_rank=0,
        mode="ep",
        geom=g,
        base_exec=ex_base,
        buckets=(2,),
        verify_buckets=(),
    )
    driver.bind(SimpleNamespace(model=model, _unified_kv_layout=True))
    attn_md_call = SimpleNamespace(
        block_tables=torch.zeros((4,), dtype=torch.int32),
        seq_lens=torch.ones((2,), dtype=torch.int32),
        query_start_loc=None,
    )
    m_mds_call = [
        SimpleNamespace(
            mamba_state_indices=torch.zeros((2,), dtype=torch.int32),
            mamba_ckpt_indices=None,
            mamba_slot_read_offsets=None,
            mamba_request_distribution=meta_dist,
        )
        for _ in range(3)
    ]
    md_dict = {driver.program.attn_layer_name: attn_md_call}
    for name, m_md in zip(driver.program.mamba_layer_names, m_mds_call, strict=True):
        md_dict[name] = m_md
    monkeypatch.setattr(
        driver_mod,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=md_dict),
    )
    out = driver(
        torch.zeros((2,), dtype=torch.int32),
        torch.zeros((2,), dtype=torch.int32),
        num_tokens=2,
        runner=None,
    )
    assert out.shape == (2, g.H)
