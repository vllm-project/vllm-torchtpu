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
"""CPU unit tests for `Glm53DecodeOwner` and the `glm53` megakernel plugin."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel.adapter import config as mk_config
from vllm_torchtpu.megakernel.adapter import decode_owner
from vllm_torchtpu.megakernel.adapter.decode_owner_scheduler import SCHEDULER_CLS
from vllm_torchtpu.megakernel.core.runtime.contract import Decline
from vllm_torchtpu.models.vllm.glm5_next.megakernel import (
    Glm53DecodeOwner,
    SlotTable,
    _HostHeads,
    _StepTimer,
    unpack_heads,
)


class _FakeBatch:
    def __init__(self, req_ids: list[str]) -> None:
        self.req_ids = list(req_ids)
        self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
        self.num_computed_tokens_cpu = np.arange(len(req_ids), dtype=np.int32) * 10
        self.token_ids_cpu = np.zeros((len(req_ids), 64), dtype=np.int32)
        self.num_tokens = [10 * i for i in range(len(req_ids))]
        self.num_tokens_no_spec = [10 * i for i in range(len(req_ids))]

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)


def _jax_to_cpu_torch(arr: Any) -> torch.Tensor:
    a = np.ascontiguousarray(np.asarray(arr))
    name = str(a.dtype)
    if name == "bfloat16":
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    if name == "float8_e4m3fn":
        return torch.from_numpy(a.view(np.uint8)).view(torch.float8_e4m3fn)
    return torch.from_numpy(a)


def test_glm53_shorthand_selects_spec_and_configures_scheduler(
    monkeypatch: Any,
) -> None:
    hf_cfg = SimpleNamespace(
        architectures=["Glm5NextForCausalLM"],
        model_type="glm5_next",
    )
    cfg, resolved = mk_config.select_plugin("1", None, hf_cfg)
    assert cfg.enabled
    assert resolved is not None
    ref, options = resolved
    assert ref.module == "vllm_torchtpu.kernels.glm53_megakernel"
    assert ref.attr == "SPEC"
    assert ref.shorthand == "glm53"
    assert dict(options) == {}
    spec, _ = mk_config.import_plugin(ref)
    assert spec.name == "glm53"
    assert spec.tp_sizes == (8,)

    monkeypatch.setattr(envs, "VLLM_TPU_MEGAKERNEL", "plugin=glm53,max_req=16")
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_cfg),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=False),
        additional_config={},
    )
    decode_owner.configure_scheduler(vcfg)
    assert vcfg.scheduler_config.scheduler_cls == SCHEDULER_CLS

    # Config-level glm53 declines (`unsupported_max_req`, `unsupported_mla_impl`).
    monkeypatch.setattr(envs, "VLLM_TPU_MEGAKERNEL", "plugin=glm53,max_req=12")
    vcfg_bad_req = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_cfg),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=False),
        additional_config={},
    )
    decode_owner.configure_scheduler(vcfg_bad_req)
    assert vcfg_bad_req.scheduler_config.scheduler_cls is None

    monkeypatch.setattr(envs, "VLLM_TPU_MEGAKERNEL", "plugin=glm53,max_req=16")
    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_MLA_IMPL", "absorbed")
    vcfg_bad_mla = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_cfg),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(scheduler_cls=None, async_scheduling=False),
        additional_config={},
    )
    decode_owner.configure_scheduler(vcfg_bad_mla)
    assert vcfg_bad_mla.scheduler_config.scheduler_cls is None


def test_glm53_create_declines_invalid_options(monkeypatch: Any) -> None:
    import vllm_torchtpu.models.vllm.glm5_next.megakernel as mk_mod

    view = SimpleNamespace(speculative_tokens=2, _runner=SimpleNamespace())
    dec = mk_mod.create(view, {})
    assert isinstance(dec, Decline) and dec.code == "unsupported_spec_tokens"

    view_ok = SimpleNamespace(speculative_tokens=0, _runner=SimpleNamespace())
    dec2 = mk_mod.create(view_ok, {"max_req": "12"})
    assert isinstance(dec2, Decline) and dec2.code == "unsupported_max_req"

    with pytest.raises(mk_config.MegakernelConfigError, match="unknown option"):
        mk_mod.create(view_ok, {"bogus_opt": "1"})

    view_ngram = SimpleNamespace(
        speculative_tokens=3,
        _runner=SimpleNamespace(
            vllm_config=SimpleNamespace(
                speculative_config=SimpleNamespace(
                    method="ngram", num_speculative_tokens=3
                )
            )
        ),
    )
    dec_method = mk_mod.create(view_ngram, {})
    assert (
        isinstance(dec_method, Decline) and dec_method.code == "unsupported_spec_method"
    )

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        mk_mod,
        "Glm53DecodeOwner",
        lambda runner, **kw: captured.update(kw) or SimpleNamespace(**kw),
    )
    hf_cfg = SimpleNamespace(
        architectures=["Glm5NextForConditionalGeneration"], model_type="glm5_next"
    )
    cfg, resolved = mk_config.select_plugin(
        "plugin=glm53,max_req=16,debug=0,timing=1,profile_scopes=1", None, hf_cfg
    )
    assert cfg.enabled and resolved is not None
    _, opts = resolved
    mk_mod.create(view_ok, opts)
    assert captured == {
        "max_req": 16,
        "cache_kpages": None,
        "debug": False,
        "timing": True,
        "profile_scopes": True,
    }


def test_mla_attention_latent_sink_hook() -> None:
    """`Glm5NextMLAAttention._latent_sink` captures `kv_latent` cleanly."""
    from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextMLAAttention

    captured: list[torch.Tensor] = []
    attn = object.__new__(Glm5NextMLAAttention)
    torch.nn.Module.__init__(attn)
    attn.q_lora_rank = 4
    attn.kv_lora_rank = 4
    attn.num_local_heads = 1
    attn.qk_nope_head_dim = 4
    attn.qk_head_dim = 4
    attn.v_head_dim = 4
    attn.mla_attn = None
    attn.q_a_proj = lambda x: (x[:, :4], None)
    attn.q_a_layernorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5)
    attn.kv_a_proj_with_mqa = lambda x: (x[:, :4], None)
    attn.kv_a_layernorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5)
    attn.q_b_proj = lambda q: (q, None)
    attn.kv_b_proj = lambda kv: (torch.cat([kv, kv], dim=-1), None)
    attn.attn = lambda q, k, v: v
    attn.o_proj = lambda o: (o, None)
    attn._latent_sink = lambda lat: captured.append(lat.detach().clone())

    x = torch.ones((3, 4), dtype=torch.float32)
    pos = torch.arange(3, dtype=torch.int64)
    out = attn.forward(x, pos)
    assert out.shape == (3, 4)
    assert len(captured) == 1 and captured[0].shape == (3, 4)


def test_unpack_heads_plain_and_spec() -> None:
    b = 2
    plain_lp = np.ones((b, 64), dtype=np.float32)
    plain_id = np.arange(b * 64, dtype=np.int32).reshape(b, 64)
    plain_nxt = np.array([11, 22], dtype=np.int32)
    flat_plain = np.concatenate(
        [
            plain_lp.view(np.int32).reshape(-1),
            plain_id.reshape(-1),
            plain_nxt.reshape(-1),
        ]
    )
    lp, tid, nxt = unpack_heads(flat_plain, b, spec=False)
    np.testing.assert_allclose(lp, plain_lp)
    np.testing.assert_array_equal(tid, plain_id)
    np.testing.assert_array_equal(nxt, plain_nxt)

    spec_lp = np.ones((b, 4, 64), dtype=np.float32) * -0.5
    spec_id = np.zeros((b, 4, 64), dtype=np.int32)
    target = np.arange(b * 4, dtype=np.int32).reshape(b, 4)
    n_acc = np.array([3, 1], dtype=np.int32)
    nxt_v = np.arange(100, 100 + b * 4, dtype=np.int32).reshape(b, 4)
    nxt_p = np.array([10, 20], dtype=np.int32)
    flat_spec = np.concatenate(
        [
            spec_lp.view(np.int32).reshape(-1),
            spec_id.reshape(-1),
            target.reshape(-1),
            n_acc.reshape(-1),
            nxt_v.reshape(-1),
            nxt_p.reshape(-1),
        ]
    )
    s_lp, _s_id, s_tgt, s_acc, s_nv, s_np = unpack_heads(flat_spec, b, spec=True)
    np.testing.assert_allclose(s_lp, spec_lp)
    np.testing.assert_array_equal(s_tgt, target)
    np.testing.assert_array_equal(s_acc, n_acc)
    np.testing.assert_array_equal(s_nv, nxt_v)
    np.testing.assert_array_equal(s_np, nxt_p)


def test_invalidate_cached_block_evicts_recycled_block_ids() -> None:
    mk = object.__new__(Glm53DecodeOwner)
    mk.world = 4
    mk.rank = 0
    mk._cache_subpage_base = 100
    mk._cache_subpage_cap = 8
    mk._rank_cache_cursors = [0, 0, 0, 0]
    mk._rank_cached_blocks = [{}, {}, {}, {}]
    mk._rank_subpage_to_block = [{}, {}, {}, {}]

    sub0 = mk._alloc_cache_subpage_for_rank(0, 42)
    assert mk._rank_cached_blocks[0][42] == sub0
    # Reusing block 42 on rank 2 must evict the stale rank-0 mapping.
    mk._invalidate_cached_block(42, keep_rank=2)
    assert 42 not in mk._rank_cached_blocks[0]
    assert sub0 not in mk._rank_subpage_to_block[0]
    sub2 = mk._alloc_cache_subpage_for_rank(2, 42)
    assert mk._rank_cached_blocks[2][42] == sub2
    # Overwriting block 42 without a megakernel slot evicts all ranks.
    mk._invalidate_cached_block(42)
    assert all(42 not in b2s for b2s in mk._rank_cached_blocks)


def test_glm53_step_op_preserves_out_shardings_and_buffer_aliasing() -> None:
    """`Glm53DecodeOwner._get_step_op` must preserve `@main` output shardings
    (`sdy.sharding` on `"tp"`), XLA input-output aliasing (`tf.aliasing_output`
    1..4 on `conv, rec, kvq, kvs`), and per-rank local output tensor shapes."""
    import jax
    import jax.numpy as jnp
    from jax._src import mesh as jax_mesh
    from jax.sharding import AbstractMesh, AxisType
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.kernels.glm53_megakernel import (
        Glm53MegakernelConfig,
        latent_pool_shapes,
        prepare_kernel_weights_local,
        synthetic_rank_weights,
        to_per_channel_fp8,
    )
    from vllm_torchtpu.kernels.glm53_megakernel import (
        kernel as mk_kernel,
    )

    world = 2
    cfg = Glm53MegakernelConfig.small_lane_aligned_for_test(
        world_size=world, page_size=32
    )
    abs_dev = jax_mesh.AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    abs_mesh = AbstractMesh(
        (world,), ("tp",), axis_types=(AxisType.Auto,), abstract_device=abs_dev
    )

    w_raw = synthetic_rank_weights(
        cfg, rank=0, world_size=world, seed=0, include_mtp=False
    )
    cpu0 = jax.devices("cpu")[0]
    with jax.default_device(cpu0):
        w_jnp = {k: jnp.asarray(v) for k, v in w_raw.items()}
        kw_local = prepare_kernel_weights_local(
            cfg, to_per_channel_fp8(cfg, w_jnp), world_size=world, experts_are_tp=True
        )
    weight_names = list(mk_kernel.KERNEL_WEIGHT_NAMES)
    kw_torch = {k: _jax_to_cpu_torch(kw_local[k])[None] for k in weight_names}
    l_moe = cfg.num_hidden_layers - cfg.first_k_dense_replace
    e_cnt = cfg.n_routed_experts
    i_loc = cfg.moe_intermediate_size // world
    h_dim = cfg.hidden_size
    kw_torch["moe_expert_gate_up_q"] = torch.zeros(
        (1, l_moe, e_cnt, h_dim, 2 * i_loc), dtype=torch.float8_e4m3fn
    )
    kw_torch["moe_expert_gate_up_s"] = torch.ones(
        (1, l_moe, e_cnt, 1, 2 * i_loc), dtype=torch.float32
    )
    kw_torch["moe_expert_down_q"] = torch.zeros(
        (1, l_moe, e_cnt, i_loc, h_dim), dtype=torch.float8_e4m3fn
    )
    kw_torch["moe_expert_down_s"] = torch.ones(
        (1, l_moe, e_cnt, 1, h_dim), dtype=torch.float32
    )

    owner: Any = object.__new__(Glm53DecodeOwner)
    owner.cfg = cfg
    owner.world = world
    owner.spec = False
    owner.n_slots = 8
    owner.n_pages = 32
    owner._ops = {}
    owner.mesh = abs_mesh
    owner.weight_names = weight_names
    owner.kw = kw_torch

    op = owner._get_step_op(1)
    n_kpages = 10
    kvq_sh, kvs_sh = latent_pool_shapes(cfg, len(cfg.mla_layers), n_kpages)
    lk = len(cfg.kda_layers)
    hk = cfg.kda_num_heads // world
    dk = cfg.kda_head_dim

    packed = torch.zeros((4,), dtype=torch.int32)
    conv = torch.zeros(
        (1, lk, owner.n_slots, cfg.kda_conv_kernel_size - 1, 3 * hk * dk),
        dtype=torch.bfloat16,
    )
    rec = torch.zeros((1, lk, owner.n_slots, hk, dk, dk), dtype=torch.float32)
    kvq = torch.zeros((1, *kvq_sh), dtype=torch.int32)
    kvs = torch.ones((1, *kvs_sh), dtype=torch.float32)
    kpt = torch.ones((owner.n_slots, 1), dtype=torch.int32)
    rank_t = torch.tensor([0], dtype=torch.int32)
    dev_t = torch.arange(world, dtype=torch.int32)
    call_args = [
        *(kw_torch[k] for k in weight_names),
        packed,
        conv,
        rec,
        kvq,
        kvs,
        kpt,
        rank_t,
        dev_t,
    ]

    wrapped: Any = op._init_fn
    tp_spec = jax.sharding.PartitionSpec("tp")
    tp = jax.sharding.NamedSharding(abs_mesh, tp_spec)
    rep = jax.sharding.NamedSharding(abs_mesh, jax.sharding.PartitionSpec())
    to_jnp = {
        torch.bfloat16: jnp.bfloat16,
        torch.float32: jnp.float32,
        torch.int32: jnp.int32,
        torch.float8_e4m3fn: jnp.float8_e4m3fn,
    }
    jax_args = [
        jax.ShapeDtypeStruct(
            (world, *t.shape[1:]) if s == tp_spec else tuple(t.shape),
            to_jnp[t.dtype],
            sharding=tp if s == tp_spec else rep,
        )
        for t, s in zip(call_args, wrapped.input_partition_specs)
    ]
    mlir = jax.export.export(wrapped.jit_fn, platforms=("tpu",))(
        *jax_args
    ).mlir_module()
    main_header = next(
        line for line in mlir.splitlines() if "func.func public @main(" in line
    )
    for alias_idx in (1, 2, 3, 4):
        assert f"tf.aliasing_output = {alias_idx} : i32" in main_header
    ret_part = main_header.split(") -> (", 1)[1]
    assert ret_part.count('sdy.sharding = #sdy.sharding<@mesh, [{"tp"}') == 5

    with FakeTensorMode() as mode:
        fake_args = [mode.from_tensor(t) for t in call_args]
        outs = op(*fake_args)
    expected_head_w = 1 * (2 * mk_kernel.TOPK_OUT + 1)
    assert outs[0].shape == (1, expected_head_w) and outs[0].dtype == torch.int32
    assert [t.shape for t in outs[1:]] == [conv.shape, rec.shape, kvq.shape, kvs.shape]

    # Also verify `_StepTimer` records pure vs mixed steps in `run_decode`.
    owner.slots = SlotTable(8)
    st0 = owner.slots.acquire("r0")
    assert st0 is not None
    st0.owned = True
    owner.timer = _StepTimer()
    owner.debug = False
    fake_heads = (
        np.zeros((1, mk_kernel.TOPK_OUT), dtype=np.float32),
        np.zeros((1, mk_kernel.TOPK_OUT), dtype=np.int32),
        np.array([9], dtype=np.int32),
    )
    owner.decode_step = lambda _t, _p, _s: (_HostHeads(fake_heads), 1)
    runner_pure = SimpleNamespace(input_batch=_FakeBatch(["r0"]))
    owner.run_decode(runner_pure, [0])
    runner_mixed = SimpleNamespace(input_batch=_FakeBatch(["r0", "r1"]))
    owner.run_decode(runner_mixed, [0])
    assert owner.timer.counts == {"pure": 1, "mixed": 1}


def test_glm53_decode_owner_uses_rank_ordered_mesh_coordinates(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """`dev_table` holds `DeviceIdType.MESH` coordinates `0..world-1`, not raw
    PCI device IDs `[0, 1, 4, 5, 6, 7, 2, 3]`, because `get_stage_mesh` already
    orders `mesh.devices` by TP rank."""
    from vllm.distributed import parallel_state

    from vllm_torchtpu.distributed import mesh_utils
    from vllm_torchtpu.kernels.glm53_megakernel.config import Glm53MegakernelConfig

    cfg_dict = {
        "text_config": {
            "num_hidden_layers": 45,
            "first_k_dense_replace": 3,
            "hidden_size": 4096,
            "vocab_size": 154880,
            "rms_norm_eps": 1e-5,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "layer_types": [
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                for i in range(45)
            ],
            "linear_attn_config": {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -9.0,
            },
            "num_attention_heads": 64,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "intermediate_size": 12288,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "moe_intermediate_size": 2048,
            "routed_scaling_factor": 2.5,
            "swiglu_limit": 10.0,
            "norm_topk_prob": True,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "num_nextn_predict_layers": 1,
            "eos_token_id": [154820],
        },
        "quantization_config": {"weight_block_size": [128, 128]},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    Glm53MegakernelConfig.from_checkpoint(tmp_path).assert_production_glm53()

    tpu7x_pci_order = (0, 1, 4, 5, 6, 7, 2, 3)
    fake_tp = SimpleNamespace(world_size=8, rank_in_group=2)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: fake_tp)
    monkeypatch.setattr(
        mesh_utils,
        "get_cp_group_layout",
        lambda _g: SimpleNamespace(device_ids=tpu7x_pci_order),
    )
    monkeypatch.setattr(
        mesh_utils, "get_stage_mesh", lambda *_a, **_kw: "fake_rank_ordered_mesh"
    )

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model=str(tmp_path), max_model_len=8192),
            cache_config=SimpleNamespace(block_size=144),
            speculative_config=None,
        )
    )
    owner = Glm53DecodeOwner(runner, max_req=8)
    assert owner.device_ids == list(tpu7x_pci_order)
    assert owner.dev_table == [0, 1, 2, 3, 4, 5, 6, 7]

    # Cross-rank SlotTable invariance: releasing the same finished requests in
    # different iteration orders (as unsorted `set[str]` across spawned workers
    # with distinct `PYTHONHASHSEED`s would produce) must keep `slots.free`
    # identical on every rank so remote superpage DMAs address the same slot.
    owner_b = Glm53DecodeOwner(runner, max_req=8)
    for rid in ("req-a", "req-b", "req-c", "req-d"):
        s1, s2 = owner.slots.acquire(rid), owner_b.slots.acquire(rid)
        assert s1 is not None and s2 is not None and s1.slot == s2.slot
    owner.release(["req-d", "req-a", "req-c"])
    owner_b.release(["req-c", "req-d", "req-a"])
    assert owner.slots.free == owner_b.slots.free
    for rid in ("req-e", "req-f", "req-g"):
        s1, s2 = owner.slots.acquire(rid), owner_b.slots.acquire(rid)
        assert s1 is not None and s2 is not None and s1.slot == s2.slot
