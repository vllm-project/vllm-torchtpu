# SPDX-License-Identifier: Apache-2.0
"""Coverage tests for Deepseek V4.1 v2 model SP and mHC pre-sums."""

import unittest.mock
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from vllm.compilation.decorators import CompilationMode

from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc as v2_mhc
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model

# ---------------------------------------------------------------------------
# 1. v2/mhc.py:162 - _flat_pre_sums Pallas kernel assertion and computation
# ---------------------------------------------------------------------------


def test_mhc_flat_pre_sums_kernel_execution():
    """Verify _flat_pre_sums kernel executes line 162 assertions."""

    class MockRef:
        def __init__(self, data: np.ndarray):
            self.data = data
            self.dtype = getattr(data, "dtype", None)

        def __getitem__(self, idx: Any) -> Any:
            return self.data if idx is Ellipsis else self.data[idx]

        def __setitem__(self, idx: Any, val: Any) -> None:
            if idx is Ellipsis:
                self.data = np.asarray(val, dtype=self.data.dtype)
            else:
                self.data[idx] = np.asarray(val)

    def fake_pallas_call(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        # mult=2, dim=4, mixes=8, tokens=16, tile=16, width=mult*dim=8
        x_data = np.ones((16, 8), dtype=np.float32)
        w_data = np.ones((8, 24), dtype=np.float32)
        pre_data = np.ones((16, 2), dtype=np.float32)
        mix_data = np.zeros((16, 8), dtype=np.float32)
        square_data = np.zeros((16, 1), dtype=np.float32)
        value_data = np.zeros((16, 4), dtype=np.float32)

        x_ref = MockRef(x_data)
        w_ref = MockRef(w_data)
        pre_ref = MockRef(pre_data)
        mix_ref = MockRef(mix_data)
        square_ref = MockRef(square_data)
        value_ref = MockRef(value_data)

        kernel(x_ref, w_ref, pre_ref, mix_ref, square_ref, value_ref)

        def _invoke(flat, weight, previous_pre):
            return (
                jnp.array(mix_ref.data),
                jnp.array(square_ref.data),
                jnp.array(value_ref.data),
            )

        return _invoke

    with unittest.mock.patch(
        "jax.experimental.pallas.pallas_call", side_effect=fake_pallas_call
    ):
        flat = jnp.ones((16, 8), dtype=jnp.float32)
        weight = jnp.ones((8, 24), dtype=jnp.float32)
        previous_pre = jnp.ones((16, 2), dtype=jnp.float32)
        mixes, square, value = v2_mhc._flat_pre_sums(
            flat, weight, previous_pre, mixes=8, tile=16
        )
        assert mixes.shape == (16, 8)
        assert square.shape == (16, 1)
        assert value.shape == (16, 4)


# ---------------------------------------------------------------------------
# 2. v2/model.py: DeepseekV41DecoderLayer.__init__ and _use_sequence_parallel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("upstream_sp", "env_sp", "pp", "ep", "tp", "expected"),
    [
        (True, False, 1, False, 1, True),
        (False, True, 1, True, 4, True),
        (False, False, 1, True, 4, False),
        (False, True, 2, True, 4, False),
        (False, True, 1, False, 4, False),
        (False, True, 1, True, 1, False),
    ],
)
def test_use_sequence_parallel(
    monkeypatch: pytest.MonkeyPatch,
    upstream_sp: bool,
    env_sp: bool,
    pp: int,
    ep: bool,
    tp: int,
    expected: bool,
):
    monkeypatch.setattr(
        v2_model, "_upstream_use_sequence_parallel", lambda _: upstream_sp
    )
    monkeypatch.setattr(v2_model.envs, "TPU_DEEPSEEK_V41_SP", env_sp)
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
            tensor_parallel_size=tp,
        )
    )
    assert v2_model._use_sequence_parallel(vllm_config) is expected


def test_decoder_layer_init_branching(monkeypatch: pytest.MonkeyPatch):
    observed_sp: list[bool] = []

    def fake_init(self: Any, vllm_config: Any, prefix: str = "", **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        observed_sp.append(bool(v2_model.upstream._use_sequence_parallel(vllm_config)))

    monkeypatch.setattr(v2_model.upstream.DeepseekV4DecoderLayer, "__init__", fake_init)
    monkeypatch.setattr(v2_model.upstream, "_use_sequence_parallel", lambda _: True)
    monkeypatch.setattr(v2_model.envs, "TPU_DEEPSEEK_V41_CED", True)

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=4)),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    # Layer 0 (< ced_boundary 2): else branch
    v2_model.DeepseekV41DecoderLayer(vllm_config, prefix="model.layers.0")
    # Layer 2 (>= ced_boundary 2): if branch with _use_sequence_parallel forced False
    v2_model.DeepseekV41DecoderLayer(vllm_config, prefix="model.layers.2")
    assert observed_sp == [True, False]


# ---------------------------------------------------------------------------
# 3. v2/model.py: _StageBase and forward stage wrappers
# ---------------------------------------------------------------------------


def test_stage_base_and_wrappers():
    owner = SimpleNamespace(
        _forward_stage=lambda *args, **kwargs: (args, kwargs),
    )
    stage_with_tag = v2_model._StageBase(
        owner=owner, vllm_config=None, prefix="base", compile_tag="tag1"
    )
    assert stage_with_tag.prefix == "tag1"
    assert object.__getattribute__(stage_with_tag, "owner") is owner

    stage_no_tag = v2_model._StageBase(
        owner=owner, vllm_config=None, prefix="base", compile_tag=None
    )
    assert not hasattr(stage_no_tag, "prefix")

    cfg = SimpleNamespace(compilation_config=SimpleNamespace(mode=CompilationMode.NONE))

    # Full stage
    full = v2_model.DeepseekV41FullStage(
        owner=owner, vllm_config=cfg, prefix="f", compile_tag="full_tag"
    )
    res = full.forward("ids", "pos", "inter", "embeds", "lookback")
    assert res[1]["stage"] == "full"

    # Encoder stage
    encoder = v2_model.DeepseekV41EncoderStage(
        owner=owner, vllm_config=cfg, prefix="e", compile_tag="enc_tag"
    )
    res = encoder.forward("ids", "pos", "inter", "embeds", "lookback")
    assert res[1]["stage"] == "encoder"

    # Decoder stage
    decoder = v2_model.DeepseekV41DecoderStage(
        owner=owner, vllm_config=cfg, prefix="d", compile_tag="dec_tag"
    )
    res = decoder.forward("ids", "pos", "embeds", "pre_mix")
    assert res[1]["stage"] == "decoder"
    assert res[1]["encoder_pre_mix"] == "pre_mix"


# ---------------------------------------------------------------------------
# 4. v2/model.py: DeepseekV41Model helper methods and _prepare_engram
# ---------------------------------------------------------------------------


def test_has_cpu_offloaded_engram():
    model = SimpleNamespace(
        start_layer=0,
        end_layer=3,
        layers=[
            SimpleNamespace(engram=None),
            SimpleNamespace(
                engram=SimpleNamespace(embed_tokens=SimpleNamespace(host_offload=False))
            ),
            SimpleNamespace(
                engram=SimpleNamespace(embed_tokens=SimpleNamespace(host_offload=True))
            ),
        ],
    )
    assert v2_model.DeepseekV41Model._has_cpu_offloaded_engram(model) is True

    model.layers[2].engram.embed_tokens.host_offload = False
    assert v2_model.DeepseekV41Model._has_cpu_offloaded_engram(model) is False


def test_prepare_engram_early_exits(monkeypatch: pytest.MonkeyPatch):
    model = SimpleNamespace(
        engram_hash=None,
        _has_cpu_offloaded_engram=lambda: False,
    )
    pos = torch.zeros(4, dtype=torch.int32)
    ids = torch.zeros(4, dtype=torch.int32)

    # stage == 'decoder'
    assert v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, None, stage="decoder"
    ) == (None, None)

    # engram_hash is None
    assert v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, None, stage="full"
    ) == (None, None)

    # input_ids is None
    model.engram_hash = SimpleNamespace()
    assert v2_model.DeepseekV41Model._prepare_engram(
        model, None, pos, None, stage="full"
    ) == (None, None)

    # forward context unavailable
    monkeypatch.setattr(
        v2_model.upstream, "is_forward_context_available", lambda: False
    )
    assert v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, None, stage="full"
    ) == (None, None)


def test_prepare_engram_full_path(monkeypatch: pytest.MonkeyPatch):
    @dataclass
    class _FakeSwaMeta:
        seq_lens: torch.Tensor
        query_start_loc: torch.Tensor
        block_table: torch.Tensor | None = None

    @dataclass
    class _FakeForwardCtx:
        attn_metadata: Any
        dp_metadata: Any = None

    prepared_embeddings: list[torch.Tensor] = []

    class _FakeEngram:
        layer_hash_index = 0

        def prepare_embeddings(self, h: torch.Tensor) -> None:
            prepared_embeddings.append(h)

    class _FakeEngramHash:
        def __init__(self):
            self.multipliers = torch.zeros((3, 1))
            self.lookback_depth = 4
            self.use_slot_cache = True
            self.swa_cache_module = SimpleNamespace(kv_cache=torch.zeros(1))

        def ensure_cache(self) -> bool:
            return True

        def dummy_hashes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros((x.shape[0], 3, 1)), torch.ones(
                x.shape[0], dtype=torch.bool
            )

        def __call__(self, *args: Any) -> torch.Tensor:
            return torch.ones((4, 3, 1), dtype=torch.int32)

    model = SimpleNamespace(
        engram_hash=_FakeEngramHash(),
        engram_swa_prefix="swa",
        engram_dp_shared_memory=False,
        start_layer=0,
        end_layer=1,
        layers=[SimpleNamespace(engram=_FakeEngram())],
        _has_cpu_offloaded_engram=lambda: True,
        # Device hash path; the host path is covered by test_dsv41_v2_host_engram.
        _host_hash_inputs=lambda *args: None,
    )

    monkeypatch.setattr(v2_model.upstream, "is_forward_context_available", lambda: True)
    meta = _FakeSwaMeta(
        seq_lens=torch.tensor([4]), query_start_loc=torch.tensor([0, 4])
    )
    ctx = _FakeForwardCtx(attn_metadata=[{"swa": meta}])
    monkeypatch.setattr(v2_model.upstream, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(v2_model.upstream, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(
        v2_model.upstream,
        "image_sentinel_mask",
        lambda t: torch.zeros_like(t, dtype=torch.bool),
    )
    monkeypatch.setattr(
        v2_model.upstream, "gather_engram_hashes", lambda h, dp_shared_memory: h
    )
    monkeypatch.setattr(v2_model, "_cache_slots", lambda *a: None)
    monkeypatch.setattr(v2_model, "_block_table", lambda *a: None)

    ids = torch.arange(4, dtype=torch.int32)
    pos = torch.arange(4, dtype=torch.int32)

    # 1. skip_lookup=True
    h, m = v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, None, skip_lookup=True
    )
    assert h is not None and h.shape == (4, 3, 1)
    assert m is not None

    # 2. skip_lookup=False, lookback_token_ids=None, use_slot_cache=False
    model.engram_hash.use_slot_cache = False
    with pytest.raises(NotImplementedError, match="lookback_token_ids"):
        v2_model.DeepseekV41Model._prepare_engram(
            model, ids, pos, None, skip_lookup=False
        )

    # 3. skip_lookup=False, lookback_token_ids=None, use_slot_cache=True
    model.engram_hash.use_slot_cache = True
    h, m = v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, None, skip_lookup=False
    )
    assert h is not None and h.shape == (4, 3, 1)
    assert len(prepared_embeddings) == 1

    # 4. Fallback dummy_hashes collective branch (ensure_cache() is False, DP size > 1)
    model.engram_hash.ensure_cache = lambda: False
    monkeypatch.setattr(v2_model.upstream, "get_engram_dp_size", lambda: 2)
    h_dummy, m_dummy = v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, None, skip_lookup=False
    )
    assert h_dummy is not None and m_dummy is not None

    # 5. lookback_token_ids already provided (not None)
    model.engram_hash.ensure_cache = lambda: True
    lookback = torch.zeros((1, 4), dtype=torch.int32)
    h_lookback, _ = v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, lookback, skip_lookup=False
    )
    assert h_lookback is not None

    # 6. attn_metadata is direct dict (not list)
    ctx.attn_metadata = {"swa": meta}
    h_dict, _ = v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, lookback, skip_lookup=False
    )
    assert h_dict is not None

    # 7. gathered_hashes on TPU/non-cpu converted to cpu
    class _FakeTpuHashes(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="tpu")

        def to(self, device):
            return torch.zeros((4, 3, 1), dtype=torch.int32)

    monkeypatch.setattr(
        v2_model.upstream,
        "gather_engram_hashes",
        lambda h, dp_shared_memory: torch.zeros(
            (4, 3, 1), dtype=torch.int32
        ).as_subclass(_FakeTpuHashes),
    )
    model.layers.append(SimpleNamespace(engram=None))
    model.end_layer = len(model.layers)
    h_cpu, _ = v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, lookback, skip_lookup=False
    )
    assert h_cpu is not None

    # 8. engram_dp_shared_memory=True when ensure_cache() is False
    model.engram_hash.ensure_cache = lambda: False
    model.engram_dp_shared_memory = True
    h_none, m_none = v2_model.DeepseekV41Model._prepare_engram(
        model, ids, pos, lookback, skip_lookup=False
    )
    assert h_none is None and m_none is None


# ---------------------------------------------------------------------------
# 5. v2/model.py: DeepseekV41Model.forward branching
# ---------------------------------------------------------------------------


def test_model_forward_full_and_ced_paths(monkeypatch: pytest.MonkeyPatch):
    @dataclass
    class _FakeMeta:
        seq_lens: torch.Tensor
        query_start_loc: torch.Tensor
        input_positions: torch.Tensor | None = None

    @dataclass
    class _FakeCtx:
        attn_metadata: Any
        dp_metadata: Any = None
        is_padding: Any = None
        additional_kwargs: dict[str, Any] = field(default_factory=dict)

    class _FakeModel:
        def __init__(self):
            self._ced_enabled = False
            self.use_native_mega_moe = False
            self.use_sequence_parallel = False
            self.config = SimpleNamespace(hidden_size=16)
            self._ced_window = 4
            self._ced_compile_sizes = (4,)

        def _has_cpu_offloaded_engram(self):
            return False

        def _prepare_engram(self, *args, **kwargs):
            return None, None

        def full_stage(self, *args, **kwargs):
            return torch.ones((4, 16), dtype=torch.bfloat16)

        def encoder_stage(self, *args, **kwargs):
            return [
                (
                    torch.ones((4, 16), dtype=torch.bfloat16),
                    torch.ones((4, 4), dtype=torch.float32),
                )
            ]

        def decoder_stage(self, *args, **kwargs):
            return [torch.ones((4, 16), dtype=torch.bfloat16)]

        _stage_uses_sequence_parallel = (
            v2_model.DeepseekV41Model._stage_uses_sequence_parallel
        )

    model = _FakeModel()
    ids = torch.arange(4, dtype=torch.int32)
    pos = torch.arange(4, dtype=torch.int32)

    # 1. offload_engram = True with use_native_mega_moe = True
    model._has_cpu_offloaded_engram = lambda: True
    model.use_native_mega_moe = True
    out_full = v2_model.DeepseekV41Model.forward(model, ids, pos)
    assert out_full.shape == (4, 16)

    # 2. CED enabled, plan is None
    model._ced_enabled = True
    ctx = _FakeCtx(attn_metadata=None)
    monkeypatch.setattr(v2_model.upstream, "get_forward_context", lambda: ctx)
    out_ced_no_plan = v2_model.DeepseekV41Model.forward(model, ids, pos)
    assert out_ced_no_plan.shape == (4, 16)

    # 3. CED enabled, plan is not None with SP and TPU device synchronization
    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    hidden_tpu = torch.ones((4, 16), dtype=torch.bfloat16).as_subclass(_FakeTpuTensor)
    pre_mix_tpu = torch.ones((4, 4), dtype=torch.float32).as_subclass(_FakeTpuTensor)

    model.use_sequence_parallel = True
    model.encoder_stage = lambda *args, **kwargs: (hidden_tpu, pre_mix_tpu)
    model.decoder_stage = lambda *args, **kwargs: torch.ones(
        (4, 16), dtype=torch.bfloat16
    )

    plan_obj = SimpleNamespace(
        indices=(0, 1),
        padded_num_tokens=4,
        query_start_loc=(0, 2),
        replay_starts=(0,),
    )
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *args: plan_obj)
    monkeypatch.setattr(v2_model.upstream, "sp_all_gather", lambda t: t)
    synced: list[Any] = []
    monkeypatch.setattr(
        v2_model, "synchronize_tensors", lambda pair: synced.append(pair)
    )

    meta_mock = _FakeMeta(
        seq_lens=torch.tensor([4]), query_start_loc=torch.tensor([0, 4])
    )
    ctx_plan = _FakeCtx(attn_metadata={"l0": meta_mock})
    monkeypatch.setattr(v2_model.upstream, "get_forward_context", lambda: ctx_plan)

    out_plan = v2_model.DeepseekV41Model.forward(model, ids, pos)
    assert out_plan.shape == (4, 16)
    assert len(synced) == 1

    # 4. CED enabled with empty plan.indices -> early return zeros
    plan_empty = SimpleNamespace(
        indices=(),
        padded_num_tokens=0,
        query_start_loc=(0, 0),
        replay_starts=(0,),
    )
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *args: plan_empty)
    out_empty = v2_model.DeepseekV41Model.forward(model, ids, pos)
    assert out_empty.shape == (4, 16)
    assert torch.count_nonzero(out_empty) == 0

    # 5. CED enabled with hidden on CPU (skips synchronize_tensors)
    hidden_cpu = torch.ones((4, 16), dtype=torch.bfloat16)
    pre_mix_cpu = torch.ones((4, 4), dtype=torch.float32)
    model.encoder_stage = lambda *args, **kwargs: (hidden_cpu, pre_mix_cpu)
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *args: plan_obj)
    synced.clear()
    out_cpu = v2_model.DeepseekV41Model.forward(model, ids, pos)
    assert out_cpu.shape == (4, 16)
    assert len(synced) == 0

    # 6. offload_engram = False with use_native_mega_moe = False
    model._has_cpu_offloaded_engram = lambda: False
    model.use_native_mega_moe = False
    out_no_offload = v2_model.DeepseekV41Model.forward(model, ids, pos)
    assert out_no_offload.shape == (4, 16)


# ---------------------------------------------------------------------------
# 6. v2/model.py: DeepseekV41Model._forward_stage
# ---------------------------------------------------------------------------


def test_forward_stage_decoder_and_encoder_paths(monkeypatch: pytest.MonkeyPatch):
    class _FakeCallableLayer:
        def __call__(
            self,
            hidden: torch.Tensor,
            pos: torch.Tensor,
            ids: torch.Tensor,
            pre: torch.Tensor,
            post: Any,
            res_mix: Any,
            res: Any,
            eh: Any,
            em: Any,
            capture_previous_aux: bool = False,
        ) -> tuple[Any, ...]:
            out_pre = (
                pre
                if pre is not None
                else torch.ones((hidden.shape[0], 4), dtype=torch.float32)
            )
            return hidden, None, None, None, out_pre, None

    model = SimpleNamespace(
        _ced_boundary=2,
        start_layer=0,
        end_layer=4,
        _stage_uses_sequence_parallel=lambda stage: False,
        use_native_mega_moe=False,
        _prepare_engram=lambda *a, **kw: (None, None),
        aux_hidden_state_layers=set(),
        layers=[_FakeCallableLayer() for _ in range(4)],
        norm=lambda x: x,
        _mtp_hidden_buffer=None,
    )
    ids = torch.arange(4, dtype=torch.int32)
    pos = torch.arange(4, dtype=torch.int32)
    embeds = torch.ones((4, 16), dtype=torch.bfloat16)
    pre_mix = torch.ones((4, 4), dtype=torch.float32)

    monkeypatch.setattr(
        v2_model.upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(v2_model, "post", lambda h, *a: h)
    monkeypatch.setattr(v2_model, "collapse", lambda h, pm: h)

    out_dec = v2_model.DeepseekV41Model._forward_stage(
        model,
        ids,
        pos,
        None,
        inputs_embeds=embeds,
        encoder_pre_mix=pre_mix,
        stage="decoder",
    )
    assert out_dec.shape == (4, 16)

    # Test stage == "encoder" with SP and global kv write
    written_kv: list[torch.Tensor] = []

    class _FakeDecoderLayer:
        def attn_norm(self, x: torch.Tensor) -> torch.Tensor:
            return x

        attn = SimpleNamespace(
            write_encoder_global_kv=lambda norm, pos, sp_sharded=False: (
                written_kv.append(norm)
            )
        )

        def __call__(self, *args, **kwargs):
            hidden = args[0]
            pre = (
                args[3]
                if len(args) > 3 and args[3] is not None
                else torch.ones((hidden.shape[0], 4), dtype=torch.float32)
            )
            return hidden, None, None, None, pre, None

    model.layers[2] = _FakeDecoderLayer()
    model._stage_uses_sequence_parallel = lambda stage: True
    monkeypatch.setattr(
        v2_model.upstream, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(v2_model.upstream, "sp_shard", lambda t: t)
    monkeypatch.setattr(v2_model.upstream, "sp_all_gather", lambda t: t)
    monkeypatch.setattr(
        v2_model.upstream, "is_forward_context_available", lambda: False
    )

    h_enc, pm_enc = v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, None, inputs_embeds=embeds, stage="encoder"
    )
    assert h_enc.shape == (4, 16)
    assert pm_enc.shape == (4, 4)
    assert len(written_kv) == 1

    # embeds is None on first rank -> calls embed_input_ids
    model.embed_input_ids = lambda i: torch.ones((i.shape[0], 16), dtype=torch.bfloat16)
    out_embed = v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, None, inputs_embeds=None, stage="full"
    )
    assert out_embed.shape == (4, 16)

    # stage_sp=False for encoder
    model._stage_uses_sequence_parallel = lambda stage: False
    h_enc_nosp, pm_enc_nosp = v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, None, inputs_embeds=embeds, stage="encoder"
    )
    assert h_enc_nosp.shape == (4, 16)
    assert pm_enc_nosp.shape == (4, 4)


def test_forward_stage_pipeline_parallel_and_aux_hidden_states(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeAuxLayer:
        def __call__(
            self,
            hidden: torch.Tensor,
            pos: torch.Tensor,
            ids: torch.Tensor,
            pre: torch.Tensor,
            post: Any,
            res_mix: Any,
            res: Any,
            eh: Any,
            em: Any,
            capture_previous_aux: bool = False,
        ) -> tuple[Any, ...]:
            prev = (
                torch.ones((4, 16), dtype=torch.bfloat16)
                if capture_previous_aux
                else None
            )
            return hidden, None, None, None, pre, prev

    model = SimpleNamespace(
        _ced_boundary=2,
        start_layer=0,
        end_layer=2,
        _stage_uses_sequence_parallel=lambda stage: True,
        use_native_mega_moe=True,
        _prepare_engram=lambda *a, **kw: (None, None),
        aux_hidden_state_layers={1, 2},
        layers=[_FakeAuxLayer(), _FakeAuxLayer()],
        norm=lambda x: x,
        _mtp_hidden_buffer=torch.zeros((8, 16), dtype=torch.bfloat16),
    )
    ids = torch.arange(4, dtype=torch.int32)
    pos = torch.arange(4, dtype=torch.int32)

    # Intermediate non-first rank
    monkeypatch.setattr(
        v2_model.upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    monkeypatch.setattr(
        v2_model.upstream, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(v2_model.upstream, "sp_shard", lambda t: t)
    monkeypatch.setattr(v2_model.upstream, "sp_all_gather", lambda t: t)
    monkeypatch.setattr(v2_model.upstream, "is_forward_context_available", lambda: True)

    @dataclass
    class _FakeForwardCtx:
        is_padding: Any = torch.zeros(4, dtype=torch.bool)

    ctx = _FakeForwardCtx()
    monkeypatch.setattr(v2_model.upstream, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(v2_model.upstream.envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(v2_model.upstream, "sp_padding_mask", lambda mask, h: mask)
    monkeypatch.setattr(v2_model, "post", lambda h, *a: h)
    monkeypatch.setattr(v2_model, "collapse", lambda h, pm: h)

    inter_tensors = {
        "hidden_states": torch.ones((4, 16), dtype=torch.bfloat16),
        "pre_mix": torch.ones((4, 4), dtype=torch.float32),
    }
    res_inter = v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, intermediate_tensors=inter_tensors, stage="full"
    )
    assert isinstance(res_inter, v2_model.upstream.IntermediateTensors)

    # Now make it is_last_rank=True with aux_hidden_states returned
    monkeypatch.setattr(
        v2_model.upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    res_last = v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, intermediate_tensors=inter_tensors, stage="full"
    )
    assert isinstance(res_last, tuple)
    h_out, aux_out = res_last
    assert h_out.shape == (4, 16)
    assert len(aux_out) > 0

    # Without MTP buffer with stage_sp=True -> gathers collapsed hidden states
    model._mtp_hidden_buffer = None
    res_no_mtp = v2_model.DeepseekV41Model._forward_stage(
        model,
        ids,
        pos,
        intermediate_tensors=inter_tensors,
        encoder_pre_mix=inter_tensors["pre_mix"],
        stage="full",
    )
    assert isinstance(res_no_mtp, tuple)

    # stage_sp=False with previous_aux and final_aux
    model._stage_uses_sequence_parallel = lambda stage: False
    res_nosp_aux = v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, intermediate_tensors=inter_tensors, stage="full"
    )
    assert isinstance(res_nosp_aux, tuple)

    # MTP buffer with stage_sp=False
    model._mtp_hidden_buffer = torch.zeros((8, 16), dtype=torch.bfloat16)
    res_mtp_nosp = v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, intermediate_tensors=inter_tensors, stage="full"
    )
    assert isinstance(res_mtp_nosp, tuple)


# ---------------------------------------------------------------------------
# 7. v2/model.py: finalize_mhc_broadcast_weights
# ---------------------------------------------------------------------------


def test_finalize_mhc_broadcast_weights(monkeypatch: pytest.MonkeyPatch):
    class _FakeLayer:
        def __init__(self, dtype: torch.dtype, dev: str):
            self.hc_attn_fn = torch.ones((4, 4), dtype=dtype, device=dev)
            self.hc_ffn_fn = torch.ones((4, 4), dtype=dtype, device=dev)

    class _MockModel(v2_model.DeepseekV41Model):
        def __init__(self, layers: list[Any]):
            torch.nn.Module.__init__(self)
            self.start_layer = 0
            self.end_layer = len(layers)
            self.layers = layers

    # 1. Already packed (dtype != float32)
    model = _MockModel([_FakeLayer(torch.bfloat16, "cpu")])
    model.finalize_mhc_broadcast_weights()
    assert model.layers[0].hc_attn_fn.dtype == torch.bfloat16

    # 2. Float32 on TPU -> packed
    super_called: list[bool] = []
    monkeypatch.setattr(
        v2_model.upstream.DeepseekV4Model,
        "finalize_mhc_broadcast_weights",
        lambda self: super_called.append(True),
    )
    monkeypatch.setattr(v2_mhc, "pack_mix_weight", lambda t: t.to(torch.bfloat16))

    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

    layer_tpu = SimpleNamespace(
        hc_attn_fn=torch.ones((4, 4), dtype=torch.float32).as_subclass(_FakeTpuTensor),
        hc_ffn_fn=torch.ones((4, 4), dtype=torch.float32).as_subclass(_FakeTpuTensor),
    )
    model_tpu = _MockModel([layer_tpu])
    model_tpu.finalize_mhc_broadcast_weights()
    assert len(super_called) == 1
    assert model_tpu.layers[0].hc_attn_fn.dtype == torch.bfloat16

    # 3. Layer with CPU device (not TPU) remains un-packed
    layer_tpu2 = SimpleNamespace(
        hc_attn_fn=torch.ones((4, 4), dtype=torch.float32).as_subclass(_FakeTpuTensor),
        hc_ffn_fn=torch.ones((4, 4), dtype=torch.float32).as_subclass(_FakeTpuTensor),
    )
    layer_cpu = SimpleNamespace(
        hc_attn_fn=torch.ones((4, 4), dtype=torch.float32),
        hc_ffn_fn=torch.ones((4, 4), dtype=torch.float32),
    )
    model_mixed = _MockModel([layer_tpu2, layer_cpu])
    model_mixed.finalize_mhc_broadcast_weights()
    assert model_mixed.layers[0].hc_attn_fn.dtype == torch.bfloat16
    assert model_mixed.layers[1].hc_attn_fn.dtype == torch.float32


# ---------------------------------------------------------------------------
# 8. v2/model.py: DeepseekV41Model.__init__ validation and setup
# ---------------------------------------------------------------------------


def test_model_init_validation_and_ced_setup(monkeypatch: pytest.MonkeyPatch):
    def fake_super_init(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
        torch.nn.Module.__init__(self)
        self.engram_dp_shared_memory = False
        self.use_sequence_parallel = bool(v2_model._use_sequence_parallel(vllm_config))
        self.config = SimpleNamespace(num_hidden_layers=4, hidden_size=16)
        self.start_layer = 0
        self.end_layer = 4
        self.layers = [
            SimpleNamespace(
                attn=SimpleNamespace(
                    window_size=4, compressor=None if i != 2 else SimpleNamespace()
                ),
                engram=None,
            )
            for i in range(4)
        ]
        if getattr(vllm_config, "_test_host_offload", False):
            self.layers[0].engram = SimpleNamespace(
                embed_tokens=SimpleNamespace(host_offload=True)
            )

    monkeypatch.setattr(v2_model.upstream.DeepseekV4Model, "__init__", fake_super_init)
    monkeypatch.setattr(v2_model, "initialize_cache_ops", lambda: None)
    monkeypatch.setattr(v2_model, "initialize_delayed_mhc", lambda: None)
    monkeypatch.setattr(v2_model, "initialize_mhc_post", lambda: None)
    monkeypatch.setattr(v2_model, "initialize_paged_attention", lambda: None)
    monkeypatch.setattr(v2_model, "initialize_indexer", lambda: None)
    monkeypatch.setattr(v2_model, "compact_rope_caches", lambda *a: None)

    # 1. SP validation: compile_sizes < 2 * tp
    monkeypatch.setattr(v2_model, "_use_sequence_parallel", lambda _: True)
    vllm_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, prefill_context_parallel_size=1
        ),
        compilation_config=SimpleNamespace(
            compile_sizes=[4], mode=CompilationMode.NONE
        ),
        model_config=SimpleNamespace(max_model_len=128),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    with pytest.raises(ValueError, match="needs every token bucket >= 2 \\* TP"):
        v2_model.DeepseekV41Model(vllm_config=vllm_cfg)

    # 2. CED with prefix caching -> warning and disables CED
    vllm_cfg.compilation_config.compile_sizes = [16]
    vllm_cfg.cache_config.enable_prefix_caching = True
    monkeypatch.setattr(v2_model.envs, "TPU_DEEPSEEK_V41_CED", True)
    m = v2_model.DeepseekV41Model(vllm_config=vllm_cfg)
    assert m._ced_enabled is False
    assert m.engram_dp_shared_memory is False
    vllm_cfg._test_host_offload = True
    shared = v2_model.DeepseekV41Model(vllm_config=vllm_cfg)
    assert shared.engram_dp_shared_memory is True
    vllm_cfg._test_host_offload = False

    # 3. CED with prefill_context_parallel_size != 1 -> NotImplementedError
    vllm_cfg.cache_config.enable_prefix_caching = False
    vllm_cfg.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(NotImplementedError, match="unsharded token rows"):
        v2_model.DeepseekV41Model(vllm_config=vllm_cfg)

    # 4. CED with wrong KV sources -> NotImplementedError
    vllm_cfg.parallel_config.prefill_context_parallel_size = 1

    def fake_super_bad_sources(
        self: Any, *, vllm_config: Any, prefix: str = ""
    ) -> None:
        fake_super_init(self, vllm_config=vllm_config, prefix=prefix)
        self.layers[3].attn.compressor = SimpleNamespace()

    monkeypatch.setattr(
        v2_model.upstream.DeepseekV4Model, "__init__", fake_super_bad_sources
    )
    with pytest.raises(NotImplementedError, match="one decoder KV source"):
        v2_model.DeepseekV41Model(vllm_config=vllm_cfg)

    # 5. CED with decoder engram layers -> NotImplementedError
    def fake_super_dec_engram(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
        fake_super_init(self, vllm_config=vllm_config, prefix=prefix)
        self.layers[3].engram = SimpleNamespace(
            embed_tokens=SimpleNamespace(host_offload=False)
        )

    monkeypatch.setattr(
        v2_model.upstream.DeepseekV4Model, "__init__", fake_super_dec_engram
    )
    with pytest.raises(NotImplementedError, match="Engram layers in the encoder"):
        v2_model.DeepseekV41Model(vllm_config=vllm_cfg)

    # 6. Valid CED setup
    monkeypatch.setattr(v2_model.upstream.DeepseekV4Model, "__init__", fake_super_init)
    valid_model = v2_model.DeepseekV41Model(vllm_config=vllm_cfg)
    assert valid_model._ced_enabled is True
    assert hasattr(valid_model, "encoder_stage")
    assert hasattr(valid_model, "decoder_stage")

    # 7. use_sequence_parallel is False (skips SP check)
    monkeypatch.setattr(v2_model, "_use_sequence_parallel", lambda _: False)
    vllm_cfg.compilation_config.compile_sizes = [4]
    m_nosp = v2_model.DeepseekV41Model(vllm_config=vllm_cfg)
    assert m_nosp.use_sequence_parallel is False

    # 8. compile_sizes is empty with SP
    monkeypatch.setattr(v2_model, "_use_sequence_parallel", lambda _: True)
    vllm_cfg.compilation_config.compile_sizes = []
    m_empty = v2_model.DeepseekV41Model(vllm_config=vllm_cfg)
    assert m_empty.use_sequence_parallel is True


# ---------------------------------------------------------------------------
# 9. v2/model.py: DeepseekV41ForCausalLM
# ---------------------------------------------------------------------------


def test_deepseek_v41_causal_lm(monkeypatch: pytest.MonkeyPatch):
    def fake_causal_init(self, *a, **kw):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(expert_dtype="fp8")

    monkeypatch.setattr(
        v2_model.upstream.DeepseekV41LLMForCausalLM,
        "__init__",
        fake_causal_init,
    )
    monkeypatch.setattr(
        v2_model.upstream.DeepseekV41LLMForCausalLM,
        "forward",
        lambda self, *a, **kw: "causal_out",
    )
    monkeypatch.setattr(
        v2_model.upstream, "_make_deepseek_v4_weights_mapper", lambda *a: "mapper"
    )

    vllm_cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )

    with unittest.mock.patch(
        "vllm_torchtpu.models.vllm.deepseek_v41.v2.async_fence.install_lookback_fence"
    ) as mock_fence:
        causal = v2_model.DeepseekV41ForCausalLM(vllm_config=vllm_cfg)
        mock_fence.assert_called_once()
        assert causal.hf_to_vllm_mapper == "mapper"

        causal.model = SimpleNamespace(
            engram_hash=SimpleNamespace(ensure_cache=unittest.mock.MagicMock())
        )
        out = causal.forward()
        assert out == "causal_out"
        causal.model.engram_hash.ensure_cache.assert_called_once()

        # causal.model.engram_hash is None -> skips ensure_cache
        causal.model.engram_hash = None
        out_no_hash = causal.forward()
        assert out_no_hash == "causal_out"

    # async_scheduling is False -> skips install_lookback_fence
    vllm_cfg.scheduler_config.async_scheduling = False
    causal_no_async = v2_model.DeepseekV41ForCausalLM(vllm_config=vllm_cfg)
    assert causal_no_async is not None

    # Capability guard exceptions
    vllm_cfg.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(NotImplementedError, match="requires PP1"):
        v2_model.DeepseekV41ForCausalLM(vllm_config=vllm_cfg)

    vllm_cfg.parallel_config.pipeline_parallel_size = 1
    vllm_cfg.speculative_config = SimpleNamespace(method="mtp")
    with pytest.raises(
        NotImplementedError, match="does not support speculative method"
    ):
        v2_model.DeepseekV41ForCausalLM(vllm_config=vllm_cfg)
