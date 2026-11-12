"""The row-gated sequence parallelism: the gate module, the communicator's step helpers and the parallel-config
property, on the CPU (no devices, no process groups)."""
import importlib

import pytest


def _reload_envs(monkeypatch, min_rows):
    monkeypatch.setenv("TPU_SP_MIN_ROWS", str(min_rows))
    from vllm_torchtpu import envs
    importlib.reload(envs)
    return envs


def test_gate_module_reads_the_threshold(monkeypatch):
    _reload_envs(monkeypatch, 1024)
    from vllm_torchtpu.layers.vllm import sequence_parallel as sp
    assert sp.enabled() and sp.min_rows() == 1024
    assert sp.step_is_sequence_parallel(8192)
    assert sp.step_is_sequence_parallel(1024)
    assert not sp.step_is_sequence_parallel(32)
    sp.set_active(True)
    assert sp.is_active()
    sp.set_active(False)
    assert not sp.is_active()


def test_gate_module_off_by_default(monkeypatch):
    _reload_envs(monkeypatch, 0)
    from vllm_torchtpu.layers.vllm import sequence_parallel as sp
    assert not sp.enabled()
    assert not sp.step_is_sequence_parallel(8192)


@pytest.mark.parametrize("fp8_gather", [False, True])
def test_attention_interface_does_not_require_qk_fusion(
        monkeypatch, fp8_gather):
    import types

    import torch
    from vllm.model_executor.models import qwen3_5, qwen3_next

    import vllm_torchtpu
    from vllm_torchtpu.layers.vllm import linear_common

    monkeypatch.setenv("TPU_FUSED_QK_NORM_ROPE", "0")
    monkeypatch.setenv("TPU_SP_FP8_GATHER", "1" if fp8_gather else "0")
    _reload_envs(monkeypatch, 1024)

    class Attention:

        def forward(self, positions, hidden_states):
            return hidden_states + 1

        def _project_qkv_gate(self, qkv, positions):
            return qkv, qkv, qkv, None

        def attn(self, q, k, v):
            return q + k + v

        def o_proj(self, x):
            return x, None

    class Decoder:
        pass

    monkeypatch.setattr(qwen3_next, "Qwen3NextAttention", Attention)
    monkeypatch.setattr(qwen3_next, "Qwen3NextDecoderLayer", Decoder)
    monkeypatch.setattr(qwen3_5, "Qwen3_5DecoderLayer", Decoder)
    monkeypatch.setattr(linear_common, "prebuild_prequant_ops", lambda: None)
    vllm_torchtpu._patch_qwen3_next_sequence_parallel_by_rows()
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()

    attention = Attention()
    attention.qkv_proj = types.SimpleNamespace(
        skip_bias_add=True,
        quant_method=types.SimpleNamespace(
            apply_prequant=lambda layer, q, s, bias: q * s))
    x = torch.ones(8, 16)
    positions = torch.arange(8)
    # The decoder supplies prequant=None even on ordinary decode steps.
    torch.testing.assert_close(attention.forward(positions, x, prequant=None),
                               x + 1)
    torch.testing.assert_close(
        attention.forward(positions, x, prequant=(x, 2 * x)), 6 * x)


def test_communicator_step_helpers_follow_the_flag(monkeypatch):
    _reload_envs(monkeypatch, 1024)
    from vllm_torchtpu.distributed.tpu_communicator import \
        TpuDeviceCommunicator as C
    from vllm_torchtpu.layers.vllm import sequence_parallel as sp
    sp.set_active(True)
    assert C._sequence_parallel_step(True)
    assert not C._replicated_sequence_parallel_step(True)
    sp.set_active(False)
    assert not C._sequence_parallel_step(True)
    assert C._replicated_sequence_parallel_step(True)
    # a config that is not sequence-parallel never takes either branch
    assert not C._sequence_parallel_step(False)
    assert not C._replicated_sequence_parallel_step(False)


def test_parallel_config_property_under_the_threshold(monkeypatch):
    _reload_envs(monkeypatch, 1024)
    from vllm.config.parallel import ParallelConfig

    import vllm_torchtpu
    monkeypatch.setattr(ParallelConfig,
                        "_tpu_no_sp_moe_patch",
                        False,
                        raising=False)
    vllm_torchtpu._patch_disable_sequence_parallel_moe()
    tp8 = ParallelConfig(tensor_parallel_size=8,
                         data_parallel_size=1,
                         enable_expert_parallel=True)
    dp8 = ParallelConfig(tensor_parallel_size=1,
                         data_parallel_size=8,
                         enable_expert_parallel=True)
    tp8_no_ep = ParallelConfig(tensor_parallel_size=8,
                               data_parallel_size=1,
                               enable_expert_parallel=False)
    assert tp8.use_sequence_parallel_moe
    assert not dp8.use_sequence_parallel_moe
    assert not tp8_no_ep.use_sequence_parallel_moe


def test_parallel_config_property_off_without_the_threshold(monkeypatch):
    _reload_envs(monkeypatch, 0)
    from vllm.config.parallel import ParallelConfig

    import vllm_torchtpu
    monkeypatch.setattr(ParallelConfig,
                        "_tpu_no_sp_moe_patch",
                        False,
                        raising=False)
    vllm_torchtpu._patch_disable_sequence_parallel_moe()
    tp8 = ParallelConfig(tensor_parallel_size=8,
                         data_parallel_size=1,
                         enable_expert_parallel=True)
    assert not tp8.use_sequence_parallel_moe


def test_decoder_patch_installs_the_row_gated_forward(monkeypatch):
    _reload_envs(monkeypatch, 1024)
    from vllm.model_executor.models import qwen3_next as qn

    import vllm_torchtpu
    monkeypatch.setattr(qn.Qwen3NextDecoderLayer,
                        "_tpu_sp_by_rows_patch",
                        False,
                        raising=False)
    vllm_torchtpu._patch_qwen3_next_sequence_parallel_by_rows()
    assert qn.Qwen3NextDecoderLayer._tpu_sp_by_rows_patch
    assert qn.Qwen3NextDecoderLayer.forward.__name__ == "forward"
    # the Qwen3.5 layer inherits the patched forward but has its own __init__
    # (it skips Qwen3NextDecoderLayer.__init__), so both inits carry the rank
    # buffer wrapper.
    from vllm.model_executor.models import qwen3_5 as q35
    assert q35.Qwen3_5DecoderLayer.forward is qn.Qwen3NextDecoderLayer.forward
    assert getattr(qn.Qwen3NextDecoderLayer.__init__, "_tpu_sp_rank_init",
                   False)
    assert getattr(q35.Qwen3_5DecoderLayer.__init__, "_tpu_sp_rank_init",
                   False)


def test_runner_sets_the_flag_outside_the_compiled_region():
    # the decoder forward must not write the flag (Dynamo would emit the new
    # value as a graph output, which the TPU export refuses); the runner does
    import inspect

    import vllm_torchtpu
    from vllm_torchtpu.runner import tpu_runner
    assert "sequence_parallel.set_active(" in inspect.getsource(
        tpu_runner.TPUModelRunner.forward_model)
    src = inspect.getsource(
        vllm_torchtpu._patch_qwen3_next_sequence_parallel_by_rows)
    assert "set_active(" not in src
    # and the decoder's own decision is a tensor-shape comparison: the compile
    # flow forks a trace only on a shape guard (a Python value is frozen at the
    # first trace and cannot select a different collective for larger buckets).
    assert "positions.shape[-1] >= _sp.min_rows()" in src


def test_prequant_ops_fake_shapes(monkeypatch):
    # the two custom ops of the fp8 all-gather: their fakes give the shapes the
    # real ops return (fp8 rows, [rows, 1] float32 scales; bf16 [rows, n_out])
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.layers.vllm import linear_common as lc
    lc.prebuild_prequant_ops()
    with FakeTensorMode():
        x = torch.empty(1024, 4096, dtype=torch.bfloat16)
        q, s = lc.quantize_rows(x)
        assert q.shape == (1024, 4096) and q.dtype == torch.float8_e4m3fn
        assert s.shape == (1024, 1) and s.dtype == torch.float32
        w = torch.empty(4096, 2560, dtype=torch.float8_e4m3fn)
        ws = torch.empty(2560, dtype=torch.float32)
        y = lc.quantized_matmul_prequant(q, s, w, ws)
        assert y.shape == (1024, 2560) and y.dtype == torch.bfloat16


def test_fp8_gather_flag(monkeypatch):
    _reload_envs(monkeypatch, 1024)
    monkeypatch.setenv("TPU_SP_FP8_GATHER", "1")
    from vllm_torchtpu import envs
    importlib.reload(envs)
    from vllm_torchtpu.layers.vllm import sequence_parallel as sp
    assert sp.fp8_gather()


def test_decode_batch_flag_raises_above_the_cap(monkeypatch):
    monkeypatch.setenv("TPU_RPA_DECODE_BATCH", "16")
    monkeypatch.setenv("TPU_RPA_DECODE_VMEM_MIB",
                       "32")  # the overrides exist only with a VMEM limit set
    from vllm_torchtpu import envs
    importlib.reload(envs)
    from vllm_torchtpu.layers.common import attention_interface as ai
    importlib.reload(ai)
    with pytest.raises(ValueError):
        ai.batched_rpa_decode_overrides(64, 16)


@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("language_model_only", [None, False, True])
def test_qk_fusion_checks_positional_multimodal_config(monkeypatch, positional,
                                                       language_model_only):
    from types import SimpleNamespace

    import torch
    from vllm.model_executor.models import qwen3_next

    import vllm_torchtpu
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm import fused_qk_norm_rope_tpu as qk

    class Attention:

        def __init__(self, config, model_config=None):
            self.attn_output_gate = True
            self.rotary_emb = SimpleNamespace(is_neox_style=True,
                                              rotary_dim=64)
            self.num_heads = self.num_kv_heads = 1
            self.head_dim = 256
            self.q_norm = SimpleNamespace(variance_epsilon=1e-6)

        def _project_qkv_gate(self, qkv, positions):
            return "original"

    monkeypatch.setattr(envs, "TPU_FUSED_QK_NORM_ROPE", 1)
    monkeypatch.setattr(qwen3_next, "Qwen3NextAttention", Attention)
    monkeypatch.setattr(qwen3_next,
                        "_tpu_fused_qk_norm_rope_patch",
                        False,
                        raising=False)
    calls = []
    monkeypatch.setattr(qk, "prebuild", lambda *args: calls.append(args))
    vllm_torchtpu._patch_qwen3_next_fused_qk_norm_rope_gate()
    mm = (None if language_model_only is None else SimpleNamespace(
        language_model_only=language_model_only))
    model_config = SimpleNamespace(multimodal_config=mm)
    attention = (Attention(None, model_config)
                 if positional else Attention(None, model_config=model_config))
    expected = language_model_only is not False
    assert attention._tpu_fused_qk == expected
    assert bool(calls) == expected
    if not expected:
        assert attention._project_qkv_gate(torch.empty(1, 1024),
                                           torch.tensor([[1], [2],
                                                         [3]])) == "original"


@pytest.mark.parametrize("gathered_rows,expected", [(8, True), (256, True),
                                                    (512, True), (520, False),
                                                    (131072, False)])
def test_dp_combine_caps_output_rows_per_rank(monkeypatch, gathered_rows,
                                              expected):
    import torch

    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.vllm import small_all_reduce_op as sar

    monkeypatch.setattr(envs, "TPU_SMALL_ALL_REDUCE_MAX_ROWS", 64)
    monkeypatch.setattr(sar, "_CORES", 8)
    monkeypatch.setattr(sar, "_DP_COMBINE", True)
    monkeypatch.setattr(sar, "_RANK", None)
    monkeypatch.setattr(sar, "_MY_RANK", None)
    monkeypatch.setattr(sar, "_CORE_OF_RANK", None)
    calls = []
    monkeypatch.setattr(sar, "_RS_OP",
                        lambda x, *args: calls.append(x.shape) or "custom")
    block = torch.empty(gathered_rows,
                        4096,
                        dtype=torch.bfloat16,
                        device="meta")
    assert (sar.dp_combine(block) == "custom") == expected
    assert bool(calls) == expected
