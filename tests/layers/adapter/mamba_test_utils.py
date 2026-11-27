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
"""Shared Mamba-2 and Mamba-3 test fixtures, JAX op bridges, and reference oracle."""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from jax.sharding import Mesh
from safetensors.torch import load_file, save_file

# Offline golden fixtures generated directly from upstream `state-spaces/mamba`
# (`mamba_ssm/modules/mamba3.py`, `mamba_ssm/modules/mlp.py`,
# `tests/ops/triton/test_mamba3_siso.py::mamba3_siso_step_ref`, and
# `tests/ops/tilelang/test_mamba3_mimo.py::mamba3_MIMO_chunk_ref`) at commit:
UPSTREAM_MAMBA_SSM_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
MAMBA3_GOLDEN_PROMPT_IDS: tuple[int, ...] = (4, 11, 27, 42, 19)
MAMBA3_SISO_UPSTREAM_GOLDEN_GREEDY: tuple[int, ...] = (19, 67, 67, 89)
MAMBA3_MIMO_UPSTREAM_GOLDEN_GREEDY: tuple[int, ...] = (19, 19, 19, 19)
MAMBA3_SISO_UPSTREAM_GOLDEN_LOGITS: tuple[float, ...] = (
    0.5430256,
    -0.3927312,
    0.709416,
    0.5319416,
    -0.0802019,
    -0.3364887,
    -0.0626547,
    -0.0283399,
    -0.2379595,
    0.5643261,
    -0.0622821,
    0.2925138,
    0.5277873,
    0.7601159,
    -0.9880454,
    1.4054391,
    -0.4664281,
    0.2663198,
    -0.6431195,
    2.3820136,
    -0.1218599,
    0.0671847,
    1.4990749,
    0.813377,
    0.0548268,
    1.5502981,
    0.4840742,
    0.8494359,
    -0.1359196,
    -0.0614817,
    0.1566301,
    -0.223757,
    -0.2024407,
    0.2960829,
    0.7545869,
    0.9602647,
    -0.5566549,
    -0.181175,
    0.1435542,
    -0.842068,
    0.1796866,
    0.3602853,
    -1.1902617,
    -0.0084473,
    -0.821399,
    -0.6492442,
    0.6980948,
    0.439073,
    0.3711567,
    -0.129061,
    0.5815379,
    -0.1641704,
    0.020886,
    0.6016452,
    0.240787,
    0.4969108,
    -0.4759234,
    -0.4108926,
    0.2090384,
    0.1444948,
    -0.6900774,
    -0.0603351,
    -0.1099939,
    0.160911,
    -0.5254852,
    0.6556898,
    0.1271058,
    1.5192497,
    -0.4493614,
    -0.3659217,
    1.177982,
    -0.454566,
    0.5611523,
    1.0556568,
    0.8881601,
    -0.4491705,
    0.9324099,
    0.528851,
    -0.8668822,
    0.6883888,
    -0.0719894,
    0.9793128,
    -0.6624606,
    -0.2003762,
    -1.9088064,
    -0.1801521,
    -0.0627327,
    0.0671564,
    -0.0881618,
    -0.9100243,
    0.4026805,
    0.7902833,
    0.6360714,
    0.3248921,
    0.421901,
    -0.3698012,
    0.2625831,
    -0.5998358,
    -1.2006359,
    0.4890454,
)
MAMBA3_MIMO_UPSTREAM_GOLDEN_LOGITS: tuple[float, ...] = (
    0.2179261,
    0.0776894,
    -0.5186676,
    0.480095,
    0.9575841,
    0.4218766,
    0.507672,
    0.0514421,
    0.0487948,
    -0.1511241,
    -0.0332729,
    -0.580453,
    0.1224549,
    -0.1221695,
    -0.2206071,
    0.578696,
    0.7422899,
    0.2397944,
    0.9647764,
    3.249033,
    -0.6339355,
    0.9303834,
    -0.9200488,
    -0.4612656,
    0.3859085,
    1.3755946,
    -0.3631333,
    0.1304649,
    -0.4660384,
    1.6492231,
    0.238002,
    -0.2627761,
    0.3464696,
    -0.122054,
    0.420892,
    -0.3811922,
    -0.2008603,
    0.1796682,
    0.5855665,
    -0.8171188,
    0.9661843,
    0.459278,
    0.8711135,
    0.0376102,
    1.1911844,
    0.881109,
    -0.0804695,
    -0.9203762,
    0.4569453,
    -0.6290105,
    -0.0535309,
    -0.4761862,
    -0.8724931,
    0.1358095,
    0.2218869,
    0.8021012,
    0.4433436,
    -0.2532985,
    0.7547399,
    -0.2239999,
    0.7567552,
    0.4418865,
    -0.2356245,
    -0.7673675,
    -0.4207637,
    0.089606,
    0.4800101,
    1.0009947,
    -0.1032782,
    -1.0574561,
    -0.6036469,
    -0.7475856,
    0.0254278,
    0.3834137,
    0.0722501,
    -0.061595,
    0.3496621,
    0.5124313,
    0.0899352,
    -0.7529867,
    1.3784992,
    0.4585281,
    -0.8736275,
    0.062727,
    -0.21903,
    0.1509587,
    0.2558814,
    0.9270047,
    -0.2935192,
    0.7642206,
    -1.0880972,
    -0.6262525,
    0.741909,
    -1.0242316,
    -0.8615136,
    0.7936982,
    0.0543754,
    -0.4252556,
    0.360875,
    0.3126954,
)


def make_mock_mesh(tp_size: int = 1) -> SimpleNamespace:
    """Return a lightweight single-axis mock mesh for adapter unit tests."""
    return SimpleNamespace(shape={"model": tp_size}, axis_names=("model",))


def make_mamba_vllm_config(
    *,
    dp_size: int = 1,
    pcp_size: int = 1,
    dcp_size: int = 1,
    speculative_config: object | None = None,
    model_config: object | None = None,
) -> SimpleNamespace:
    """Return a minimal `VllmConfig`-like namespace for layer validation tests."""
    return SimpleNamespace(
        model_config=model_config,
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
    )


def setup_cpu_platform_for_dist(request: Any, monkeypatch: Any) -> None:
    """Pin `current_platform` to CPU before initializing `dist_ctx`."""
    from vllm.platforms import current_platform

    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)
        monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(current_platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(current_platform, "device_type", "cpu", raising=False)
    request.getfixturevalue("dist_ctx")


def jax_cpu_mesh() -> Mesh:
    """Return a single-device JAX `Mesh` lazily inside test execution."""
    devices = np.asarray(jax.devices()[:1]).reshape((1,))
    return Mesh(devices, axis_names=("model",))


_JIT_MIXER_CACHE: dict[tuple[Any, ...], Any] = {}


def jit_run_jax_mamba_mixer(*args: Any, **kwargs: Any) -> Any:
    """Cached `jax.jit` wrapper around `run_jax_mamba_mixer` for CPU tests."""
    from vllm_torchtpu.layers.core.mamba_mixer2 import run_jax_mamba_mixer

    mesh = kwargs["mesh"]
    if not isinstance(mesh, Mesh):
        return run_jax_mamba_mixer(*args, **kwargs)
    static_kw = {k: v for k, v in kwargs.items() if k != "mesh"}
    key = (
        "m2",
        tuple(mesh.devices.flat),
        mesh.axis_names,
        tuple(sorted(static_kw.items())),
    )
    jitted = _JIT_MIXER_CACHE.get(key)
    if jitted is None:
        jitted = jax.jit(functools.partial(run_jax_mamba_mixer, mesh=mesh, **static_kw))
        _JIT_MIXER_CACHE[key] = jitted
    return jitted(*args)


def jit_run_jax_mamba_mixer3(*args: Any, **kwargs: Any) -> Any:
    """Cached `jax.jit` wrapper around `run_jax_mamba_mixer3` for CPU tests."""
    from vllm_torchtpu.layers.core.mamba_mixer3 import run_jax_mamba_mixer3

    mesh = kwargs["mesh"]
    if not isinstance(mesh, Mesh):
        return run_jax_mamba_mixer3(*args, **kwargs)
    static_kw = {k: v for k, v in kwargs.items() if k != "mesh"}
    key = (
        "m3",
        tuple(mesh.devices.flat),
        mesh.axis_names,
        tuple(sorted(static_kw.items())),
    )
    jitted = _JIT_MIXER_CACHE.get(key)
    if jitted is None:
        jitted = jax.jit(
            functools.partial(run_jax_mamba_mixer3, mesh=mesh, **static_kw)
        )
        _JIT_MIXER_CACHE[key] = jitted
    return jitted(*args)


def passthrough_jax_op(
    op_name: str,
    fn: Any,
    donate_argnums: tuple[int, ...] = (),
) -> Any:
    """CPU test bridge matching `pallas.jax_op` for arbitrary donated state arity."""
    del op_name, donate_argnums
    jitted_fn = jax.jit(fn)

    def _op(*torch_args: Any) -> tuple[torch.Tensor, ...]:
        jax_args = [
            jnp.asarray(t.detach().cpu().to(torch.float32).numpy()).astype(
                jnp.bfloat16 if t.dtype == torch.bfloat16 else jnp.float32
            )
            if isinstance(t, torch.Tensor) and t.is_floating_point()
            else (
                jnp.asarray(t.detach().cpu().numpy())
                if isinstance(t, torch.Tensor)
                else None
            )
            for t in torch_args
        ]
        results = jitted_fn(*jax_args)
        out_tensors: list[torch.Tensor] = []
        for item in results:
            if item.dtype == jnp.bfloat16:
                arr_f32 = np.array(item.astype(jnp.float32), copy=True)
                out_tensors.append(torch.from_numpy(arr_f32).to(torch.bfloat16))
            else:
                out_tensors.append(torch.from_numpy(np.array(item, copy=True)))
        return tuple(out_tensors)

    _op.register_fake = lambda _: None
    return _op


def make_mamba_attn_metadata(
    *,
    mamba_state_indices: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    request_distribution: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor | None = None,
    mamba_request_distribution: torch.Tensor | None = None,
) -> SimpleNamespace:
    """Build mock `MambaAttentionMetadata` with all declared protocol attributes."""
    return SimpleNamespace(
        mamba_state_indices=mamba_state_indices,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
        mamba_request_distribution=mamba_request_distribution,
        seq_lens=seq_lens,
        block_tables=block_tables,
    )


def make_forward_context(
    prefixes: Sequence[str],
    metadata: SimpleNamespace,
) -> SimpleNamespace:
    """Build a mock `ForwardContext` mapping each layer prefix to `metadata`."""
    return SimpleNamespace(attn_metadata={p: metadata for p in prefixes})


def write_minimal_fast_tokenizer(ckpt_dir: Path, vocab_size: int) -> None:
    """Write a deterministic offline `PreTrainedTokenizerFast` into `ckpt_dir`."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab: dict[str, int] = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for idx in range(4, vocab_size):
        vocab[f"tok{idx}"] = idx

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    fast_tok.save_pretrained(str(ckpt_dir))


def write_synthetic_mamba2_checkpoint(
    ckpt_dir: Path,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    """Create a deterministic HF `Mamba2ForCausalLM` safetensors checkpoint."""
    from transformers import Mamba2Config, Mamba2ForCausalLM

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Seed CPU RNG only so callers before TPU worker spawn never trigger PJRT init.
    torch.default_generator.manual_seed(seed)
    cfg = Mamba2Config(
        vocab_size=128,
        hidden_size=64,
        state_size=16,
        num_hidden_layers=2,
        head_dim=16,
        num_heads=8,
        # HF MambaRMSNormGated hardcodes ungrouped RMSNorm (ignoring n_groups > 1),
        # whereas vLLM Mixer2RMSNormGated normalizes per group. Official
        # state-spaces/mamba2-* checkpoints use n_groups=1; n_groups=2 is tested
        # on TPU in tests/layers/adapter/test_mamba_mixer2_op.py.
        n_groups=1,
        expand=2,
        conv_kernel=4,
        chunk_size=8,
        tie_word_embeddings=False,
        use_conv_bias=True,
        use_bias=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    hf_model = Mamba2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for name, p in hf_model.named_parameters():
            if "A_log" in name:
                p.copy_(torch.linspace(-0.5, 0.2, p.numel()).reshape_as(p))
            elif "norm" in name or name.endswith(".D"):
                p.copy_(torch.linspace(0.8, 1.2, p.numel()).reshape_as(p))
            elif "dt_bias" in name:
                p.zero_()
            else:
                torch.nn.init.normal_(p, mean=0.0, std=0.06)

    hf_model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    write_minimal_fast_tokenizer(ckpt_dir, cfg.vocab_size)
    return {"vocab_size": cfg.vocab_size}


@torch.no_grad()
def eval_hf_mamba2_reference(
    ckpt_dir: Path,
    prompt_token_ids: list[int],
    gen_steps: int,
) -> tuple[torch.Tensor, list[int]]:
    """Run HF `Mamba2ForCausalLM` CPU reference (`(last_prompt_logits, greedy_ids)`)."""
    from transformers import Mamba2ForCausalLM

    hf_model = Mamba2ForCausalLM.from_pretrained(str(ckpt_dir)).to(torch.float32).eval()
    seq = list(prompt_token_ids)
    prompt_logits: torch.Tensor | None = None
    generated: list[int] = []

    for step in range(gen_steps):
        inp = torch.tensor([seq], dtype=torch.long)
        logits = hf_model(input_ids=inp).logits[0, -1].detach().cpu()
        if step == 0:
            prompt_logits = logits.clone()
        next_tok = int(torch.argmax(logits).item())
        generated.append(next_tok)
        seq.append(next_tok)

    assert prompt_logits is not None
    return prompt_logits, generated


def write_synthetic_mamba3_checkpoint(
    ckpt_dir: Path,
    *,
    is_mimo: bool = False,
    seed: int = 3030,
) -> dict[str, Any]:
    """Write a deterministic raw `state-spaces/mamba3-*` safetensors checkpoint.

    Uses raw `ssm_cfg.layer = "Mamba3"` (loaded via `--config-format mamba3`),
    `vocab_size = 100` padded to `112` (`pad_vocab_size_multiple = 16`), and
    `d_intermediate = 96` padded to `128` (`multiple_of = 128` in `GatedMLP`).
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Seed CPU RNG only so callers before TPU worker spawn never trigger PJRT init.
    torch.default_generator.manual_seed(seed)

    d_model = 64
    n_layer = 2
    org_vocab_size = 100
    pad_mult = 16
    padded_vocab_size = 112
    d_intermediate = 96
    padded_d_inter = 128
    d_state = 16
    expand = 2
    d_inner = d_model * expand
    head_dim = 16
    num_heads = d_inner // head_dim
    n_groups = 2 if is_mimo else 1
    rope_fraction = 0.5
    num_rope_angles = int((d_state // 2) * rope_fraction)
    mimo_rank = 4 if is_mimo else 1
    chunk_size = 16 if is_mimo else 64

    config_json = {
        "d_model": d_model,
        "d_intermediate": d_intermediate,
        "n_layer": n_layer,
        "vocab_size": org_vocab_size,
        "ssm_cfg": {
            "layer": "Mamba3",
            "d_state": d_state,
            "expand": expand,
            "headdim": head_dim,
            "ngroups": n_groups,
            "rope_fraction": rope_fraction,
            "A_floor": 1e-4,
            "chunk_size": chunk_size,
            "is_mimo": is_mimo,
            "mimo_rank": 4,
            "is_outproj_norm": False,
        },
        "rms_norm": True,
        "residual_in_fp32": True,
        "tie_embeddings": True,
        "pad_vocab_size_multiple": pad_mult,
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config_json, indent=2))

    total_proj = (
        2 * d_inner
        + 2 * mimo_rank * n_groups * d_state
        + 3 * num_heads
        + num_rope_angles
    )
    emb = torch.randn(padded_vocab_size, d_model) * 0.08
    tensors: dict[str, torch.Tensor] = {
        "backbone.embedding.weight": emb,
        "backbone.norm_f.weight": torch.linspace(0.85, 1.15, d_model),
        "lm_head.weight": emb.clone(),
    }
    for layer_idx in range(n_layer):
        p = f"backbone.layers.{layer_idx}"
        tensors[f"{p}.norm.weight"] = torch.linspace(0.9, 1.1, d_model)
        tensors[f"{p}.mixer.in_proj.weight"] = torch.randn(total_proj, d_model) * 0.06
        tensors[f"{p}.mixer.out_proj.weight"] = torch.randn(d_model, d_inner) * 0.06
        tensors[f"{p}.mixer.B_norm.weight"] = torch.linspace(0.8, 1.2, d_state)
        tensors[f"{p}.mixer.C_norm.weight"] = torch.linspace(0.8, 1.2, d_state)
        tensors[f"{p}.mixer.B_bias"] = torch.randn(num_heads, mimo_rank, d_state) * 0.05
        tensors[f"{p}.mixer.C_bias"] = torch.randn(num_heads, mimo_rank, d_state) * 0.05
        tensors[f"{p}.mixer.D"] = torch.linspace(0.5, 1.0, num_heads)
        tensors[f"{p}.mixer.dt_bias"] = torch.linspace(-0.2, 0.2, num_heads)
        if is_mimo:
            tensors[f"{p}.mixer.mimo_x"] = (
                torch.randn(num_heads, mimo_rank, head_dim) * 0.2
            )
            tensors[f"{p}.mixer.mimo_o"] = (
                torch.randn(num_heads, mimo_rank, head_dim) * 0.2
            )
            tensors[f"{p}.mixer.mimo_z"] = (
                torch.randn(num_heads, mimo_rank, head_dim) * 0.3
            )
        tensors[f"{p}.norm2.weight"] = torch.linspace(0.9, 1.1, d_model)
        y_up = torch.randn(d_intermediate, d_model) * 0.06
        gate = torch.randn(d_intermediate, d_model) * 0.06
        pad_zeros = torch.zeros(padded_d_inter - d_intermediate, d_model)
        tensors[f"{p}.mlp.fc1.weight"] = torch.cat(
            [y_up, pad_zeros, gate, pad_zeros], dim=0
        )
        fc2_valid = torch.randn(d_model, d_intermediate) * 0.06
        tensors[f"{p}.mlp.fc2.weight"] = torch.cat(
            [fc2_valid, torch.zeros(d_model, padded_d_inter - d_intermediate)], dim=1
        )

    save_file(tensors, str(ckpt_dir / "model.safetensors"))
    write_minimal_fast_tokenizer(ckpt_dir, org_vocab_size)
    return {
        "org_vocab_size": org_vocab_size,
        "padded_vocab_size": padded_vocab_size,
    }


def _rms_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def _heavy_tail_activation_torch(x: torch.Tensor) -> torch.Tensor:
    """Upstream `mamba_ssm.modules.mamba3.heavy_tail_activation`."""
    x_f32 = x.float()
    return x_f32.clamp_min(0.0) + torch.reciprocal(1.0 - x_f32.clamp_max(0.0))


@torch.no_grad()
def eval_mamba3_mixer_step_reference(
    proj: np.ndarray | torch.Tensor,
    aux_init: np.ndarray | torch.Tensor,
    rec_init: np.ndarray | torch.Tensor,
    B_norm_w: np.ndarray | torch.Tensor,
    C_norm_w: np.ndarray | torch.Tensor,
    B_bias: np.ndarray | torch.Tensor,
    C_bias: np.ndarray | torch.Tensor,
    D: np.ndarray | torch.Tensor,
    dt_bias: np.ndarray | torch.Tensor,
    norm_w: np.ndarray | torch.Tensor | None,
    mimo_x: np.ndarray | torch.Tensor | None,
    mimo_o: np.ndarray | torch.Tensor | None,
    mimo_z: np.ndarray | torch.Tensor | None,
    state_indices: np.ndarray | torch.Tensor,
    query_start_loc: np.ndarray | torch.Tensor,
    distribution: np.ndarray | torch.Tensor,
    seq_lens: np.ndarray | torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    num_rope_angles: int,
    is_mimo: bool,
    mimo_rank: int,
    A_floor: float = 1e-4,
    rms_norm_eps: float = 1e-5,
    apply_rms_norm: bool = True,
    flip_trap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single shared step-by-step PyTorch Mamba-3 mixer oracle (SISO & MIMO)."""

    def _f32_tensor(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().to(torch.float32).clone()
        return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))

    proj_t = _f32_tensor(proj)
    b_norm_t = _f32_tensor(B_norm_w)
    c_norm_t = _f32_tensor(C_norm_w)
    b_bias_t = _f32_tensor(B_bias)
    c_bias_t = _f32_tensor(C_bias)
    d_t = _f32_tensor(D)
    dt_bias_t = _f32_tensor(dt_bias)
    norm_t = None if norm_w is None else _f32_tensor(norm_w)
    mx_t = None if mimo_x is None else _f32_tensor(mimo_x)
    mo_t = None if mimo_o is None else _f32_tensor(mimo_o)
    mz_t = None if mimo_z is None else _f32_tensor(mimo_z)

    r = mimo_rank if is_mimo else 1
    if b_bias_t.ndim == 2:
        b_bias_t = b_bias_t.unsqueeze(1)
    if c_bias_t.ndim == 2:
        c_bias_t = c_bias_t.unsqueeze(1)

    num_tokens = proj_t.shape[0]
    aux_out = _f32_tensor(aux_init)
    rec_out = _f32_tensor(rec_init)
    num_slots = rec_out.shape[0]
    slots = np.asarray(state_indices)
    qsl = np.asarray(query_start_loc)
    dist = np.asarray(distribution)
    slens = np.asarray(seq_lens)
    max_reqs = slens.shape[0]
    num_valid = int(np.clip(dist[2], 0, max_reqs))

    d_inner = num_heads * head_dim
    d_bc = r * n_groups * d_state
    group_rep = num_heads // n_groups
    dim_x_aux = r * head_dim
    dim_k_aux = r * d_state
    y_out = torch.zeros((num_tokens, d_inner), dtype=torch.float32)

    s1 = d_inner
    s2 = s1 + d_inner
    s3 = s2 + d_bc
    s4 = s3 + d_bc
    s5 = s4 + num_heads
    s6 = s5 + num_heads
    s7 = s6 + num_heads

    for seq_idx in range(num_valid):
        start = int(qsl[seq_idx])
        end = int(qsl[seq_idx + 1])
        q_len = max(end - start, 0)
        if q_len == 0:
            continue
        slot = int(slots[seq_idx])
        has_slot = 0 <= slot < num_slots
        has_init = has_slot and (int(slens[seq_idx]) - q_len > 0)

        if has_init:
            h = rec_out[slot].clone()
            aux_s = aux_out[slot]
            prev_x = aux_s[:, :dim_x_aux].reshape(num_heads, r, head_dim).clone()
            prev_k = (
                aux_s[:, dim_x_aux : dim_x_aux + dim_k_aux]
                .reshape(num_heads, r, d_state)
                .clone()
            )
            theta = aux_s[:, dim_x_aux + dim_k_aux :].clone()
        else:
            h = torch.zeros((num_heads, head_dim, d_state), dtype=torch.float32)
            prev_x = torch.zeros((num_heads, r, head_dim), dtype=torch.float32)
            prev_k = torch.zeros((num_heads, r, d_state), dtype=torch.float32)
            theta = torch.zeros((num_heads, num_rope_angles), dtype=torch.float32)

        for local_t, tok in enumerate(range(start, end)):
            row = proj_t[tok]
            z_raw = row[:s1]
            v_t = row[s1:s2].reshape(num_heads, head_dim)
            b_raw = row[s2:s3].reshape(r, n_groups, d_state).permute(1, 0, 2)
            c_raw = row[s3:s4].reshape(r, n_groups, d_state).permute(1, 0, 2)
            dd_dt = row[s4:s5]
            dd_a = row[s5:s6]
            trap_raw = row[s6:s7]
            angle_raw = row[s7:]

            b_exp = _rms_norm_torch(b_raw, b_norm_t, rms_norm_eps).repeat_interleave(
                group_rep, dim=0
            )
            c_exp = _rms_norm_torch(c_raw, c_norm_t, rms_norm_eps).repeat_interleave(
                group_rep, dim=0
            )
            k_pre = b_exp + b_bias_t
            q_pre = c_exp + c_bias_t

            a_t = torch.clamp(-_heavy_tail_activation_torch(dd_a), max=-A_floor)
            dt_t = F.softplus(dd_dt + dt_bias_t)
            trap_t = torch.sigmoid(trap_raw)
            if flip_trap:
                trap_t = 1.0 - trap_t
            gamma_t = dt_t * trap_t
            beta_t = dt_t * (1.0 - trap_t)

            theta = torch.remainder(
                theta
                + torch.tanh(angle_raw).unsqueeze(0) * math.pi * dt_t.unsqueeze(1),
                2.0 * math.pi,
            )
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)

            k_rot = k_pre.clone()
            q_rot = q_pre.clone()
            if is_mimo:
                assert mx_t is not None and mo_t is not None and mz_t is not None
                half = d_state // 2
                for arr in (k_rot, q_rot):
                    a0 = arr[:, :, :num_rope_angles].clone()
                    a1 = arr[:, :, half : half + num_rope_angles].clone()
                    arr[:, :, :num_rope_angles] = a0 * cos_t.unsqueeze(
                        1
                    ) - a1 * sin_t.unsqueeze(1)
                    arr[:, :, half : half + num_rope_angles] = a0 * sin_t.unsqueeze(
                        1
                    ) + a1 * cos_t.unsqueeze(1)
                x_t = v_t.unsqueeze(1) * mx_t
            else:
                rot_dim = 2 * num_rope_angles
                for arr in (k_rot, q_rot):
                    pairs = arr[:, 0, :rot_dim].reshape(num_heads, num_rope_angles, 2)
                    p0 = pairs[:, :, 0].clone()
                    p1 = pairs[:, :, 1].clone()
                    arr[:, 0, :rot_dim] = torch.stack(
                        [p0 * cos_t - p1 * sin_t, p0 * sin_t + p1 * cos_t], dim=-1
                    ).reshape(num_heads, rot_dim)
                x_t = v_t.unsqueeze(1)

            use_prev = (local_t > 0) or has_init
            prev_outer = (
                torch.einsum("hrp,hrn->hpn", prev_x, prev_k)
                if use_prev
                else torch.zeros_like(h)
            )
            cur_outer = torch.einsum("hrp,hrn->hpn", x_t, k_rot)
            h = (
                torch.exp(a_t * dt_t)[:, None, None]
                * (h + beta_t[:, None, None] * prev_outer)
                + gamma_t[:, None, None] * cur_outer
            )
            y_ssm = torch.einsum("hrn,hpn->hrp", q_rot, h) + d_t[:, None, None] * x_t

            if is_mimo:
                assert mo_t is not None and mz_t is not None
                z_r = z_raw.reshape(num_heads, 1, head_dim)
                y_tok = (y_ssm * F.silu(z_r * mz_t) * mo_t).sum(dim=1).reshape(d_inner)
                gate_tok = None
            else:
                y_tok = y_ssm[:, 0, :].reshape(d_inner)
                gate_tok = z_raw

            if apply_rms_norm:
                val = y_tok * F.silu(gate_tok) if gate_tok is not None else y_tok
                if norm_t is not None:
                    grouped = val.reshape(n_groups, d_inner // n_groups)
                    var = grouped.pow(2).mean(dim=-1, keepdim=True)
                    val = (grouped * torch.rsqrt(var + rms_norm_eps)).reshape(
                        d_inner
                    ) * norm_t
                y_out[tok] = val
            else:
                y_out[tok] = y_tok

            prev_x = x_t.clone()
            prev_k = k_rot.clone()

        if has_slot:
            rec_out[slot] = h
            aux_out[slot] = torch.cat(
                [
                    prev_x.reshape(num_heads, dim_x_aux),
                    prev_k.reshape(num_heads, dim_k_aux),
                    theta,
                ],
                dim=-1,
            )

    return aux_out.numpy(), rec_out.numpy(), y_out.numpy()


@torch.no_grad()
def eval_torch_mamba3_reference(
    ckpt_dir: Path,
    prompt_token_ids: list[int],
    gen_steps: int,
) -> tuple[torch.Tensor, list[int]]:
    """Model-level Mamba-3 oracle built on shared `eval_mamba3_mixer_step_reference`."""
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    sd = {
        k: v.float() for k, v in load_file(str(ckpt_dir / "model.safetensors")).items()
    }

    d_model = int(cfg["d_model"])
    n_layer = int(cfg["n_layer"])
    org_vocab_size = int(cfg["vocab_size"])
    d_inter = int(cfg["d_intermediate"])
    ssm = cfg["ssm_cfg"]
    d_state = int(ssm["d_state"])
    expand = int(ssm["expand"])
    d_inner = d_model * expand
    head_dim = int(ssm["headdim"])
    num_heads = d_inner // head_dim
    n_groups = int(ssm["ngroups"])
    is_mimo = bool(ssm["is_mimo"])
    r = int(ssm["mimo_rank"]) if is_mimo else 1
    num_rope_angles = int((d_state // 2) * float(ssm["rope_fraction"]))
    a_floor = float(ssm.get("A_floor", 1e-4))
    use_outproj_norm = bool(ssm.get("is_outproj_norm", False))
    aux_dim = r * (head_dim + d_state) + num_rope_angles
    eps = 1e-5

    def _forward_full(tokens: list[int]) -> torch.Tensor:
        inp = torch.tensor(tokens, dtype=torch.long)
        hidden = sd["backbone.embedding.weight"][inp]
        residual: torch.Tensor | None = None
        seq_len = len(tokens)

        for layer_idx in range(n_layer):
            p = f"backbone.layers.{layer_idx}"
            residual = hidden if residual is None else (hidden + residual)
            hidden = _rms_norm_torch(residual, sd[f"{p}.norm.weight"], eps)

            proj = hidden @ sd[f"{p}.mixer.in_proj.weight"].T
            dt_bias = sd.get(f"{p}.mixer.dt_bias", sd.get(f"{p}.mixer.b_dt"))
            assert dt_bias is not None
            norm_w = sd.get(f"{p}.mixer.norm.weight") if use_outproj_norm else None
            _, _, y_np = eval_mamba3_mixer_step_reference(
                proj,
                torch.zeros((1, num_heads, aux_dim)),
                torch.zeros((1, num_heads, head_dim, d_state)),
                sd[f"{p}.mixer.B_norm.weight"],
                sd[f"{p}.mixer.C_norm.weight"],
                sd[f"{p}.mixer.B_bias"],
                sd[f"{p}.mixer.C_bias"],
                sd[f"{p}.mixer.D"],
                dt_bias,
                norm_w,
                sd.get(f"{p}.mixer.mimo_x"),
                sd.get(f"{p}.mixer.mimo_o"),
                sd.get(f"{p}.mixer.mimo_z"),
                np.array([0], dtype=np.int32),
                np.array([0, seq_len], dtype=np.int32),
                np.array([0, 1, 1], dtype=np.int32),
                np.array([seq_len], dtype=np.int32),
                num_heads=num_heads,
                head_dim=head_dim,
                n_groups=n_groups,
                d_state=d_state,
                num_rope_angles=num_rope_angles,
                is_mimo=is_mimo,
                mimo_rank=r,
                A_floor=a_floor,
                rms_norm_eps=eps,
                apply_rms_norm=True,
            )
            hidden = torch.from_numpy(y_np) @ sd[f"{p}.mixer.out_proj.weight"].T

            if d_inter > 0:
                residual = hidden + residual
                mlp_norm_w = sd.get(f"{p}.norm2.weight", sd.get(f"{p}.mlp_norm.weight"))
                assert mlp_norm_w is not None
                hidden = _rms_norm_torch(residual, mlp_norm_w, eps)
                fc1_out = hidden @ sd[f"{p}.mlp.fc1.weight"].T
                # Upstream `mamba_ssm.modules.mlp.GatedMLP`: `(y, gate) = chunk(2)`.
                y_up, gate = fc1_out.chunk(2, dim=-1)
                hidden = (y_up * F.silu(gate)) @ sd[f"{p}.mlp.fc2.weight"].T

        assert residual is not None
        final_hidden = _rms_norm_torch(
            hidden + residual, sd["backbone.norm_f.weight"], eps
        )
        head_w = sd.get("lm_head.weight", sd["backbone.embedding.weight"])
        return final_hidden @ head_w[:org_vocab_size].T

    seq = list(prompt_token_ids)
    prompt_logits: torch.Tensor | None = None
    generated: list[int] = []
    for step in range(gen_steps):
        logits_all = _forward_full(seq)
        last_logits = logits_all[-1].detach().cpu()
        if step == 0:
            prompt_logits = last_logits.clone()
        next_tok = int(torch.argmax(last_logits).item())
        generated.append(next_tok)
        seq.append(next_tok)

    assert prompt_logits is not None
    return prompt_logits, generated
