# OLMo 3.5 (`olmo35`)

TPU model directory for Ai2's OLMo 3.5 hybrid models
(`Olmo35ForCausalLM`, `model_type: olmo35`): Kimi Delta Attention on seven of
every eight layers, NoPE full attention with scalable softmax on the rest, and
a latent mixture of experts.

- OLMo 3.5 Small (72.6B total, 4.1B active). No public checkpoint yet; the
  weights are converted from the MaxText training checkpoint
  (`decoder_block: olmoe3`).

- **Support tier**: `Experimental`
- **Owners**: `@sierraisland`, `@wenxindongwork`

## Directory Layout

- `config.py`: the HF config class (transformers has none for this
  architecture yet), its `AutoConfig` registration, and `TPUOlmo35Config`,
  the vLLM config hook that turns prefix caching off and rejects the features
  the model does not support.
- `model.py`: the model and its checkpoint loader. The KDA, latent-MoE and
  MLP layers are the shared ones in `layers/adapter/` (`kda_attention.py`,
  `latent_moe.py`, `gated_mlp.py`), also used by Kimi K3.

The compile cache key covers the paths in `_RUNTIME_CACHE_KEY_PATHS` plus
every `kernels/` and `ops/` directory under `vllm_torchtpu/models/`, so keep
custom-op bridges and kernel copies in those directories.
