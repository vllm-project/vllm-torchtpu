# Omni (vLLM-Omni on TPU)

This package is the TPU platform plugin for
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). vLLM-Omni loads it
through the `vllm_omni.platform_plugins` entry point `torchtpu`
(`vllm_torchtpu.omni:register_omni_tpu_platform`), which returns
`vllm_torchtpu.omni.platform.OmniTpuPlatform` when TPU chips are present.
Model code comes from vLLM-Omni; TPU fixes are applied as runtime patches.

## Files

- `__init__.py`: entry point that returns `OmniTpuPlatform` on TPU hosts.
- `attention.py`: Pallas flash attention backend with a PyTorch SDPA fallback.
- `patches.py`: runtime patches that make vLLM-Omni diffusion models run on TPU.
- `platform.py`: `OmniTpuPlatform`, the vLLM-Omni platform built on `TpuPlatform`.
- `profiler.py`: profiler wrapper that records TPU activity in diffusion workers.
- `rope.py`: TPU-friendly rotary embeddings for Wan and LTX-2 models.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Contributing

Before you work on any new feature, bug fix or model in omni:

1. Open a [GitHub issue](https://github.com/vllm-project/vllm-torchtpu/issues/new/choose)
   first. Start the title with `[Omni]`.
2. Wait for an omni owner to confirm on the issue before you start work.
   Owners are listed in [CODEOWNERS](../../../.github/CODEOWNERS) for
   `/src/vllm_torchtpu/omni/` (currently @mbohlool, @pi314ever,
   @chandrasekhard2).
3. Link the issue in your PR and start the PR title with `[Omni]`.
4. Follow the repository [CONTRIBUTING.md](../../../CONTRIBUTING.md).
