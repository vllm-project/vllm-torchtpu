# NVFP4 block-512 fused EP MoE

The TPU plugin can serve NVFP4 routed experts through `fused_ep_moe_v2`
in W4A8 mode. Enable both settings before starting the service:

```bash
export USE_MOE_FUSED_EP_KERNEL=1
export MOE_REQUANTIZE_BLOCK_SIZE=512
```

This composes with selective online FP8 quantization of BF16 Linear layers
through `additional_config.tpu_online_fp8`. The two features affect different
weights: selected dense/shared-expert projections become FP8, while routed
experts remain FP4.

During loading, the checkpoint's per-block and global scales are combined,
then the original block-16 weights are dequantized and requantized to
block-512. The native FP4 K-major weights and block scales are prepared once.
The fused kernel dynamically quantizes input and intermediate activations to
FP8, handles routing and EP exchange, and returns the local token shard.
FP4 weights are packed in memory and widened to FP8 inside the kernel; this
is not a native FP4 matrix instruction.

Only admitted layers claim communication ownership. Admission retains the
existing FP8 path's checks for expert placement, supported routing, bias,
TP/PCP configuration and token threshold, with additional FP4 dtype and
block-512 scale checks. Unsupported layers retain the GMM path. The GMM path also uses W4A8 when
requantization is enabled, as implemented by the existing GMM W4A8 support.
Requantizing weights is controlled by `MOE_REQUANTIZE_BLOCK_SIZE`.
With that variable unset, original block-16 NVFP4 continues using W4A16.

Successful admission logs both `Fused EP MoE armed` with `format=fp4
rhs_qb=512` and `NVFP4 fused EP W4A8 enabled: block-512`. An enabled environment
flag alone does not prove that the fused kernel is serving a layer.

Block-512 requantization and FP8 activation/transport introduce numerical
changes relative to the original checkpoint. Kernel numerical tests compare
against a dense reference using the same dequantized FP4 weights. Model-level
quality and throughput must be evaluated separately with real weights.
