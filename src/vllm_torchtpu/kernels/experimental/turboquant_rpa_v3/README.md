# turboquant_rpa_v3: fused TurboQuant Ragged Paged Attention v3 kernel fork

This directory is an experimental fork of `kernels/ragged_paged_attention/v3`
for the TurboQuant compressed `uint8` KV-cache path
(`turboquant_{k8v4,4bit_nc,k3v4_nc,3bit_nc}`), where compressed 128-lane KV
pages are streamed between HBM and VMEM via DMA, updated in VREGs, and
dequantized on the fly inside the inner `RPAd` / `RPAm` Pallas loops. Standard
`bf16` and `fp8` attention layers continue to use
`kernels/ragged_paged_attention/v3` unchanged.

Policy:

- Forked from `kernels/ragged_paged_attention/v3/kernel.py` at commit
  `132a64aee` (`2026-10-09`). Upstream V3 changes are **not** picked up
  automatically; port them by hand when TurboQuant needs them.
- Once the compressed-page DMA and VREG dequantization hooks settle, the design
  is expected to be folded back into `kernels/ragged_paged_attention/v3` (or
  `batched_rpa`) through design review, and this fork deleted.

Shared contract with `layers/core/turboquant.py` and V3:

- Compressed page geometry (`(num_pages, num_kv_heads, page_words, 128)`) and
  128-lane VREG update/dequantization callbacks (`tq_update_page_fn`,
  `tq_dequant_page_fn`) are defined in `vllm_torchtpu.layers.core.turboquant`.
- `tests/kernels/experimental/turboquant_rpa_v3/test_kernel.py` verifies exact
  cache-byte and attention-output parity against `ref_ragged_paged_attention`
  from `kernels/ragged_paged_attention/v3`.
