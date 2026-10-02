# gdn_pcp_fused: fused GDN prefill-context-parallel (PCP) kernel fork

This directory is a fork of `kernels/gdn/v3` (synced through #1422), plus the
PCP-specific kernel code that used to live there (`pcp_wrapper.py`,
`pcp_metadata.py`, `projection_scale.py`).

Scope: only the fused pooled-PCP path uses this fork, where the QKVZ
projection, PCP remote-DMA communication and conv1d + GDN run in one
Pallas kernel (`pcp_wrapper.fused_qkvz_projection_pcp_gdn`). The
non-fused PCP paths in `layers/core/gdn_attention.py` exchange tokens
outside the kernel and call the V3 front-end directly.

Policy:

- `kernels/gdn/v3` is provided as-is; only its front-end
  (`wrapper.fused_conv1d_gdn`) is a supported interface. Code here must not
  import from `kernels/gdn/v3`.
- Upstream V3 changes are **not** picked up automatically. Port them by hand
  when PCP needs them.
- Once PCP work settles, the ideas here are expected to be folded back into
  `kernels/gdn/v3` through a proper design review, and this fork deleted.

Shared contract with V3: the GDN state layout inside the unified KV pool.
PCP prefill writes state that the V3 decode path later reads, so both sides
must agree on granule addressing and the conv/recurrent offsets.
`tests/layers/core/test_gdn_attention_pcp.py` checks this against the V3
reference and the V3 host view of the pool.
