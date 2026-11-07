# TPU-C MoE bundle (`TPUC_MOE=1`)

Everything `vllm_torchtpu.layers.adapter.tpuc_moe` needs to run the routed
experts of FP8 MoE layers on the Ghostfish (TPU v7x) TPU-C MoE kernel. Built in
google3 under `experimental/users/hning/torchtpu_tpuc_cdk/`
(`build_and_stage_bundle.sh`).

| Path | What |
|---|---|
| `lib/libtpuc_serializer.so` | Serializes TPU-C assembly into a PJRT executable (google3 build). |
| `lib/tpuc_serialize_cli.py` | Runs the serializer in a child process (it must not run inside the libtpu process, b/575455577). |
| `lib/deepsea_torch_bridge.so` | The adapter: registers `torch.ops.deepsea_moe.*`, loads and launches the kernel through the PJRT C API. Built in the container by `build_adapter.sh`. |
| `src/deepsea_torch_bridge_gxx.cc`, `src/pjrt_c_api.h` | Adapter sources. |
| `runfiles/google3/torchtpu_tpuc_kernels/moe/*.s` | Kernel assembly: `libmoe_fused_ffn_kernel_gf.s` (H=4096, I=1024; Qwen3.5-397B), `libmoe_fused_ffn_kernel_h2048_gf.s` (H=2048, I=512; Qwen3.5-35B-A3B). |
| `plugins/fused_moe_tpuc_layer.py` | Routing / SparseCore dispatch and combine glue around the kernel. |
| `test_qwen35_moe.py`, `test_smoke.py` | Standalone EP8 parity / benchmark tests (`TPU_MOE_MODEL=qwen35_35b` for 35B dims). |

Serving:

```bash
bash tpuc_bundle/build_adapter.sh   # once per container image (also done lazily)
TPUC_MOE=1 \
LIBTPU_INIT_ARGS="--tpu_use_continuations=true --DANGEROUS_tpu_runtime_abi_verification_disabled=true" \
vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --data-parallel-size 8 --enable-expert-parallel ...
```

Requirements: MoE dims (hidden, moe_intermediate) of (4096, 1024) or (2048, 512),
per-channel FP8 MoE weights (default `MOE_REQUANTIZE_*`), at most 80 local
experts, and none of the MoE kernels that own the collectives
(`MOE_FUSED_EP_KERNEL_IMPL`, `USE_MOE_FUSED_EP_KERNEL`,
`TPU_MOE_COLLECTION_CHUNK_SIZE`, `TPU_MOE_FP8_ALL_GATHER`,
`FUSED_MOE_DECODE_KERNEL`, `USE_MOE_TP_DECODE_KERNEL`).
