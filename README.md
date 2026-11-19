# TPU7x XPlane Traces for vllm-torchtpu PR #908 (`TPU_MOE_JOINT_ROUTING_128`)

Raw TPU7x `.xplane.pb.gz` device traces and paired A/B measurements comparing `main`'s default packed single sort (`TPU_MOE_JOINT_ROUTING_128=0`, `_single_sort_permutation`) against PR #908's fixed-size Pallas kernel (`TPU_MOE_JOINT_ROUTING_128=1`, `stable_order_and_inverse_128`) on PR #908 `HEAD` (`4ea2b80db93f1d82a7225d0fcf1d3aaa8e13b91f`).

## Configuration

- **Hardware / runtime**: TPU v7x (`2x2x1`, single chip execution via `TPU_VISIBLE_CHIPS=0`), `jax==0.11.2`, `jaxlib==0.11.2`, `libtpu==0.0.50`, `torch==2.13.0+cpu`, `torch_tpu==0.2.0`
- **Routing flags**: `TPU_MOE_COUNTING_SORT=1`, `TPU_MOE_SINGLE_SORT=1`, `USE_MOE_SPARSE_CORE=0`, `TPU_MOE_SKIP_PADDED_TOKENS=0`, `TPU_MOE_DENSE_EXPERT_THRESHOLD=0`, `ONEHOT_MOE_PERMUTE_THRESHOLD=3840`
- **Workload shape**: `num_tokens=16`, `topk=8` (`128` routed rows), `local_num_experts=256`, `hidden_size=2048`, `intermediate_size=128`, `bfloat16`, `activation="silu"`, `use_ep=False`
- **Correctness on TPU7x**: `native_kernel_cases=5` passed natively via `torch.compile(backend="tpu")`, and all 6 outputs of `prepare_routed_gmm_inputs` matched `_single_sort_permutation` bit-for-bit (`prepare_bit_exact_tpu=True`).

## Paired Results (`pair-01` A/B and `pair-02` B/A)

| Workload | Variant | `XLA Modules` Median (`µs`, 5 profiled calls) | Chained Median (`µs`/call, 9 rounds) | Single-Call HLO (`sort` / `scatter` / `stable_order_and_inverse_128`) | Trace (`.xplane.pb.gz`) | Size | SHA-256 |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| `prepare_routed_gmm_inputs` (`pair-01`) | `baseline` (`0`) | `5.776 µs` | `12.634 µs` | `1 / 1 / 0` | [`prep_baseline_tl10.xplane.pb.gz`](traces/prep_baseline_tl10.xplane.pb.gz) | `2.42 MiB` | `2d749df1a8ff489873a42a278b1a4f6e4b6c888d395d4677155943c0e1743219` |
| `prepare_routed_gmm_inputs` (`pair-01`) | `candidate` (`1`) | `3.559 µs` (**-38.4%**) | `8.955 µs` (**-29.1%**) | `0 / 0 / 1 call` | [`prep_candidate_tl10.xplane.pb.gz`](traces/prep_candidate_tl10.xplane.pb.gz) | `2.43 MiB` | `293abc5d9bf2b890f5991e6fb6b774d9e17fc5cfb0cd5d135b398df08d1d6f84` |
| `prepare_routed_gmm_inputs` (`pair-02`) | `candidate` (`1`) | `3.559 µs` (**-38.6%**) | `7.954 µs` (**-34.6%**) | `0 / 0 / 1 call` | [`prep_candidate_r2_tl10.xplane.pb.gz`](traces/prep_candidate_r2_tl10.xplane.pb.gz) | `2.42 MiB` | `cba5ec808627323b3f614cfbb08a82fe7b3dc39453382e6ba24c8941fca1add1` |
| `prepare_routed_gmm_inputs` (`pair-02`) | `baseline` (`0`) | `5.796 µs` | `12.166 µs` | `1 / 1 / 0` | [`prep_baseline_r2_tl10.xplane.pb.gz`](traces/prep_baseline_r2_tl10.xplane.pb.gz) | `2.43 MiB` | `34bb99a1e7e4fc713b903a40b8405ef407747c9d29a0c4eb66d25b9eae75ffec` |
| `fused_moe_func` (`pair-01`) | `baseline` (`0`) | `125.885 µs` | `155.234 µs` | `1 / 1 / 0` | [`moe_baseline_tl10.xplane.pb.gz`](traces/moe_baseline_tl10.xplane.pb.gz) | `2.58 MiB` | `fa47987e62525e6c35367765105ad2b9c359460b6c5c077b2a0765391ac97f00` |
| `fused_moe_func` (`pair-01`) | `candidate` (`1`) | `122.974 µs` (**-2.31%**) | `153.397 µs` (**-1.18%**) | `0 / 0 / 1 call` | [`moe_candidate_tl10.xplane.pb.gz`](traces/moe_candidate_tl10.xplane.pb.gz) | `2.59 MiB` | `e1535d94d56093ad3742b7f82c0f7ae3269bd4996ee25be6ae352f00a94d43ff` |
| `fused_moe_func` (`pair-02`) | `candidate` (`1`) | `123.330 µs` (**-2.05%**) | `151.516 µs` (**-1.34%**) | `0 / 0 / 1 call` | [`moe_candidate_r2_tl10.xplane.pb.gz`](traces/moe_candidate_r2_tl10.xplane.pb.gz) | `2.59 MiB` | `c3278e08342c395bc205c1247a2c4bcdec2b361bf23ba6e114e31c82343c14b6` |
| `fused_moe_func` (`pair-02`) | `baseline` (`0`) | `125.916 µs` | `153.572 µs` | `1 / 1 / 0` | [`moe_baseline_r2_tl10.xplane.pb.gz`](traces/moe_baseline_r2_tl10.xplane.pb.gz) | `2.59 MiB` | `f2205cbe4b6e598a4721a5a6daca633d96ef64ac7f53702a6a6ec2dee8482738` |

## Device `XLA Ops` Breakdown (`/device:TPU:0`, mean over post-warmup profiled calls)

- **Baseline (`_single_sort_permutation`)**:
  - `sort.5`: **`1.959 µs`** (`prep`), **`1.957 µs`** (`moe`)
  - `scatter_custom_fusion`: **`0.982 µs`** (`prep`), **`0.952 µs`** (`moe`)
  - Combined sort + scatter cost: **`2.941 µs`** (`prep`) / **`2.909 µs`** (`moe`)
- **Candidate (`stable_order_and_inverse_128`)**:
  - `stable_order_and_inverse_128.1`: **`0.209 µs`** (`prep` & `moe` across both pairs)
  - Net module savings: **`-2.22 µs`** (`-38.5%`) on isolated `prepare_routed_gmm_inputs` (`5.786 µs -> 3.559 µs`) and **`-2.59 µs` to `-2.91 µs`** (`-2.05%` to `-2.31%`) on full 1-layer `fused_moe_func`.

To inspect locally:

```bash
gunzip -k traces/*.xplane.pb.gz
xprof --port 8791 --logdir traces
```
