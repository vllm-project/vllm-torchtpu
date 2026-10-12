# Kubernetes & GKE Deployment

`vllm-torchtpu` includes the Kubernetes [JobSet](https://jobset.sigs.k8s.io/) workload manifests its CI uses to run multi-host TPU slices and Prefill/Decode (P/D) disaggregated serving on Google Kubernetes Engine (GKE), under `.buildkite/kubernetes/manifests/`. They are templates for the CI launcher: variables such as `${WORKLOAD_NAME}`, `${IMAGE}` and `${MULTIHOST_TOPOLOGY}` are substituted before submission, and caches, tolerations and retries come from the launcher's `tpu-ci.google.com/defaults` annotation. Adapt them before applying them to your own cluster.

---

## Available Reference Manifests

| Manifest Path | Topology / Use Case |
| :--- | :--- |
| `.buildkite/kubernetes/manifests/workloads/mp-multihost-slice.yaml` | Multi-host TPU v7x slice using vLLM's native multiprocessing (`mp`) data/tensor parallelism without Ray. |
| `.buildkite/kubernetes/manifests/workloads/ray-multihost-slice.yaml` | Multi-host TPU v7x slice bringing up a Ray cluster (`JOB_COMPLETION_INDEX=0` head + worker pods). |
| `.buildkite/kubernetes/manifests/workloads/qwen3.5-397b-1p1d.yaml` | 1-Prefill / 1-Decode (1P1D) disaggregated serving for `Qwen/Qwen3.5-397B-A17B` with Raiden KV transfer. |
| `.buildkite/kubernetes/manifests/workloads/qwen3.5-p8d8-disagg.yaml` | PCP8/DP8 P/D disaggregation correctness test for `Qwen3.5` (two single-host `2x2x1` slices). |
| `.buildkite/kubernetes/manifests/workloads/qwen3.5-gpc-two-replica.yaml` | Global prefix caching across two single-host (`2x2x1`) `Qwen3.5` serving replicas. |
| `.buildkite/kubernetes/manifests/workloads/kimi-k3-1p1d-4dp8tp.yaml` | 1P1D disaggregated serving for `Kimi-K3` (`DP=4, TP=8`). |
| `.buildkite/kubernetes/manifests/workloads/kimi-k3-agentx-1p1d.yaml` | `Kimi-K3` 1P1D replaying the InferenceX AgentX trace, driven by a chipless client pod. |
| `.buildkite/kubernetes/manifests/workloads/dsv4-pd-disagg.yaml` | P/D disaggregation correctness test for `DeepSeek-V4-Flash` (DP8 prefill host + DP8 decode host). |

---

## Manifest Design & Conventions

---8<--- ".buildkite/kubernetes/manifests/README.md:3"

---

## Example: Multi-Host `mp` Slice JobSet

Below is the reference `JobSet` manifest for running `vllm-torchtpu` across a multi-host TPU v7x slice using the native `mp` backend (`.buildkite/kubernetes/manifests/workloads/mp-multihost-slice.yaml`):

```yaml
---8<--- ".buildkite/kubernetes/manifests/workloads/mp-multihost-slice.yaml"
```

---

## Key GKE Configuration Best Practices

1. **Indexed Job Completion for Multi-Host Slices**: Always set `completionMode: Indexed`, `parallelism: <NUM_HOSTS>`, and `completions: <NUM_HOSTS>` on the `replicatedJobs` template, with `enableDNSHostnames: true` on the `JobSet`. GKE schedules all pods of an indexed job onto a single interconnected TPU slice and assigns deterministic DNS hostnames (`<jobset>-slice-0-0.<jobset>`).
2. **TPU Node Selectors**: Specify both `cloud.google.com/gke-tpu-accelerator` (e.g., `tpu7x` or `tpu-v6e-slice`) and `cloud.google.com/gke-tpu-topology` (e.g., `2x2x1`, `2x2x2`, `2x2x4`) alongside `google.com/tpu` resource limits (`4` chips per host on TPU v7x).
3. **Persistent XLA & vLLM Caches**: Point `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` and `VLLM_CACHE_ROOT` to a persistent volume or GCS-backed mount rather than the container root disk, while keeping `profile_cache` on an in-memory `emptyDir` (`medium: Memory`) so free-HBM probing always runs fresh on startup. The CI manifests still set the legacy `VLLM_XLA_CACHE_PATH`, which `vllm-torchtpu` does not read. Set `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` when you adapt them.
