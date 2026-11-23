# DeepSeek-V4.1-Flash on TPU7x: 1P1D recipe from public sources only

This recipe needs no private wheels, no private buckets and no Google-internal tooling.
It was run end to end on 2026-11-23 and reached **21.7K output tok/s per 4-chip decode host**.

| Input | Public source | Pinned to |
|---|---|---|
| vllm-torchtpu | `github.com/vllm-project/vllm-torchtpu`, branch `cpk/dsv41-mh-decode-main` | runtime code validated at `630a3aae6`; later commits on the branch add only this kit and its results |
| vLLM | `github.com/vllm-project/vllm`, tag `v0.33.0` | plus the 4 patches in `../patches/vllm-v0.33.0/` |
| Container image | Docker Hub `vllm/vllm-tpu:nightly` | resolved to `sha256:97da0de1581c928ea8a76b4e3ba501a980df6bfafca1b5c22b810827c6ca8e87` |
| Runtime wheels | PyPI and `download.pytorch.org/whl/cpu` | `torch 2.13.0+cpu`, `torchvision 0.28.0`, `torch_tpu 0.2.0`, `libtpu 0.0.50`, `tpu-sync-torch 0.0.1.dev20261029081917`, `jax`/`jaxlib 0.11.2`, `frozendict 2.4.7`, `immutabledict 4.3.1` |
| Checkpoint | Hugging Face `deepseek-ai/DeepSeek-V4.1-Flash` (ungated) | revision `314113f8a7da`, 48 shards, 510 GB |

Do not use the older branch `cpk/dsv41-mh-decode` or `../whls/SHA256SUMS.txt`.
Both describe the earlier stack, which does need unpublished `torch_tpu 0.1.1.dev…` and `tpu-sync-torch` builds.

## Topology

| Role | Host | Shape | Ports |
|---|---|---|---|
| Prefill | 1 × tpu7x `2x2x1` (4 chips, 8 TensorCores) | 4 servers, each TP2/EP2, 20-layer CED, 8,192-token chunk | 8400–8403 |
| Decode | 1 × tpu7x `2x2x1` | DP8/TP1/EP8, DSpark K=5, 128 seqs/rank, 768 batched tokens/rank | 9400 |

## Prerequisites

- A GKE cluster with two free tpu7x `2x2x1` nodes (ours are `tpu7x-standard-4t`, about 940 GB of host RAM) and `kubectl` access.
- Pods must be allowed `hostNetwork`, `privileged` and a `hostPath` mount of `/dev/shm`.
- Outbound access from the pods to GitHub, PyPI, `download.pytorch.org`, Docker Hub and Hugging Face.

## Steps

```bash
RAW=https://raw.githubusercontent.com/vllm-project/vllm-torchtpu/cpk/dsv41-mh-decode-main/benchmarks/dsv41_tpu7x/public
K=/workspace/vllm-torchtpu/benchmarks/dsv41_tpu7x/public   # where the kit lands inside each pod

# 1. Pods (one per role), about 2 min
curl -fsSL $RAW/pod.yaml -o pod.yaml
for r in prefill decode; do sed "s/ROLE/$r/g" pod.yaml | kubectl apply -f -; done
kubectl wait --for=condition=Ready pod/dsv41-prefill pod/dsv41-decode --timeout=15m

# 2. Bootstrap both pods in parallel (detached), about 11 min each
for r in prefill decode; do
  kubectl exec dsv41-$r -- bash -c "curl -fsSL $RAW/bootstrap.sh -o /tmp/bootstrap.sh && \
    nohup setsid bash /tmp/bootstrap.sh > /dev/shm/bootstrap.log 2>&1 < /dev/null &"
done
#    Re-run this until both print BOOTSTRAP_OK
for r in prefill decode; do kubectl exec dsv41-$r -- tail -n1 /dev/shm/bootstrap.log; done

# 3. Start both roles with the same RUN_ID. Prefill is healthy in about 10 min, decode in 20–35 min.
RUN_ID=dsv41-$(date +%s)
for r in decode prefill; do kubectl exec dsv41-$r -- env RUN_ID=$RUN_ID bash $K/serve.sh $r; done

# 4. Correctness gates, on the prefill pod. Exits 1 with "not healthy yet" until both roles are up; re-run it.
DECODE_IP=$(kubectl get pod dsv41-decode -o jsonpath='{.status.podIP}')
kubectl exec dsv41-prefill -- bash $K/check.sh $DECODE_IP

# 5. Decode throughput, on the decode pod, about 5 min for 3 repeats (also saved to /dev/shm/sweep_rep*.log)
kubectl exec dsv41-decode -- bash $K/sweep.sh 3
```

Server logs are `/dev/shm/pd_prefill.log` and `/dev/shm/pd_decode.log` inside the pods.

## Pass criteria

Step 4 must print all of:

- `[PD-GATES] hard_gates_passed=True`
- `[GATE-EQ] … equal=True` at 64, 700, 1,500, 5,000 and 20,000 prompt tokens (prefill-then-decode output equals decode-local output, greedy)
- `STAGE3_SKIPPED_SENDS 0` and `STAGE3_CHECK_OK 1` (every KV transfer completed)

The server logs must contain none of: `Traceback`, `annotation groups with gaps`, `sized globally`, `must match world size`.

## Expected performance

Decode-only, greedy, 4,000-token context, 1,024 output tokens, DSpark K=5.
Read repeats 2 and 3; repeat 1 runs on a cold prefix cache and is about 10% lower.

| Load | Per decode host (4 chips) | Per chip | Per user | Step time | Accepted tokens/step |
|---|---|---|---|---|---|
| 64 seqs/rank (512 concurrent) | **21,715 / 21,720 tok/s** | 5,430 tok/s | 42.4 tok/s | 72.7–72.9 ms | 3.08–3.09 |
| 32 seqs/rank (256 concurrent) | **16,280 / 16,420 tok/s** | 4,070–4,105 tok/s | 63.6–64.1 tok/s | 48.3–48.4 ms | 3.08–3.10 |

Read the `interior=` figure on each `[X1]` line. The `@3.5=` figure on the same line is a projection to an
acceptance of 3.5 tokens/step, not a measurement.

The raw output of the validation run is in `../results/port_main/public_recipe_20261123/`.

## Settings that matter on the public runtime

`serve.sh` sets all of these. They are listed because changing them breaks the run.

| Setting | Why |
|---|---|
| `VLLM_DSV41_STREAMIDX_SCHED_GROUPS=0` | The stream-index kernel's scheduling groups do not compile on `torch_tpu 0.2.0` |
| `PREFILL_4X_TP2_MNBT=8192` | A 16,384-token prefill program does not fit in memory on this runtime |
| `VLLM_DSV41_CSA_FUSED_MAX_TOKENS=768` | Must match the decode `MAX_NUM_BATCHED_TOKENS=768` |
| `COMPILE_SIZES=[16, 40, 48, 64, 96, 192, 384, 512, 768]` | Decode buckets compiled at boot; this is most of the decode boot time |

## The two failures seen with `torch-tpu 0.2.0` on the old branch

| Symptom | Cause | What this branch does |
|---|---|---|
| Pallas `jax_op` accepts only full world-size meshes; DP8/TP1 sub-mesh kernels fail | The old branch passes a size-1 per-rank mesh, which `torch_tpu 0.2.0` rejects | Uses upstream `get_pallas_op_mesh()`, which returns no mesh for a single-device rank so the kernel runs on the local device. Do not bypass the check. |
| MoE output 8× off: `tensor a (384) must match tensor b (3072)` | The old branch's `sharded_jax_op` shim falls back silently on 0.2.0 and sizes sharded outputs globally | Uses stock `torch_tpu._internal.pallas.jax_op` with `output_partition_specs` |

## Limits

- The validation run copied the five kit files into the pods with `kubectl cp` because they were not yet on GitHub.
  Everything else was cloned or downloaded from the public sources above. Fetching the kit from `$RAW` has not been run.
- Pinning the image by digest in `pod.yaml` has not been run; the `nightly` tag resolved to the digest above.
  If the tag has moved, replace the image with `vllm/vllm-tpu@sha256:97da0de1…`.
- The numbers are decode-only. Prefill throughput at the 8,192-token chunk and the 600 s mixed-length production
  benchmark have not been measured on this stack.
- `torch-tpu 0.3.0` has not been tried.
- The branch is based on upstream `main` at `aba4aad88`, not on current `main`.
- Creating the tpu7x node pool was not part of the validation; we used an existing pool.
- The serve wrapper deletes the checkpoint from `/dev/shm` once all ranks have loaded, to free host RAM.
  Restarting a role means re-running `bootstrap.sh`, or set `PD_DELETE_CKPT=0` before `serve.sh`.
- If the two hosts do not share a pod network, set `ADVERTISE_HOST` to each host's routable IP before `serve.sh`.
