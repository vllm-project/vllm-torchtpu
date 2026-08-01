# GCE Setup for Multi-Host vLLM-TorchTPU

This guide provides step-by-step instructions for deploying a Ray cluster and serving models with `vllm-torchtpu` across multiple TPU VM hosts on Google Compute Engine (GCE).

## Prerequisites

Find the IP addresses of your TPU VMs by running:

```bash
gcloud compute tpus tpu-vm describe <VM_NAME> \
  --zone=<VM_ZONE> \
  --project=<YOUR_PROJECT>
```

---

## Step 1: Deploy Ray Cluster

On your local machine, clone the repository and run the deployment script:

```bash
git clone https://github.com/vllm-project/vllm-torchtpu.git
cd vllm-torchtpu/

# Replace the IPs with your own IP addresses
./scripts/multihost/deploy_cluster.sh \
  -s ./scripts/multihost/run_cluster.sh \
  -d "<link_to_your_docker_image>" \
  -t "<HF_TOKEN>" \
  -H "<IP_0>" \
  -i "<INTERNAL_IP_0>" \
  -W "<IP_1>,<IP_2>"
```

### Parameter Descriptions
- `-H`: External IP of host0
- `-i`: Internal IP of host0
- `-W`: External IPs of worker hosts (`host1,host2,...`)

---

## Step 2: Start vLLM Server

1. SSH into worker 0 (the head node):

```bash
gcloud compute tpus tpu-vm ssh <VM_NAME> \
  --zone=<VM_ZONE> \
  --project=<YOUR_PROJECT> \
  --worker=0
```

1. Log into the Docker container:

```bash
sudo docker exec -it node /bin/bash
```

1. Start the vLLM server (example command given below, adjust parameters as needed):

```bash
vllm serve "Qwen/Qwen3-8B" \
  --tensor_parallel_size=16 \
  --max-model-len=256 \
  --max-num-batched-tokens=256 \
  --async-scheduling
```

---

## Step 3: Verification

Start another terminal, log into the docker container, and verify serving using `curl`:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "prompt": "Hello, my name is",
    "max_tokens": 20,
    "temperature": 0
  }'
```
