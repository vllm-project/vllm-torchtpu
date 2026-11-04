# Cloud TPU Setup

Running `vllm-torchtpu` requires access to Google Cloud TPUs (such as **TPU v7x / Ironwood** or **TPU v6e / Trillium**). This guide walks through choosing a TPU generation, setting up a Google Cloud project, and provisioning a TPU VM.

## TPU Versions and Topologies

Tensor Processing Units (TPUs) are Google's custom-developed application-specific
integrated circuits (ASICs) used to accelerate machine learning workloads. TPUs
are available in different versions each with different hardware specifications.
For more information about TPUs, see [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm).

The following TPU versions are compatible with `vllm-torchtpu`. Selecting a topology allows you to configure the physical arrangements of the TPU chips, improving throughput and networking performance.

### Recommended

<div class="grid cards" markdown>

- __TPU v7x (Ironwood)__

    <span class="cg-badge cg-badge-recommended">Recommended</span>

    Next-generation architecture for bleeding-edge research and ultra-large models. This is the generation `vllm-torchtpu` is continuously validated on in CI.

    [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/tpu7x)

</div>

### Supported

<div class="grid cards" markdown>

- __TPU v6e (Trillium)__

    <span class="cg-badge cg-badge-supported">Supported</span>

    Optimal performance for mainstream AI workloads with balanced memory and compute. Not covered by CI.

    [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v6e#configurations)

</div>

### Experimental

<div class="grid cards" markdown>

- __TPU v5e__

    <span class="cg-badge cg-badge-experimental">Experimental</span>

    Cost-effective performance for medium-to-large scale inference and training.

    [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v5e#tpu-v5e-config)

- __TPU v5p__

    <span class="cg-badge cg-badge-experimental">Experimental</span>

    High-performance architecture optimized for peak compute and memory bandwidth.

    [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v5p#tpu-v5p-config)

- __TPU v4__

    <span class="cg-badge cg-badge-experimental">Experimental</span>

    Previous generation flagship architecture for large-scale training.

    [View Topology Guidelines &rarr;](https://cloud.google.com/tpu/docs/v4#tpu-v4-config)

</div>

## Prerequisites

Before provisioning a TPU VM, ensure you have:

1. **A Google Cloud project** with billing enabled.
2. **The Google Cloud CLI (`gcloud`)** installed and authenticated on your workstation:

    ```bash
    gcloud auth login
    gcloud config set project PROJECT_ID
    ```

3. **The Cloud TPU and Compute Engine APIs enabled**:

    ```bash
    gcloud services enable tpu.googleapis.com compute.googleapis.com
    ```

4. **TPU quota** in your target zone. You can check it under **IAM & Admin -> Quotas & System Limits** in the Google Cloud Console. For more information, see [TPU quota](https://cloud.google.com/tpu/docs/quota#tpu_quota).

For TPU pricing information, see [Cloud TPU pricing](https://cloud.google.com/tpu/pricing).

## Provisioning Cloud TPUs

Google Cloud supports two primary APIs for provisioning TPUs:

- **[Compute Engine API](https://cloud.google.com/tpu/docs/tpus-in-compute-engine) (`gcloud compute instances create`)**: Used for **TPU v6e (Trillium)** and **TPU v5p**.
- **[Cloud TPU API](https://cloud.google.com/tpu/docs/queued-resources) (`gcloud alpha compute tpus queued-resources create`)**: Used for **TPU v7x (Ironwood)**, **TPU v5e**, and **TPU v4**.

You can also choose between two capacity models:

- **Standard (On-Demand)**: Immediate allocation at standard pay-as-you-go rates.
- **Flex-start (DWS)**: Discounted capacity via Dynamic Workload Scheduler (DWS) that runs uninterrupted for up to 7 days. Flex-start is supported on **TPU v7x**, **TPU v6e**, **TPU v5p**, and **TPU v5e** (not on **TPU v4**). For more details, see [About Flex-start VMs](https://cloud.google.com/compute/docs/instances/about-flex-start-vms) and [DWS Pricing](https://cloud.google.com/products/dws/pricing#flex-start-tpu-vm-pricing).

Select your target TPU generation, chip count, and provisioning model below to generate the `gcloud` provisioning command. Be sure to replace placeholder variables (like `PROJECT_ID` and `ZONE`) with your own values before running.

<div class="command-generator-container" id="prov-generator">
  <div class="cg-row">
    <div class="cg-label">1. Hardware Generation</div>
    <div class="cg-options">
      <button type="button" class="cg-btn active" data-group="prov_hw" data-val="v7x" aria-pressed="true">TPU v7x (Ironwood)</button>
      <button type="button" class="cg-btn" data-group="prov_hw" data-val="v6e" aria-pressed="false">TPU v6e (Trillium)</button>
      <button type="button" class="cg-btn" data-group="prov_hw" data-val="v5p" aria-pressed="false">TPU v5p</button>
      <button type="button" class="cg-btn" data-group="prov_hw" data-val="v5e" aria-pressed="false">TPU v5e</button>
      <button type="button" class="cg-btn" data-group="prov_hw" data-val="v4" aria-pressed="false">TPU v4</button>
    </div>
  </div>
  <div class="cg-row">
    <div class="cg-label">2. Chip Count</div>
    <div class="cg-options">
      <button type="button" class="cg-btn" data-group="prov_chips" data-val="1" aria-pressed="false">1 Chip</button>
      <button type="button" class="cg-btn active" data-group="prov_chips" data-val="4" aria-pressed="true">4 Chips</button>
      <button type="button" class="cg-btn" data-group="prov_chips" data-val="8" aria-pressed="false">8 Chips</button>
      <button type="button" class="cg-btn" data-group="prov_chips" data-val="16" aria-pressed="false">16 Chips</button>
      <button type="button" class="cg-btn" data-group="prov_chips" data-val="32" aria-pressed="false">32 Chips</button>
      <button type="button" class="cg-btn" data-group="prov_chips" data-val="64" aria-pressed="false">64 Chips</button>
    </div>
  </div>
  <div class="cg-row">
    <div class="cg-label">3. Provisioning Model</div>
    <div class="cg-options">
      <button type="button" class="cg-btn active" data-group="prov_model" data-val="standard" aria-pressed="true">Standard / Reserved</button>
      <button type="button" class="cg-btn" data-group="prov_model" data-val="flex_start" aria-pressed="false">Flex-start (DWS)</button>
    </div>
  </div>
  <div class="cg-output-container">
    <div class="cg-instructions" id="prov-output-instructions"></div>
    <div class="highlight"><pre><code id="prov-output-command"></code></pre></div>
  </div>
</div>

| Parameter | Description |
|-----------|-------------|
| `PROJECT_ID` | Your Google Cloud project ID. |
| `ZONE` | The Google Cloud zone where you have TPU quota (e.g., `us-central1-c`, `us-east5-a`, `europe-west4-a`, `us-central2-b`). See [TPU regions and zones](https://cloud.google.com/tpu/docs/regions-zones) for availability. |
| `SERVICE_ACCOUNT` | The email address for your service account, found in the Cloud Console under IAM Service Accounts (e.g., `tpu-service-account@<your_project_ID>.iam.gserviceaccount.com`). Optional for Cloud TPU API (queued resources) commands; if you omit `--service-account`, the Compute Engine default service account is used. |
| `RUNTIME_VERSION` | Automatically populated by the generator above based on your selected TPU hardware generation. |

!!! note "TPU v7x dual-core architecture"
    Each physical **TPU v7x (Ironwood)** chip exposes **2 logical TensorCore devices** to PyTorch and JAX. A single 4-chip TPU v7x host (`tpu7x-8`) provides **8 logical TPU devices** (`--tensor-parallel-size=8`) and 768 GB of total HBM.

## Connecting and Checking Status

Once your queued resource or GCE instance reaches the `ACTIVE` / `RUNNING` state, SSH into the host.

### Compute Engine (TPU v6e, v5p)

Check the status of your instance:

```bash
gcloud compute instances describe my-tpu-vm \
  --project=PROJECT_ID \
  --zone=ZONE
```

Connect using SSH:

```bash
gcloud compute ssh my-tpu-vm \
  --project=PROJECT_ID \
  --zone=ZONE
```

### Queued Resources (TPU v7x, v5e, v4)

Check the status of your queued resource:

```bash
gcloud alpha compute tpus queued-resources describe my-queued-resource \
  --project=PROJECT_ID \
  --zone=ZONE
```

Connect to worker 0 using SSH:

```bash
gcloud compute tpus tpu-vm ssh my-tpu-name \
  --project=PROJECT_ID \
  --zone=ZONE \
  --worker=0
```

## Verify TPU Devices

Inside the TPU VM, verify that the accelerator devices are visible before installing `vllm-torchtpu`:

```bash
ls /dev/vfio/
```

Once `vllm-torchtpu` is installed (which includes `tpu-info`), you can inspect chip utilization and HBM memory usage at any time with:

```bash
tpu-info
```

## Clean Up Resources

To avoid incurring unnecessary charges when you are finished, delete your TPU resources:

### Compute Engine (TPU v6e, v5p)

```bash
gcloud compute instances delete my-tpu-vm \
  --project=PROJECT_ID \
  --zone=ZONE
```

### Queued Resources (TPU v7x, v5e, v4)

```bash
gcloud alpha compute tpus queued-resources delete my-queued-resource \
  --project=PROJECT_ID \
  --zone=ZONE \
  --force
```

## Next Steps

Once your TPU VM is running, follow the [installation and quickstart instructions](https://github.com/vllm-project/vllm-torchtpu#readme) to install `vllm-torchtpu` and serve your first model.
