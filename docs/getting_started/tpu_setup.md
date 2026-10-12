# Cloud TPU Setup

Running `vllm-torchtpu` requires access to Google Cloud TPUs (such as **TPU v7x / Ironwood** or **TPU v6e / Trillium**). This guide walks through setting up a Google Cloud project, requesting quota, and provisioning a TPU VM slice.

---

## 1. Prerequisites

Before provisioning a TPU VM, ensure you have:

1. **A Google Cloud Project** with billing enabled.
2. **The Google Cloud CLI (`gcloud`)** installed and authenticated on your workstation:

   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Cloud TPU API Enabled**:

   ```bash
   gcloud services enable tpu.googleapis.com compute.googleapis.com
   ```

4. **TPU Quota**: Verify that your project has sufficient TPU quota in your target zone under **IAM & Admin -> Quotas & System Limits** in the Google Cloud Console.
5. **Torch TPU registry access**: `torch-tpu` is installed from a private registry. Request read access from the Torch TPU team (see [Installation](installation.md#prerequisites)).

---

## 2. Provision a Cloud TPU VM

Select your target TPU generation, chip count, and provisioning model below to generate the `gcloud` provisioning command:

<div class="command-generator-container" id="prov-generator">
  <div class="cg-row">
    <div class="cg-label">1. Hardware Generation</div>
    <div class="cg-options">
      <button type="button" class="cg-btn active" data-group="prov_hw" data-val="v7x" aria-pressed="true">TPU v7x (Ironwood)</button>
      <button type="button" class="cg-btn" data-group="prov_hw" data-val="v6e" aria-pressed="false">TPU v6e (Trillium)</button>
      <button type="button" class="cg-btn" data-group="prov_hw" data-val="v5p" aria-pressed="false">TPU v5p</button>
      <button type="button" class="cg-btn" data-group="prov_hw" data-val="v5e" aria-pressed="false">TPU v5e</button>
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

> [!NOTE]
> **TPU v7x Dual-Core Architecture**: Each physical **TPU v7x (Ironwood)** chip exposes **2 logical TensorCore devices** to PyTorch and JAX. A single 4-chip TPU v7x host (`tpu7x-8`) provides **8 logical TPU devices** (`--tensor-parallel-size=8`) and 768 GB of total HBM.

---

## 3. Connect to Your TPU VM

Once your queued resource or GCE instance reaches the `ACTIVE` / `RUNNING` state, SSH into the host:

### Queued Resources / TPU VM API

```bash
# Check status of queued resource
gcloud alpha compute tpus queued-resources list \
  --project=PROJECT_ID \
  --zone=ZONE

# SSH into worker 0
gcloud compute tpus tpu-vm ssh my-tpu-name \
  --project=PROJECT_ID \
  --zone=ZONE \
  --worker=0
```

### Compute Engine (GCE) Instance

```bash
gcloud compute ssh my-tpu-vm \
  --project=PROJECT_ID \
  --zone=ZONE
```

---

## 4. Verify TPU Devices

Inside the TPU VM, verify that the accelerator devices are visible before installing `vllm-torchtpu`. Depending on the TPU generation, they appear under `/dev/vfio/` or as `/dev/accel*`:

```bash
ls /dev/vfio/ /dev/accel* 2>/dev/null
```

Once `vllm-torchtpu` is installed (which includes `tpu-info`), you can inspect chip utilization and HBM memory usage at any time with:

```bash
tpu-info
```

---

## 5. Clean Up Resources

To avoid incurring unnecessary charges when you are finished testing, delete your TPU resources:

### Queued Resources

```bash
gcloud alpha compute tpus queued-resources delete my-queued-resource \
  --project=PROJECT_ID \
  --zone=ZONE \
  --force
```

### Compute Engine (GCE) Instance

```bash
gcloud compute instances delete my-tpu-vm \
  --project=PROJECT_ID \
  --zone=ZONE
```
