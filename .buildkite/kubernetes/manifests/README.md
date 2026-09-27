# Workload manifests

Handed to the launcher with `--manifest`, for workloads the built-in one-pod
Job cannot express: several roles that talk to each other, or the hosts of one
multi-host slice.

Named for what they contain. `qwen3.5-397b-1p1d.yaml` serves that model in that
topology; `ray-multihost-slice.yaml` brings Ray up across a slice and runs
whatever command the step passes, so it is named for the bring-up rather than
any one benchmark.

`v7x/` predates this and is not referenced by any pipeline.

## What a manifest has to say

Only the hardware it wants, and what is genuinely its own:

- `nodeSelector` with the accelerator and topology, and a `google.com/tpu`
  count. Those three pick the queue, so they cannot come from anywhere else.
- Its containers, and `activeDeadlineSeconds` if the fleet's default is wrong
  for it.

Everything else comes from the fleet, through one annotation on the JobSet:

```yaml
metadata:
  annotations:
    tpu-ci.google.com/defaults: standard
```

That supplies the compilation and model caches with their mounts, the gcsfuse
sidecar settings, the TPU toleration, the service account, `restartPolicy`, the
TTL, the memory request for whichever host the pod lands on, and the retry rules
that let a run survive its node being repaired. It is defined once in ci-infra,
in `kueue/launcher/pod_defaults.yaml`.

The merge is additive, so a role can still declare a volume of its own — both
manifests here add a per-pod profile cache that shadows a subtree of the shared
one — or override anything it needs to differ on.

Roles that hold no chips inherit only the eviction policy: the caches are sized
from a TPU host's memory and a chipless role does not run on one, so it states
what it needs itself.

Do not add a CPU request. It is a scheduling floor checked against the template
the autoscaler builds for the shape, and one large enough to matter stops the
pool building nodes at all.
