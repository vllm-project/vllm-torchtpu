# Buildkite pipelines

`pipeline_tests.yml` runs tests in `tests/` marked `cpu_test` on `cpu_64_core` and excludes
them from the general TPU jobs. PR steps exclude `nightly` tests; nightly
steps include them. Both use the CI image built by `pipeline_build.yml`.
The CPU steps set `CPU_ONLY=1` to run Docker without privileged mode or device
mounts and select the JAX CPU backend. Host model caches and test results
use `$HOME/.cache/vllm-torchtpu-ci` on CPU agents and `/mnt/disks/persist`
on TPU agents.

For a test file that runs entirely without accelerator hardware, add
`pytestmark = pytest.mark.cpu_test` after its imports. Use
`@pytest.mark.cpu_test` for individual tests in mixed files. Module imports
must work on CPU because pytest collects tests before applying marker filters.
Mocks that use `torch.tpu` must supply that object on CPU hosts.

Run the CPU tests locally with the project's test dependencies installed:

```bash
JAX_PLATFORMS=cpu pytest tests/ -m cpu_test
```
