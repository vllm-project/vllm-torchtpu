# Golden check (temporary)

This directory checks the CI scenarios against what runs today:

- `test_golden.py` runs today's kube scripts unmodified, with stub programs,
  and compares every call they make with the scenario's Plan. A difference
  fails unless `allowlist.yaml` accepts it with a reason.
- `test_baselines.py` compares each scenario's inline baselines with the
  files under `scripts/vllm/benchmarking/baselines`.

It is temporary because it compares the scenarios with what they replace.
M4 deletes this directory, `.github/workflows/scenario-golden.yml` and
`../tests/test_golden_boundary.py` together with today's kube scripts. Until
then:

- nothing outside this directory may import from it;
- new features get plan snapshot tests in `../tests`, not cases here.

Run it from the repository root. It needs bash 4.4 or newer.

```
uv run --project tools/scenario --frozen pytest -c tools/scenario/pyproject.toml .buildkite/kubernetes/scenario/golden
```
