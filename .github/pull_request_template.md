<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED.

## Purpose
<!-- Explain the problem this PR solves or the feature it introduces. Link to any related issues (e.g., Fixes #123). -->

## Type of Change
<!-- Please select the canonical category matching your PR title prefix (multi-tagging is permitted, e.g., [Bugfix][Core]): -->
- [ ] `[Bugfix]`: Bug fixes (please include regression tests)
- [ ] `[Core]`: Core runtime logic (Runner, KV cache pools, scheduler, speculative decoding, compilation, distributed)
- [ ] `[Kernel]`: Compute kernels & custom ops (JAX / Pallas kernels, custom op bridges)
- [ ] `[Model]`: Adding or improving models (model name must be in PR title, e.g., `[Model][Qwen] ...`)
- [ ] `[Frontend]`: OpenAI API server, CLI, entrypoints, proxy/disaggregated serving
- [ ] `[CI/Build]`: Docker images, Buildkite pipelines, dependency pins, pre-commit configuration
- [ ] `[Doc]`: Documentation improvements, guides, docstrings
- [ ] `[Misc]`: Miscellaneous changes not fitting above (please use sparingly)

## Test Plan
<!-- Describe the tests you ran to verify your changes and provide instructions/commands so others can reproduce. -->
<!-- TODO: show how to test it on cpu or tpu -->

## Test Results
<!-- Paste the passing test output or describe reproduction results. -->

---
<details>
<summary> PR Checklist (Click to Expand) </summary>

- [ ] **DCO Sign-off**: Every commit has a Developer Certificate of Origin sign-off (`Signed-off-by: Your Name <email>`) created via `git commit -s`.
- [ ] **No Duplicate Work**: I have searched existing issues and open pull requests to ensure this work is not duplicated.
- [ ] **Architectural RFC**: For major changes (>500 lines of core logic excluding tests/configs), an RFC issue was opened and discussed first.
- [ ] **AI-Assisted Contributions**: If AI tools were used, I have reviewed every changed line, validated behavior end-to-end, and included appropriate commit trailers (`Co-authored-by:` / `Assisted-by:`).
- [ ] **CI Execution (`ready` label)**: Full CI on physical TPU hardware will run once a repository collaborator or reviewer inspects the PR for safety and attaches the `ready` label.

</details>

**BEFORE SUBMITTING, PLEASE READ OUR [CONTRIBUTION GUIDE](https://github.com/vllm-project/vllm-torchtpu/blob/main/CONTRIBUTING.md)**
