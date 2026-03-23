# Ray LLM Module — Claude Code Context

## Overview

This is the `ray/llm` module within the Ray project (`ray-project/ray`). It provides LLM serving infrastructure on top of Ray Serve, with engine backends (currently vLLM), placement group management for multi-accelerator deployments, and configuration abstractions.

## Repository

- **GitHub:** `ray-project/ray`
- **Module path:** `python/ray/llm/`
- **Key subsystems:**
  - `_internal/serve/` — Core serving infrastructure (configs, engines, deployments)
  - `_internal/serve/engines/vllm/` — vLLM engine integration
  - `_internal/serve/core/configs/` — LLMConfig and related configuration
  - `_internal/batch/` — Batch processing infrastructure
  - `tests/` — Test suites (BUILD.bazel for test targets)

## Skills

When the user says `/ray-review <PR URL>` or asks to review a pull request, read and follow the instructions in `.claude/skills/review-pr/SKILL.md` (relative to this file).

When you need to interact with the GitHub PR API (posting reviews, fetching PR data), read `.claude/skills/gh-pr-api/SKILL.md` for the correct endpoints and patterns. In particular, always use the batch review endpoint for inline comments — the individual comment endpoint is unreliable.

Available skills:
- **review-pr** — Co-review GitHub PRs with a Ray maintainer. Gathers context, checks out a worktree, analyzes before/after impact, reviews code, and posts feedback after user approval.
- **gh-pr-api** — Reference for GitHub PR API operations (correct endpoints, posting reviews, common pitfalls).

## Review Context

When reviewing PRs to this module, keep in mind:

### Architecture Principles
- **Prefer general primitives over narrow hardware-specific ones.** If a change can be generalized without excessive complexity, the general approach is preferred.
- **Deep modules over shallow ones.** Interfaces should be simple; complexity belongs in implementations, not leaking through APIs.
- **Manage complexity deliberately.** Watch for change amplification (one change forcing many others), cognitive load, and unknown unknowns.

### CI Infrastructure
- CPU tests: tagged `"cpu"` in BUILD.bazel, run in standard Buildkite
- GPU tests: tagged `"gpu"`, run on GPU instances (`g6-large`, `gpu-large`)
- All LLM tests are tagged `"team:llm"`
- CI config lives in `.buildkite/llm.rayci.yml`