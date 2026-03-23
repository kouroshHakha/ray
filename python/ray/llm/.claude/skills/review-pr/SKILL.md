---
name: review-pr
description: Co-review a GitHub PR with the user acting as a Ray maintainer. Use when the user says /ray-review <PR URL> or asks to review a pull request. Walks through understanding the contribution, building context, reviewing code, and posting feedback — all with user approval at each step.
---

# Co-Review a Pull Request

You are co-reviewing a PR with a Ray maintainer. Your role is to accelerate their review by doing the legwork — gathering context, understanding the contribution, identifying issues, and drafting feedback — while they make the final calls.

**Announce at start:** "Starting PR co-review. Let me gather context."

## Phase 1: Gather Context

Collect all PR data in parallel:

```bash
# Run all of these in parallel
gh pr view <number> --repo <owner>/<repo> --json title,body,state,author,baseRefName,headRefName,files,additions,deletions,changedFiles,commits,labels
gh pr diff <number> --repo <owner>/<repo>
gh api repos/<owner>/<repo>/pulls/<number>/comments --paginate
gh api repos/<owner>/<repo>/pulls/<number>/reviews --paginate
gh pr view <number> --repo <owner>/<repo> --json comments --jq '.comments[]'
```

Also fetch the HEAD SHA for later use:
```bash
gh api repos/<owner>/<repo>/pulls/<number> --jq '.head.sha'
```

## Phase 2: Checkout Worktree

Create a worktree to browse the code at the PR's state:

1. Navigate to the repo root
2. Fetch the PR branch:
   ```bash
   git fetch origin pull/<number>/head:pr-<number>
   ```
3. Create a worktree:
   ```bash
   git worktree add .worktrees/pr-<number> pr-<number>
   ```
4. Use the worktree path for all file reads during review

After review is complete, clean up:
```bash
git worktree remove .worktrees/pr-<number>
git branch -D pr-<number>
```

## Phase 3: Understand the Contribution

Present a structured summary to the reviewer:

### 3a. Contribution Overview
- **What:** One-sentence summary of what the PR does
- **Why:** The motivation — link to issues, prior discussion, or problem statement
- **Who:** Author context — are they a regular contributor? First-time?
- **Scope:** Files changed, lines added/deleted, number of commits

### 3b. Before/After Impact

This is critical for the reviewer. For each significant change, show the before and after states clearly:

```
### Change: [short description]

**Before:**
- [How the code/system behaved before]
- [Relevant code snippet or architecture]

**After:**
- [How it behaves now]
- [New code snippet or architecture]

**Impact:** [What this means for users, performance, correctness, etc.]
```

For architectural changes, draw ASCII diagrams showing the before/after:

```
BEFORE:                          AFTER:
┌──────────┐                     ┌──────────┐
│ ConfigA  │──▶ logic            │ ConfigA  │──▶ StrategyA
└──────────┘                     └──────────┘
                                 ┌──────────┐
                                 │ ConfigB  │──▶ StrategyB
                                 └──────────┘
```

### 3c. Review Map

Suggest which files to review first, ordered by risk/importance:

```
## Suggested Review Order (derisk first)

1. **path/to/core_change.py** — [why this is highest risk]
2. **path/to/config.py** — [why review this second]
3. **path/to/tests.py** — [verify coverage]
4. **path/to/build.yaml** — [minor, check last]
```

Factors for ordering:
- Files with the most architectural impact go first
- Files that change public APIs or contracts go before implementation details
- Test files come after the code they test
- Config/build files come last unless they're the main concern

## Phase 4: Perform Review

Read the changed files in the worktree. For each file, examine the diff in context of the surrounding code. Identify issues and categorize them.

### Issue Categories

**Critical (must fix before merge):**
- Correctness bugs
- Security vulnerabilities
- Data loss risks
- Breaking changes to public APIs without migration path
- Violations of architectural invariants

**Important (should fix, can discuss):**
- Design issues that increase maintenance burden
- Missing validation at system boundaries
- Incomplete error handling for realistic failure modes
- Test gaps for important code paths
- Inconsistencies with existing patterns in the codebase

**Suggestions (nice to have):**
- Naming improvements
- Minor simplifications
- Style consistency
- Type hint improvements
- Documentation suggestions

### Review Philosophy

- **Prefer general primitives over narrow specific ones.** If a change solves one case but could be generalized without much added complexity, suggest the general approach.
- **Think about complexity symptoms:** change amplification (does this change force future changes in many places?), cognitive load (is this harder to understand than it needs to be?), and unknown unknowns (are there non-obvious interactions?).
- **Deep modules over shallow ones.** Interfaces should be simple; implementations can be complex. Flag cases where complexity leaks into interfaces.
- **Write from the maintainer's perspective.** You have influence over the project's direction. Frame feedback as decisions about what the codebase should become, not suggestions from the outside.
- **Acknowledge what's good.** If the approach is sound, say so. Don't manufacture issues.

## Phase 5: Present Review to User

Present findings to the user for iteration **before** posting anything to GitHub.

### Structure

1. **Top-level comment** (architectural questions + overall assessment)
2. **Inline comments** (specific code feedback on exact lines)

Present them clearly:

```
## Top-Level Comment (will be posted as review body):

[draft text]

## Inline Comments:

1. **file.py:42** — [category: Critical/Important/Suggestion]
   > [draft comment text]

2. **file.py:130** — [category: Important]
   > [draft comment text]

...
```

### User Iteration

After presenting:
- Ask the user to approve, modify, or remove each comment
- The user may add their own comments to the batch
- The user may ask you to investigate something further before finalizing
- Iterate until the user says to post

Do NOT post to GitHub until the user explicitly approves.

## Phase 6: Post Review

Once approved, use the batch review endpoint (see `gh-pr-api` skill):

```bash
gh api repos/<owner>/<repo>/pulls/<number>/reviews \
  --method POST \
  --input - <<'PAYLOAD'
{
  "commit_id": "<head_sha>",
  "event": "COMMENT",
  "body": "<top-level comment with AI disclaimer>",
  "comments": [
    {
      "path": "<file>",
      "line": <line_number>,
      "side": "RIGHT",
      "body": "<comment text>"
    }
  ]
}
PAYLOAD
```

The top-level comment body MUST include this disclaimer at the end:

```markdown
> [!NOTE]
> This review was co-written with AI assistance (Claude Code).
```

After posting, report the review ID and link to the user.

## Phase 7: Cleanup

Remove the worktree:
```bash
git worktree remove .worktrees/pr-<number>
git branch -D pr-<number>
```

## Red Flags

**Never:**
- Post a review without user approval
- Guess at line numbers — always verify against the actual file
- Leave the worktree behind after review is complete
- Approve or request changes without explicit user instruction (default to `"COMMENT"`)

**Always:**
- Show before/after impact for significant changes
- Categorize every issue (Critical/Important/Suggestion)
- Include the AI assistance disclaimer
- Use the batch review endpoint for inline comments
- Let the user iterate on comments before posting