---
name: gh-pr-api
description: Reference for GitHub PR API operations using gh CLI. Use when posting review comments, fetching PR data, or interacting with GitHub pull requests programmatically. Covers the correct endpoints, parameter formats, and common pitfalls.
---

# GitHub PR API Operations

Reference for interacting with GitHub pull requests via `gh` CLI. This skill documents the correct API patterns — several GitHub endpoints have subtle gotchas that cause silent failures.

## Fetching PR Data

### PR metadata
```bash
gh pr view <number> --repo <owner>/<repo> --json title,body,state,author,baseRefName,headRefName,files,additions,deletions,changedFiles,commits
```

### PR diff
```bash
gh pr diff <number> --repo <owner>/<repo>
```

### PR review comments (inline)
```bash
gh api repos/<owner>/<repo>/pulls/<number>/comments --paginate
```

### PR reviews (top-level)
```bash
gh api repos/<owner>/<repo>/pulls/<number>/reviews --paginate
```

### PR issue comments (conversation)
```bash
gh pr view <number> --repo <owner>/<repo> --json comments --jq '.comments[]'
```

### PR checks/status
```bash
gh pr checks <number> --repo <owner>/<repo>
```

## Posting Reviews

### Batch review with inline comments (REQUIRED approach)

Always use the **Create a review** endpoint to post inline comments. This is the only reliable method.

```bash
gh api repos/<owner>/<repo>/pulls/<number>/reviews \
  --method POST \
  --input - <<'PAYLOAD'
{
  "commit_id": "<head_sha>",
  "event": "COMMENT",
  "body": "Top-level review body text",
  "comments": [
    {
      "path": "path/to/file.py",
      "line": 42,
      "side": "RIGHT",
      "body": "Inline comment text"
    },
    {
      "path": "path/to/other.py",
      "line": 10,
      "side": "RIGHT",
      "body": "Another inline comment"
    }
  ]
}
PAYLOAD
```

Key details:
- `line` is the **new file line number** (the actual line in the file, not a diff offset)
- `side: "RIGHT"` means the new version of the file (use this for comments on added/changed lines)
- `side: "LEFT"` for comments on deleted lines (use the old file line number)
- `event` can be: `"COMMENT"`, `"APPROVE"`, or `"REQUEST_CHANGES"`
- All comments are submitted atomically as one review
- `commit_id` should be the PR's HEAD SHA: `gh api repos/<owner>/<repo>/pulls/<number> --jq '.head.sha'`

### DO NOT use the individual comment endpoint

The `POST /repos/{owner}/{repo}/pulls/{number}/comments` endpoint with the `position` parameter is unreliable:
- The `position` numbering is ambiguous (per-file vs whole-diff)
- Comments consistently land on wrong lines
- The `line` + `subject_type` fields are not supported on this endpoint despite documentation suggesting otherwise

## Fetching File Contents at PR Revision

To read a specific file at the PR's HEAD:
```bash
gh api repos/<owner>/<repo>/contents/<path>?ref=<branch_or_sha> --jq '.content' | base64 -d
```

Or clone/worktree the branch and read locally (preferred for multiple files).

## Parsing Review Comments

Useful jq patterns for extracting review comment data:

```bash
# Get all non-bot review comments with file positions
gh api repos/<owner>/<repo>/pulls/<number>/comments --paginate | \
  python3 -c "
import json, sys
comments = json.load(sys.stdin)
for c in comments:
    user = c['user']['login']
    path = c.get('path','')
    line = c.get('original_line','')
    body = c.get('body','')[:400]
    print(f'--- {user} on {path}:{line} ---')
    print(body)
    print()
"
```

## Common Patterns

| Task | Command |
|------|---------|
| Get HEAD SHA | `gh api repos/o/r/pulls/N --jq '.head.sha'` |
| Get base branch | `gh api repos/o/r/pulls/N --jq '.base.ref'` |
| Get changed files | `gh pr view N --repo o/r --json files --jq '.files[].path'` |
| Get PR author | `gh pr view N --repo o/r --json author --jq '.author.login'` |
| List PR labels | `gh pr view N --repo o/r --json labels --jq '.labels[].name'` |