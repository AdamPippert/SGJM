# AGENTS.md

Guidance for autonomous coding agents (Codex, Claude Code, etc.) working in this repo.

This project is sometimes run by an agent in fully autonomous mode (no interactive
approval). Behave as if no human will review a prompt mid-run: prefer reversible
steps, do not run destructive commands unless the task explicitly calls for it, and
stop and report rather than guess when a step is genuinely ambiguous.

## Commit & author identity — required

**No AI tool may appear as a committer, co-author, or in commit metadata.**

- All commits must be authored as `Adam Pippert <adam.pippert@gmail.com>`.
- Do **not** add `Co-Authored-By:` trailers naming Claude, Codex, an AI, or a model.
- Do **not** add generated-by / "🤖 Generated with ..." footers.
- Do **not** set `user.name` or `user.email` to anything containing `claude`,
  `codex`, `openai`, or `anthropic`.
- If the repo's git `user.name`/`user.email` is unset or contains any of those
  strings, **stop and report** — do not silently reconfigure it and do not commit.

Adam Pippert remains the sole author of record for all work, regardless of which
tool drafted it.

See `CLAUDE.md` for the Claude-specific statement of the same policy.

## Pushing

`origin` points at GitHub. Do **not** push unless the task explicitly asks you to.
When in doubt, commit locally and report what's ready to push.
