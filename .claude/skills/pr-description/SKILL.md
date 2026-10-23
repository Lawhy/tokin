---
name: pr-description
description: Write or rewrite a tokin pull request description in its fixed, technical, terse format (Issue / Solution / Behaviour / Evidence / References). Use when opening a PR or editing one's body.
---

# PR description

A reviewer reads the description to learn what was wrong, what changed, and why to believe it works. Give exactly that. Every line either states a fact a reviewer needs or goes.

## Format

Title: the commit's, in Conventional Commits form. When the PR needs another merged first, its first line is
`Depends on #N.`

```markdown
## Issue
<What is wrong and what it costs, in 1–3 lines, with a number.>

## Solution
- <One bullet per change, naming the identifier: `Class.method`, `module`, a renamed field.>
- <A design choice gets half a line of reason only where a reviewer would otherwise undo it.>

## Behaviour
<Only when what a harness, trainer or user observes changes: before/after, as a table when there are cases.
Say "Breaking:" first when an interface they call changes.>

| Case | Before | After |
|---|---|---|

## Evidence
- Tests: <added or changed tests, by name or by what they cover; which fail on `main`>.
- Suites: <default / real-tokenizer / integration, pass>.
- Measured: <numbers on real data, with sample size: before vs after, or the change's footprint>.

## References
- <project / file / doc whose behaviour this matches or departs from, one line each, a link where public>.
```

Leave out a section with nothing to say, except Issue, Solution and Evidence.

Each fact goes in one section only: the defect and its cost in Issue, what other projects do in References, what
was checked in Evidence (including how clients react to the new output).

## Rules

- Lead with the defect, not the history. No "Previously we…", no account of how it was found.
- Numbers over adjectives: "8 of 356 runs", not "a noticeable share".
- A survey of other projects is the References list, one line per project, not prose.
- No restating the commit message, the diff, or the docstrings.
- Identifiers in single backticks.
- The repository is public: nothing internal to the developer's organisation (cluster, team, internal checkpoint
  names, paths). Say "a checkpoint" or name only public models.
- Ask before `gh pr create`. Editing an open PR's body when asked is fine.

## Example

```markdown
## Issue
A `<tool_call>` block whose tool name tokin cannot read stays in the content, so the reply carries no call and pi or
opencode ends the run: 8 of 356 Qwen3.6 pi runs, on `<function,bash>`, `<parameter=bash>`, `<function read>`.

## Solution
- `ToolParser.parse`: every block is a call; an unreadable name becomes `name = ""`, the block its `arguments`.
  No name rather than a placeholder a harness could hold a tool by.
- `ChatTemplate.parse`: reasoning left open ends at the first block.

## Behaviour
| Block | Before | After |
|---|---|---|
| name unreadable | text | call, `name=""`, raw arguments |
| mentioned in prose | text | call, `name=""` |

## Evidence
- Tests: blocks naming no tool now expect a nameless call (hermes, qwen3_coder, glm45, template).
- Suites: default and real-tokenizer pass.
- Measured: re-parse of ~113k generations from five terminal-bench sweeps: 29 replies change.

## References
- SGLang `Qwen3CoderDetector.parse_streaming_increment`: streams a nameless call for such blocks.
- vLLM `qwen3_coder`, SGLang non-stream: drop the block.
```
