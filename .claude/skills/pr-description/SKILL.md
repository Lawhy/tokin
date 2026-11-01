---
name: pr-description
description: Write or rewrite a tokin pull request description in its fixed, technical, terse format (Issue / Solution / Decisions / Behaviour / Evidence / References). Use when opening a PR or editing one's body.
---

# PR description

A reviewer reads the description to learn what was wrong, what changed, who settled each choice, and why to believe it works. Give exactly that. Every line either states a fact a reviewer needs or goes.

## Format

Title: the commit's, in Conventional Commits form. When the PR needs another merged first, its first line is
`Depends on #N.`

```markdown
## Issue
<What is wrong and what it costs, in 1–3 lines, with a number.>

## Solution
- <One bullet per change, naming the identifier: `Class.method`, `module`, a renamed field.>

## Decisions
- Human: <a choice the human made or imposed>, over <the alternative it replaced>.
- Agent, agreed: <a choice the agent proposed and the human explicitly accepted>: <half a line of reason>.
- Agent, unreviewed: <a choice no human weighed>: <half a line of reason>.

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

Leave out a section with nothing to say, except Issue, Solution, Decisions and Evidence.

Each fact goes in one section only: the defect and its cost in Issue, what changed in Solution, why it is that way and
who chose it in Decisions, what other projects do in References, what was checked in Evidence (including how clients
react to the new output).

## Decisions

The share of a PR's design a human settled is part of its quality: a human choice was weighed by the person
accountable for the code, an agent's default by nobody. So Decisions lists every choice a reviewer could reasonably
make otherwise, each tagged by who settled it, and a reviewer starts from the `Agent, unreviewed` lines.

- Human: the human chose it or overrode the agent's proposal; name what it replaced.
- Agent, agreed: the human said yes to it. Silence, or merging the PR, is not a yes.
- Agent, unreviewed: everything else, including choices the agent made without raising them.
- A decision, not the conversation: no count of rounds, no slips the human caught (a wrong test, a stale comment).
  Those are the agent's to learn from, in its memory.
- No `Human:` line is a finding in itself: say so rather than leave the section out.

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
- `ChatTemplate.parse`: reasoning left open ends at the first block.

## Decisions
- Human: an unreadable block becomes a call the harness refuses, over leaving it as text that ends the run.
- Agent, agreed: no name rather than a placeholder: a harness could hold a tool by it.
- Agent, unreviewed: reasoning left open ends at the first block, as sglang's reasoning parser takes it.

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
