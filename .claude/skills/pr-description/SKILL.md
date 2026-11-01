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

## Behaviour
<Only when what a harness, trainer or user observes changes: before/after, as a table when there are cases.
Say "Breaking:" first when an interface they call changes.>

| Case | Before | After |
|---|---|---|

## Evidence
- Tests: <added or changed tests, by name or by what they cover; which fail on `main`>.
- Suites: <default / real-tokenizer / integration, pass>.
- Measured: <numbers on real data, with sample size: before vs after, or the change's footprint>.

## Decisions
- @<author>: <choice>, over <the alternative>; <choice>, over <the alternative>.
- Claude, agreed: <choice>: <half a line of reason>; <choice>: <reason>.
- Claude, unreviewed: <choice>: <reason>; <choice>: <reason>.

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
make otherwise, one line per party, and a reviewer starts from the `unreviewed` line.

- `@<author>`, the human who opens the PR: what they chose or overrode, each with what it replaced.
- `Claude, agreed`: what the agent proposed and the author said yes to. Silence, or merging the PR, is not a yes.
- `Claude, unreviewed`: everything else, including choices the agent made without raising them.
- The agent goes by its name without `@`: `@agent`, `@claude` are GitHub users.
- A decision, not the conversation: no count of rounds, no slips the human caught (a wrong test, a stale comment).
  Those are the agent's to learn from, in its memory.
- No `@<author>` line is a finding in itself: say so rather than leave the line out.

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
A call the tool parser cannot read reaches the harness with its block as `arguments`, which is not JSON. A harness
that echoes it back has its next turn refused by `ChatTemplate.conform` and answered 400: 24 of ~87k OpenAI Agents
SDK runs in a 3 h soak.

## Solution
- `ToolParser.parse`: an unreadable block's `arguments` is the whole block, tags included.
- `ChatTemplate.conform`: a call whose `arguments` is exactly one such block is appended to the reply's content.
- `ChatTemplate.tool_call_parser`: a family names its grammar by its `get_tool_parser` name, which returns an instance.

## Behaviour
Breaking: a family sets `tool_call_parser = "<name>"` instead of `tool_parser = <instance>`.

| Case | Before | After |
|---|---|---|
| unreadable call, `arguments` | block body | whole block |
| that call echoed back | 400 | rendered as the model wrote it |

## Evidence
- Tests: an unread call renders as its block and keeps its result; a turn cut inside a call renders in the next
  prompt (both fail on `main`).
- Suites: default and real-tokenizer pass.

## Decisions
- @Lawhy: an unreadable call renders back as the text the model wrote, over a 400; `tool_call_parser` a required
  string, over an optional instance; `get_tool_parser` returns an instance, over a class; the check inline in
  `conform`, over a helper.
- Claude, agreed: `arguments` holds the whole block: a cut one has no closing tag to restore; a restored block renders
  before any readable call beside it: content renders before `tool_calls`.
- Claude, unreviewed: other non-JSON `arguments` are still refused: only a parser block has a known text; the name
  `tool_call_parser`, sglang's flag; it resolves in the constructor, so an unknown name fails at build.

## References
- miles `normalize_tool_arguments`: `json.loads` every `arguments`, raises otherwise.
```
