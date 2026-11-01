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
A call the tool parser cannot read reaches the harness with its block as `arguments`, which is not JSON. A harness
that echoes it back has its next turn refused by `ChatTemplate.conform` and answered 400: 24 of ~87k OpenAI Agents
SDK runs in a 3 h soak.

## Solution
- `ToolParser.parse`: an unreadable block's `arguments` is the whole block, tags included.
- `ChatTemplate.conform`: a call whose `arguments` is exactly one such block is appended to the reply's content.
- `ChatTemplate.tool_call_parser`: a family names its grammar by its `get_tool_parser` name, which returns an instance.

## Decisions
- Human: an unreadable call renders back as the text the model wrote, over refusing the turn with 400.
- Human: `tool_call_parser` is a required string, over an optional instance, so a family's facts are all primitives.
- Human: `get_tool_parser` returns an instance, over a class: a parser takes no arguments.
- Human: the check sits inline in `conform`, over a helper.
- Agent, agreed: `arguments` holds the whole block, not its body: a cut block has no closing tag to restore.
- Agent, agreed: a restored block renders before any readable call beside it: content renders before `tool_calls`.
- Agent, unreviewed: other non-JSON `arguments` are still refused: only a parser block has a known text.
- Agent, unreviewed: the attribute is `tool_call_parser`, sglang's flag name, leaving `tool_parser` to the instance.
- Agent, unreviewed: the name resolves in the constructor, so an unknown one fails when the template is built.

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

## References
- miles `normalize_tool_arguments`: `json.loads` every `arguments`, raises otherwise.
```
