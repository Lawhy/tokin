# tokin

A gateway that sits between an agent harness and an inference server: the harness
speaks text over `/v1/chat/completions`, the server only ever receives token ids.

## Commands

```bash
uv sync                       # install, including dev tooling
uv run pytest tests -q        # tests
uv run ruff check .           # lint
uv run ruff format .          # format
uv run mypy src               # types
uv run pre-commit run --all-files
```

`pre-commit` runs mypy and the fast tests through the project venv, so `uv sync`
has to have run first.

## Less

损之又损: subtract, then subtract again. Before adding a field, a module, a
helper or a test, ask what can be left out or derived from what already exists;
before shipping, ask what can still come out.

- Build the path that runs today. A second implementation, API or option arrives
  when something needs it, not when it can be foreseen.
- One source per fact. Derive a list from the thing it lists rather than keep a
  copy that can drift.

## No silent fallback

When the code cannot do what it was asked, it raises an error that says why. A
fallback, some value or path the caller did not choose, needs a reason that
holds for every caller, and the code states it where it falls back. Without one
it is a bug that hides itself: a run that quietly did something else is worse
than one that failed, since nobody learns that it happened.

## Style

Formatting is `ruff`'s job. What a linter can't check:

### Files

No license or copyright headers — the licence lives at the repo root.

No module or package docstrings. A file starts at its first import or definition;
`D100` and `D104` are unselected for that reason.

A pluggable concept gets a package: `<concept>s/<concept>.py` holds the base class
and its error type, each implementation gets its own module, and `__init__`
re-exports the base and the registry. `chat_templates/`, `tool_parsers/` and
`inference_engines/` follow this; a module with no implementation family stays at the top
level.

### Docstrings

Document what the signature can't say. Types are in the signature already.

Three tiers, most code being the first:

- **None.** The name and types are enough. `def token_ids(self) -> list[int]`
  needs nothing.
- **One line.** There is a constraint or consequence a caller must know:
  `"""Append tokens the model sampled; must follow at least one prompt turn."""`
- **A short paragraph**, only when a design choice would otherwise look arbitrary
  and someone might reasonably undo it.

No `Args:` / `Returns:` sections to restate types — the signature carries them and
we don't generate API docs from docstrings. Use one only when two or more
parameters each carry a constraint the signature can't say; a single such
parameter goes in the one-line form. Don't restate the function name.

Roughly half the functions in a mature codebase carry no docstring at all, and
that is the target rather than a gap. `D1` stays unselected.

### Comments

A comment does one of two jobs. It states a non-obvious reason at the boundary
that owns it, in one sentence, adding a constraint or an expiry condition only
when a maintainer needs it to judge when the reason stops holding. Or it marks
the structure of a long method, such as the numbered stages of a pipeline, so a
reader sees its shape before its lines.

A comment never narrates: it does not say again what the next line does, tell how
the code came to be or what was tried before it, or list work that may come later.

Behaviour goes in code; durable contracts go in the doc that owns them. If an
explanation runs to a paragraph, it is either describing what the code already
says — delete it — or it is a contract, and belongs here or in the commit
message.

### Private helpers

Extract a helper because the logic has a *name*, not because it repeats.
Repetition is usually better removed with a parameter or a loop; a helper earns
its place by naming a concept.

Three thresholds:

- **Under four lines: inline it.** A helper that short is almost unheard of in
  code that has been maintained; the indirection costs more than it saves.
- **Called once: the name has to say something concrete.** Single-use helpers are
  normal, so call count is not the test — nameability is. If the best name
  available is `_add` or `_process`, there is no concept to extract.
- **Keep private under ~20% of the functions in a module.** Past that a function
  is being sliced up rather than having concepts lifted out of it.

Eight to fifteen lines is the size at which naming something starts to pay.

When a helper does survive, define it before its callers so reading top to bottom
never requires jumping ahead.

### Module-level private functions

**Default is zero per file.** Adding one means arguing why it belongs to neither a
class nor its single caller. A module-level `_helper` has no owner, which is why
these accumulate: anyone can add one and nothing says what it pairs with.

Where they actually belong:

- Used by one class → make it a method on that class.
- Used once inside one function → inline it, or nest it in that function.
- A genuinely general pure function used from several places → it is probably
  public, or it belongs in its own module.

Module-level private *constants* (`_HEADERS = (...)`) are exempt — constants
belong at module scope.

The two numbers move together: files that accumulate module-level privates are
also the ones where private share creeps past half.

### Prose

Single backticks around identifiers and endpoints: `input_ids`, not
``input_ids``. Applies to docstrings, comments, and Markdown.

## Families

A family is one chat template lineage: a `ChatTemplate` subclass under
`src/tokin/chat_templates/` stating its facts as class attributes, with `models` the
checkpoints it was verified against.

A family is named for the generation that introduced its template; later
generations that keep it are appended to `models`, so `qwen35` covers Qwen3.6
and Qwen3.8 the way `llama3` covers 3.3 everywhere. A generation that changes
the template gets a new family, whatever it changed. Names describe the start of
a lineage, not a property of it, so they don't go stale.

A tool-call grammar under `src/tokin/tool_parsers/` is named the same way, for the
model that introduced it, and with sglang's `--tool-call-parser` name where it has
one: `hermes`, `qwen3_coder`, `glm45`. A module's name is its registry name.

`models` lists the mainline of a vendor completely, base and instruct alike;
special-purpose variants (Coder, VL, Math, QwQ, …) are left out until someone
needs them and then get a family of their own.

## Tests

Every test answers one question from a fixed list; a test with no question is
not written:

- **Declarations** — a family's stated facts hold against the real tokenizer.
- **Behaviour** — one contract per method, on a fake tokenizer.
- **Properties** — invariants across methods on real tokenizers. The one that
  matters most: the increment's ids are a suffix of the full render's ids.
- **Refusals** — one input per error contract, asserting the exception and its
  message.

Layout follows what a test needs, and only that decides whether it runs by
default:

- `tests/*.py` needs nothing and runs on every commit. The file name is the
  module under test.
- `tests/tokenizers/` needs real tokenizers (network once, then the cache); its
  `conftest.py` marks everything there `tokenizer`.
- `tests/integration/` needs a running inference server; marked `integration`,
  run by hand.

`addopts` skips the last two. The same module at another layer keeps its file
name and changes directory.

Default to module-level test functions, no class grouping — the name carries the
context.

Group into a class only when one function has three or more scenarios worth
covering. Then the class is named for the function under test
(`TestAddResponse`), and the methods name only the scenario — don't repeat the
function name in both. Grouped method names should read short, three or four
words, because the class already said what is being tested.

No docstrings on tests; the name is the documentation. If a case needs
justification, a one-line comment above the assertion.

## Commits

Conventional Commits, enforced by a `commit-msg` hook: `feat`, `fix`, `docs`,
`style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

`main` takes pull requests only, squash-merged, with `ci-ok` green.

The repository is public. Anything internal to a developer's organization stays
out of commits, pull requests, issues and tracked files, unless its owner has
said it may be shared.
