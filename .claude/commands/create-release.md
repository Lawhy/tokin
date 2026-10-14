Create a new release: generate release notes, create a GitHub release (which triggers the PyPI publish), and verify the pipeline.

The user optionally provides a version as $ARGUMENTS (e.g. `0.4.0`). If not provided, determine the next version from the latest git tag and the nature of the changes (patch for fixes and chores, minor for features, major for breaking changes).

## 0. Preconditions

- Release from an up-to-date `main`: `git checkout main && git pull --ff-only`. `gh release create` tags the tip of `main` on GitHub.
- `ci-ok` must be green on that commit: `gh run list --branch main --limit 1`.
- First release only: the pending trusted publisher must already exist on PyPI (project `tokin`, owner `Lawhy`, repository `tokin`, workflow `publish.yml`, environment `pypi`). Without it the publish job fails with `invalid-publisher`. Only the project owner can register it.

## 1. Determine version

- Find the latest git tag: `git tag --sort=-v:refname | head -1`
- No tag yet means this is the first release: use `0.0.1` unless the user says otherwise, and list every commit.
- If the user didn't provide a version, suggest one based on the commits since the last tag.
- Confirm the version with the user before proceeding.

## 2. Generate release notes

- List commits since the last tag: `git log --oneline <last_tag>..HEAD`
- `main` is squash-merged, so each subject ends in `(#N)`; link it as `[#N](https://github.com/Lawhy/tokin/pull/N)`.
- Categorize using the conventional commit prefixes:
  - `feat` → **Features**
  - `fix` → **Bug Fixes**
  - `refactor`, `perf` → **Refactoring**
  - `docs` → **Documentation**
  - `test` → **Tests**
  - `build`, `ci`, `chore`, `style` → **Maintenance** (one section)
  - `revert` → note under the section of what it reverted
  - Breaking changes (a `!` after the type, e.g. `feat!:`) → **Breaking Changes** at the top
- Skip merge commits and commits that only touch `.claude/`.
- Write descriptions from the **user's perspective**, not the commit message verbatim.
- Omit empty sections. Group related commits into a single bullet.
- Keep it concise: release notes are for users, not a git log dump.
- Minor and major releases open with a **Highlights** paragraph, two or three sentences on what the release means, before the sections. Patch releases don't.
- Notes cover the changes since the previous tag, whatever its size; a minor does not repeat its patches. The one exception: when the previous tag was a name-claim (`0.0.x`), nobody read its notes, so write this release's notes cumulatively from the first commit.

## 3. Show the user the release notes for review

Present the full release notes in the body format below and ask the user to confirm before creating the release.

**Body format**:
```markdown
## What's New

Highlights paragraph, minor and major releases only.

### Features
- Brief description of the feature ([#N](https://github.com/Lawhy/tokin/pull/N))

### Bug Fixes
- Brief description of the fix

**Full Changelog**: https://github.com/Lawhy/tokin/compare/<last_tag>...v<new_version>
```

For the first release the changelog line is `https://github.com/Lawhy/tokin/commits/v<new_version>`.

## 4. Create the GitHub release

Once the user confirms, create the release with `gh release create`. This creates the git tag.

```bash
gh release create v<version> --title "v<version>" --notes "<body>"
```

Use a HEREDOC for the body to preserve formatting.

## 5. Verify the publish pipeline

The release triggers `.github/workflows/publish.yml`, which:
1. Builds the sdist and wheel (`uv build`; hatch-vcs derives the version from the tag)
2. Checks the metadata renders (`twine check --strict`)
3. Publishes to PyPI through the `pypi` environment with trusted publishing (OIDC) and PEP 740 attestations

After creating the release:
- Run `gh run list --workflow publish.yml --limit 3` to confirm the "Publish" workflow was triggered.
- Watch the run: `gh run watch <run_id>` (timeout 2 minutes).
- Report the result to the user; if it failed, show the logs with `gh run view <run_id> --log-failed`.
- Confirm the version is live: `curl -s https://pypi.org/pypi/tokin/<version>/json | head -c 200`.

A dry run against TestPyPI is `gh workflow run publish.yml -f target=testpypi`; it needs its own pending publisher on test.pypi.org.

## Important

- Always confirm the version and the release notes with the user before creating the release.
- The version is derived from the git tag by hatch-vcs; there is no version field in `pyproject.toml` to update. Do not modify `pyproject.toml`.
- Get the number right the first time: the `tags-are-immutable` ruleset blocks moving or deleting a tag, and PyPI refuses a second upload of the same filename.
