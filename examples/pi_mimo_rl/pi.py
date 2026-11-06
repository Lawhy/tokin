from __future__ import annotations

import io
import json
import platform
import shlex
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

from harbor.agents.installed.pi import _REMOTE_PI_CONFIG_DIR, Pi, PiOptions
from harbor.agents.model_connection import ResolvedModelConnection
from harbor.environments.base import BaseEnvironment
from harbor.environments.e2b import E2BEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import model_validator

VERSION = "1.3.0"
NODE = "v22.23.3"
RIPGREP = "15.2.0"
FD = "10.5.0"
SANDBOX = "/opt/tokin-pi"
# Bumped when the build changes, so an archive cached from before it is built again.
REVISION = 1
# pi's bash tool with a default timeout: its call, minified and not, and the schema the model reads.
TIMEOUT = [
    ("{command,timeout},signal", "{command,timeout=1200},signal"),
    ("{ command, timeout }, signal", "{ command, timeout = 1200 }, signal"),
    ("optional, no default timeout", "optional, default 1200"),
]
# The example's six tools: claude-code's Bash, Read, Edit, Write, Glob and Grep, one for one.
TOOLS = "read,bash,edit,write,grep,find"
# slime's `SWE_CC_PROMPT`, with the statement inline rather than in `PROBLEM_STATEMENT.md`, which the verifier would
# also diff, and without the `.harness` directory pi does not have.
PROMPT = (
    "Resolve the issue below. Edit source files only; do not modify tests. Implement the fix, run the relevant "
    "tests, then print a one-line summary and exit. Do not commit."
)
# How every MiMo code task's instruction opens.
LEAD = "Fix the following issue:"
# Printed when the extension stops pi; free of "tokin", the session's API key, which Harbor redacts from its logs.
CAP_MARK = "compaction-cap-reached"


@dataclass(frozen=True)
class Prebuilt:
    """An agent and everything it runs on, one tarball that unpacks to `sandbox`, the path what runs it names."""

    name: str
    version: str
    sandbox: str
    build: Callable[[Path], None]

    def path(self, cache: Path) -> Path:
        """The tarball, built into `cache` on first use."""
        path = cache / f"{self.name}-{self.version}-r{REVISION}.tar.gz"
        if path.exists():
            return path
        if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
            raise RuntimeError(f"the {self.name} archive runs in linux x86_64 sandboxes, so it is built on such a host")
        with tempfile.TemporaryDirectory(dir=cache) as work:
            root = Path(work) / "root"
            root.mkdir()
            self.build(root)
            part = path.with_suffix(".part")
            with tarfile.open(part, "w:gz") as archive:
                archive.add(root, arcname=self.sandbox.lstrip("/"))
            part.rename(path)
        return path

    async def carry(self, environment: BaseEnvironment, path: str) -> None:
        """Unpack the tarball at `path` into the e2b sandbox of `environment`, which must be an e2b one.

        Streamed, with room to finish: Harbor's `upload_file` reads a file whole on the event loop and keeps e2b's
        default request timeout, which a wide launch of these uploads outlasts.
        """
        sandbox = environment._sandbox if isinstance(environment, E2BEnvironment) else None
        if sandbox is None:
            raise RuntimeError(
                f"a prebuilt agent unpacks into a started e2b sandbox, not {type(environment).__name__}'s"
            )
        remote = f"/tmp/{Path(path).name}"
        with open(path, "rb") as data:  # noqa: ASYNC230  # a file e2b reads a chunk at a time as it uploads
            await sandbox.files.write(remote, data, user="root", request_timeout=900)
        unpacked = await environment.exec(
            f"tar -C / -xzf {remote} && rm {remote} && chmod -R a+rX {self.sandbox}", user="root"
        )
        if unpacked.return_code:
            raise RuntimeError(f"the {self.name} archive did not unpack: {unpacked.stdout}{unpacked.stderr}")


def build(root: Path) -> None:
    """Node's official build, pi from npm, and the ripgrep and fd its `grep` and `find` run, in `bin/`.

    pi downloads ripgrep and fd on first use, which a sandbox that blocks the network fails.
    """
    url = f"https://nodejs.org/dist/{NODE}/node-{NODE}-linux-x64.tar.xz"
    with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())) as node:
        node.extractall(root, filter="tar")
    (root / f"node-{NODE}-linux-x64").rename(root / "node")
    package, npm = "@earendil-works/pi-coding-agent", root / "node" / "bin" / "npm"
    install = [str(npm), "install", "-g", "--ignore-scripts", "--prefix", str(root), f"{package}@{VERSION}"]
    subprocess.run(install, check=True, env={"PATH": f"{npm.parent}:/usr/bin:/bin"})
    # pi's bash tool waits on a command with no timeout for as long as it runs, so a server the model starts in the
    # foreground holds the trial until the agent's timeout; such a command gets 1200 s.
    dist = root / "lib" / "node_modules" / package / "dist"
    for script in dist.glob("**/*.js"):
        text = script.read_text()
        for unset, default in TIMEOUT:
            text = text.replace(unset, default)
        script.write_text(text)
    if (patched := sum(TIMEOUT[0][1] in script.read_text() for script in dist.glob("bundle/**/*.js"))) != 1:
        raise RuntimeError(
            f"pi {VERSION}'s bash tool is not the one the timeout patch fits: patched in {patched} files"
        )
    releases = {
        "rg": f"https://github.com/BurntSushi/ripgrep/releases/download/{RIPGREP}/ripgrep-{RIPGREP}-x86_64-unknown-linux-musl.tar.gz",
        "fd": f"https://github.com/sharkdp/fd/releases/download/v{FD}/fd-v{FD}-x86_64-unknown-linux-musl.tar.gz",
    }
    for name, url in releases.items():
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())) as release:
            binary = release.extractfile(f"{Path(url).name.removesuffix('.tar.gz')}/{name}")
            assert binary is not None
            (root / "bin" / name).write_bytes(binary.read())
        (root / "bin" / name).chmod(0o755)


PREBUILT = Prebuilt("pi-rg-fd", VERSION, SANDBOX, build)


def extension(max_compactions: int) -> str:
    """A pi extension that lets `max_compactions` compactions through and stops pi at the next attempt."""
    return f"""export default function (pi: any) {{
  pi.on("session_before_compact", (event: any, ctx: any) => {{
    const done = event.branchEntries.filter((entry: any) => entry.type === "compaction").length;
    if (done < {max_compactions}) return;
    console.error("{CAP_MARK}: " + done + " compactions done");
    ctx.abort();
    ctx.shutdown();
    return {{ cancel: true }};
  }});
}}
"""


class PrebuiltPiOptions(PiOptions):
    archive: str
    context_window: int
    max_tokens: int
    compact_pct: int = 70
    max_compactions: int = 2

    @property
    def reserve_tokens(self) -> int:
        return self.context_window * (100 - self.compact_pct) // 100

    @model_validator(mode="after")
    def reserve_holds_a_turn(self) -> Self:
        """A reply fits in what compaction leaves, so pi compacts before a reply can reach the context's end."""
        if self.reserve_tokens < self.max_tokens:
            raise ValueError(
                f"pi's compaction reserve {self.reserve_tokens} is below a turn's {self.max_tokens} tokens"
            )
        return self


class PrebuiltPi(Pi):
    """Harbor's pi from its prebuilt archive, sized to its tokin session, with the example's tools, prompt, compaction.

    `max_tokens` is tokin's per-turn cap and `context_window` its context, so a reply cut at the cap is not one pi
    recovers from by compacting. pi compacts at `compact_pct` of the context, at most `max_compactions` times, and the
    extension stops it at the next attempt, printing `CAP_MARK`.
    """

    options_model: ClassVar[type[PiOptions]] = PrebuiltPiOptions
    options: PrebuiltPiOptions

    async def install(self, environment: BaseEnvironment) -> None:
        await PREBUILT.carry(environment, self.options.archive)
        # Harbor's run command sources nvm before it calls pi; this one puts the archive on the path instead.
        path = f"export PATH={SANDBOX}/bin:{SANDBOX}/node/bin:\\$PATH"
        await self.exec_as_agent(environment, command=f'mkdir -p $HOME/.nvm && echo "{path}" > $HOME/.nvm/nvm.sh')

    def _build_custom_models_json(self, access: ResolvedModelConnection, model_id: str) -> dict[str, Any] | None:
        models = super()._build_custom_models_json(access, model_id)
        if models is None:
            raise ValueError("pi is sized to a tokin session, so it needs the session's URL as its endpoint")
        for provider in models["providers"].values():
            for model in provider["models"]:
                model["maxTokens"] = self.options.max_tokens
                model["contextWindow"] = self.options.context_window
        return models

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not instruction.startswith(LEAD):
            raise ValueError(f"the task's instruction does not open with {LEAD!r}, so the prompt has no place")
        instruction = f"{PROMPT}\n\n{instruction.removeprefix(LEAD).lstrip()}"
        await self._ensure_config_dir(environment)
        compaction = {"enabled": True, "reserveTokens": self.options.reserve_tokens, "keepRecentTokens": 20000}
        configs = {
            "settings.json": json.dumps({"compaction": compaction}) + "\n",
            "compaction-cap.ts": extension(self.options.max_compactions),
        }
        for filename, content in configs.items():
            await self._upload_config_text(
                environment, content=content, remote_path=f"{_REMOTE_PI_CONFIG_DIR}/{filename}", filename=filename
            )
        # The instruction as Harbor's `run` renders it, which is what its command quotes.
        self.instruction = self.render_instruction(instruction)
        await super().run(instruction, environment, context)

    async def exec_as_agent(self, environment: BaseEnvironment, command: str, *args: Any, **kwargs: Any) -> Any:
        # pi takes a message that opens with `-` for an option and exits before its first turn; past `--` every
        # argument is a message.
        if " pi --print " in command:
            quoted = f" {shlex.quote(self.instruction)} 2>&1 </dev/null"
            if command.count(quoted) != 1:
                raise RuntimeError("pi's run command does not end in the instruction, so its flags have no place")
            flags = f" --tools {TOOLS} --extension {_REMOTE_PI_CONFIG_DIR}/compaction-cap.ts --"
            command = command.replace(quoted, f"{flags}{quoted}")
        return await super().exec_as_agent(environment, command, *args, **kwargs)


if __name__ == "__main__":
    cache = Path(sys.argv[1])
    cache.mkdir(parents=True, exist_ok=True)
    print(PREBUILT.path(cache))
