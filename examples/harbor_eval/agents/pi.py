from __future__ import annotations

import io
import shlex
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

from harbor.agents.installed.pi import Pi, PiOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .bundle import Bundle

VERSION = "1.3.0"
NODE = "v22.23.3"
SANDBOX = "/opt/tokin-pi"
# pi's bash tool with a default timeout: its call, minified and not, and the schema the model reads.
TIMEOUT = [
    ("{command,timeout},signal", "{command,timeout=1200},signal"),
    ("{ command, timeout }, signal", "{ command, timeout = 1200 }, signal"),
    ("optional, no default timeout", "optional, default 1200"),
]


def build(root: Path) -> None:
    """Node's official build and pi from npm, its `bin/` linked relative to where it is unpacked."""
    url = f"https://nodejs.org/dist/{NODE}/node-{NODE}-linux-x64.tar.xz"
    with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:xz") as node:
        node.extractall(root, filter="tar")
    (root / f"node-{NODE}-linux-x64").rename(root / "node")
    npm = root / "node" / "bin" / "npm"
    env = {"PATH": f"{root / 'node' / 'bin'}:/usr/bin:/bin"}
    subprocess.run(
        [
            str(npm),
            "install",
            "-g",
            "--ignore-scripts",
            "--prefix",
            str(root),
            f"@earendil-works/pi-coding-agent@{VERSION}",
        ],
        check=True,
        env=env,
    )
    # pi's bash tool waits on a command with no timeout for as long as it runs, so a server the model starts in the
    # foreground holds the trial until the agent's timeout; such a command gets the 1200 s mini-swe-agent's does.
    package = root / "lib" / "node_modules" / "@earendil-works" / "pi-coding-agent"
    for script in package.glob("dist/**/*.js"):
        text = script.read_text()
        for unset, default in TIMEOUT:
            text = text.replace(unset, default)
        script.write_text(text)
    patched = sum(TIMEOUT[0][1] in script.read_text() for script in package.glob("dist/bundle/**/*.js"))
    if patched != 1:
        raise RuntimeError(
            f"pi {VERSION}'s bash tool is not the one the timeout patch fits: patched in {patched} files"
        )


BUNDLE = Bundle("pi", VERSION, SANDBOX, build)


class BundledPiOptions(PiOptions):
    bundle: str


class BundledPi(Pi):
    """Harbor's pi, unpacked from a bundle of Node and pi rather than installed with nvm and npm in each sandbox."""

    options_model: ClassVar[type[PiOptions]] = BundledPiOptions
    options: BundledPiOptions

    async def install(self, environment: BaseEnvironment) -> None:
        await BUNDLE.carry(environment, self.options.bundle)
        # Harbor's run command sources nvm before it calls pi; this one puts the bundle on the path instead.
        path = f"export PATH={SANDBOX}/bin:{SANDBOX}/node/bin:\\$PATH"
        await self.exec_as_agent(environment, command=f'mkdir -p $HOME/.nvm && echo "{path}" > $HOME/.nvm/nvm.sh')

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        # The instruction as Harbor's `run` renders it, which is what its command quotes.
        self.instruction = self.render_instruction(instruction)
        await super().run(instruction, environment, context)

    async def exec_as_agent(self, environment: BaseEnvironment, command: str, *args: Any, **kwargs: Any) -> Any:
        # pi takes a message that opens with `-` for an option and exits before its first turn; past `--` every
        # argument is a message.
        if " pi --print " in command:
            quoted = f" {shlex.quote(self.instruction)} 2>&1 </dev/null"
            if command.count(quoted) != 1:
                raise RuntimeError("pi's run command does not end in the instruction, so `--` has no place before it")
            command = command.replace(quoted, f" --{quoted}")
        return await super().exec_as_agent(environment, command, *args, **kwargs)
