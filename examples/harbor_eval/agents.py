from __future__ import annotations

import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from harbor.agents.installed.mini_swe_agent import MiniSweAgent, MiniSweAgentOptions
from harbor.agents.installed.opencode import OpenCode, OpenCodeOptions
from harbor.agents.installed.pi import Pi, PiOptions
from harbor.environments.base import BaseEnvironment
from harbor.environments.e2b import E2BEnvironment
from harbor.models.agent.context import AgentContext
from uv import find_uv_bin

MINI_SWE_AGENT = "2.4.6"
PI = "1.3.0"
NODE = "v22.23.3"
OPENCODE = "1.19.3"
# Bumped when a build changes, so a bundle cached from before it is built again.
REVISION = 2
# mini-swe-agent's command timeout, made to hold: since 2.4.1 its `_run` drains a killed command's pipe with no
# timeout, so a process that left the group still holding stdout, as `qemu -daemonize` does, hangs the agent for
# good. Waiting on the shell instead, as `subprocess.run` and 2.4.0 do, keeps the output from before the timeout.
TIMEOUT_DRAIN = """    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)"""
TIMEOUT_WAIT = """    except subprocess.TimeoutExpired as expired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        process.wait()
        raise subprocess.TimeoutExpired(command, timeout, output=expired.output)"""
# pi's bash tool with a default timeout: its call, minified and not, and the schema the model reads.
PI_TIMEOUT = [
    ("{command,timeout},signal", "{command,timeout=1200},signal"),
    ("{ command, timeout }, signal", "{ command, timeout = 1200 }, signal"),
    ("optional, no default timeout", "optional, default 1200"),
]
OPENCODE_PROVIDER = "@ai-sdk/openai-compatible"


def build_mini_swe_agent(root: Path) -> None:
    """A standalone CPython with mini-swe-agent installed into it, and `bin/mini-swe-agent` to start it.

    No venv and no `PYTHONPATH`: the interpreter finds its packages from where it is unpacked, and the commands
    the agent runs inherit no path of ours.
    """
    uv = find_uv_bin()
    subprocess.run([uv, "python", "install", "3.12", "--install-dir", str(root / "pythons")], check=True)
    # uv links `cpython-3.12-<platform>` to the patch release it installed; the release is what moves.
    release = next(p for p in (root / "pythons").glob("cpython-3.12*") if p.is_dir() and not p.is_symlink())
    release.rename(root / "python")
    shutil.rmtree(root / "pythons")
    python = root / "python" / "bin" / "python3"
    # Wheels for glibc 2.28, which Debian 10 and Ubuntu 20.04 on carry, so one bundle runs in any of their images.
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--break-system-packages",
            "--python-platform",
            "x86_64-manylinux_2_28",
            f"mini-swe-agent=={MINI_SWE_AGENT}",
        ],
        check=True,
    )
    local = next((root / "python" / "lib").glob("python3.12/site-packages/minisweagent/environments/local.py"))
    text = local.read_text()
    if TIMEOUT_DRAIN not in text:
        raise RuntimeError(f"mini-swe-agent {MINI_SWE_AGENT}'s `_run` is not the one the timeout patch replaces")
    local.write_text(text.replace(TIMEOUT_DRAIN, TIMEOUT_WAIT))
    (root / "bin").mkdir()
    start = root / "bin" / "mini-swe-agent"
    start.write_text(f'#!/bin/sh\nexec {SANDBOX["mini-swe-agent"]}/python/bin/python3 -m minisweagent.run.mini "$@"\n')
    start.chmod(0o755)


def build_pi(root: Path) -> None:
    """Node's official build and pi from npm, its `bin/` linked relative to where it is unpacked."""
    url = f"https://nodejs.org/dist/{NODE}/node-{NODE}-linux-x64.tar.xz"
    with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:xz") as node:
        node.extractall(root, filter="tar")
    (root / f"node-{NODE}-linux-x64").rename(root / "node")
    npm = root / "node" / "bin" / "npm"
    env = {"PATH": f"{root / 'node' / 'bin'}:/usr/bin:/bin"}
    subprocess.run(
        [str(npm), "install", "-g", "--ignore-scripts", "--prefix", str(root), f"@earendil-works/pi-coding-agent@{PI}"],
        check=True,
        env=env,
    )
    # pi's bash tool waits on a command with no timeout for as long as it runs, so a server the model starts in the
    # foreground holds the trial until the agent's timeout; such a command gets the 1200 s mini-swe-agent's does.
    package = root / "lib" / "node_modules" / "@earendil-works" / "pi-coding-agent"
    for script in package.glob("dist/**/*.js"):
        text = script.read_text()
        for unset, default in PI_TIMEOUT:
            text = text.replace(unset, default)
        script.write_text(text)
    patched = sum(PI_TIMEOUT[0][1] in script.read_text() for script in package.glob("dist/bundle/**/*.js"))
    if patched != 1:
        raise RuntimeError(f"pi {PI}'s bash tool is not the one the timeout patch fits: patched in {patched} files")


def build_opencode(root: Path) -> None:
    """opencode's native builds for glibc and musl, `bin/opencode` to pick one, and the cache its first run fills.

    The openai-compatible provider tokin is served through ships in the binary, and the models.dev catalogue it
    fetches on first run ships in the cache, so nothing is fetched in a sandbox that blocks the network.
    """
    for libc, package in [("glibc", "opencode-linux-x64"), ("musl", "opencode-linux-x64-musl")]:
        url = f"https://registry.npmjs.org/{package}/-/{package}-{OPENCODE}.tgz"
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:gz") as npm:
            binary = npm.extractfile("package/bin/opencode")
            assert binary is not None
            (root / libc).mkdir()
            (root / libc / "opencode").write_bytes(binary.read())
        (root / libc / "opencode").chmod(0o755)
    sandbox = SANDBOX["opencode"]
    (root / "bin").mkdir()
    start = root / "bin" / "opencode"
    start.write_text(
        "#!/bin/sh\n"
        "libc=glibc; if [ -f /etc/alpine-release ] || ldd --version 2>&1 | grep -qi musl; then libc=musl; fi\n"
        f'XDG_CACHE_HOME={sandbox}/cache exec {sandbox}/$libc/opencode "$@"\n'
    )
    start.chmod(0o755)
    # A run against nothing fills the cache; getting as far as the connection shows the provider needs no fetch.
    provider = {"npm": OPENCODE_PROVIDER, "options": {"baseURL": "http://127.0.0.1:9/v1"}, "models": {"policy": {}}}
    config = {"provider": {"tokin": provider}}
    with tempfile.TemporaryDirectory() as home:
        tried = subprocess.run(
            [str(root / "glibc" / "opencode"), "--model=tokin/policy", "run", "hi"],
            env=os.environ
            | {"HOME": home, "XDG_CACHE_HOME": str(root / "cache"), "OPENCODE_CONFIG_CONTENT": json.dumps(config)},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
    if "Cannot connect to API" not in tried.stdout + tried.stderr:
        raise RuntimeError(f"opencode {OPENCODE} never reached the endpoint: {tried.stdout}{tried.stderr}")


BUILDS: dict[str, Callable[[Path], None]] = {
    "mini-swe-agent": build_mini_swe_agent,
    "pi": build_pi,
    "opencode": build_opencode,
}
# Where each bundle unpacks in the sandbox: what runs it names this path.
SANDBOX = {"mini-swe-agent": "/opt/tokin-mini-swe-agent", "pi": "/opt/tokin-pi", "opencode": "/opt/tokin-opencode"}
VERSIONS = {"mini-swe-agent": MINI_SWE_AGENT, "pi": PI, "opencode": OPENCODE}


def bundle(agent: str, cache: Path) -> Path:
    """The agent's bundle, a tarball that unpacks to its sandbox path, built into `cache` on first use."""
    path = cache / f"{agent}-{VERSIONS[agent]}-r{REVISION}.tar.gz"
    if path.exists():
        return path
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError(f"the {agent} bundle runs in linux x86_64 sandboxes, so it is built on such a host")
    with tempfile.TemporaryDirectory(dir=cache) as work:
        root = Path(work) / "root"
        root.mkdir()
        BUILDS[agent](root)
        part = path.with_suffix(".part")
        with tarfile.open(part, "w:gz") as archive:
            archive.add(root, arcname=SANDBOX[agent].lstrip("/"))
        part.rename(path)
    return path


async def carry(environment: BaseEnvironment, path: str, agent: str) -> None:
    """Unpack the bundle at `path` into the e2b sandbox of `environment`, which must be an e2b one.

    Streamed, with room to finish: Harbor's `upload_file` reads a file whole on the event loop and keeps e2b's
    default request timeout, which a wide launch of these uploads outlasts.
    """
    sandbox = environment._sandbox if isinstance(environment, E2BEnvironment) else None
    if sandbox is None:
        raise RuntimeError(f"a bundled agent unpacks into a started e2b sandbox, not {type(environment).__name__}'s")
    remote = f"/tmp/{Path(path).name}"
    with open(path, "rb") as data:  # noqa: ASYNC230  # a file e2b reads a chunk at a time as it uploads
        await sandbox.files.write(remote, data, user="root", request_timeout=900)
    unpacked = await environment.exec(
        f"tar -C / -xzf {remote} && rm {remote} && chmod -R a+rX {SANDBOX[agent]}", user="root"
    )
    if unpacked.return_code:
        raise RuntimeError(f"the {agent} bundle did not unpack: {unpacked.stdout}{unpacked.stderr}")


class BundledMiniSweAgentOptions(MiniSweAgentOptions):
    bundle: str


class BundledMiniSweAgent(MiniSweAgent):
    """Harbor's mini-swe-agent, unpacked from a bundle rather than installed in each sandbox.

    Installing it takes apt, uv, a CPython where the image's is older than 3.10, and litellm's dependencies: each a
    download that a wide launch, an image past its distribution's support, or a task that blocks package indexes can
    fail. The bundle is one upload, its command timeout already patched to hold.
    """

    options_model: ClassVar[type[MiniSweAgentOptions]] = BundledMiniSweAgentOptions
    options: BundledMiniSweAgentOptions

    async def install(self, environment: BaseEnvironment) -> None:
        await carry(environment, self.options.bundle, "mini-swe-agent")
        # Where Harbor's run command looks for the CLI.
        root = SANDBOX["mini-swe-agent"]
        await self.exec_as_agent(
            environment, command=f"mkdir -p $HOME/.local/bin && ln -sf {root}/bin/* $HOME/.local/bin/"
        )

    def get_version_command(self) -> str | None:
        return (
            f"{SANDBOX['mini-swe-agent']}/python/bin/python3 -c 'import minisweagent; print(minisweagent.__version__)'"
        )


class BundledPiOptions(PiOptions):
    bundle: str


class BundledPi(Pi):
    """Harbor's pi, unpacked from a bundle of Node and pi rather than installed with nvm and npm in each sandbox."""

    options_model: ClassVar[type[PiOptions]] = BundledPiOptions
    options: BundledPiOptions

    async def install(self, environment: BaseEnvironment) -> None:
        await carry(environment, self.options.bundle, "pi")
        # Harbor's run command sources nvm before it calls pi; this one puts the bundle on the path instead.
        root = SANDBOX["pi"]
        path = f"export PATH={root}/bin:{root}/node/bin:\\$PATH"
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


class BundledOpenCodeOptions(OpenCodeOptions):
    bundle: str


class BundledOpenCode(OpenCode):
    """Harbor's opencode, unpacked from a bundle of its native builds rather than installed with nvm and npm."""

    options_model: ClassVar[type[OpenCodeOptions]] = BundledOpenCodeOptions
    options: BundledOpenCodeOptions

    async def install(self, environment: BaseEnvironment) -> None:
        await carry(environment, self.options.bundle, "opencode")
        # Where Harbor's run command finds it, with a cache opencode may rewrite as whichever user it runs as.
        root = SANDBOX["opencode"]
        await self.exec_as_root(
            environment,
            command=f"chmod -R a+rwX {root}/cache && mkdir -p /usr/local/bin && ln -sf {root}/bin/opencode /usr/local/bin/",
        )
