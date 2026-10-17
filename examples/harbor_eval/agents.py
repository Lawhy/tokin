from __future__ import annotations

import io
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from harbor.agents.installed.mini_swe_agent import MiniSweAgent, MiniSweAgentOptions
from harbor.agents.installed.pi import Pi, PiOptions
from harbor.environments.base import BaseEnvironment
from harbor.environments.e2b import E2BEnvironment
from uv import find_uv_bin

MINI_SWE_AGENT = "2.4.6"
PI = "1.3.0"
NODE = "v22.23.3"
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


BUILDS: dict[str, Callable[[Path], None]] = {"mini-swe-agent": build_mini_swe_agent, "pi": build_pi}
# Where each bundle unpacks in the sandbox: what runs it names this path.
SANDBOX = {"mini-swe-agent": "/opt/tokin-mini-swe-agent", "pi": "/opt/tokin-pi"}
VERSIONS = {"mini-swe-agent": MINI_SWE_AGENT, "pi": PI}


def bundle(agent: str, cache: Path) -> Path:
    """The agent's bundle, a tarball that unpacks to its sandbox path, built into `cache` on first use."""
    path = cache / f"{agent}-{VERSIONS[agent]}.tar.gz"
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
