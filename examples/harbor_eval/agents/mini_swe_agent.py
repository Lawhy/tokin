from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

from harbor.agents.installed.mini_swe_agent import MiniSweAgent, MiniSweAgentOptions
from harbor.environments.base import BaseEnvironment
from uv import find_uv_bin

from .bundle import Bundle

VERSION = "2.4.6"
SANDBOX = "/opt/tokin-mini-swe-agent"
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


def build(root: Path) -> None:
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
            f"mini-swe-agent=={VERSION}",
        ],
        check=True,
    )
    local = next((root / "python" / "lib").glob("python3.12/site-packages/minisweagent/environments/local.py"))
    text = local.read_text()
    if TIMEOUT_DRAIN not in text:
        raise RuntimeError(f"mini-swe-agent {VERSION}'s `_run` is not the one the timeout patch replaces")
    local.write_text(text.replace(TIMEOUT_DRAIN, TIMEOUT_WAIT))
    (root / "bin").mkdir()
    start = root / "bin" / "mini-swe-agent"
    start.write_text(f'#!/bin/sh\nexec {SANDBOX}/python/bin/python3 -m minisweagent.run.mini "$@"\n')
    start.chmod(0o755)


BUNDLE = Bundle("mini-swe-agent", VERSION, SANDBOX, build)


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
        await BUNDLE.carry(environment, self.options.bundle)
        # Where Harbor's run command looks for the CLI.
        await self.exec_as_agent(
            environment, command=f"mkdir -p $HOME/.local/bin && ln -sf {SANDBOX}/bin/* $HOME/.local/bin/"
        )

    def get_version_command(self) -> str | None:
        return f"{SANDBOX}/python/bin/python3 -c 'import minisweagent; print(minisweagent.__version__)'"
