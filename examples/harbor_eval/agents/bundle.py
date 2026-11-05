from __future__ import annotations

import platform
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from harbor.environments.base import BaseEnvironment
from harbor.environments.e2b import E2BEnvironment

# Bumped when a build changes, so a bundle cached from before it is built again.
REVISION = 2


@dataclass(frozen=True)
class Bundle:
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
            raise RuntimeError(f"the {self.name} bundle runs in linux x86_64 sandboxes, so it is built on such a host")
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
                f"a bundled agent unpacks into a started e2b sandbox, not {type(environment).__name__}'s"
            )
        remote = f"/tmp/{Path(path).name}"
        with open(path, "rb") as data:  # noqa: ASYNC230  # a file e2b reads a chunk at a time as it uploads
            await sandbox.files.write(remote, data, user="root", request_timeout=900)
        unpacked = await environment.exec(
            f"tar -C / -xzf {remote} && rm {remote} && chmod -R a+rX {self.sandbox}", user="root"
        )
        if unpacked.return_code:
            raise RuntimeError(f"the {self.name} bundle did not unpack: {unpacked.stdout}{unpacked.stderr}")
