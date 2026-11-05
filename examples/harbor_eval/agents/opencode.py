from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import ClassVar

from harbor.agents.installed.opencode import OpenCode, OpenCodeOptions
from harbor.environments.base import BaseEnvironment

from .prebuilt import Prebuilt

VERSION = "1.19.3"
SANDBOX = "/opt/tokin-opencode"
PROVIDER = "@ai-sdk/openai-compatible"


def build(root: Path) -> None:
    """opencode's native builds for glibc and musl, `bin/opencode` to pick one, and the cache its first run fills.

    The openai-compatible provider tokin is served through ships in the binary, and the models.dev catalogue it
    fetches on first run ships in the cache, so nothing is fetched in a sandbox that blocks the network.
    """
    for libc, package in [("glibc", "opencode-linux-x64"), ("musl", "opencode-linux-x64-musl")]:
        url = f"https://registry.npmjs.org/{package}/-/{package}-{VERSION}.tgz"
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:gz") as npm:
            binary = npm.extractfile("package/bin/opencode")
            assert binary is not None
            (root / libc).mkdir()
            (root / libc / "opencode").write_bytes(binary.read())
        (root / libc / "opencode").chmod(0o755)
    (root / "bin").mkdir()
    start = root / "bin" / "opencode"
    start.write_text(
        "#!/bin/sh\n"
        "libc=glibc; if [ -f /etc/alpine-release ] || ldd --version 2>&1 | grep -qi musl; then libc=musl; fi\n"
        f'XDG_CACHE_HOME={SANDBOX}/cache exec {SANDBOX}/$libc/opencode "$@"\n'
    )
    start.chmod(0o755)
    # A run against nothing fills the cache; getting as far as the connection shows the provider needs no fetch.
    provider = {"npm": PROVIDER, "options": {"baseURL": "http://127.0.0.1:9/v1"}, "models": {"policy": {}}}
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
        raise RuntimeError(f"opencode {VERSION} never reached the endpoint: {tried.stdout}{tried.stderr}")


PREBUILT = Prebuilt("opencode", VERSION, SANDBOX, build)


class PrebuiltOpenCodeOptions(OpenCodeOptions):
    archive: str


class PrebuiltOpenCode(OpenCode):
    """Harbor's opencode, installed from a prebuilt archive of its native builds rather than with nvm and npm."""

    options_model: ClassVar[type[OpenCodeOptions]] = PrebuiltOpenCodeOptions
    options: PrebuiltOpenCodeOptions

    async def install(self, environment: BaseEnvironment) -> None:
        await PREBUILT.carry(environment, self.options.archive)
        # Where Harbor's run command finds it, with a cache opencode may rewrite as whichever user it runs as.
        await self.exec_as_root(
            environment,
            command=f"chmod -R a+rwX {SANDBOX}/cache && mkdir -p /usr/local/bin && ln -sf {SANDBOX}/bin/opencode /usr/local/bin/",
        )
