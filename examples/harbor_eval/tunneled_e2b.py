import asyncio
import json
import secrets
from pathlib import Path

from harbor.environments.e2b import E2BEnvironment
from harbor.models.trial.paths import EnvironmentPaths

TUNNEL_PORT = 9000
LOCAL_PORT = 8000
SANDBOX_CHISEL = "/opt/tokin-tunnel/chisel"


class TunneledE2BEnvironment(E2BEnvironment):
    """Harbor's e2b environment, with tokin served on the sandbox's own `localhost:8000`.

    An agent installed in the sandbox calls its model from inside, and a sandbox cannot dial the host tokin runs
    on. So `start` opens a reverse tunnel: `chisel server --reverse` in the sandbox, behind the https port e2b
    exposes, and a `chisel client` here asking for `R:8000:<tokin>`.

    Attributes:
        tokin (str): `host:port` of tokin, as this host reaches it.
        chisel (dict[str, str]): The chisel binaries, `{"host": ..., "sandbox": ...}`, this host's and linux amd64.
        templates (str | None): A JSON file naming a pre-built template per task, for a self-hosted e2b that
            cannot build them; `None` builds them as Harbor does.
    """

    def __init__(self, *args, tokin: str, chisel: dict[str, str], templates: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokin, self.chisel = tokin, chisel
        self.sandbox_chisel = Path(chisel["sandbox"]).read_bytes()
        self.prebuilt = templates is not None
        if templates is not None:
            self._template_name = json.loads(Path(templates).read_text())[self.environment_name.split("/")[-1]]
        self.client: asyncio.subprocess.Process | None = None

    async def start(self, force_build: bool) -> None:
        if not self.prebuilt:
            await super().start(force_build)
        else:
            await self._create_sandbox()
            # The template build skipped here is what bakes Harbor's log and test dirs into the image.
            paths = EnvironmentPaths
            dirs = [paths.logs_dir, paths.agent_dir, paths.verifier_dir, paths.artifacts_dir, paths.tests_dir]
            await self.ensure_dirs([*dirs, *self._mount_targets(writable_only=True)])
            await self._upload_environment_dir_after_start()
        await self.open_tunnel()

    async def open_tunnel(self) -> None:
        """Serve tokin on the sandbox's `localhost:8000`, or raise if the sandbox cannot reach it."""
        sandbox = self._sandbox
        auth = f"tokin:{secrets.token_hex(16)}"
        await sandbox.files.write(SANDBOX_CHISEL, self.sandbox_chisel, user="root", gzip=True)
        await sandbox.commands.run(f"chmod +x {SANDBOX_CHISEL}", user="root")
        # `timeout=0`: e2b kills a command, in the background or not, once its 60 s default runs out.
        await sandbox.commands.run(
            f"{SANDBOX_CHISEL} server --port {TUNNEL_PORT} --reverse --auth {auth}",
            background=True,
            timeout=0,
            user="root",
        )
        log = await asyncio.to_thread(open, self.trial_paths.trial_dir / "tunnel.log", "ab")
        with log:
            self.client = await asyncio.create_subprocess_exec(
                self.chisel["host"],
                "client",
                "--auth",
                auth,
                "--keepalive",
                "25s",
                f"https://{sandbox.get_host(TUNNEL_PORT)}",
                f"R:{LOCAL_PORT}:{self.tokin}",
                stdout=log,
                stderr=log,
            )
        # Bash's `/dev/tcp`, since a task image may have no curl, wget or python.
        probe = (
            f"timeout 5 bash -c 'exec 3<>/dev/tcp/127.0.0.1/{LOCAL_PORT}; "
            'printf "GET /health HTTP/1.0\\r\\n\\r\\n" >&3; head -c 200 <&3\' 2>&1 || true'
        )
        for _ in range(60):
            if "200 OK" in (await sandbox.commands.run(probe, timeout=15, user="root")).stdout:
                return
            await asyncio.sleep(1)
        raise RuntimeError(
            f"the sandbox cannot reach tokin through the tunnel; see {self.trial_paths.trial_dir}/tunnel.log"
        )

    async def stop(self, delete: bool) -> None:
        if self.client and self.client.returncode is None:
            self.client.kill()
            await self.client.wait()
        await super().stop(delete)
