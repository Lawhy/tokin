from harbor.agents.installed.mini_swe_agent import MiniSweAgent

OLD = """    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout)"""
NEW = """    except subprocess.TimeoutExpired as expired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        process.wait()
        raise subprocess.TimeoutExpired(command, timeout, output=expired.output)"""
PATCH = f"""
"$HOME/.local/share/uv/tools/mini-swe-agent/bin/python" - <<'PY'
import minisweagent.environments.local as local
text = open(local.__file__).read()
if {OLD!r} not in text:
    raise SystemExit(f"mini-swe-agent's _run changed; the timeout patch no longer applies to {{local.__file__}}")
open(local.__file__, "w").write(text.replace({OLD!r}, {NEW!r}))
PY
"""


class PatchedMiniSweAgent(MiniSweAgent):
    """Harbor's mini-swe-agent, its command timeout made to hold.

    Since 2.4.1, `_run` drains the pipe with no timeout after killing a command's process group: a process that
    left the group still holding stdout, as `qemu -daemonize` does, hangs the agent for good. The patch waits on
    the shell instead, as `subprocess.run` and 2.4.0 do, keeping the output that came before the timeout. It fails
    the install when `_run` no longer reads as patched, rather than run unpatched.
    """

    async def install(self, environment) -> None:
        await super().install(environment)
        await self.exec_as_agent(environment, command=PATCH)
