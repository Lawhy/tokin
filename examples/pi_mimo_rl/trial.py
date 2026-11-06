import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from pi import CAP_MARK
from sandbox import LOCAL_PORT, fetch_chisel

TRIALS = Path(os.environ["TRIALS_DIR"])
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


async def run(spec: dict[str, Any]) -> dict[str, Any]:
    """The trial's reward, and its exception with the phase that raised it; SIGTERM cancels it, so Harbor tears the
    sandbox down.
    """
    task = asyncio.current_task()
    assert task is not None
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    config = TrialConfig(
        task=TaskConfig(path=Path(os.environ["MIMO_TASKS"]) / spec["task"]),
        trial_name=spec["name"],
        trials_dir=TRIALS,
        agent=AgentConfig(
            import_path="pi:PrebuiltPi",
            model_name="openai/policy",
            # Harbor's 360 s is less than the 900 s `carry` gives the archive's upload.
            override_setup_timeout_sec=1800,
            env={
                "OPENAI_API_BASE": f"http://localhost:{LOCAL_PORT}/sessions/{spec['session']}/v1",
                "OPENAI_API_KEY": "tokin",
            },
            kwargs={
                "model_api": "openai-completions",
                "archive": os.environ["PI_ARCHIVE"],
                "context_window": spec["context_window"],
                "max_tokens": spec["max_tokens"],
            },
        ),
        environment=EnvironmentConfig(
            import_path="sandbox:TunneledE2BEnvironment",
            kwargs={
                "tokin": f"127.0.0.1:{spec['port']}",
                "chisel": fetch_chisel(TRIALS / "chisel"),
                "templates": os.environ.get("MIMO_TEMPLATES") or None,
            },
        ),
    )
    row: dict[str, Any] = {"reward": None}
    try:
        try:
            trial = await Trial.create(config)
        except Exception:
            row["phase"] = "environment_setup"
            raise
        result = await trial.run()
        rewards = result.verifier_result.rewards if result.verifier_result else None
        row["reward"] = rewards.get("reward") if rewards else None
        if info := result.exception_info:
            row["exception"] = f"{info.exception_type}: {info.exception_message[:4000]}"
            stamp = info.occurred_at.replace(tzinfo=None)
            started = [
                name
                for name in PHASES
                if (timing := getattr(result, name))
                and timing.started_at
                and timing.started_at.replace(tzinfo=None) <= stamp
            ]
            row["phase"] = started[-1] if started else "environment_setup"
    except Exception as e:
        row["exception"] = f"{type(e).__name__}: {str(e)[:4000]}"
        row.setdefault("phase", "unknown")
    log = TRIALS / spec["name"] / "agent" / "pi.txt"
    row["compaction_cap"] = log.exists() and CAP_MARK in log.read_text(errors="replace")
    return row


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    try:
        row = asyncio.run(run(spec))
    except asyncio.CancelledError:
        row = {"reward": None, "exception": "Cancelled: the rollout stopped the trial", "phase": "cancelled"}
    (TRIALS / spec["name"] / "result.json").write_text(json.dumps(row))
