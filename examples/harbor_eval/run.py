import asyncio
import json
import math
import random
import signal
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import click
import httpx2
from agents import bundle
from environment import LOCAL_PORT, fetch_chisel
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

from tokin import Session


@dataclass(frozen=True)
class Benchmark:
    """A Harbor dataset at a fixed commit: its tasks are the directories under `subdir`."""

    git: str
    ref: str
    subdir: str


BENCHMARKS = {
    "terminal-bench-2.1": Benchmark(
        "https://github.com/harbor-framework/terminal-bench-2-1.git",
        "7131e4375048a0e408a8fb404b5f499d726b695b",
        "tasks",
    ),
    "swebench-verified": Benchmark(
        "https://github.com/laude-institute/harbor-datasets.git",
        "37db108843a49bb31a592e37a75e2c40dc3f9749",
        "datasets/swebench-verified",
    ),
}
# How Harbor runs each harness against tokin, unpacked from a bundle: the model it names, which tokin answers as the
# one it fronts whatever the name, and its defaults in `kwargs`, which `--agent-kwargs` replaces key by key.
AGENTS: dict[str, dict[str, Any]] = {
    # mini-swe-agent defaults to 30 s a command and no step limit.
    "mini-swe-agent": {
        "import_path": "agents:BundledMiniSweAgent",
        "model": "openai/policy",
        "kwargs": {"config": {"agent": {"step_limit": 400}, "environment": {"timeout": 1200}}},
    },
    # No `max_turns`: pi states it to the model in the system prompt.
    "pi": {"import_path": "agents:BundledPi", "model": "openai/policy", "kwargs": {"model_api": "openai-completions"}},
    # opencode sends a provider named `openai` to the Responses API whatever its package; tokin serves chat
    # completions, which the openai-compatible provider speaks under any other name.
    "opencode": {
        "import_path": "agents:BundledOpenCode",
        "model": "tokin/policy",
        "kwargs": {
            "opencode_config": {
                "provider": {
                    "tokin": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": "{env:OPENAI_API_BASE}", "apiKey": "{env:OPENAI_API_KEY}"},
                        "models": {"policy": {}},
                    }
                },
                "autoupdate": False,
                "share": "disabled",
            }
        },
    },
}
# A sample that failed outside the policy runs again, since that biases nothing: the sandbox's connection dropping
# (exit -1), the sandbox dying under the agent (exit 135), Harbor finding no reward, or the verifier timing out.
INFRA = (
    "ProtocolError",
    "RemoteProtocolError",
    "NetworkConnectionError",
    "AddTestsDirError",
    "DownloadVerifierDirError",
    "RewardFileNotFoundError",
    "VerifierTimeoutError",
)
ATTEMPTS = 3


def fetch_tasks(benchmark: Benchmark, cache: Path) -> Path:
    """The benchmark's task directory, sparse-checked out into `cache` at its commit on first use."""
    repo = cache / benchmark.ref
    if not (repo / benchmark.subdir).is_dir():

        def git(*args: str) -> None:
            subprocess.run(["git", "-C", str(repo), *args], check=True)

        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", benchmark.git, str(repo)],
            check=True,
        )
        git("sparse-checkout", "init", "--cone")
        git("sparse-checkout", "set", benchmark.subdir)
        git("fetch", "--depth", "1", "origin", benchmark.ref)
        git("checkout", benchmark.ref)
    return repo / benchmark.subdir


def as_json(ctx: click.Context, param: click.Parameter, value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"not JSON: {e}") from e


def pass_at(rows: list[dict], k: int, n: int) -> float | None:
    """Unbiased pass@k over the tasks with `n` scored samples; a reward of 1 is a pass."""
    by_task: dict[str, list[bool]] = {}
    for row in rows:
        if row["reward"] is not None:
            by_task.setdefault(row["task"], []).append(row["reward"] >= 1)
    scores = [1 - math.comb(n - sum(v), k) / math.comb(n, k) for v in by_task.values() if len(v) == n]
    return sum(scores) / len(scores) if scores else None


@dataclass(frozen=True)
class Run:
    tokin: str
    agent: str
    agent_kwargs: dict[str, Any]
    params: dict[str, Any]
    out: Path
    environment: dict[str, Any]
    agent_timeout: float | None
    memory_scale: float


async def attempt(http: httpx2.AsyncClient, run: Run, task_dir: Path, k: int) -> dict:
    """One run of sample `k` of `task_dir`, its packed session saved beside the trial."""
    memory = Task(task_dir).config.environment.memory_mb
    trial = TrialConfig(
        task=TaskConfig(path=task_dir),
        trial_name=f"{task_dir.name}__{k}",
        trials_dir=run.out / "trials",
        agent=AgentConfig(
            import_path=AGENTS[run.agent]["import_path"],
            model_name=AGENTS[run.agent]["model"],
            override_timeout_sec=run.agent_timeout,
            kwargs=run.agent_kwargs,
        ),
        environment=EnvironmentConfig(
            import_path="environment:TunneledE2BEnvironment",
            override_memory_mb=int(memory * run.memory_scale) if memory and run.memory_scale != 1 else None,
            kwargs=run.environment,
        ),
    )
    start = time.time()
    row: dict[str, Any] = {"task": task_dir.name, "k": k}
    sid = await Session.open(http, run.params)
    # The harness reaches tokin on the sandbox's own localhost, which `TunneledE2BEnvironment` serves.
    env = {"OPENAI_API_BASE": f"http://localhost:{LOCAL_PORT}/sessions/{sid}/v1", "OPENAI_API_KEY": "tokin"}
    trial = trial.model_copy(
        update={"trial_name": f"{trial.trial_name}__{sid[:6]}", "agent": trial.agent.model_copy(update={"env": env})}
    )
    try:
        result = await (await Trial.create(trial)).run()
        rewards = result.verifier_result.rewards if result.verifier_result else None
        row["reward"] = rewards.get("reward") if rewards else None
        if info := result.exception_info:
            row["exception"] = f"{info.exception_type}: {info.exception_message}"
    except Exception as e:
        row["reward"], row["exception"] = None, f"{type(e).__name__}: {e}"
    finally:
        session = await Session.collect(http, sid)
    trial_dir = run.out / "trials" / trial.trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "session.safetensors").write_bytes(session.pack())
    finishes = [turn.generation.finish_reason for turn in session.turns]
    return row | {
        "session": session.id,
        "seconds": round(time.time() - start),
        "rollouts": len(session.rollouts),
        "turns": len(finishes),
        "tokens": len(session),
        "finish_reasons": dict(Counter(finishes)),
    }


def unfinished(row: dict) -> bool:
    """A sample to run again: one with no turn or no score has no result, and one that failed outside the policy."""
    exception = row.get("exception") or ""
    crashed = exception.startswith("NonZeroAgentExitCodeError") and (
        "(exit -1)" in exception or "(exit 135)" in exception
    )
    return not row["turns"] or row["reward"] is None or exception.split(":")[0] in INFRA or crashed


async def sample(
    http: httpx2.AsyncClient, run: Run, task_dir: Path, k: int, slots: asyncio.Semaphore, jitter: float, results
) -> None:
    async with slots:
        # Sandbox creates spread over `jitter` seconds, since e2b drops part of a burst of them.
        await asyncio.sleep(random.uniform(0, jitter))
        for _ in range(ATTEMPTS):
            row = await attempt(http, run, task_dir, k)
            results.write(json.dumps(row) + "\n")
            results.flush()
            click.echo(
                f"{row['task']}#{k} reward={row['reward']} turns={row['turns']} {row.get('exception', '')[:200]}"
            )
            if not unfinished(row):
                return
            await asyncio.sleep(random.uniform(10, 60))


def latest(path: Path) -> dict[tuple[str, int], dict]:
    """The last row of each sample in `path`, since an unfinished one is run again."""
    rows = map(json.loads, path.read_text().splitlines()) if path.exists() else []
    return {(r["task"], r["k"]): r for r in rows}


async def evaluate(run: Run, tasks: list[Path], samples: int, concurrency: int, jitter: float) -> dict[str, Any]:
    run.out.mkdir(parents=True, exist_ok=True)
    path = run.out / "results.jsonl"
    done = {key for key, row in latest(path).items() if not unfinished(row)}
    todo = [(d, k) for k in range(samples) for d in tasks if (d.name, k) not in done]
    click.echo(f"{len(todo)} samples to run, {len(done)} done")
    slots = asyncio.Semaphore(concurrency)
    async with httpx2.AsyncClient(base_url=run.tokin, timeout=None) as http:
        (await http.get("/health")).raise_for_status()
        with path.open("a") as results:
            await asyncio.gather(*(sample(http, run, d, k, slots, jitter, results) for d, k in todo))
    rows = list(latest(path).values())
    summary: dict[str, Any] = {f"pass@{k}": pass_at(rows, k, samples) for k in range(1, samples + 1)}
    summary |= {"samples": len(rows), "unscored": sum(r["reward"] is None for r in rows)}
    (run.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


@click.command(context_settings={"auto_envvar_prefix": "HARBOR_EVAL", "show_default": True})
@click.option("--benchmark", type=click.Choice(sorted(BENCHMARKS)), default="terminal-bench-2.1")
@click.option("--agent", type=click.Choice(sorted(AGENTS)), default="mini-swe-agent")
@click.option("--tokin-url", default="http://127.0.0.1:8000", help="Where tokin serves, as this host reaches it.")
@click.option("--agent-kwargs", default="{}", callback=as_json, help="JSON over the harness's defaults, key by key.")
@click.option("--params", default="{}", callback=as_json, help="JSON session params, e.g. '{\"temperature\": 1.0}'.")
@click.option("--samples", type=click.IntRange(min=1), default=1, help="Samples per task, pass@k up to it.")
@click.option("--only", default="", help="Comma-separated task names to run, all by default.")
@click.option("--concurrency", type=click.IntRange(min=1), default=32)
@click.option("--jitter", type=float, help="Seconds sandbox starts spread over; one per slot by default.")
@click.option("--agent-timeout", type=float, help="Seconds an agent may run; the task's own by default.")
@click.option("--memory-scale", type=float, default=1.0, help="Scale each task's memory.")
@click.option("--templates", type=click.Path(exists=True), help="JSON of pre-built e2b templates by task name.")
@click.option("--out", type=click.Path(path_type=Path), default=Path("harbor-eval"))
@click.option("--cache", type=click.Path(path_type=Path), default=Path.home() / ".cache" / "tokin-harbor-eval")
def main(
    benchmark,
    agent,
    agent_kwargs,
    tokin_url,
    params,
    samples,
    only,
    concurrency,
    jitter,
    agent_timeout,
    memory_scale,
    templates,
    out,
    cache,
) -> None:
    """Evaluate a harness on a Harbor benchmark in e2b sandboxes, its model served through tokin."""
    bench = BENCHMARKS[benchmark]
    cache.mkdir(parents=True, exist_ok=True)
    wanted = set(filter(None, only.split(",")))
    tasks = sorted(d for d in fetch_tasks(bench, cache).iterdir() if d.is_dir() and (not wanted or d.name in wanted))
    environment = {"tokin": urlsplit(tokin_url).netloc, "chisel": fetch_chisel(cache), "templates": templates}
    kwargs = {**AGENTS[agent]["kwargs"], **agent_kwargs, "bundle": str(bundle(agent, cache))}
    run = Run(tokin_url, agent, kwargs, params, out / benchmark / agent, environment, agent_timeout, memory_scale)
    # One second per slot: a sandbox takes a bundle and chisel as it starts, and e2b drops part of a burst.
    spread = concurrency if jitter is None else jitter

    async def cancellable() -> dict[str, Any]:
        # SIGTERM cancels the samples, so each Harbor trial tears its sandbox down rather than leave it running.
        task = asyncio.current_task()
        assert task is not None
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        return await evaluate(run, tasks, samples, concurrency, spread)

    click.echo(json.dumps(asyncio.run(cancellable()), indent=2))


if __name__ == "__main__":
    main()
