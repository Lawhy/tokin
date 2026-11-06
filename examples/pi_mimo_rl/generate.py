import asyncio
import base64
import copy
import json
import logging
import os
import random
import sys
import threading
from argparse import Namespace
from pathlib import Path
from typing import Any

import httpx2
import torch
import uvicorn
from slime.utils.types import Sample
from transformers import AutoTokenizer

from tokin import FinishReason, Gateway, GenerationParams, Session, SGLangEngine, get_chat_template
from tokin.chat_templates import ChatTemplate
from tokin.rollout import Generation
from tokin.server import Server

HARBOR_DIR = os.environ["HARBOR_DIR"]
TRIALS = Path(os.environ["TRIALS_DIR"])
PORT = 18100
ATTEMPTS = 3
# Seconds a trial's start is spread over, since e2b drops part of a burst of sandbox creates.
JITTER = 10
# Past a slow trial's sandbox setup, 3,600 s of agent and 2,100 s of verifier.
GUARD = 8100
# Failures outside the policy, run again since that biases nothing.
INFRA = (
    "ProtocolError",
    "RemoteProtocolError",
    "NetworkConnectionError",
    "AddTestsDirError",
    "DownloadVerifierDirError",
    "RewardFileNotFoundError",
    "VerifierTimeoutError",
)
SERVER: dict[str, Any] = {}

logger = logging.getLogger(__name__)


async def tokin_client(args: Namespace) -> httpx2.AsyncClient:
    """A client of the tokin Server this process starts once, on its own thread and loop, in front of slime's router."""
    async with SERVER.setdefault("lock", asyncio.Lock()):
        if "http" not in SERVER:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            family = get_chat_template(os.environ.get("TOKIN_CHAT_TEMPLATE") or args.hf_checkpoint)
            template = family(tokenizer, **(args.apply_chat_template_kwargs or {}))
            # The context is the engine's, which tokin reads through the router; slime sets it with
            # --sglang-context-length.
            engine = SGLangEngine(f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
            config = uvicorn.Config(
                Server(Gateway(template, engine)).app, port=PORT, log_level="warning", timeout_keep_alive=75
            )
            server = uvicorn.Server(config)
            thread = threading.Thread(target=server.run, name="tokin-server", daemon=True)
            thread.start()
            while not server.started:
                if not thread.is_alive():
                    raise RuntimeError(f"tokin's server on 127.0.0.1:{PORT} stopped before it started")
                await asyncio.sleep(1)
            (TRIALS / "chisel").mkdir(parents=True, exist_ok=True)
            SERVER.update(http=httpx2.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=None), template=template)
    http: httpx2.AsyncClient = SERVER["http"]
    return http


def session_params(args: Namespace, sampling_params: dict[str, Any]) -> GenerationParams:
    """slime's sampling, fixed for every turn: what the trainer replays."""
    params = GenerationParams(
        temperature=sampling_params["temperature"],
        top_p=sampling_params.get("top_p", 1.0),
        max_tokens=sampling_params["max_new_tokens"],
        return_logprobs=True,
    )
    if sampling_params.get("top_k", -1) not in (-1, 0, None):
        params["top_k"] = sampling_params["top_k"]
    # slime replays the top-p support whenever top_p < 1.
    if params["top_p"] < 1:
        params["return_support"] = True
    if args.use_rollout_routing_replay:
        params["return_routed_experts"] = True
    return params


async def run_trial(args: Namespace, params: GenerationParams, task: str, name: str, sid: str) -> dict[str, Any]:
    """The trial's row; cancelling this stops the trial, which tears its sandbox down.

    The trial runs in a process of its own with Harbor's dependencies first on its path: they differ from slime's,
    which they would replace in this one.
    """
    trial_dir = TRIALS / name
    trial_dir.mkdir(parents=True)
    spec = {"task": task, "name": name, "session": sid, "port": PORT}
    spec |= {"context_window": args.rollout_max_context_len, "max_tokens": params["max_tokens"]}
    with open(trial_dir / "runner.log", "ab") as log:  # noqa: ASYNC230  # the child writes it, not this loop
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("trial.py")),
            json.dumps(spec),
            env={**os.environ, "PYTHONPATH": HARBOR_DIR},
            stdout=log,
            stderr=log,
        )
    try:
        await child.wait()
    except asyncio.CancelledError:
        child.terminate()
        await asyncio.shield(asyncio.wait_for(child.wait(), 300))
        raise
    result = trial_dir / "result.json"
    if not result.exists():
        return {"reward": None, "exception": f"RunnerError: the trial exited {child.returncode}", "phase": "unknown"}
    row: dict[str, Any] = json.loads(result.read_text())
    return row


def unfinished(row: dict[str, Any], session: Session) -> bool:
    """A trial to run again: no turn or no reward, a failure outside the policy, or one before the agent acted.

    Outside the policy are the `INFRA` exceptions and the sandbox's connection dropping (exit -1) or the sandbox
    dying under the agent (exit 135).
    """
    exception = row.get("exception") or ""
    crashed = exception.startswith("NonZeroAgentExitCodeError") and (
        "(exit -1)" in exception or "(exit 135)" in exception
    )
    return (
        not session.turns
        or row["reward"] is None
        or exception.split(":")[0] in INFRA
        or crashed
        or row.get("phase") in ("environment_setup", "agent_setup")
    )


def support(args: Namespace, generation: Generation) -> dict[str, Any]:
    """The top-p support `generation` was sampled from, as sglang's meta info carries it to slime."""
    if args.rollout_top_p >= 1:
        return {}
    if (s := generation.support) is None:
        raise ValueError("a generation came back without the top-p support slime replays")
    fields = {"token_ids": s.ids, "token_offsets": s.offsets, "log_probs": s.logprobs}
    return {f"top_p_{key}": base64.b64encode(value).decode() for key, value in fields.items()}


def to_samples(
    args: Namespace, base: Sample, session: Session, template: ChatTemplate, truncation: str | None
) -> list[Sample]:
    """One Sample per leaf of the session, a turn shared by several leaves trained once, in the first.

    TRUNCATED only for a `truncation`, the compaction cap or the time budget; a last turn stopped on length is
    recorded as `truncated` in the metadata, as the example's are.
    """
    trained: set[str] = set()
    samples: list[Sample] = []
    for rollout in session.rollouts:
        s = copy.copy(base)
        # A shallow copy shares the base's lists, which `append_response_tokens` extends.
        s.tokens = list(rollout.turns[0].prompt.token_ids)
        s.response, s.response_length, s.loss_mask, s.rollout_log_probs = "", 0, None, None
        s.rollout_top_p_token_ids = s.rollout_top_p_token_offsets = s.rollout_top_p_log_probs = None
        s.weight_versions = []
        for j, turn in enumerate(rollout.turns):
            if j:
                s.append_response_tokens(
                    args, tokens=turn.prompt.token_ids, trainable=False, update_terminal_info=False
                )
            g, fresh = turn.generation, turn.id not in trained
            trained.add(turn.id)
            s.append_response_tokens(
                args,
                tokens=g.token_ids,
                log_probs=g.logprobs if fresh else None,
                trainable=fresh,
                meta_info=support(args, g) if fresh else None,
                update_terminal_info=False,
            )
            if fresh:
                s.weight_versions.extend(version for version, _, _ in g.weight_versions)
        cut = rollout.turns[-1].generation.finish_reason is FinishReason.LENGTH
        s.response = template.decode(s.tokens[len(s.tokens) - s.response_length :])
        s.status = Sample.Status.TRUNCATED if truncation else Sample.Status.COMPLETED
        s.metadata = {**(base.metadata or {}), "tokin_session": session.id, "truncated": cut or bool(truncation)}
        if truncation:
            s.metadata["truncation_reason"] = truncation
        # Siblings from one session are one rollout to slime's loss reducer, which requires them to share it.
        s.rollout_id = base.rollout_id if base.rollout_id is not None else base.index
        if args.use_rollout_routing_replay:
            # `(len(tokens) - 1, layers, top_k)`, the shape slime's routing replay reads.
            experts = rollout.routed_experts(args.num_layers, args.moe_router_topk)
            s.rollout_routed_experts = torch.from_numpy(experts.copy())
        samples.append(s)
    return samples


async def generate(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    """The Samples of one pi trial on the MiMo task `sample.label`, rewarded by its verifier.

    A trial that failed outside the policy runs again, up to `ATTEMPTS` times, then leaves the sample ABORTED, as
    does one past `GUARD`; fully async slime requeues it.
    """
    http = await tokin_client(args)
    task = sample.label
    params = session_params(args, sampling_params)
    for attempt in range(ATTEMPTS):
        await asyncio.sleep(random.uniform(0, JITTER))
        sid = await Session.open(http, params, resume_aborted=args.partial_rollout)
        name = f"{task}__{sample.index}__{sid[:6]}"
        try:
            async with asyncio.timeout(GUARD):
                row = await run_trial(args, params, task, name, sid)
        except TimeoutError:
            row = {"reward": None, "exception": f"GuardTimeout: the trial outlasted {GUARD} s", "phase": "guard"}
        finally:
            session = await Session.collect(http, sid)
            (TRIALS / name / "session.safetensors").write_bytes(session.pack())
        row |= {"task": task, "index": sample.index, "attempt": attempt, "session": sid}
        with open(TRIALS / "results.jsonl", "a") as results:  # noqa: ASYNC230  # one line, appended
            results.write(json.dumps(row) + "\n")
        if row.get("phase") == "guard" or not unfinished(row, session):
            break
        logger.warning("trial %s failed outside the policy: %s", name, (row.get("exception") or "no reward")[:300])
        await asyncio.sleep(random.uniform(10, 60))
    if row.get("phase") == "guard" or unfinished(row, session):
        sample.status = Sample.Status.ABORTED
        return [sample]
    timed_out = (row.get("exception") or "").startswith("AgentTimeoutError")
    truncation = "compact_limit" if row["compaction_cap"] else "time_budget" if timed_out else None
    samples = to_samples(args, sample, session, SERVER["template"], truncation)
    for s in samples:
        s.reward = row["reward"]
    # A list even of one: slime flattens a batch by its first item's shape.
    return samples


# slime cancels a generation it no longer needs, and the trial with it.
generate.abort_mode = "request"  # type: ignore[attr-defined]
