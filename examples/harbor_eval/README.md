# Harbor Evaluation

Evaluate an agent harness on a [Harbor](https://github.com/laude-institute/harbor) benchmark, its model served
through tokin. Each sample runs in its own e2b sandbox with its own tokin session, and the session, token ids,
logprobs and all, is kept beside the trial.

| Benchmark | Tasks |
|---|---|
| `terminal-bench-2.1` | 89 |
| `swebench-verified` | 500 |

| Harness | Harbor agent | Defaults |
|---|---|---|
| `mini-swe-agent` | `mini-swe-agent`, patched | 400 steps, 1200 s a command, where its own are none and 30 s |
| `pi` | `pi` | the `openai-completions` API, which a custom endpoint needs |

`--agent-kwargs` replaces these key by key: `--agent pi --agent-kwargs '{"thinking": "low"}'` keeps pi's
`model_api`, while a mini-swe-agent `config` given whole replaces both of its defaults.

## Run

From the repository root, install tokin from this checkout and the example's dependencies, resolved together:

```bash
pip install -e . -r examples/harbor_eval/requirements.txt    # or, in a uv venv: uv pip install -e . -r ...
```

Serve the model with sglang, and tokin in front of it:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B --port 30000
python -m tokin --model-path Qwen/Qwen3.6-35B-A3B --engine-url http://localhost:30000 --port 8000
```

Then, with an e2b key in `E2B_API_KEY`:

```bash
python examples/harbor_eval/run.py --benchmark terminal-bench-2.1 --agent mini-swe-agent --samples 4 \
    --params '{"temperature": 1.0, "top_p": 1.0}'
```

Every option can also be set as `HARBOR_EVAL_<OPTION>`; `--help` lists them. `--params` are session params:
they fill what the harness leaves out of a request and overrule what it sets, so every turn samples as the
evaluation asks.

## Output

```
harbor-eval/<benchmark>/<agent>/
  results.jsonl                    one row per sample: reward, turns, tokens, finish reasons
  metrics.json                     pass@k over the tasks whose every sample was scored
  trials/<task>__<k>__<session>/   Harbor's trial, with `session.safetensors` beside it
```

A session reads back as tokin wrote it:

```python
from pathlib import Path
from tokin import Session

session = Session.unpack(Path("…/session.safetensors").read_bytes())
for rollout in session.rollouts:
    rollout.token_ids, rollout.loss_mask, rollout.logprobs, rollout.messages
```

A session holds more than one rollout when the harness rewrote its history, as mini-swe-agent does when it
drops a reply that called no tool and asks again.

Running the same command again resumes: samples with a row are skipped. A sample that ended before its first
turn failed in the sandbox rather than the policy, so it is retried, up to three times, and run again on resume.

## How the sandbox reaches tokin

The harness runs inside the sandbox, and a sandbox cannot dial the host tokin runs on. `TunneledE2BEnvironment`
opens a reverse tunnel when the sandbox starts: [chisel](https://github.com/jpillora/chisel) serves
`--reverse` in the sandbox behind the https port e2b exposes, and a client here asks for `R:8000:<tokin>`. The
harness then calls `http://localhost:8000/sessions/<id>/v1`. Chisel is downloaded into `--cache` on first use.

A self-hosted e2b that cannot build templates takes `--templates`, a JSON file naming a pre-built template per
task, with the cluster's domain in `E2B_DOMAIN`.

## mini-swe-agent's command timeout

Since 2.4.1, mini-swe-agent drains a timed-out command's output with no timeout of its own after killing its
process group, so a process that left the group still holding stdout, as `qemu -daemonize` does, hangs the
agent until the trial's timeout. `patched_mini_swe.py` restores 2.4.0's behaviour once Harbor has installed it,
and fails the install if the code it patches has changed.
