# Harbor eval

Evaluate an agent harness on a [Harbor](https://github.com/laude-institute/harbor) benchmark through tokin: one e2b
sandbox and one tokin session per sample, the session saved beside the trial.

## Run

On a Linux x86_64 host, from the repository root:

```bash
uv sync --group harbor
python -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B --port 30000
tokin serve --model-path Qwen/Qwen3.6-35B-A3B --engine-url http://localhost:30000
E2B_API_KEY=... python examples/harbor_eval/run.py --benchmark terminal-bench-2.1 --agent pi --samples 4
```

`--benchmark` is `terminal-bench-2.1` or `swebench-verified`, and `--agent` is `mini-swe-agent`, `pi` or `opencode`.
`--help` lists the rest. Running the same command again resumes.

Each sandbox reaches tokin through a reverse tunnel (`sandbox.py`). Each harness is installed from a prebuilt archive
rather than in every sandbox (`agents/`). An archive changes how its harness is installed, not what it does, apart
from the patches commented in its module.

## Output

```
harbor-eval/<benchmark>/<agent>/
  results.jsonl                    one row per sample
  metrics.json                     pass@k
  trials/<task>__<k>__<session>/   Harbor's trial, with `session.safetensors`
```

## Reference

Qwen3.6-35B-A3B on Terminal-Bench 2.1, 89 tasks × 4 samples, with
`--params '{"temperature": 1.0, "top_p": 1.0, "max_tokens": 65536}' --memory-scale 2`:

| Harness | pass@1 | pass@4 |
|---|---|---|
| `mini-swe-agent` | 0.451 | 0.616 |
| `opencode` | 0.454 | 0.655 |
| `pi` | 0.438 | 0.625 |
