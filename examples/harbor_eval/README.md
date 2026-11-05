# Harbor Evaluation

Evaluate an agent harness on a [Harbor](https://github.com/laude-institute/harbor) benchmark through tokin. Each
sample runs in its own e2b sandbox with its own tokin session, saved beside the trial.

## Run

On a Linux x86_64 host, from the repository root.

**1. Install** tokin with its `harbor` extra:

```bash
pip install "tokin[harbor]"
```

**2. Serve** a model with sglang, and tokin in front of it:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B --port 30000
tokin serve --model-path Qwen/Qwen3.6-35B-A3B --engine-url http://localhost:30000
```

**3. Evaluate**, with an e2b key:

```bash
export E2B_API_KEY=...
python examples/harbor_eval/run.py --benchmark terminal-bench-2.1 --agent pi --samples 4
```

| Option | Values |
|---|---|
| `--benchmark` | `terminal-bench-2.1`, `swebench-verified` |
| `--agent` | `mini-swe-agent`, `pi`, `opencode` |
| `--samples` | samples per task, pass@k up to it |

`--help` lists the rest. Running the same command again resumes.

## How it works

- **Sandbox:** each sandbox reaches tokin through a reverse tunnel (`sandbox.py`).
- **Harnesses:** each is installed from a prebuilt archive rather than in every sandbox (`agents/`). An archive
  changes how its harness is installed, not what it does, apart from the patches commented in its module.

## Output

```
harbor-eval/<benchmark>/<agent>/
├── results.jsonl                  one row per sample
├── metrics.json                   pass@k
└── trials/<task>__<k>__<session>/ Harbor's trial, with session.safetensors
```

## Results

Qwen3.6-35B-A3B:

| | `mini-swe-agent` | `opencode` | `pi` |
|---|:---:|:---:|:---:|
| Terminal-Bench 2.1, pass@1 | 0.451 | 0.454 | 0.438 |
| Terminal-Bench 2.1, pass@4 | 0.616 | 0.655 | 0.625 |
| SWE-bench Verified, pass@1 | 0.689 | 0.666 | 0.690 |

Terminal-Bench 2.1 is 89 tasks × 4 samples and SWE-bench Verified 500 tasks × 1, both with:

```bash
--params '{"temperature": 1.0, "top_p": 1.0, "max_tokens": 65536}' --memory-scale 2
```

tokin (with TITO) does not move these scores: against the same gateway re-rendering each turn from text, Qwen3.6's
2,568 paired samples on Terminal-Bench 2.1 and SWE-bench Verified differ by −0.002 [−0.018, +0.015]. What it removes
is the gap to training: every token a trainer reads back is one the model sampled.
