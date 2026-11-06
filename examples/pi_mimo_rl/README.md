# Train pi on MiMo-RL

Fully asynchronous coding agent RL of [pi](https://github.com/earendil-works/pi) on [MiMo-V2.6-RL-oss](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss), with REINFORCE + score centering. slime trains the model, tokin manages the rollout sessions, and Harbor runs each trial in an e2b sandbox.

## Run

**0. Requirements:** slime at `859938dc` or later on its image `nightly-dev-20261016b-cu130` or later, which score
centering needs.

**1. Install**, on every node, tokin into slime's python, and tokin with its `harbor` extra into a directory the
trials run from, so Harbor's dependencies stay apart from slime's:

```bash
pip install tokin
pip install --target "$HARBOR_DIR" "tokin[harbor]"
```

**2. Prepare the data**, holding out the tasks in `heldout.txt`:

```bash
hf download FineEnvs/MiMo-V2.6-RL-harbor-code --repo-type dataset \
    --revision cf87dbe426ae26d15ca416ec599e549f65f84249 --include "tasks/*" --local-dir "$DATA"
python - "$DATA" <<'PY'
import json, sys
from pathlib import Path
data = Path(sys.argv[1])
held = set(Path("examples/pi_mimo_rl/heldout.txt").read_text().split())
with open(data / "train.jsonl", "w") as rows:
    for task in sorted(d.name for d in (data / "tasks").iterdir() if d.name not in held):
        rows.write(json.dumps({"prompt": [{"role": "user", "content": task}], "label": task, "metadata": {"task": task}}) + "\n")
PY
```

**3. Build the pi archive**, on a Linux x86_64 host:

```bash
PYTHONPATH="$HARBOR_DIR" python examples/pi_mimo_rl/pi.py "$RUN_ROOT/cache"
```

**4. Launch** on a Ray cluster:

```bash
export E2B_API_KEY=...
SLIME_DIR=... MEGATRON_DIR=... HF_CHECKPOINT=.../Qwen3.8-27B RUN_ROOT=... SOCKET_IFNAME=... \
RAY_ADDRESS=... RAY_DASHBOARD_ADDRESS=... PROMPT_DATA="$DATA/train.jsonl" MIMO_TASKS="$DATA/tasks" \
HARBOR_DIR=... PI_ARCHIVE=... examples/pi_mimo_rl/run_qwen38_27b.sh
```

The recipe and every other setting are in `run_qwen38_27b.sh`.

## How it works

- `generate.py`: slime's custom generate hook. Each sample opens a tokin session and runs one Harbor trial on it.
- `trial.py`: the trial, in a process of its own.
- `pi.py`: pi, installed from a prebuilt archive.
- `sandbox.py`: the e2b sandbox, with a reverse tunnel to tokin.

## Notes

- slime's `--save-hf` writes only the trained parameters ([`hf_checkpoint_saver.py#L94`](https://github.com/THUDM/slime/blob/859938dcb2b0791dbae9084a4d9ae91a4f38f3b7/slime/backends/megatron_utils/hf_checkpoint_saver.py#L94)), so a Qwen3.8 save lacks the base's `mtp.*` and `model.visual.*` tensors. To serve a save, convert its torch_dist checkpoint with slime's `tools/convert_torch_dist_to_hf.py --add-missing-from-origin-hf`.
- A save waits for every in-flight trial to finish ([`rollout.py#L414-L428`](https://github.com/THUDM/slime/blob/859938dcb2b0791dbae9084a4d9ae91a4f38f3b7/slime/ray/rollout.py#L414-L428)), up to the agent's time budget.

## Acknowledgement

This example is adapted from slime's [`coding_agent_rl`](https://github.com/THUDM/slime/tree/859938dc/examples/coding_agent_rl) example.
