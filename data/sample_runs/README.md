# Sample runs

| File | Contents |
|---|---|
| `calibration_runs.jsonl` | 50 trajectories (stratified over outcome and failure mode) that are labeled in `../human_labels/labels.jsonl` |
| `react_gpt4o_style.jsonl` | 60 trajectories: ReAct + gpt-4o-style on all 20 tasks × 3 trials (`scripts/run_eval.py --out ...`) |

Each line is a `agenteval.types.Trajectory` serialised with `to_dict()`: every step's thought,
tool call and result, the final answer, termination reason, policy audit log, evaluator verdict,
token usage and (for simulated models only) the injected-fault log.

```python
from agenteval.types import Trajectory, load_jsonl
runs = [Trajectory.from_dict(d) for d in load_jsonl("data/sample_runs/calibration_runs.jsonl")]
```

All runs come from **simulated** models. Regenerate them with `scripts/build_sample_data.py`.
