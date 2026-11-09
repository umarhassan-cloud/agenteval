# AgentEval leaderboard (simulated models)

20 tasks x 5 trials per configuration. 95% bootstrap CIs over tasks.

## Overall

| # | Configuration | pass^1 | pass^3 | pass^5 | pass@3 | Policy viol. | Avg steps |
|---:|---|---|---|---|---|---:|---:|
| 1 | Claude-style + ReAct | 0.79 [0.73, 0.86] | 0.47 [0.33, 0.64] | 0.30 [0.10, 0.50] | 1.00 | 9.0% | 4.2 |
| 2 | GPT-4o-style + ReAct | 0.72 [0.60, 0.83] | 0.44 [0.27, 0.62] | 0.30 [0.10, 0.50] | 0.95 | 10.0% | 4.3 |
| 3 | GPT-4o-style + Plan-then-Act | 0.68 [0.59, 0.77] | 0.32 [0.18, 0.48] | 0.15 [0.00, 0.35] | 0.96 | 12.0% | 4.1 |
| 4 | Claude-style + Plan-then-Act | 0.63 [0.52, 0.72] | 0.26 [0.14, 0.40] | 0.10 [0.00, 0.25] | 0.92 | 9.0% | 3.9 |
| 5 | Llama-3-70B-style + Plan-then-Act | 0.53 [0.40, 0.66] | 0.24 [0.10, 0.41] | 0.15 [0.00, 0.30] | 0.84 | 13.0% | 4.2 |
| 6 | Claude-style + Naive loop | 0.60 [0.51, 0.69] | 0.19 [0.10, 0.30] | 0.05 [0.00, 0.15] | 0.95 | 13.0% | 4.8 |
| 7 | Llama-3-70B-style + ReAct | 0.54 [0.43, 0.65] | 0.17 [0.08, 0.30] | 0.05 [0.00, 0.15] | 0.90 | 13.0% | 4.1 |
| 8 | GPT-4o-style + Naive loop | 0.56 [0.48, 0.65] | 0.15 [0.07, 0.27] | 0.05 [0.00, 0.15] | 0.94 | 13.0% | 4.4 |
| 9 | Llama-3-70B-style + Naive loop | 0.29 [0.18, 0.41] | 0.07 [0.00, 0.20] | 0.05 [0.00, 0.15] | 0.57 | 19.0% | 5.5 |

## pass^3 by domain

| Configuration | booking pass^3 | coding pass^3 | support pass^3 |
|---|---:|---:|---:|
| Claude-style + ReAct | 0.49 | 0.65 | 0.31 |
| GPT-4o-style + ReAct | 0.49 | 0.25 | 0.56 |
| GPT-4o-style + Plan-then-Act | 0.33 | 0.35 | 0.29 |
| Claude-style + Plan-then-Act | 0.30 | 0.18 | 0.29 |
| Llama-3-70B-style + Plan-then-Act | 0.47 | 0.25 | 0.00 |
| Claude-style + Naive loop | 0.31 | 0.17 | 0.09 |
| Llama-3-70B-style + ReAct | 0.29 | 0.15 | 0.09 |
| GPT-4o-style + Naive loop | 0.20 | 0.23 | 0.04 |
| Llama-3-70B-style + Naive loop | 0.14 | 0.07 | 0.01 |

## Failure mix (share of failed runs)

| Configuration | IGNORED_POLICY | LOOPING | GAVE_UP_EARLY | WRONG_TOOL | HALLUCINATED_ARG | HALLUCINATED_FACT | PARTIAL_SUCCESS | WRONG_STATE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Claude-style + ReAct | 43% | 0% | 24% | 0% | 5% | 24% | 5% | 0% |
| GPT-4o-style + ReAct | 36% | 0% | 11% | 0% | 7% | 32% | 14% | 0% |
| GPT-4o-style + Plan-then-Act | 38% | 0% | 16% | 0% | 6% | 22% | 12% | 6% |
| Claude-style + Plan-then-Act | 24% | 8% | 27% | 5% | 3% | 22% | 11% | 0% |
| Llama-3-70B-style + Plan-then-Act | 28% | 2% | 26% | 0% | 6% | 21% | 11% | 6% |
| Claude-style + Naive loop | 32% | 15% | 25% | 0% | 2% | 18% | 5% | 2% |
| Llama-3-70B-style + ReAct | 28% | 2% | 24% | 2% | 2% | 26% | 9% | 7% |
| GPT-4o-style + Naive loop | 30% | 5% | 23% | 7% | 2% | 23% | 9% | 2% |
| Llama-3-70B-style + Naive loop | 27% | 21% | 28% | 0% | 1% | 14% | 7% | 1% |
