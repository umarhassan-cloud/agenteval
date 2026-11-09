# Leaderboard

- `index.html` is a static dashboard with the results JSON embedded. Open it directly in a browser;
  no server or network access is needed.
- `RESULTS.md` contains the same results as Markdown tables.
- `template.html` is the dashboard source. `scripts/generate_leaderboard.py` injects
  `data/leaderboard_results.json` into it.

Regenerate everything (900 episodes, about 35 s):

```bash
python scripts/generate_leaderboard.py --trials 5
```

Compare real models by passing adapter model strings:

```bash
python scripts/generate_leaderboard.py --models openai:gpt-4o,anthropic:claude-sonnet-4-5 --agents react,naive --trials 3
```

The dashboard shows a sortable leaderboard with pass^k CIs (overall or per domain), pass^k decay
curves, a ReAct vs plan-then-act vs naive comparison per model, the failure-taxonomy mix per
configuration, and a per-domain pass^3 heatmap.
