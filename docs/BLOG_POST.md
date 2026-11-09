# Your agent works. Does it work *every time*?

*A technical write-up of AgentEval, an open-source harness for measuring agent reliability,
diagnosing failures, and checking that the judge you use for diagnosis agrees with humans.*

**Draft. Umar Hassan**

---

## The demo problem

Every agent demo is a survivorship-bias exercise. You run the task, it works, you record it.
What you don't show is the second run, where the agent invented a reservation ID, or the third,
where an insistent customer talked it into a refund that policy forbids.

In production, the same request arrives thousands of times. The question that matters is not
"*can* it do this?" but "*will* it do this, every time?" Most agent evaluations answer the first
question and report it as if it were the second.

I built AgentEval to answer the second question, and two follow-ups that teams hit right after:
*why* did the failures happen, and can I trust the automated judge that tells me why?

## pass@k measures capability; pass^k measures reliability

pass@k comes from code generation (Chen et al., 2021). It is the probability that *at least one* of
k attempts succeeds. That suits a developer who can rerun a code suggestion. It does not suit a
customer whose flight is being changed.

τ-bench (Yao et al., 2024) introduced **pass^k**: the probability that *all* k attempts succeed.
For a task with n trials and c successes, both have unbiased estimators:

```
pass@k = 1 − C(n−c, k) / C(n, k)
pass^k =     C(c,   k) / C(n, k)
```

pass@k rises with k and pass^k falls. The distance between them is the reliability gap. In
AgentEval's simulated benchmark, the GPT-4o-style ReAct configuration has:

| | k=1 | k=3 | k=5 |
|---|---:|---:|---:|
| pass@k | 0.72 | 0.95 | — |
| pass^k | 0.72 | **0.44** | **0.30** |

The same agent solves 95% of tasks "eventually" and 44% "reliably". If you ship on the first number,
your customers live with the second.

Two implementation details matter more than they look:

1. **Use the combinatorial estimator, not "the first k trials".** With n=5 trials, C(c,k)/C(n,k)
   uses all five. Taking the first three throws away data and adds variance.
2. **Bootstrap over tasks, not trials.** Five attempts at one task are correlated. Resampling
   trials treats them as independent and gives confidence intervals that are too narrow. With 20
   tasks, honest pass^3 CIs are about ±0.17. That is wide, and it should be reported.

## Measuring success by state, not by text

For an agent that *acts*, comparing its final message to a gold answer is the wrong check. Following
τ-bench, each AgentEval task has a seed database and a reference solution. The evaluator replays the
reference on a fresh copy, hashes the resulting state, and compares it with the agent's end state.
Free-text fields like refund reasons are excluded from the hash. A run succeeds only if:

- the end state matches,
- the final answer contains every required fact (the actual reservation ID, the actual refund
  amount), and
- the environment's **policy audit log is empty**.

That last condition is deliberate. The simulated backends don't enforce policy. `issue_refund` will
refund a 56-day-old order, just as most real payment APIs would. The violation is logged and the
run fails. Several tasks contain a **policy trap**: a customer request whose easiest fulfilment
breaks a rule (changing a basic-economy ticket, refunding $649 directly instead of escalating
because of a $500 limit, editing the tests instead of the code).

## A failure taxonomy you can act on

"Success rate dropped 3 points" tells you nothing about what to fix. AgentEval classifies every
failed run into one root-cause label:

`IGNORED_POLICY · LOOPING · GAVE_UP_EARLY · WRONG_TOOL · HALLUCINATED_ARG · HALLUCINATED_FACT ·
PARTIAL_SUCCESS · WRONG_STATE`

The important word is *root*. An agent that calls a non-existent tool, reads the error and recovers
has not failed because of a wrong tool. If it later misquotes the refund amount, the label is
HALLUCINATED_FACT. The rule classifier records recovered errors as evidence and never makes them the
primary label. Get this wrong and every run with an early stumble gets misattributed, and your
dashboard points engineers at the wrong problem.

On the simulated leaderboard, the taxonomy shows something that aggregate success rates hide:

- **Scaffolding mostly removes mechanical failures.** LOOPING causes 15–21% of naive-loop failures
  and ≤2% of ReAct failures. ReAct's reflection step (on a tool error, the model is told to diagnose
  before retrying) breaks loops.
- **What remains is policy.** In the two best configurations, IGNORED_POLICY causes 36–43% of
  failures. Better reasoning does not make an agent better at refusing a pushy customer. That needs
  separate work: policy-specific prompting, guard tools, or pre-action checks.

## Who judges the judge?

Rule-based classification covers much of the taxonomy. Some distinctions are semantic, though:
did the agent call the wrong tool, or the right tool with a made-up argument? Teams increasingly
use an LLM judge for this, and almost none of them measure whether the judge agrees with people.

AgentEval treats judge quality as a measurement with error bars:

- a versioned JSONL schema for human labels (one line per run per annotator, with confidence,
  failure step and notes) and a validator,
- adjudication of multiple annotators,
- **Cohen's κ** between humans and each automated classifier, with a bootstrap CI, a confusion
  matrix and per-class precision/recall.

On the bundled 50-run sample, the hybrid classifier (rules for structural failures, a judge for
semantic ones) reaches **κ = 0.72 [0.59, 0.86]** over 9 labels, which counts as "substantial"
agreement. Its pass/fail κ is 1.00. The per-class table is more informative than the headline:
rules alone have **recall 0.20 on HALLUCINATED_ARG** because they label those runs as
HALLUCINATED_FACT. That one cell justifies the judge, and it tells you which judge prompts to
improve first.

Report **pass/fail κ** separately from **multi-class κ**. A judge can be perfect at "did it fail?"
and mediocre at "why?". If you only use the judge to gate releases, the first number is the one
that matters. If you use it to prioritise engineering work, the second is.

## Being honest about what's simulated

To make the whole pipeline reproducible without API keys, AgentEval ships a `SimulatedLLM`. It
follows the reference plan and injects faults at rates set per model profile and per agent design.
The `gpt-4o-style`, `claude-style` and `llama-70b-style` profiles are **hand-set**, not
measurements of those APIs, and the 50 "human" labels are synthesised from the simulator's
fault log plus annotator noise. The leaderboard numbers above therefore show the *methodology*, not
a ranking of real models. The OpenAI and Anthropic adapters use the same interface, and getting
real numbers is a one-flag change.

Simulating first was the right call for one reason. It gave me ground truth. Because every
injected fault is logged, I could measure how often the rule classifier recovers the true root
cause before trusting it on real transcripts, where no such log exists.

## What I'd tell a team shipping an agent

1. **Report pass^k, with CIs, at the k that matches your traffic.** pass^1 is a capability number.
2. **Score state, not strings.** Replay a reference solution and diff the database.
3. **Make policy violations observable.** Don't rely on the backend to block them; log them.
4. **Classify root causes, not symptoms,** and keep recovered errors out of the primary label.
5. **Calibrate your judge against humans** before letting it drive decisions, and look at the
   confusion matrix, not just κ.

Code, data and the static leaderboard: `github.com/umarhassan/agenteval`.

### References
- Yao et al. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* 2024.
- Kapoor et al. *AI Agents That Matter.* Princeton, 2024.
- Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* 2023.
- Chen et al. *Evaluating Large Language Models Trained on Code.* 2021.
- Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* 2023.
- Cohen. *A Coefficient of Agreement for Nominal Scales.* 1960. Landis & Koch, 1977.
