"""AgentEval: reliability and evaluation platform for tool-using AI agents.

Subpackages:
    envs      Stateful simulated backends (booking, support, coding).
    tasks     Task suites with ground-truth success criteria and policy constraints.
    agents    Reference agents (ReAct, plan-then-act, naive) and LLM backends.
    harness   Episode loop, evaluator and suite runner.
    taxonomy  Failure classification (rule-based, LLM-as-judge, hybrid).
    judge     Human labels, Cohen's kappa and judge calibration reports.
    metrics   pass@k, pass^k, bootstrap confidence intervals, aggregation.
"""

__version__ = "0.1.0"
