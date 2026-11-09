"""Hybrid classifier: deterministic rules for structural failures, LLM judge for the rest."""

from __future__ import annotations

from agenteval.agents.llm import LLMClient
from agenteval.tasks.base import Task
from agenteval.tasks.registry import get_task
from agenteval.taxonomy.llm_judge import LLMJudgeClassifier, SimulatedJudgeLLM
from agenteval.taxonomy.rules import classify_rules
from agenteval.taxonomy.types import FailureMode, FailureReport
from agenteval.types import Trajectory

STRUCTURAL_MODES: frozenset[FailureMode] = frozenset({FailureMode.IGNORED_POLICY, FailureMode.LOOPING})
"""Modes the rules detect from hard evidence (audit log, exact repeats); rules win here."""


class FailureClassifier:
    """Classify trajectories with ``rules``, ``judge`` or ``hybrid`` strategy.

    ``hybrid`` trusts rules when they are structurally certain (policy audit log, exact
    repeated calls) and when the run succeeded (the evaluator is ground truth), and
    defers semantic distinctions (wrong tool vs hallucinated argument, partial vs wrong
    state, hallucinated facts) to the judge.
    """

    def __init__(self, strategy: str = "hybrid", judge_llm: LLMClient | None = None) -> None:
        if strategy not in ("rules", "judge", "hybrid"):
            raise ValueError("strategy must be 'rules', 'judge' or 'hybrid'")
        self.strategy = strategy
        self.judge = LLMJudgeClassifier(judge_llm or SimulatedJudgeLLM())

    def classify(self, traj: Trajectory, task: Task | None = None) -> FailureReport:
        task = task or get_task(traj.task_id)
        if self.strategy == "rules":
            return classify_rules(traj, task)
        if self.strategy == "judge":
            return self.judge.classify(traj, task)
        rules = classify_rules(traj, task)
        if rules.success or rules.primary in STRUCTURAL_MODES:
            rules.classifier = "hybrid(rules)"
            return rules
        judged = self.judge.classify(traj, task)
        judged.evidence = rules.evidence
        judged.classifier = "hybrid(judge)"
        if judged.primary is None:  # judge cannot overturn the evaluator's failure verdict
            judged.primary = rules.primary
            judged.success = False
        return judged

    def classify_many(self, trajs: list[Trajectory]) -> list[FailureReport]:
        return [self.classify(t) for t in trajs]
