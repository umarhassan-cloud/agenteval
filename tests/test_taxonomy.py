"""Tests for the failure taxonomy classifiers."""

from __future__ import annotations

from agenteval.agents import FaultRates, NaiveAgent, SimulatedLLM
from agenteval.harness import run_episode
from agenteval.tasks import get_task
from agenteval.taxonomy import FailureClassifier, FailureMode, classify_rules, parse_verdict
from agenteval.types import EvalResult, Step, ToolCall, ToolResult, Trajectory


def _traj(task_id: str, steps: list[Step], *, final: str | None, termination: str, ev: EvalResult,
          violations: list[str] | None = None) -> Trajectory:
    task = get_task(task_id)
    return Trajectory(run_id="r", task_id=task_id, domain=task.domain, agent="a", model="m", trial=0, seed=0,
                      steps=steps, final_answer=final, termination=termination,  # type: ignore[arg-type]
                      policy_violations=violations or [], eval=ev)


def _step(i: int, name: str, args: dict, *, error: str | None = None, output=None) -> Step:
    tc = ToolCall(name, args, id=f"c{i}")
    return Step(index=i, tool_call=tc, result=ToolResult(tc.id, name, output=output, error=error))


def test_success_has_no_failure_mode():
    ev = EvalResult(True, True, True, True, 1.0)
    rep = classify_rules(_traj("booking-001", [], final="ok", termination="final_answer", ev=ev), get_task("booking-001"))
    assert rep.success and rep.primary is None and rep.label == "SUCCESS"


def test_policy_violation_wins():
    ev = EvalResult(False, False, True, False, 0.0)
    t = _traj("booking-003", [_step(0, "change_flight", {"reservation_id": "R1002", "new_flight_number": "AE421"},
                                    output={})],
              final="done", termination="final_answer", ev=ev, violations=["NO_BASIC_ECONOMY_CHANGES: x"])
    assert classify_rules(t, get_task("booking-003")).primary is FailureMode.IGNORED_POLICY


def test_looping_detected():
    steps = [_step(i, "get_reservation", {"reservation_id": "R1030"}, error="Reservation 'R1030' not found")
             for i in range(6)]
    ev = EvalResult(False, False, False, True, 0.0, missing_facts=["AE310"])
    rep = classify_rules(_traj("booking-001", steps, final=None, termination="max_steps", ev=ev), get_task("booking-001"))
    assert rep.primary is FailureMode.LOOPING
    assert any(e.mode is FailureMode.HALLUCINATED_ARG for e in rep.evidence)  # R1030 is ungrounded


def test_gave_up_detected():
    ev = EvalResult(False, False, False, True, 0.0, missing_facts=["R2001"])
    t = _traj("booking-002", [], final="I'm sorry, but I wasn't able to complete this request.",
              termination="final_answer", ev=ev)
    assert classify_rules(t, get_task("booking-002")).primary is FailureMode.GAVE_UP_EARLY


def test_hallucinated_fact_when_state_correct_but_answer_wrong():
    ev = EvalResult(False, True, False, True, 1.0, missing_facts=["320"])
    t = _traj("booking-004", [], final="Cancelled; refund of $330.", termination="final_answer", ev=ev)
    assert classify_rules(t, get_task("booking-004")).primary is FailureMode.HALLUCINATED_FACT


def test_hallucinated_arg_on_mutation():
    steps = [_step(0, "issue_refund", {"order_id": "O-8806", "amount": 22.5, "reason": "x"}, output={})]
    ev = EvalResult(False, False, True, True, 0.0)
    rep = classify_rules(_traj("support-006", steps, final="done", termination="final_answer", ev=ev),
                         get_task("support-006"))
    assert rep.primary is FailureMode.HALLUCINATED_ARG


def test_partial_success():
    steps = [_step(0, "change_flight", {"reservation_id": "R1001", "new_flight_number": "AE311"}, output={})]
    ev = EvalResult(False, False, False, True, 0.5, missing_facts=["HB501"])
    rep = classify_rules(_traj("booking-006", steps, final="Moved.", termination="final_answer", ev=ev),
                         get_task("booking-006"))
    assert rep.primary is FailureMode.PARTIAL_SUCCESS


def test_wrong_tool_mutation():
    steps = [_step(0, "cancel_reservation", {"reservation_id": "R1001"}, output={})]
    ev = EvalResult(False, False, False, True, 0.0)
    rep = classify_rules(_traj("booking-006", steps, final="ok", termination="final_answer", ev=ev),
                         get_task("booking-006"))
    assert rep.primary is FailureMode.WRONG_TOOL


def test_parse_verdict():
    v = parse_verdict('Sure! {"label": "looping", "secondary": ["GAVE_UP_EARLY", "nope"], "confidence": 1.7}')
    assert v == {"label": "LOOPING", "secondary": ["GAVE_UP_EARLY"], "confidence": 1.0, "rationale": ""}
    assert parse_verdict("no json here") is None
    assert parse_verdict('{"label": "NOT_A_LABEL"}') is None


def test_classifiers_on_real_episode():
    slipper = SimulatedLLM(FaultRates(policy_slip=0.98), name="slipper")
    traj = run_episode(NaiveAgent(slipper), get_task("support-002"))
    for strategy in ("rules", "judge", "hybrid"):
        rep = FailureClassifier(strategy).classify(traj)
        assert not rep.success
    assert FailureClassifier("hybrid").classify(traj).primary is FailureMode.IGNORED_POLICY


def test_hybrid_never_overturns_failed_run():
    giver = SimulatedLLM(FaultRates(give_up=0.95), name="giver")
    traj = run_episode(NaiveAgent(giver), get_task("booking-006"))
    assert not traj.success
    assert FailureClassifier("hybrid").classify(traj).label != "SUCCESS"
