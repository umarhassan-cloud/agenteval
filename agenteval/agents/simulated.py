"""Offline simulated LLM with calibrated, seeded fault injection.

**Demo mode.** ``SimulatedLLM`` lets the whole platform (harness, taxonomy, judge
calibration, pass^k, leaderboard) run end-to-end without API keys. It is *not* a
language model: it follows the task's reference plan and injects faults drawn from a
per-model :class:`FaultRates` profile, modulated by the calling agent's scaffold
(ReAct / plan-then-act / naive) and the task's difficulty.

The fault types map one-to-one onto the failure taxonomy, and every injected fault is
logged in ``injected_faults``. That log is the ground truth used to synthesise the
bundled human-label sample, so classifier accuracy can be measured honestly.

Profiles are named ``*-style`` deliberately: their rates are hand-set to reflect
qualitative behaviour reported for those model families on agentic benchmarks, not
measurements of the real models. Swap in :mod:`agenteval.agents.adapters` for real runs.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any

from agenteval.agents.llm import LLMClient, LLMRequest, LLMResponse, estimate_tokens, messages_text
from agenteval.envs.base import FREE_TEXT_KEYS, ToolSpec
from agenteval.types import Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from agenteval.tasks.base import Task


@dataclass(frozen=True)
class FaultRates:
    """Per-decision fault probabilities for a simulated model."""

    wrong_tool: float = 0.0
    hallucinated_arg: float = 0.0
    give_up: float = 0.0
    loop: float = 0.0
    premature_finish: float = 0.0
    policy_slip: float = 0.0
    hallucinated_fact: float = 0.0
    plan_omission: float = 0.0
    recover: float = 1.0
    """Probability of recovering correctly after a tool error."""
    loop_persist: float = 0.0
    """Probability of continuing a loop once in one."""

    def apply(self, multipliers: dict[str, float]) -> FaultRates:
        updates = {}
        for f in fields(self):
            m = multipliers.get(f.name, 1.0)
            updates[f.name] = min(0.98, getattr(self, f.name) * m)
        return replace(self, **updates)


MODEL_PROFILES: dict[str, FaultRates] = {
    "oracle": FaultRates(),
    "gpt-4o-style": FaultRates(
        wrong_tool=0.030, hallucinated_arg=0.040, give_up=0.015, loop=0.025, premature_finish=0.025,
        policy_slip=0.30, hallucinated_fact=0.050, plan_omission=0.10, recover=0.72, loop_persist=0.74,
    ),
    "claude-style": FaultRates(
        wrong_tool=0.025, hallucinated_arg=0.030, give_up=0.025, loop=0.020, premature_finish=0.020,
        policy_slip=0.20, hallucinated_fact=0.040, plan_omission=0.08, recover=0.76, loop_persist=0.70,
    ),
    "llama-70b-style": FaultRates(
        wrong_tool=0.060, hallucinated_arg=0.070, give_up=0.030, loop=0.045, premature_finish=0.045,
        policy_slip=0.45, hallucinated_fact=0.090, plan_omission=0.16, recover=0.52, loop_persist=0.80,
    ),
}

SCAFFOLD_MULTIPLIERS: dict[str, dict[str, float]] = {
    # Explicit reasoning + error reflection: fewer slips, strong recovery, short loops.
    "react": {"wrong_tool": 0.8, "hallucinated_arg": 0.75, "give_up": 0.7, "loop": 0.6,
              "premature_finish": 0.8, "policy_slip": 0.75, "hallucinated_fact": 0.7,
              "plan_omission": 0.0, "recover": 1.2, "loop_persist": 0.7},
    # Up-front plan: tool choice and policy checks improve, but plans can omit steps and
    # execution adapts less well to surprises.
    "planact": {"wrong_tool": 0.6, "hallucinated_arg": 0.9, "give_up": 0.8, "loop": 0.4,
                "premature_finish": 0.5, "policy_slip": 0.7, "hallucinated_fact": 0.8,
                "plan_omission": 1.0, "recover": 0.85, "loop_persist": 0.8},
    # No reasoning scaffold, no reflection.
    "naive": {"wrong_tool": 1.35, "hallucinated_arg": 1.3, "give_up": 1.3, "loop": 1.4,
              "premature_finish": 1.3, "policy_slip": 1.25, "hallucinated_fact": 1.3,
              "plan_omission": 0.0, "recover": 0.6, "loop_persist": 1.15},
}

PLAUSIBLE_FAKE_TOOLS: dict[str, list[str]] = {
    "issue_refund": ["refund_order", "process_refund"],
    "get_order": ["lookup_order", "fetch_order"],
    "get_customer_orders": ["list_orders"],
    "verify_identity": ["authenticate_customer", "verify_customer"],
    "change_flight": ["modify_reservation", "update_reservation"],
    "cancel_reservation": ["cancel_booking"],
    "book_flight": ["reserve_flight", "create_booking"],
    "book_hotel": ["reserve_hotel"],
    "get_reservation": ["lookup_reservation"],
    "search_flights": ["find_flights"],
    "write_file": ["edit_file", "apply_patch"],
    "run_tests": ["run_pytest", "pytest"],
    "read_file": ["open_file", "cat_file"],
    "escalate_ticket": ["transfer_ticket"],
    "create_ticket": ["open_ticket"],
}

AIRPORTS = ("SFO", "JFK", "LAX", "SEA", "BOS", "ORD")
GIVE_UP_ANSWER = (
    "I'm sorry, but I wasn't able to complete this request. "
    "Please contact a human agent for further help."
)
FABRICATED_ANSWER = "All set! I've taken care of that for you and everything has been processed."


def bump_digits(value: str) -> str:
    """Change the last digit of a string (R1003 -> R1004, 79.99 -> 79.90)."""
    for i in range(len(value) - 1, -1, -1):
        if value[i].isdigit():
            return value[:i] + str((int(value[i]) + 1) % 10) + value[i + 1:]
    return value


@dataclass
class _Episode:
    cursor: int = 0
    queue: deque = field(default_factory=deque)
    last: tuple[ToolCall, str, int | None] | None = None
    pending_fault: int | None = None
    prev_call: ToolCall | None = None
    looping: bool = False
    trap_checked: bool = False
    plan_skip: set[int] = field(default_factory=set)
    consecutive_errors: int = 0
    n_emitted: int = 0
    faults: list[dict[str, Any]] = field(default_factory=list)


class SimulatedLLM(LLMClient):
    """Seeded stand-in for a tool-calling LLM (see module docstring)."""

    def __init__(self, profile: str | FaultRates = "gpt-4o-style", name: str | None = None) -> None:
        if isinstance(profile, str):
            if profile not in MODEL_PROFILES:
                raise ValueError(f"Unknown profile '{profile}'. Choose from {sorted(MODEL_PROFILES)}")
            self.rates = MODEL_PROFILES[profile]
            self.name = name or profile
        else:
            self.rates = profile
            self.name = name or "custom-sim"
        self._task: Task | None = None
        self._rng = random.Random(0)
        self._ep = _Episode()

    # --------------------------------------------------------------- public
    def bind_task(self, task: Task, seed: int) -> None:
        self._task = task
        self._rng = random.Random(seed)
        self._ep = _Episode()

    @property
    def injected_faults(self) -> list[dict[str, Any]]:
        return list(self._ep.faults)

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self._task is None:
            raise RuntimeError("SimulatedLLM needs bind_task() before use (the harness does this).")
        rates = self.rates.apply(SCAFFOLD_MULTIPLIERS.get(request.scaffold, {}))
        if request.mode in ("plan", "replan"):
            resp = LLMResponse(content=self._plan(rates, replan=request.mode == "replan"))
        elif request.mode == "judge":
            raise ValueError("SimulatedLLM does not judge; use taxonomy.SimulatedJudgeLLM")
        else:
            resp = self._act(request, rates)
        resp.prompt_tokens = estimate_tokens(messages_text(request.messages)) + 60 * len(request.tools)
        resp.completion_tokens = estimate_tokens(resp.content) + 25 * len(resp.tool_calls)
        return resp

    # -------------------------------------------------------------- helpers
    def _p(self, rate: float) -> float:
        assert self._task is not None
        return min(0.95, rate * self._task.difficulty_factor)

    def _fault(self, kind: str, detail: str) -> None:
        self._ep.faults.append({"type": kind, "step": self._ep.n_emitted, "detail": detail,
                                "errored": False, "recovered": False})
        self._ep.pending_fault = len(self._ep.faults) - 1

    def _recover_errored(self) -> None:
        """A correct step succeeded: faults whose calls merely errored did no lasting damage."""
        for f in self._ep.faults:
            if f["errored"]:
                f["recovered"] = True

    def _recover_loops(self) -> None:
        for f in self._ep.faults:
            if f["type"] == "loop":
                f["recovered"] = True

    def _emit(self, call: dict[str, Any] | ToolCall, kind: str, thought: str = "") -> LLMResponse:
        ep = self._ep
        ep.n_emitted += 1
        if isinstance(call, ToolCall):
            tc = ToolCall(call.name, dict(call.arguments), id=f"sim_{ep.n_emitted}")
        else:
            tc = ToolCall(call["name"], dict(call["arguments"]), id=f"sim_{ep.n_emitted}")
        ep.last = (tc, kind, ep.pending_fault)
        ep.pending_fault = None
        return LLMResponse(content=thought or f"Next I will call {tc.name}.", tool_calls=[tc])

    def _final(self, text: str) -> LLMResponse:
        self._ep.n_emitted += 1
        self._ep.last = None
        return LLMResponse(content=text)

    @staticmethod
    def _find_result(messages: list[Message], call_id: str) -> ToolResult | None:
        for m in reversed(messages):
            if m.role == "tool" and m.tool_result is not None and m.tool_result.call_id == call_id:
                return m.tool_result
        return None

    # ----------------------------------------------------------------- plan
    def _plan(self, rates: FaultRates, replan: bool) -> str:
        assert self._task is not None
        ref = self._task.reference_actions
        env = self._task.make_env()
        if not replan and self._rng.random() < self._p(rates.plan_omission):
            candidates = [i for i, a in enumerate(ref) if i > 0 and (s := env.get_spec(a["name"])) and s.mutating]
            if candidates:
                idx = self._rng.choice(candidates)
                self._ep.plan_skip.add(idx)
                self._fault("plan_omission", f"plan omitted step {idx} ({ref[idx]['name']})")
        steps = [a["name"] for i, a in enumerate(ref) if i not in self._ep.plan_skip and i >= self._ep.cursor]
        lines = [f"{n}. Call {name}" for n, name in enumerate(steps, start=1)]
        lines.append(f"{len(steps) + 1}. Report the outcome to the user.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ act
    def _act(self, req: LLMRequest, rates: FaultRates) -> LLMResponse:
        assert self._task is not None
        task, ep, rng = self._task, self._ep, self._rng
        ref = task.reference_actions
        mutating = {t.name: t.mutating for t in req.tools}

        # 1. Digest the observation for our previous call.
        last = self._find_result(req.messages, ep.last[0].id) if ep.last is not None else None
        if ep.last is not None and last is not None:
            call, kind, fault_idx = ep.last
            ep.last = None
            ep.prev_call = call
            if last.ok:
                ep.consecutive_errors = 0
                if kind in ("ref", "halluc", "wrong"):
                    ep.cursor += 1
                if kind == "ref":
                    self._recover_errored()
            else:
                ep.consecutive_errors += 1
                if fault_idx is not None:
                    ep.faults[fault_idx]["errored"] = True
                if not ep.looping:
                    p_recover = rates.recover / (1.0 + 0.5 * (ep.consecutive_errors - 1))
                    if rng.random() >= p_recover:
                        r = rng.random()
                        if r < 0.5:
                            self._fault("loop", f"repeats failing call {call.name}")
                            ep.looping = True
                            return self._emit(call, "loop", "Let me try that again.")
                        if r < 0.8:
                            self._fault("give_up", f"gave up after error from {call.name}")
                            return self._final(GIVE_UP_ANSWER)
                        if ep.cursor < len(ref):
                            self._fault("wrong_tool", f"switched to a wrong tool after {call.name} failed")
                            return self._emit(self._wrong_tool(ref[ep.cursor], req.tools), "wrong")

        # 2. Loop persistence.
        if ep.looping:
            if ep.prev_call is not None and rng.random() < rates.loop_persist:
                return self._emit(ep.prev_call, "loop", "Let me try that again.")
            ep.looping = False
            self._recover_loops()

        # 3. Queued actions (policy-trap shortcuts).
        if ep.queue:
            call_d, kind = ep.queue.popleft()
            return self._emit(call_d, kind)

        # 4. Policy temptation.
        trap = task.trap
        if trap is not None and not ep.trap_checked and ep.cursor == trap.at:
            ep.trap_checked = True
            if rng.random() < min(0.95, rates.policy_slip):
                self._fault("ignored_policy", trap.description)
                ep.cursor = trap.resume
                ep.queue.extend((a, "trap") for a in trap.actions)
                if ep.queue:
                    call_d, kind = ep.queue.popleft()
                    return self._emit(call_d, kind)

        # 5. Steps the plan forgot.
        while ep.cursor in ep.plan_skip:
            ep.cursor += 1

        # 6. Done: answer (possibly with a hallucinated fact).
        if ep.cursor >= len(ref):
            if rng.random() < self._p(rates.hallucinated_fact):
                self._fault("hallucinated_fact", "final answer misstates an outcome")
                return self._final(self._corrupt_answer(task.reference_answer, task.required_facts))
            return self._final(task.reference_answer)

        # 7. Per-step fault draw for the next reference action.
        step = ref[ep.cursor]
        remaining_mutating = any(mutating.get(a["name"], False) for a in ref[ep.cursor:])
        can_loop = ep.prev_call is not None and not mutating.get(ep.prev_call.name, False)
        corrupt = self._corrupt_args(step["arguments"])
        options: list[tuple[str, float]] = [
            ("give_up", self._p(rates.give_up)),
            ("premature_finish", self._p(rates.premature_finish) if ep.cursor > 0 and remaining_mutating else 0.0),
            ("wrong_tool", self._p(rates.wrong_tool)),
            ("hallucinated_arg", self._p(rates.hallucinated_arg) if corrupt is not None else 0.0),
            ("loop", self._p(rates.loop) if can_loop else 0.0),
        ]
        r, acc = rng.random(), 0.0
        for kind, prob in options:
            acc += prob
            if r >= acc:
                continue
            if kind == "give_up":
                self._fault("give_up", f"gave up before {step['name']}")
                return self._final(GIVE_UP_ANSWER)
            if kind == "premature_finish":
                self._fault("premature_finish", f"declared success before {step['name']}")
                return self._final(task.reference_answer)
            if kind == "wrong_tool":
                self._fault("wrong_tool", f"called a wrong tool instead of {step['name']}")
                return self._emit(self._wrong_tool(step, req.tools), "wrong")
            if kind == "hallucinated_arg":
                key, args = corrupt  # type: ignore[misc]
                self._fault("hallucinated_arg", f"{step['name']}.{key}={args[key]!r}")
                return self._emit({"name": step["name"], "arguments": args}, "halluc")
            if kind == "loop":
                self._fault("loop", f"repeats {ep.prev_call.name}")  # type: ignore[union-attr]
                ep.looping = True
                return self._emit(ep.prev_call, "loop", "Let me double-check that.")  # type: ignore[arg-type]
        return self._emit(step, "ref", f"Based on what I know so far, I should call {step['name']}.")

    # ---------------------------------------------------------- corruptions
    def _wrong_tool(self, step: dict[str, Any], tools: list[ToolSpec]) -> dict[str, Any]:
        rng = self._rng
        others = [t for t in tools if t.name != step["name"]]
        if not others or rng.random() < 0.35:
            fakes = PLAUSIBLE_FAKE_TOOLS.get(step["name"], [f"{step['name']}_v2"])
            return {"name": rng.choice(fakes), "arguments": dict(step["arguments"])}
        overlapping = [t for t in others if set(t.parameters["properties"]) & set(step["arguments"])]
        spec = rng.choice(overlapping or others)
        args: dict[str, Any] = {}
        for pname in spec.parameters["properties"]:
            if pname in step["arguments"]:
                args[pname] = step["arguments"][pname]
            elif pname in spec.required:
                args[pname] = "unknown"
        return {"name": spec.name, "arguments": args}

    def _corrupt_args(self, args: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        candidates = [
            k for k, v in args.items()
            if k not in FREE_TEXT_KEYS and k != "content"
            and ((isinstance(v, (int, float)) and not isinstance(v, bool))
                 or (isinstance(v, str) and (any(c.isdigit() for c in v) or "/" in v or "@" in v or v in AIRPORTS)))
        ]
        if not candidates:
            return None
        key = self._rng.choice(sorted(candidates))
        v = args[key]
        if isinstance(v, int) and not isinstance(v, bool):
            new: Any = v + 1
        elif isinstance(v, float):
            new = round(v * self._rng.choice([0.5, 0.9, 1.1]), 2)
        elif v in AIRPORTS:
            new = self._rng.choice([a for a in AIRPORTS if a != v])
        elif "@" in v:
            new = v.split("@")[0] + "@example.org"
        elif any(c.isdigit() for c in v):
            new = bump_digits(v)
        else:
            new = "lib/" + v.split("/", 1)[1] if v.startswith("src/") else "src/" + v
        out = dict(args)
        out[key] = new
        return key, out

    def _corrupt_answer(self, answer: str, facts: list[str]) -> str:
        numeric = [f for f in facts if any(c.isdigit() for c in f) and f in answer]
        if numeric:
            fact = self._rng.choice(numeric)
            return answer.replace(fact, bump_digits(fact))
        return FABRICATED_ANSWER
