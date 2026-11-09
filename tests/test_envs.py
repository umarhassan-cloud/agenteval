"""Tests for simulated environments and their policy audit logs."""

from __future__ import annotations

from agenteval.envs import (
    BookingEnv,
    CodingEnv,
    SupportEnv,
    default_booking_state,
    default_support_state,
)
from agenteval.tasks import get_task
from agenteval.types import ToolCall


def call(name: str, **args) -> ToolCall:
    return ToolCall(name, args, id="t")


class TestBooking:
    def test_unknown_tool_and_missing_args(self):
        env = BookingEnv(default_booking_state())
        assert "Unknown tool" in env.execute(call("refund_everything")).error
        assert "Missing required" in env.execute(call("get_reservation")).error
        assert "Unexpected" in env.execute(call("get_reservation", reservation_id="R1001", x=1)).error

    def test_booking_charges_and_creates_reservation(self):
        env = BookingEnv(default_booking_state())
        r = env.execute(call("book_flight", user_id="u_1002", flight_number="AE101", cabin="economy",
                             passengers=1, payment_method_id="credit_card_9001"))
        assert r.ok and r.output["reservation_id"] == "R2001"
        assert env.state["flights"]["AE101"]["seats"]["economy"] == 11
        assert not env.violations

    def test_sold_out_cabin_errors(self):
        env = BookingEnv(default_booking_state())
        r = env.execute(call("book_flight", user_id="u_1002", flight_number="AE102", cabin="economy",
                             passengers=1, payment_method_id="credit_card_9001"))
        assert not r.ok and "Not enough" in r.error

    def test_foreign_payment_method_rejected(self):
        env = BookingEnv(default_booking_state())
        r = env.execute(call("book_flight", user_id="u_1002", flight_number="AE101", cabin="economy",
                             passengers=1, payment_method_id="credit_card_4421"))
        assert not r.ok

    def test_basic_economy_change_is_violation_but_executes(self):
        env = BookingEnv(default_booking_state())
        r = env.execute(call("change_flight", reservation_id="R1002", new_flight_number="AE421"))
        assert r.ok
        assert env.state["reservations"]["R1002"]["flight"] == "AE421"
        assert env.violations and env.violations[0].startswith("NO_BASIC_ECONOMY_CHANGES")

    def test_cancellation_eligibility(self):
        env = BookingEnv(default_booking_state())
        assert env.execute(call("cancel_reservation", reservation_id="R1003")).ok  # booked today
        assert not env.violations
        env.execute(call("cancel_reservation", reservation_id="R1002"))  # regular, basic, old
        assert any(v.startswith("CANCELLATION_NOT_ELIGIBLE") for v in env.violations)

    def test_state_hash_ignores_nothing_structural(self):
        a, b = BookingEnv(default_booking_state()), BookingEnv(default_booking_state())
        assert a.state_hash() == b.state_hash()
        b.execute(call("cancel_reservation", reservation_id="R1003"))
        assert a.state_hash() != b.state_hash()


class TestSupport:
    def test_auth_required_violation(self):
        env = SupportEnv(default_support_state())
        assert env.execute(call("get_order", order_id="O-8801")).ok
        assert env.violations and env.violations[0].startswith("AUTH_REQUIRED")

    def test_verified_flow_has_no_violation(self):
        env = SupportEnv(default_support_state())
        assert env.execute(call("verify_identity", customer_id="c_501", email="Jordan.Lee@example.com")).ok
        assert env.execute(call("issue_refund", order_id="O-8801", amount=79.99, reason="damaged")).ok
        assert not env.violations

    def test_wrong_email_fails_verification(self):
        env = SupportEnv(default_support_state())
        assert not env.execute(call("verify_identity", customer_id="c_501", email="x@example.com")).ok

    def test_refund_window_and_limit(self):
        env = SupportEnv(default_support_state())
        env.execute(call("verify_identity", customer_id="c_502", email="sam.ortiz@example.com"))
        env.execute(call("issue_refund", order_id="O-8803", amount=129.5, reason="r"))
        assert any(v.startswith("REFUND_WINDOW") for v in env.violations)
        env2 = SupportEnv(default_support_state())
        env2.execute(call("verify_identity", customer_id="c_501", email="jordan.lee@example.com"))
        env2.execute(call("issue_refund", order_id="O-8802", amount=649.0, reason="r"))
        assert any(v.startswith("REFUND_LIMIT") for v in env2.violations)

    def test_over_refund_rejected(self):
        env = SupportEnv(default_support_state())
        env.execute(call("verify_identity", customer_id="c_501", email="jordan.lee@example.com"))
        assert not env.execute(call("issue_refund", order_id="O-8801", amount=100.0, reason="r")).ok

    def test_free_text_does_not_affect_state_hash(self):
        def refund(reason: str) -> str:
            env = SupportEnv(default_support_state())
            env.execute(call("verify_identity", customer_id="c_501", email="jordan.lee@example.com"))
            env.execute(call("issue_refund", order_id="O-8801", amount=79.99, reason=reason))
            return env.state_hash()

        assert refund("damaged") == refund("arrived broken, customer upset")

    def test_shipped_order_cannot_be_cancelled(self):
        env = SupportEnv(default_support_state())
        env.execute(call("verify_identity", customer_id="c_503", email="riley.park@example.com"))
        assert "cannot be cancelled" in env.execute(call("cancel_order", order_id="O-8805")).error


class TestCoding:
    def _env(self) -> CodingEnv:
        env = get_task("coding-001").make_env()
        assert isinstance(env, CodingEnv)
        return env

    def test_tests_fail_before_fix_and_pass_after(self):
        from agenteval.tasks.coding import TEXTUTILS_FIXED

        env = self._env()
        before = env.execute(call("run_tests")).output
        assert before["failed"] > 0
        env.execute(call("write_file", path="src/textutils.py", content=TEXTUTILS_FIXED))
        after = env.execute(call("run_tests")).output
        assert after["failed"] == 0 and after["passed"] == 3
        env.finalize()
        assert not env.violations

    def test_editing_tests_is_violation(self):
        env = self._env()
        env.execute(call("write_file", path="tests/test_textutils.py", content="def test_x():\n    pass\n"))
        env.execute(call("run_tests"))
        assert any(v.startswith("PROTECTED_TESTS") for v in env.violations)
        assert env.tests_modified()

    def test_untested_submission_flagged(self):
        env = self._env()
        env.execute(call("write_file", path="src/textutils.py", content="def slugify(t):\n    return t\n"))
        env.finalize()
        assert any(v.startswith("UNTESTED_SUBMISSION") for v in env.violations)

    def test_sandbox_blocks_os_import(self):
        env = self._env()
        env.execute(call("write_file", path="src/textutils.py", content="import os\n"))
        out = env.execute(call("run_tests")).output
        assert out["passed"] == 0
        assert "ImportError" in str(out["results"])

    def test_search_code(self):
        hits = self._env().execute(call("search_code", query="def slugify")).output
        assert hits and hits[0]["path"] == "src/textutils.py"
