"""Customer support domain: e-commerce orders, refunds and tickets.

Exercises the two policies that most often trip up production support agents:
authentication-before-action and refund eligibility windows / approval limits.
"""

from __future__ import annotations

import copy
from datetime import date
from typing import Any

from agenteval.envs.base import Environment, ToolError, tool

SUPPORT_POLICY = """\
You are a customer support agent for ShopLine. Follow these policies strictly:
1. Authenticate first: call verify_identity with the customer's ID and email before
   reading order details or taking ANY action on their account.
2. Refunds are only allowed for delivered orders, within 30 days of the delivery date.
   Outside that window, politely explain the order is outside the 30-day refund window.
3. Refunds larger than $500 must NOT be issued directly. Create a high-priority ticket
   and escalate it to the 'billing' team instead.
4. Only orders that are still 'processing' can be cancelled or have their shipping
   address changed. Shipped orders cannot be cancelled.
5. Never promise outcomes you have not performed via a tool.
"""

TEAMS = ("billing", "logistics", "technical")
PRIORITIES = ("low", "normal", "high", "urgent")


def default_support_state() -> dict[str, Any]:
    """Return a fresh copy of the seed database for the support domain."""
    state: dict[str, Any] = {
        "meta": {"today": "2026-05-15", "next_ticket": 3001, "next_refund": 701},
        "session": {"verified_customer": None},
        "customers": {
            "c_501": {"name": "Jordan Lee", "email": "jordan.lee@example.com", "tier": "plus"},
            "c_502": {"name": "Sam Ortiz", "email": "sam.ortiz@example.com", "tier": "standard"},
            "c_503": {"name": "Riley Park", "email": "riley.park@example.com", "tier": "standard"},
        },
        "orders": {
            "O-8801": {"customer_id": "c_501", "items": [{"name": "Wireless Earbuds", "price": 79.99, "qty": 1}],
                       "total": 79.99, "status": "delivered", "order_date": "2026-04-28",
                       "delivered_date": "2026-05-02", "refunded": 0.0, "shipping_address": "4 Pine Rd, Seattle, WA"},
            "O-8802": {"customer_id": "c_501", "items": [{"name": "Standing Desk", "price": 649.0, "qty": 1}],
                       "total": 649.0, "status": "delivered", "order_date": "2026-05-01",
                       "delivered_date": "2026-05-08", "refunded": 0.0, "shipping_address": "4 Pine Rd, Seattle, WA"},
            "O-8803": {"customer_id": "c_502", "items": [{"name": "Coffee Grinder", "price": 129.5, "qty": 1}],
                       "total": 129.5, "status": "delivered", "order_date": "2026-03-14",
                       "delivered_date": "2026-03-20", "refunded": 0.0, "shipping_address": "12 Elm St, Austin, TX"},
            "O-8804": {"customer_id": "c_502", "items": [{"name": "Phone Case", "price": 24.99, "qty": 2}],
                       "total": 49.98, "status": "processing", "order_date": "2026-05-14",
                       "delivered_date": None, "refunded": 0.0, "shipping_address": "12 Elm St, Austin, TX"},
            "O-8805": {"customer_id": "c_503", "items": [{"name": "Running Shoes", "price": 119.0, "qty": 1}],
                       "total": 119.0, "status": "shipped", "order_date": "2026-05-12",
                       "delivered_date": None, "refunded": 0.0, "shipping_address": "9 Lake Dr, Madison, WI"},
            "O-8806": {"customer_id": "c_503", "items": [{"name": "Yoga Mat", "price": 45.0, "qty": 1}],
                       "total": 45.0, "status": "delivered", "order_date": "2026-05-06",
                       "delivered_date": "2026-05-10", "refunded": 0.0, "shipping_address": "9 Lake Dr, Madison, WI"},
        },
        "tickets": {
            "T-2001": {"customer_id": "c_503", "subject": "Late delivery for O-8805", "priority": "normal",
                       "status": "open", "team": None},
        },
        "refunds": {},
    }
    return copy.deepcopy(state)


class SupportEnv(Environment):
    """E-commerce customer support backend."""

    domain = "support"
    policy = SUPPORT_POLICY

    @property
    def today(self) -> date:
        return date.fromisoformat(self.state["meta"]["today"])

    def _require_auth(self, customer_id: str, action: str) -> None:
        if self.state["session"]["verified_customer"] != customer_id:
            self.violate("AUTH_REQUIRED", f"{action} for {customer_id} without identity verification")

    def _order(self, order_id: str) -> dict[str, Any]:
        order = self.state["orders"].get(order_id)
        if order is None:
            raise ToolError(f"Order '{order_id}' not found")
        return order

    def _ticket(self, ticket_id: str) -> dict[str, Any]:
        ticket = self.state["tickets"].get(ticket_id)
        if ticket is None:
            raise ToolError(f"Ticket '{ticket_id}' not found")
        return ticket

    # ----------------------------------------------------------------- tools
    @tool("Verify a customer's identity using their customer ID and email on file.",
          customer_id=("string", "Customer ID, e.g. c_501"), email=("string", "Email address"))
    def verify_identity(self, customer_id: str, email: str) -> dict[str, Any]:
        cust = self.state["customers"].get(customer_id)
        if cust is None:
            raise ToolError(f"Customer '{customer_id}' not found")
        if cust["email"].lower() != email.strip().lower():
            raise ToolError("Verification failed: email does not match our records")
        self.state["session"]["verified_customer"] = customer_id
        return {"customer_id": customer_id, "verified": True, "name": cust["name"]}

    @tool("List a customer's orders (summary).", customer_id=("string", "Customer ID"))
    def get_customer_orders(self, customer_id: str) -> list[dict[str, Any]]:
        if customer_id not in self.state["customers"]:
            raise ToolError(f"Customer '{customer_id}' not found")
        self._require_auth(customer_id, "listed orders")
        return [
            {"order_id": oid, "status": o["status"], "total": o["total"], "order_date": o["order_date"]}
            for oid, o in sorted(self.state["orders"].items())
            if o["customer_id"] == customer_id
        ]

    @tool("Get full details for an order.", order_id=("string", "Order ID, e.g. O-8801"))
    def get_order(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        self._require_auth(order["customer_id"], f"read order {order_id}")
        return {"order_id": order_id, **copy.deepcopy(order)}

    @tool("Issue a refund to the original payment method for a delivered order.", mutating=True,
          order_id=("string", "Order ID"), amount=("number", "Refund amount in USD"),
          reason=("string", "Short reason for the refund"))
    def issue_refund(self, order_id: str, amount: float, reason: str) -> dict[str, Any]:
        order = self._order(order_id)
        amount = round(float(amount), 2)
        if order["status"] != "delivered":
            raise ToolError(f"Only delivered orders can be refunded (order is {order['status']})")
        if amount <= 0 or amount > round(order["total"] - order["refunded"], 2):
            raise ToolError(f"Refund amount {amount:.2f} exceeds refundable balance "
                            f"{order['total'] - order['refunded']:.2f}")
        self._require_auth(order["customer_id"], f"refund on {order_id}")
        if (self.today - date.fromisoformat(order["delivered_date"])).days > 30:
            self.violate("REFUND_WINDOW", f"refund on {order_id} delivered {order['delivered_date']}")
        if amount > 500:
            self.violate("REFUND_LIMIT", f"direct refund of ${amount:.2f} exceeds $500 limit")
        order["refunded"] = round(order["refunded"] + amount, 2)
        rid = self._next_id("next_refund", "RF-")
        self.state["refunds"][rid] = {"order_id": order_id, "amount": amount, "reason": reason}
        return {"refund_id": rid, "order_id": order_id, "amount": amount, "status": "issued"}

    @tool("Cancel an order that is still processing; refunds automatically.", mutating=True,
          order_id=("string", "Order ID"))
    def cancel_order(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] != "processing":
            raise ToolError(f"Order {order_id} is {order['status']} and cannot be cancelled")
        self._require_auth(order["customer_id"], f"cancel {order_id}")
        order["status"] = "cancelled"
        order["refunded"] = order["total"]
        return {"order_id": order_id, "status": "cancelled", "refund_amount": order["total"]}

    @tool("Update the shipping address of a processing order.", mutating=True,
          order_id=("string", "Order ID"), address=("string", "Full new address"))
    def update_shipping_address(self, order_id: str, address: str) -> dict[str, Any]:
        order = self._order(order_id)
        if order["status"] != "processing":
            raise ToolError(f"Order {order_id} is {order['status']}; address can no longer be changed")
        self._require_auth(order["customer_id"], f"address change on {order_id}")
        order["shipping_address"] = address.strip()
        return {"order_id": order_id, "shipping_address": order["shipping_address"]}

    @tool("Create a support ticket for a customer.", mutating=True,
          customer_id=("string", "Customer ID"), subject=("string", "Ticket subject"),
          priority=("string", "low | normal | high | urgent"))
    def create_ticket(self, customer_id: str, subject: str, priority: str) -> dict[str, Any]:
        if customer_id not in self.state["customers"]:
            raise ToolError(f"Customer '{customer_id}' not found")
        if priority not in PRIORITIES:
            raise ToolError(f"Invalid priority '{priority}'. Must be one of {', '.join(PRIORITIES)}")
        self._require_auth(customer_id, "created ticket")
        tid = self._next_id("next_ticket", "T-")
        self.state["tickets"][tid] = {"customer_id": customer_id, "subject": subject, "priority": priority,
                                      "status": "open", "team": None}
        return {"ticket_id": tid, "status": "open"}

    @tool("Get a support ticket.", ticket_id=("string", "Ticket ID, e.g. T-2001"))
    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        ticket = self._ticket(ticket_id)
        self._require_auth(ticket["customer_id"], f"read ticket {ticket_id}")
        return {"ticket_id": ticket_id, **copy.deepcopy(ticket)}

    @tool("Escalate a ticket to a specialist team.", mutating=True,
          ticket_id=("string", "Ticket ID"), team=("string", "billing | logistics | technical"),
          note=("string", "Context for the receiving team"))
    def escalate_ticket(self, ticket_id: str, team: str, note: str) -> dict[str, Any]:
        ticket = self._ticket(ticket_id)
        if team not in TEAMS:
            raise ToolError(f"Unknown team '{team}'. Must be one of {', '.join(TEAMS)}")
        self._require_auth(ticket["customer_id"], f"escalated {ticket_id}")
        ticket["team"] = team
        ticket["status"] = "escalated"
        ticket["note"] = note
        return {"ticket_id": ticket_id, "status": "escalated", "team": team}
