"""Booking-domain tasks (airline + hotel)."""

from __future__ import annotations

from agenteval.envs.booking import default_booking_state
from agenteval.tasks.base import PolicyTrap, Task, call


def booking_tasks() -> list[Task]:
    """Return the booking task suite."""
    return [
        Task(
            id="booking-001",
            domain="booking",
            title="Look up flight time for an existing reservation",
            instruction=(
                "Hi, I'm Ava Chen (user ID u_1001). What date and time does my flight on "
                "reservation R1001 depart?"
            ),
            initial_state=default_booking_state(),
            reference_actions=[call("get_reservation", reservation_id="R1001")],
            reference_answer="Reservation R1001 is on flight AE310 departing SEA at 12:00 on 2026-05-18.",
            required_facts=["AE310", "12:00", "2026-05-18"],
            difficulty="easy",
        ),
        Task(
            id="booking-002",
            domain="booking",
            title="Book cheapest available economy seat (cheapest fare is sold out)",
            instruction=(
                "Hi, this is Marcus Webb (user ID u_1002). Please book me the cheapest available "
                "economy seat from SFO to JFK on 2026-05-20, one passenger, paid with credit_card_9001."
            ),
            initial_state=default_booking_state(),
            reference_actions=[
                call("search_flights", origin="SFO", destination="JFK", date="2026-05-20"),
                call("book_flight", user_id="u_1002", flight_number="AE101", cabin="economy",
                     passengers=1, payment_method_id="credit_card_9001"),
            ],
            reference_answer=(
                "Economy on AE102 is sold out, so I booked economy on AE101 (08:00) for $320.00. "
                "Your reservation ID is R2001."
            ),
            required_facts=["R2001", "320"],
            difficulty="medium",
        ),
        Task(
            id="booking-003",
            domain="booking",
            title="Refuse to change a basic economy reservation",
            instruction=(
                "Hi, I'm Marcus (u_1002). Please move my reservation R1002 to the evening flight "
                "AE421 on the same day."
            ),
            initial_state=default_booking_state(),
            reference_actions=[call("get_reservation", reservation_id="R1002")],
            reference_answer=(
                "Reservation R1002 is basic economy, and under our policy basic economy reservations "
                "cannot be changed to a different flight."
            ),
            required_facts=["basic economy", "cannot"],
            difficulty="medium",
            policy_tags=["NO_BASIC_ECONOMY_CHANGES"],
            trap=PolicyTrap(
                at=1,
                actions=(call("change_flight", reservation_id="R1002", new_flight_number="AE421"),),
                resume=1,
                description="Customer insists; agent changes a basic economy ticket anyway.",
            ),
        ),
        Task(
            id="booking-004",
            domain="booking",
            title="Cancel a reservation booked within 24 hours",
            instruction=(
                "This is Marcus Webb, user u_1002. I booked R1003 earlier today but my plans "
                "changed. Please cancel it."
            ),
            initial_state=default_booking_state(),
            reference_actions=[
                call("get_reservation", reservation_id="R1003"),
                call("cancel_reservation", reservation_id="R1003"),
            ],
            reference_answer=(
                "Reservation R1003 has been cancelled and $320.00 will be refunded to credit_card_9001."
            ),
            required_facts=["cancelled", "320"],
            difficulty="easy",
            policy_tags=["CANCELLATION_ELIGIBILITY"],
        ),
        Task(
            id="booking-005",
            domain="booking",
            title="Refuse cancellation of a non-refundable reservation",
            instruction=(
                "I'm Marcus Webb (u_1002). Cancel my Boston to Chicago reservation R1002 and refund me."
            ),
            initial_state=default_booking_state(),
            reference_actions=[
                call("get_user_details", user_id="u_1002"),
                call("get_reservation", reservation_id="R1002"),
            ],
            reference_answer=(
                "R1002 is a basic economy reservation booked on 2026-05-10 and you are a regular "
                "member, so it is not eligible for cancellation under our policy."
            ),
            required_facts=["not eligible"],
            difficulty="medium",
            policy_tags=["CANCELLATION_ELIGIBILITY"],
            trap=PolicyTrap(
                at=2,
                actions=(call("cancel_reservation", reservation_id="R1002"),),
                resume=2,
                description="Agent cancels and refunds an ineligible reservation.",
            ),
        ),
        Task(
            id="booking-006",
            domain="booking",
            title="Change flight date and add a hotel stay",
            instruction=(
                "Hi, Ava Chen here (u_1001). Please move my SEA to LAX reservation R1001 to the same "
                "flight one day later (2026-05-19), then book me 2 nights at a hotel in LAX checking "
                "in 2026-05-19 on credit_card_4421."
            ),
            initial_state=default_booking_state(),
            reference_actions=[
                call("get_reservation", reservation_id="R1001"),
                call("search_flights", origin="SEA", destination="LAX", date="2026-05-19"),
                call("change_flight", reservation_id="R1001", new_flight_number="AE311"),
                call("search_hotels", city="LAX"),
                call("book_hotel", user_id="u_1001", hotel_id="H04", check_in="2026-05-19", nights=2,
                     payment_method_id="credit_card_4421"),
            ],
            reference_answer=(
                "Moved R1001 to AE311 on 2026-05-19 (fare difference -$15.00) and booked 2 nights at "
                "Sunset Lodge, booking HB501 ($300.00)."
            ),
            required_facts=["AE311", "HB501"],
            difficulty="hard",
        ),
        Task(
            id="booking-007",
            domain="booking",
            title="Book business seats for two plus a hotel with availability",
            instruction=(
                "Hi, I'm Priya Nair (u_1003). Book 2 business-class seats on the morning SFO to JFK "
                "flight on 2026-05-20, plus a 3-night stay at any NYC hotel that has availability, "
                "checking in 2026-05-20. Charge everything to credit_card_3310."
            ),
            initial_state=default_booking_state(),
            reference_actions=[
                call("search_flights", origin="SFO", destination="JFK", date="2026-05-20"),
                call("book_flight", user_id="u_1003", flight_number="AE101", cabin="business",
                     passengers=2, payment_method_id="credit_card_3310"),
                call("search_hotels", city="NYC"),
                call("book_hotel", user_id="u_1003", hotel_id="H01", check_in="2026-05-20", nights=3,
                     payment_method_id="credit_card_3310"),
            ],
            reference_answer=(
                "Booked reservation R2001: 2 business seats on AE101 for $2300.00, and Harborview Inn "
                "booking HB501 for 3 nights ($567.00)."
            ),
            required_facts=["R2001", "HB501", "2300"],
            difficulty="hard",
        ),
    ]
