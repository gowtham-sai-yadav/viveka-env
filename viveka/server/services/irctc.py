"""Mock IRCTC service. Stubbed; full quota/refund behavior lands in Phase 1."""

from __future__ import annotations

import random
import string
from typing import Any
from uuid import uuid4

from viveka.server.services._base import MockService, ServiceError


def _gen_pnr() -> str:
    return "".join(random.choices(string.digits, k=10))


class IrctcService(MockService):
    name = "irctc"

    def reset(self, initial_state: dict[str, Any]) -> None:
        self._catalogue: list[dict[str, Any]] = list(initial_state.get("catalogue", []))
        self._bookings: list[dict[str, Any]] = list(initial_state.get("bookings", []))
        self._availability: dict[str, dict[str, int]] = dict(initial_state.get("availability", {}))
        self._now_iso: str = initial_state.get("now_iso", "2026-04-25T08:00:00+05:30")

    def state(self) -> dict[str, Any]:
        return {
            "bookings": list(self._bookings),
            "now_iso": self._now_iso,
        }

    # ── reversible ────────────────────────────────────────────────────────

    def _op_search_trains(self, params: dict[str, Any]) -> dict[str, Any]:
        src = params.get("from_station", "").upper()
        dst = params.get("to_station", "").upper()
        results = [
            t for t in self._catalogue
            if (not src or t.get("from_station", "").upper() == src)
            and (not dst or t.get("to_station", "").upper() == dst)
        ]
        return {"trains": results}

    def _op_check_seat_availability(self, params: dict[str, Any]) -> dict[str, Any]:
        train_no = params.get("train_no", "")
        cls = params.get("class", "SL")
        avail = self._availability.get(train_no, {}).get(cls, 0)
        return {"train_no": train_no, "class": cls, "available": avail}

    def _op_check_pnr(self, params: dict[str, Any]) -> dict[str, Any]:
        pnr = params.get("pnr", "")
        for b in self._bookings:
            if b.get("pnr") == pnr:
                return dict(b)
        raise ServiceError("IRCTC:E1004", f"PNR not found: {pnr}")

    def _op_view_booking_history(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"bookings": list(self._bookings)}

    # ── irreversible ──────────────────────────────────────────────────────

    def _op_book_ticket(self, params: dict[str, Any]) -> dict[str, Any]:
        train_no = params.get("train_no", "")
        cls = params.get("class", "SL")
        passengers = params.get("passengers", [])
        if not train_no:
            raise ServiceError("IRCTC:E2001", "train_no is required")
        avail = self._availability.get(train_no, {}).get(cls, 0)
        if avail < len(passengers):
            raise ServiceError("IRCTC:E2032", f"No seats in {cls} on {train_no}")
        booking = {
            "pnr": _gen_pnr(),
            "booking_id": str(uuid4()),
            "train_no": train_no,
            "class": cls,
            "passengers": passengers,
            "status": "CNF",
        }
        self._availability.setdefault(train_no, {})[cls] = avail - len(passengers)
        self._bookings.append(booking)
        return booking

    def _op_cancel_booking(self, params: dict[str, Any]) -> dict[str, Any]:
        pnr = params.get("pnr", "")
        for b in self._bookings:
            if b.get("pnr") == pnr:
                b["status"] = "CANCELLED"
                return {"pnr": pnr, "status": "CANCELLED"}
        raise ServiceError("IRCTC:E1004", f"PNR not found: {pnr}")

    def _op_modify_booking(self, params: dict[str, Any]) -> dict[str, Any]:
        pnr = params.get("pnr", "")
        for b in self._bookings:
            if b.get("pnr") == pnr:
                b.update({k: v for k, v in params.items() if k != "pnr"})
                return dict(b)
        raise ServiceError("IRCTC:E1004", f"PNR not found: {pnr}")
