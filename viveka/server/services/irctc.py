"""Mock IRCTC service with PNR/quota/refund/tatkal semantics.

Error codes:
IRCTC:E1004 PNR not found | IRCTC:E1010 already cancelled | IRCTC:E2001 train_no required/unknown |
IRCTC:E2032 tatkal closed or no tatkal seats | IRCTC:E3001 refund window expired |
IRCTC:E4001 modify after chart prep | IRCTC:E4002 modify cancelled booking
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from viveka.server.services._base import MockService, ServiceError

IST = timezone(timedelta(hours=5, minutes=30))
TATKAL_AC_CLASSES = {"3A", "2A", "1A", "CC", "EC"}
TATKAL_SL_CLASSES = {"SL", "2S"}


def _gen_pnr() -> str:
    return "".join(random.choices(string.digits, k=10))


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt


class IrctcService(MockService):
    name = "irctc"

    def reset(self, initial_state: dict[str, Any]) -> None:
        self._catalogue: list[dict[str, Any]] = list(initial_state.get("catalogue", []))
        self._bookings: list[dict[str, Any]] = list(initial_state.get("bookings", []))
        self._availability: dict[str, dict[str, int]] = {
            tn: dict(cls_map) for tn, cls_map in dict(initial_state.get("availability", {})).items()
        }
        self._now_iso: str = initial_state.get("now_iso", "2026-04-25T08:00:00+05:30")

    def state(self) -> dict[str, Any]:
        return {
            "catalogue": list(self._catalogue),
            "bookings": list(self._bookings),
            "availability": {tn: dict(cls_map) for tn, cls_map in self._availability.items()},
            "now_iso": self._now_iso,
        }

    def _advance_clock(self, minutes: int) -> None:
        self._now_iso = (_parse_iso(self._now_iso) + timedelta(minutes=minutes)).isoformat()

    def _now_dt(self) -> datetime:
        return _parse_iso(self._now_iso)

    # ── reversible ────────────────────────────────────────────────────────

    def _op_search_trains(self, params: dict[str, Any]) -> dict[str, Any]:
        src = params.get("from_station", "").upper()
        dst = params.get("to_station", "").upper()
        cls_filter = params.get("class")
        results = [
            t for t in self._catalogue
            if (not src or t.get("from_station", "").upper() == src)
            and (not dst or t.get("to_station", "").upper() == dst)
        ]
        if cls_filter:
            results = [
                t for t in results
                if self._availability.get(t.get("train_no", ""), {}).get(cls_filter, 0) > 0
            ]
        return {"trains": results}

    def _op_check_seat_availability(self, params: dict[str, Any]) -> dict[str, Any]:
        train_no = params.get("train_no", "")
        cls = params.get("class", "SL")
        avail = self._availability.get(train_no, {}).get(cls, 0)
        if avail >= 4:
            status = "AVAILABLE"
        elif avail >= 1:
            status = "RAC"
        else:
            status = "WL"
        return {"train_no": train_no, "class": cls, "available": avail, "status": status}

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
        passengers = list(params.get("passengers", []))
        if not train_no:
            raise ServiceError("IRCTC:E2001", "train_no is required")
        train_entry = next((t for t in self._catalogue if t.get("train_no") == train_no), None)
        if train_entry is None:
            raise ServiceError("IRCTC:E2001", f"train_no not in catalogue: {train_no}")
        if cls.startswith("TKT-"):
            self._enforce_tatkal_window(cls)
        avail = self._availability.get(train_no, {}).get(cls, 0)
        if avail <= 0 and cls.startswith("TKT-"):
            raise ServiceError("IRCTC:E2032", "No tatkal seats remaining")
        if avail < len(passengers):
            raise ServiceError("IRCTC:E2032", f"No seats in {cls} on {train_no}")
        booking = {
            "pnr": _gen_pnr(),
            "booking_id": str(uuid4()),
            "train_no": train_no,
            "class": cls,
            "passengers": passengers,
            "status": "CNF",
            "booked_at": self._now_iso,
            "departure_iso": params.get("departure_iso") or train_entry.get("departure_iso"),
            "chart_prepared": False,
        }
        self._availability.setdefault(train_no, {})[cls] = avail - len(passengers)
        self._bookings.append(booking)
        return dict(booking)

    def _op_cancel_booking(self, params: dict[str, Any]) -> dict[str, Any]:
        pnr = params.get("pnr", "")
        booking = next((b for b in self._bookings if b.get("pnr") == pnr), None)
        if booking is None:
            raise ServiceError("IRCTC:E1004", f"PNR not found: {pnr}")
        if str(booking.get("status", "")).startswith("CANCELLED"):
            raise ServiceError("IRCTC:E1010", "Booking already cancelled")
        departure_iso = booking.get("departure_iso")
        passengers = booking.get("passengers", []) or []
        if departure_iso:
            hours_to_departure = (_parse_iso(departure_iso) - self._now_dt()).total_seconds() / 3600.0
            if hours_to_departure < 4:
                raise ServiceError("IRCTC:E3001", "Refund window expired")
            elif hours_to_departure < 12:
                booking["status"] = "CANCELLED_PARTIAL_REFUND"
                booking["refund_pct"] = 50
            else:
                booking["status"] = "CANCELLED_FULL_REFUND"
                booking["refund_pct"] = 75
        else:
            booking["status"] = "CANCELLED_FULL_REFUND"
            booking["refund_pct"] = 75
        self._restore_seats(booking["train_no"], booking["class"], len(passengers))
        return {
            "pnr": pnr,
            "status": booking["status"],
            "refund_pct": booking["refund_pct"],
        }

    def _op_modify_booking(self, params: dict[str, Any]) -> dict[str, Any]:
        pnr = params.get("pnr", "")
        if "class" not in params and "passengers" not in params:
            raise ServiceError("IRCTC:E2001", "Provide at least one of class, passengers to modify")
        booking = next((b for b in self._bookings if b.get("pnr") == pnr), None)
        if booking is None:
            raise ServiceError("IRCTC:E1004", f"PNR not found: {pnr}")
        if booking.get("chart_prepared"):
            raise ServiceError("IRCTC:E4001", "Modification not allowed after chart preparation")
        if str(booking.get("status", "")).startswith("CANCELLED"):
            raise ServiceError("IRCTC:E4002", "Cannot modify cancelled booking")
        old_class = booking["class"]
        old_count = len(booking.get("passengers", []) or [])
        new_class = params.get("class", old_class)
        new_passengers = list(params.get("passengers", booking.get("passengers", []) or []))
        train_no = booking["train_no"]
        delta = len(new_passengers) - old_count
        if new_class != old_class:
            self._restore_seats(train_no, old_class, old_count)
            avail_new = self._availability.get(train_no, {}).get(new_class, 0)
            if avail_new < len(new_passengers):
                self._availability.setdefault(train_no, {})[old_class] = (
                    self._availability.get(train_no, {}).get(old_class, 0) - old_count
                )
                raise ServiceError("IRCTC:E2032", f"No seats in {new_class} on {train_no}")
            self._availability.setdefault(train_no, {})[new_class] = avail_new - len(new_passengers)
        elif delta > 0:
            avail = self._availability.get(train_no, {}).get(new_class, 0)
            if avail < delta:
                raise ServiceError("IRCTC:E2032", f"No seats in {new_class} on {train_no}")
            self._availability[train_no][new_class] = avail - delta
        elif delta < 0:
            self._restore_seats(train_no, new_class, -delta)
        booking["class"] = new_class
        booking["passengers"] = new_passengers
        return dict(booking)

    # ── helpers ───────────────────────────────────────────────────────────

    def _enforce_tatkal_window(self, cls: str) -> None:
        sub = cls[len("TKT-"):]
        now_ist = self._now_dt().astimezone(IST)
        if sub in TATKAL_AC_CLASSES or sub == "AC":
            if now_ist.hour < 10:
                raise ServiceError("IRCTC:E2032", "Tatkal AC window opens at 10:00 IST")
        elif sub in TATKAL_SL_CLASSES or sub == "SL":
            if now_ist.hour < 11:
                raise ServiceError("IRCTC:E2032", "Tatkal SL window opens at 11:00 IST")

    def _restore_seats(self, train_no: str, cls: str, count: int) -> None:
        if count <= 0:
            return
        cur = self._availability.get(train_no, {}).get(cls, 0)
        self._availability.setdefault(train_no, {})[cls] = cur + count
