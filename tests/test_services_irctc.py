"""IRCTC mock service behavior tests."""

from __future__ import annotations

import pytest

from viveka.server.services._base import ServiceError
from viveka.server.services.irctc import IrctcService


def _service(**overrides):
    state = {
        "catalogue": [
            {"train_no": "12345", "name": "Rajdhani Exp", "from_station": "NDLS", "to_station": "BCT",
             "departure_iso": "2026-04-26T16:00:00+05:30"},
            {"train_no": "12952", "name": "Mumbai Rajdhani", "from_station": "NDLS", "to_station": "BCT",
             "departure_iso": "2026-04-26T16:30:00+05:30"},
            {"train_no": "22691", "name": "Rajdhani BLR", "from_station": "NDLS", "to_station": "SBC",
             "departure_iso": "2026-04-26T20:00:00+05:30"},
        ],
        "bookings": [],
        "availability": {
            "12345": {"SL": 10, "3A": 4, "2A": 0, "TKT-AC": 2, "TKT-SL": 0},
            "12952": {"SL": 0, "3A": 2, "2A": 1, "TKT-AC": 5},
            "22691": {"SL": 0, "3A": 0, "2A": 0},
        },
        "now_iso": "2026-04-25T08:00:00+05:30",
    }
    state.update(overrides)
    s = IrctcService()
    s.reset(state)
    return s


def test_search_trains_filters_by_from_to_station():
    s = _service()
    out = s.execute("search_trains", {"from_station": "NDLS", "to_station": "BCT"})
    train_nos = {t["train_no"] for t in out["trains"]}
    assert train_nos == {"12345", "12952"}


def test_search_trains_class_filter_excludes_zero_availability():
    s = _service()
    out = s.execute("search_trains", {"from_station": "NDLS", "to_station": "BCT", "class": "SL"})
    train_nos = [t["train_no"] for t in out["trains"]]
    assert train_nos == ["12345"]
    out2 = s.execute("search_trains", {"from_station": "NDLS", "to_station": "SBC", "class": "3A"})
    assert out2["trains"] == []


def test_check_seat_availability_status_boundaries():
    s = _service(availability={
        "T1": {"SL": 10},
        "T2": {"SL": 4},
        "T3": {"SL": 3},
        "T4": {"SL": 1},
        "T5": {"SL": 0},
    })
    assert s.execute("check_seat_availability", {"train_no": "T1", "class": "SL"})["status"] == "AVAILABLE"
    assert s.execute("check_seat_availability", {"train_no": "T2", "class": "SL"})["status"] == "AVAILABLE"
    assert s.execute("check_seat_availability", {"train_no": "T3", "class": "SL"})["status"] == "RAC"
    assert s.execute("check_seat_availability", {"train_no": "T4", "class": "SL"})["status"] == "RAC"
    assert s.execute("check_seat_availability", {"train_no": "T5", "class": "SL"})["status"] == "WL"


def test_check_pnr_happy_and_missing():
    s = _service(bookings=[
        {"pnr": "1234567890", "train_no": "12345", "class": "SL", "passengers": ["A"], "status": "CNF"},
    ])
    out = s.execute("check_pnr", {"pnr": "1234567890"})
    assert out["train_no"] == "12345"
    with pytest.raises(ServiceError) as exc:
        s.execute("check_pnr", {"pnr": "0000000000"})
    assert exc.value.code == "IRCTC:E1004"


def test_book_ticket_happy_path_pnr_and_decrement():
    s = _service()
    out = s.execute("book_ticket", {"train_no": "12345", "class": "SL", "passengers": ["A", "B"]})
    assert out["status"] == "CNF"
    assert len(out["pnr"]) == 10 and out["pnr"].isdigit()
    assert out["booked_at"] == "2026-04-25T08:00:00+05:30"
    assert s.state()["availability"]["12345"]["SL"] == 8
    assert len(s.state()["bookings"]) == 1


def test_book_ticket_unknown_train_raises_e2001():
    s = _service()
    with pytest.raises(ServiceError) as exc:
        s.execute("book_ticket", {"train_no": "99999", "class": "SL", "passengers": ["A"]})
    assert exc.value.code == "IRCTC:E2001"

    with pytest.raises(ServiceError) as exc2:
        s.execute("book_ticket", {"train_no": "", "class": "SL", "passengers": ["A"]})
    assert exc2.value.code == "IRCTC:E2001"


def test_book_ticket_tatkal_ac_before_10_raises_e2032():
    s = _service(now_iso="2026-04-25T09:30:00+05:30")
    with pytest.raises(ServiceError) as exc:
        s.execute("book_ticket", {"train_no": "12345", "class": "TKT-AC", "passengers": ["A"]})
    assert exc.value.code == "IRCTC:E2032"
    assert "10:00" in exc.value.message


def test_book_ticket_tatkal_ac_after_10_succeeds():
    s = _service(now_iso="2026-04-25T10:15:00+05:30")
    out = s.execute("book_ticket", {"train_no": "12345", "class": "TKT-AC", "passengers": ["A"]})
    assert out["status"] == "CNF"
    assert out["class"] == "TKT-AC"
    assert s.state()["availability"]["12345"]["TKT-AC"] == 1


def test_book_ticket_tatkal_sl_before_11_raises_e2032():
    s = _service(
        now_iso="2026-04-25T10:30:00+05:30",
        availability={"12345": {"TKT-SL": 5}},
    )
    with pytest.raises(ServiceError) as exc:
        s.execute("book_ticket", {"train_no": "12345", "class": "TKT-SL", "passengers": ["A"]})
    assert exc.value.code == "IRCTC:E2032"
    assert "11:00" in exc.value.message


def test_book_ticket_tatkal_no_seats_raises_e2032():
    s = _service(now_iso="2026-04-25T11:30:00+05:30")
    with pytest.raises(ServiceError) as exc:
        s.execute("book_ticket", {"train_no": "12345", "class": "TKT-SL", "passengers": ["A"]})
    assert exc.value.code == "IRCTC:E2032"
    assert "tatkal" in exc.value.message.lower()


def test_cancel_booking_more_than_12h_full_refund_and_seats_restored():
    s = _service(now_iso="2026-04-25T08:00:00+05:30")
    booked = s.execute("book_ticket", {"train_no": "12345", "class": "SL", "passengers": ["A", "B"]})
    avail_after_book = s.state()["availability"]["12345"]["SL"]
    out = s.execute("cancel_booking", {"pnr": booked["pnr"]})
    assert out["status"] == "CANCELLED_FULL_REFUND"
    assert out["refund_pct"] == 75
    assert s.state()["availability"]["12345"]["SL"] == avail_after_book + 2


def test_cancel_booking_4_to_12h_partial_refund():
    s = _service(now_iso="2026-04-26T08:00:00+05:30")
    booked = s.execute("book_ticket", {"train_no": "12345", "class": "SL", "passengers": ["A"]})
    out = s.execute("cancel_booking", {"pnr": booked["pnr"]})
    assert out["status"] == "CANCELLED_PARTIAL_REFUND"
    assert out["refund_pct"] == 50


def test_cancel_booking_less_than_4h_raises_e3001():
    s = _service(now_iso="2026-04-26T13:30:00+05:30")
    booked = s.execute("book_ticket", {"train_no": "12345", "class": "SL", "passengers": ["A"]})
    with pytest.raises(ServiceError) as exc:
        s.execute("cancel_booking", {"pnr": booked["pnr"]})
    assert exc.value.code == "IRCTC:E3001"


def test_cancel_booking_already_cancelled_raises_e1010():
    s = _service(now_iso="2026-04-25T08:00:00+05:30")
    booked = s.execute("book_ticket", {"train_no": "12345", "class": "SL", "passengers": ["A"]})
    s.execute("cancel_booking", {"pnr": booked["pnr"]})
    with pytest.raises(ServiceError) as exc:
        s.execute("cancel_booking", {"pnr": booked["pnr"]})
    assert exc.value.code == "IRCTC:E1010"


def test_cancel_booking_missing_pnr_raises_e1004():
    s = _service()
    with pytest.raises(ServiceError) as exc:
        s.execute("cancel_booking", {"pnr": "9999999999"})
    assert exc.value.code == "IRCTC:E1004"


def test_modify_booking_post_chart_raises_e4001():
    s = _service(bookings=[{
        "pnr": "1111111111", "train_no": "12345", "class": "SL", "passengers": ["A"],
        "status": "CNF", "chart_prepared": True,
    }])
    with pytest.raises(ServiceError) as exc:
        s.execute("modify_booking", {"pnr": "1111111111", "class": "3A"})
    assert exc.value.code == "IRCTC:E4001"


def test_modify_booking_cancelled_raises_e4002():
    s = _service(now_iso="2026-04-25T08:00:00+05:30")
    booked = s.execute("book_ticket", {"train_no": "12345", "class": "SL", "passengers": ["A"]})
    s.execute("cancel_booking", {"pnr": booked["pnr"]})
    with pytest.raises(ServiceError) as exc:
        s.execute("modify_booking", {"pnr": booked["pnr"], "class": "3A"})
    assert exc.value.code == "IRCTC:E4002"


def test_modify_booking_class_change_updates_availability():
    s = _service(now_iso="2026-04-25T08:00:00+05:30")
    booked = s.execute("book_ticket", {"train_no": "12345", "class": "SL", "passengers": ["A"]})
    sl_after_book = s.state()["availability"]["12345"]["SL"]
    a3_before = s.state()["availability"]["12345"]["3A"]
    out = s.execute("modify_booking", {"pnr": booked["pnr"], "class": "3A"})
    assert out["class"] == "3A"
    assert s.state()["availability"]["12345"]["SL"] == sl_after_book + 1
    assert s.state()["availability"]["12345"]["3A"] == a3_before - 1


def test_advance_clock_moves_now_iso_forward():
    s = _service(now_iso="2026-04-25T08:00:00+05:30")
    s._advance_clock(150)
    assert s._now_iso.startswith("2026-04-25T10:30:00")
