"""UPI mock service behavior tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from viveka.server.services._base import ServiceError
from viveka.server.services.upi import UpiService


def _service(**overrides):
    state = {
        "balance": 5000.0,
        "payer_vpa": "alice@okicici",
        "transactions": [],
        "mandates": [],
        "cards": [],
        "fraud_vpa": ["scam@fakebank"],
    }
    state.update(overrides)
    s = UpiService()
    s.reset(state)
    return s


def test_check_balance_returns_balance_and_inr():
    s = _service(balance=1234.5)
    out = s.execute("check_balance", {})
    assert out == {"balance": 1234.5, "currency": "INR"}


def test_list_transactions_respects_limit():
    txns = [
        {"transaction_ref_id": f"t{i}", "payee_vpa": f"p{i}@bank", "amount": float(i), "status": "SUCCESS"}
        for i in range(20)
    ]
    s = _service(transactions=txns)
    out = s.execute("list_transactions", {"limit": 3})
    assert len(out["transactions"]) == 3
    assert [t["transaction_ref_id"] for t in out["transactions"]] == ["t17", "t18", "t19"]


def test_list_transactions_filters_by_payee_substring():
    txns = [
        {"transaction_ref_id": "t1", "payee_vpa": "shop@okhdfc", "amount": 100.0, "status": "SUCCESS"},
        {"transaction_ref_id": "t2", "payee_vpa": "friend@okicici", "amount": 50.0, "status": "SUCCESS"},
        {"transaction_ref_id": "t3", "payee_vpa": "shopkeeper@okaxis", "amount": 25.0, "status": "SUCCESS"},
    ]
    s = _service(transactions=txns)
    out = s.execute("list_transactions", {"payee": "shop"})
    ids = [t["transaction_ref_id"] for t in out["transactions"]]
    assert ids == ["t1", "t3"]


def test_list_transactions_filters_by_since_iso():
    base = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
    txns = [
        {"transaction_ref_id": "old", "payee_vpa": "a@b", "amount": 1.0, "status": "SUCCESS",
         "timestamp": (base - timedelta(hours=2)).isoformat()},
        {"transaction_ref_id": "new", "payee_vpa": "a@b", "amount": 2.0, "status": "SUCCESS",
         "timestamp": (base + timedelta(hours=1)).isoformat()},
    ]
    s = _service(transactions=txns)
    out = s.execute("list_transactions", {"since_iso": base.isoformat()})
    assert [t["transaction_ref_id"] for t in out["transactions"]] == ["new"]


def test_send_money_happy_path_decrements_balance_and_appends_txn():
    s = _service(balance=1000.0)
    out = s.execute("send_money", {"payee_vpa": "merchant@okhdfc", "amount": 250.0})
    assert out["status"] == "SUCCESS"
    assert out["amount"] == 250.0
    UUID(out["transaction_ref_id"], version=4)
    assert "timestamp" in out
    datetime.fromisoformat(out["timestamp"])
    assert s.state()["balance"] == 750.0
    assert len(s.state()["transactions"]) == 1


def test_send_money_invalid_vpa_format_raises_5001():
    s = _service()
    with pytest.raises(ServiceError) as exc:
        s.execute("send_money", {"payee_vpa": "not-a-vpa", "amount": 10.0})
    assert exc.value.code == "UPI:5001"


def test_send_money_insufficient_balance_raises_5012():
    s = _service(balance=10.0)
    with pytest.raises(ServiceError) as exc:
        s.execute("send_money", {"payee_vpa": "shop@okhdfc", "amount": 500.0})
    assert exc.value.code == "UPI:5012"


def test_send_money_mandate_cap_exceeded_raises_5031():
    s = _service(balance=10_000_000.0)
    with pytest.raises(ServiceError) as exc:
        s.execute("send_money", {"payee_vpa": "shop@okhdfc", "amount": 200_000.0})
    assert exc.value.code == "UPI:5031"


def test_send_money_fraud_vpa_blocked_raises_5050():
    s = _service(balance=10_000.0, fraud_vpa=["scam@fakebank"])
    with pytest.raises(ServiceError) as exc:
        s.execute("send_money", {"payee_vpa": "scam@fakebank", "amount": 100.0})
    assert exc.value.code == "UPI:5050"


def test_approve_mandate_transitions_pending_and_debits_balance():
    s = _service(
        balance=2000.0,
        mandates=[{"mandate_id": "M1", "merchant": "netflix@okhdfc", "amount": 499.0, "status": "pending"}],
    )
    out = s.execute("approve_mandate", {"mandate_id": "M1"})
    assert out["status"] == "approved"
    assert out["mandate_id"] == "M1"
    UUID(out["transaction_ref_id"], version=4)
    state = s.state()
    assert state["balance"] == 1501.0
    assert len(state["transactions"]) == 1
    assert state["transactions"][0]["mandate_id"] == "M1"
    assert state["transactions"][0]["amount"] == 499.0


def test_approve_mandate_idempotency_raises_7012():
    s = _service(
        balance=2000.0,
        mandates=[{"mandate_id": "M1", "merchant": "netflix@okhdfc", "amount": 499.0, "status": "pending"}],
    )
    s.execute("approve_mandate", {"mandate_id": "M1"})
    with pytest.raises(ServiceError) as exc:
        s.execute("approve_mandate", {"mandate_id": "M1"})
    assert exc.value.code == "UPI:7012"


def test_block_card_idempotency_raises_8010():
    s = _service(cards=[{"last4": "1234", "blocked": False}])
    s.execute("block_card", {"card_last4": "1234"})
    with pytest.raises(ServiceError) as exc:
        s.execute("block_card", {"card_last4": "1234"})
    assert exc.value.code == "UPI:8010"


def test_raise_dispute_existing_txn_returns_dispute_id_missing_raises_9001():
    s = _service(balance=1000.0)
    sent = s.execute("send_money", {"payee_vpa": "shop@okhdfc", "amount": 100.0})
    out = s.execute("raise_dispute", {"transaction_ref_id": sent["transaction_ref_id"]})
    assert out["status"] == "filed"
    UUID(out["dispute_id"], version=4)
    assert out["transaction_ref_id"] == sent["transaction_ref_id"]
    assert s.state()["disputes"][0]["dispute_id"] == out["dispute_id"]

    with pytest.raises(ServiceError) as exc:
        s.execute("raise_dispute", {"transaction_ref_id": "does-not-exist"})
    assert exc.value.code == "UPI:9001"


def test_view_pending_mandates_filters_by_merchant_substring():
    s = _service(mandates=[
        {"mandate_id": "M1", "merchant": "netflix@okhdfc", "amount": 499.0, "status": "pending"},
        {"mandate_id": "M2", "merchant": "spotify@okicici", "amount": 119.0, "status": "pending"},
        {"mandate_id": "M3", "merchant": "netflix@okhdfc", "amount": 199.0, "status": "approved"},
    ])
    out = s.execute("view_pending_mandates", {"merchant": "netflix"})
    assert len(out["mandates"]) == 1
    assert out["mandates"][0]["mandate_id"] == "M1"


def test_reject_mandate_idempotency_raises_7013():
    s = _service(mandates=[{"mandate_id": "M1", "merchant": "x@y", "amount": 10.0, "status": "pending"}])
    s.execute("reject_mandate", {"mandate_id": "M1"})
    with pytest.raises(ServiceError) as exc:
        s.execute("reject_mandate", {"mandate_id": "M1"})
    assert exc.value.code == "UPI:7013"
