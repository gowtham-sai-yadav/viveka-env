"""Mock UPI service with NPCI-flavored semantics."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from viveka.server.services._base import MockService, ServiceError

VPA_RE = re.compile(r"^[a-zA-Z0-9._-]+@[a-zA-Z][a-zA-Z0-9]+$")
MANDATE_CAP = 100_000.0  # NPCI per-transaction mandate cap, ₹1 lakh.


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UpiService(MockService):
    name = "upi"

    def reset(self, initial_state: dict[str, Any]) -> None:
        self._balance: float = float(initial_state.get("balance", 0.0))
        self._transactions: list[dict[str, Any]] = list(initial_state.get("transactions", []))
        self._mandates: list[dict[str, Any]] = list(initial_state.get("mandates", []))
        self._cards: list[dict[str, Any]] = list(initial_state.get("cards", []))
        self._contacts: dict[str, str] = dict(initial_state.get("contacts", {}))
        self._fraud_vpa: set[str] = set(initial_state.get("fraud_vpa", []))
        self._payer_vpa: str = initial_state.get("payer_vpa", "user@upi")
        self._disputes: list[dict[str, Any]] = list(initial_state.get("disputes", []))

    def state(self) -> dict[str, Any]:
        return {
            "balance": round(self._balance, 2),
            "payer_vpa": self._payer_vpa,
            "transactions": list(self._transactions),
            "mandates": list(self._mandates),
            "cards": list(self._cards),
            "disputes": list(self._disputes),
        }

    # ── reversible ────────────────────────────────────────────────────────

    def _op_check_balance(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"balance": round(self._balance, 2), "currency": "INR"}

    def _op_list_transactions(self, params: dict[str, Any]) -> dict[str, Any]:
        limit = int(params.get("limit", 10))
        payee = params.get("payee")
        since_iso = params.get("since_iso")
        txns = list(self._transactions)
        if payee:
            txns = [t for t in txns if payee in t.get("payee_vpa", "")]
        if since_iso:
            txns = [t for t in txns if "timestamp" not in t or t["timestamp"] >= since_iso]
        return {"transactions": txns[-limit:]}

    def _op_view_pending_mandates(self, params: dict[str, Any]) -> dict[str, Any]:
        merchant = params.get("merchant")
        pending = [m for m in self._mandates if m.get("status") == "pending"]
        if merchant:
            pending = [m for m in pending if merchant in m.get("merchant", "")]
        return {"mandates": pending}

    def _op_lookup_vpa(self, params: dict[str, Any]) -> dict[str, Any]:
        vpa = params.get("vpa", "")
        flagged = vpa in self._fraud_vpa
        valid = bool(VPA_RE.match(vpa))
        return {"vpa": vpa, "valid": valid, "fraud_flag": flagged}

    # ── irreversible ──────────────────────────────────────────────────────

    def _op_send_money(self, params: dict[str, Any]) -> dict[str, Any]:
        payee = params.get("payee_vpa", "")
        amount = float(params.get("amount", 0.0))
        if not VPA_RE.match(payee):
            raise ServiceError("UPI:5001", f"Invalid VPA format: {payee}")
        if amount <= 0:
            raise ServiceError("UPI:5004", "Amount must be positive")
        if payee in self._fraud_vpa:
            raise ServiceError("UPI:5050", "Payee on fraud watchlist — blocked")
        if amount > self._balance:
            raise ServiceError("UPI:5012", "Insufficient balance")
        if amount > MANDATE_CAP:
            raise ServiceError("UPI:5031", f"Mandate cap exceeded (₹{MANDATE_CAP:.0f})")
        txn = {
            "transaction_ref_id": str(uuid4()),
            "payer_vpa": self._payer_vpa,
            "payee_vpa": payee,
            "amount": round(amount, 2),
            "status": "SUCCESS",
            "timestamp": _now_iso(),
        }
        self._balance -= amount
        self._transactions.append(txn)
        return txn

    def _op_approve_mandate(self, params: dict[str, Any]) -> dict[str, Any]:
        mandate_id = params.get("mandate_id", "")
        for m in self._mandates:
            if m.get("mandate_id") != mandate_id:
                continue
            if m.get("status") == "approved":
                raise ServiceError("UPI:7012", "Mandate already approved")
            amount = float(m.get("amount", 0.0))
            if amount > self._balance:
                raise ServiceError("UPI:5012", "Insufficient balance")
            m["status"] = "approved"
            txn = {
                "transaction_ref_id": str(uuid4()),
                "payer_vpa": self._payer_vpa,
                "payee_vpa": m.get("merchant", ""),
                "amount": round(amount, 2),
                "status": "SUCCESS",
                "mandate_id": mandate_id,
                "timestamp": _now_iso(),
            }
            self._balance -= amount
            self._transactions.append(txn)
            return {"mandate_id": mandate_id, "status": "approved", "transaction_ref_id": txn["transaction_ref_id"]}
        raise ServiceError("UPI:7001", f"Mandate not found: {mandate_id}")

    def _op_reject_mandate(self, params: dict[str, Any]) -> dict[str, Any]:
        mandate_id = params.get("mandate_id", "")
        for m in self._mandates:
            if m.get("mandate_id") != mandate_id:
                continue
            if m.get("status") == "rejected":
                raise ServiceError("UPI:7013", "Mandate already rejected")
            m["status"] = "rejected"
            return {"mandate_id": mandate_id, "status": "rejected"}
        raise ServiceError("UPI:7001", f"Mandate not found: {mandate_id}")

    def _op_block_card(self, params: dict[str, Any]) -> dict[str, Any]:
        last4 = params.get("card_last4", "")
        for c in self._cards:
            if c.get("last4") != last4:
                continue
            if c.get("blocked"):
                raise ServiceError("UPI:8010", "Card already blocked")
            c["blocked"] = True
            return {"card_last4": last4, "blocked": True}
        raise ServiceError("UPI:8003", f"Card not found: {last4}")

    def _op_raise_dispute(self, params: dict[str, Any]) -> dict[str, Any]:
        txn_id = params.get("transaction_ref_id", "")
        match = next((t for t in self._transactions if t.get("transaction_ref_id") == txn_id), None)
        if match is None:
            raise ServiceError("UPI:9001", "Transaction not found for dispute")
        dispute = {
            "dispute_id": str(uuid4()),
            "transaction_ref_id": txn_id,
            "status": "filed",
            "filed_at": _now_iso(),
        }
        self._disputes.append(dispute)
        return dispute
