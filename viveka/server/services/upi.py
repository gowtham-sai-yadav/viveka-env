"""Mock UPI service. Stubbed write ops; full NPCI-flavored behavior lands in Phase 1."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from viveka.server.services._base import MockService, ServiceError

VPA_RE = re.compile(r"^[a-zA-Z0-9._-]+@[a-zA-Z][a-zA-Z0-9]+$")
MANDATE_CAP = 100_000.0  # NPCI per-transaction mandate cap, ₹1 lakh.


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

    def state(self) -> dict[str, Any]:
        return {
            "balance": round(self._balance, 2),
            "payer_vpa": self._payer_vpa,
            "transactions": list(self._transactions),
            "mandates": list(self._mandates),
            "cards": list(self._cards),
        }

    # ── reversible ────────────────────────────────────────────────────────

    def _op_check_balance(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"balance": round(self._balance, 2), "currency": "INR"}

    def _op_list_transactions(self, params: dict[str, Any]) -> dict[str, Any]:
        n = int(params.get("limit", 10))
        return {"transactions": list(self._transactions[-n:])}

    def _op_view_pending_mandates(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"mandates": [m for m in self._mandates if m.get("status") == "pending"]}

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
        }
        self._balance -= amount
        self._transactions.append(txn)
        return txn

    def _op_approve_mandate(self, params: dict[str, Any]) -> dict[str, Any]:
        mandate_id = params.get("mandate_id", "")
        for m in self._mandates:
            if m.get("mandate_id") == mandate_id:
                m["status"] = "approved"
                return {"mandate_id": mandate_id, "status": "approved"}
        raise ServiceError("UPI:7001", f"Mandate not found: {mandate_id}")

    def _op_reject_mandate(self, params: dict[str, Any]) -> dict[str, Any]:
        mandate_id = params.get("mandate_id", "")
        for m in self._mandates:
            if m.get("mandate_id") == mandate_id:
                m["status"] = "rejected"
                return {"mandate_id": mandate_id, "status": "rejected"}
        raise ServiceError("UPI:7001", f"Mandate not found: {mandate_id}")

    def _op_block_card(self, params: dict[str, Any]) -> dict[str, Any]:
        last4 = params.get("card_last4", "")
        for c in self._cards:
            if c.get("last4") == last4:
                c["blocked"] = True
                return {"card_last4": last4, "blocked": True}
        raise ServiceError("UPI:8003", f"Card not found: {last4}")

    def _op_raise_dispute(self, params: dict[str, Any]) -> dict[str, Any]:
        txn_id = params.get("transaction_ref_id", "")
        return {"transaction_ref_id": txn_id, "dispute_id": str(uuid4()), "status": "filed"}
