"""Mock Banking service. RBI/NPCI-flavored error codes per RBI cyberfraud bulletins (RBI/2023-24/85): BNK:1001 account-not-found, BNK:1404 invalid-statement-range, BNK:2001 invalid-IFSC, BNK:3010 cooling-violation, BNK:3015 duplicate-beneficiary, BNK:3022 beneficiary-not-found, BNK:4001 NEFT-cap-exceeded, BNK:4012 cooling-window-violation, BNK:4050 mule-flag, BNK:5001 CVV-mismatch, BNK:5099 CVV-lockout, BNK:6010 ATM-PIN-mismatch, BNK:7001 virtual-card-daily-cap."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from viveka.server.services._base import MockService, ServiceError

IFSC_RE = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
NEFT_PER_TXN_CAP = 1_000_000.0  # RBI NEFT retail cap ₹10 lakh per transaction.
COOLING_THRESHOLD = 50_000.0  # RBI 30-min cooling for first transfer >₹50k to new beneficiary.
COOLING_MINUTES = 30
CVV_LOCKOUT_THRESHOLD = 3
VIRTUAL_CARD_DAILY_CAP = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BankingService(MockService):
    name = "banking"

    def reset(self, initial_state: dict[str, Any]) -> None:
        self._accounts: list[dict[str, Any]] = list(initial_state.get("accounts", []))
        self._beneficiaries: list[dict[str, Any]] = list(initial_state.get("beneficiaries", []))
        self._cards: list[dict[str, Any]] = list(initial_state.get("cards", []))
        self._pending_neft: list[dict[str, Any]] = list(initial_state.get("pending_neft", []))
        self._cvv_attempt_counts: dict[str, int] = dict(initial_state.get("cvv_attempt_counts", {}))
        self._virtual_cards_today: int = int(initial_state.get("virtual_cards_today", 0))

    def state(self) -> dict[str, Any]:
        return {
            "accounts": list(self._accounts),
            "beneficiaries": list(self._beneficiaries),
            "cards": list(self._cards),
            "pending_neft": list(self._pending_neft),
            "cvv_attempt_counts": dict(self._cvv_attempt_counts),
            "virtual_cards_today": self._virtual_cards_today,
        }

    def _find_account(self, account_id: str | None) -> dict[str, Any] | None:
        if not self._accounts:
            return None
        if account_id is None:
            return self._accounts[0]
        for a in self._accounts:
            if a.get("account_id") == account_id:
                return a
        return None

    def _find_beneficiary(self, beneficiary_id: str) -> dict[str, Any] | None:
        for b in self._beneficiaries:
            if b.get("beneficiary_id") == beneficiary_id:
                return b
        return None

    def _find_card(self, card_id: str) -> dict[str, Any] | None:
        for c in self._cards:
            if c.get("card_id") == card_id:
                return c
        return None

    # ── reversible ────────────────────────────────────────────────────────

    def _op_check_account_balance(self, params: dict[str, Any]) -> dict[str, Any]:
        account_id = params.get("account_id")
        acct = self._find_account(account_id)
        if acct is None:
            raise ServiceError("BNK:1001", f"Account not found: {account_id}")
        return {
            "account_id": acct.get("account_id"),
            "account_number": acct.get("account_number", ""),
            "balance": round(float(acct.get("balance", 0.0)), 2),
            "currency": "INR",
        }

    def _op_list_beneficiaries(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"beneficiaries": list(self._beneficiaries)}

    def _op_view_statement(self, params: dict[str, Any]) -> dict[str, Any]:
        account_id = params.get("account_id")
        from_iso = params.get("from_iso", "")
        to_iso = params.get("to_iso", "")
        acct = self._find_account(account_id)
        if acct is None:
            raise ServiceError("BNK:1001", f"Account not found: {account_id}")
        if not from_iso or not to_iso or from_iso > to_iso:
            raise ServiceError("BNK:1404", "Invalid statement date range")
        txns = [
            t for t in acct.get("transactions", [])
            if from_iso <= t.get("timestamp", "") <= to_iso
        ]
        return {"account_id": acct.get("account_id"), "transactions": txns}

    def _op_verify_ifsc(self, params: dict[str, Any]) -> dict[str, Any]:
        ifsc = params.get("ifsc", "")
        if not IFSC_RE.match(ifsc):
            raise ServiceError("BNK:2001", f"Invalid IFSC format: {ifsc}")
        return {"ifsc": ifsc, "valid": True, "bank_code": ifsc[:4], "branch_code": ifsc[5:]}

    # ── irreversible ──────────────────────────────────────────────────────

    def _op_add_beneficiary(self, params: dict[str, Any]) -> dict[str, Any]:
        account_number = params.get("account_number", "")
        ifsc = params.get("ifsc", "")
        name = params.get("name", "")
        nickname = params.get("nickname", "")
        if not IFSC_RE.match(ifsc):
            raise ServiceError("BNK:2001", f"Invalid IFSC format: {ifsc}")
        for b in self._beneficiaries:
            if b.get("account_number") == account_number and b.get("ifsc") == ifsc:
                raise ServiceError("BNK:3015", "Duplicate beneficiary")
        cooling_until = (datetime.now(timezone.utc) + timedelta(minutes=COOLING_MINUTES)).isoformat()
        beneficiary = {
            "beneficiary_id": str(uuid4()),
            "account_number": account_number,
            "ifsc": ifsc,
            "name": name,
            "nickname": nickname,
            "added_at": _now_iso(),
            "cooling_until_iso": cooling_until,
            "mule_flag": False,
        }
        self._beneficiaries.append(beneficiary)
        return dict(beneficiary)

    def _op_initiate_neft(self, params: dict[str, Any]) -> dict[str, Any]:
        beneficiary_id = params.get("beneficiary_id", "")
        amount = float(params.get("amount", 0.0))
        remarks = params.get("remarks", "")
        bene = self._find_beneficiary(beneficiary_id)
        if bene is None:
            raise ServiceError("BNK:3022", f"Beneficiary not found: {beneficiary_id}")
        if amount <= 0 or amount > NEFT_PER_TXN_CAP:
            raise ServiceError("BNK:4001", f"NEFT cap exceeded (₹{NEFT_PER_TXN_CAP:.0f})")
        if bene.get("mule_flag"):
            raise ServiceError("BNK:4050", "Beneficiary flagged as mule account")
        cooling_until = bene.get("cooling_until_iso", "")
        if amount > COOLING_THRESHOLD and cooling_until and _now_iso() < cooling_until:
            raise ServiceError("BNK:4012", f"Cooling-window violation; first >₹{COOLING_THRESHOLD:.0f} transfer blocked till {cooling_until}")
        acct = self._find_account(None)
        if acct is None:
            raise ServiceError("BNK:1001", "Source account not found")
        if amount > float(acct.get("balance", 0.0)):
            raise ServiceError("BNK:4001", "Insufficient balance for NEFT")
        neft = {
            "neft_ref_id": "N" + uuid4().hex[:15].upper(),
            "beneficiary_id": beneficiary_id,
            "amount": round(amount, 2),
            "remarks": remarks,
            "status": "INITIATED",
            "timestamp": _now_iso(),
        }
        acct["balance"] = float(acct.get("balance", 0.0)) - amount
        self._pending_neft.append(neft)
        return dict(neft)

    def _op_verify_cvv_for_cnp(self, params: dict[str, Any]) -> dict[str, Any]:
        card_id = params.get("card_id", "")
        cvv = params.get("cvv", "")
        amount = float(params.get("amount", 0.0))
        card = self._find_card(card_id)
        if card is None:
            raise ServiceError("BNK:5001", f"Card not found: {card_id}")
        attempts = self._cvv_attempt_counts.get(card_id, 0)
        if attempts >= CVV_LOCKOUT_THRESHOLD:
            raise ServiceError("BNK:5099", "Card locked after 3 failed CVV attempts")
        if cvv != card.get("cvv"):
            self._cvv_attempt_counts[card_id] = attempts + 1
            raise ServiceError("BNK:5001", "CVV mismatch")
        self._cvv_attempt_counts[card_id] = 0
        return {"card_id": card_id, "verified": True, "amount": round(amount, 2), "auth_ref": str(uuid4())}

    def _op_change_atm_pin(self, params: dict[str, Any]) -> dict[str, Any]:
        card_id = params.get("card_id", "")
        old_pin = params.get("old_pin", "")
        new_pin = params.get("new_pin", "")
        card = self._find_card(card_id)
        if card is None:
            raise ServiceError("BNK:6010", f"Card not found: {card_id}")
        if card.get("atm_pin") != old_pin:
            raise ServiceError("BNK:6010", "Old ATM PIN mismatch")
        card["atm_pin"] = new_pin
        card["pin_changed_at"] = _now_iso()
        return {"card_id": card_id, "pin_changed": True}

    # ── irreversible_trivial ──────────────────────────────────────────────

    def _op_remove_beneficiary(self, params: dict[str, Any]) -> dict[str, Any]:
        beneficiary_id = params.get("beneficiary_id", "")
        bene = self._find_beneficiary(beneficiary_id)
        if bene is None:
            raise ServiceError("BNK:3022", f"Beneficiary not found: {beneficiary_id}")
        self._beneficiaries = [b for b in self._beneficiaries if b.get("beneficiary_id") != beneficiary_id]
        return {"beneficiary_id": beneficiary_id, "removed": True}

    def _op_generate_virtual_card(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._virtual_cards_today >= VIRTUAL_CARD_DAILY_CAP:
            raise ServiceError("BNK:7001", f"Virtual card daily cap reached ({VIRTUAL_CARD_DAILY_CAP})")
        token = "VC-" + uuid4().hex[:12].upper()
        self._virtual_cards_today += 1
        return {"virtual_card_token": token, "issued_at": _now_iso(), "ttl_minutes": 60}
