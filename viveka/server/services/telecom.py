"""Mock Telecom service with TRAI/DoT-flavored MNP, SIM-swap, and TAF-COP semantics.

References: TRAI MNP Regulations 2009 (UPC 4-day validity, 90-day lock-in),
DoT 2023 SIM-swap 24h cooling directive, TAF-COP 9-SIMs-per-Aadhaar cap,
CERT-In CIAD-2024-0019 (SIM-swap fraud advisory).

Error codes:
TEL:1001 sim not found | TEL:1010 kyc pending |
TEL:2001 OTP rate-limit | TEL:2010 DND-blocked |
TEL:3001 OTP expired | TEL:3010 OTP mismatch | TEL:3099 OTP max-attempts |
TEL:4001 SMS already blocked | TEL:5001 SIM already deactive |
TEL:6001 MNP lock-in 90d | TEL:6010 dues pending | TEL:6020 inactive >60d |
TEL:6099 UPC invalid | TEL:6100 port window expired | TEL:6105 MNP gateway locked |
TEL:7001 aadhaar invalid | TEL:7010 ekyc timeout | TEL:7099 TAF-COP 9-SIM cap |
TEL:8001 kyc fail | TEL:8010 biometric mismatch | TEL:8099 SIM-swap cooling 24h
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from viveka.server.services._base import MockService, ServiceError

IST = timezone(timedelta(hours=5, minutes=30))
TAF_COP_CAP = 9
UPC_VALIDITY_DAYS = 4
MNP_LOCK_IN_DAYS = 90
SIM_SWAP_COOLING_HOURS = 24
INACTIVE_PORT_THRESHOLD_DAYS = 60
OTP_TTL_SECONDS = 600
OTP_MAX_ATTEMPTS = 3
OTP_RATE_LIMIT = 5


def _gen_upc() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def _gen_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt


class TelecomService(MockService):
    name = "telecom"

    def reset(self, initial_state: dict[str, Any]) -> None:
        # Accept list-of-dicts in `initial_state` (scenario JSON convention,
        # matching upi/digilocker/irctc patterns). Convert to dict-of-dicts
        # keyed by msisdn internally for efficient lookup/mutation in ops.
        # Also tolerate dict-of-dicts on input for backward compatibility.
        def _to_dict_by_msisdn(raw: Any) -> dict[str, dict[str, Any]]:
            if isinstance(raw, list):
                return {item["msisdn"]: dict(item)
                        for item in raw
                        if isinstance(item, dict) and "msisdn" in item}
            if isinstance(raw, dict):
                return {m: dict(s) for m, s in raw.items()}
            return {}
        self._sims: dict[str, dict[str, Any]] = _to_dict_by_msisdn(
            initial_state.get("sims", []))
        self._active_ports: dict[str, dict[str, Any]] = _to_dict_by_msisdn(
            initial_state.get("active_ports", []))
        self._otps: dict[str, dict[str, Any]] = _to_dict_by_msisdn(
            initial_state.get("otps", []))
        self._now_iso: str = initial_state.get("now_iso", "2026-04-25T08:00:00+05:30")

    def state(self) -> dict[str, Any]:
        # Emit list-of-dicts so scenario post_state assertions (which use the
        # same convention as upi/digilocker/irctc) match correctly.
        return {
            "sims": [dict(s) for s in self._sims.values()],
            "active_ports": [dict(p) for p in self._active_ports.values()],
            "otps": [dict(o) for o in self._otps.values()],
            "now_iso": self._now_iso,
        }

    def _now_dt(self) -> datetime:
        return _parse_iso(self._now_iso)

    def _get_sim(self, msisdn: str) -> dict[str, Any]:
        sim = self._sims.get(msisdn)
        if sim is None:
            raise ServiceError("TEL:1001", f"SIM not found: {msisdn}")
        return sim

    # ── reversible ────────────────────────────────────────────────────────

    def _op_check_sim_status(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        sim = self._get_sim(msisdn)
        if sim.get("kyc_status") == "pending":
            raise ServiceError("TEL:1010", f"KYC pending for {msisdn}")
        return {
            "msisdn": msisdn,
            "operator": sim.get("operator", ""),
            "circle": sim.get("circle", ""),
            "status": sim.get("status", "active"),
            "kyc_status": sim.get("kyc_status", "verified"),
            "aadhaar_last4": sim.get("aadhaar_last4"),
            "activated_at": sim.get("activated_at"),
        }

    def _op_check_taf_cop(self, params: dict[str, Any]) -> dict[str, Any]:
        last4 = params.get("aadhaar_last4", "")
        linked = [m for m, s in self._sims.items() if s.get("aadhaar_last4") == last4]
        return {
            "aadhaar_last4": last4,
            "sim_count": len(linked),
            "msisdns": linked,
            "cap": TAF_COP_CAP,
            "remaining": max(0, TAF_COP_CAP - len(linked)),
        }

    def _op_send_otp(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        purpose = params.get("purpose", "verify")
        sim = self._get_sim(msisdn)
        if sim.get("dnd_blocked"):
            raise ServiceError("TEL:2010", f"DND-blocked: {msisdn}")
        existing = self._otps.get(msisdn, {})
        sent_count = int(existing.get("sent_count", 0))
        if sent_count >= OTP_RATE_LIMIT:
            raise ServiceError("TEL:2001", f"OTP rate-limit exceeded for {msisdn}")
        otp = _gen_otp()
        self._otps[msisdn] = {
            "otp": otp,
            "purpose": purpose,
            "issued_at": self._now_iso,
            "attempts": 0,
            "sent_count": sent_count + 1,
        }
        return {"msisdn": msisdn, "purpose": purpose, "otp_ref": str(uuid4()), "issued_at": self._now_iso}

    def _op_verify_otp(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        otp = params.get("otp", "")
        record = self._otps.get(msisdn)
        if record is None:
            raise ServiceError("TEL:3010", f"No OTP issued for {msisdn}")
        if int(record.get("attempts", 0)) >= OTP_MAX_ATTEMPTS:
            raise ServiceError("TEL:3099", "OTP max attempts exceeded")
        age = (self._now_dt() - _parse_iso(record["issued_at"])).total_seconds()
        if age > OTP_TTL_SECONDS:
            raise ServiceError("TEL:3001", "OTP expired")
        record["attempts"] = int(record.get("attempts", 0)) + 1
        if record.get("otp") != otp:
            raise ServiceError("TEL:3010", "OTP mismatch")
        record["verified"] = True
        return {"msisdn": msisdn, "verified": True, "purpose": record.get("purpose")}

    # ── irreversible_trivial ──────────────────────────────────────────────

    def _op_block_sms(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        sim = self._get_sim(msisdn)
        if sim.get("dnd_blocked"):
            raise ServiceError("TEL:4001", f"SMS already blocked on {msisdn}")
        sim["dnd_blocked"] = True
        sim["dnd_blocked_at"] = self._now_iso
        return {"msisdn": msisdn, "dnd_blocked": True}

    def _op_deactivate_sim(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        sim = self._get_sim(msisdn)
        if sim.get("status") == "deactivated":
            raise ServiceError("TEL:5001", f"SIM already deactivated: {msisdn}")
        sim["status"] = "deactivated"
        sim["deactivated_at"] = self._now_iso
        sim["reactivation_window_days"] = 30
        return {"msisdn": msisdn, "status": "deactivated", "reactivation_window_days": 30}

    # ── irreversible ──────────────────────────────────────────────────────

    def _op_request_port_out(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        target_operator = params.get("target_operator", "")
        sim = self._get_sim(msisdn)
        last_port_iso = sim.get("last_port_at")
        if last_port_iso:
            days_since = (self._now_dt() - _parse_iso(last_port_iso)).total_seconds() / 86400.0
            if days_since < MNP_LOCK_IN_DAYS:
                raise ServiceError("TEL:6001", f"MNP lock-in {MNP_LOCK_IN_DAYS}d not elapsed")
        if float(sim.get("dues_inr", 0.0)) > 0:
            raise ServiceError("TEL:6010", "Outstanding dues pending")
        last_active_iso = sim.get("last_active_at", sim.get("activated_at"))
        if last_active_iso:
            days_inactive = (self._now_dt() - _parse_iso(last_active_iso)).total_seconds() / 86400.0
            if days_inactive > INACTIVE_PORT_THRESHOLD_DAYS:
                raise ServiceError("TEL:6020", f"SIM inactive >{INACTIVE_PORT_THRESHOLD_DAYS}d")
        upc = _gen_upc()
        self._active_ports[msisdn] = {
            "upc": upc,
            "target_operator": target_operator,
            "issued_at": self._now_iso,
            "expires_at": (self._now_dt() + timedelta(days=UPC_VALIDITY_DAYS)).isoformat(),
            "status": "issued",
        }
        return {"msisdn": msisdn, "upc": upc, "target_operator": target_operator,
                "valid_days": UPC_VALIDITY_DAYS}

    def _op_confirm_port_out(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        upc = params.get("upc", "")
        port = self._active_ports.get(msisdn)
        if port is None or port.get("upc") != upc:
            raise ServiceError("TEL:6099", "UPC invalid")
        if port.get("status") == "gateway_locked":
            raise ServiceError("TEL:6105", "MNP gateway locked")
        if self._now_dt() > _parse_iso(port["expires_at"]):
            raise ServiceError("TEL:6100", "Port window expired")
        sim = self._get_sim(msisdn)
        sim["operator"] = port["target_operator"]
        sim["last_port_at"] = self._now_iso
        port["status"] = "completed"
        port["completed_at"] = self._now_iso
        return {"msisdn": msisdn, "operator": sim["operator"], "status": "ported"}

    def _op_link_aadhaar_to_sim(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        last4 = params.get("aadhaar_last4", "")
        sim = self._get_sim(msisdn)
        if not last4 or not last4.isdigit() or len(last4) != 4:
            raise ServiceError("TEL:7001", "Aadhaar last-4 invalid")
        if params.get("ekyc_timeout"):
            raise ServiceError("TEL:7010", "eKYC timeout")
        linked = [m for m, s in self._sims.items() if s.get("aadhaar_last4") == last4 and m != msisdn]
        if len(linked) >= TAF_COP_CAP:
            raise ServiceError("TEL:7099", f"TAF-COP cap of {TAF_COP_CAP} SIMs reached")
        sim["aadhaar_last4"] = last4
        sim["kyc_status"] = "verified"
        sim["ekyc_at"] = self._now_iso
        return {"msisdn": msisdn, "aadhaar_last4": last4, "kyc_status": "verified"}

    def _op_request_new_sim(self, params: dict[str, Any]) -> dict[str, Any]:
        msisdn = params.get("msisdn", "")
        last4 = params.get("aadhaar_last4", "")
        biometric_consent = bool(params.get("biometric_consent", False))
        if not last4 or not last4.isdigit() or len(last4) != 4:
            raise ServiceError("TEL:8001", "KYC failure: aadhaar last-4 invalid")
        if not biometric_consent:
            raise ServiceError("TEL:8010", "Biometric consent / mismatch")
        existing = self._sims.get(msisdn)
        if existing is not None:
            last_port_iso = existing.get("last_port_at")
            if last_port_iso:
                hours = (self._now_dt() - _parse_iso(last_port_iso)).total_seconds() / 3600.0
                if hours < SIM_SWAP_COOLING_HOURS:
                    raise ServiceError("TEL:8099", f"SIM-swap cooling {SIM_SWAP_COOLING_HOURS}h active")
        linked = [m for m, s in self._sims.items() if s.get("aadhaar_last4") == last4]
        if len(linked) >= TAF_COP_CAP:
            raise ServiceError("TEL:7099", f"TAF-COP cap of {TAF_COP_CAP} SIMs reached")
        self._sims[msisdn] = {
            "operator": params.get("operator", "Jio"),
            "circle": params.get("circle", "KA"),
            "status": "active",
            "kyc_status": "verified",
            "aadhaar_last4": last4,
            "activated_at": self._now_iso,
            "last_active_at": self._now_iso,
            "dnd_blocked": False,
            "dues_inr": 0.0,
        }
        return {"msisdn": msisdn, "status": "active", "sim_id": str(uuid4())}
