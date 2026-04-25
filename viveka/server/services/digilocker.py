"""Mock DigiLocker service. Stubbed; full doc-locker behavior lands in Phase 1."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from viveka.server.services._base import MockService, ServiceError


class DigiLockerService(MockService):
    name = "digilocker"

    def reset(self, initial_state: dict[str, Any]) -> None:
        self._documents: list[dict[str, Any]] = list(initial_state.get("documents", []))
        self._consents: list[dict[str, Any]] = list(initial_state.get("consents", []))
        self._shared: list[dict[str, Any]] = list(initial_state.get("shared", []))

    def state(self) -> dict[str, Any]:
        return {
            "documents": list(self._documents),
            "consents": list(self._consents),
            "shared": list(self._shared),
        }

    # ── reversible ────────────────────────────────────────────────────────

    def _op_list_documents(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"documents": [{k: d[k] for k in ("doc_id", "doc_type", "issuer") if k in d} for d in self._documents]}

    def _op_view_document(self, params: dict[str, Any]) -> dict[str, Any]:
        doc_id = params.get("doc_id", "")
        for d in self._documents:
            if d.get("doc_id") == doc_id:
                return dict(d)
        raise ServiceError("DGL:404", f"Document not found: {doc_id}")

    def _op_fetch_document(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._op_view_document(params)

    def _op_list_consents(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"consents": list(self._consents)}

    def _op_revoke_consent(self, params: dict[str, Any]) -> dict[str, Any]:
        cid = params.get("consent_id", "")
        for c in self._consents:
            if c.get("consent_id") == cid:
                c["status"] = "revoked"
                return {"consent_id": cid, "status": "revoked"}
        raise ServiceError("DGL:601", f"Consent not found: {cid}")

    # ── irreversible ──────────────────────────────────────────────────────

    def _op_issue_consent_token(self, params: dict[str, Any]) -> dict[str, Any]:
        consent = {
            "consent_id": str(uuid4()),
            "doc_id": params.get("doc_id", ""),
            "audience": params.get("audience", ""),
            "scope": params.get("scope", []),
            "ttl_minutes": int(params.get("ttl_minutes", 30)),
            "status": "active",
        }
        self._consents.append(consent)
        return consent

    def _op_share_document(self, params: dict[str, Any]) -> dict[str, Any]:
        doc_id = params.get("doc_id", "")
        recipient = params.get("recipient", "")
        share = {"share_id": str(uuid4()), "doc_id": doc_id, "recipient": recipient}
        self._shared.append(share)
        return share

    def _op_delete_document(self, params: dict[str, Any]) -> dict[str, Any]:
        doc_id = params.get("doc_id", "")
        before = len(self._documents)
        self._documents = [d for d in self._documents if d.get("doc_id") != doc_id]
        if len(self._documents) == before:
            raise ServiceError("DGL:404", f"Document not found: {doc_id}")
        return {"doc_id": doc_id, "deleted": True}
