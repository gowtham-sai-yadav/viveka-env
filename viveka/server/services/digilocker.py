"""Mock DigiLocker service. Errors: DGL:404 doc-not-found, DGL:601 consent-not-found, DGL:702 consent-expired, DGL:801 invalid-scope, DGL:901 already-deleted."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from viveka.server.services._base import MockService, ServiceError


class DigiLockerService(MockService):
    name = "digilocker"

    def reset(self, initial_state: dict[str, Any]) -> None:
        self._documents: list[dict[str, Any]] = []
        for d in initial_state.get("documents", []):
            doc = dict(d)
            doc.setdefault("fetch_count", 0)
            self._documents.append(doc)
        self._consents: list[dict[str, Any]] = list(initial_state.get("consents", []))
        self._shared: list[dict[str, Any]] = list(initial_state.get("shared", []))
        self._deleted_doc_ids: set[str] = set(initial_state.get("deleted_doc_ids", []))

    def state(self) -> dict[str, Any]:
        return {
            "documents": list(self._documents),
            "consents": list(self._consents),
            "shared": list(self._shared),
        }

    def _find_doc(self, doc_id: str) -> dict[str, Any] | None:
        for d in self._documents:
            if d.get("doc_id") == doc_id:
                return d
        return None

    # ── reversible ────────────────────────────────────────────────────────

    def _op_list_documents(self, params: dict[str, Any]) -> dict[str, Any]:
        minimal = [
            {k: d[k] for k in ("doc_id", "doc_type", "issuer", "issued_at") if k in d}
            for d in self._documents
        ]
        return {"documents": minimal}

    def _op_view_document(self, params: dict[str, Any]) -> dict[str, Any]:
        doc_id = params.get("doc_id", "")
        doc = self._find_doc(doc_id)
        if doc is None:
            raise ServiceError("DGL:404", f"Document not found: {doc_id}")
        return dict(doc)

    def _op_fetch_document(self, params: dict[str, Any]) -> dict[str, Any]:
        doc_id = params.get("doc_id", "")
        doc = self._find_doc(doc_id)
        if doc is None:
            raise ServiceError("DGL:404", f"Document not found: {doc_id}")
        doc["fetch_count"] = int(doc.get("fetch_count", 0)) + 1
        return dict(doc)

    def _op_list_consents(self, params: dict[str, Any]) -> dict[str, Any]:
        status_filter = params.get("status")
        out = []
        for c in self._consents:
            if status_filter and c.get("status") != status_filter:
                continue
            entry = dict(c)
            entry.setdefault("created_at", c.get("created_at"))
            entry.setdefault("expires_at", c.get("expires_at"))
            out.append(entry)
        return {"consents": out}

    def _op_revoke_consent(self, params: dict[str, Any]) -> dict[str, Any]:
        cid = params.get("consent_id", "")
        for c in self._consents:
            if c.get("consent_id") == cid:
                if c.get("status") == "revoked":
                    return {"consent_id": cid, "status": "revoked", "already_revoked": True}
                c["status"] = "revoked"
                c["revoked_at"] = datetime.now(timezone.utc).isoformat()
                return {"consent_id": cid, "status": "revoked"}
        raise ServiceError("DGL:601", f"Consent not found: {cid}")

    # ── irreversible ──────────────────────────────────────────────────────

    def _op_issue_consent_token(self, params: dict[str, Any]) -> dict[str, Any]:
        doc_id = params.get("doc_id", "")
        audience = params.get("audience", "")
        scope = params.get("scope", [])
        ttl_minutes = int(params.get("ttl_minutes", 30))
        if self._find_doc(doc_id) is None:
            raise ServiceError("DGL:404", f"Document not found: {doc_id}")
        if not scope or not isinstance(scope, list):
            raise ServiceError("DGL:801", "Scope must be a non-empty list")
        created_dt = datetime.now(timezone.utc)
        expires_dt = created_dt + timedelta(minutes=ttl_minutes)
        consent = {
            "consent_id": str(uuid4()),
            "doc_id": doc_id,
            "audience": audience,
            "scope": list(scope),
            "ttl_minutes": ttl_minutes,
            "status": "active",
            "created_at": created_dt.isoformat(),
            "expires_at": expires_dt.isoformat(),
        }
        self._consents.append(consent)
        return dict(consent)

    def _op_share_document(self, params: dict[str, Any]) -> dict[str, Any]:
        doc_id = params.get("doc_id", "")
        recipient = params.get("recipient", "")
        if self._find_doc(doc_id) is None:
            raise ServiceError("DGL:404", f"Document not found: {doc_id}")
        share = {
            "share_id": str(uuid4()),
            "doc_id": doc_id,
            "recipient": recipient,
            "shared_at": datetime.now(timezone.utc).isoformat(),
        }
        self._shared.append(share)
        return dict(share)

    def _op_delete_document(self, params: dict[str, Any]) -> dict[str, Any]:
        doc_id = params.get("doc_id", "")
        if doc_id in self._deleted_doc_ids:
            raise ServiceError("DGL:901", "Document already deleted")
        if self._find_doc(doc_id) is None:
            raise ServiceError("DGL:404", f"Document not found: {doc_id}")
        cascaded: list[str] = []
        revoked_at = datetime.now(timezone.utc).isoformat()
        for c in self._consents:
            if c.get("doc_id") == doc_id and c.get("status") != "revoked":
                c["status"] = "revoked"
                c["revoked_at"] = revoked_at
                cascaded.append(c.get("consent_id", ""))
        self._documents = [d for d in self._documents if d.get("doc_id") != doc_id]
        self._deleted_doc_ids.add(doc_id)
        return {"doc_id": doc_id, "deleted": True, "cascaded_consents": cascaded}
