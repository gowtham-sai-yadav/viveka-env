"""Tests for DigiLockerService — issued docs, consent TTL, share audit, delete cascade."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from viveka.server.services._base import ServiceError
from viveka.server.services.digilocker import DigiLockerService


def _seed() -> DigiLockerService:
    svc = DigiLockerService()
    svc.reset({
        "documents": [
            {
                "doc_id": "DGL-AADHAAR-001",
                "doc_type": "aadhaar",
                "issuer": "UIDAI",
                "issued_at": "2024-01-15T10:00:00+00:00",
                "data": {"name": "Gowtham", "dob": "1998-05-12"},
                "attributes": {"verified": True},
            },
            {
                "doc_id": "DGL-PAN-002",
                "doc_type": "pan",
                "issuer": "Income Tax Dept",
                "issued_at": "2023-08-01T09:00:00+00:00",
                "data": {"pan_number": "ABCDE1234F"},
            },
        ],
        "consents": [
            {
                "consent_id": "consent-existing-1",
                "doc_id": "DGL-AADHAAR-001",
                "audience": "icici-bank",
                "scope": ["read"],
                "status": "active",
                "created_at": "2026-04-25T08:00:00+00:00",
                "expires_at": "2026-04-25T08:30:00+00:00",
            },
            {
                "consent_id": "consent-existing-2",
                "doc_id": "DGL-PAN-002",
                "audience": "hdfc-bank",
                "scope": ["read"],
                "status": "revoked",
                "created_at": "2026-04-20T08:00:00+00:00",
                "expires_at": "2026-04-20T08:30:00+00:00",
            },
        ],
    })
    return svc


def test_list_documents_minimal_fields_only():
    svc = _seed()
    out = svc.execute("list_documents", {})
    docs = out["documents"]
    assert len(docs) == 2
    for d in docs:
        assert set(d.keys()) <= {"doc_id", "doc_type", "issuer", "issued_at"}
        assert "data" not in d
        assert "attributes" not in d


def test_view_document_returns_full_doc():
    svc = _seed()
    out = svc.execute("view_document", {"doc_id": "DGL-AADHAAR-001"})
    assert out["doc_id"] == "DGL-AADHAAR-001"
    assert out["data"]["name"] == "Gowtham"
    assert out["attributes"]["verified"] is True


def test_view_document_missing_raises_404():
    svc = _seed()
    with pytest.raises(ServiceError) as exc:
        svc.execute("view_document", {"doc_id": "DGL-NOPE-999"})
    assert exc.value.code == "DGL:404"


def test_fetch_document_increments_fetch_count():
    svc = _seed()
    first = svc.execute("fetch_document", {"doc_id": "DGL-AADHAAR-001"})
    assert first["fetch_count"] == 1
    second = svc.execute("fetch_document", {"doc_id": "DGL-AADHAAR-001"})
    assert second["fetch_count"] == 2
    state = svc.state()
    found = next(d for d in state["documents"] if d["doc_id"] == "DGL-AADHAAR-001")
    assert found["fetch_count"] == 2


def test_list_consents_filters_by_status():
    svc = _seed()
    active = svc.execute("list_consents", {"status": "active"})
    assert len(active["consents"]) == 1
    assert active["consents"][0]["consent_id"] == "consent-existing-1"
    assert active["consents"][0]["created_at"] is not None
    assert active["consents"][0]["expires_at"] is not None
    revoked = svc.execute("list_consents", {"status": "revoked"})
    assert len(revoked["consents"]) == 1
    assert revoked["consents"][0]["consent_id"] == "consent-existing-2"
    all_consents = svc.execute("list_consents", {})
    assert len(all_consents["consents"]) == 2


def test_issue_consent_token_happy_path_sets_expires_at():
    svc = _seed()
    before = datetime.now(timezone.utc)
    out = svc.execute("issue_consent_token", {
        "doc_id": "DGL-AADHAAR-001",
        "audience": "axis-bank",
        "scope": ["read", "verify"],
        "ttl_minutes": 60,
    })
    assert out["status"] == "active"
    assert out["doc_id"] == "DGL-AADHAAR-001"
    assert out["audience"] == "axis-bank"
    assert out["scope"] == ["read", "verify"]
    assert out["ttl_minutes"] == 60
    created = datetime.fromisoformat(out["created_at"])
    expires = datetime.fromisoformat(out["expires_at"])
    delta = expires - created
    assert delta == timedelta(minutes=60)
    assert created >= before - timedelta(seconds=2)


def test_issue_consent_token_missing_doc_raises_404():
    svc = _seed()
    with pytest.raises(ServiceError) as exc:
        svc.execute("issue_consent_token", {
            "doc_id": "DGL-NOPE-999",
            "audience": "axis-bank",
            "scope": ["read"],
        })
    assert exc.value.code == "DGL:404"


def test_issue_consent_token_empty_scope_raises_801():
    svc = _seed()
    with pytest.raises(ServiceError) as exc:
        svc.execute("issue_consent_token", {
            "doc_id": "DGL-AADHAAR-001",
            "audience": "axis-bank",
            "scope": [],
        })
    assert exc.value.code == "DGL:801"


def test_revoke_consent_happy_path_flips_status():
    svc = _seed()
    out = svc.execute("revoke_consent", {"consent_id": "consent-existing-1"})
    assert out["status"] == "revoked"
    state = svc.state()
    consent = next(c for c in state["consents"] if c["consent_id"] == "consent-existing-1")
    assert consent["status"] == "revoked"
    assert "revoked_at" in consent


def test_revoke_consent_idempotent_on_already_revoked():
    svc = _seed()
    out = svc.execute("revoke_consent", {"consent_id": "consent-existing-2"})
    assert out["status"] == "revoked"
    assert out.get("already_revoked") is True


def test_revoke_consent_missing_raises_601():
    svc = _seed()
    with pytest.raises(ServiceError) as exc:
        svc.execute("revoke_consent", {"consent_id": "consent-nonexistent"})
    assert exc.value.code == "DGL:601"


def test_share_document_happy_and_missing_doc():
    svc = _seed()
    out = svc.execute("share_document", {
        "doc_id": "DGL-PAN-002",
        "recipient": "employer@example.com",
    })
    assert out["doc_id"] == "DGL-PAN-002"
    assert out["recipient"] == "employer@example.com"
    assert "share_id" in out
    assert "shared_at" in out
    state = svc.state()
    assert len(state["shared"]) == 1
    with pytest.raises(ServiceError) as exc:
        svc.execute("share_document", {
            "doc_id": "DGL-NOPE-999",
            "recipient": "employer@example.com",
        })
    assert exc.value.code == "DGL:404"


def test_delete_document_cascades_consent_revocation():
    svc = _seed()
    out = svc.execute("delete_document", {"doc_id": "DGL-AADHAAR-001"})
    assert out["deleted"] is True
    assert "consent-existing-1" in out["cascaded_consents"]
    state = svc.state()
    assert all(d["doc_id"] != "DGL-AADHAAR-001" for d in state["documents"])
    consent = next(c for c in state["consents"] if c["consent_id"] == "consent-existing-1")
    assert consent["status"] == "revoked"
    assert "revoked_at" in consent


def test_delete_document_already_deleted_raises_901():
    svc = _seed()
    svc.execute("delete_document", {"doc_id": "DGL-PAN-002"})
    with pytest.raises(ServiceError) as exc:
        svc.execute("delete_document", {"doc_id": "DGL-PAN-002"})
    assert exc.value.code == "DGL:901"


def test_delete_document_missing_raises_404():
    svc = _seed()
    with pytest.raises(ServiceError) as exc:
        svc.execute("delete_document", {"doc_id": "DGL-NOPE-999"})
    assert exc.value.code == "DGL:404"
