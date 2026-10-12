"""Tests for the Healthcare Management System."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.data_store import (
    get_doctor, get_patient, recommend_doctors_for_symptoms,
    book_appointment, DOCTORS, PATIENTS
)
from app.rag import _chunk
from app.models import UrgencyLevel, AppointmentStatus

client = TestClient(app)


# ── Data store tests ──────────────────────────────────────────────
def test_get_doctor_exists():
    doc = get_doctor("D001")
    assert doc is not None
    assert "name" in doc
    assert "specialisation" in doc

def test_get_doctor_not_found():
    assert get_doctor("D999") is None

def test_get_patient_exists():
    p = get_patient("P001")
    assert p is not None
    assert p["name"] == "Arjun Kumar"

def test_recommend_doctors_headache():
    docs = recommend_doctors_for_symptoms(["severe headache"])
    ids = [d["doctor_id"] for d in docs]
    # Should include GP (D001) and/or Neurologist (D002)
    assert any(did in ids for did in ["D001","D002"])

def test_recommend_doctors_always_includes_gp():
    docs = recommend_doctors_for_symptoms(["chest pain"])
    ids = [d["doctor_id"] for d in docs]
    assert "D001" in ids  # GP always included

def test_book_appointment_sandbox():
    result = book_appointment("P001","D001","2024-12-20 10:00","headache test",sandbox=True)
    assert result["success"] is True
    assert result.get("sandbox") is True
    assert "appointment_id" in result

def test_book_appointment_invalid_doctor():
    result = book_appointment("P001","D999","2024-12-20 10:00","test",sandbox=True)
    assert result["success"] is False


# ── RAG chunking tests ────────────────────────────────────────────
def test_chunk_short_text():
    text = "Short medical text."
    chunks = _chunk(text)
    assert len(chunks) == 1
    assert chunks[0] == "Short medical text."

def test_chunk_empty():
    assert _chunk("") == []

def test_chunk_long_text():
    text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five."
    chunks = _chunk(text)
    assert len(chunks) >= 1


# ── API endpoint tests ────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    assert "total_chunks" in r.json()

def test_get_patient_p001():
    r = client.get("/patients/P001")
    assert r.status_code == 200
    assert r.json()["name"] == "Arjun Kumar"

def test_get_patient_not_found():
    r = client.get("/patients/P999")
    assert r.status_code == 404

def test_query_empty():
    r = client.post("/query", json={"patient_id":"P001","query":""})
    assert r.status_code == 400

def test_query_too_long():
    r = client.post("/query", json={"patient_id":"P001","query":"x"*2001})
    assert r.status_code == 400

MOCK_RESPONSE = {
    "patient_id":"P001","query":"headache and fever",
    "symptom_analysis":{"urgency":"soon","possible_conditions":["Tension headache","Viral fever"],
        "recommended_specialisation":"General Physician","red_flags":[],
        "home_care_advice":["Rest","Stay hydrated"],"disclaimer":"Consult a doctor."},
    "recommended_doctors":[{"doctor_id":"D001","name":"Dr. Priya Sharma",
        "specialisation":"General Physician","available_slots":["2024-12-20 10:00"],"reason":"Initial evaluation"}],
    "appointment":{"appointment_id":"A0001","patient_id":"P001","doctor_name":"Dr. Priya Sharma",
        "specialisation":"General Physician","scheduled_time":"2024-12-20 10:00",
        "location":"OPD Block A","preparation_notes":"Bring reports","status":"booked"},
    "contextual_summary":"Based on your symptoms...","sources_used":["Neurological Guide"],
    "agents_involved":["SymptomAnalysisAgent","DoctorMatchingAgent","AppointmentAgent"],
    "disclaimer":"This is AI-assisted guidance only."
}

@patch("app.main.healthcare_orchestrator")
def test_query_success(mock_orch):
    from app.models import HealthcareResponse
    mock_orch.return_value = HealthcareResponse(**MOCK_RESPONSE)
    r = client.post("/query", json={"patient_id":"P001","query":"I have headache and fever"})
    assert r.status_code == 200
    data = r.json()
    assert "symptom_analysis" in data
    assert "recommended_doctors" in data
    assert len(data["agents_involved"]) >= 1
