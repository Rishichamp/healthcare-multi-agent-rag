"""
AI-Powered Healthcare Management System — FastAPI Application
=============================================================
Endpoints:
  POST /query              — process patient health query through multi-agent pipeline
  POST /appointments       — book an appointment directly
  GET  /patients/{id}      — get patient profile
  POST /ingest/directory   — ingest medical knowledge base documents
  GET  /stats              — system statistics
  GET  /health             — health check
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings
from app.models import PatientQuery, HealthcareResponse, AppointmentRequest
from app.coordinator import healthcare_orchestrator
from app.data_store import get_patient, book_appointment
from app.rag import ingest_directory, get_collection_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Healthcare Management System",
    description="Multi-agent AI system for symptom analysis, doctor matching, and appointment scheduling.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
    <h1>🏥 AI Healthcare Management System</h1>
    <p>Multi-agent: SymptomAnalysis + DoctorMatching + AppointmentScheduling + RAG</p>
    <p><a href="/docs">📖 API Docs</a></p>
    <h3>Sample patient IDs: P001, P002, P003</h3>
    </body></html>"""


@app.get("/health")
async def health():
    stats = get_collection_stats()
    return {"status": "ok", "sandbox_mode": settings.SANDBOX_MODE, **stats}


@app.post("/query", response_model=HealthcareResponse)
async def process_query(req: PatientQuery):
    """
    Process a patient health query through the full multi-agent pipeline:
    1. SymptomAnalysisAgent — assesses urgency, identifies red flags
    2. DoctorMatchingAgent  — recommends appropriate specialists (RAG-grounded)
    3. AppointmentAgent     — schedules appointment with best available doctor
    """
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    if len(req.query) > 2000:
        raise HTTPException(400, "Query too long.")
    try:
        return healthcare_orchestrator(req.patient_id, req.query)
    except Exception as e:
        logger.error(f"Healthcare pipeline error: {e}", exc_info=True)
        raise HTTPException(500, f"Processing error: {str(e)}")


@app.get("/patients/{patient_id}")
async def get_patient_profile(patient_id: str):
    patient = get_patient(patient_id)
    if not patient:
        raise HTTPException(404, f"Patient {patient_id} not found.")
    return {"patient_id": patient_id, **patient}


@app.post("/ingest/directory")
async def ingest_knowledge_base():
    """Ingest all medical documents from the configured DOCS_PATH directory."""
    try:
        results = ingest_directory()
        return {"status":"completed","ingested":results["success"],"failed":results["failed"]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/stats")
async def get_stats():
    return {**get_collection_stats(), "sandbox_mode": settings.SANDBOX_MODE,
            "orchestrator_model": settings.ORCHESTRATOR_MODEL, "worker_model": settings.WORKER_MODEL}
