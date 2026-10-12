"""
Healthcare Multi-Agent Coordinator.

Three specialist agents coordinated by an orchestrator:
  1. SymptomAnalysisAgent  — analyses symptoms, assesses urgency, checks for red flags
  2. DoctorMatchingAgent   — recommends appropriate specialists based on symptom analysis
  3. AppointmentAgent      — schedules appointments and generates patient summaries

The orchestrator calls each agent in sequence, passing outputs forward as context.
RAG grounds every agent's response in the medical knowledge base.
"""
import json, logging, re
from typing import Optional
from openai import OpenAI

from app.config import settings
from app.data_store import (
    get_patient, recommend_doctors_for_symptoms,
    book_appointment, get_doctors_by_specialisation
)
from app.rag import get_medical_context
from app.models import (
    SymptomAnalysis, DoctorRecommendation, AppointmentConfirmation,
    HealthcareResponse, UrgencyLevel, AppointmentStatus
)

logger = logging.getLogger(__name__)


def _llm(messages: list[dict], model: str = None, temperature: float = 0.1) -> str:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    model = model or settings.WORKER_MODEL
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=1024
    )
    return resp.choices[0].message.content


def _parse_json(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


# ── Agent 1: Symptom Analysis ─────────────────────────────────────────────────

def symptom_analysis_agent(symptoms_description: str, patient_data: dict) -> SymptomAnalysis:
    """
    Analyses symptoms using both LLM reasoning and RAG-retrieved medical context.
    Returns a structured SymptomAnalysis with urgency, possible conditions, and red flags.
    """
    logger.info("[SymptomAgent] Analysing symptoms...")

    # RAG: retrieve relevant medical knowledge
    medical_context = get_medical_context(symptoms_description)

    patient_context = ""
    if patient_data:
        patient_context = (
            f"Patient age: {patient_data.get('age','unknown')}, "
            f"existing conditions: {', '.join(patient_data.get('conditions',[]) or ['none'])}, "
            f"current medications: {', '.join(patient_data.get('medications',[]) or ['none'])}, "
            f"allergies: {', '.join(patient_data.get('allergies',[]) or ['none'])}"
        )

    system = """You are a clinical triage specialist. Analyse symptoms and return structured JSON.
IMPORTANT: This is for triage assistance only — always recommend professional consultation.
Respond ONLY with valid JSON, no preamble."""

    user = f"""Analyse these symptoms and return a JSON triage assessment.

Patient context: {patient_context or 'Not provided'}
Symptoms: {symptoms_description}

{"Medical knowledge context:" + chr(10) + medical_context if medical_context else ""}

JSON schema:
{{
  "urgency": "routine" | "soon" | "urgent" | "emergency",
  "possible_conditions": [list of 2-4 possible conditions, general not definitive],
  "recommended_specialisation": "most appropriate medical speciality",
  "red_flags": [list of any serious warning signs found, or empty list],
  "home_care_advice": [list of 2-3 safe home care steps for non-emergency cases],
  "disclaimer": "Always consult a qualified doctor for proper diagnosis."
}}

urgency guide: emergency=immediate ER | urgent=within 24h | soon=within week | routine=scheduled appt"""

    raw = _llm([{"role":"system","content":system},{"role":"user","content":user}])
    data = _parse_json(raw)

    return SymptomAnalysis(
        urgency=UrgencyLevel(data.get("urgency","routine")),
        possible_conditions=data.get("possible_conditions",[]),
        recommended_specialisation=data.get("recommended_specialisation","General Physician"),
        red_flags=data.get("red_flags",[]),
        home_care_advice=data.get("home_care_advice",[]),
        disclaimer=data.get("disclaimer","Always consult a qualified doctor.")
    )


# ── Agent 2: Doctor Matching ──────────────────────────────────────────────────

def doctor_matching_agent(
    symptom_analysis: SymptomAnalysis,
    symptoms_description: str
) -> list[DoctorRecommendation]:
    """
    Matches patient symptoms to appropriate doctors using:
    1. Data-driven keyword matching (recommend_doctors_for_symptoms)
    2. LLM-generated reasons for each recommendation
    """
    logger.info("[DoctorAgent] Matching doctors...")

    # Get candidate doctors from data store
    candidates = recommend_doctors_for_symptoms([symptoms_description])
    # Also try by specialisation from symptom analysis
    spec_matches = get_doctors_by_specialisation(symptom_analysis.recommended_specialisation)
    seen = {d["doctor_id"] for d in candidates}
    for doc in spec_matches:
        if doc["doctor_id"] not in seen:
            candidates.append(doc)
            seen.add(doc["doctor_id"])

    if not candidates:
        candidates = [{"doctor_id":"D001","name":"Dr. Priya Sharma",
                       "specialisation":"General Physician",
                       "available_slots":["2024-12-20 10:00"],"location":"OPD Block A","rating":4.8}]

    # Use LLM to rank and explain recommendations
    candidates_text = json.dumps([
        {"id":d["doctor_id"],"name":d["name"],"spec":d["specialisation"],"slots":d.get("available_slots",[])}
        for d in candidates[:6]
    ])

    user = f"""Given these symptoms: {symptoms_description}
Urgency: {symptom_analysis.urgency}
Recommended specialisation: {symptom_analysis.recommended_specialisation}

Available doctors: {candidates_text}

Rank the top 3 most appropriate doctors and explain why each is suitable.
Return JSON: [{{"doctor_id":str,"reason":str}}]"""

    raw = _llm([{"role":"user","content":user}])
    try:
        rankings = _parse_json(raw)
        if not isinstance(rankings, list):
            rankings = rankings.get("doctors", list(rankings.values())[:3])
    except Exception:
        rankings = [{"doctor_id": d["doctor_id"], "reason": "Matches symptom profile"} for d in candidates[:3]]

    recommendations = []
    doc_map = {d["doctor_id"]: d for d in candidates}
    for rank in rankings[:3]:
        did = rank.get("doctor_id","")
        doc = doc_map.get(did)
        if not doc:
            continue
        recommendations.append(DoctorRecommendation(
            doctor_id=did,
            name=doc["name"],
            specialisation=doc["specialisation"],
            available_slots=doc.get("available_slots",[]),
            reason=rank.get("reason","Recommended based on symptom analysis")
        ))

    return recommendations or [DoctorRecommendation(
        doctor_id="D001",name="Dr. Priya Sharma",specialisation="General Physician",
        available_slots=["2024-12-20 10:00"],reason="General practitioner for initial assessment"
    )]


# ── Agent 3: Appointment Scheduling ──────────────────────────────────────────

def appointment_agent(
    patient_id: str,
    recommended_doctors: list[DoctorRecommendation],
    reason: str,
    urgency: UrgencyLevel,
) -> Optional[AppointmentConfirmation]:
    """
    Attempts to book an appointment with the highest-ranked available doctor.
    Respects sandbox mode — books are logged but not persisted when SANDBOX_MODE=True.
    """
    logger.info("[AppointmentAgent] Scheduling appointment...")

    for doc_rec in recommended_doctors:
        if not doc_rec.available_slots:
            continue
        # For urgent cases, pick earliest slot; for routine, first slot is fine
        slot = sorted(doc_rec.available_slots)[0] if urgency in (UrgencyLevel.URGENT, UrgencyLevel.EMERGENCY) else doc_rec.available_slots[0]
        result = book_appointment(patient_id, doc_rec.doctor_id, slot, reason, sandbox=settings.SANDBOX_MODE)
        if result["success"]:
            logger.info(f"[AppointmentAgent] Booked {doc_rec.doctor_id} at {slot} (sandbox={settings.SANDBOX_MODE})")
            return AppointmentConfirmation(
                appointment_id=result["appointment_id"],
                patient_id=patient_id,
                doctor_name=result["doctor_name"],
                specialisation=result["specialisation"],
                scheduled_time=result["scheduled_time"],
                location=result["location"],
                preparation_notes=result["preparation_notes"],
                status=AppointmentStatus.BOOKED,
            )
    return None


# ── Orchestrator ──────────────────────────────────────────────────────────────

def healthcare_orchestrator(patient_id: str, query: str) -> HealthcareResponse:
    """
    Coordinates the three healthcare agents in sequence:
      1. SymptomAnalysis  → assesses urgency and possible conditions
      2. DoctorMatching   → recommends appropriate specialists
      3. Appointment      → schedules the appointment

    Uses RAG throughout to ground responses in medical documentation.
    """
    logger.info(f"[Orchestrator] Processing query for patient {patient_id}")
    agents_involved = []

    # Get patient context
    patient_data = get_patient(patient_id) or {}
    patient_name = patient_data.get("name", f"Patient {patient_id}")

    # Agent 1: Symptom analysis
    agents_involved.append("SymptomAnalysisAgent")
    symptom_analysis = symptom_analysis_agent(query, patient_data)

    # Emergency check — stop and direct to ER immediately
    if symptom_analysis.urgency == UrgencyLevel.EMERGENCY:
        return HealthcareResponse(
            patient_id=patient_id,
            query=query,
            symptom_analysis=symptom_analysis,
            recommended_doctors=[],
            appointment=None,
            contextual_summary=(
                f"⚠️ EMERGENCY ALERT for {patient_name}: "
                "Symptoms indicate a potentially life-threatening condition. "
                "Please go to the nearest Emergency Room IMMEDIATELY or call 112. "
                "Do not wait for an appointment."
            ),
            sources_used=[],
            agents_involved=agents_involved,
        )

    # Agent 2: Doctor matching
    agents_involved.append("DoctorMatchingAgent")
    recommended_doctors = doctor_matching_agent(symptom_analysis, query)

    # Agent 3: Appointment scheduling
    appointment = None
    if recommended_doctors and symptom_analysis.urgency != UrgencyLevel.ROUTINE:
        agents_involved.append("AppointmentAgent")
        appointment = appointment_agent(patient_id, recommended_doctors, query, symptom_analysis.urgency)

    # Generate contextual summary using orchestrator model
    rag_context = get_medical_context(query)
    summary_prompt = f"""Generate a compassionate, clear patient communication summary.

Patient: {patient_name}
Query: {query}
Urgency: {symptom_analysis.urgency}
Possible conditions: {', '.join(symptom_analysis.possible_conditions)}
Red flags: {', '.join(symptom_analysis.red_flags) if symptom_analysis.red_flags else 'None'}
Appointment: {"Scheduled with " + appointment.doctor_name + " at " + appointment.scheduled_time if appointment else "Not scheduled - routine case"}
Home care: {', '.join(symptom_analysis.home_care_advice)}

{"Relevant medical context: " + rag_context[:500] if rag_context else ""}

Write a 3-4 sentence summary for the patient. Be caring but clear. Include next steps."""

    summary = _llm(
        [{"role":"user","content":summary_prompt}],
        model=settings.ORCHESTRATOR_MODEL,
        temperature=0.3
    )

    return HealthcareResponse(
        patient_id=patient_id,
        query=query,
        symptom_analysis=symptom_analysis,
        recommended_doctors=recommended_doctors,
        appointment=appointment,
        contextual_summary=summary,
        sources_used=[d["metadata"].get("title","Medical Reference") for d in retrieve_sources(query)],
        agents_involved=agents_involved,
    )


def retrieve_sources(query: str) -> list[dict]:
    from app.rag import retrieve
    return retrieve(query)
