"""Pydantic models for the Healthcare Management System."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class AppointmentStatus(str, Enum):
    AVAILABLE  = "available"
    BOOKED     = "booked"
    CANCELLED  = "cancelled"
    COMPLETED  = "completed"


class UrgencyLevel(str, Enum):
    ROUTINE   = "routine"
    SOON      = "soon"
    URGENT    = "urgent"
    EMERGENCY = "emergency"


# ── Request models ────────────────────────────────────────────────
class PatientQuery(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    query: str = Field(..., description="Patient's health query or symptom description")
    model_config = {
        "json_schema_extra": {
            "example": {
                "patient_id": "P001",
                "query": "I have had a persistent headache for 3 days with mild fever and neck stiffness."
            }
        }
    }

class AppointmentRequest(BaseModel):
    patient_id: str
    doctor_id: str
    preferred_date: str = Field(..., description="Preferred date in YYYY-MM-DD format")
    reason: str

class SymptomCheckRequest(BaseModel):
    symptoms: list[str]
    duration_days: int = Field(1, ge=0)
    severity: str = Field("moderate", description="mild / moderate / severe")


# ── Response models ───────────────────────────────────────────────
class DoctorRecommendation(BaseModel):
    doctor_id: str
    name: str
    specialisation: str
    available_slots: list[str]
    reason: str

class SymptomAnalysis(BaseModel):
    urgency: UrgencyLevel
    possible_conditions: list[str]
    recommended_specialisation: str
    red_flags: list[str]
    home_care_advice: list[str]
    disclaimer: str

class AppointmentConfirmation(BaseModel):
    appointment_id: str
    patient_id: str
    doctor_name: str
    specialisation: str
    scheduled_time: str
    location: str
    preparation_notes: str
    status: AppointmentStatus

class PatientSummary(BaseModel):
    patient_id: str
    name: str
    age: int
    conditions: list[str]
    medications: list[str]
    last_visit: Optional[str]
    upcoming_appointments: list[dict]

class HealthcareResponse(BaseModel):
    patient_id: str
    query: str
    symptom_analysis: Optional[SymptomAnalysis]
    recommended_doctors: list[DoctorRecommendation]
    appointment: Optional[AppointmentConfirmation]
    contextual_summary: str
    sources_used: list[str]
    agents_involved: list[str]
    disclaimer: str = "This is AI-assisted guidance only. Always consult a qualified medical professional for diagnosis and treatment."
