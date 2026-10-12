"""
Healthcare data store — simulates a hospital management system.
In production: replace with real EHR system API calls.
"""
from datetime import datetime, timedelta
from typing import Optional
import random

_base = datetime.now()

DOCTORS = {
    "D001": {"name": "Dr. Priya Sharma",    "specialisation": "General Physician",   "available_slots": ["2024-12-20 10:00","2024-12-20 14:00","2024-12-21 09:00"], "location": "OPD Block A, Room 101", "rating": 4.8},
    "D002": {"name": "Dr. Rahul Mehta",     "specialisation": "Neurologist",         "available_slots": ["2024-12-20 11:00","2024-12-22 15:00"], "location": "Neurology Dept, Room 205", "rating": 4.9},
    "D003": {"name": "Dr. Anita Verma",     "specialisation": "Cardiologist",        "available_slots": ["2024-12-21 10:00","2024-12-23 14:00"], "location": "Cardiology Dept, Room 310", "rating": 4.7},
    "D004": {"name": "Dr. Suresh Kumar",    "specialisation": "Orthopaedic Surgeon", "available_slots": ["2024-12-20 16:00","2024-12-21 11:00"], "location": "Ortho Block, Room 410", "rating": 4.6},
    "D005": {"name": "Dr. Meera Iyer",      "specialisation": "Dermatologist",       "available_slots": ["2024-12-21 14:00","2024-12-22 10:00"], "location": "Skin Clinic, Room 105", "rating": 4.8},
    "D006": {"name": "Dr. Vikram Singh",    "specialisation": "Gastroenterologist",  "available_slots": ["2024-12-22 11:00","2024-12-23 16:00"], "location": "GI Dept, Room 220", "rating": 4.7},
    "D007": {"name": "Dr. Kavya Nair",      "specialisation": "Psychiatrist",        "available_slots": ["2024-12-20 15:00","2024-12-23 10:00"], "location": "Mental Health Clinic, Room 115", "rating": 4.9},
    "D008": {"name": "Dr. Arun Patel",      "specialisation": "Paediatrician",       "available_slots": ["2024-12-21 09:00","2024-12-22 14:00"], "location": "Paediatric OPD, Room 130", "rating": 4.8},
    "D009": {"name": "Dr. Sunita Rao",      "specialisation": "Gynaecologist",       "available_slots": ["2024-12-20 12:00","2024-12-21 16:00"], "location": "Women's Health, Room 315", "rating": 4.9},
    "D010": {"name": "Dr. Deepak Joshi",    "specialisation": "Pulmonologist",       "available_slots": ["2024-12-22 09:00","2024-12-23 11:00"], "location": "Respiratory Dept, Room 210", "rating": 4.6},
}

PATIENTS = {
    "P001": {"name": "Arjun Kumar", "age": 34, "blood_group": "O+", "conditions": ["Hypertension"], "medications": ["Amlodipine 5mg"], "allergies": ["Penicillin"], "last_visit": "2024-11-15"},
    "P002": {"name": "Priya Sharma","age": 28, "blood_group": "A+", "conditions": [], "medications": [], "allergies": [], "last_visit": "2024-10-20"},
    "P003": {"name": "Ravi Singh",  "age": 55, "blood_group": "B+", "conditions": ["Type 2 Diabetes","Hypertension"], "medications": ["Metformin 500mg","Losartan 50mg"], "allergies": [], "last_visit": "2024-12-01"},
}

APPOINTMENTS = {}
_appt_counter = [1]

# Specialisation → symptom keyword mapping for recommendation
SPECIALISATION_MAP = {
    "headache":        "D001,D002",
    "chest pain":      "D003",
    "back pain":       "D004",
    "skin rash":       "D005",
    "stomach":         "D006",
    "anxiety":         "D007",
    "depression":      "D007",
    "child":           "D008",
    "pregnancy":       "D009",
    "cough":           "D010",
    "breathing":       "D010",
    "fever":           "D001",
    "joint pain":      "D004",
    "heart":           "D003",
    "palpitation":     "D003",
    "weight loss":     "D001,D006",
}


def get_doctor(doctor_id: str) -> Optional[dict]:
    return DOCTORS.get(doctor_id)

def get_all_doctors() -> list[dict]:
    return [{"doctor_id": k, **v} for k, v in DOCTORS.items()]

def get_doctors_by_specialisation(specialisation: str) -> list[dict]:
    spec_lower = specialisation.lower()
    return [
        {"doctor_id": k, **v}
        for k, v in DOCTORS.items()
        if spec_lower in v["specialisation"].lower()
    ]

def get_patient(patient_id: str) -> Optional[dict]:
    return PATIENTS.get(patient_id)

def recommend_doctors_for_symptoms(symptoms: list[str]) -> list[dict]:
    """Simple keyword matching to recommend relevant doctors."""
    doctor_ids = set()
    for symptom in symptoms:
        s_lower = symptom.lower()
        for keyword, doc_ids in SPECIALISATION_MAP.items():
            if keyword in s_lower:
                for did in doc_ids.split(","):
                    doctor_ids.add(did.strip())
    # Always include GP
    doctor_ids.add("D001")
    return [{"doctor_id": did, **DOCTORS[did]} for did in doctor_ids if did in DOCTORS]

def book_appointment(patient_id: str, doctor_id: str, slot: str, reason: str, sandbox: bool = True) -> dict:
    if doctor_id not in DOCTORS:
        return {"success": False, "error": f"Doctor {doctor_id} not found"}
    doctor = DOCTORS[doctor_id]
    appt_id = f"A{_appt_counter[0]:04d}"
    _appt_counter[0] += 1
    prep_notes = {
        "General Physician": "Bring previous prescription and reports. Fast for 2 hours before visit.",
        "Neurologist": "Bring MRI/CT scans if available. List all current medications.",
        "Cardiologist": "Bring ECG reports if available. Avoid caffeine 4 hours before visit.",
    }.get(doctor["specialisation"], "Bring all previous medical records and reports.")

    appt = {
        "appointment_id": appt_id,
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "doctor_name": doctor["name"],
        "specialisation": doctor["specialisation"],
        "scheduled_time": slot,
        "location": doctor["location"],
        "reason": reason,
        "preparation_notes": prep_notes,
        "status": "booked",
    }
    if not sandbox:
        APPOINTMENTS[appt_id] = appt
    return {"success": True, "sandbox": sandbox, **appt}

def get_patient_appointments(patient_id: str) -> list[dict]:
    return [a for a in APPOINTMENTS.values() if a["patient_id"] == patient_id]
