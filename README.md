# Healthcare Multi-Agent Assistant (RAG)

[![tests](https://github.com/Rishichamp/healthcare-multi-agent-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishichamp/healthcare-multi-agent-rag/actions/workflows/tests.yml)

A multi-agent pipeline that takes a patient's description of their symptoms and walks it through triage, specialist matching and appointment booking. Each step is grounded in a medical knowledge base through retrieval-augmented generation.

> **Not medical advice.** This is a portfolio/learning project that runs on simulated patients, doctors and appointments.

## How it works

```
patient query ──► SymptomAnalysisAgent ──► DoctorMatchingAgent ──► AppointmentAgent
                   urgency + red flags      ranked specialists       booking + summary
                          ▲                        ▲                        ▲
                          └──────── RAG over the medical knowledge base ────┘
```

An orchestrator runs the three agents in sequence, passing each one's structured output to the next, and returns one response with the sources used.

## Features

- **Symptom analysis:** urgency level and red-flag detection.
- **Doctor matching:** specialist recommendations, always including a GP fallback.
- **Appointment booking:** in sandbox mode, plus a patient-friendly summary.
- **RAG grounding:** a ChromaDB knowledge base built from `sample_docs/`.
- **Structured outputs:** Pydantic models at every step (`SymptomAnalysis`, `DoctorRecommendation`, `AppointmentConfirmation`).
- **Two model tiers:** `gpt-4o` orchestrator, `gpt-4o-mini` workers.

## Tech stack

Python 3.11 · OpenAI · ChromaDB · FastAPI · Pydantic · pytest

## Project structure

```
app/
  coordinator.py   the three agents and the orchestrator
  rag.py           knowledge-base retrieval
  data_store.py    simulated patients, doctors and appointments
  main.py          FastAPI app and endpoints
  models.py        structured schemas
  config.py        settings from .env
sample_docs/       medical knowledge base (Markdown)
tests/             17 tests
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # then add your OPENAI_API_KEY
python run.py                  # http://127.0.0.1:8000/docs
```

Call `POST /ingest/directory` once to build the knowledge base.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/query` | Run a patient query through the multi-agent pipeline |
| GET | `/patients/{patient_id}` | Patient profile |
| POST | `/ingest/directory` | Ingest the knowledge-base documents |
| GET | `/stats` | System statistics |
| GET | `/health` | Health check |

## Tests

```bash
pytest tests/ -q
```

17 tests covering the data layer, doctor recommendations, sandboxed booking, chunking and the API, with the LLM mocked.

## Notes

Built as part of Module 4 of a Generative AI engineering programme.
