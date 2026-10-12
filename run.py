"""Entry point. Usage: python run.py"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"\n🏥  AI Healthcare Management System")
    print(f"   Orchestrator : {settings.ORCHESTRATOR_MODEL}")
    print(f"   Workers      : {settings.WORKER_MODEL}")
    print(f"   Sandbox      : {'ON' if settings.SANDBOX_MODE else 'OFF'}")
    print(f"   Docs         : http://127.0.0.1:{settings.APP_PORT}/docs")
    print(f"\n   Step 1: POST /ingest/directory  to load medical knowledge base")
    print(f"   Step 2: POST /query  with patient_id and symptoms\n")
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT,
                reload=settings.DEBUG, log_level="info")
