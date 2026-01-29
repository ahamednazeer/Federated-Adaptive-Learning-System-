"""
Doctor Router
API endpoints for doctor dashboard and patient review
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import aiosqlite

router = APIRouter()


class DoctorLogin(BaseModel):
    email: str
    password: str


class DoctorFeedback(BaseModel):
    diagnosis_id: int
    confirmed: bool
    corrected_cvd_risk: float = None
    corrected_parkinsons_risk: float = None
    corrected_diabetes_risk: float = None
    notes: str = ""


@router.post("/login")
async def doctor_login(credentials: DoctorLogin):
    """Doctor authentication"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM users WHERE email = ? AND role = ?",
                (credentials.email, "doctor")
            )
            user = await cursor.fetchone()
            
            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            
            # In production, verify password hash
            return {
                "message": "Login successful",
                "doctor_id": user['id'],
                "email": user['email']
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/dashboard")
async def get_dashboard():
    """Get doctor dashboard data"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            
            # Get recent diagnoses
            cursor = await db.execute(
                """SELECT d.*, p.name as patient_name, p.age
                   FROM diagnoses d
                   JOIN patients p ON d.patient_id = p.id
                   ORDER BY d.created_at DESC
                   LIMIT 20"""
            )
            diagnoses = await cursor.fetchall()
            
            # Get high-risk patients
            cursor = await db.execute(
                """SELECT d.*, p.name as patient_name, p.age
                   FROM diagnoses d
                   JOIN patients p ON d.patient_id = p.id
                   WHERE d.cvd_risk > 0.7 OR d.parkinsons_risk > 0.7 OR d.diabetes_risk > 0.7
                   ORDER BY d.created_at DESC
                   LIMIT 10"""
            )
            high_risk = await cursor.fetchall()
            
            return {
                "recent_diagnoses": [dict(row) for row in diagnoses],
                "high_risk_patients": [dict(row) for row in high_risk]
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/patients/{patient_id}/insights")
async def get_patient_insights(patient_id: int):
    """Get processed insights for a patient"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            
            # Get patient info
            cursor = await db.execute(
                "SELECT * FROM patients WHERE id = ?",
                (patient_id,)
            )
            patient = await cursor.fetchone()
            
            if not patient:
                raise HTTPException(status_code=404, detail="Patient not found")
            
            # Get latest diagnosis
            cursor = await db.execute(
                """SELECT * FROM diagnoses 
                   WHERE patient_id = ? 
                   ORDER BY created_at DESC 
                   LIMIT 1""",
                (patient_id,)
            )
            diagnosis = await cursor.fetchone()
            
            # Get explanations
            explanations = []
            if diagnosis:
                cursor = await db.execute(
                    "SELECT * FROM explanations WHERE diagnosis_id = ?",
                    (diagnosis['id'],)
                )
                explanations = await cursor.fetchall()
            
            return {
                "patient": dict(patient),
                "latest_diagnosis": dict(diagnosis) if diagnosis else None,
                "explanations": [dict(row) for row in explanations]
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/patients/{patient_id}/feedback")
async def submit_feedback(patient_id: int, feedback: DoctorFeedback):
    """Submit clinical feedback for a diagnosis"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            # Get doctor ID (in production, from auth token)
            doctor_id = 3  # Hardcoded for demo
            
            await db.execute(
                """INSERT INTO feedback (diagnosis_id, doctor_id, confirmed, 
                   corrected_cvd_risk, corrected_parkinsons_risk, corrected_diabetes_risk, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (feedback.diagnosis_id, doctor_id, feedback.confirmed,
                 feedback.corrected_cvd_risk, feedback.corrected_parkinsons_risk,
                 feedback.corrected_diabetes_risk, feedback.notes)
            )
            await db.commit()
            
            return {"message": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
