"""
Patient Router
API endpoints for patient management
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import aiosqlite
from datetime import datetime

router = APIRouter()

# Pydantic models
class PatientCreate(BaseModel):
    email: str
    password: str
    name: str
    age: int
    medical_history: Optional[str] = None


class PatientConsent(BaseModel):
    consent_privacy: bool
    consent_federated_learning: bool


class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    medical_history: Optional[str] = None
    
    
class PatientResponse(BaseModel):
    id: int
    name: str
    age: int
    consent_privacy: bool
    consent_federated_learning: bool


@router.post("/register", response_model=dict)
async def register_patient(patient: PatientCreate):
    """Register a new patient"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            # Create user
            cursor = await db.execute(
                "INSERT INTO users (email, password_hash, role) VALUES (?, ?, ?)",
                (patient.email, f"hashed_{patient.password}", "patient")
            )
            user_id = cursor.lastrowid
            
            # Create patient profile
            await db.execute(
                """INSERT INTO patients (user_id, name, age, medical_history) 
                   VALUES (?, ?, ?, ?)""",
                (user_id, patient.name, patient.age, patient.medical_history or "")
            )
            
            await db.commit()
            
            return {
                "message": "Patient registered successfully",
                "user_id": user_id,
                "email": patient.email
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{patient_id}/consent")
async def update_consent(patient_id: int, consent: PatientConsent):
    """Update patient consent"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            await db.execute(
                """UPDATE patients 
                   SET consent_privacy = ?, consent_federated_learning = ?
                   WHERE id = ?""",
                (consent.consent_privacy, consent.consent_federated_learning, patient_id)
            )
            await db.commit()
            
            
            return {"message": "Consent updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{patient_id}")
async def update_patient(patient_id: int, patient_update: PatientUpdate):
    """Update patient information"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            # Build query dynamically based on provided fields
            update_fields = []
            params = []
            
            if patient_update.name is not None:
                update_fields.append("name = ?")
                params.append(patient_update.name)
            
            if patient_update.age is not None:
                update_fields.append("age = ?")
                params.append(patient_update.age)
                
            if patient_update.medical_history is not None:
                update_fields.append("medical_history = ?")
                params.append(patient_update.medical_history)
            
            if not update_fields:
                return {"message": "No fields to update"}
            
            params.append(patient_id)
            query = f"UPDATE patients SET {', '.join(update_fields)} WHERE id = ?"
            
            cursor = await db.execute(query, tuple(params))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Patient not found")
                
            await db.commit()
            
            return {"message": "Patient updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{patient_id}", response_model=dict)
async def get_patient(patient_id: int):
    """Get patient details"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM patients WHERE id = ?",
                (patient_id,)
            )
            row = await cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Patient not found")
            
            return dict(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{patient_id}/devices")
async def get_patient_devices(patient_id: int):
    """Get patient's paired devices"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM devices WHERE patient_id = ?",
                (patient_id,)
            )
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{patient_id}/diagnoses")
async def get_patient_diagnoses(patient_id: int):
    """Get patient's diagnosis history"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM diagnoses 
                   WHERE patient_id = ? 
                   ORDER BY created_at DESC
                   LIMIT 10""",
                (patient_id,)
            )
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{patient_id}/explanations/{diagnosis_id}")
async def get_diagnosis_explanations(patient_id: int, diagnosis_id: int):
    """Get XAI explanations for a diagnosis"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT e.* FROM explanations e
                   JOIN diagnoses d ON e.diagnosis_id = d.id
                   WHERE d.id = ? AND d.patient_id = ?""",
                (diagnosis_id, patient_id)
            )
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
