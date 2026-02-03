"""
Verification Script for Patient Management
Tests Add and Edit Patient functionality
"""

import sys
import os
import aiosqlite
import asyncio
from typing import Dict

# Add necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.routers.patients import PatientCreate, PatientUpdate, update_patient, register_patient

# Mock database context for testing without running full server
# but we need the actual DB file to be present/accessible or use a test DB

# Since the functions use "database/federated_learning.db" hardcoded, 
# we should ensure we run this from the backend root or mock the connection.
# For simplicity in this environment, we'll run from backend root.

async def test_patient_management():
    print("\n--- Testing Patient Management ---")
    
    # 1. Test Registration (Add)
    print("Testing Patient Registration...")
    new_patient = PatientCreate(
        email="test_edit@example.com",
        password="password123",
        name="Test Patient For Edit",
        age=30,
        medical_history="None"
    )
    
    # Clean up potentially existing user from previous runs
    async with aiosqlite.connect("database/federated_learning.db") as db:
        await db.execute("DELETE FROM users WHERE email = ?", (new_patient.email,))
        await db.commit()
    
    try:
        response = await register_patient(new_patient)
        print("✓ Registration successful")
        patient_id = response['user_id'] # Note: register returns user_id, but inserts into patients too
        # Wait, register_patient returns user_id. We need patient_id for update.
        # Let's check the code:
        # INSERT INTO patients (user_id, ...)
        # We need to fetch the patient_id associated with this user_id
        
        async with aiosqlite.connect("database/federated_learning.db") as db:
            cursor = await db.execute("SELECT id FROM patients WHERE user_id = ?", (patient_id,))
            row = await cursor.fetchone()
            real_patient_id = row[0]
            print(f"  Got patient ID: {real_patient_id}")
            
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return

    # 2. Test Editing (Update)
    print("\nTesting Patient Update...")
    update_data = PatientUpdate(
        name="Updated Patient Name",
        medical_history="Updated History: Mild Hypertension"
    )
    
    try:
        update_response = await update_patient(real_patient_id, update_data)
        print(f"✓ Update response: {update_response}")
        
        # Verify update
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM patients WHERE id = ?", (real_patient_id,))
            patient = await cursor.fetchone()
            
            print(f"  Verifying name: {patient['name']}")
            print(f"  Verifying history: {patient['medical_history']}")
            
            assert patient['name'] == "Updated Patient Name", "Name update failed"
            assert patient['medical_history'] == "Updated History: Mild Hypertension", "History update failed"
            assert patient['age'] == 30, "Age should remain unchanged"
            
            print("✓ Verification successful: Data updated correctly")
            
    except Exception as e:
        print(f"❌ Update failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure we are in backend dir so database path is correct because code uses relative path
    if not os.path.exists("database"):
        print("Error: Please run this script from the backend directory.")
        sys.exit(1)
        
    asyncio.run(test_patient_management())
