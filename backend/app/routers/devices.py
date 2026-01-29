"""
Device Router
API endpoints for device management and data collection
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import aiosqlite
from datetime import datetime
import numpy as np
import sys
sys.path.append('..')

# Import dataset loaders and preprocessors
from datasets.loaders import DatasetFactory
from datasets.preprocessors import PreprocessorFactory

# Import ML encoders
from app.ml.encoders.ecg_encoder import create_ecg_encoder
from app.ml.encoders.voice_encoder import create_voice_encoder
from app.ml.encoders.handwriting_encoder import create_handwriting_encoder
from app.ml.encoders.wearable_encoder import create_wearable_encoder
from app.ml.encoders.glucose_encoder import create_glucose_encoder
from app.ml.fusion.multimodal_fusion import create_multimodal_fusion
from app.ml.xai.explainer import create_explainability_engine

router = APIRouter()

def _to_serializable(obj):
    """Recursively convert numpy objects to native python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(x) for x in obj]
    return obj

# Lazy-loaded models
_models = {}

def get_models():
    """Lazy load ML models"""
    if not _models:
        _models['ecg_encoder'] = create_ecg_encoder()
        _models['voice_encoder'] = create_voice_encoder()
        _models['handwriting_encoder'] = create_handwriting_encoder()
        _models['wearable_encoder'] = create_wearable_encoder()
        _models['glucose_encoder'] = create_glucose_encoder()
        _models['fusion'] = create_multimodal_fusion()
        _models['explainer'] = create_explainability_engine()
    return _models

# Pydantic models
class DevicePair(BaseModel):
    patient_id: int
    device_type: str  # ecg, voice, handwriting, wearable, glucose
    dataset_name: str  # PTB-XL, UCI, HandPD, WESAD, OhioT1DM

class DataCollect(BaseModel):
    device_id: int

@router.post("/pair")
async def pair_device(device: DevicePair):
    """Pair a new device to patient"""
    try:
        async with aiosqlite.connect("database/federated_learning.db") as db:
            cursor = await db.execute(
                """INSERT INTO devices (patient_id, device_type, dataset_name, status, last_sync)
                   VALUES (?, ?, ?, ?, ?)""",
                (device.patient_id, device.device_type, device.dataset_name, 
                 "paired", datetime.now().isoformat())
            )
            device_id = cursor.lastrowid
            await db.commit()
            
            return {
                "message": "Device paired successfully",
                "device_id": device_id,
                "device_type": device.device_type,
                "dataset": device.dataset_name
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/collect")
async def collect_data(request: DataCollect):
    """Simulate data collection from device"""
    try:
        # Get device info
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM devices WHERE id = ?",
                (request.device_id,)
            )
            device = await cursor.fetchone()
            
            if not device:
                raise HTTPException(status_code=404, detail="Device not found")
            
            device = dict(device)
        
        # Load dataset
        loader = DatasetFactory.create_loader(device['dataset_name'])
        loader.load()
        
        # Get sample
        sample = loader.get_sample(device['patient_id'])
        
        # Preprocess
        preprocessor = PreprocessorFactory.create_preprocessor(device['device_type'])
        processed = preprocessor.preprocess(sample)
        
        # Update device status
        async with aiosqlite.connect("database/federated_learning.db") as db:
            await db.execute(
                "UPDATE devices SET status = ?, last_sync = ? WHERE id = ?",
                ("active", datetime.now().isoformat(), request.device_id)
            )
            await db.commit()
        
        return {
            "message": "Data collected successfully",
            "device_id": request.device_id,
            "device_type": device['device_type'],
            "dataset": device['dataset_name'],
            "sample_info": {
                "patient_id": processed.get('patient_id'),
                "timestamp": sample.get('timestamp')
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/diagnose/{patient_id}")
async def run_diagnosis(patient_id: int):
    """Run multi-modal diagnosis for patient"""
    try:
        # Get patient devices
        async with aiosqlite.connect("database/federated_learning.db") as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM devices WHERE patient_id = ?",
                (patient_id,)
            )
            devices = await cursor.fetchall()
            devices = [dict(d) for d in devices]
        
        if not devices:
            raise HTTPException(status_code=400, detail="No devices paired for patient")
        
        # Load models
        models = get_models()
        
        # Collect data from all devices and encode
        embeddings = {}
        modality_features = {}
        
        for device in devices:
            # Load dataset and get sample
            loader = DatasetFactory.create_loader(device['dataset_name'])
            loader.load()
            sample = loader.get_sample(patient_id)
            
            # Preprocess
            preprocessor = PreprocessorFactory.create_preprocessor(device['device_type'])
            processed = preprocessor.preprocess(sample)
            modality_features[device['device_type']] = processed
            
            # Encode
            encoder_map = {
                'ecg': 'ecg_encoder',
                'voice': 'voice_encoder',
                'handwriting': 'handwriting_encoder',
                'wearable': 'wearable_encoder',
                'glucose': 'glucose_encoder'
            }
            
            encoder = models[encoder_map[device['device_type']]]
            
            # Get appropriate input for encoder
            if device['device_type'] == 'ecg':
                embedding = encoder.encode_sample(processed['processed_signal'])
            elif device['device_type'] == 'voice':
                embedding = encoder.encode_sample(processed['mfcc_features'])
            elif device['device_type'] == 'handwriting':
                embedding = encoder.encode_sample(processed['features'])
            elif device['device_type'] == 'wearable':
                embedding = encoder.encode_sample(processed['features'])
            elif device['device_type'] == 'glucose':
                embedding = encoder.encode_sample(processed['processed_signal'])
            
            embeddings[device['device_type']] = embedding
        
        # Fuse and diagnose
        fusion = models['fusion']
        diagnosis_result = fusion.predict(embeddings)
        
        # Generate explanations
        explainer = models['explainer']
        explanations = explainer.generate_explanation(
            diagnosis_result,
            diagnosis_result['attention_weights'],
            modality_features
        )
        
        # Save diagnosis to database
        async with aiosqlite.connect("database/federated_learning.db") as db:
            cursor = await db.execute(
                """INSERT INTO diagnoses (patient_id, cvd_risk, parkinsons_risk, diabetes_risk)
                   VALUES (?, ?, ?, ?)""",
                (patient_id, diagnosis_result['cvd_risk'], 
                 diagnosis_result['parkinsons_risk'], diagnosis_result['diabetes_risk'])
            )
            diagnosis_id = cursor.lastrowid
            
            # Save explanations
            for modality in ['ecg', 'voice', 'handwriting', 'wearable', 'glucose', 'fusion']:
                exp_key = f'{modality}_explanation' if modality != 'fusion' else 'summary'
                if exp_key in explanations:
                    exp = explanations[exp_key]
                    natural_language = exp if isinstance(exp, str) else exp.get('natural_language', '')
                    
                    await db.execute(
                        """INSERT INTO explanations (diagnosis_id, modality, natural_language)
                           VALUES (?, ?, ?)""",
                        (diagnosis_id, modality, natural_language)
                    )
            
            await db.commit()
        
        return {
            "diagnosis_id": diagnosis_id,
            "patient_id": patient_id,
            "risks": {
                "cvd": diagnosis_result['cvd_risk'],
                "parkinsons": diagnosis_result['parkinsons_risk'],
                "diabetes": diagnosis_result['diabetes_risk']
            },
            "attention_weights": diagnosis_result['attention_weights'],
            "explanations": explanations,
            "visualization_data": _to_serializable(modality_features)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
