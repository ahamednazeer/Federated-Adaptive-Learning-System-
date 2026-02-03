"""
Dataset Loaders for Multi-Modal Medical Data
Simulates real-time data streaming from edge devices using public datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from scipy import signal

class BaseDatasetLoader:
    """Base class for all dataset loaders"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.current_index = 0
    
    def load(self):
        """Load dataset into memory"""
        raise NotImplementedError
    
    def get_sample(self, patient_id: Optional[int] = None, medical_history: str = "") -> Dict:
        """Get a single sample (simulates real-time data collection)"""
        raise NotImplementedError
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get a batch of samples for federated training"""
        raise NotImplementedError


class PTBXLLoader(BaseDatasetLoader):
    """
    PTB-XL ECG Dataset Loader
    21,837 clinical 12-lead ECG records from 18,885 patients
    """
    
    def load(self):
        """Load PTB-XL dataset or use Scipy Real ECG"""
        try:
            # Load metadata
            metadata_path = self.dataset_path / 'ptbxl_database.csv'
            if metadata_path.exists():
                self.data = pd.read_csv(metadata_path)
                print(f"✓ Loaded PTB-XL: {len(self.data)} records")
            else:
                print(f"⚠ PTB-XL metadata not found at {metadata_path}")
                self._load_scipy_ecg()
        except Exception as e:
            print(f"⚠ Error loading PTB-XL: {e}, using synthetic data")
            self._create_synthetic_ecg()

    def _load_scipy_ecg(self):
        """Load real ECG from Scipy (Lightweight alternative)"""
        try:
            try:
                from scipy.datasets import electrocardiogram
            except ImportError:
                from scipy.misc import electrocardiogram
            
            self.real_ecg = electrocardiogram()
            self.fs = 360  # Scipy ECG is 360Hz
            print("✓ Loaded Scipy Real ECG dataset (5 minutes record)")
            
            # Create a mock dataframe structure to satisfy the rest of the code
            num_samples = 1000
            self.data = pd.DataFrame({
                'ecg_id': range(num_samples),
                'patient_id': np.random.randint(1, 100, num_samples),
                'age': np.random.randint(20, 90, num_samples),
                'sex': np.random.choice([0, 1], num_samples),
                'scp_codes': ['{"NORM": 100}'] * num_samples
            })
            self.use_scipy = True
        except Exception as e:
            print(f"⚠ Could not load Scipy ECG: {e}")
            self._create_synthetic_ecg()
            self.use_scipy = False

    def _create_synthetic_ecg(self):
        """Create synthetic ECG data for demonstration"""
        print("  Creating synthetic ECG data...")
        num_samples = 1000
        self.data = pd.DataFrame({
            'ecg_id': range(num_samples),
            'patient_id': np.random.randint(1, 100, num_samples),
            'age': np.random.randint(20, 90, num_samples),
            'sex': np.random.choice([0, 1], num_samples),
            'scp_codes': ['{"NORM": 100}'] * num_samples
        })
        self.use_scipy = False
    
    def get_sample(self, patient_id: Optional[int] = None, medical_history: str = "") -> Dict:
        """Get ECG waveform sample"""
        if self.data is None:
            self.load()
        
        # Simulate 12-lead ECG (5000 samples at 500Hz = 10 seconds)
        sampling_rate = 500
        duration = 10
        num_samples = sampling_rate * duration
        num_samples = sampling_rate * duration
        
        # Ensure determinism but variation based on history
        if patient_id is not None:
            # simple hash of the history string to vary the seed
            history_hash = sum(ord(c) for c in medical_history) if medical_history else 0
            np.random.seed(patient_id + history_hash)
        
        if getattr(self, 'use_scipy', False):
            # Use real data from Scipy
            # Random starting point
            start_idx = np.random.randint(0, len(self.real_ecg) - num_samples)
            ecg_signal = self.real_ecg[start_idx : start_idx + num_samples]
            
            # Resample if needed (Scipy is 360Hz, we want 500Hz)
            if self.fs != sampling_rate:
                target_len = int(len(ecg_signal) * sampling_rate / self.fs)
                # Simple linear interpolation for speed (or just take simpler slice)
                # Actually, just taking a slice of length num_samples is fine for visualization
                # but to be correct on time:
                raw_samples = int(duration * self.fs)
                start_idx = np.random.randint(0, len(self.real_ecg) - raw_samples)
                ecg_signal = signal.resample(self.real_ecg[start_idx : start_idx + raw_samples], num_samples)
                
        else:
            # Generate synthetic ECG waveform
            t = np.linspace(0, duration, num_samples)
            
            # Simulate normal sinus rhythm (Healthy baseline)
            heart_rate = np.random.uniform(60, 80)  # BPM (Healthy)
            
            # BIAS: If history suggests heart issues, adjust heart rate or waveform
            if 'cardiac' in medical_history.lower() or 'heart' in medical_history.lower() or 'cad' in medical_history.lower() or 'coronary' in medical_history.lower():
                # Simulate Higher HR or Irregularity
                heart_rate = np.random.uniform(90, 130)  # Tachycardia
            
            ecg_signal = self._generate_ecg_waveform(t, heart_rate)
            
            # BIAS: Add noise/irregularity
            noise_level = 0.02  # Low noise (Healthy)
            if 'arrhythmia' in medical_history.lower() or 'fibrillation' in medical_history.lower():
                noise_level = 0.2
            
            # Add noise
            noise = np.random.normal(0, noise_level, num_samples)
            ecg_signal += noise
        
        return {
            'signal': ecg_signal.tolist(),
            'sampling_rate': sampling_rate,
            'duration': duration,
            'patient_id': patient_id or np.random.randint(1, 100),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _generate_ecg_waveform(self, t: np.ndarray, heart_rate: float) -> np.ndarray:
        """Generate synthetic ECG waveform"""
        # Simple QRS complex simulation
        freq = heart_rate / 60  # Hz
        
        # P wave
        p_wave = 0.1 * np.sin(2 * np.pi * freq * t)
        
        # QRS complex (dominant)
        qrs = 1.0 * np.sin(2 * np.pi * freq * 3 * t) * np.exp(-((t % (1/freq) - 0.3)**2) / 0.01)
        
        # T wave
        t_wave = 0.2 * np.sin(2 * np.pi * freq * t - np.pi/4)
        
        return p_wave + qrs + t_wave
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get batch for training"""
        return [self.get_sample() for _ in range(batch_size)]


class UCIVoiceLoader(BaseDatasetLoader):
    """
    UCI Parkinson's Voice Dataset Loader
    Voice measurements from 31 people (23 with Parkinson's disease)
    """
    
    def load(self):
        """Load UCI Parkinson's dataset"""
        try:
            data_path = self.dataset_path / 'parkinsons.data'
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                print(f"✓ Loaded UCI Parkinson's: {len(self.data)} records")
            else:
                print(f"⚠ UCI data not found at {data_path}")
                self._create_synthetic_voice()
        except Exception as e:
            print(f"⚠ Error loading UCI: {e}, using synthetic data")
            self._create_synthetic_voice()
    
    def _create_synthetic_voice(self):
        """Create synthetic voice features"""
        print("  Creating synthetic voice data...")
        num_samples = 200
        self.data = pd.DataFrame({
            'name': [f'patient_{i}' for i in range(num_samples)],
            'MDVP:Fo(Hz)': np.random.uniform(80, 250, num_samples),
            'MDVP:Fhi(Hz)': np.random.uniform(100, 300, num_samples),
            'MDVP:Flo(Hz)': np.random.uniform(60, 150, num_samples),
            'MDVP:Jitter(%)': np.random.uniform(0.001, 0.02, num_samples),
            'MDVP:Shimmer': np.random.uniform(0.01, 0.1, num_samples),
            'status': np.random.choice([0, 1], num_samples)
        })
    
    def get_sample(self, patient_id: Optional[int] = None, medical_history: str = "") -> Dict:
        """Get voice features sample"""
        if self.data is None:
            self.load()
        
        # Simulate MFCC features (13 coefficients × 100 frames)
        num_mfcc = 13
        num_frames = 100
        
        # Ensure determinism but variation based on history
        if patient_id is not None:
            history_hash = sum(ord(c) for c in medical_history) if medical_history else 0
            np.random.seed(patient_id + history_hash)
        
        # Generate synthetic MFCCs
        mfcc = np.random.randn(num_mfcc, num_frames)
        
        # Add tremor characteristics for Parkinson's simulation
        # Default: Very low chance of random tremor in healthy people
        has_tremor = np.random.random() > 0.95
        
        # BIAS: Force tremor if Parkinson's in history
        if 'parkinson' in medical_history.lower():
            has_tremor = True
            
        if has_tremor:
            tremor_freq = np.random.uniform(4, 6)  # Hz (typical Parkinson's tremor)
            t = np.linspace(0, 3, num_frames)
            tremor = 0.3 * np.sin(2 * np.pi * tremor_freq * t)
            mfcc[0, :] += tremor
        
        return {
            'mfcc': mfcc.tolist(),
            'num_coefficients': num_mfcc,
            'num_frames': num_frames,
            'patient_id': patient_id or np.random.randint(1, 100),
            'has_tremor': has_tremor,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get batch for training"""
        return [self.get_sample() for _ in range(batch_size)]


class HandwritingSyntheticLoader(BaseDatasetLoader):
    """
    Synthetic Handwriting Dataset Loader
    Simulates handwriting tasks (spirals, writing) with tremor patterns
    """
    
    def load(self):
        """Initialize synthetic handwriting generator"""
        print("✓ Synthetic handwriting generator ready")
        self.data = True
    
    def get_sample(self, patient_id: Optional[int] = None, medical_history: str = "") -> Dict:
        """Generate synthetic handwriting sample"""
        # Simulate spiral drawing task
        num_points = 500
        
        # Ensure determinism but variation based on history
        if patient_id is not None:
            history_hash = sum(ord(c) for c in medical_history) if medical_history else 0
            np.random.seed(patient_id + history_hash)
            
        t = np.linspace(0, 4 * np.pi, num_points)
        
        # Base spiral
        r = t / (4 * np.pi)
        x = r * np.cos(t)
        r = t / (4 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        # Add baseline natural variation (everyone writes differently)
        x += np.random.normal(0, 0.005, num_points)
        y += np.random.normal(0, 0.005, num_points)
        
        # Add tremor if simulating Parkinson's
        # Default: Very low chance of random tremor in healthy people
        has_tremor = np.random.random() > 0.95
        
        # BIAS: Force tremor if Parkinson's in history
        if 'parkinson' in medical_history.lower():
            has_tremor = True
            
        if has_tremor:
            tremor_freq = np.random.uniform(4, 7)
            tremor_amplitude = np.random.uniform(0.02, 0.05)
            # Enhance tremor for known cases
            if 'parkinson' in medical_history.lower():
                tremor_amplitude *= 1.5
                
            tremor_x = tremor_amplitude * np.sin(2 * np.pi * tremor_freq * t)
            tremor_y = tremor_amplitude * np.cos(2 * np.pi * tremor_freq * t)
            x += tremor_x
            y += tremor_y
        
        # Compute velocity and pressure
        velocity = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        velocity = np.append(velocity, velocity[-1])
        
        pressure = np.random.uniform(0.3, 1.0, num_points)
        if has_tremor:
            pressure *= np.random.uniform(0.7, 0.9)  # Reduced pressure control
        
        return {
            'x': x.tolist(),
            'y': y.tolist(),
            'velocity': velocity.tolist(),
            'pressure': pressure.tolist(),
            'num_points': num_points,
            'patient_id': patient_id or np.random.randint(1, 100),
            'has_tremor': has_tremor,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get batch for training"""
        return [self.get_sample() for _ in range(batch_size)]


class WESADLoader(BaseDatasetLoader):
    """
    WESAD (Wearable Stress and Affect Detection) Dataset Loader
    Physiological signals from wearable sensors
    """
    
    def load(self):
        """Load WESAD dataset or create synthetic"""
        # WESAD requires manual download
        if not self.dataset_path.exists():
            print("⚠ WESAD not found, using synthetic wearable data")
            self._create_synthetic_wearable()
        else:
            print("✓ WESAD dataset found")
            self.data = True
    
    def _create_synthetic_wearable(self):
        """Create synthetic wearable sensor data"""
        print("  Creating synthetic wearable data...")
        self.data = True
    
    def get_sample(self, patient_id: Optional[int] = None, medical_history: str = "") -> Dict:
        """Get wearable sensor sample (30-second window)"""
        sampling_rate = 32  # Hz
        duration = 30  # seconds
        num_samples = sampling_rate * duration
        
        sampling_rate = 32  # Hz
        duration = 30  # seconds
        num_samples = sampling_rate * duration
        
        # Ensure determinism but variation based on history
        if patient_id is not None:
            history_hash = sum(ord(c) for c in medical_history) if medical_history else 0
            np.random.seed(patient_id + history_hash)
        
        t = np.linspace(0, duration, num_samples)
        
        # Heart rate (60-100 BPM baseline)
        baseline_hr = np.random.uniform(60, 100)
        
        # BIAS: Stress or Anxiety
        if 'anxiety' in medical_history.lower() or 'stress' in medical_history.lower():
            baseline_hr += 15  # Elevated HR
            
        hr = baseline_hr + 5 * np.sin(2 * np.pi * 0.1 * t)  # Respiratory sinus arrhythmia
        hr += np.random.normal(0, 2, num_samples)  # Noise
        
        # Skin conductance (stress indicator)
        baseline_sc = np.random.uniform(2, 10)  # μS
        
        if 'anxiety' in medical_history.lower():
            baseline_sc += 5 # Higher sweating/conductance
            
        sc = baseline_sc + np.random.normal(0, 0.5, num_samples)
        
        # Accelerometer (movement)
        acc_x = np.random.normal(0, 0.1, num_samples)
        acc_y = np.random.normal(0, 0.1, num_samples)
        acc_z = np.random.normal(9.8, 0.1, num_samples)  # Gravity
        
        return {
            'heart_rate': hr.tolist(),
            'skin_conductance': sc.tolist(),
            'acc_x': acc_x.tolist(),
            'acc_y': acc_y.tolist(),
            'acc_z': acc_z.tolist(),
            'sampling_rate': sampling_rate,
            'duration': duration,
            'patient_id': patient_id or np.random.randint(1, 100),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get batch for training"""
        return [self.get_sample() for _ in range(batch_size)]


class OhioT1DMLoader(BaseDatasetLoader):
    """
    OhioT1DM Dataset Loader
    Blood glucose levels from Type 1 Diabetes patients
    """
    
    def load(self):
        """Load OhioT1DM dataset or create synthetic"""
        if not self.dataset_path.exists():
            print("⚠ OhioT1DM not found, using synthetic glucose data")
            self._create_synthetic_glucose()
        else:
            print("✓ OhioT1DM dataset found")
            self.data = True
    
    def _create_synthetic_glucose(self):
        """Create synthetic CGM data"""
        print("  Creating synthetic glucose data...")
        self.data = True
    
    def get_sample(self, patient_id: Optional[int] = None, medical_history: str = "") -> Dict:
        """Get CGM sample (24 hours, 5-minute intervals = 288 readings)"""
        num_readings = 288
        interval_minutes = 5
        
        # Ensure determinism but variation based on history
        if patient_id is not None:
            history_hash = sum(ord(c) for c in medical_history) if medical_history else 0
            np.random.seed(patient_id + history_hash)
        
        # Simulate daily glucose pattern
        t = np.linspace(0, 24, num_readings)
        
        # Baseline glucose (mg/dL) - Healthy range defaults
        baseline = np.random.uniform(70, 100)
        
        # BIAS: Diabetes
        if 'diabetes' in medical_history.lower():
            baseline = np.random.uniform(140, 200) # Hyperglycemia baseline
        elif 'hypoglycemia' in medical_history.lower() or 'insulin' in medical_history.lower():
            baseline = np.random.uniform(70, 100) # Lower baseline
        
        # Meal spikes (breakfast, lunch, dinner)
        meal_times = [7, 12, 19]
        glucose = np.full(num_readings, baseline)
        
        for meal_time in meal_times:
            spike = 60 * np.exp(-((t - meal_time)**2) / 2)
            glucose += spike
        
        # Add random variation
        glucose += np.random.normal(0, 10, num_readings)
        
        # Ensure physiological range
        glucose = np.clip(glucose, 70, 300)
        
        return {
            'glucose': glucose.tolist(),
            'timestamps': t.tolist(),
            'interval_minutes': interval_minutes,
            'num_readings': num_readings,
            'patient_id': patient_id or np.random.randint(1, 100),
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """Get batch for training"""
        return [self.get_sample() for _ in range(batch_size)]


# Dataset factory
class DatasetFactory:
    """Factory for creating dataset loaders"""
    
    LOADERS = {
        'PTB-XL': PTBXLLoader,
        'UCI': UCIVoiceLoader,
        'HandPD': HandwritingSyntheticLoader,
        'WESAD': WESADLoader,
        'OhioT1DM': OhioT1DMLoader
    }
    
    @staticmethod
    def create_loader(dataset_name: str, base_path: str = 'datasets/raw') -> BaseDatasetLoader:
        """Create a dataset loader instance"""
        if dataset_name not in DatasetFactory.LOADERS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        loader_class = DatasetFactory.LOADERS[dataset_name]
        
        # Map dataset names to directory names
        dir_mapping = {
            'PTB-XL': 'ptb-xl',
            'UCI': 'uci-parkinsons',
            'HandPD': 'handwriting-synthetic',
            'WESAD': 'wesad',
            'OhioT1DM': 'ohio-t1dm'
        }
        
        dataset_path = Path(base_path) / dir_mapping[dataset_name]
        return loader_class(str(dataset_path))
