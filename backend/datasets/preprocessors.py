"""
Preprocessing Pipelines for Multi-Modal Medical Data
Applies modality-specific transformations to raw sensor data.
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ECGPreprocessor:
    """Preprocessing for ECG signals"""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        self.lowcut = 0.5  # Hz
        self.highcut = 40.0  # Hz
    
    def bandpass_filter(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove baseline wander and high-freq noise"""
        nyquist = self.sampling_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        return filtered
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks in ECG signal"""
        # Simple peak detection
        peaks, _ = signal.find_peaks(ecg_signal, distance=self.sampling_rate//3, height=0.5)
        return peaks
    
    def segment_beats(self, ecg_signal: np.ndarray, r_peaks: np.ndarray, 
                     window_size: int = 200) -> List[np.ndarray]:
        """Segment ECG into individual heartbeats"""
        beats = []
        for peak in r_peaks:
            start = max(0, peak - window_size // 2)
            end = min(len(ecg_signal), peak + window_size // 2)
            beat = ecg_signal[start:end]
            
            # Pad if necessary
            if len(beat) < window_size:
                beat = np.pad(beat, (0, window_size - len(beat)), mode='edge')
            
            beats.append(beat)
        
        return beats
    
    def compute_hrv_features(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """Compute Heart Rate Variability features"""
        # RR intervals (in ms)
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000
        
        if len(rr_intervals) < 2:
            return {'sdnn': 0, 'rmssd': 0, 'pnn50': 0}
        
        # Time-domain HRV metrics
        sdnn = np.std(rr_intervals)  # Standard deviation
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))  # Root mean square of successive differences
        
        # pNN50: percentage of successive RR intervals that differ by more than 50 ms
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = (nn50 / len(rr_intervals)) * 100 if len(rr_intervals) > 0 else 0
        
        return {
            'sdnn': float(sdnn),
            'rmssd': float(rmssd),
            'pnn50': float(pnn50),
            'mean_hr': float(60000 / np.mean(rr_intervals)) if np.mean(rr_intervals) > 0 else 0
        }
    
    def preprocess(self, ecg_data: Dict) -> Dict:
        """Full preprocessing pipeline"""
        ecg_signal = np.array(ecg_data['signal'])
        
        # Apply bandpass filter
        filtered = self.bandpass_filter(ecg_signal)
        
        # Normalize
        normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
        
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(filtered)
        
        # Compute HRV features
        hrv_features = self.compute_hrv_features(r_peaks)
        
        # Segment beats
        beats = self.segment_beats(normalized, r_peaks)
        
        return {
            'processed_signal': normalized,
            'r_peaks': r_peaks,
            'hrv_features': hrv_features,
            'beats': beats,
            'num_beats': len(beats),
            'patient_id': ecg_data.get('patient_id')
        }


class VoicePreprocessor:
    """Preprocessing for voice/audio signals"""
    
    def __init__(self, num_mfcc: int = 13):
        self.num_mfcc = num_mfcc
    
    def remove_silence(self, mfcc: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silent frames based on energy"""
        energy = np.sum(mfcc**2, axis=0)
        active_frames = energy > threshold
        return mfcc[:, active_frames]
    
    def compute_delta_features(self, mfcc: np.ndarray) -> np.ndarray:
        """Compute delta (velocity) features"""
        delta = np.zeros_like(mfcc)
        for i in range(1, mfcc.shape[1] - 1):
            delta[:, i] = (mfcc[:, i + 1] - mfcc[:, i - 1]) / 2
        return delta
    
    def compute_tremor_features(self, mfcc: np.ndarray) -> Dict[str, float]:
        """Extract tremor-related features from voice"""
        # Analyze first MFCC coefficient (fundamental frequency variation)
        f0_variation = mfcc[0, :]
        
        # Compute tremor frequency using FFT
        fft = np.fft.fft(f0_variation)
        freqs = np.fft.fftfreq(len(f0_variation), d=0.03)  # Assuming 30ms frames
        
        # Parkinson's tremor typically 4-6 Hz
        tremor_band = (freqs >= 4) & (freqs <= 6)
        tremor_power = np.sum(np.abs(fft[tremor_band])**2)
        total_power = np.sum(np.abs(fft)**2)
        
        tremor_ratio = tremor_power / (total_power + 1e-8)
        
        # Jitter and shimmer (voice quality measures)
        jitter = np.std(np.diff(f0_variation)) / (np.mean(np.abs(f0_variation)) + 1e-8)
        shimmer = np.std(mfcc[1, :]) / (np.mean(np.abs(mfcc[1, :])) + 1e-8)
        
        return {
            'tremor_ratio': float(tremor_ratio),
            'jitter': float(jitter),
            'shimmer': float(shimmer)
        }
    
    def preprocess(self, voice_data: Dict) -> Dict:
        """Full preprocessing pipeline"""
        mfcc = np.array(voice_data['mfcc'])
        
        # Remove silence
        mfcc_active = self.remove_silence(mfcc)
        
        # Normalize
        mfcc_norm = (mfcc_active - np.mean(mfcc_active, axis=1, keepdims=True)) / \
                    (np.std(mfcc_active, axis=1, keepdims=True) + 1e-8)
        
        # Compute delta features
        delta = self.compute_delta_features(mfcc_norm)
        
        # Compute tremor features
        tremor_features = self.compute_tremor_features(mfcc_active)
        
        # Concatenate MFCC and delta
        features = np.vstack([mfcc_norm, delta])
        
        return {
            'mfcc_features': features,
            'tremor_features': tremor_features,
            'num_frames': features.shape[1],
            'patient_id': voice_data.get('patient_id')
        }


class HandwritingPreprocessor:
    """Preprocessing for handwriting/drawing data"""
    
    def smooth_trajectory(self, x: np.ndarray, y: np.ndarray, 
                         window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth handwriting trajectory"""
        x_smooth = signal.savgol_filter(x, window_size, 2)
        y_smooth = signal.savgol_filter(y, window_size, 2)
        return x_smooth, y_smooth
    
    def compute_velocity_acceleration(self, x: np.ndarray, y: np.ndarray) -> Dict:
        """Compute velocity and acceleration profiles"""
        # Velocity
        vx = np.diff(x)
        vy = np.diff(y)
        velocity = np.sqrt(vx**2 + vy**2)
        velocity = np.append(velocity, velocity[-1])
        
        # Acceleration
        ax = np.diff(vx)
        ay = np.diff(vy)
        acceleration = np.sqrt(ax**2 + ay**2)
        acceleration = np.append(acceleration, [acceleration[-1], acceleration[-1]])
        
        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'mean_velocity': np.mean(velocity),
            'std_velocity': np.std(velocity)
        }
    
    def compute_tremor_features(self, x: np.ndarray, y: np.ndarray, 
                               velocity: np.ndarray) -> Dict[str, float]:
        """Extract tremor features from handwriting"""
        # Frequency analysis of position
        fft_x = np.fft.fft(x)
        fft_y = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(x), d=0.01)  # Assuming 100Hz sampling
        
        # Tremor band (4-7 Hz for Parkinson's)
        tremor_band = (freqs >= 4) & (freqs <= 7)
        tremor_power_x = np.sum(np.abs(fft_x[tremor_band])**2)
        tremor_power_y = np.sum(np.abs(fft_y[tremor_band])**2)
        
        total_power_x = np.sum(np.abs(fft_x)**2)
        total_power_y = np.sum(np.abs(fft_y)**2)
        
        tremor_ratio = (tremor_power_x + tremor_power_y) / (total_power_x + total_power_y + 1e-8)
        
        # Velocity variation (motor control)
        velocity_cv = np.std(velocity) / (np.mean(velocity) + 1e-8)
        
        return {
            'tremor_ratio': float(tremor_ratio),
            'velocity_cv': float(velocity_cv),
            'mean_velocity': float(np.mean(velocity))
        }
    
    def preprocess(self, handwriting_data: Dict) -> Dict:
        """Full preprocessing pipeline"""
        x = np.array(handwriting_data['x'])
        y = np.array(handwriting_data['y'])
        pressure = np.array(handwriting_data['pressure'])
        
        # Smooth trajectory
        x_smooth, y_smooth = self.smooth_trajectory(x, y)
        
        # Compute velocity and acceleration
        kinematics = self.compute_velocity_acceleration(x_smooth, y_smooth)
        
        # Compute tremor features
        tremor_features = self.compute_tremor_features(x_smooth, y_smooth, kinematics['velocity'])
        
        # Normalize coordinates
        x_norm = (x_smooth - np.mean(x_smooth)) / (np.std(x_smooth) + 1e-8)
        y_norm = (y_smooth - np.mean(y_smooth)) / (np.std(y_smooth) + 1e-8)
        
        # Create feature matrix
        features = np.stack([x_norm, y_norm, kinematics['velocity'], 
                           kinematics['acceleration'], pressure], axis=0)
        
        return {
            'features': features,
            'tremor_features': tremor_features,
            'num_points': len(x),
            'patient_id': handwriting_data.get('patient_id')
        }


class WearablePreprocessor:
    """Preprocessing for wearable sensor data"""
    
    def __init__(self, window_size: int = 30, sampling_rate: int = 32):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
    
    def compute_hrv_features(self, heart_rate: np.ndarray) -> Dict[str, float]:
        """Compute HRV features from heart rate"""
        # Convert HR to RR intervals
        rr_intervals = 60000 / (heart_rate + 1e-8)  # ms
        
        # Time-domain features
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
        
        return {
            'sdnn': float(sdnn),
            'rmssd': float(rmssd),
            'mean_hr': float(np.mean(heart_rate))
        }
    
    def compute_stress_features(self, skin_conductance: np.ndarray) -> Dict[str, float]:
        """Compute stress-related features from skin conductance"""
        # SCL: Skin Conductance Level (tonic component)
        scl = signal.medfilt(skin_conductance, kernel_size=31)
        
        # SCR: Skin Conductance Response (phasic component)
        scr = skin_conductance - scl
        
        # Number of SCR peaks (stress responses)
        peaks, _ = signal.find_peaks(scr, height=0.05)
        num_peaks = len(peaks)
        
        return {
            'mean_scl': float(np.mean(scl)),
            'std_scr': float(np.std(scr)),
            'num_stress_responses': int(num_peaks)
        }
    
    def compute_activity_features(self, acc_x: np.ndarray, acc_y: np.ndarray, 
                                  acc_z: np.ndarray) -> Dict[str, float]:
        """Compute activity features from accelerometer"""
        # Magnitude
        magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        # Activity level
        activity_level = np.std(magnitude)
        
        return {
            'activity_level': float(activity_level),
            'mean_magnitude': float(np.mean(magnitude))
        }
    
    def preprocess(self, wearable_data: Dict) -> Dict:
        """Full preprocessing pipeline"""
        hr = np.array(wearable_data['heart_rate'])
        sc = np.array(wearable_data['skin_conductance'])
        acc_x = np.array(wearable_data['acc_x'])
        acc_y = np.array(wearable_data['acc_y'])
        acc_z = np.array(wearable_data['acc_z'])
        
        # Compute features
        hrv_features = self.compute_hrv_features(hr)
        stress_features = self.compute_stress_features(sc)
        activity_features = self.compute_activity_features(acc_x, acc_y, acc_z)
        
        # Normalize signals (Use fixed scaling to preserve absolute levels)
        # HR: Map 40-200 BPM to approx -1 to 1 range (using 0-1 scaling is safer for ReLU)
        # Actually standard scaling is usually best for DL, but we want to preserve the OFFSET
        # So we subtract a FIXED mean (e.g. 70) rather than the sample mean
        hr_norm = (hr - 70) / 30.0  # 100bpm -> 1.0, 70bpm -> 0.0, 130bpm -> 2.0
        
        # SC: Map 0-20 muS
        sc_norm = (sc - 5) / 5.0
        
        # Stack features
        features = np.stack([hr_norm, sc_norm, acc_x, acc_y, acc_z], axis=0)
        
        return {
            'features': features,
            'hrv_features': hrv_features,
            'stress_features': stress_features,
            'activity_features': activity_features,
            'patient_id': wearable_data.get('patient_id')
        }


class GlucosePreprocessor:
    """Preprocessing for continuous glucose monitoring data"""
    
    def interpolate_missing(self, glucose: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Interpolate missing glucose readings"""
        # Simple linear interpolation
        valid_indices = ~np.isnan(glucose)
        if np.sum(valid_indices) < 2:
            return glucose
        
        interpolated = np.interp(timestamps, timestamps[valid_indices], glucose[valid_indices])
        return interpolated
    
    def smooth_signal(self, glucose: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Smooth glucose signal"""
        return signal.savgol_filter(glucose, window_size, 2)
    
    def compute_glycemic_features(self, glucose: np.ndarray) -> Dict[str, float]:
        """Compute glycemic variability features"""
        # Mean glucose
        mean_glucose = np.mean(glucose)
        
        # Standard deviation (glycemic variability)
        std_glucose = np.std(glucose)
        
        # Coefficient of variation
        cv = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0
        
        # Time in range (70-180 mg/dL)
        tir = np.sum((glucose >= 70) & (glucose <= 180)) / len(glucose) * 100
        
        # Time above range (>180 mg/dL)
        tar = np.sum(glucose > 180) / len(glucose) * 100
        
        # Time below range (<70 mg/dL)
        tbr = np.sum(glucose < 70) / len(glucose) * 100
        
        return {
            'mean_glucose': float(mean_glucose),
            'std_glucose': float(std_glucose),
            'cv': float(cv),
            'time_in_range': float(tir),
            'time_above_range': float(tar),
            'time_below_range': float(tbr)
        }
    
    def compute_trend_features(self, glucose: np.ndarray) -> Dict[str, float]:
        """Compute glucose trend features"""
        # Rate of change
        rate_of_change = np.diff(glucose)
        
        # Mean absolute rate
        mean_rate = np.mean(np.abs(rate_of_change))
        
        # Number of rapid changes (>2 mg/dL per 5 min)
        rapid_changes = np.sum(np.abs(rate_of_change) > 2)
        
        return {
            'mean_rate_of_change': float(mean_rate),
            'num_rapid_changes': int(rapid_changes)
        }
    
    def preprocess(self, glucose_data: Dict) -> Dict:
        """Full preprocessing pipeline"""
        glucose = np.array(glucose_data['glucose'])
        timestamps = np.array(glucose_data['timestamps'])
        
        # Interpolate missing values
        glucose_interp = self.interpolate_missing(glucose, timestamps)
        
        # Smooth signal
        glucose_smooth = self.smooth_signal(glucose_interp)
        
        # Normalize (Use fixed scaling to preserve hyperglycemia signal)
        # Map 0-400 mg/dL to roughly 0-1 range
        glucose_norm = glucose_smooth / 200.0  # 100 -> 0.5, 200 -> 1.0
        
        # Compute features
        glycemic_features = self.compute_glycemic_features(glucose_smooth)
        trend_features = self.compute_trend_features(glucose_smooth)
        
        return {
            'processed_signal': glucose_norm,
            'glycemic_features': glycemic_features,
            'trend_features': trend_features,
            'patient_id': glucose_data.get('patient_id')
        }


# Preprocessor factory
class PreprocessorFactory:
    """Factory for creating preprocessors"""
    
    PREPROCESSORS = {
        'ecg': ECGPreprocessor,
        'voice': VoicePreprocessor,
        'handwriting': HandwritingPreprocessor,
        'wearable': WearablePreprocessor,
        'glucose': GlucosePreprocessor
    }
    
    @staticmethod
    def create_preprocessor(modality: str):
        """Create a preprocessor instance"""
        if modality not in PreprocessorFactory.PREPROCESSORS:
            raise ValueError(f"Unknown modality: {modality}")
        
        return PreprocessorFactory.PREPROCESSORS[modality]()
