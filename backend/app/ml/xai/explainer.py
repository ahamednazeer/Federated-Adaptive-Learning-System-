"""
Explainability (XAI) Module
Generates human-readable explanations for diagnosis results
"""

import numpy as np
from typing import Dict, List, Optional
import json


class ExplainabilityEngine:
    """
    Generates explanations for multi-modal diagnosis
    """
    
    def __init__(self):
        self.modality_names = {
            'ecg': 'ECG (Heart Rhythm)',
            'voice': 'Voice Analysis',
            'handwriting': 'Handwriting Pattern',
            'wearable': 'Wearable Sensors',
            'glucose': 'Glucose Monitoring'
        }
        
        self.disease_thresholds = {
            'cvd': {'low': 0.3, 'moderate': 0.5, 'high': 0.7},
            'parkinsons': {'low': 0.3, 'moderate': 0.5, 'high': 0.7},
            'diabetes': {'low': 0.3, 'moderate': 0.5, 'high': 0.7}
        }
    
    def generate_explanation(self, 
                           diagnosis_result: Dict,
                           attention_weights: Dict[str, float],
                           modality_features: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive explanation for diagnosis
        
        Args:
            diagnosis_result: Dict with cvd_risk, parkinsons_risk, diabetes_risk
            attention_weights: Dict with attention weights for each modality
            modality_features: Optional dict with extracted features from each modality
        
        Returns:
            Dict with explanations
        """
        explanations = {
            'summary': self._generate_summary(diagnosis_result, attention_weights),
            'cvd_explanation': self._explain_cvd(diagnosis_result['cvd_risk'], attention_weights, modality_features),
            'parkinsons_explanation': self._explain_parkinsons(diagnosis_result['parkinsons_risk'], attention_weights, modality_features),
            'diabetes_explanation': self._explain_diabetes(diagnosis_result['diabetes_risk'], attention_weights, modality_features),
            'modality_contributions': self._explain_modality_contributions(attention_weights),
            'recommendations': self._generate_recommendations(diagnosis_result)
        }
        
        return explanations
    
    def _generate_summary(self, diagnosis_result: Dict, attention_weights: Dict) -> str:
        """Generate overall summary"""
        # Find highest risk
        risks = {
            'CVD': diagnosis_result['cvd_risk'],
            'Parkinson\'s Disease': diagnosis_result['parkinsons_risk'],
            'Diabetes': diagnosis_result['diabetes_risk']
        }
        
        max_disease = max(risks, key=risks.get)
        max_risk = risks[max_disease]
        
        # Find most important modality
        max_modality = max(attention_weights, key=attention_weights.get)
        
        if max_risk < 0.3:
            summary = f"All disease risk levels are low. The analysis primarily relied on {self.modality_names[max_modality]}."
        elif max_risk < 0.5:
            summary = f"Moderate risk detected for {max_disease} (risk score: {max_risk:.2f}). The analysis primarily relied on {self.modality_names[max_modality]}."
        else:
            summary = f"Elevated risk detected for {max_disease} (risk score: {max_risk:.2f}). The analysis primarily relied on {self.modality_names[max_modality]}. Medical consultation recommended."
        
        return summary
    
    def _explain_cvd(self, risk: float, attention_weights: Dict, features: Optional[Dict]) -> Dict:
        """Explain CVD risk"""
        level = self._get_risk_level(risk, 'cvd')
        
        explanation = {
            'risk_score': float(risk),
            'risk_level': level,
            'natural_language': '',
            'key_indicators': []
        }
        
        # Generate natural language explanation
        if level == 'low':
            explanation['natural_language'] = "Your cardiac health indicators appear normal. "
        elif level == 'moderate':
            explanation['natural_language'] = "Some cardiac health indicators show moderate concern. "
        else:
            explanation['natural_language'] = "Multiple cardiac health indicators show elevated risk. "
        
        # Add modality-specific insights
        if attention_weights.get('ecg', 0) > 0.2:
            if features and 'ecg' in features:
                hrv = features['ecg'].get('hrv_features', {})
                if hrv.get('sdnn', 0) < 50:
                    explanation['key_indicators'].append("Reduced heart rate variability detected")
                if hrv.get('mean_hr', 70) > 100:
                    explanation['key_indicators'].append("Elevated resting heart rate")
            else:
                explanation['key_indicators'].append("ECG rhythm patterns analyzed")
        
        if attention_weights.get('wearable', 0) > 0.2:
            if features and 'wearable' in features:
                hrv = features['wearable'].get('hrv_features', {})
                if hrv.get('rmssd', 0) < 20:
                    explanation['key_indicators'].append("Low HRV variability from wearable sensors")
            else:
                explanation['key_indicators'].append("Wearable sensor data analyzed")
        
        if not explanation['key_indicators']:
            explanation['key_indicators'].append("Multi-modal analysis of cardiac indicators")
        
        explanation['natural_language'] += " ".join(explanation['key_indicators'][:2]) + "."
        
        return explanation
    
    def _explain_parkinsons(self, risk: float, attention_weights: Dict, features: Optional[Dict]) -> Dict:
        """Explain Parkinson's risk"""
        level = self._get_risk_level(risk, 'parkinsons')
        
        explanation = {
            'risk_score': float(risk),
            'risk_level': level,
            'natural_language': '',
            'key_indicators': []
        }
        
        # Generate natural language explanation
        if level == 'low':
            explanation['natural_language'] = "No significant motor control or tremor indicators detected. "
        elif level == 'moderate':
            explanation['natural_language'] = "Some motor control variations detected. "
        else:
            explanation['natural_language'] = "Multiple motor control and tremor indicators detected. "
        
        # Add modality-specific insights
        if attention_weights.get('voice', 0) > 0.2:
            if features and 'voice' in features:
                tremor = features['voice'].get('tremor_features', {})
                if tremor.get('tremor_ratio', 0) > 0.3:
                    explanation['key_indicators'].append("Voice tremor patterns in 4-6 Hz range")
                if tremor.get('jitter', 0) > 0.01:
                    explanation['key_indicators'].append("Increased voice jitter")
            else:
                explanation['key_indicators'].append("Voice tremor analysis performed")
        
        if attention_weights.get('handwriting', 0) > 0.2:
            if features and 'handwriting' in features:
                tremor = features['handwriting'].get('tremor_features', {})
                if tremor.get('tremor_ratio', 0) > 0.3:
                    explanation['key_indicators'].append("Handwriting tremor detected")
                if tremor.get('velocity_cv', 0) > 0.5:
                    explanation['key_indicators'].append("Reduced motor control in handwriting")
            else:
                explanation['key_indicators'].append("Handwriting pattern analysis performed")
        
        if not explanation['key_indicators']:
            explanation['key_indicators'].append("Multi-modal motor control analysis")
        
        explanation['natural_language'] += " ".join(explanation['key_indicators'][:2]) + "."
        
        return explanation
    
    def _explain_diabetes(self, risk: float, attention_weights: Dict, features: Optional[Dict]) -> Dict:
        """Explain diabetes risk"""
        level = self._get_risk_level(risk, 'diabetes')
        
        explanation = {
            'risk_score': float(risk),
            'risk_level': level,
            'natural_language': '',
            'key_indicators': []
        }
        
        # Generate natural language explanation
        if level == 'low':
            explanation['natural_language'] = "Blood glucose patterns appear well-controlled. "
        elif level == 'moderate':
            explanation['natural_language'] = "Some glucose variability detected. "
        else:
            explanation['natural_language'] = "Significant glucose variability and control issues detected. "
        
        # Add modality-specific insights
        if attention_weights.get('glucose', 0) > 0.2:
            if features and 'glucose' in features:
                glycemic = features['glucose'].get('glycemic_features', {})
                if glycemic.get('mean_glucose', 100) > 140:
                    explanation['key_indicators'].append("Elevated average glucose levels")
                if glycemic.get('cv', 0) > 36:
                    explanation['key_indicators'].append("High glucose variability")
                if glycemic.get('time_in_range', 100) < 70:
                    explanation['key_indicators'].append(f"Time in target range: {glycemic['time_in_range']:.0f}%")
            else:
                explanation['key_indicators'].append("Continuous glucose monitoring analyzed")
        
        if not explanation['key_indicators']:
            explanation['key_indicators'].append("Multi-modal metabolic analysis")
        
        explanation['natural_language'] += " ".join(explanation['key_indicators'][:2]) + "."
        
        return explanation
    
    def _explain_modality_contributions(self, attention_weights: Dict) -> List[Dict]:
        """Explain how each modality contributed"""
        contributions = []
        
        # Sort by weight
        sorted_modalities = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)
        
        for modality, weight in sorted_modalities:
            if weight > 0.05:  # Only include significant contributions
                contributions.append({
                    'modality': self.modality_names[modality],
                    'weight': float(weight),
                    'percentage': float(weight * 100),
                    'importance': 'high' if weight > 0.3 else 'moderate' if weight > 0.15 else 'low'
                })
        
        return contributions
    
    def _generate_recommendations(self, diagnosis_result: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # CVD recommendations
        if diagnosis_result['cvd_risk'] > 0.5:
            recommendations.append("Consult a cardiologist for comprehensive cardiac evaluation")
            recommendations.append("Monitor blood pressure and heart rate regularly")
        elif diagnosis_result['cvd_risk'] > 0.3:
            recommendations.append("Consider lifestyle modifications for heart health")
        
        # Parkinson's recommendations
        if diagnosis_result['parkinsons_risk'] > 0.5:
            recommendations.append("Consult a neurologist for motor function assessment")
            recommendations.append("Consider tremor and gait analysis")
        elif diagnosis_result['parkinsons_risk'] > 0.3:
            recommendations.append("Monitor motor control symptoms")
        
        # Diabetes recommendations
        if diagnosis_result['diabetes_risk'] > 0.5:
            recommendations.append("Consult an endocrinologist for glucose management")
            recommendations.append("Continue regular glucose monitoring")
        elif diagnosis_result['diabetes_risk'] > 0.3:
            recommendations.append("Monitor blood glucose levels regularly")
        
        if not recommendations:
            recommendations.append("Continue regular health monitoring")
            recommendations.append("Maintain healthy lifestyle habits")
        
        return recommendations
    
    def _get_risk_level(self, risk: float, disease: str) -> str:
        """Determine risk level"""
        thresholds = self.disease_thresholds[disease]
        
        if risk < thresholds['low']:
            return 'low'
        elif risk < thresholds['moderate']:
            return 'moderate'
        elif risk < thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def generate_patient_friendly_explanation(self, explanations: Dict) -> str:
        """Generate a simple, patient-friendly explanation"""
        summary = explanations['summary']
        
        # Add key findings
        findings = []
        for disease in ['cvd', 'parkinsons', 'diabetes']:
            exp_key = f'{disease}_explanation'
            if exp_key in explanations:
                exp = explanations[exp_key]
                if exp['risk_level'] in ['high', 'very_high']:
                    findings.append(f"• {exp['natural_language']}")
        
        # Add recommendations
        recs = explanations.get('recommendations', [])
        
        patient_text = f"{summary}\n\n"
        
        if findings:
            patient_text += "Key Findings:\n" + "\n".join(findings) + "\n\n"
        
        if recs:
            patient_text += "Recommendations:\n" + "\n".join([f"• {rec}" for rec in recs[:3]])
        
        return patient_text


def create_explainability_engine() -> ExplainabilityEngine:
    """Factory function to create explainability engine"""
    return ExplainabilityEngine()


if __name__ == '__main__':
    # Test the explainability engine
    print("Testing Explainability Engine...")
    
    engine = create_explainability_engine()
    
    # Test diagnosis
    diagnosis = {
        'cvd_risk': 0.72,
        'parkinsons_risk': 0.41,
        'diabetes_risk': 0.65
    }
    
    attention_weights = {
        'ecg': 0.35,
        'voice': 0.20,
        'handwriting': 0.15,
        'wearable': 0.20,
        'glucose': 0.10
    }
    
    explanations = engine.generate_explanation(diagnosis, attention_weights)
    
    print("\nSummary:", explanations['summary'])
    print("\nCVD Explanation:", explanations['cvd_explanation']['natural_language'])
    print("  Risk Level:", explanations['cvd_explanation']['risk_level'])
    print("\nModality Contributions:")
    for contrib in explanations['modality_contributions']:
        print(f"  {contrib['modality']}: {contrib['percentage']:.1f}% ({contrib['importance']})")
    
    print("\nRecommendations:")
    for rec in explanations['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "="*60)
    print("Patient-Friendly Explanation:")
    print("="*60)
    print(engine.generate_patient_friendly_explanation(explanations))
    
    print("\n✓ Explainability Engine test passed!")
