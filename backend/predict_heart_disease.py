"""
Heart Disease Prediction ML Model Backend
This module handles the machine learning model for heart disease prediction.
"""

import joblib
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

FEATURE_NAMES = [
    'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
    'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
    'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
]

MODEL_DIR = Path("backend/heart_disease_model")

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_and_scaler() -> Tuple[Any, Any, Optional[Dict], Optional[Dict]]:
    """Load the trained model and scaler"""
    try:
        model = joblib.load(MODEL_DIR / "gradient_boosting_model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        
        # Load additional info if available
        feature_info = None
        metrics = None
        
        if (MODEL_DIR / "feature_info.json").exists():
            with open(MODEL_DIR / "feature_info.json", "r") as f:
                feature_info = json.load(f)
        
        if (MODEL_DIR / "performance_metrics.json").exists():
            with open(MODEL_DIR / "performance_metrics.json", "r") as f:
                metrics = json.load(f)
        
        return model, scaler, feature_info, metrics
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

# ============================================================================
# RISK LEVEL DETERMINATION
# ============================================================================

def get_risk_level(disease_prob: float) -> Tuple[str, str, str]:
    """Determine risk level based on disease probability"""
    if disease_prob >= 0.8:
        risk_level = "Very High"
        risk_color = "#dc3545"
        recommendation = "🚨 URGENT: Immediate medical consultation is strongly recommended!"
    elif disease_prob >= 0.6:
        risk_level = "High"
        risk_color = "#fd7e14"
        recommendation = "⚠️ Schedule a medical consultation soon for further evaluation."
    elif disease_prob >= 0.4:
        risk_level = "Medium"
        risk_color = "#ffc107"
        recommendation = "📋 Consider scheduling a check-up with your physician."
    elif disease_prob >= 0.2:
        risk_level = "Low"
        risk_color = "#28a745"
        recommendation = "✅ Continue monitoring your health and maintain healthy habits."
    else:
        risk_level = "Very Low"
        risk_color = "#20c997"
        recommendation = "💚 Excellent! Keep maintaining your healthy lifestyle!"
    
    return risk_level, risk_color, recommendation

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_heart_disease(model: Any, scaler: Any, patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction for patient data"""
    
    # Create DataFrame with correct feature order
    df = pd.DataFrame([patient_data])
    df = df[FEATURE_NAMES]
    
    # Scale features
    scaled_data = scaler.transform(df)
    
    # Predict
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    # Determine risk level
    disease_prob = probabilities[1]
    risk_level, risk_color, recommendation = get_risk_level(disease_prob)
    
    result = {
        "prediction": int(prediction),
        "prediction_label": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "probability_no_disease": float(probabilities[0]),
        "probability_disease": float(probabilities[1]),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "patient_data": patient_data
    }
    
    return result

# ============================================================================
# BATCH PREDICTION
# ============================================================================

def batch_predict(model: Any, scaler: Any, patients_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Make predictions for multiple patients"""
    results = []
    
    for idx, row in patients_df.iterrows():
        patient_data = row.to_dict()
        result = predict_heart_disease(model, scaler, patient_data)
        result['patient_id'] = idx + 1
        results.append(result)
    
    return results

# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    # Test the model loading
    model, scaler, feature_info, metrics = load_model_and_scaler()
    
    if model is not None:
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Test prediction with sample data
        sample_patient = {
            "age": 55,
            "gender": 1,
            "chestpain": 2,
            "restingBP": 140,
            "serumcholestrol": 260,
            "fastingbloodsugar": 0,
            "restingrelectro": 1,
            "maxheartrate": 150,
            "exerciseangia": 0,
            "oldpeak": 1.5,
            "slope": 1,
            "noofmajorvessels": 0
        }
        
        result = predict_heart_disease(model, scaler, sample_patient)
        print(f"\nPrediction: {result['prediction_label']}")
        print(f"Disease Probability: {result['probability_disease']:.2%}")
        print(f"Risk Level: {result['risk_level']}")
    else:
        print("Failed to load model!")
